#!/usr/bin/env python3
"""Supervise the released dictation bot with no command-line configuration."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


IS_WINDOWS = os.name == "nt"
ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
PROGRAM = ROOT / ("dictation-bot.exe" if IS_WINDOWS else "dictation-bot")
CONFIG = ROOT / "dictation.toml"
CONTROL_PAGE = ROOT / "control.html"
RUNTIME_CONFIG = ROOT / "cadmus-runtime.json"
DEFAULTS_FILE = ROOT / "cadmus-defaults.json"
LOG = LOGS / "cadmus-watchdog.log"
PID_FILE = LOGS / "cadmus-watchdog.pid"
STOP_FILE = LOGS / "cadmus.stop"
DUMP_DIR = LOGS / "dumps"
CRASH_DIR = LOGS / "crashes"
CRASH_BUNDLES = 10
STABLE_SECONDS = 300.0
MAX_RESTART_DELAY = 60.0
STOP_TIMEOUT = 1200.0
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 8091
MAX_CONTROL_BODY = 512 * 1024
ACTIVITY_WINDOW = 64 * 1024
# A browser will happily send a cross-site form POST to a loopback port, and DNS rebinding defeats
# binding to 127.0.0.1. Pinning Host and Origin, and requiring a JSON content type (which a form
# cannot set without a preflight), keeps the control API reachable only from its own page.
ALLOWED_HOSTS = frozenset(
    {f"{CONTROL_HOST}:{CONTROL_PORT}", f"localhost:{CONTROL_PORT}", f"[::1]:{CONTROL_PORT}"}
)
ALLOWED_ORIGINS = frozenset(f"http://{host}" for host in ALLOWED_HOSTS)
WER_LOCAL_DUMPS = r"SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps"

STATE_LOCK = threading.Lock()
# Saving settings is a read-modify-write (merge current, bump version, replace the file) and the
# control server is threaded, so concurrent savers must not interleave.
SETTINGS_LOCK = threading.Lock()
STATE: dict[str, object] = {
    "watchdog_started_at": time.time(),
    "child_pid": 0,
    "child_started_at": 0.0,
    "child_exit_code": None,
    "restart_delay": 0.0,
}
CONTROL_LOGGER: logging.Logger | None = None


def make_logger() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cadmus-watchdog")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    rotating = RotatingFileHandler(
        LOG, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    rotating.setFormatter(formatter)
    logger.addHandler(rotating)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def emit(logger: logging.Logger, event: str, **fields: object) -> None:
    logger.info(
        json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": event,
                **fields,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def windows_balloon(title: str, message: str) -> None:
    if not IS_WINDOWS:
        return
    safe_title = title.replace("'", "''")
    safe_message = message.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "Add-Type -AssemblyName System.Drawing;"
        "$n=New-Object System.Windows.Forms.NotifyIcon;"
        "$n.Icon=[System.Drawing.SystemIcons]::Error;"
        "$n.Visible=$true;"
        f"$n.ShowBalloonTip(10000,'{safe_title}','{safe_message}',"
        "[System.Windows.Forms.ToolTipIcon]::Error);"
        "Start-Sleep -Seconds 11;"
        "$n.Dispose()"
    )
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        pass


def windows_local_dump_config(exe_name: str) -> tuple[int | None, Path | None]:
    if not IS_WINDOWS:
        return None, None
    import winreg

    for subkey, app_specific in (
        (f"{WER_LOCAL_DUMPS}\\{exe_name}", True),
        (WER_LOCAL_DUMPS, False),
    ):
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                subkey,
                0,
                winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
            ) as key:
                if not app_specific:
                    try:
                        winreg.QueryValueEx(key, "DumpType")
                    except FileNotFoundError:
                        continue
                try:
                    dump_type = int(winreg.QueryValueEx(key, "DumpType")[0])
                except FileNotFoundError:
                    dump_type = 1
                try:
                    folder = Path(
                        os.path.expandvars(str(winreg.QueryValueEx(key, "DumpFolder")[0]))
                    )
                except FileNotFoundError:
                    folder = Path(os.environ["LOCALAPPDATA"]) / "CrashDumps"
                return dump_type, folder
        except (FileNotFoundError, OSError):
            continue
    return None, None


def wait_for_dump(exe_name: str, started_at: float) -> Path | None:
    deadline = time.monotonic() + 15.0
    prefix = exe_name.lower()
    while time.monotonic() < deadline:
        try:
            candidates = [
                path
                for path in DUMP_DIR.glob("*.dmp")
                if path.name.lower().startswith(prefix)
                and path.stat().st_mtime >= started_at - 5.0
            ]
        except OSError:
            candidates = []
        if candidates:
            newest = max(candidates, key=lambda path: path.stat().st_mtime)
            try:
                with newest.open("rb"):
                    pass
                return newest
            except OSError:
                pass
        time.sleep(0.25)
    return None


def prune_crash_bundles() -> None:
    if not CRASH_DIR.is_dir():
        return
    bundles = sorted(
        (path for path in CRASH_DIR.glob("cadmus-*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in bundles[CRASH_BUNDLES:]:
        shutil.rmtree(path, ignore_errors=True)


def collect_crash_bundle(
    child_pid: int, return_code: int, started_at: float, dump_path: Path | None
) -> Path:
    bundle = CRASH_DIR / f"cadmus-{time.strftime('%Y%m%d-%H%M%S')}-pid{child_pid}"
    bundle.mkdir(parents=True, exist_ok=False)
    artifacts = [PROGRAM, PROGRAM.with_suffix(".map"), PROGRAM.with_suffix(".pdb")]
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pid": child_pid,
        "exit_code": return_code,
        "exit_code_hex": f"0x{return_code & 0xFFFFFFFF:08X}",
        "started_at": started_at,
        "command": [str(PROGRAM), "--config", str(CONFIG)],
        "cwd": str(ROOT),
        "dump": dump_path.name if dump_path is not None else "",
        "program_artifacts": [path.name for path in artifacts if path.is_file()],
    }
    (bundle / "crash.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if LOG.is_file():
        shutil.copy2(LOG, bundle / LOG.name)
    if dump_path is not None and dump_path.is_file():
        shutil.copy2(dump_path, bundle / dump_path.name)
    for source in artifacts:
        if source.is_file():
            shutil.copy2(source, bundle / source.name)
    prune_crash_bundles()
    return bundle


def stream_child(proc: subprocess.Popen[str], logger: logging.Logger) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        emit(logger, "child", pid=proc.pid, message=line.rstrip("\r\n"))


CONFIG_CACHE: tuple[object, dict[str, object]] = (None, {})


def read_bot_config() -> dict[str, object]:
    """Parsed dictation.toml, cached by mtime. Treat the result as read-only — it is shared."""
    global CONFIG_CACHE
    try:
        status = CONFIG.stat()
    except OSError:
        return {}
    stamp = (status.st_mtime_ns, status.st_size)
    cached_stamp, cached = CONFIG_CACHE
    if cached_stamp == stamp:
        return cached
    try:
        with CONFIG.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    result = data if isinstance(data, dict) else {}
    CONFIG_CACHE = (stamp, result)
    return result


def published_defaults() -> dict[str, object]:
    """Defaults the bot publishes at startup. Empty until it has run once."""
    try:
        value = json.loads(DEFAULTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def generation_defaults(config: dict[str, object]) -> dict[str, dict[str, object]]:
    """Cold-start fallback only. Once the bot has run, its published defaults win — it owns them."""

    def stage(temperature: float, max_tokens: int) -> dict[str, object]:
        return {
            "temperature": temperature,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repeat_penalty": 1.0,
            "max_tokens": max_tokens,
            "thinking": False,
        }

    cleanup_temp = float(config.get("temp", 0.0))
    assistant_temp = float(config.get("assistant_temp", 0.2))
    cleanup_tokens = int(config.get("max_tokens", 16384))
    assistant_tokens = int(config.get("assistant_max_tokens", 4096))
    return {
        "cleanup": stage(cleanup_temp, cleanup_tokens),
        "planner": stage(0.0, 512),
        "assistant": stage(assistant_temp, assistant_tokens),
        "summary": stage(assistant_temp, assistant_tokens),
    }


def default_runtime_config(config: dict[str, object]) -> dict[str, object]:
    published = published_defaults()
    features = published.get("features")
    generation = published.get("generation")
    return {
        "version": 1,
        "features": features
        if isinstance(features, dict)
        else {
            "transcription": True,
            "cleanup": True,
            "assistant": bool(config.get("reply_when_mentioned", False)),
            "history_retrieval": True,
            "summaries": True,
            "web_search": bool(config.get("web_search_enabled", True)),
            "history_import": bool(str(config.get("history_media_root", "")).strip()),
        },
        "generation": generation if isinstance(generation, dict) else generation_defaults(config),
        # Empty means "no override" — the bot falls back to its built-in prompt, and the control page
        # shows that built-in as placeholder text. Seeding these with the config's prompt would make
        # clearing a box in the UI silently ineffective.
        "prompts": {"cleanup": "", "planner": "", "assistant": "", "summary": ""},
    }


def builtin_prompts() -> dict[str, str]:
    prompts = published_defaults().get("prompts")
    if not isinstance(prompts, dict):
        return {}
    return {key: value for key, value in prompts.items() if isinstance(value, str)}


def merged_runtime_config() -> dict[str, object]:
    result = default_runtime_config(read_bot_config())
    try:
        saved = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    if not isinstance(saved, dict):
        return result
    if isinstance(saved.get("version"), int):
        result["version"] = saved["version"]
    for section in ("features", "generation", "prompts"):
        incoming = saved.get(section)
        target = result[section]
        if not isinstance(incoming, dict) or not isinstance(target, dict):
            continue
        for key, value in incoming.items():
            if key not in target:
                continue
            if section == "features" and isinstance(value, bool):
                target[key] = value
            elif section == "generation" and isinstance(value, dict):
                stage = target[key]
                if isinstance(stage, dict):
                    stage.update({k: v for k, v in value.items() if k in stage})
            elif section == "prompts" and isinstance(value, str):
                target[key] = value
    return result


def validate_runtime_config(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("settings must be a JSON object")
    current = merged_runtime_config()
    features = value.get("features")
    generation = value.get("generation")
    if not isinstance(features, dict) or not isinstance(generation, dict):
        raise ValueError("features and generation must be JSON objects")
    known_features = current["features"]
    assert isinstance(known_features, dict)
    for name in known_features:
        enabled = features.get(name)
        if not isinstance(enabled, bool):
            raise ValueError(f"feature '{name}' must be true or false")
        known_features[name] = enabled
    known_generation = current["generation"]
    assert isinstance(known_generation, dict)
    numeric_limits = {
        "temperature": (0.0, 4.0),
        "top_k": (0, 1_000_000),
        "top_p": (0.0, 1.0),
        "min_p": (0.0, 1.0),
        "presence_penalty": (-2.0, 2.0),
        "frequency_penalty": (-2.0, 2.0),
        "repeat_penalty": (0.0, 4.0),
        "max_tokens": (1, 1_000_000),
    }
    for stage_name, stage_defaults in known_generation.items():
        stage_in = generation.get(stage_name)
        if not isinstance(stage_in, dict) or not isinstance(stage_defaults, dict):
            raise ValueError(f"generation stage '{stage_name}' must be an object")
        for key, (lower, upper) in numeric_limits.items():
            number = stage_in.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"{stage_name}.{key} must be numeric")
            if number < lower or number > upper:
                raise ValueError(f"{stage_name}.{key} must be between {lower} and {upper}")
            stage_defaults[key] = int(number) if key in ("top_k", "max_tokens") else float(number)
        thinking = stage_in.get("thinking")
        if not isinstance(thinking, bool):
            raise ValueError(f"{stage_name}.thinking must be true or false")
        stage_defaults["thinking"] = thinking
    prompts = value.get("prompts")
    known_prompts = current["prompts"]
    if not isinstance(prompts, dict) or not isinstance(known_prompts, dict):
        raise ValueError("prompts must be a JSON object")
    for name in known_prompts:
        prompt = prompts.get(name)
        if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > 128 * 1024:
            raise ValueError(f"prompt '{name}' must be a string under 128 KiB")
        known_prompts[name] = prompt
    current["version"] = int(current.get("version", 0)) + 1
    current["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return current


def write_runtime_config(value: dict[str, object]) -> None:
    # The temp file lives beside the target, not under logs/: os.replace is only atomic within one
    # filesystem, and a unique name keeps a concurrent writer off this one's partial file.
    handle, temporary = tempfile.mkstemp(
        dir=RUNTIME_CONFIG.parent, prefix=RUNTIME_CONFIG.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as target:
            target.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, RUNTIME_CONFIG)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def bot_log_path(config: dict[str, object]) -> Path:
    configured = str(config.get("log_file", "")).strip()
    if not configured:
        return ROOT / "dictation.log"
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


ACTIVITY_CACHE: tuple[object, list[object]] = (None, [])


def recent_activity(limit: int = 80) -> list[object]:
    global ACTIVITY_CACHE
    path = bot_log_path(read_bot_config())
    try:
        status = path.stat()
    except OSError:
        return []
    stamp = (str(path), status.st_mtime_ns, status.st_size, limit)
    cached_stamp, cached = ACTIVITY_CACHE
    if cached_stamp == stamp:
        return cached
    try:
        with path.open("rb") as source:
            source.seek(max(0, status.st_size - ACTIVITY_WINDOW))
            raw = source.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = raw.splitlines()
    # A window that does not start at the top of the file begins mid-line; drop that fragment.
    if status.st_size > ACTIVITY_WINDOW and lines:
        lines = lines[1:]
    rows: list[object] = []
    for line in lines[-max(1, min(limit, 300)) :]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"level": "info", "cat": "process", "msg": line})
    ACTIVITY_CACHE = (stamp, rows)
    return rows


def server_base_url() -> str:
    configured = str(read_bot_config().get("server", "http://127.0.0.1:8080")).rstrip("/")
    parsed = urlparse(configured)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the dashboard only proxies a loopback HTTP model server")
    return configured


def server_json(method: str, route: str, body: object | None = None, timeout: float = 30.0) -> object:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        server_base_url() + route,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return json.loads(payload) if payload else {}
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model server HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"model server unavailable: {error.reason}") from error


class RequestRejected(Exception):
    """Request failed the Host / Origin / content-type guard."""


class ControlHandler(BaseHTTPRequestHandler):
    server_version = "CadmusControl/1"

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def send_json(self, status: int, value: object) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def check_origin(self, require_json_body: bool) -> None:
        if (self.headers.get("Host") or "").strip().lower() not in ALLOWED_HOSTS:
            raise RequestRejected("unexpected Host header")
        origin = self.headers.get("Origin")
        if origin is not None and origin.strip().lower() not in ALLOWED_ORIGINS:
            raise RequestRejected("cross-origin request rejected")
        if require_json_body:
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type != "application/json":
                raise RequestRejected("Content-Type must be application/json")

    def read_json(self) -> object:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length <= 0 or length > MAX_CONTROL_BODY:
            raise ValueError("request body is empty or too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        try:
            self.check_origin(require_json_body=False)
            if route == "/":
                payload = CONTROL_PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; "
                    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                    "font-src 'self' https://fonts.gstatic.com; "
                    "script-src 'self' 'unsafe-inline'; "
                    "img-src 'self' data:",
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if route == "/api/status":
                with STATE_LOCK:
                    state = dict(STATE)
                state["watchdog_pid"] = os.getpid()
                state["watchdog_uptime_s"] = time.time() - float(state["watchdog_started_at"])
                try:
                    state["server"] = server_json("GET", "/v1/stats", timeout=3.0)
                except (RuntimeError, ValueError) as error:
                    state["server_error"] = str(error)
                self.send_json(200, state)
                return
            if route == "/api/settings":
                config = read_bot_config()
                safe_config = {k: v for k, v in config.items() if k != "bot_token"}
                self.send_json(
                    200,
                    {
                        "runtime": merged_runtime_config(),
                        "config": safe_config,
                        "builtin_prompts": builtin_prompts(),
                    },
                )
                return
            if route == "/api/activity":
                self.send_json(200, {"activity": recent_activity()})
                return
            if route == "/api/server/config":
                self.send_json(200, server_json("GET", "/config", timeout=5.0))
                return
            self.send_json(404, {"error": "not found"})
        except RequestRejected as error:
            self.send_json(403, {"error": str(error)})
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            self.send_json(502, {"error": str(error)})

    def do_POST(self) -> None:
        route = self.path.split("?", 1)[0]
        try:
            self.check_origin(require_json_body=True)
            body = self.read_json()
            if route == "/api/settings":
                with SETTINGS_LOCK:
                    settings = validate_runtime_config(body)
                    write_runtime_config(settings)
                if CONTROL_LOGGER is not None:
                    emit(CONTROL_LOGGER, "runtime_settings_saved", version=settings["version"])
                self.send_json(200, {"saved": True, "runtime": settings})
                return
            if route == "/api/playground":
                if not isinstance(body, dict):
                    raise ValueError("playground body must be an object")
                messages = body.get("messages")
                generation = body.get("generation")
                if not isinstance(messages, list) or not isinstance(generation, dict):
                    raise ValueError("playground requires messages and generation")
                request_body = {"messages": messages, "stream": False, "truncation": "auto"}
                for key in (
                    "temperature", "top_k", "top_p", "min_p", "presence_penalty",
                    "frequency_penalty", "repeat_penalty", "max_tokens",
                ):
                    request_body[key] = generation.get(key)
                request_body["enable_thinking"] = bool(generation.get("thinking", False))
                self.send_json(200, server_json("POST", "/v1/chat/completions", request_body, timeout=900.0))
                return
            if route == "/api/server/model":
                if not isinstance(body, dict) or not isinstance(body.get("model"), str):
                    raise ValueError("model must be a string")
                surface = server_json("GET", "/config", timeout=5.0)
                if not isinstance(surface, dict) or not isinstance(surface.get("surface"), dict):
                    raise RuntimeError("model server returned no editable config")
                config_surface = surface["surface"]
                assert isinstance(config_surface, dict)
                effective = config_surface.get("config")
                if not isinstance(effective, dict):
                    raise RuntimeError("model server returned no effective config")
                selected = body["model"].strip()
                # Only a bare file name from the server's own model directory; a path here would let
                # a caller point the server at an arbitrary file.
                if not selected or Path(selected).name != selected:
                    raise ValueError("model must be a file name in the server's model directory")
                old_model = Path(str(effective.get("model", "")))
                selected = str(old_model.parent / selected)
                effective["model"] = selected
                saved = server_json("POST", "/config", effective, timeout=10.0)
                try:
                    restarted = server_json("POST", "/restart", {}, timeout=10.0)
                except RuntimeError as error:
                    restarted = {"accepted": True, "connection": str(error)}
                self.send_json(200, {"saved": saved, "restart": restarted, "model": selected})
                return
            self.send_json(404, {"error": "not found"})
        except RequestRejected as error:
            self.send_json(403, {"error": str(error)})
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})


def start_control_server(logger: logging.Logger) -> ThreadingHTTPServer:
    global CONTROL_LOGGER
    CONTROL_LOGGER = logger
    server = ThreadingHTTPServer((CONTROL_HOST, CONTROL_PORT), ControlHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="cadmus-control", daemon=True).start()
    emit(logger, "control_started", url=f"http://{CONTROL_HOST}:{CONTROL_PORT}/")
    return server


def main() -> int:
    global DUMP_DIR

    logger = make_logger()
    if not PROGRAM.is_file():
        emit(logger, "program_missing", path=str(PROGRAM))
        return 2
    if not CONFIG.is_file():
        emit(logger, "config_missing", path=str(CONFIG))
        return 2
    if not CONTROL_PAGE.is_file():
        emit(logger, "control_page_missing", path=str(CONTROL_PAGE))
        return 2

    dump_type, configured_dump_dir = windows_local_dump_config(PROGRAM.name)
    if configured_dump_dir is not None:
        DUMP_DIR = configured_dump_dir
    if dump_type == 1:
        emit(logger, "wer_ready", dump_dir=str(DUMP_DIR), dump_type="minidump")
    elif IS_WINDOWS:
        emit(
            logger,
            "wer_not_ready",
            reason="not configured" if dump_type is None else f"unsafe dump type {dump_type}",
        )

    command = [str(PROGRAM), "--config", str(CONFIG)]
    child_env = os.environ.copy()
    child_env["CADMUS_STOP_FILE"] = str(STOP_FILE)
    stopping = threading.Event()
    child: subprocess.Popen[str] | None = None
    try:
        control_server = start_control_server(logger)
    except OSError as error:
        # Binding the control port is the single-instance check, so claim the PID file only after it
        # succeeds — otherwise a second watchdog would overwrite and then delete the running one's.
        emit(logger, "control_start_failed", host=CONTROL_HOST, port=CONTROL_PORT, error=str(error))
        return 2
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")

    def stop_handler(_signum: int, _frame: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    failures = 0
    emit(logger, "watchdog_started", pid=os.getpid(), command=command, cwd=str(ROOT))

    try:
        while not stopping.is_set():
            try:
                STOP_FILE.unlink()
            except FileNotFoundError:
                pass
            started_at = time.time()
            try:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=(
                        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                        if IS_WINDOWS
                        else 0
                    ),
                    env=child_env,
                )
            except OSError as error:
                emit(logger, "spawn_failed", error=str(error))
                return 2

            emit(logger, "child_started", pid=child.pid)
            with STATE_LOCK:
                STATE["child_pid"] = child.pid
                STATE["child_started_at"] = started_at
                STATE["child_exit_code"] = None
                STATE["restart_delay"] = 0.0
            output = threading.Thread(target=stream_child, args=(child, logger), daemon=True)
            output.start()
            stop_requested_at = 0.0
            while child.poll() is None:
                if stopping.is_set() and stop_requested_at == 0.0:
                    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
                    STOP_FILE.write_text("stop\n", encoding="ascii")
                    stop_requested_at = time.monotonic()
                    emit(logger, "stop_file_requested", path=str(STOP_FILE))
                if stop_requested_at and time.monotonic() - stop_requested_at > STOP_TIMEOUT:
                    child.terminate()
                time.sleep(0.25)

            child_pid = child.pid
            return_code = child.wait()
            output.join(timeout=2.0)
            uptime = time.time() - started_at
            emit(logger, "child_exited", pid=child_pid, code=return_code, uptime_seconds=uptime)
            with STATE_LOCK:
                STATE["child_pid"] = 0
                STATE["child_exit_code"] = return_code
            child = None
            if stopping.is_set() or return_code == 0:
                return 0

            dump_path = wait_for_dump(PROGRAM.name, started_at) if dump_type == 1 else None
            failures = 0 if uptime >= STABLE_SECONDS else failures + 1
            delay = min(2.0 ** min(failures, 6), MAX_RESTART_DELAY)
            with STATE_LOCK:
                STATE["restart_delay"] = delay
            bundle: Path | None = None
            try:
                bundle = collect_crash_bundle(child_pid, return_code, started_at, dump_path)
            except OSError as error:
                emit(logger, "crash_bundle_failed", error=str(error))
            emit(
                logger,
                "crash",
                code=return_code,
                uptime_seconds=uptime,
                restart_delay_seconds=delay,
                dump=str(dump_path) if dump_path is not None else "",
                bundle=str(bundle) if bundle is not None else "",
            )
            windows_balloon(
                "Cadmus crashed",
                f"Exit {return_code}; restarting in {delay:.0f}s"
                + (f"\nDump: {dump_path}" if dump_path is not None else "\nNo minidump produced"),
            )
            stopping.wait(delay)
    finally:
        control_server.shutdown()
        control_server.server_close()
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10.0)
        try:
            if PID_FILE.read_text(encoding="ascii").strip() == str(os.getpid()):
                PID_FILE.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
