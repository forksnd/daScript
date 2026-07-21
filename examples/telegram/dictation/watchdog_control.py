"""Dictation-bot control page, loaded by utils/watchdog/watchdog.py as its control plugin.

The watchdog supervises; this module owns everything that knows what a dictation bot is --
dictation.toml, the prompt set, generation defaults and the recent-activity feed. The watchdog
calls start(logger, args) and injects `host`, its own module, before executing this file.

Saving settings writes cadmus-runtime.json and asks the watchdog for a config restart (exit 4).
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# Injected by watchdog.py before this module is executed; see start_control_plugin there.
host: object

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
CONFIG = ROOT / "dictation.toml"
CONTROL_PAGE = ROOT / "control.html"
RUNTIME_CONFIG = ROOT / "cadmus-runtime.json"
DEFAULTS_FILE = ROOT / "cadmus-defaults.json"
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
# Loop name is not `host`: that would read as the injected supervisor module.
ALLOWED_ORIGINS = frozenset(f"http://{hostport}" for hostport in ALLOWED_HOSTS)

# Saving settings is a read-modify-write (merge current, bump version, replace the file) and the
# control server is threaded, so concurrent savers must not interleave.
SETTINGS_LOCK = threading.Lock()
CONTROL_LOGGER: logging.Logger | None = None


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
                state = host.read_state()
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
                    host.emit(CONTROL_LOGGER, "runtime_settings_saved", version=settings["version"])
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


def start(logger: logging.Logger, args) -> ThreadingHTTPServer:
    """Entry point the watchdog calls; args is the supervisor's parsed namespace."""
    global CONTROL_LOGGER
    CONTROL_LOGGER = logger
    server = ThreadingHTTPServer((CONTROL_HOST, CONTROL_PORT), ControlHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="cadmus-control", daemon=True).start()
    host.emit(logger, "control_started", url=f"http://{CONTROL_HOST}:{CONTROL_PORT}/")
    return server
