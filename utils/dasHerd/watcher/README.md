# dasHerd watcher

The watcher is dasHerd's local detachable terminal host. One persistent JIT
process owns multiple PTYs, terminal emulators, and durable session logs while
native/live-command surfaces attach as clients. The browser panel is an
interactive debug and recovery fallback; the ImGui client remains the primary UI.

Run it from the repository root:

```powershell
bin\Release\daslang.exe -jit utils\dasHerd\watcher\main.das
```

Run the docked native client with the token printed by the watcher:

```powershell
bin\Release\daslang-live.exe utils\dasHerd\watcher\rich_client.das -- --token=<token>
```

The client opens maximized at 150% zoom. Its left pane lists sessions and can
launch or terminate them; the right pane attaches with control and renders raw
PTY bytes through the daScript terminal emulator. `--port=9191` is available on
both processes when the default port is occupied.

The watcher prints a token-bearing control-panel URL. Optional arguments after
`--` are `--port=9191`, `--token=<token>`, and `--log-root=<directory>`.

This first slice binds only to `127.0.0.1`. Its startup token prevents
accidental access from an unrelated local page; it is not the future remote
authentication design. Do not expose this port through a proxy or tunnel.

HTTP endpoints under `/api/v1/` provide health, session listing, launch,
input, resize, termination, and a deferred `POST /gc` diagnostic. `/ws` provides attach/detach, controller
leases, heartbeats, session listing/launch/termination, input, resize, terminal
snapshots, and bounded raw PTY replay.
Raw output is delivered as binary WebSocket frames, so the browser can render
the same color and control sequences and send keyboard input back to the PTY.
Every request must include the startup token as the `token` query parameter.

The script exposes `init`, `update`, and `shutdown` for lifecycle hosts. Its
standalone loop performs optional GC only after `update` returns, when request
temporaries are dead; all state that must survive collection is globally rooted.
Automatic collection is fragmentation-driven and rate-limited. `POST /gc`
coalesces a request for a validated collection at that same safe boundary.

The browser terminal vendors xterm.js 5.5.0 with the matching fit 0.10.0,
Unicode 11 0.8.0, and web-links 0.11.0 addons under `vendor/xterm/`. It works
offline; the corresponding MIT license files are kept beside the assets.

Session logs are stored under
`logs/dasHerd/watcher/sessions/<session-id>/`. Closing a client releases its
controller lease but does not stop its child. Killing the watcher necessarily
loses its live PTYs; already-flushed logs remain readable.

The native terminal registers its full `ImGuiTerminalState` with the live-command
surface. `imgui_snapshot` can therefore read terminal text, selection, geometry,
cursor/blink state, performance counters, and input revisions without a screenshot;
`imgui_click` plus `imgui_key_type` drives the same input path as a human user.
