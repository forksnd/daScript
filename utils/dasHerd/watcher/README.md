# dasHerd watcher

The watcher is dasHerd's local detachable terminal host. One persistent JIT
process owns multiple PTYs, terminal emulators, and durable session logs while
browser and future native/live-command surfaces attach as clients.

Run it from the repository root:

```powershell
bin\Release\daslang.exe -jit utils\dasHerd\watcher\main.das
```

The watcher prints a token-bearing control-panel URL. Optional arguments after
`--` are `--port=9191`, `--token=<token>`, and `--log-root=<directory>`.

This first slice binds only to `127.0.0.1`. Its startup token prevents
accidental access from an unrelated local page; it is not the future remote
authentication design. Do not expose this port through a proxy or tunnel.

HTTP endpoints under `/api/v1/` provide health, session listing, launch,
input, resize, and termination. `/ws` provides attach/detach, controller
leases, heartbeats, input, resize, and terminal snapshots. Every request must
include the startup token as the `token` query parameter.

Session logs are stored under
`logs/dasHerd/watcher/sessions/<session-id>/`. Closing a client releases its
controller lease but does not stop its child. Killing the watcher necessarily
loses its live PTYs; already-flushed logs remain readable.
