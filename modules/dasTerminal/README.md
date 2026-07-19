# dasTerminal

`dasTerminal` provides a renderless terminal state machine and a separate PTY
transport boundary. The emulator consumes terminal byte streams and exposes a
canonical value snapshot; neither layer draws pixels. Long-lived process
ownership belongs to the session host, and rendering belongs to consumers such
as dasImgui.

The snapshot deliberately includes both screen buffers, scrollback, grapheme
width and style per cell, cursor state, terminal modes, title/CWD metadata, and
unknown-sequence diagnostics. That schema is also intended to become the
session-host-to-UI IPC contract.

The first conformance slice covers text controls, wrapping and scrolling,
cursor movement, erase operations, common SGR colors/attributes, OSC title/CWD
and hyperlinks, alternate screen, selected DEC modes, device replies, and
incremental UTF-8 decoding. It also encodes normalized key events and paste
payloads into mode-sensitive VT input bytes while keeping device replies on a
separate drain path for the future PTY host. `tests-cpp/small/test_terminal.cpp`
asserts semantic snapshots and one-shot/byte-at-a-time/random chunk equivalence.

`src/pty.h` is the transport-only process boundary. On Windows it launches a
child through dynamically resolved ConPTY APIs, exposes nonblocking reads,
writes, resize, root-process wait/exit status, and explicit termination, and
does not couple process bytes to a particular emulator or renderer. The live
test prefers PowerShell 7 (`pwsh`) and falls back to Windows PowerShell when it
is not installed; readiness and echo completion are observed through terminal
snapshots rather than sleeps.

The daScript `terminal_launch` convenience handle composes those separate
interfaces for local applications. `terminal_poll` feeds nonblocking PTY output
into the emulator and routes device replies back to the child; write, resize,
process-status, exit-code, and explicit-termination operations complete the
local process pump. The first ImGui consumer remains outside this module at
`modules/dasImgui/terminal/imgui_terminal.das`.
