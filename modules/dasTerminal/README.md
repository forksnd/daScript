# dasTerminal

`dasTerminal` is a renderless terminal state machine. It consumes terminal byte
streams and exposes a canonical value snapshot; it does not create a PTY or draw
pixels. Process transport belongs to the session host, and rendering belongs to
consumers such as dasImgui.

The snapshot deliberately includes both screen buffers, scrollback, grapheme
width and style per cell, cursor state, terminal modes, title/CWD metadata, and
unknown-sequence diagnostics. That schema is also intended to become the
session-host-to-UI IPC contract.

The first conformance slice covers text controls, wrapping and scrolling,
cursor movement, erase operations, common SGR colors/attributes, OSC title/CWD
and hyperlinks, alternate screen, selected DEC modes, device replies, and
incremental UTF-8 decoding. `tests-cpp/small/test_terminal.cpp` asserts semantic
snapshots and one-shot/byte-at-a-time/random chunk equivalence.
