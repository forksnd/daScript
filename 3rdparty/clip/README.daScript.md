# daScript vendoring notes

This directory contains the text-clipboard subset of
[`dacap/clip`](https://github.com/dacap/clip), version 1.15, pinned at commit
`f2bd226fd96f3431f4675d9c13aa7334dd06ddda`.

The upstream MIT license is preserved in `LICENSE.txt`. Image support,
examples, tests, and build-system files are intentionally omitted. daScript
builds this copy with `CLIP_ENABLE_IMAGE=0` and
`CLIP_ENABLE_LIST_FORMATS=0`.

Native backends are selected by the top-level daScript CMake build:

- Windows: Win32 clipboard
- macOS: Cocoa pasteboard
- Linux: X11/XWayland through XCB when development files are available
- other targets, including Wayland-only sessions and Emscripten: explicit
  unsupported/unavailable status from the public wrapper
