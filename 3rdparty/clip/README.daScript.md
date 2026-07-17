# daScript vendoring notes

This directory contains the text and custom-format clipboard subset of
[`dacap/clip`](https://github.com/dacap/clip), version 1.15, pinned at commit
`f2bd226fd96f3431f4675d9c13aa7334dd06ddda`.

The upstream MIT license is preserved in `LICENSE.txt`. Examples, tests, and
build-system files are intentionally omitted. daScript builds this copy with
`CLIP_ENABLE_LIST_FORMATS=0`. clip image support is enabled on macOS; Windows
uses the core wrapper's direct `CF_DIBV5` rail; and Linux image support remains
explicitly unavailable until its PNG backend is vendored with the dependency.

daScript's core wrapper writes standards-visible registered formats without
clip's private Windows custom-data length prefix. It also implements Windows
RGBA8 image transport directly through `CF_DIBV5`; no GPU readback is used.

Native backends are selected by the top-level daScript CMake build:

- Windows: Win32 clipboard
- macOS: Cocoa pasteboard
- Linux: X11/XWayland through XCB when development files are available
- other targets, including Wayland-only sessions and Emscripten: explicit
  unsupported/unavailable status from the public wrapper
