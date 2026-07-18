# dasClipboard

`dasClipboard` provides a platform-neutral daScript clipboard API and a native
desktop transport. Use it from daScript with:

```das
require clipboard/clipboard
```

The portable in-memory transport is suitable for deterministic tests and
headless tools. Native Windows, macOS, and X11 builds expose the supported OS
clipboard formats; unsupported targets retain the API and report
`ClipboardStatus.unsupported`.

Scripts under `tools/clipboard_smoke` deliberately touch the real desktop
clipboard. They are manual interoperability checks and are not run by regular
tests or CI.
