# Clipboard smoke tests

These tests intentionally exercise the real desktop clipboard. They are not
under `/tests`, are not discovered by `dastest`, and must not run in regular CI.

Run the native supported-subset roundtrip explicitly from the repository root:

```powershell
.\bin\Release\daslang.exe modules\dasClipboard\tools\clipboard_smoke\native_roundtrip.das
```

The probe offers plain UTF-8 text, Markdown, an HTML fragment, a URI list, and
a host-supplied 2x2 straight-alpha RGBA8 image in one native transaction. It
then takes one native snapshot, verifies every representation, and restores the
previous supported subset only if its unique plain-text marker still owns the
clipboard. A clipboard change made by the user or another app always wins.

This native-only script cannot preserve arbitrary third-party/private formats.
Do not run it when the clipboard contains data that cannot be reconstructed
from daScript's platform-independent subset. Use the guarded Windows smoke
below when stronger restoration and external interoperability checks matter.

On Windows, the external-consumer smoke adds a `.NET WinForms` consumer in both
directions and guards restoration in `finally`, including when a daScript child
fails:

```powershell
.\modules\dasClipboard\tools\clipboard_smoke\external_consumers.ps1
```

It eagerly asks WinForms and raw Win32 code to consume daScript's plain text,
CF_HTML, raw `text/markdown`, raw `text/uri-list`, and image offers. It then
publishes those formats in one raw Win32 transaction and asks daScript to
consume them. Before writing, the guardian eagerly snapshots every
HGLOBAL-backed clipboard format as exact bytes. Those bytes are restored only
while one of the smoke markers still owns the clipboard; a newer
user/application copy is never overwritten. Non-HGLOBAL handle formats remain
outside this restoration subset.
