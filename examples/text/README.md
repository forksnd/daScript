# Text viewer

A small Unicode plain-text and Markdown viewer built with dasImgui.

```text
daslang examples/text/main.das -- README.md
daslang examples/text/main.das -- --plain LICENSE
daslang-live examples/text/main.das -- README.md
```

`.md` and `.markdown` files start in rendered Markdown mode; other files start
in plain-text mode. Use `--markdown` or `--plain` before the path to override
that choice. The View menu and bottom mode buttons can switch at runtime.

Markdown source keeps every original byte visible while styling markers such
as **strong delimiters**, `code fences`, and link punctuation. Unicode remains
cell-aligned across combining text, CJK, and color emoji: é · 中文 · 👩‍💻.

The main document window is dockable. Zoom is a fixed 5% step from 50–300% at
the bottom-right, and Ctrl+mouse-wheel adjusts it while the document is hovered.
On macOS, the usual Command key is used in place of Ctrl.

**File / Open...** and Ctrl+O launch the selected document in a new viewer
window, leaving the current document and its view state intact.
