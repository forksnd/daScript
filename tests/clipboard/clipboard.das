options gen2
options indenting = 4

require dastest/testing_boost public
require daslib/clipboard

def private restore_clipboard(status : ClipboardStatus; text : string) {
    if (status == ClipboardStatus.ok) {
        clipboard_set_text(text)
    } else {
        clipboard_clear()
    }
}

[test]
def test_clipboard_utf8_text_roundtrip(t : T?) {
    let caps = clipboard_capabilities()
    if (!caps.read_text || !caps.write_text || !caps.clear) {
        t |> skip("native system clipboard is unavailable on this platform/session")
        return
    }

    var original : string
    let original_status = clipboard_get_text(original)
    if (original_status != ClipboardStatus.ok &&
        original_status != ClipboardStatus.no_text) {
        t |> skip("could not preserve the current system clipboard")
        return
    }
    let payload = "clipboard Привет 日本語 👩‍💻\nsecond line\r\nthird line"
    let set_status = clipboard_set_text(payload)
    var actual : string
    let get_status = clipboard_get_text(actual)
    restore_clipboard(original_status, original)

    t |> equal(set_status, ClipboardStatus.ok, "write UTF-8 text")
    t |> equal(get_status, ClipboardStatus.ok, "read UTF-8 text")
    t |> equal(actual, payload, "preserve Unicode and line endings")
}

[test]
def test_clipboard_empty_text_clears(t : T?) {
    let caps = clipboard_capabilities()
    if (!caps.read_text || !caps.write_text || !caps.clear) {
        t |> skip("native system clipboard is unavailable on this platform/session")
        return
    }

    var original : string
    let original_status = clipboard_get_text(original)
    if (original_status != ClipboardStatus.ok &&
        original_status != ClipboardStatus.no_text) {
        t |> skip("could not preserve the current system clipboard")
        return
    }
    let set_status = clipboard_set_text("")
    var actual : string
    let get_status = clipboard_get_text(actual)
    restore_clipboard(original_status, original)

    t |> equal(set_status, ClipboardStatus.ok, "empty write clears clipboard")
    t |> equal(get_status, ClipboardStatus.no_text, "cleared clipboard has no text")
    t |> success(empty(actual), "non-ok read clears output")
}
