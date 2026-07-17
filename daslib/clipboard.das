options gen2
options indenting = 4

module clipboard shared

require clipboard_core public
require daslib/utf8_utils
require strings

//! Platform-neutral system clipboard access.
//!
//! Text crossing this boundary is UTF-8. Desktop operations are synchronous;
//! unsupported platforms and sessions without an available desktop report an
//! explicit status instead of falling back to a process-local clipboard.

enum public ClipboardStatus {
    ok
    no_text
    busy
    unavailable
    unsupported
    invalid_utf8
    failed
}

bitfield public ClipboardCapabilities {
    read_text
    write_text
    clear
    synchronous
}

def public clipboard_capabilities() : ClipboardCapabilities {
    return unsafe(reinterpret<ClipboardCapabilities>(_clipboard_capabilities()))
}

def public clipboard_set_text(text : string) : ClipboardStatus {
    //! Replace the system clipboard with UTF-8 text. Empty text clears it.
    if (!is_utf8_string_valid(text)) return ClipboardStatus.invalid_utf8
    return unsafe(reinterpret<ClipboardStatus>(
        _clipboard_set_text(text, length(text))))
}

def public clipboard_get_text(var text : string&) : ClipboardStatus {
    //! Read UTF-8 text into `text`. The output is empty for every non-ok result.
    var raw_status = int(ClipboardStatus.failed)
    text := _clipboard_get_text(raw_status)
    let status = unsafe(reinterpret<ClipboardStatus>(raw_status))
    if (status == ClipboardStatus.ok && !is_utf8_string_valid(text)) {
        text = ""
        return ClipboardStatus.invalid_utf8
    }
    return status
}

def public clipboard_clear() : ClipboardStatus {
    return unsafe(reinterpret<ClipboardStatus>(_clipboard_clear()))
}
