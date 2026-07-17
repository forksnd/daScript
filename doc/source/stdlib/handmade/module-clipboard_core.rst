The clipboard module provides synchronous access to the operating system's
text clipboard. Text is always exchanged as UTF-8, including supplementary
Unicode characters such as emoji.

All public symbols are in the ``clipboard`` module. Require the promoted
wrapper rather than the internal ``clipboard_core`` module:

.. code-block:: das

    require clipboard/clipboard

    [export]
    def main() {
        if (clipboard_set_text("Hello, UTF-8") == ClipboardStatus.ok) {
            var text : string
            if (clipboard_get_text(text) == ClipboardStatus.ok) {
                print("{text}\n")
            }
        }
    }

Use ``clipboard_capabilities`` before enabling clipboard UI. Unsupported
targets, unavailable desktop sessions, transient contention, and invalid
UTF-8 are reported through ``ClipboardStatus``; the API does not substitute a
process-local clipboard.
