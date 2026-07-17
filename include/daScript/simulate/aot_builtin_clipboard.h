#pragma once

#include "daScript/simulate/simulate.h"

namespace das {

    // Integer values are mirrored by daslib/clipboard.das.  Keep the C++
    // boundary primitive so the public script API can remain a normal enum.
    enum class ClipboardStatus : int32_t {
        ok = 0,
        no_text,
        busy,
        unavailable,
        unsupported,
        invalid_utf8,
        failed
    };

    enum ClipboardCapability : int32_t {
        clipboard_read_text = 1 << 0,
        clipboard_write_text = 1 << 1,
        clipboard_clear = 1 << 2,
        clipboard_synchronous = 1 << 3
    };

    DAS_API int32_t builtin_clipboard_capabilities();
    DAS_API int32_t builtin_clipboard_set_text(const char * text, int32_t text_size);
    DAS_API char * builtin_clipboard_get_text(int32_t & status, Context * context, LineInfoArg * at);
    // Stable until the next call on the same thread. Intended for native UI
    // callback APIs (such as ImGuiPlatformIO), not exposed to daScript.
    DAS_API const char * builtin_clipboard_get_text_temporary(int32_t & status);
    DAS_API int32_t builtin_clipboard_clear();

}
