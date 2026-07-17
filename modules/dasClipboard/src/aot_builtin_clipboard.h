#pragma once

#include "daScript/simulate/simulate.h"

namespace das {

    // Integer values are mirrored by clipboard/clipboard.das. Keep the C++
    // boundary primitive so the public script API can remain a normal enum.
    enum class ClipboardStatus : int32_t {
        ok = 0,
        no_text,
        busy,
        unavailable,
        unsupported,
        invalid_utf8,
        failed,
        no_content,
        invalid_image
    };

    enum ClipboardCapability : int32_t {
        clipboard_read_text = 1 << 0,
        clipboard_write_text = 1 << 1,
        clipboard_clear = 1 << 2,
        clipboard_synchronous = 1 << 3,
        clipboard_read_rich_text = 1 << 4,
        clipboard_write_rich_text = 1 << 5,
        clipboard_read_image = 1 << 6,
        clipboard_write_image = 1 << 7
    };

    enum ClipboardFormat : int32_t {
        clipboard_plain_text = 1 << 0,
        clipboard_markdown = 1 << 1,
        clipboard_html = 1 << 2,
        clipboard_uri_list = 1 << 3,
        clipboard_image_rgba8 = 1 << 4
    };

#if DAS_ENABLE_DLL && defined(DAS_CLIPBOARD_IMPORTS)
#define DAS_CLIPBOARD_API DAS_IMPORT_DLL
#elif DAS_ENABLE_DLL && defined(DAS_MOD_EXPORTS)
#define DAS_CLIPBOARD_API DAS_EXPORT_DLL
#else
#define DAS_CLIPBOARD_API
#endif

    DAS_CLIPBOARD_API int32_t builtin_clipboard_capabilities();
    DAS_CLIPBOARD_API int32_t builtin_clipboard_set_text(const char * text, int32_t text_size);
    DAS_CLIPBOARD_API char * builtin_clipboard_get_text(int32_t & status, Context * context, LineInfoArg * at);
    // Stable until the next call on the same thread. Intended for native UI
    // callback APIs (such as ImGuiPlatformIO), not exposed to daScript.
    DAS_CLIPBOARD_API const char * builtin_clipboard_get_text_temporary(int32_t & status);
    DAS_CLIPBOARD_API int32_t builtin_clipboard_clear();
    DAS_CLIPBOARD_API int32_t builtin_clipboard_set_content(
        int32_t formats,
        const char * plainText, int32_t plainTextSize,
        const char * markdown, int32_t markdownSize,
        const char * html, int32_t htmlSize,
        const char * uriList, int32_t uriListSize,
        const TArray<uint8_t> & imageRgba8,
        int32_t imageWidth, int32_t imageHeight, int32_t imageStride);
    DAS_CLIPBOARD_API int32_t builtin_clipboard_get_content(
        int32_t & formats,
        char * & plainText, char * & markdown, char * & html, char * & uriList,
        TArray<uint8_t> & imageRgba8,
        int32_t & imageWidth, int32_t & imageHeight, int32_t & imageStride,
        Context * context, LineInfoArg * at);

}

#undef DAS_CLIPBOARD_API
