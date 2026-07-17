#include "daScript/ast/ast.h"
#include "daScript/ast/ast_interop.h"
#include "daScript/daScriptModule.h"
#include "daScript/simulate/aot_builtin_clipboard.h"
#include "clip.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace das {

    namespace {

        constexpr int32_t desktopCapabilities =
            clipboard_read_text | clipboard_write_text |
            clipboard_clear | clipboard_synchronous;

        void ignoreClipError(clip::ErrorCode) {
            // The public API reports failures as ClipboardStatus.  The vendored
            // library's default handler throws, which would erase that contract.
        }

        bool nativeClipboardAvailable() {
#if !DAS_CLIPBOARD_NATIVE_SUPPORTED
            return false;
#elif defined(__linux__)
            // clip's Linux backend is XCB.  A Wayland session with XWayland has
            // DISPLAY and works; a native Wayland-only session is unavailable.
            const char * display = std::getenv("DISPLAY");
            return display && *display;
#else
            return true;
#endif
        }

        ClipboardStatus unavailableStatus() {
#if DAS_CLIPBOARD_NATIVE_SUPPORTED
            return ClipboardStatus::unavailable;
#else
            return ClipboardStatus::unsupported;
#endif
        }

        ClipboardStatus lockClipboard(clip::lock & lock) {
            return lock.locked() ? ClipboardStatus::ok : ClipboardStatus::busy;
        }

        int32_t statusCode(ClipboardStatus status) {
            return static_cast<int32_t>(status);
        }

        ClipboardStatus getClipboardText(std::string & text) {
            text.clear();
            if (!nativeClipboardAvailable()) return unavailableStatus();
            clip::lock lock;
            ClipboardStatus status = lockClipboard(lock);
            if (status != ClipboardStatus::ok) return status;
            const clip::format format = clip::text_format();
            if (!lock.is_convertible(format)) return ClipboardStatus::no_text;
            const size_t bufferSize = lock.get_data_length(format);
            if (bufferSize == 0) return ClipboardStatus::ok;
            std::vector<char> buffer(bufferSize + 1, 0);
            if (!lock.get_data(format, buffer.data(), bufferSize)) {
                return ClipboardStatus::failed;
            }
            size_t textSize = bufferSize;
            while (textSize > 0 && buffer[textSize - 1] == 0) --textSize;
            text.assign(buffer.data(), textSize);
            return ClipboardStatus::ok;
        }

    }

    int32_t builtin_clipboard_capabilities() {
        return nativeClipboardAvailable() ? desktopCapabilities : 0;
    }

    int32_t builtin_clipboard_set_text(const char * text, int32_t text_size) {
        if (text_size < 0 || (!text && text_size != 0)) {
            return statusCode(ClipboardStatus::failed);
        }
        if (!nativeClipboardAvailable()) return statusCode(unavailableStatus());
        clip::lock lock;
        ClipboardStatus status = lockClipboard(lock);
        if (status != ClipboardStatus::ok) {
            return statusCode(status);
        }
        if (!lock.clear()) {
            return statusCode(ClipboardStatus::failed);
        }
        // Clearing is the correct system representation of empty clipboard
        // text.  clip's Windows text backend intentionally rejects zero bytes.
        if (text_size == 0) {
            return statusCode(ClipboardStatus::ok);
        }
        return statusCode(lock.set_data(clip::text_format(), text,
                                        static_cast<size_t>(text_size))
            ? ClipboardStatus::ok : ClipboardStatus::failed);
    }

    char * builtin_clipboard_get_text(int32_t & resultStatus, Context * context, LineInfoArg * at) {
        std::string text;
        ClipboardStatus status = getClipboardText(text);
        resultStatus = statusCode(status);
        return context->allocateString(text, at);
    }

    const char * builtin_clipboard_get_text_temporary(int32_t & resultStatus) {
        static thread_local std::string text;
        ClipboardStatus status = getClipboardText(text);
        resultStatus = statusCode(status);
        return status == ClipboardStatus::ok ? text.c_str() : nullptr;
    }

    int32_t builtin_clipboard_clear() {
        if (!nativeClipboardAvailable()) return statusCode(unavailableStatus());
        clip::lock lock;
        ClipboardStatus status = lockClipboard(lock);
        if (status != ClipboardStatus::ok) {
            return statusCode(status);
        }
        return statusCode(lock.clear() ? ClipboardStatus::ok
                                      : ClipboardStatus::failed);
    }

    class Module_Clipboard final : public Module {
    public:
        Module_Clipboard() : Module("clipboard_core") {
            ModuleLibrary lib(this);
            lib.addBuiltInModule();
            addBuiltinDependency(lib, Module::require("strings"));

            clip::set_error_handler(ignoreClipError);

            addExtern<DAS_BIND_FUN(builtin_clipboard_capabilities)>(*this, lib,
                "_clipboard_capabilities", SideEffects::accessExternal,
                "builtin_clipboard_capabilities");
            addExtern<DAS_BIND_FUN(builtin_clipboard_set_text)>(*this, lib,
                "_clipboard_set_text", SideEffects::modifyExternal,
                "builtin_clipboard_set_text")
                    ->args({"text", "text_size"});
            addExtern<DAS_BIND_FUN(builtin_clipboard_get_text)>(*this, lib,
                "_clipboard_get_text", SideEffects::modifyArgumentAndExternal,
                "builtin_clipboard_get_text")
                    ->args({"status", "context", "at"});
            addExtern<DAS_BIND_FUN(builtin_clipboard_clear)>(*this, lib,
                "_clipboard_clear", SideEffects::modifyExternal,
                "builtin_clipboard_clear");

            verifyBuiltinNames(uint32_t(VerifyBuiltinFlags::verifyAll));
            verifyAotReady();
        }

        ModuleAotType aotRequire(TextWriter & tw) const override {
            tw << "#include \"daScript/simulate/aot_builtin_clipboard.h\"\n";
            return ModuleAotType::cpp;
        }
    };

}

REGISTER_MODULE_IN_NAMESPACE(Module_Clipboard, das);
