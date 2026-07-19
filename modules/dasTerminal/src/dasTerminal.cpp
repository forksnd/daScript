#include "daScript/misc/platform.h"
#include "daScript/ast/ast.h"
#include "daScript/ast/ast_handle.h"
#include "daScript/ast/ast_interop.h"

#include "aot_builtin_terminal.h"

#include <algorithm>

MAKE_TYPE_FACTORY(terminal_handle, das_terminal::TerminalHandle)

namespace das_terminal {
namespace {

const BufferSnapshot * selectBuffer(const Snapshot & snapshot, int32_t screen) {
    if (screen == 0) return &snapshot.normal;
    if (screen == 1) return &snapshot.alternate;
    return nullptr;
}

int32_t colorKind(const Color & color) {
    return static_cast<int32_t>(color.kind);
}

} // namespace

das::smart_ptr<TerminalHandle> builtin_terminal_create(int32_t columns, int32_t rows) {
    return das::make_smart<TerminalHandle>(columns, rows);
}

void builtin_terminal_feed(
    const das::smart_ptr<TerminalHandle> & terminal, const char * bytes, int32_t count) {
    if (!terminal || !bytes || count <= 0) return;
    terminal->state.feed(reinterpret_cast<const uint8_t *>(bytes), static_cast<size_t>(count));
}

void builtin_terminal_resize(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t columns, int32_t rows) {
    if (terminal) terminal->state.resize(columns, rows);
}

int32_t builtin_terminal_columns(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal ? terminal->state.snapshot().columns : 0;
}

int32_t builtin_terminal_rows(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal ? terminal->state.snapshot().rows : 0;
}

bool builtin_terminal_alternate_active(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal && terminal->state.snapshot().alternate_active;
}

int32_t builtin_terminal_mode_bits(const das::smart_ptr<TerminalHandle> & terminal) {
    if (!terminal) return 0;
    const Modes & modes = terminal->state.snapshot().modes;
    int32_t result = 0;
    if (modes.auto_wrap) result |= 1 << 0;
    if (modes.application_cursor_keys) result |= 1 << 1;
    if (modes.bracketed_paste) result |= 1 << 2;
    if (modes.focus_reporting) result |= 1 << 3;
    if (modes.mouse_button_reporting) result |= 1 << 4;
    if (modes.mouse_any_reporting) result |= 1 << 5;
    if (modes.sgr_mouse_encoding) result |= 1 << 6;
    return result;
}

char * builtin_terminal_title(
    const das::smart_ptr<TerminalHandle> & terminal, das::Context * context, das::LineInfoArg * at) {
    const std::string value = terminal ? terminal->state.snapshot().title : std::string();
    return context->allocateString(value, at);
}

char * builtin_terminal_current_directory(
    const das::smart_ptr<TerminalHandle> & terminal, das::Context * context, das::LineInfoArg * at) {
    const std::string value = terminal ? terminal->state.snapshot().current_directory : std::string();
    return context->allocateString(value, at);
}

int32_t builtin_terminal_scrollback_rows(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen) {
    if (!terminal) return 0;
    const Snapshot snapshot = terminal->state.snapshot();
    const BufferSnapshot * buffer = selectBuffer(snapshot, screen);
    return buffer ? static_cast<int32_t>(buffer->scrollback.size()) : 0;
}

void builtin_terminal_cursor(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen,
    const das::TBlock<void, int32_t, int32_t, bool, bool, int32_t> & block,
    das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const Snapshot snapshot = terminal->state.snapshot();
    const BufferSnapshot * buffer = selectBuffer(snapshot, screen);
    if (!buffer) return;
    das::das_invoke<void>::invoke<int32_t, int32_t, bool, bool, int32_t>(
        context, at, block, buffer->cursor.row, buffer->cursor.column,
        buffer->cursor.visible, buffer->cursor.blinking, buffer->cursor.style);
}

void builtin_terminal_visit_cells(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen, bool scrollback,
    const CellBlock & block, das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const Snapshot snapshot = terminal->state.snapshot();
    const BufferSnapshot * buffer = selectBuffer(snapshot, screen);
    if (!buffer) return;
    const std::vector<std::vector<Cell>> & rows = scrollback ? buffer->scrollback : buffer->rows;
    for (size_t row = 0; row != rows.size(); ++row) {
        for (size_t column = 0; column != rows[row].size(); ++column) {
            const Cell & cell = rows[row][column];
            das::das_invoke<void>::invoke<
                int32_t, int32_t, const char *, int32_t, int32_t,
                int32_t, int32_t, int32_t, int32_t, int32_t,
                int32_t, int32_t, int32_t, int32_t, int32_t, const char *>(
                context, at, block,
                static_cast<int32_t>(row), static_cast<int32_t>(column), cell.grapheme.c_str(),
                cell.width, cell.attributes,
                colorKind(cell.foreground), cell.foreground.index,
                cell.foreground.red, cell.foreground.green, cell.foreground.blue,
                colorKind(cell.background), cell.background.index,
                cell.background.red, cell.background.green, cell.background.blue,
                cell.hyperlink.c_str());
        }
    }
}

void builtin_terminal_visit_unknown(
    const das::smart_ptr<TerminalHandle> & terminal,
    const das::TBlock<void, das::TTemporary<const char *>> & block,
    das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const std::vector<std::string> values = terminal->state.snapshot().unknown_sequences;
    for (const std::string & value : values)
        das::das_invoke<void>::invoke<const char *>(context, at, block, value.c_str());
}

void builtin_terminal_drain_replies(
    const das::smart_ptr<TerminalHandle> & terminal,
    const das::TBlock<void, das::TTemporary<const char *>> & block,
    das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const std::vector<std::string> values = terminal->state.drainReplies();
    for (const std::string & value : values)
        das::das_invoke<void>::invoke<const char *>(context, at, block, value.c_str());
}

void builtin_terminal_take_events(
    const das::smart_ptr<TerminalHandle> & terminal,
    const das::TBlock<void, das::TTemporary<const char *>> & block,
    das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const std::vector<std::string> values = terminal->state.takeEvents();
    for (const std::string & value : values)
        das::das_invoke<void>::invoke<const char *>(context, at, block, value.c_str());
}

} // namespace das_terminal

namespace das {

struct TerminalHandleAnnotation final
    : ManagedStructureAnnotation<das_terminal::TerminalHandle, false, true> {
    TerminalHandleAnnotation(ModuleLibrary & library)
        : ManagedStructureAnnotation("terminal_handle", library, "das_terminal::TerminalHandle") {}
};

class Module_Terminal final : public Module {
public:
    Module_Terminal() : Module("terminal_core") {
        ModuleLibrary lib(this);
        lib.addBuiltInModule();
        addAnnotation(new TerminalHandleAnnotation(lib));

        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_create)>(*this, lib,
            "_terminal_create", SideEffects::modifyExternal,
            "das_terminal::builtin_terminal_create")->args({"columns", "rows"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_feed)>(*this, lib,
            "_terminal_feed", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_feed")->args({"terminal", "bytes", "count"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_resize)>(*this, lib,
            "_terminal_resize", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_resize")->args({"terminal", "columns", "rows"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_columns)>(*this, lib,
            "_terminal_columns", SideEffects::none,
            "das_terminal::builtin_terminal_columns")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_rows)>(*this, lib,
            "_terminal_rows", SideEffects::none,
            "das_terminal::builtin_terminal_rows")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_alternate_active)>(*this, lib,
            "_terminal_alternate_active", SideEffects::none,
            "das_terminal::builtin_terminal_alternate_active")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_mode_bits)>(*this, lib,
            "_terminal_mode_bits", SideEffects::none,
            "das_terminal::builtin_terminal_mode_bits")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_title)>(*this, lib,
            "_terminal_title", SideEffects::modifyExternal,
            "das_terminal::builtin_terminal_title")->args({"terminal", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_current_directory)>(*this, lib,
            "_terminal_current_directory", SideEffects::modifyExternal,
            "das_terminal::builtin_terminal_current_directory")->args({"terminal", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_scrollback_rows)>(*this, lib,
            "_terminal_scrollback_rows", SideEffects::none,
            "das_terminal::builtin_terminal_scrollback_rows")->args({"terminal", "screen"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_cursor)>(*this, lib,
            "_terminal_cursor", SideEffects::invoke,
            "das_terminal::builtin_terminal_cursor")
                ->args({"terminal", "screen", "block", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_visit_cells)>(*this, lib,
            "_terminal_visit_cells", SideEffects::invoke,
            "das_terminal::builtin_terminal_visit_cells")
                ->args({"terminal", "screen", "scrollback", "block", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_visit_unknown)>(*this, lib,
            "_terminal_visit_unknown", SideEffects::invoke,
            "das_terminal::builtin_terminal_visit_unknown")
                ->args({"terminal", "block", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_drain_replies)>(*this, lib,
            "_terminal_drain_replies", SideEffects::invoke,
            "das_terminal::builtin_terminal_drain_replies")
                ->args({"terminal", "block", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_take_events)>(*this, lib,
            "_terminal_take_events", SideEffects::invoke,
            "das_terminal::builtin_terminal_take_events")
                ->args({"terminal", "block", "context", "at"});

        verifyBuiltinNames(uint32_t(VerifyBuiltinFlags::verifyAll));
        verifyAotReady();
    }

    ModuleAotType aotRequire(TextWriter & writer) const override {
        writer << "#include \"../modules/dasTerminal/src/aot_builtin_terminal.h\"\n";
        return ModuleAotType::cpp;
    }
};

REGISTER_DYN_MODULE(Module_Terminal, Module_Terminal);

} // namespace das

REGISTER_MODULE_IN_NAMESPACE(Module_Terminal, das);
