#include "daScript/misc/platform.h"
#include "daScript/ast/ast.h"
#include "daScript/ast/ast_handle.h"
#include "daScript/ast/ast_interop.h"

#include "aot_builtin_terminal.h"

#include <algorithm>

MAKE_TYPE_FACTORY(terminal_handle, das_terminal::TerminalHandle)

namespace das_terminal {
namespace {

int32_t colorKind(const Color & color) {
    return static_cast<int32_t>(color.kind);
}

bool refreshProcessStatus(TerminalHandle & terminal) {
    if (terminal.process_exited) return true;
    if (!terminal.process) {
        if (terminal.transport_error.empty())
            terminal.transport_error = "terminal has no PTY process";
        return false;
    }
    std::string error;
    uint32_t exit_code = 0;
    if (terminal.process->wait(0, exit_code, error)) {
        terminal.process_exited = true;
        terminal.process_exit_code = exit_code;
        terminal.transport_error.clear();
        return true;
    }
    if (!error.empty()) terminal.transport_error = error;
    return false;
}

} // namespace

das::smart_ptr<TerminalHandle> builtin_terminal_create(int32_t columns, int32_t rows) {
    return das::make_smart<TerminalHandle>(columns, rows);
}

das::smart_ptr<TerminalHandle> builtin_terminal_launch(
    const char * command_line, int32_t command_count,
    const char * working_directory, int32_t directory_count,
    int32_t columns, int32_t rows) {
    das::smart_ptr<TerminalHandle> terminal = das::make_smart<TerminalHandle>(columns, rows);
    PtyProcessOptions options;
    if (command_line && command_count > 0)
        options.command_line.assign(command_line, static_cast<size_t>(command_count));
    if (working_directory && directory_count > 0)
        options.working_directory.assign(working_directory, static_cast<size_t>(directory_count));
    options.columns = columns;
    options.rows = rows;
    terminal->process = launchPtyProcess(options, terminal->transport_error);
    return terminal;
}

void builtin_terminal_feed(
    const das::smart_ptr<TerminalHandle> & terminal, const char * bytes, int32_t count) {
    if (!terminal || !bytes || count <= 0) return;
    terminal->state.feed(reinterpret_cast<const uint8_t *>(bytes), static_cast<size_t>(count));
}

void builtin_terminal_resize(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t columns, int32_t rows) {
    if (!terminal) return;
    terminal->state.resize(columns, rows);
    if (terminal->process) {
        if (!terminal->process->resize(columns, rows, terminal->transport_error)) return;
        terminal->transport_error.clear();
    }
}

int32_t builtin_terminal_poll(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t maximum_bytes) {
    if (!terminal) return static_cast<int32_t>(PtyReadStatus::error);
    if (!terminal->process) {
        if (terminal->transport_error.empty())
            terminal->transport_error = "terminal has no PTY process";
        return static_cast<int32_t>(PtyReadStatus::error);
    }
    terminal->transport_error.clear();
    std::string bytes;
    const PtyReadStatus status = terminal->process->read(
        bytes, terminal->transport_error,
        static_cast<size_t>(std::max(1, maximum_bytes)));
    if (status != PtyReadStatus::data) return static_cast<int32_t>(status);
    terminal->state.feed(bytes);
    const std::vector<std::string> replies = terminal->state.drainReplies();
    for (const std::string & reply : replies) {
        if (!terminal->process->write(reply, terminal->transport_error))
            return static_cast<int32_t>(PtyReadStatus::error);
    }
    terminal->transport_error.clear();
    return static_cast<int32_t>(status);
}

bool builtin_terminal_write(
    const das::smart_ptr<TerminalHandle> & terminal, const char * bytes, int32_t count) {
    if (!terminal) return false;
    if (!terminal->process) {
        if (terminal->transport_error.empty())
            terminal->transport_error = "terminal has no PTY process";
        return false;
    }
    if (count < 0 || (!bytes && count)) {
        terminal->transport_error = "terminal input bytes are invalid";
        return false;
    }
    if (!terminal->process->write(
            reinterpret_cast<const uint8_t *>(bytes), static_cast<size_t>(count),
            terminal->transport_error)) return false;
    terminal->transport_error.clear();
    return true;
}

int32_t builtin_terminal_process_id(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal && terminal->process
        ? static_cast<int32_t>(terminal->process->processId()) : 0;
}

bool builtin_terminal_process_exited(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal && refreshProcessStatus(*terminal);
}

int64_t builtin_terminal_exit_code(const das::smart_ptr<TerminalHandle> & terminal) {
    if (!terminal || !refreshProcessStatus(*terminal)) return -1;
    return static_cast<int64_t>(terminal->process_exit_code);
}

bool builtin_terminal_terminate(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t exit_code) {
    if (!terminal) return false;
    if (!terminal->process) {
        if (terminal->transport_error.empty())
            terminal->transport_error = "terminal has no PTY process";
        return false;
    }
    if (refreshProcessStatus(*terminal)) return true;
    terminal->transport_error.clear();
    if (!terminal->process->terminate(static_cast<uint32_t>(exit_code),
                                      terminal->transport_error)) return false;
    refreshProcessStatus(*terminal);
    return true;
}

char * builtin_terminal_transport_error(
    const das::smart_ptr<TerminalHandle> & terminal,
    das::Context * context, das::LineInfoArg * at) {
    const std::string value = terminal ? terminal->transport_error : std::string();
    return context->allocateString(value.data(), value.size(), at);
}

char * builtin_terminal_encode_key(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t key,
    const char * text, int32_t count, bool shift, bool alt, bool control,
    das::Context * context, das::LineInfoArg * at) {
    std::string value;
    if (terminal && key >= static_cast<int32_t>(Key::text) &&
        key <= static_cast<int32_t>(Key::f12)) {
        const std::string key_text = text && count > 0
            ? std::string(text, static_cast<size_t>(count)) : std::string();
        value = terminal->state.encodeKey(KeyEvent(
            static_cast<Key>(key), key_text, shift, alt, control));
    }
    return context->allocateString(value.data(), value.size(), at);
}

char * builtin_terminal_encode_paste(
    const das::smart_ptr<TerminalHandle> & terminal, const char * text, int32_t count,
    das::Context * context, das::LineInfoArg * at) {
    const std::string paste = text && count > 0
        ? std::string(text, static_cast<size_t>(count)) : std::string();
    const std::string value = terminal ? terminal->state.encodePaste(paste) : std::string();
    return context->allocateString(value.data(), value.size(), at);
}

int32_t builtin_terminal_columns(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal ? terminal->state.columns() : 0;
}

int32_t builtin_terminal_rows(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal ? terminal->state.rows() : 0;
}

bool builtin_terminal_alternate_active(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal && terminal->state.alternateActive();
}

int64_t builtin_terminal_revision(const das::smart_ptr<TerminalHandle> & terminal) {
    return terminal ? static_cast<int64_t>(terminal->state.revision()) : 0;
}

int32_t builtin_terminal_mode_bits(const das::smart_ptr<TerminalHandle> & terminal) {
    if (!terminal) return 0;
    const Modes & modes = terminal->state.modes();
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
    const std::string value = terminal ? terminal->state.title() : std::string();
    return context->allocateString(value, at);
}

char * builtin_terminal_current_directory(
    const das::smart_ptr<TerminalHandle> & terminal, das::Context * context, das::LineInfoArg * at) {
    const std::string value = terminal ? terminal->state.currentDirectory() : std::string();
    return context->allocateString(value, at);
}

int32_t builtin_terminal_scrollback_rows(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen) {
    if (!terminal) return 0;
    const Terminal::CellRows * rows = terminal->state.cellRows(screen, true);
    return rows ? static_cast<int32_t>(rows->size()) : 0;
}

void builtin_terminal_cursor(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen,
    const das::TBlock<void, int32_t, int32_t, bool, bool, int32_t> & block,
    das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const Cursor * cursor = terminal->state.cursor(screen);
    if (!cursor) return;
    das::das_invoke<void>::invoke<int32_t, int32_t, bool, bool, int32_t>(
        context, at, block, cursor->row, cursor->column,
        cursor->visible, cursor->blinking, cursor->style);
}

void visitCells(const Terminal::CellRows & rows, size_t first_row, size_t end_row,
                const CellBlock & block, das::Context * context, das::LineInfoArg * at) {
    for (size_t row = first_row; row != end_row; ++row) {
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

void builtin_terminal_visit_cells(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen, bool scrollback,
    const CellBlock & block, das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const Terminal::CellRows * rows = terminal->state.cellRows(screen, scrollback);
    if (!rows) return;
    visitCells(*rows, 0, rows->size(), block, context, at);
}

void builtin_terminal_visit_viewport_scrollback(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen, int32_t scroll_offset,
    const CellBlock & block, das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const Terminal::CellRows * rows = terminal->state.cellRows(screen, true);
    if (!rows) return;
    const size_t offset = static_cast<size_t>(std::clamp(
        scroll_offset, 0, static_cast<int32_t>(rows->size())));
    const size_t first_row = rows->size() - offset;
    const size_t end_row = std::min(
        rows->size(), first_row + static_cast<size_t>(terminal->state.rows()));
    visitCells(*rows, first_row, end_row, block, context, at);
}

void builtin_terminal_visit_unknown(
    const das::smart_ptr<TerminalHandle> & terminal,
    const das::TBlock<void, das::TTemporary<const char *>> & block,
    das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const std::vector<std::string> & values = terminal->state.unknownSequences();
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
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_launch)>(*this, lib,
            "_terminal_launch", SideEffects::modifyExternal,
            "das_terminal::builtin_terminal_launch")
                ->args({"command_line", "command_count", "working_directory",
                    "directory_count", "columns", "rows"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_feed)>(*this, lib,
            "_terminal_feed", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_feed")->args({"terminal", "bytes", "count"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_resize)>(*this, lib,
            "_terminal_resize", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_resize")->args({"terminal", "columns", "rows"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_poll)>(*this, lib,
            "_terminal_poll", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_poll")->args({"terminal", "maximum_bytes"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_write)>(*this, lib,
            "_terminal_write", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_write")->args({"terminal", "bytes", "count"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_process_id)>(*this, lib,
            "_terminal_process_id", SideEffects::none,
            "das_terminal::builtin_terminal_process_id")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_process_exited)>(*this, lib,
            "_terminal_process_exited", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_process_exited")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_exit_code)>(*this, lib,
            "_terminal_exit_code", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_exit_code")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_terminate)>(*this, lib,
            "_terminal_terminate", SideEffects::modifyArgument,
            "das_terminal::builtin_terminal_terminate")->args({"terminal", "exit_code"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_transport_error)>(*this, lib,
            "_terminal_transport_error", SideEffects::modifyExternal,
            "das_terminal::builtin_terminal_transport_error")
                ->args({"terminal", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_encode_key)>(*this, lib,
            "_terminal_encode_key", SideEffects::modifyExternal,
            "das_terminal::builtin_terminal_encode_key")
                ->args({"terminal", "key", "text", "count", "shift", "alt", "control",
                    "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_encode_paste)>(*this, lib,
            "_terminal_encode_paste", SideEffects::modifyExternal,
            "das_terminal::builtin_terminal_encode_paste")
                ->args({"terminal", "text", "count", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_columns)>(*this, lib,
            "_terminal_columns", SideEffects::none,
            "das_terminal::builtin_terminal_columns")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_rows)>(*this, lib,
            "_terminal_rows", SideEffects::none,
            "das_terminal::builtin_terminal_rows")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_alternate_active)>(*this, lib,
            "_terminal_alternate_active", SideEffects::none,
            "das_terminal::builtin_terminal_alternate_active")->arg("terminal");
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_revision)>(*this, lib,
            "_terminal_revision", SideEffects::none,
            "das_terminal::builtin_terminal_revision")->arg("terminal");
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
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_visit_viewport_scrollback)>(
            *this, lib, "_terminal_visit_viewport_scrollback", SideEffects::invoke,
            "das_terminal::builtin_terminal_visit_viewport_scrollback")
                ->args({"terminal", "screen", "scroll_offset", "block", "context", "at"});
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
