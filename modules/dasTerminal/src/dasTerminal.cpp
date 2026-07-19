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

size_t boundedScrollOffset(int32_t scroll_offset, size_t history_rows) {
    return static_cast<size_t>(std::max(
        0, std::min(scroll_offset, static_cast<int32_t>(history_rows))));
}

void appendScreenTextRow(const std::vector<Cell> & row, bool first_row,
                         std::string & value, size_t & last_content_end) {
    if (!first_row) value.push_back('\n');
    const size_t row_begin = value.size();
    for (const Cell & cell : row)
        if (cell.width != 0) value += cell.grapheme;
    while (value.size() != row_begin && value.back() == ' ') value.pop_back();
    if (value.size() != row_begin) last_content_end = value.size();
}

const std::string & screenText(TerminalHandle & terminal) {
    if (terminal.screen_text_revision == terminal.state.revision())
        return terminal.screen_text_cache;
    terminal.screen_text_cache.clear();
    size_t last_content_end = 0;
    const int32_t screen = terminal.state.alternateActive() ? 1 : 0;
    const Terminal::CellRows * rows = terminal.state.cellRows(screen, false);
    if (rows) {
        for (size_t row = 0; row != rows->size(); ++row)
            appendScreenTextRow((*rows)[row], row == 0,
                terminal.screen_text_cache, last_content_end);
        terminal.screen_text_cache.resize(last_content_end);
    }
    terminal.screen_text_revision = terminal.state.revision();
    return terminal.screen_text_cache;
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
    if (terminal->process) {
        if (!terminal->process->resize(columns, rows, terminal->transport_error)) return;
        terminal->transport_error.clear();
    }
    terminal->state.resize(columns, rows);
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
    return context->allocateString(value.data(), value.size(), at);
}

char * builtin_terminal_current_directory(
    const das::smart_ptr<TerminalHandle> & terminal, das::Context * context, das::LineInfoArg * at) {
    const std::string value = terminal ? terminal->state.currentDirectory() : std::string();
    return context->allocateString(value.data(), value.size(), at);
}

char * builtin_terminal_screen_text(
    const das::smart_ptr<TerminalHandle> & terminal,
    das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return context->allocateString("", 0, at);
    const std::string & value = screenText(*terminal);
    return context->allocateString(value.data(), value.size(), at);
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
    const size_t offset = boundedScrollOffset(scroll_offset, rows->size());
    const size_t first_row = rows->size() - offset;
    const size_t end_row = std::min(
        rows->size(), first_row + static_cast<size_t>(terminal->state.rows()));
    visitCells(*rows, first_row, end_row, block, context, at);
}

struct TextRunGroup {
    Color color;
    bool default_is_background = false;
    int32_t first_column = 0;
    int32_t last_column = 0;
    int32_t glyph_cells = 0;
    std::string text;
};

bool batchableAscii(const Cell & cell) {
    // TextRun carries color, but not attributes or OSC 8 metadata. Keep those
    // cells in the sparse paint-cell projection so their semantics survive.
    if (cell.width != 1 || cell.attributes != 0 || !cell.hyperlink.empty() ||
        cell.grapheme.size() != 1) return false;
    const uint8_t value = static_cast<uint8_t>(cell.grapheme[0]);
    return value >= 32 && value <= 126;
}

bool effectiveForeground(const Cell & cell, Color & color, bool & default_is_background) {
    default_is_background = (cell.attributes & attr_inverse) != 0;
    color = default_is_background ? cell.background : cell.foreground;
    return true;
}

bool sameEffectiveForeground(const Cell & left, const Cell & right) {
    Color left_color;
    Color right_color;
    bool left_background = false;
    bool right_background = false;
    effectiveForeground(left, left_color, left_background);
    effectiveForeground(right, right_color, right_background);
    return left_background == right_background && left_color == right_color;
}

void emitTextRun(int32_t global_row, int32_t column, const std::string & text,
                 const Color & color, bool default_is_background, int32_t glyph_cells,
                 const TextRunBlock & block, das::Context * context, das::LineInfoArg * at) {
    das::das_invoke<void>::invoke<
        int32_t, int32_t, const char *, int32_t, int32_t, int32_t, int32_t, int32_t,
        bool, int32_t>(
        context, at, block, global_row, column, text.c_str(),
        colorKind(color), color.index, color.red, color.green, color.blue,
        default_is_background, glyph_cells);
}

void visitTextRows(const Terminal::CellRows & rows, size_t first_row, size_t end_row,
                   int32_t row_base, const TextRunBlock & block,
                   das::Context * context, das::LineInfoArg * at,
                   std::string * screen_text = nullptr, size_t * last_content_end = nullptr) {
    for (size_t row = first_row; row != end_row; ++row) {
        const std::vector<Cell> & cells = rows[row];
        if (screen_text && last_content_end)
            appendScreenTextRow(cells, row == first_row, *screen_text, *last_content_end);
        std::vector<TextRunGroup> groups;
        groups.reserve(17);
        bool too_many_colors = false;
        for (size_t column = 0; column != cells.size(); ++column) {
            const Cell & cell = cells[column];
            if (!batchableAscii(cell) || cell.grapheme == " ") continue;
            Color color;
            bool default_is_background = false;
            effectiveForeground(cell, color, default_is_background);
            size_t found = groups.size();
            for (size_t index = 0; index != groups.size(); ++index) {
                if (groups[index].default_is_background == default_is_background &&
                    groups[index].color == color) {
                    found = index;
                    break;
                }
            }
            if (found == groups.size()) {
                if (groups.size() == 16) {
                    too_many_colors = true;
                    break;
                }
                TextRunGroup group;
                group.color = color;
                group.default_is_background = default_is_background;
                group.first_column = static_cast<int32_t>(column);
                group.last_column = static_cast<int32_t>(column);
                group.text.assign(cells.size(), ' ');
                groups.push_back(group);
                found = groups.size() - 1;
            }
            TextRunGroup & group = groups[found];
            group.text[column] = cell.grapheme[0];
            group.last_column = static_cast<int32_t>(column);
            ++group.glyph_cells;
        }
        const int32_t global_row = row_base + static_cast<int32_t>(row);
        if (!too_many_colors) {
            for (const TextRunGroup & group : groups) {
                const size_t first = static_cast<size_t>(group.first_column);
                const size_t count = static_cast<size_t>(
                    group.last_column - group.first_column + 1);
                emitTextRun(global_row, group.first_column, group.text.substr(first, count),
                    group.color, group.default_is_background, group.glyph_cells,
                    block, context, at);
            }
            continue;
        }
        size_t column = 0;
        while (column != cells.size()) {
            const Cell & first = cells[column];
            if (!batchableAscii(first) || first.grapheme == " ") {
                ++column;
                continue;
            }
            const size_t run_start = column;
            size_t ink_end = column;
            int32_t glyph_cells = 0;
            while (column != cells.size() && batchableAscii(cells[column]) &&
                   sameEffectiveForeground(first, cells[column])) {
                if (cells[column].grapheme != " ") {
                    ink_end = column + 1;
                    ++glyph_cells;
                }
                ++column;
            }
            std::string text;
            text.reserve(ink_end - run_start);
            for (size_t item = run_start; item != ink_end; ++item)
                text += cells[item].grapheme;
            Color color;
            bool default_is_background = false;
            effectiveForeground(first, color, default_is_background);
            emitTextRun(global_row, static_cast<int32_t>(run_start), text,
                color, default_is_background, glyph_cells, block, context, at);
        }
    }
}

void builtin_terminal_visit_viewport_text_runs(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t scroll_offset,
    const TextRunBlock & block, das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const int32_t screen = terminal->state.alternateActive() ? 1 : 0;
    const Terminal::CellRows * history = terminal->state.cellRows(screen, true);
    const Terminal::CellRows * screen_rows = terminal->state.cellRows(screen, false);
    if (!history || !screen_rows) return;
    const size_t offset = boundedScrollOffset(scroll_offset, history->size());
    const size_t first_history_row = history->size() - offset;
    const size_t end_history_row = std::min(
        history->size(), first_history_row + static_cast<size_t>(terminal->state.rows()));
    visitTextRows(*history, first_history_row, end_history_row, 0, block, context, at);
    if (terminal->screen_text_revision == terminal->state.revision()) {
        visitTextRows(*screen_rows, 0, screen_rows->size(),
            static_cast<int32_t>(history->size()), block, context, at);
    } else {
        terminal->screen_text_cache.clear();
        size_t last_content_end = 0;
        visitTextRows(*screen_rows, 0, screen_rows->size(),
            static_cast<int32_t>(history->size()), block, context, at,
            &terminal->screen_text_cache, &last_content_end);
        terminal->screen_text_cache.resize(last_content_end);
        terminal->screen_text_revision = terminal->state.revision();
    }
}

bool needsPaintCell(const Cell & cell) {
    if (cell.width == 0) return false;
    const bool chrome = cell.background.kind != ColorKind::default_color ||
        (cell.attributes & (attr_inverse | attr_underline | attr_strike)) != 0;
    const bool fallback_glyph = !batchableAscii(cell) && cell.grapheme != " " &&
        !cell.grapheme.empty() && (cell.attributes & attr_hidden) == 0;
    return chrome || fallback_glyph;
}

void visitPaintCells(const Terminal::CellRows & rows, size_t first_row, size_t end_row,
                     const CellBlock & block, das::Context * context, das::LineInfoArg * at) {
    for (size_t row = first_row; row != end_row; ++row) {
        for (size_t column = 0; column != rows[row].size(); ++column) {
            const Cell & cell = rows[row][column];
            if (!needsPaintCell(cell)) continue;
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

void builtin_terminal_visit_viewport_paint_cells(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen, bool scrollback,
    int32_t scroll_offset, const CellBlock & block,
    das::Context * context, das::LineInfoArg * at) {
    if (!terminal) return;
    const Terminal::CellRows * rows = terminal->state.cellRows(screen, scrollback);
    if (!rows) return;
    if (!scrollback) {
        visitPaintCells(*rows, 0, rows->size(), block, context, at);
        return;
    }
    const size_t offset = boundedScrollOffset(scroll_offset, rows->size());
    const size_t first_row = rows->size() - offset;
    const size_t end_row = std::min(
        rows->size(), first_row + static_cast<size_t>(terminal->state.rows()));
    visitPaintCells(*rows, first_row, end_row, block, context, at);
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
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_screen_text)>(*this, lib,
            "_terminal_screen_text", SideEffects::modifyExternal,
            "das_terminal::builtin_terminal_screen_text")->args({"terminal", "context", "at"});
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
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_visit_viewport_text_runs)>(
            *this, lib, "_terminal_visit_viewport_text_runs", SideEffects::invoke,
            "das_terminal::builtin_terminal_visit_viewport_text_runs")
                ->args({"terminal", "scroll_offset", "block", "context", "at"});
        addExtern<DAS_BIND_FUN(das_terminal::builtin_terminal_visit_viewport_paint_cells)>(
            *this, lib, "_terminal_visit_viewport_paint_cells", SideEffects::invoke,
            "das_terminal::builtin_terminal_visit_viewport_paint_cells")
                ->args({"terminal", "screen", "scrollback", "scroll_offset",
                    "block", "context", "at"});
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
