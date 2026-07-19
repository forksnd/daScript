#pragma once

#include "daScript/misc/platform.h"
#include "daScript/simulate/bind_enum.h"
#include "daScript/simulate/simulate.h"

#include "terminal.h"
#include "pty.h"

namespace das_terminal {

struct TerminalHandle : das::ptr_ref_count {
    Terminal state;
    std::unique_ptr<PtyProcess> process;
    std::string transport_error;
    std::string screen_text_cache;
    uint64_t screen_text_revision = uint64_t(-1);
    bool process_exited = false;
    uint32_t process_exit_code = 0;

    TerminalHandle(int columns, int rows) : state(columns, rows) {}
};

using CellBlock = das::TBlock<void,
    int32_t, int32_t, das::TTemporary<const char *>, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t,
    das::TTemporary<const char *>>;

using TextRunBlock = das::TBlock<void,
    int32_t, int32_t, das::TTemporary<const char *>,
    int32_t, int32_t, int32_t, int32_t, int32_t, bool, int32_t>;

das::smart_ptr<TerminalHandle> builtin_terminal_create(int32_t columns, int32_t rows);
das::smart_ptr<TerminalHandle> builtin_terminal_launch(
    const char * command_line, int32_t command_count,
    const char * working_directory, int32_t directory_count,
    int32_t columns, int32_t rows);
void builtin_terminal_feed(
    const das::smart_ptr<TerminalHandle> & terminal, const char * bytes, int32_t count);
void builtin_terminal_resize(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t columns, int32_t rows);
int32_t builtin_terminal_poll(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t maximum_bytes);
bool builtin_terminal_write(
    const das::smart_ptr<TerminalHandle> & terminal, const char * bytes, int32_t count);
int32_t builtin_terminal_process_id(const das::smart_ptr<TerminalHandle> & terminal);
bool builtin_terminal_process_exited(const das::smart_ptr<TerminalHandle> & terminal);
int64_t builtin_terminal_exit_code(const das::smart_ptr<TerminalHandle> & terminal);
bool builtin_terminal_terminate(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t exit_code);
char * builtin_terminal_transport_error(
    const das::smart_ptr<TerminalHandle> & terminal,
    das::Context * context, das::LineInfoArg * at);
char * builtin_terminal_encode_key(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t key,
    const char * text, int32_t count, bool shift, bool alt, bool control,
    das::Context * context, das::LineInfoArg * at);
char * builtin_terminal_encode_paste(
    const das::smart_ptr<TerminalHandle> & terminal, const char * text, int32_t count,
    das::Context * context, das::LineInfoArg * at);
int32_t builtin_terminal_columns(const das::smart_ptr<TerminalHandle> & terminal);
int32_t builtin_terminal_rows(const das::smart_ptr<TerminalHandle> & terminal);
bool builtin_terminal_alternate_active(const das::smart_ptr<TerminalHandle> & terminal);
int64_t builtin_terminal_revision(const das::smart_ptr<TerminalHandle> & terminal);
int32_t builtin_terminal_mode_bits(const das::smart_ptr<TerminalHandle> & terminal);
char * builtin_terminal_title(
    const das::smart_ptr<TerminalHandle> & terminal, das::Context * context, das::LineInfoArg * at);
char * builtin_terminal_current_directory(
    const das::smart_ptr<TerminalHandle> & terminal, das::Context * context, das::LineInfoArg * at);
char * builtin_terminal_screen_text(
    const das::smart_ptr<TerminalHandle> & terminal, das::Context * context, das::LineInfoArg * at);
int32_t builtin_terminal_scrollback_rows(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen);
void builtin_terminal_cursor(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen,
    const das::TBlock<void, int32_t, int32_t, bool, bool, int32_t> & block,
    das::Context * context, das::LineInfoArg * at);
void builtin_terminal_visit_cells(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen, bool scrollback,
    const CellBlock & block, das::Context * context, das::LineInfoArg * at);
void builtin_terminal_visit_viewport_scrollback(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen, int32_t scroll_offset,
    const CellBlock & block, das::Context * context, das::LineInfoArg * at);
void builtin_terminal_visit_viewport_text_runs(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t scroll_offset,
    const TextRunBlock & block, das::Context * context, das::LineInfoArg * at);
void builtin_terminal_visit_viewport_paint_cells(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t screen, bool scrollback,
    int32_t scroll_offset, const CellBlock & block,
    das::Context * context, das::LineInfoArg * at);
void builtin_terminal_visit_unknown(
    const das::smart_ptr<TerminalHandle> & terminal,
    const das::TBlock<void, das::TTemporary<const char *>> & block,
    das::Context * context, das::LineInfoArg * at);
void builtin_terminal_drain_replies(
    const das::smart_ptr<TerminalHandle> & terminal,
    const das::TBlock<void, das::TTemporary<const char *>> & block,
    das::Context * context, das::LineInfoArg * at);
void builtin_terminal_take_events(
    const das::smart_ptr<TerminalHandle> & terminal,
    const das::TBlock<void, das::TTemporary<const char *>> & block,
    das::Context * context, das::LineInfoArg * at);

} // namespace das_terminal
