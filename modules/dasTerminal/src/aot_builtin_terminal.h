#pragma once

#include "daScript/misc/platform.h"
#include "daScript/simulate/bind_enum.h"
#include "daScript/simulate/simulate.h"

#include "terminal.h"

namespace das_terminal {

struct TerminalHandle : das::ptr_ref_count {
    Terminal state;

    TerminalHandle(int columns, int rows) : state(columns, rows) {}
};

using CellBlock = das::TBlock<void,
    int32_t, int32_t, das::TTemporary<const char *>, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t,
    int32_t, int32_t, int32_t, int32_t, int32_t,
    das::TTemporary<const char *>>;

das::smart_ptr<TerminalHandle> builtin_terminal_create(int32_t columns, int32_t rows);
void builtin_terminal_feed(
    const das::smart_ptr<TerminalHandle> & terminal, const char * bytes, int32_t count);
void builtin_terminal_resize(
    const das::smart_ptr<TerminalHandle> & terminal, int32_t columns, int32_t rows);
int32_t builtin_terminal_columns(const das::smart_ptr<TerminalHandle> & terminal);
int32_t builtin_terminal_rows(const das::smart_ptr<TerminalHandle> & terminal);
bool builtin_terminal_alternate_active(const das::smart_ptr<TerminalHandle> & terminal);
int32_t builtin_terminal_mode_bits(const das::smart_ptr<TerminalHandle> & terminal);
char * builtin_terminal_title(
    const das::smart_ptr<TerminalHandle> & terminal, das::Context * context, das::LineInfoArg * at);
char * builtin_terminal_current_directory(
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
