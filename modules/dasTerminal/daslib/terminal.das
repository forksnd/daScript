options gen2
options indenting = 4

module terminal shared

require terminal_core public
require strings

enum public TerminalColorKind {
    default_color
    indexed
    rgb
}

enum public TerminalKey {
    text
    enter
    tab
    backspace
    escape
    up
    down
    right
    left
    home
    end
    insert
    delete_key
    page_up
    page_down
    f1
    f2
    f3
    f4
    f5
    f6
    f7
    f8
    f9
    f10
    f11
    f12
}

enum public TerminalPollStatus {
    data
    idle
    closed
    error
}

struct public TerminalKeyEvent {
    key : TerminalKey = TerminalKey.text
    text : string
    shift : bool
    alt : bool
    control : bool
}

struct public TerminalColor {
    kind : TerminalColorKind = TerminalColorKind.default_color
    index : int
    red : int
    green : int
    blue : int
}

struct public TerminalCell {
    row : int
    column : int
    grapheme : string
    width : int = 1
    attributes : int
    foreground : TerminalColor = TerminalColor()
    background : TerminalColor = TerminalColor()
    hyperlink : string
}

struct public TerminalCursor {
    row : int
    column : int
    visible : bool = true
    blinking : bool = true
    style : int
}

struct public TerminalBufferSnapshot {
    cells : array<TerminalCell>
    scrollback : array<TerminalCell>
    scrollback_rows : int
    cursor : TerminalCursor = TerminalCursor()
}

struct public TerminalModes {
    auto_wrap : bool = true
    application_cursor_keys : bool
    bracketed_paste : bool
    focus_reporting : bool
    mouse_button_reporting : bool
    mouse_any_reporting : bool
    sgr_mouse_encoding : bool
}

struct public TerminalSnapshot {
    columns : int
    rows : int
    alternate_active : bool
    normal : TerminalBufferSnapshot = TerminalBufferSnapshot()
    alternate : TerminalBufferSnapshot = TerminalBufferSnapshot()
    modes : TerminalModes = TerminalModes()
    title : string
    current_directory : string
    unknown_sequences : array<string>
}

//! Geometry-safe ASCII paint run. Cells with wide, combining, hidden, or
//! non-ASCII graphemes remain in the viewport cell arrays for exact placement.
struct public TerminalTextRun {
    global_row : int
    column : int
    text : string
    color : TerminalColor = TerminalColor()
    default_is_background : bool
    glyph_cells : int
}

//! Bounded render snapshot: compact text runs plus only the special cells needed
//! for non-default chrome or exact-position fallback glyphs.
struct public TerminalViewportSnapshot {
    revision : int64
    columns : int
    rows : int
    alternate_active : bool
    buffer : TerminalBufferSnapshot = TerminalBufferSnapshot()
    text_runs : array<TerminalTextRun>
    screen_text : string
    cell_projection_us : int
    text_run_projection_us : int
}

typedef public Terminal = smart_ptr<terminal_handle>

def public terminal_create(columns : int = 80; rows : int = 25) : Terminal {
    return <- _terminal_create(columns, rows)
}

def public terminal_dispose(var terminal : Terminal&) {
    terminal := default<Terminal>
}

def public terminal_launch(command_line : string;
                           working_directory : string = "";
                           columns : int = 80; rows : int = 25) : Terminal {
    return <- _terminal_launch(
        command_line, length(command_line),
        working_directory, length(working_directory),
        columns, rows)
}

def public terminal_feed(terminal : Terminal const; bytes : string) {
    _terminal_feed(terminal, bytes, length(bytes))
}

def public terminal_resize(terminal : Terminal const; columns, rows : int) {
    _terminal_resize(terminal, columns, rows)
}

def public terminal_poll(terminal : Terminal const;
                         maximum_bytes : int = 64 * 1024) : TerminalPollStatus {
    return unsafe(reinterpret<TerminalPollStatus>(
        _terminal_poll(terminal, maximum_bytes)))
}

def public terminal_write(terminal : Terminal const; bytes : string) : bool {
    return _terminal_write(terminal, bytes, length(bytes))
}

def public terminal_process_id(terminal : Terminal const) : int {
    return _terminal_process_id(terminal)
}

def public terminal_process_exited(terminal : Terminal const) : bool {
    return _terminal_process_exited(terminal)
}

//! Returns ``-1`` while the root process is still running or status cannot be read.
def public terminal_exit_code(terminal : Terminal const) : int64 {
    return _terminal_exit_code(terminal)
}

def public terminal_terminate(terminal : Terminal const; exit_code : int = 1) : bool {
    return _terminal_terminate(terminal, exit_code)
}

def public terminal_transport_error(terminal : Terminal const) : string {
    return _terminal_transport_error(terminal)
}

def public terminal_revision(terminal : Terminal const) : int64 {
    return _terminal_revision(terminal)
}

def public terminal_encode_key(terminal : Terminal const;
                               event : TerminalKeyEvent const) : string {
    return _terminal_encode_key(
        terminal,
        int(event.key),
        event.text,
        length(event.text),
        event.shift,
        event.alt,
        event.control)
}

def public terminal_encode_paste(terminal : Terminal const; text : string) : string {
    return _terminal_encode_paste(terminal, text, length(text))
}

def public terminal_send_key(terminal : Terminal const;
                             event : TerminalKeyEvent const) : bool {
    return terminal_write(terminal, terminal_encode_key(terminal, event))
}

def public terminal_send_paste(terminal : Terminal const; text : string) : bool {
    return terminal_write(terminal, terminal_encode_paste(terminal, text))
}

def private collect_cells(terminal : Terminal const; screen : int; scrollback : bool;
                          var cells : array<TerminalCell>) {
    cells |> clear()
    _terminal_visit_cells(terminal, screen, scrollback) $(row, column : int;
                                                          grapheme : string const#; width, attributes : int;
                                                          foreground_kind, foreground_index, foreground_red,
                                                          foreground_green, foreground_blue : int;
                                                          background_kind, background_index, background_red,
                                                          background_green, background_blue : int;
                                                          hyperlink : string const#) {
        cells |> push(TerminalCell(
            row = row,
            column = column,
            grapheme = "{grapheme}",
            width = width,
            attributes = attributes,
            foreground = TerminalColor(
                kind = unsafe(reinterpret<TerminalColorKind>(foreground_kind)),
                index = foreground_index,
                red = foreground_red,
                green = foreground_green,
                blue = foreground_blue),
            background = TerminalColor(
                kind = unsafe(reinterpret<TerminalColorKind>(background_kind)),
                index = background_index,
                red = background_red,
                green = background_green,
                blue = background_blue),
            hyperlink = "{hyperlink}"))
    }
}

def private collect_cursor(terminal : Terminal const; screen : int;
                           var cursor : TerminalCursor) {
    _terminal_cursor(terminal, screen) $(row, column : int;
                                         visible, blinking : bool; style : int) {
        cursor = TerminalCursor(
            row = row,
            column = column,
            visible = visible,
            blinking = blinking,
            style = style)
    }
}

def private collect_viewport_text_runs(terminal : Terminal const; scroll_offset : int;
                                       var runs : array<TerminalTextRun>) {
    runs |> clear()
    _terminal_visit_viewport_text_runs(terminal, scroll_offset) $(global_row, column : int;
                                                                  text : string const#;
                                                                  color_kind, color_index,
                                                                  color_red, color_green, color_blue : int;
                                                                  default_is_background : bool;
                                                                  glyph_cells : int) {
        runs |> push(TerminalTextRun(
            global_row = global_row,
            column = column,
            text = "{text}",
            color = TerminalColor(
                kind = unsafe(reinterpret<TerminalColorKind>(color_kind)),
                index = color_index,
                red = color_red,
                green = color_green,
                blue = color_blue),
            default_is_background = default_is_background,
            glyph_cells = glyph_cells))
    }
}

def private collect_viewport_paint_cells(terminal : Terminal const; screen : int;
                                         scrollback : bool; scroll_offset : int;
                                         var cells : array<TerminalCell>) {
    cells |> clear()
    _terminal_visit_viewport_paint_cells(terminal, screen, scrollback, scroll_offset) $(row, column : int;
                                                                                        grapheme : string const#; width, attributes : int;
                                                                                        foreground_kind, foreground_index, foreground_red,
                                                                                        foreground_green, foreground_blue : int;
                                                                                        background_kind, background_index, background_red,
                                                                                        background_green, background_blue : int;
                                                                                        hyperlink : string const#) {
        cells |> push(TerminalCell(
            row = row,
            column = column,
            grapheme = "{grapheme}",
            width = width,
            attributes = attributes,
            foreground = TerminalColor(
                kind = unsafe(reinterpret<TerminalColorKind>(foreground_kind)),
                index = foreground_index,
                red = foreground_red,
                green = foreground_green,
                blue = foreground_blue),
            background = TerminalColor(
                kind = unsafe(reinterpret<TerminalColorKind>(background_kind)),
                index = background_index,
                red = background_red,
                green = background_green,
                blue = background_blue),
            hyperlink = "{hyperlink}"))
    }
}

def public terminal_viewport_snapshot(terminal : Terminal const; scroll_offset : int;
                                      var snapshot : TerminalViewportSnapshot) {
    snapshot.revision = _terminal_revision(terminal)
    snapshot.columns = _terminal_columns(terminal)
    snapshot.rows = _terminal_rows(terminal)
    snapshot.alternate_active = _terminal_alternate_active(terminal)
    let screen = snapshot.alternate_active ? 1 : 0
    let cells_started = ref_time_ticks()
    collect_viewport_paint_cells(terminal, screen, false, scroll_offset, snapshot.buffer.cells)
    collect_viewport_paint_cells(terminal, screen, true, scroll_offset, snapshot.buffer.scrollback)
    snapshot.cell_projection_us = get_time_usec(cells_started)
    let runs_started = ref_time_ticks()
    collect_viewport_text_runs(terminal, scroll_offset, snapshot.text_runs)
    //! Text-run projection refreshes the revision-keyed accessibility text cache
    //! while it already walks screen rows, avoiding a second grid traversal.
    snapshot.screen_text = _terminal_screen_text(terminal)
    snapshot.text_run_projection_us = get_time_usec(runs_started)
    snapshot.buffer.scrollback_rows = _terminal_scrollback_rows(terminal, screen)
    collect_cursor(terminal, screen, snapshot.buffer.cursor)
}

def public terminal_snapshot(terminal : Terminal const;
                             var snapshot : TerminalSnapshot) {
    snapshot.columns = _terminal_columns(terminal)
    snapshot.rows = _terminal_rows(terminal)
    snapshot.alternate_active = _terminal_alternate_active(terminal)
    snapshot.title = _terminal_title(terminal)
    snapshot.current_directory = _terminal_current_directory(terminal)

    collect_cells(terminal, 0, false, snapshot.normal.cells)
    collect_cells(terminal, 0, true, snapshot.normal.scrollback)
    snapshot.normal.scrollback_rows = _terminal_scrollback_rows(terminal, 0)
    collect_cursor(terminal, 0, snapshot.normal.cursor)

    collect_cells(terminal, 1, false, snapshot.alternate.cells)
    collect_cells(terminal, 1, true, snapshot.alternate.scrollback)
    snapshot.alternate.scrollback_rows = _terminal_scrollback_rows(terminal, 1)
    collect_cursor(terminal, 1, snapshot.alternate.cursor)

    let mode_bits = _terminal_mode_bits(terminal)
    snapshot.modes = TerminalModes(
        auto_wrap = (mode_bits & (1 << 0)) != 0,
        application_cursor_keys = (mode_bits & (1 << 1)) != 0,
        bracketed_paste = (mode_bits & (1 << 2)) != 0,
        focus_reporting = (mode_bits & (1 << 3)) != 0,
        mouse_button_reporting = (mode_bits & (1 << 4)) != 0,
        mouse_any_reporting = (mode_bits & (1 << 5)) != 0,
        sgr_mouse_encoding = (mode_bits & (1 << 6)) != 0)

    snapshot.unknown_sequences |> clear()
    _terminal_visit_unknown(terminal) $(sequence : string const#) {
        snapshot.unknown_sequences |> push("{sequence}")
    }
}

def public terminal_drain_replies(terminal : Terminal const;
                                  var replies : array<string>) {
    replies |> clear()
    _terminal_drain_replies(terminal) $(reply : string const#) {
        replies |> push("{reply}")
    }
}

def public terminal_take_events(terminal : Terminal const;
                                var events : array<string>) {
    events |> clear()
    _terminal_take_events(terminal) $(event : string const#) {
        events |> push("{event}")
    }
}
