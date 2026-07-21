options gen2
options rtti
options persistent_heap
options gc
options indenting = 4

module watcher_core shared public

require terminal/terminal public
require daslib/fio public
require daslib/json_boost public
require daslib/strings_boost
require strings public
require math

let private DEFAULT_DRAIN_QUIET_MS = 250l
let private MAX_COMMAND_BYTES = 32 * 1024
let private MAX_INPUT_BYTES = 1024 * 1024
let private MAX_RAW_HISTORY_BYTES = 8 * 1024 * 1024
let private RAW_HISTORY_TRIM_SLACK = 1024 * 1024

struct public WatcherLaunchRequest {
    command_line : string
    argv : array<string>
    cwd : string
    display_name : string
    kind : string
    target_id : string = "local"
    repository_id : string
    worktree_path : string
    columns : int = 100
    rows : int = 30
    timeout_ms : int64
}

def public watcher_target_id_or_local(target_id : string) : string {
    return empty(target_id) ? "local" : target_id
}

struct public WatcherInputRequest {
    session_id : string
    text : string
}

struct public WatcherResizeRequest {
    session_id : string
    columns : int
    rows : int
}

struct public WatcherSessionRequest {
    session_id : string
}

struct public WatcherEvent {
    session_id : string
    event : string
    elapsed_ms : int64
    detail : string
}

struct public WatcherSessionInfo {
    id : string
    command_line : string
    cwd : string
    display_name : string
    kind : string
    target_id : string
    repository_id : string
    worktree_path : string
    state : string
    reason : string
    pid : int = 0
    exit_code : int64 = -1l
    elapsed_ms : int64
    output_bytes : int64
    input_bytes : int64
    columns : int
    rows : int
    revision : int64
    status_revision : int64
    controller_id : int
    drained : bool
}

struct public WatcherSessionSnapshot {
    session : WatcherSessionInfo = WatcherSessionInfo()
    screen_text : string
    title : string
    current_directory : string
    scrollback_rows : int
    cursor_row : int
    cursor_column : int
    cursor_visible : bool
    cursor_blinking : bool
}

struct public WatcherLaunchResult {
    ok : bool
    session_id : string
    error : string
}

struct public WatcherOutputChunk {
    bytes : array<uint8>
    next_offset : int64
    reset : bool
}

class private WatcherSession {
    id : string
    command_line : string
    cwd : string
    display_name : string
    kind : string
    target_id : string
    repository_id : string
    worktree_path : string
    session_dir : string
    output_path : string
    events_path : string
    state : string = "starting"
    reason : string
    pid : int = 0
    terminal : Terminal = Terminal()
    output_file : FILE const? = null
    events_file : FILE const? = null
    started : int64
    timeout_ms : int64
    last_data_ms : int64
    exit_seen_ms : int64 = -1l
    exit_code : int64 = -1l
    output_bytes : int64
    input_bytes : int64
    raw_history : array<uint8>
    raw_history_start : int64
    status_revision : int64
    controller_id : int
    drained : bool
    def operator delete {
        if (output_file != null) {
            fflush(output_file)
            fclose(output_file)
            output_file = null
        }
        if (events_file != null) {
            fflush(events_file)
            fclose(events_file)
            events_file = null
        }
        terminal_dispose(terminal)
        delete raw_history
    }
}

def private retain_raw_output(var session : WatcherSession?; bytes : array<uint8> const) {
    if (empty(bytes)) return
    let old_length = length(session.raw_history)
    session.raw_history |> resize(old_length + length(bytes))
    unsafe {
        memcpy(addr(session.raw_history[old_length]), addr(bytes[0]), length(bytes))
    }
    if (length(session.raw_history) > MAX_RAW_HISTORY_BYTES + RAW_HISTORY_TRIM_SLACK) {
        let discard = length(session.raw_history) - MAX_RAW_HISTORY_BYTES
        session.raw_history |> erase(0, discard)
        session.raw_history_start += int64(discard)
    }
}

var private g_sessions : array<WatcherSession?>
var private g_log_root : string
var private g_next_session_id = 0

def private elapsed_ms(session : WatcherSession?) : int64 {
    return get_time_nsec(session.started) / 1000000l
}

def private write_text_file(path, body : string) : bool {
    var file = fopen(path, "wb")
    if (file == null) return false
    fwrite(file, body)
    fflush(file)
    fclose(file)
    return true
}

def private append_event(session : WatcherSession?; event, detail : string) {
    if (session.events_file == null) return
    let line = sprint_json(WatcherEvent(session_id = session.id, event = event,
        elapsed_ms = elapsed_ms(session), detail = detail), false) + "\n"
    fwrite(session.events_file, line)
    fflush(session.events_file)
}

def private session_info(session : WatcherSession?) : WatcherSessionInfo {
    return WatcherSessionInfo(id = session.id, command_line = session.command_line,
        cwd = session.cwd, display_name = session.display_name, kind = session.kind,
        target_id = session.target_id, repository_id = session.repository_id,
        worktree_path = session.worktree_path,
        state = session.state, reason = session.reason,
        pid = session.pid, exit_code = session.exit_code,
        elapsed_ms = elapsed_ms(session), output_bytes = session.output_bytes,
        input_bytes = session.input_bytes,
        columns = session.terminal.columns, rows = session.terminal.rows,
        revision = terminal_revision(session.terminal), status_revision = session.status_revision,
        controller_id = session.controller_id,
        drained = session.drained)
}

def public watcher_init(log_root : string) : string {
    g_log_root = log_root
    var error : string
    if (!mkdir_rec(g_log_root, error)) return error
    return ""
}

def public watcher_find_session(session_id : string) : int {
    for (index in range(length(g_sessions))) {
        if (g_sessions[index] != null && g_sessions[index].id == session_id) return index
    }
    return -1
}

def public watcher_accepts_input(session_id : string) : bool {
    let index = watcher_find_session(session_id)
    return (index >= 0 && g_sessions[index] != null &&
        g_sessions[index].state == "running" && !g_sessions[index].drained)
}

def public watcher_launch(request : WatcherLaunchRequest) : WatcherLaunchResult {
    if (empty(request.command_line) && empty(request.argv)) {
        return WatcherLaunchResult(error = "command_line or argv is required")
    }
    if (length(request.command_line) > MAX_COMMAND_BYTES) {
        return WatcherLaunchResult(error = "command_line is too large")
    }
    if (length(request.display_name) > MAX_COMMAND_BYTES) {
        return WatcherLaunchResult(error = "display_name is too large")
    }
    var argv_bytes = max(0, length(request.argv) - 1)
    for (argument in request.argv) {
        argv_bytes += length(argument)
        if (argv_bytes > MAX_COMMAND_BYTES) {
            return WatcherLaunchResult(error = "argv is too large")
        }
    }
    if (empty(g_log_root)) {
        return WatcherLaunchResult(error = "watcher core is not initialized")
    }
    g_next_session_id++
    let session_id = "s{ref_time_ticks()}_{g_next_session_id}"
    let session_dir = path_join(g_log_root, session_id)
    var mkdir_error : string
    if (!mkdir_rec(session_dir, mkdir_error)) {
        return WatcherLaunchResult(error = "could not create session directory: {mkdir_error}")
    }

    var visible_command = request.display_name
    if (empty(visible_command)) {
        visible_command = !empty(request.command_line) ? request.command_line : join(request.argv, " ")
    }
    var session = new WatcherSession(id = session_id,
        command_line = visible_command, cwd = request.cwd,
        display_name = request.display_name, kind = request.kind,
        target_id = request.target_id, repository_id = request.repository_id,
        worktree_path = request.worktree_path,
        session_dir = session_dir,
        output_path = path_join(session_dir, "output.raw"),
        events_path = path_join(session_dir, "events.jsonl"),
        started = ref_time_ticks(), timeout_ms = max(0l, request.timeout_ms))
    session.output_file = fopen(session.output_path, "wb")
    session.events_file = fopen(session.events_path, "wb")
    if (session.output_file == null || session.events_file == null) {
        unsafe { delete session }
        return WatcherLaunchResult(error = "could not open session logs")
    }
    let columns = clamp(request.columns, 20, 400)
    let rows = clamp(request.rows, 5, 200)
    var inscope persisted_request := request
    persisted_request.columns = columns
    persisted_request.rows = rows
    if (!write_text_file(path_join(session_dir, "session.json"), sprint_json(persisted_request, true))) {
        unsafe { delete session }
        return WatcherLaunchResult(error = "could not write session metadata")
    }

    if (!empty(request.argv)) {
        var inscope launched <- terminal_launch_argv(request.argv, request.cwd, columns, rows)
        session.terminal := launched
    } else {
        var inscope launched <- terminal_launch(request.command_line, request.cwd, columns, rows)
        session.terminal := launched
    }
    if (terminal_process_id(session.terminal) == 0) {
        session.state = "failed"
        session.reason = terminal_transport_error(session.terminal)
        if (empty(session.reason)) { session.reason = "terminal launch failed" }
        session.drained = true
        session.status_revision++
        append_event(session, "launch_failed", session.reason)
    } else {
        session.pid = terminal_process_id(session.terminal)
        session.state = "running"
        session.status_revision++
        append_event(session, "launched", "pid={terminal_process_id(session.terminal)}")
    }
    g_sessions |> push(session)
    if (session.state == "failed") {
        return WatcherLaunchResult(session_id = session_id, error = session.reason)
    }
    return WatcherLaunchResult(ok = true, session_id = session_id)
}

def private observe_exit(var session : WatcherSession?; now_ms : int64) {
    if (session.exit_seen_ms >= 0l || !terminal_process_exited(session.terminal)) return
    session.exit_seen_ms = now_ms
    session.exit_code = terminal_exit_code(session.terminal)
    if (session.state != "failed") {
        session.state = "exited"
    }
    session.status_revision++
    if (session.last_data_ms == 0l) {
        session.last_data_ms = now_ms
    }
    append_event(session, "process_exited", "exit_code={session.exit_code}")
}

def private finish_drain(var session : WatcherSession?; detail : string) {
    if (session.drained) return
    session.drained = true
    session.status_revision++
    if (session.output_file != null) {
        fflush(session.output_file)
        fclose(session.output_file)
        session.output_file = null
    }
    append_event(session, "output_drained", detail)
}

def private enforce_timeout(var session : WatcherSession?; now_ms : int64) {
    if (session.state != "running" || session.timeout_ms <= 0l || now_ms < session.timeout_ms) return
    session.reason = "wall_timeout"
    session.state = "terminating"
    session.status_revision++
    let terminated = terminal_terminate(session.terminal)
    append_event(session, "wall_timeout", "terminate={terminated}")
}

def public watcher_tick(maximum_bytes_per_session : int = 64 * 1024) {
    let bounded_bytes = clamp(maximum_bytes_per_session, 1024, 1024 * 1024)
    for (session in g_sessions) {
        if (session == null) continue
        let now_ms = elapsed_ms(session)
        if (session.drained) {
            // Pipe closure can race slightly ahead of the process handle. Keep
            // observing lifecycle/timeout state after output is fully drained.
            observe_exit(session, now_ms)
            enforce_timeout(session, now_ms)
            continue
        }
        let status = terminal_poll(session.terminal, bounded_bytes)
        if (status == TerminalPollStatus.data) {
            session.last_data_ms = now_ms
            session.output_bytes += int64(length(session.terminal.read_buffer))
            retain_raw_output(session, session.terminal.read_buffer)
            if (session.output_file != null) {
                let _ = fwrite(session.output_file, session.terminal.read_buffer)
                fflush(session.output_file)
            }
        } elif (status == TerminalPollStatus.error) {
            session.state = "failed"
            session.reason = terminal_transport_error(session.terminal)
            session.status_revision++
            append_event(session, "transport_error", session.reason)
            finish_drain(session, "transport_error")
            continue
        }
        observe_exit(session, now_ms)
        enforce_timeout(session, now_ms)
        if (status == TerminalPollStatus.closed) {
            finish_drain(session, "pipe_closed")
        } elif (session.exit_seen_ms >= 0l && now_ms - session.last_data_ms >= DEFAULT_DRAIN_QUIET_MS) {
            finish_drain(session, "quiet_interval")
        }
    }
}

def public watcher_output_since(session_id : string; requested_offset : int64;
                                maximum_bytes : int = 256 * 1024) : WatcherOutputChunk {
    var result = WatcherOutputChunk()
    let index = watcher_find_session(session_id)
    if (index < 0 || g_sessions[index] == null) return <- result
    let session = g_sessions[index]
    var offset = requested_offset
    if (offset < session.raw_history_start || offset > session.output_bytes) {
        offset = session.raw_history_start
        result.reset = true
    }
    let available = session.output_bytes - offset
    let count = min(int64(clamp(maximum_bytes, 1024, 1024 * 1024)), available)
    if (count > 0l) {
        let source = int(offset - session.raw_history_start)
        result.bytes |> resize(int(count))
        unsafe {
            memcpy(addr(result.bytes[0]), addr(session.raw_history[source]), int(count))
        }
    }
    result.next_offset = offset + count
    return <- result
}

def public watcher_get_session_info(session_id : string; var info : WatcherSessionInfo) : bool {
    let index = watcher_find_session(session_id)
    if (index < 0 || g_sessions[index] == null) return false
    info = session_info(g_sessions[index])
    return true
}

def public watcher_output_text(session_id : string) : string {
    let index = watcher_find_session(session_id)
    if (index < 0 || g_sessions[index] == null) return ""
    return string(g_sessions[index].raw_history)
}

def public watcher_strip_terminal_controls(value : string) : string {
    // Task processes launched through ConPTY are wrapped in cursor, screen,
    // title, and mode sequences even when the child itself emits plain text.
    // Machine consumers need the original printable byte stream, not a
    // bounded reconstructed screen.
    var inscope output : array<uint8>
    output |> reserve(length(value))
    var state = 0 // 0=text, 1=ESC, 2=CSI, 3=OSC, 4=OSC ESC, 5=ST string, 6=ST ESC
    peek_data(value) $(bytes) {
        for (index in range(length(value))) {
            let byte = int(bytes[index])
            if (state == 0) {
                if (byte == 0x1b) {
                    state = 1
                }
                elif (byte == '\t' || byte == '\r' || byte == '\n' || byte >= 0x20) {
                    output |> push(uint8(byte))
                }
            } elif (state == 1) {
                if (byte == '[') {
                    state = 2
                } elif (byte == ']') {
                    state = 3
                } elif (byte == 'P' || byte == '^' || byte == '_' || byte == 'X') {
                    state = 5
                } else {
                    state = 0
                }
            } elif (state == 2) {
                if (byte >= 0x40 && byte <= 0x7e) {
                    state = 0
                }
            } elif (state == 3) {
                if (byte == 0x07) {
                    state = 0
                } elif (byte == 0x1b) {
                    state = 4
                }
            } elif (state == 4) {
                state = byte == '\\' ? 0 : 3
            } elif (state == 5) {
                if (byte == 0x1b) {
                    state = 6
                }
            } else {
                state = byte == '\\' ? 0 : 5
            }
        }
    }
    return string(output)
}

def public watcher_machine_output_text(session_id : string;
                                       var truncated : bool) : string {
    truncated = false
    let index = watcher_find_session(session_id)
    if (index < 0 || g_sessions[index] == null) return ""
    let session = g_sessions[index]
    truncated = session.raw_history_start != 0l
    return watcher_strip_terminal_controls(string(session.raw_history))
}

def public watcher_terminal_text(session_id : string; var scrollback_rows : int) : string {
    scrollback_rows = 0
    let index = watcher_find_session(session_id)
    if (index < 0 || g_sessions[index] == null) return ""
    var viewport = TerminalViewportSnapshot()
    terminal_viewport_snapshot(g_sessions[index].terminal, 0, viewport)
    scrollback_rows = viewport.buffer.scrollback_rows
    let result = viewport.screen_text
    terminal_viewport_snapshot_dispose(viewport)
    return result
}

def public watcher_compact_session(session_id : string;
                                   columns : int = 160; rows : int = 50) : bool {
    let index = watcher_find_session(session_id)
    if (index < 0 || g_sessions[index] == null) return false
    var session = g_sessions[index]
    if (!session.drained || session.state == "running" || session.state == "terminating") return false
    let compact_columns = clamp(columns, 20, 400)
    let compact_rows = clamp(rows, 5, 200)
    if (session.terminal.transport == null && session.terminal.columns == compact_columns
            && session.terminal.rows == compact_rows) return true
    var inscope compact <- terminal_create(compact_columns, compact_rows)
    terminal_feed_bytes(compact, session.raw_history)
    terminal_dispose(session.terminal)
    session.terminal := compact
    session.status_revision++
    append_event(session, "terminal_compacted", "{compact_columns}x{compact_rows}")
    return true
}

def public watcher_session_artifact_path(session_id, name : string) : string {
    if (empty(name) || name == "." || name == ".." || find(name, "/") >= 0
            || find(name, "\\") >= 0 || find(name, ":") >= 0) return ""
    let index = watcher_find_session(session_id)
    if (index < 0 || g_sessions[index] == null) return ""
    return path_join(g_sessions[index].session_dir, name)
}

def public watcher_prune_completed_sessions(kind : string;
                                            repository_id : string;
                                            worktree_path : string) : int {
    var removed = 0
    var index = length(g_sessions)
    while (index > 0) {
        index--
        var session = g_sessions[index]
        if (session == null || !session.drained || session.state == "running"
                || session.state == "terminating" || session.kind != kind
                || session.repository_id != repository_id
                || session.worktree_path != worktree_path) continue
        unsafe { delete session }
        g_sessions |> erase(index)
        removed++
    }
    return removed
}

def private input_session(var session : WatcherSession?; text : string; client_id : int) : string {
    if (session.controller_id != 0 && session.controller_id != client_id) return "controller lease is held by another client"
    if (session.state != "running" || session.drained) return "session is not running"
    if (!terminal_write(session.terminal, text)) return terminal_transport_error(session.terminal)
    session.input_bytes += int64(length(text))
    return ""
}

def public watcher_input(session_id, text : string; client_id : int = 0) : string {
    let index = watcher_find_session(session_id)
    if (index < 0) return "unknown session"
    if (length(text) > MAX_INPUT_BYTES) return "input is too large"
    return input_session(g_sessions[index], text, client_id)
}

def private resize_session(var session : WatcherSession?; columns, rows, client_id : int) : string {
    if (session.controller_id != 0 && session.controller_id != client_id) return "controller lease is held by another client"
    terminal_resize(session.terminal, clamp(columns, 20, 400), clamp(rows, 5, 200))
    append_event(session, "resized", "{session.terminal.columns}x{session.terminal.rows}")
    return ""
}

def public watcher_resize(session_id : string; columns, rows, client_id : int) : string {
    let index = watcher_find_session(session_id)
    if (index < 0) return "unknown session"
    return resize_session(g_sessions[index], columns, rows, client_id)
}

def private terminate_session(var session : WatcherSession?; detail : string) : string {
    if ((session.state != "running" && session.state != "terminating")
            || terminal_process_exited(session.terminal)) return ""
    session.reason = detail
    session.state = "terminating"
    session.status_revision++
    if (!terminal_terminate(session.terminal)) return terminal_transport_error(session.terminal)
    append_event(session, "terminate_requested", detail)
    return ""
}

def public watcher_terminate(session_id : string; detail : string = "requested") : string {
    let index = watcher_find_session(session_id)
    if (index < 0) return "unknown session"
    return terminate_session(g_sessions[index], detail)
}

def private claim_session(var session : WatcherSession?; client_id : int) : bool {
    if (session.controller_id != 0 && session.controller_id != client_id) return false
    if (session.controller_id == 0) append_event(session, "controller_claimed", "client={client_id}")
    session.controller_id = client_id
    session.status_revision++
    return true
}

def public watcher_claim_controller(session_id : string; client_id : int) : bool {
    let index = watcher_find_session(session_id)
    if (index < 0 || client_id == 0) return false
    return claim_session(g_sessions[index], client_id)
}

def private release_session(var session : WatcherSession?; client_id : int; detail : string) {
    if (session.controller_id != client_id) return
    session.controller_id = 0
    session.status_revision++
    append_event(session, "controller_released", "client={client_id}; reason={detail}")
}

def public watcher_release_controller(session_id : string; client_id : int; detail : string = "released") {
    let index = watcher_find_session(session_id)
    if (index < 0) return
    release_session(g_sessions[index], client_id, detail)
}

def public watcher_list_json : string {
    var inscope result : array<WatcherSessionInfo>
    result |> reserve(length(g_sessions))
    for (session in g_sessions) {
        if (session != null) result |> push(session_info(session))
    }
    return sprint_json(result, false)
}

def private snapshot_session_json(session : WatcherSession?) : string {
    var viewport = TerminalViewportSnapshot()
    terminal_viewport_snapshot(session.terminal, 0, viewport)
    let cursor = viewport.buffer.cursor
    let snapshot = WatcherSessionSnapshot(session = session_info(session),
        screen_text = viewport.screen_text, title = session.terminal.title,
        current_directory = session.terminal.current_directory,
        scrollback_rows = viewport.buffer.scrollback_rows,
        cursor_row = cursor.row, cursor_column = cursor.column,
        cursor_visible = cursor.visible, cursor_blinking = cursor.blinking)
    let result = sprint_json(snapshot, false)
    terminal_viewport_snapshot_dispose(viewport)
    return result
}

def public watcher_snapshot_json(session_id : string) : string {
    let index = watcher_find_session(session_id)
    if (index < 0) return ""
    return snapshot_session_json(g_sessions[index])
}

def public watcher_update_key(session_id : string) : string {
    let index = watcher_find_session(session_id)
    if (index < 0 || g_sessions[index] == null) return ""
    let session = g_sessions[index]
    return "{terminal_revision(session.terminal)}:{session.status_revision}"
}

def public watcher_shutdown(terminate_running : bool = false) {
    for (session in g_sessions) {
        if (session == null) continue
        if (terminate_running && !terminal_process_exited(session.terminal)) {
            let _ = terminal_terminate(session.terminal)
        }
        if (session.output_file != null) {
            fflush(session.output_file)
            fclose(session.output_file)
            session.output_file = null
        }
        if (session.events_file != null) {
            append_event(session, "watcher_shutdown", "terminate_running={terminate_running}")
            fflush(session.events_file)
            fclose(session.events_file)
            session.events_file = null
        }
        unsafe { delete session }
    }
    g_sessions |> clear()
    unsafe { delete g_sessions }
}
