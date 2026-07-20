options gen2
options gc
options persistent_heap
options no_unused_function_arguments = false

require imgui/imgui_harness
require imgui/imgui_terminal
require imgui/imgui_fonts
require dashv/dashv_boost
require clipboard/clipboard
require daslib/safe_addr
require daslib/json_boost
require ./watcher_core.das
require math
require strings

struct private WatcherEnvelope {
    @rename = "type" _type : string
}

struct private WatcherConnectedMessage {
    @rename = "type" _type : string
    client_id : int
}

struct private WatcherAttachedMessage {
    @rename = "type" _type : string
    session_id : string
    controls : bool
}

struct private WatcherSessionMessage {
    @rename = "type" _type : string
    session_id : string
}

struct private WatcherErrorMessage {
    @rename = "type" _type : string
    error : string
}

struct private WatcherSessionsMessage {
    @rename = "type" _type : string
    data : array<WatcherSessionInfo>
}

struct private WatcherRawResetMessage {
    @rename = "type" _type : string
    offset : int64
}

var private g_terminal = Terminal()
var private g_terminal_view = ImGuiTerminalState()
var private g_terminal_cache = ImGuiTerminalCache()
var private g_terminal_font : ImFont?
var private g_font_sources : array<string>
var private g_zoom_percent = 150
var private g_status = "Starting"
var private g_url : string
var private g_token : string
var private SESSION_ROW : table<int; ToggleState>

def private option_value(prefix : string; fallback : string = "") : string {
    for (argument in get_command_line_arguments()) {
        if (starts_with(argument, prefix)) return slice(argument, length(prefix))
    }
    return fallback
}

def private reset_terminal(columns : int = 100; rows : int = 30) {
    terminal_dispose(g_terminal)
    var inscope created <- terminal_create(columns, rows)
    g_terminal := created
    g_terminal_view = ImGuiTerminalState()
    g_terminal_cache = ImGuiTerminalCache()
}

def private feed_terminal_frame(message : string#; byte_count : int) {
    var bytes : array<uint8>
    bytes |> resize(byte_count)
    if (byte_count > 0) {
        unsafe { memcpy(addr(bytes[0]), reinterpret<void?>(message), byte_count) }
        terminal_feed_bytes(g_terminal, bytes)
    }
}

class private WatcherRichClient : HvWebSocketClient {
    connected : bool
    attached : bool
    controls : bool
    client_id : int
    selected_id : string
    attached_id : string
    sessions : array<WatcherSessionInfo>
    error : string
    last_heartbeat : int64
    last_list : int64
    sent_input_bytes : int64

    def override onOpen {
        connected = true
        error = ""
        last_heartbeat = ref_time_ticks()
        last_list = ref_time_ticks()
        send("\{\"op\":\"list\"\}")
    }

    def override onClose {
        connected = false
        attached = false
        controls = false
        error = "watcher connection closed"
    }

    def override onMessage(message : string#) {
        var envelope : WatcherEnvelope
        let body = string(message)
        if (!sscan_json(body, envelope)) {
            error = "invalid watcher message"
            return
        }
        if (envelope._type == "connected") {
            var incoming : WatcherConnectedMessage
            if (sscan_json(body, incoming)) {
                client_id = incoming.client_id
            }
        } elif (envelope._type == "sessions") {
            var incoming : WatcherSessionsMessage
            if (sscan_json(body, incoming)) {
                sessions <- incoming.data
                if (empty(selected_id)) {
                    for (session in sessions) {
                        if (session.state == "running") {
                            attach_session(session.id)
                            break
                        }
                    }
                }
            }
        } elif (envelope._type == "launched") {
            var incoming : WatcherSessionMessage
            if (sscan_json(body, incoming)) {
                attach_session(incoming.session_id)
            }
        } elif (envelope._type == "attached") {
            var incoming : WatcherAttachedMessage
            if (sscan_json(body, incoming)) {
                attached_id = incoming.session_id
                attached = true
                controls = incoming.controls
                g_status = incoming.controls ? "Attached with control" : "Attached read-only"
            }
        } elif (envelope._type == "detached") {
            attached = false
            controls = false
            attached_id = ""
        } elif (envelope._type == "raw_reset") {
            var incoming : WatcherRawResetMessage
            if (sscan_json(body, incoming)) {
                reset_terminal(g_terminal.columns, g_terminal.rows)
                g_status = "Raw replay reset at byte {incoming.offset}"
            }
        } elif (envelope._type == "error") {
            var incoming : WatcherErrorMessage
            if (sscan_json(body, incoming)) {
                error = incoming.error
            }
        }
    }

    def override onMessageFrame(message : string#; byte_count : int; opcode : ws_opcode) {
        if (opcode == ws_opcode.WS_OPCODE_TEXT) {
            onMessage(message)
        } elif (opcode == ws_opcode.WS_OPCODE_BINARY) {
            feed_terminal_frame(message, byte_count)
        }
    }

    def attach_session(session_id : string) {
        if (!connected || empty(session_id)) return
        var columns = 100
        var rows = 30
        for (session in sessions) {
            if (session.id == session_id) {
                columns = max(20, session.columns)
                rows = max(5, session.rows)
                break
            }
        }
        selected_id = session_id
        attached = false
        controls = false
        error = ""
        reset_terminal(columns, rows)
        send("\{\"op\":\"attach\",\"session_id\":{sprint_json(session_id, false)}," +
            "\"control\":true,\"raw\":true\}")
    }

    def launch(command_line, cwd : string) {
        if (!connected) {
            error = "watcher is not connected"
            return
        }
        send("\{\"op\":\"launch\",\"command_line\":{sprint_json(command_line, false)}," +
            "\"cwd\":{sprint_json(cwd, false)},\"columns\":100,\"rows\":30," +
            "\"timeout_ms\":0\}")
    }

    def send_input(text : string) {
        if (!controls || empty(text)) return
        send("\{\"op\":\"input\",\"text\":{sprint_json(text, false)}\}")
        sent_input_bytes += int64(length(text))
    }

    def resize(columns, rows : int) {
        if (!controls) return
        send("\{\"op\":\"resize\",\"columns\":{columns},\"rows\":{rows}\}")
    }

    def terminate_selected() {
        if (!connected || empty(selected_id)) return
        send("\{\"op\":\"terminate\",\"session_id\":{sprint_json(selected_id, false)}\}")
    }

    def refresh() {
        if (connected) send("\{\"op\":\"list\"\}")
    }

    def pump() {
        process_event_que()
        if (!connected) return
        if (get_time_nsec(last_heartbeat) / 1000000l >= 1000l) {
            send("\{\"op\":\"heartbeat\"\}")
            last_heartbeat = ref_time_ticks()
        }
        if (get_time_nsec(last_list) / 1000000l >= 1000l) {
            refresh()
            last_list = ref_time_ticks()
        }
    }
}

var private g_client : WatcherRichClient?

def private zoom_by(delta : int) {
    g_zoom_percent = clamp(g_zoom_percent + delta, 50, 300)
}

def private selected_session_label() : string {
    if (g_client == null || empty(g_client.selected_id)) return "No session selected"
    for (session in g_client.sessions) {
        if (session.id == g_client.selected_id) {
            return "{session.state}  pid {session.pid}  {session.command_line}"
        }
    }
    return g_client.selected_id
}

def private draw_session_list() {
    let connection = g_client != null && g_client.connected ? "connected" : "disconnected"
    text("Watcher: {connection}")
    if (small_button(REFRESH_SESSIONS, (text = "Refresh")) && g_client != null) {
        g_client->refresh()
    }
    same_line()
    if (small_button(TERMINATE_SESSION, (text = "Terminate")) && g_client != null) {
        g_client->terminate_selected()
    }
    separator()
    child(SESSIONS_CHILD, (text = "sessions", size = float2(0.0f, -150.0f),
            child_flags = ImGuiChildFlags.Borders, window_flags = ImGuiWindowFlags.None)) {
        if (g_client != null && empty(g_client.sessions)) {
            text_disabled("No sessions")
        } elif (g_client != null) {
            for (index in range(length(g_client.sessions))) {
                let session = g_client.sessions[index]
                let row_text = "[{session.state}] {session.command_line}##{session.id}"
                if (selectable(SESSION_ROW[index], (text = row_text))) {
                    g_client->attach_session(session.id)
                }
                SESSION_ROW[index].value = session.id == g_client.selected_id
            }
        }
    }
    text("New terminal")
    input_text_growable(LAUNCH_COMMAND, (text = "Command"))
    input_text_growable(LAUNCH_CWD, (text = "Working directory"))
    if (button(LAUNCH_SESSION, (text = "Launch and attach")) && g_client != null) {
        g_client->launch(LAUNCH_COMMAND.value, LAUNCH_CWD.value)
    }
}

def private draw_terminal_footer() {
    if (small_button(TERMINAL_EDIT_MENU, (text = "Edit"))) {
        open_popup("terminal_edit_menu")
    }
    popup_window(TERMINAL_EDIT_POPUP,
            (str_id = "terminal_edit_menu", flags = ImGuiWindowFlags.AlwaysAutoResize)) {
        if (menu_label(TERMINAL_COPY, (text = "Copy", shortcut = "Ctrl+C",
                enabled = g_terminal_view.selection_active))) {
            g_status = (imgui_terminal_copy_selection(g_terminal, g_terminal_view)
                ? "Selection copied" : "Nothing selected")
        }
        if (menu_label(TERMINAL_PASTE, (text = "Paste", shortcut = "Ctrl+V"))) {
            var pasted = ""
            if (clipboard_get_text(pasted) == ClipboardStatus.ok && g_client != null) {
                g_client->send_input(terminal_encode_paste(g_terminal, pasted))
                g_status = "Clipboard pasted"
            } else {
                g_status = "Clipboard unavailable"
            }
        }
        separator()
        if (menu_label(TERMINAL_SELECT_ALL, (text = "Select all"))) {
            imgui_terminal_select_all(g_terminal, g_terminal_view)
        }
        if (menu_label(TERMINAL_CLEAR_SELECTION,
                (text = "Clear selection", enabled = g_terminal_view.selection_active))) {
            imgui_terminal_clear_selection(g_terminal_view)
        }
    }
    same_line()
    if (small_button(TERMINAL_VIEW_MENU, (text = "View"))) {
        open_popup("terminal_view_menu")
    }
    popup_window(TERMINAL_VIEW_POPUP,
            (str_id = "terminal_view_menu", flags = ImGuiWindowFlags.AlwaysAutoResize)) {
        if (menu_label(TERMINAL_ZOOM_IN, (text = "Zoom in", shortcut = "Ctrl++"))) zoom_by(5)
        if (menu_label(TERMINAL_ZOOM_OUT, (text = "Zoom out", shortcut = "Ctrl+-"))) zoom_by(-5)
        if (menu_label(TERMINAL_ZOOM_RESET, (text = "Reset zoom", shortcut = "Ctrl+0"))) {
            g_zoom_percent = 100
        }
        separator()
        if (menu_label(TERMINAL_SCROLL_BOTTOM, (text = "Scroll to bottom"))) {
            g_terminal_view.scroll_offset = 0
        }
    }
    same_line()
    if (small_button(TERMINAL_ZOOM_OUT_BUTTON, (text = "-5%"))) zoom_by(-5)
    same_line()
    if (small_button(TERMINAL_ZOOM_IN_BUTTON, (text = "+5%"))) zoom_by(5)
    same_line()
    text("Zoom {g_zoom_percent}%")
    same_line()
    text_disabled("{g_terminal_view.scrollback_rows} history lines - {g_status}")
    if (g_client != null) {
        same_line()
        text_disabled("input sent {g_client.sent_input_bytes} bytes")
    }
}

[export]
def init() {
    harness_init("dasHerd", 1280, 800)
    harness_maximize_window()
    g_terminal_font = load_unicode_ui_font(14.0f, g_font_sources)
    reset_terminal()
    LAUNCH_COMMAND.value = "powershell.exe -NoLogo -NoProfile"
    LAUNCH_CWD.value = get_das_root()
    let port = clamp(to_int(option_value("--port=", "9191")), 1024, 65535)
    g_token = option_value("--token=")
    g_url = "ws://127.0.0.1:{port}/ws?token={g_token}"
    g_client = new WatcherRichClient()
    if (empty(g_token)) {
        g_client.error = "Pass the watcher token with --token=<token>"
        return
    }
    let result = g_client->init(g_url)
    if (result != 0) {
        g_client.error = "watcher connection failed ({result})"
    }
}

[export]
def update() {
    if (!harness_begin_frame()) return
    if (g_client != null) g_client->pump()
    harness_new_frame()
    GetStyle().FontScaleMain = float(g_zoom_percent) / 100.0f
    let viewport = GetMainViewport()
    SetNextWindowPos(viewport.WorkPos, ImGuiCond.Always)
    SetNextWindowSize(viewport.WorkSize, ImGuiCond.Always)
    window(HERDER_WIN, (text = "dasHerd", closable = false,
            flags = ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove)) {
        let io & = unsafe(GetIO())
        if (IsWindowHovered(ImGuiHoveredFlags.RootAndChildWindows) && io.KeyCtrl) {
            if (io.MouseWheel > 0.0f) zoom_by(5)
            elif (io.MouseWheel < 0.0f) zoom_by(-5)
        }
        child(SIDEBAR_CHILD, (text = "sidebar", size = float2(330.0f, 0.0f),
                child_flags = ImGuiChildFlags.None, window_flags = ImGuiWindowFlags.None)) {
            draw_session_list()
        }
        same_line()
        child(TERMINAL_CHILD, (text = "terminal", size = float2(0.0f, 0.0f),
                child_flags = ImGuiChildFlags.None, window_flags = ImGuiWindowFlags.None)) {
            text(SESSION_LABEL, (text = selected_session_label()))
            if (g_client != null && !empty(g_client.error)) {
                text_colored(CLIENT_ERROR,
                    (color = float4(1.0f, 0.35f, 0.30f, 1.0f), text = g_client.error))
            }
            separator()
            let terminal_style = ImGuiTerminalStyle(
                external_input = true,
                view_height = max(GetTextLineHeight(),
                    GetContentRegionAvail().y - GetFrameHeightWithSpacing()
                        - GetStyle().ItemSpacing.y))
            with_font(g_terminal_font) {
                let result <- imgui_terminal("watcher", g_terminal, g_terminal_view,
                    g_terminal_cache, terminal_style)
                if (g_client != null) {
                    if (!empty(result.input_text)) g_client->send_input(result.input_text)
                    if (result.resized) g_client->resize(result.columns, result.rows)
                }
            }
            separator()
            draw_terminal_footer()
        }
    }
    harness_end_frame()
}

[export]
def shutdown() {
    if (g_client != null) {
        g_client->close()
        g_client->cleanup()
        unsafe { delete g_client }
        g_client = null
    }
    terminal_dispose(g_terminal)
    harness_shutdown()
}

[export]
def main() {
    init()
    while (!exit_requested()) {
        update()
        harness_maybe_collect_gc()
    }
    shutdown()
}
