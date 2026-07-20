options gen2
options rtti
options persistent_heap
options gc
options indenting = 4
options no_unused_function_arguments = false
options no_unused_block_arguments = false

module watcher_server shared public

require dashv/dashv_boost public
require ./watcher_core.das public
require daslib/json_boost
require daslib/fio
require strings

let private MAX_REQUEST_BODY_BYTES = 1024 * 1024
let private CONTROLLER_HEARTBEAT_TIMEOUT_MS = 5000l
var private g_watcher_gc_requested = false

def public watcher_request_gc() : bool {
    let coalesced = g_watcher_gc_requested
    g_watcher_gc_requested = true
    return coalesced
}

def public watcher_take_gc_request() : bool {
    let requested = g_watcher_gc_requested
    g_watcher_gc_requested = false
    return requested
}

def public watcher_reset_gc_request() {
    g_watcher_gc_requested = false
}

struct private WatcherWsMessage {
    op : string
    session_id : string
    text : string
    columns : int
    rows : int
    control : bool
    raw : bool
}

class private WatcherClient {
    @safe_when_uninitialized
    channel : WebSocketChannel = default<WebSocketChannel>
    id : int
    session_id : string
    controls : bool
    last_heartbeat : int64
    last_update_key : string
    raw : bool
    raw_offset : int64
}

struct private WatcherOk {
    ok : bool = true
}

struct private WatcherError {
    ok : bool
    error : string
}

def private json_error(message : string) : string {
    return sprint_json(WatcherError(error = message), false)
}

def private query_value(url, key : string) : string {
    let query = find(url, "?")
    if (query < 0) return ""
    let wanted = key + "="
    var position = query + 1
    while (position < length(url)) {
        let amp = find(url, "&", position)
        let end = amp < 0 ? length(url) : amp
        if (starts_with(slice(url, position, end), wanted)) {
            return slice(url, position + length(wanted), end)
        }
        if (amp < 0) break
        position = amp + 1
    }
    return ""
}

class DasHerdWatcherServer : HvWebServer {
    token : string
    control_html : string
    xterm_js : string
    xterm_css : string
    xterm_fit_js : string
    xterm_unicode_js : string
    xterm_web_links_js : string
    clients : array<WatcherClient?>
    next_client_id : int

    def authorized(url : string) : bool {
        return !empty(token) && query_value(url, "token") == token
    }

    def respond_error(var resp : HttpResponse?; status : http_status; message : string) : http_status {
        return resp |> JSON(json_error(message), status)
    }

    def require_auth(var req : HttpRequest?; var resp : HttpResponse?) : bool {
        if (req != null && authorized(string(req.url))) return true
        let _ = respond_error(resp, http_status.UNAUTHORIZED, "invalid watcher token")
        return false
    }

    def override onInit {
        GET("/") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            return resp |> SERVE_FILE(self.control_html)
        }
        GET("/vendor/xterm.js") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            return resp |> SERVE_FILE(self.xterm_js)
        }
        GET("/vendor/xterm.css") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            return resp |> SERVE_FILE(self.xterm_css)
        }
        GET("/vendor/addon-fit.js") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            return resp |> SERVE_FILE(self.xterm_fit_js)
        }
        GET("/vendor/addon-unicode11.js") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            return resp |> SERVE_FILE(self.xterm_unicode_js)
        }
        GET("/vendor/addon-web-links.js") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            return resp |> SERVE_FILE(self.xterm_web_links_js)
        }
        GET("/api/v1/health") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            return resp |> JSON("\{\"ok\":true,\"service\":\"dasHerd-watcher\"\}", http_status.OK)
        }
        GET("/api/v1/sessions") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            return resp |> JSON(watcher_list_json(), http_status.OK)
        }
        POST("/api/v1/gc") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            let coalesced = watcher_request_gc()
            return resp |> JSON("\{\"ok\":true,\"scheduled\":true,\"coalesced\":{coalesced}\}",
                http_status.ACCEPTED)
        }
        POST("/api/v1/sessions") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            if (req == null || length(req.body) > MAX_REQUEST_BODY_BYTES) {
                return self.respond_error(resp, http_status.BAD_REQUEST, "request body is too large")
            }
            var launch = WatcherLaunchRequest()
            if (!sscan_json(string(req.body), launch)) {
                return self.respond_error(resp, http_status.BAD_REQUEST, "invalid launch JSON")
            }
            let result = watcher_launch(launch)
            return resp |> JSON(sprint_json(result, false),
                result.ok ? http_status.CREATED : http_status.BAD_REQUEST)
        }
        POST("/api/v1/input") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            if (req == null || length(req.body) > MAX_REQUEST_BODY_BYTES) {
                return self.respond_error(resp, http_status.BAD_REQUEST, "request body is too large")
            }
            var input : WatcherInputRequest
            if (!sscan_json(string(req.body), input)) {
                return self.respond_error(resp, http_status.BAD_REQUEST, "invalid input JSON")
            }
            let error = watcher_input(input.session_id, input.text)
            if (!empty(error)) return self.respond_error(resp, http_status.CONFLICT, error)
            return resp |> JSON(sprint_json(WatcherOk(), false), http_status.OK)
        }
        POST("/api/v1/resize") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            if (req == null || length(req.body) > MAX_REQUEST_BODY_BYTES) {
                return self.respond_error(resp, http_status.BAD_REQUEST, "request body is too large")
            }
            var resize : WatcherResizeRequest
            if (!sscan_json(string(req.body), resize)) {
                return self.respond_error(resp, http_status.BAD_REQUEST, "invalid resize JSON")
            }
            let error = watcher_resize(resize.session_id, resize.columns, resize.rows, 0)
            if (!empty(error)) return self.respond_error(resp, http_status.CONFLICT, error)
            return resp |> JSON(sprint_json(WatcherOk(), false), http_status.OK)
        }
        POST("/api/v1/terminate") <| @(var req : HttpRequest?; var resp : HttpResponse?) : http_status {
            if (!self.require_auth(req, resp)) return http_status.UNAUTHORIZED
            if (req == null || length(req.body) > MAX_REQUEST_BODY_BYTES) {
                return self.respond_error(resp, http_status.BAD_REQUEST, "request body is too large")
            }
            var request : WatcherSessionRequest
            if (!sscan_json(string(req.body), request)) {
                return self.respond_error(resp, http_status.BAD_REQUEST, "invalid terminate JSON")
            }
            let error = watcher_terminate(request.session_id)
            if (!empty(error)) return self.respond_error(resp, http_status.CONFLICT, error)
            return resp |> JSON(sprint_json(WatcherOk(), false), http_status.OK)
        }
    }

    def find_client(channel : WebSocketChannel) : int {
        for (index in range(length(clients))) {
            if (clients[index] != null && clients[index].channel == channel) return index
        }
        return -1
    }

    def send_error(channel : WebSocketChannel; message : string) {
        let body = "\{\"type\":\"error\",\"error\":{sprint_json(message, false)}\}"
        send(channel, body, ws_opcode.WS_OPCODE_TEXT, true)
    }

    def send_snapshot(var client : WatcherClient?; force : bool = false) {
        if (client == null || empty(client.session_id)) return
        let update_key = watcher_update_key(client.session_id)
        if (!force && update_key == client.last_update_key) return
        let snapshot = watcher_snapshot_json(client.session_id)
        if (empty(snapshot)) {
            send_error(client.channel, "unknown session")
            return
        }
        client.last_update_key = update_key
        send(client.channel, "\{\"type\":\"snapshot\",\"data\":{snapshot}\}",
            ws_opcode.WS_OPCODE_TEXT, true)
    }

    def send_raw_output(var client : WatcherClient?) {
        if (client == null || !client.raw || empty(client.session_id)) return
        var chunk <- watcher_output_since(client.session_id, client.raw_offset)
        if (chunk.reset) {
            let start_offset = chunk.next_offset - int64(length(chunk.bytes))
            send(client.channel, "\{\"type\":\"raw_reset\",\"offset\":{start_offset}\}",
                ws_opcode.WS_OPCODE_TEXT, true)
        }
        client.raw_offset = chunk.next_offset
        if (!empty(chunk.bytes)) {
            unsafe {
                send(client.channel, reinterpret<string#>(addr(chunk.bytes[0])), length(chunk.bytes),
                    ws_opcode.WS_OPCODE_BINARY, true)
            }
        }
    }

    def detach(var client : WatcherClient?; detail : string) {
        if (client == null) return
        if (client.controls && !empty(client.session_id)) {
            watcher_release_controller(client.session_id, client.id, detail)
        }
        client.session_id = ""
        client.controls = false
        client.last_update_key = ""
        client.raw = false
        client.raw_offset = 0l
    }

    def override onWsOpen(channel : WebSocketChannel; url : string#) {
        if (!authorized(string(url))) {
            send_error(channel, "invalid watcher token")
            close(channel)
            return
        }
        next_client_id++
        clients |> push(new WatcherClient(channel = channel, id = next_client_id,
            last_heartbeat = ref_time_ticks()))
        send(channel, "\{\"type\":\"connected\",\"client_id\":{next_client_id}\}",
            ws_opcode.WS_OPCODE_TEXT, true)
    }

    def override onWsClose(channel : WebSocketChannel) {
        let index = find_client(channel)
        if (index < 0) return
        var client = clients[index]
        detach(client, "client_closed")
        clients |> erase(index)
        unsafe { delete client }
    }

    def override onWsMessage(channel : WebSocketChannel; message : string#) {
        let index = find_client(channel)
        if (index < 0) return
        var incoming : WatcherWsMessage
        if (length(message) > MAX_REQUEST_BODY_BYTES || !sscan_json(string(message), incoming)) {
            send_error(channel, "invalid WebSocket message")
            return
        }
        var client = clients[index]
        if (client == null) return
        client.last_heartbeat = ref_time_ticks()
        if (incoming.op == "heartbeat") {
            send(channel, "\{\"type\":\"heartbeat\"\}", ws_opcode.WS_OPCODE_TEXT, true)
        } elif (incoming.op == "attach") {
            if (watcher_find_session(incoming.session_id) < 0) {
                send_error(channel, "unknown session")
                return
            }
            detach(client, "reattach")
            client.session_id = incoming.session_id
            let accepts_input = watcher_accepts_input(incoming.session_id)
            client.controls = incoming.control && accepts_input && watcher_claim_controller(incoming.session_id, client.id)
            client.raw = incoming.raw
            client.raw_offset = 0l
            if (incoming.control && !accepts_input) {
                send_error(channel, "session is not running; output is replay-only")
            } elif (incoming.control && !client.controls) {
                send_error(channel, "controller lease is held by another client")
            }
            send(channel, "\{\"type\":\"attached\",\"session_id\":{sprint_json(client.session_id, false)},\"controls\":{client.controls}\}",
                ws_opcode.WS_OPCODE_TEXT, true)
            send_snapshot(client, true)
            send_raw_output(client)
        } elif (incoming.op == "detach") {
            detach(client, "client_detached")
            send(channel, "\{\"type\":\"detached\"\}", ws_opcode.WS_OPCODE_TEXT, true)
        } elif (incoming.op == "input") {
            if (!client.controls) {
                send_error(channel, "client does not hold the controller lease")
                return
            }
            let error = watcher_input(client.session_id, incoming.text, client.id)
            if (!empty(error)) send_error(channel, error)
        } elif (incoming.op == "resize") {
            if (!client.controls) {
                send_error(channel, "client does not hold the controller lease")
                return
            }
            let error = watcher_resize(client.session_id, incoming.columns, incoming.rows, client.id)
            if (!empty(error)) send_error(channel, error)
        } else {
            send_error(channel, "unknown operation")
        }
    }

    def override onTick {
        watcher_tick()
        for (index in range(length(clients))) {
            var client = clients[index]
            if (client == null) continue
            if (client.controls && get_time_nsec(client.last_heartbeat) / 1000000l >= CONTROLLER_HEARTBEAT_TIMEOUT_MS) {
                watcher_release_controller(client.session_id, client.id, "heartbeat_timeout")
                client.controls = false
                send_error(client.channel, "controller lease expired")
            }
            send_snapshot(client)
            send_raw_output(client)
        }
    }

    def cleanup_clients {
        for (index in range(length(clients))) {
            var client = clients[index]
            if (client != null) {
                detach(client, "server_shutdown")
                unsafe { delete client }
            }
        }
        clients |> clear()
    }
}
