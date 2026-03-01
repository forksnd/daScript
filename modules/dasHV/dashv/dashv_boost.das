options gen2
options indenting = 4
options no_unused_block_arguments = false
options no_unused_function_arguments = false

module dashv_boost shared public

require dashv public

require rtti
require daslib/ast_boost


class HvWebSocketClient {
    client : WebSocketClient?
    def init(url : string) : int {
        client = make_web_socket_client(unsafe(addr(self)), class_info(self))
        return open(*client, url)
    }
    def operator delete {
        if (client != null) {
            unsafe {
                delete client
            }
        }
    }
    def abstract onOpen : void
    def abstract onClose : void
    def abstract onMessage(msg : string#) : void
    def is_connected : bool {
        return is_connected(*client)
    }
    def process_event_que {
        tick(*client)
    }
    def send(text : string) {
        send(*client, text)
    }
    def close : int {
        return close(*client)
    }
}

class HvWebServer {
    server : WebSocketServer?
    def init(port : int) {
        server = make_web_socket_server(port, 0, "", unsafe(addr(self)), class_info(self))
        onInit()
    }
    def init_wss(port, httpsPort : int; pathToCert : string = "") {
        var certPath = !empty(pathToCert) ? pathToCert : get_das_root() + "/modules/dasHV/cert"
        server = make_web_socket_server(port, httpsPort, certPath, unsafe(addr(self)), class_info(self))
        onInit()
    }
    def start : int {
        return start(server)
    }
    def stop : int {
        return stop(server)
    }
    def tick : void {
        tick(server)
    }
    def onWsOpen(channel : WebSocketChannel?; url : string#) : void {}
    def onWsClose(channel : WebSocketChannel?) : void {}
    def onWsMessage(channel : WebSocketChannel?; msg : string#) : void {}
    def onTick : void {}
    def onInit : void {}
    def GET(uri : string; lmb : lambda<(var req : HttpRequest?; var resp : HttpResponse?) : http_status>) : void {
        unsafe(GET(server, uri, lmb))
    }
    def POST(uri : string; lmb : lambda<(var req : HttpRequest?; var resp : HttpResponse?) : http_status>) : void {
        unsafe(POST(server, uri, lmb))
    }
    def PUT(uri : string; lmb : lambda<(var req : HttpRequest?; var resp : HttpResponse?) : http_status>) : void {
        unsafe(PUT(server, uri, lmb))
    }
    def DELETE(uri : string; lmb : lambda<(var req : HttpRequest?; var resp : HttpResponse?) : http_status>) : void {
        unsafe(DELETE(server, uri, lmb))
    }
    def PATCH(uri : string; lmb : lambda<(var req : HttpRequest?; var resp : HttpResponse?) : http_status>) : void {
        unsafe(PATCH(server, uri, lmb))
    }
    def HEAD(uri : string; lmb : lambda<(var req : HttpRequest?; var resp : HttpResponse?) : http_status>) : void {
        unsafe(HEAD(server, uri, lmb))
    }
    def ANY(uri : string; lmb : lambda<(var req : HttpRequest?; var resp : HttpResponse?) : http_status>) : void {
        unsafe(ANY(server, uri, lmb))
    }
    def STATIC(path, dir : string) : void {
        STATIC(server, path, dir)
    }
    def allow_cors() : void {
        allow_cors(server)
    }
    def set_document_root(dir : string) : void {
        set_document_root(server, dir)
    }
    def set_home_page(filename : string) : void {
        set_home_page(server, filename)
    }
    def set_index_of(dir : string) : void {
        set_index_of(server, dir)
    }
    def set_error_page(filename : string) : void {
        set_error_page(server, filename)
    }
}

def with_http_request(blk : block<(var req : HttpRequest?#) : void>) {
    var req = new HttpRequest
    unsafe {
        invoke(blk, reinterpret<HttpRequest?#> req)
        delete req
    }
}

def get_body_bytes(resp : HttpResponse?) : array<uint8> {
    var bytes : array<uint8>
    if (resp == null || resp.status_code != http_status.OK || resp.content_length == 0) {
        return <- bytes
    }
    bytes |> resize(resp.content_length)
    peek(resp.body) <| $(bodybytes) {
        unsafe {
            memcpy(addr(bytes[0]), reinterpret<void?> bodybytes, resp.content_length)
        }
    }
    return <- bytes
}