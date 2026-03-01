options gen2
options persistent_heap
options gc

require dashv/dashv_boost

require fio

class ExampleWebSocketClient : HvWebSocketClient {
    def override onOpen  : void {
        print("onOpen\n")
    }
    def override onClose : void {
        print("onClose\n")
    }
    def override onMessage(msg : string#) : void {
        print("onMessage: {msg}\n")
    }
}

[export]
def main() {
    var wsc = new ExampleWebSocketClient()
    wsc->init("ws://127.0.0.1:9999/test")
    while (wsc->is_connected()) {
        wsc->process_event_que()
        let st = fgets(fstdin())
        if (st == "quit\n") {
            break
        }
        wsc->send(st)
    }
}
