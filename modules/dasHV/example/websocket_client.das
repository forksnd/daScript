options gen2
options persistent_heap
options gc

require dashv/dashv_boost

require fio

[export]
def main() {
    GET("http://127.0.0.1:9999/ping") <| $(var resp : HttpResponse?) {
        if (resp != null) {
            print("GET `{resp.body}` {resp.status_code}\n")
        } else {
            print("GET failed\n")
        }
    }

    POST("http://127.0.0.1:9999/echo", "hello, world!") <| $(var resp : HttpResponse?) {
        if (resp != null) {
            print("POST `{resp.body}` {resp.status_code}\n")
        } else {
            print("POST failed\n")
        }
    }
}
