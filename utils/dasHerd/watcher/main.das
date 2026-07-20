options gen2
options rtti
options persistent_heap
options gc
options indenting = 4

require ./watcher_server.das
require daslib/random
require daslib/fio
require daslib/module_path
require live_host
require strings
require math

def private option_value(prefix : string; fallback : string = "") : string {
    for (argument in get_command_line_arguments()) {
        if (starts_with(argument, prefix)) return slice(argument, length(prefix))
    }
    return fallback
}

def private make_local_token : string {
    // This first slice is deliberately loopback-only. The token prevents an
    // accidental drive-by from another local page; remote deployment will use
    // a real authentication and pairing design.
    var seed = random_seed(int(ref_time_ticks()))
    return "{random_uint(seed)}-{random_uint(seed)}-{random_uint(seed)}-{random_uint(seed)}"
}

[export]
def main {
    let port = clamp(to_int(option_value("--port=", "9191")), 1024, 65535)
    let token_arg = option_value("--token=")
    let token = empty(token_arg) ? make_local_token() : token_arg
    let log_root = option_value("--log-root=",
        path_join(get_das_root(), "logs/dasHerd/watcher/sessions"))
    let init_error = watcher_init(log_root)
    if (!empty(init_error)) {
        print("dasHerd watcher: could not initialize logs: {init_error}\n")
        return 2
    }

    var server = new DasHerdWatcherServer()
    server.token = token
    server.control_html = path_join(get_this_module_dir(), "control.html")
    server->init(port)
    if (!server->set_bind_host("127.0.0.1")) {
        print("dasHerd watcher: could not select loopback bind address\n")
        server->cleanup()
        unsafe { delete server }
        watcher_shutdown()
        return 3
    }
    let start_result = server->start()
    if (start_result != 0) {
        print("dasHerd watcher: server start failed ({start_result})\n")
        server->cleanup()
        unsafe { delete server }
        watcher_shutdown()
        return 4
    }

    print("dasHerd watcher ready\n")
    print("  control: http://127.0.0.1:{port}/?token={token}\n")
    print("  logs:    {log_root}\n")
    print("  bind:    127.0.0.1:{port}\n")
    while (!exit_requested()) {
        server->tick()
        sleep(2u)
    }
    let _ = server->stop()
    server->cleanup_clients()
    server->cleanup()
    unsafe { delete server }
    watcher_shutdown()
    return 0
}
