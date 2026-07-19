options gen2
options persistent_heap

require dastest/testing_boost
require terminal/terminal
require strings

let COLUMNS = 78
let ROWS = 26
let OUTPUT_LINE = "line XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\r\n"

def make_output(lines : int) : string {
    return build_string() $(var writer) {
        for (_ in range(lines)) {
            writer |> write(OUTPUT_LINE)
        }
    }
}

def run_snapshot_case(b : B?; lines : int) {
    var inscope terminal = terminal_create(COLUMNS, ROWS)
    let output = make_output(lines)
    terminal_feed(terminal, output)
    var snapshot = TerminalSnapshot()
    terminal_snapshot(terminal, snapshot)
    let history_rows = snapshot.normal.scrollback_rows

    b |> run("snapshot/history_{history_rows}") {
        terminal_snapshot(terminal, snapshot)
        b |> accept(snapshot.normal.scrollback_rows)
    }

    terminal_dispose(terminal)
}

def run_viewport_case(b : B?; lines : int) {
    var inscope terminal = terminal_create(COLUMNS, ROWS)
    let output = make_output(lines)
    terminal_feed(terminal, output)
    var snapshot = TerminalViewportSnapshot()
    terminal_viewport_snapshot(terminal, 0, snapshot)
    let history_rows = snapshot.buffer.scrollback_rows

    b |> run("viewport/history_{history_rows}") {
        terminal_viewport_snapshot(terminal, 0, snapshot)
        b |> accept(snapshot.buffer.scrollback_rows)
    }

    terminal_dispose(terminal)
}

[benchmark]
def terminal_snapshot_scaling(b : B?) {
    run_snapshot_case(b, 0)
    run_snapshot_case(b, 1000)
    run_snapshot_case(b, 5000)
    run_snapshot_case(b, 10000)
    run_viewport_case(b, 0)
    run_viewport_case(b, 1000)
    run_viewport_case(b, 5000)
    run_viewport_case(b, 10000)
}
