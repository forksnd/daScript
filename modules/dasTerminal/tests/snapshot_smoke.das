options gen2

require terminal/terminal
require strings

[export]
def main {
    var inscope state <- terminal_create(8, 2)
    terminal_feed(state, "hello")

    var snapshot = TerminalSnapshot()
    terminal_snapshot(state, snapshot)

    assert(snapshot.columns == 8)
    assert(snapshot.rows == 2)
    assert(length(snapshot.normal.cells) == 16)
    assert(snapshot.normal.cells[0].grapheme == "h")
    assert(snapshot.normal.cells[4].grapheme == "o")
    assert(snapshot.normal.cursor.row == 0)
    assert(snapshot.normal.cursor.column == 5)
    assert(snapshot.modes.auto_wrap)

    let up = terminal_encode_key(state, TerminalKeyEvent(key = TerminalKey.up))
    let esc = to_char(27)
    assert(length(up) == 3)
    assert(up == "{esc}[A")

    terminal_feed(state, "{esc}[?2004h")
    let paste = terminal_encode_paste(state, "hello")
    assert(length(paste) == 17)
    assert(paste == "{esc}[200~hello{esc}[201~")

    // Failed launch remains an inspectable terminal handle instead of losing
    // the transport error at the native boundary.
    var inscope failed <- terminal_launch("")
    assert(terminal_process_id(failed) == 0)
    let poll_status = terminal_poll(failed)
    let transport_error = terminal_transport_error(failed)
    let process_exited = terminal_process_exited(failed)
    let exit_code = terminal_exit_code(failed)
    let terminated = terminal_terminate(failed)
    assert(poll_status == TerminalPollStatus.error)
    assert(length(transport_error) > 0)
    assert(!process_exited)
    assert(exit_code == -1l)
    assert(!terminated)
}
