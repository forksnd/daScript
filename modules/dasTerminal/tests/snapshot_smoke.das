options gen2

require terminal/terminal

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
}
