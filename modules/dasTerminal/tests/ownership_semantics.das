options gen2
options persistent_heap
options gc

require terminal/terminal
require strings

[export]
def main {
    unsafe { heap_collect(true, true) }
    var inscope terminal <- terminal_create(240, 100)
    let live_grid_bytes = heap_bytes_allocated()

    for (_ in range(20)) {
        terminal_feed(terminal, "{to_char(27)}[2J{to_char(27)}[Hgit status\r\n")
    }

    let after_repaint_bytes = heap_bytes_allocated()
    assert(after_repaint_bytes <= live_grid_bytes + 1024ul,
        "replacing terminal rows must release or reuse their owned cell arrays")

    terminal_dispose(terminal)
    let after_dispose_bytes = heap_bytes_allocated()
    assert(after_dispose_bytes == 0ul,
        "disposing a terminal must release all owned arrays")
}
