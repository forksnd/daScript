options gen2
require dastest/testing_boost public
require daslib/fio
require tl_launch

// Launch-rail helpers that run without a subprocess: quote-aware argv split and the
// mtime/size debounced file watcher.

[test]
def test_split_args(t : T?) {
    t |> run("quote-aware argv split") @(t : T?) {
        let a <- split_args("-m \"a b.gguf\" -p  512")
        t |> equal(4, length(a))
        t |> equal("-m", a[0])
        t |> equal("a b.gguf", a[1])
        t |> equal("-p", a[2])
        t |> equal("512", a[3])
        let b <- split_args("")
        t |> equal(0, length(b))
        let c <- split_args("single")
        t |> equal(1, length(c))
    }
}

[test]
def test_file_watch(t : T?) {
    t |> run("watch fires on change only after a stable poll") @(t : T?) {
        var terr = ""
        let dir = temp_directory(terr)
        let path = "{dir}/tl_watch_probe.txt"
        fopen(path, "wb") $(f) {
            if (f != null) {
                f |> fwrite("one")
            }
        }
        var w = FileWatch()
        watch_prime(w, path)
        t |> success(!watch_poll(w, path), "no change -> no fire")
        fopen(path, "wb") $(f) {
            if (f != null) {
                f |> fwrite("two-longer")
            }
        }
        t |> success(!watch_poll(w, path), "first poll after change arms the debounce")
        t |> success(watch_poll(w, path), "second (stable) poll fires")
        t |> success(!watch_poll(w, path), "and it rearms quietly")
    }
}
