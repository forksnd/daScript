options gen2
require dastest/testing_boost
require daslib/fio
require math

// Regression: fwrite/fread of an array past 2 GiB. The pre-fix rail computed the byte count
// as int32 `length * sizeof`, which wraps negative past INT_MAX; the C runtime sign-extends
// that to a huge size_t and streams out-of-bounds heap garbage into the file (or back over
// the heap on read) with no diagnostic. The fixed rail chunks in int64.
//
// Costs ~2.2 GB of RAM and of disk for a few seconds, so it only runs when DAS_BIG_IO=1 —
// the dasLLAMA image suite exercises the same rail end-to-end on real >2 GiB weight planes.

let BIG = int64(0x80000000) + 65536l   // just past the int32 wrap
let PROBE_STEP = int64(0x2000000)      // spot-check every 32 MiB — full compare would double the cost

[test]
def test_big_write_read(t : T?) {
    if (get_env_variable("DAS_BIG_IO") != "1") {
        t |> skip("big-IO regression needs DAS_BIG_IO=1 (2.2 GB RAM + disk)")
        return
    }
    t |> run("fwrite/fread past 2 GiB") @@(t : T?) {
        var terr : string
        let path = "{temp_directory(terr)}/das_big_io_test.bin"
        var buf : array<uint8>
        buf |> resize(BIG)
        for (k in range64(0l, BIG / PROBE_STEP)) {
            buf[k * PROBE_STEP] = uint8(k + 1l)
        }
        buf[BIG - 1l] = uint8(0xAB)
        fopen(path, "wb") $(f) {
            t |> success(f != null)
            f |> fwrite(buf)
        }
        t |> equal(int64(stat(path).size), BIG)   // pre-fix: garbage-padded file, wrong size
        for (k in range64(0l, BIG / PROBE_STEP)) {
            buf[k * PROBE_STEP] = uint8(0)
        }
        buf[BIG - 1l] = uint8(0)
        fopen(path, "rb") $(f) {
            t |> success(f != null)
            f |> fread(buf)
        }
        for (k in range64(0l, BIG / PROBE_STEP)) {
            t |> equal(buf[k * PROBE_STEP], uint8(k + 1l))
        }
        t |> equal(buf[BIG - 1l], uint8(0xAB))
        delete buf
        remove(path)
    }
}
