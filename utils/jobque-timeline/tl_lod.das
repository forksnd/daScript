options gen2
options indenting = 4

module tl_lod shared

require tl_model
require math

// Per-lane time-bucket mipmaps + level choice. Two cost levers in the viewer:
//  - bucket path: when a view holds too many events, walk fixed buckets instead
//  - run merge: adjacent same-color buckets collapse into one draw run
// The viewer additionally replays cached runs until the view changes, so
// steady-state frames never re-walk events at all.

let public MIP_BASE = 2048            // level-0 bucket count (also the overview source)
let public EVENT_PATH_MAX = 8000      // visible events above this -> bucket path

def private build_level(lane : Lane; span : double; nb : int; ncats : int) : MipLevel {
    var lvl = MipLevel(nb = nb)
    lvl.b |> resize(nb)
    var cat_acc : array<float>       // nb x ncats dominant-category accumulator
    cat_acc |> resize(nb * ncats)
    let bw = span / double(nb)
    for (e in lane.ev) {
        let kind = ev_kind(e.meta)
        let t0 = e.t0
        let t1 = e.t0 + double(e.dur)
        var b0 = int(t0 / bw)
        var b1 = int(t1 / bw)
        b0 = clamp(b0, 0, nb - 1)
        b1 = clamp(b1, 0, nb - 1)
        for (bi in range(b0, b1 + 1)) {
            var bk & = unsafe(lvl.b[bi])
            let ov = float((min(t1, double(bi + 1) * bw) - max(t0, double(bi) * bw)) / bw)
            if (ov <= 0.0f) {
                continue
            }
            if (kind == EV_CHUNK || kind == EV_FIFO || kind == EV_PUBLISH) {
                bk.busy += ov
                cat_acc[bi * ncats + ev_cat(e.meta)] += ov
            } elif (kind == EV_STAGE_WAIT) {
                bk.wait += ov
            }
            bk.count++
        }
    }
    for (bi in range(nb)) {
        var bk & = unsafe(lvl.b[bi])
        bk.busy = min(bk.busy, 1.0f)
        bk.wait = min(bk.wait, 1.0f)
        var best = -1
        var bestv = 0.0f
        for (c in range(ncats)) {
            let v = cat_acc[bi * ncats + c]
            if (v > bestv) {
                bestv = v
                best = c
            }
        }
        bk.dom = best
    }
    delete cat_acc
    return <- lvl
}

//! Build mip levels for every lane: level 0 always (the overview source), finer levels
//! while the lane has enough events to need them.
def public build_mips(var doc : TraceDoc) {
    let span = max(doc.span, 8.0lf)
    let ncats = length(doc.cats)
    for (li in range(length(doc.lanes))) {
        var lane & = unsafe(doc.lanes[li])
        delete lane.mips
        lane.mips |> reserve(4)
        var nb = MIP_BASE
        while (true) {
            lane.mips |> emplace(build_level(lane, span, nb, ncats))
            if (nb * 2 >= length(lane.ev) || nb >= 131072) {
                break
            }
            nb *= 4
        }
    }
}

//! Deepest built level whose bucket is at least `min_px` pixels wide at the current zoom.
def public pick_level(lane : Lane; span : double; px_per_us : double; min_px : double) : int {
    var best = 0
    for (li in range(length(lane.mips))) {
        let bw_px = (span / double(lane.mips[li].nb)) * px_per_us
        if (bw_px >= min_px) {
            best = li
        }
    }
    return best
}
