options gen2
require dastest/testing_boost public
require math
require tl_model
require tl_stats

// RangeStats on a hand-built doc: clipping at range edges, idle/parallelism math,
// per-category shares, cache freshness.

def make_doc() : TraceDoc {
    var doc = TraceDoc(cats <- base_categories(), span = 100.0lf)
    doc.lanes |> resize(2)
    // lane 0: one 10µs attn chunk [10,20], one publish [5,6]
    doc.lanes[0].ev <- [
        Ev(t0 = 5.0lf, dur = 1.0f, meta = ev_meta(EV_PUBLISH, 2, 2, CAT_ATTN), arg = 0u, chain = 1u),
        Ev(t0 = 10.0lf, dur = 10.0f, meta = ev_meta(EV_CHUNK, 0, 2, CAT_ATTN), arg = 0u, chain = 1u)
    ]
    doc.lanes[0].max_dur = 10.0lf
    // lane 1: a 20µs ffn chunk [15,35] (clips at range edge), a 5µs stage_wait [40,45]
    doc.lanes[1].ev <- [
        Ev(t0 = 15.0lf, dur = 20.0f, meta = ev_meta(EV_CHUNK, 1, 2, CAT_FFN), arg = 1u, chain = 1u),
        Ev(t0 = 40.0lf, dur = 5.0f, meta = ev_meta(EV_STAGE_WAIT, 1, 2, CAT_FFN), arg = 0u, chain = 1u)
    ]
    doc.lanes[1].max_dur = 20.0lf
    doc.generation = 1
    return <- doc
}

[test]
def test_range_stats(t : T?) {
    t |> run("busy/idle/parallelism with edge clipping") @(t : T?) {
        var doc <- make_doc()
        // range [10,25]: lane0 chunk fully in (10), lane1 chunk clipped to [15,25] (10);
        // publish [5,6] outside; wait outside
        let st <- compute_range_stats(doc, 10.0lf, 25.0lf)
        t |> equal(2, st.events)
        t |> equal(20.0lf, st.busy_us)
        t |> equal(0.0lf, st.wait_us)
        t |> equal(0, st.publish_count)
        // idle = 1 - 20/(15*2) = 33.33%
        t |> success(abs(st.idle_pct - 33.333f) < 0.1f, "idle pct: {st.idle_pct}")
        t |> success(abs(st.parallelism - 20.0f / 15.0f) < 0.01f, "parallelism: {st.parallelism}")
        t |> equal(2, length(st.cat))
        t |> equal(st.cat[0].busy_us, st.cat[1].busy_us)   // 10 attn + 10 ffn
    }
}

[test]
def test_range_stats_full(t : T?) {
    t |> run("full span counts publish and wait") @(t : T?) {
        var doc <- make_doc()
        let st <- compute_range_stats(doc, 0.0lf, 100.0lf)
        t |> equal(4, st.events)
        t |> equal(1, st.publish_count)
        t |> equal(31.0lf, st.busy_us)   // 10 + 20 + 1 (publish counts as work)
        t |> equal(5.0lf, st.wait_us)
    }
}

[test]
def test_stats_cache(t : T?) {
    t |> run("cache recomputes only on key change") @(t : T?) {
        var doc <- make_doc()
        var cache = StatsCache()
        cached_stats(cache, doc, 10.0lf, 25.0lf)
        t |> equal(2, cache.stats.events)
        // same key -> stays; poke the stats to detect a recompute
        cache.stats.events = 999
        cached_stats(cache, doc, 10.0lf, 25.0lf)
        t |> equal(999, cache.stats.events)
        // generation bump -> recompute
        doc.generation++
        cached_stats(cache, doc, 10.0lf, 25.0lf)
        t |> equal(2, cache.stats.events)
    }
}
