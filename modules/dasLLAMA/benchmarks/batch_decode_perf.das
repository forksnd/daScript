options gen2
options stack = 65536   // dasLLAMA program-root convention (options stack does not unify up from libs)
options _jit_fast_math = true   // ggml-parity FP laxity — same rail as decode_prof / perf

require dasllama/dasllama_transformer   // nolint:STYLE029 — umbrella fires each arch [init] registration
require dasllama/dasllama_math   // setup_dasllama_jobque_ + pin_kernel_backend (engine-level, not re-exported)
require ?das_metal dasllama/dasllama_metal_llama  // nolint:STYLE030 — per-stage report on the GPU rail (Apple builds only)
require daslib/jobque_boost
require daslib/clargs
require daslib/fio
require math

// Batched-decode throughput: B independent sessions stepped together through eval_batch vs the
// same B streams decoded one after another with forward() (fused chain ON — the honest baseline).
// The weight GEMVs amortize into B-row GEMMs; attention stays per-row by physics, so the win
// shrinks as context deepens and the KV read dominates.
//   bin/daslang -jit modules/dasLLAMA/benchmarks/batch_decode_perf.das -- -m <model.gguf> [-p 64] [-n 64] [-b 8]
// GPU (Metal) mode — both the sequential baseline (single-stream decode override) and the batched
// steps (batch override) ride the resident driver; size the mirror arena for B streams at depth:
//   DASLLAMA_METAL_KV_MIRROR_MB=3072 DASLLAMA_PIN_DECODE=metal DASLLAMA_PIN_BACKEND=portable \
//     bin/daslang -jit modules/dasLLAMA/benchmarks/batch_decode_perf.das -- -m <3B.gguf> -p 512 -n 128 -b 16 --kv f16 --wscale-f16 1

[CommandLineArgs]
struct Args {
    @clarg_required
    @clarg_short = "m"
    @clarg_doc = "Model GGUF path"
    model : string

    @clarg_short = "p"
    @clarg_doc = "Prefill depth before the timed decode (default: 64)"
    prefill : int = 64

    @clarg_short = "n"
    @clarg_doc = "Decode steps timed per stream (default: 64)"
    ngen : int = 64

    @clarg_short = "b"
    @clarg_doc = "Largest batch size — sweeps 1,2,4,... up to this (default: 8)"
    bmax : int = 8

    @clarg_short = "q"
    @clarg_doc = "Weight quant mode (default: q8)"
    quant : QuantMode = QuantMode.q8

    @clarg_doc = "KV cache dtype (default: f32)"
    kv : KVDtype = KVDtype.f32

    @clarg_doc = "Weight-scale plane A/B: 1 = convert to f16 at load (llama.cpp byte parity), 0 = keep f32, -1 = keep default"
    wscale_f16 : int = -1

    @clarg_doc = "Feed this fixed token every step instead of greedy argmax (llama-batched-bench protocol parity; -1 = greedy)"
    fixed_token : int = -1

    @clarg_short = "?"
    @clarg_doc = "Show this help and exit"
    help : bool
}

// llama-batched-bench feeds a fixed token per step (no sampling) — the greedy argmax here is a
// measured ~0.25 ms/row/step of harness cost, so cross-tool rows use --fixed-token
var private g_fixed_tok = -1l

def private build_prompt(n, salt : int64) : array<int64> {
    return <- [for (i in range64(n)); 1000l + ((i * 7l + salt * 131l) % 5000l)]
}

// team-scanned argmax (bit-identical to a serial first-wins scan) — the serial loop was a
// ~0.75ms all-lane hole PER STREAM between eval_batch steps (B=8: 6.1ms holes, 18% of the run)
def private greedy_top(lg : array<float>; vocab : int64) : int64 {
    return g_fixed_tok >= 0l ? g_fixed_tok : parallel_argmax(lg, vocab)
}

// nb fresh sessions, each prefilled to depth p with its own (salted) prompt
def private make_streams(t : Model; kvdt : KVDtype; nb, p : int64) : array<Session> {
    var ss : array<Session>
    ss |> reserve(int(nb))
    for (i in range64(nb)) {
        ss |> emplace <| make_run_state(t.config, kvdt, kvdt)
        var prompt <- build_prompt(p, i)
        eval_(t, ss[i], prompt)
        delete prompt
    }
    return <- ss
}

// today's throughput for nb conversations: each stream decoded to completion, one after another
def private run_sequential(t : Model; kvdt : KVDtype; nb, p, n : int64) : double {
    var ss <- make_streams(t, kvdt, nb, p)
    let vocab = t.config.vocab_size
    let t0 = ref_time_ticks()
    for (i in range64(nb)) {
        var tok = greedy_top(ss[i].logits, vocab)
        for (_st in range64(n)) {
            forward(t, ss[i], tok, ss[i].n_past)
            ss[i].n_past++
            tok = greedy_top(ss[i].logits, vocab)
        }
    }
    let us = get_time_usec(t0)
    delete ss
    return double(nb * n) / (double(us) * 1.0e-6lf)
}

// the same nb streams stepped together, one eval_batch per token
def private run_batched(t : Model; kvdt : KVDtype; nb, p, n : int64) : double {
    var ss <- make_streams(t, kvdt, nb, p)
    let vocab = t.config.vocab_size
    var ws <- create_batch_workspace_(t)
    var sess : array<Session?>
    sess |> reserve(int(nb))
    var cur : array<int64>
    cur |> reserve(int(nb))
    for (i in range64(nb)) {
        unsafe {
            sess |> push(addr(ss[i]))
        }
        cur |> push(greedy_top(ss[i].logits, vocab))
    }
    let t0 = ref_time_ticks()
    for (st in range64(n)) {
        trace_marker("step", int(st))   // unit boundary for the timeline viewer (no-op untraced)
        eval_batch_(t, ws, sess, cur)
        for (i in range64(nb)) {
            cur[i] = greedy_top(ss[i].logits, vocab)
        }
    }
    static_if (typeinfo builtin_module_exists(das_metal)) {
        metal_decode_flush()   // encode-ahead rails: land the in-flight step inside the window
    }
    let us = get_time_usec(t0)
    sess |> clear()   // drop the borrowed handles: delete of array<T?> cascades to the pointees
    unsafe {
        delete ws
        delete sess
    }
    delete cur
    delete ss
    return double(nb * n) / (double(us) * 1.0e-6lf)
}

[export]
def main() {
    if (!jit_enabled()) {
        print("This benchmark is JIT-only — run with: daslang -jit <file>\n")
        return
    }
    var r <- parse_args(type<Args>)
    if (r |> is_err) {
        print("error: {r |> unwrap_err}\n\n")
        print_help(get_command_info(type<Args>), "batch_decode_perf")
        return
    }
    let cfg <- r |> move_unwrap
    if (cfg.help) {
        print_help(get_command_info(type<Args>), "batch_decode_perf")
        return
    }
    let p = max(int64(cfg.prefill), 8l)
    let n = max(int64(cfg.ngen), 1l)
    let bmax = clamp(int64(cfg.bmax), 1l, 64l)
    if (has_env_variable("DASLLAMA_PIN_BACKEND")) {
        pin_kernel_backend(get_env_variable("DASLLAMA_PIN_BACKEND"))
    }
    // the wscale rail is a LOAD-TIME convert — the flag must land before load_gguf
    if (cfg.wscale_f16 >= 0) {
        set_wscale_f16(cfg.wscale_f16 != 0)
        print("wscale_f16 override: {cfg.wscale_f16 != 0}\n")
    }
    g_fixed_tok = int64(cfg.fixed_token)
    if (g_fixed_tok >= 0l) {
        print("fixed-token protocol: {g_fixed_tok} (no per-step sampling)\n")
    }
    if (has_env_variable("DASLLAMA_PIN_DECODE")) {
        let dv = get_env_variable("DASLLAMA_PIN_DECODE")
        if (dv == "metal") {
            // measurement runs are MetalMode.required: a CPU fallback anywhere in the run
            // (prefill, decode, batch) PANICS with the decline reason instead of silently
            // polluting the numbers with CPU or mixed steps
            set_metal_mode(MetalMode.required)
            print("decode override: metal (single-stream + batch + prefill), MetalMode.required — CPU fallback fails the run\n")
        } elif (!select_decode_override(dv)) {
            print("unknown decode override `{dv}`\n")
            return
        } else {
            print("decode override: {dv} (single-stream + batch)\n")
        }
    }
    print("loading {cfg.model}\n")
    var t <- load_gguf(cfg.model, cfg.quant)
    t.config.seq_len = min(t.config.seq_len, p + n + 8l)   // cap the per-session KV allocation
    let c = t.config
    print("config: dim={c.dim} layers={c.n_layers} heads={c.n_heads}/{c.n_kv_heads} vocab={c.vocab_size} moe={c.moe}\n")
    print("active backend: {active_kernel_backend()}\n")
    print("kv cache dtype: {cfg.kv}\n")
    print("prefill {p}, {n} decode steps per stream\n\n")
    with_job_que() {
        setup_dasllama_jobque_()
        // warm both code paths (JIT + kernel dispatch) before any timed window
        let _w1 = run_sequential(t, cfg.kv, 1l, 8l, 2l)
        let _w2 = run_batched(t, cfg.kv, 2l, 8l, 2l)
        // DASLLAMA_BATCH_TRACE=<path> saves a per-lane jobque trace of the LARGEST-B batched run
        // (the lane-timeline viewer's input; same rig as decode_prof --trace)
        let trace = has_env_variable("DASLLAMA_BATCH_TRACE") ? get_env_variable("DASLLAMA_BATCH_TRACE") : ""
        var nb = 1l
        while (nb <= bmax) {
            static_if (typeinfo builtin_module_exists(das_metal)) {
                metal_decode_stage_reset()   // per-B stage attribution (the report below)
            }
            let seq_tps = run_sequential(t, cfg.kv, nb, p, n)
            let doTrace = nb == bmax && !empty(trace)
            if (doTrace) {
                jobque_trace_start(1 << 21)
                set_trace_tags(true)
            }
            let bat_tps = run_batched(t, cfg.kv, nb, p, n)
            if (doTrace) {
                set_trace_tags(false)
                jobque_trace_stop()
                if (jobque_trace_save(trace)) {
                    print("jobque trace saved: {trace}\n")
                } else {
                    print("jobque trace: nothing recorded\n")
                }
            }
            let speedup = bat_tps / seq_tps
            let stream_tps = bat_tps / double(nb)
            print("B={nb}: sequential {seq_tps} tok/s | batched {bat_tps} tok/s ({speedup}x) | per-stream {stream_tps} tok/s\n")
            static_if (typeinfo builtin_module_exists(das_metal)) {
                if (metal_batch_decode_stats().steps > 0l || metal_decode_stats().decodes > 0l) {
                    metal_decode_stage_report()   // this B's sequential (single-stream) + batched stage split
                }
            }
            nb *= 2l
        }
    }
    delete t
}
