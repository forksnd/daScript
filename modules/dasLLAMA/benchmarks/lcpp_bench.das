options gen2
options stack = 65536   // dasLLAMA program-root convention (options stack does not unify up from libs)
options _jit_fast_math = true   // ggml-parity FP laxity — matches how llama.cpp is benched

require dasllama/dasllama_transformer   // nolint:STYLE029 — umbrella fires each arch [init] registration; requiring dasllama_common directly would drop them
require dasllama/dasllama_math   // setup_dasllama_jobque_ + apply_box_profile_runtime (engine-level, not re-exported)
require ../performance/profile_common.das   // profile_threads + affinity_on — the standing per-box methodology
require daslib/jobque_boost
require daslib/clargs
require daslib/fio
require math

// llama-bench mirror — EXACTLY llama.cpp's test shapes, rep counts, and timing boundaries, and
// NOTHING else. Two independent rows, each = one untimed warmup + -r timed reps, mean ± stdev:
//   pp: one batched forward_prefill of -p tokens from an empty cache per rep
//   tg: -n single-token forwards at pos 0.. per rep, feeding synthetic ids with NO logit read
//       (llama-bench never samples — sampling cost is excluded by design)
//   ours:   bin/daslang -jit modules/dasLLAMA/benchmarks/lcpp_bench.das -- -m <model.gguf>
//   theirs: llama-bench -m <model.gguf> -t 16 -p 512 -n 128 [--cpu-mask 0x55555555 --cpu-strict 1]

[CommandLineArgs]
struct Args {
    @clarg_required
    @clarg_short = "m"
    @clarg_doc = "Model GGUF path"
    model : string

    @clarg_short = "p"
    @clarg_doc = "Prompt tokens per pp rep (default: 512, llama-bench -p)"
    plen : int = 512

    @clarg_short = "n"
    @clarg_doc = "Generated tokens per tg rep (default: 128, llama-bench -n)"
    ngen : int = 128

    @clarg_short = "r"
    @clarg_doc = "Timed repetitions per row (default: 5, llama-bench -r)"
    reps : int = 5

    @clarg_short = "q"
    @clarg_doc = "Weight quant mode (default: q8)"
    quant : QuantMode = QuantMode.q8

    @clarg_doc = "A/B the dense GPU rail in-process: every row runs twice, dense routing OFF then ON. Needs DASLLAMA_GPU_DENSE=1 (uploads + marks); both arms share one process/residency, so cross-run pp drift cancels"
    dense_ab : bool

    @clarg_short = "?"
    @clarg_doc = "Show this help and exit"
    help : bool
}

// deterministic synthetic ids (vocab-safe low range, the decode_prof convention); llama-bench
// feeds std::rand() — token content doesn't affect speed, determinism keeps runs comparable
def private synth_id(i, seed : int64) : int64 => 1000l + ((i * 17l + seed * 131l) % 5000l)

def private build_prompt(n : int64; seed : int64) : array<int64> {
    return <- [for (i in range64(n)); synth_id(i, seed)]
}

def private report(tag : string; tps : array<double>) {
    var sum = 0.0lf
    for (v in tps) {
        sum += v
    }
    let n = length(tps)
    let mean = sum / double(n)
    var vsum = 0.0lf
    for (v in tps) {
        vsum += (v - mean) * (v - mean)
    }
    let sd = n > 1 ? double(sqrt(float(vsum / double(n - 1)))) : 0.0lf
    print("{tag}: {mean:.2f} ± {sd:.2f} tok/s ({n} reps)\n")
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
        print_help(get_command_info(type<Args>), "lcpp_bench")
        return
    }
    let cfg <- r |> move_unwrap
    if (cfg.help) {
        print_help(get_command_info(type<Args>), "lcpp_bench")
        return
    }
    let plen = max(int64(cfg.plen), 0l)
    let ngen = max(int64(cfg.ngen), 0l)
    let reps = max(cfg.reps, 1)
    // the standing per-box methodology, enforced here rather than left to the environment
    // (gen_profile's pattern): cap the future que's workers (computing main is the last lane) and
    // hard-pin lanes to distinct physical cores on the boxes that pin. Explicit DAS_JOBQUE_THREADS
    // / DAS_JOBQUE_AFFINITY envs still win. Must run before any with_job_que.
    let threads = has_env_variable("DAS_JOBQUE_THREADS") ? to_int(get_env_variable("DAS_JOBQUE_THREADS")) : profile_threads()
    if (!has_env_variable("DAS_JOBQUE_THREADS")) {
        set_jobque_threads_cap(max(threads - 1, 1))
    }
    let pinned = affinity_on() && !has_env_variable("DAS_JOBQUE_AFFINITY")
    if (pinned) {
        set_jobque_affinity(2)   // 2 = hard mask, the llama-bench --cpu-strict mirror (mode 1 re-rolls SMT placement)
    }
    apply_box_profile_runtime()   // the tuned per-box stack — what load_model users get
    print("loading {cfg.model}\n")
    var t <- load_gguf(cfg.model, cfg.quant)
    t.config.seq_len = min(t.config.seq_len, max(plen, ngen) + 8l)   // cap the KV allocation
    let c = t.config
    print("config: dim={c.dim} layers={c.n_layers} vocab={c.vocab_size} | backend {active_kernel_backend()} (batch {active_batch_backend()}) | t{threads} affinity {pinned ? "hard" : "env/off"} | pp{plen} tg{ngen} x {reps} reps\n")
    if (cfg.dense_ab && !has_env_variable("DASLLAMA_GPU_DENSE")) {
        print("warning: --dense-ab without DASLLAMA_GPU_DENSE=1 — no dense marks, both arms identical\n")
    }
    with_job_que() {
        setup_dasllama_jobque_()
        var s <- make_run_state(c)
        for (arm in range(cfg.dense_ab ? 2 : 1)) {
            let dtag = cfg.dense_ab ? (arm == 0 ? " dense=off" : " dense=on") : ""
            if (cfg.dense_ab) {
                set_moe_gpu_dense_route(arm != 0)
            }
            if (plen > 0l) {
                let warm <- build_prompt(plen, 999l)
                forward_prefill(t, s, warm, plen, 0l)   // warmup — untimed full-size run, as llama-bench does
                var tps : array<double>
                for (rep in range(reps)) {
                    let prompt <- build_prompt(plen, int64(rep))
                    let t0 = ref_time_ticks()
                    forward_prefill(t, s, prompt, plen, 0l)
                    let us = get_time_usec(t0)
                    tps |> push(double(plen) * 1.0e6lf / double(us))
                }
                report("pp{plen}{dtag}", tps)
                delete tps
            }
            if (ngen > 0l) {
                forward(t, s, synth_id(0l, 999l), 0l)   // warmup — one untimed token, llama-bench's test_gen(ctx, 1)
                var tps : array<double>
                for (rep in range(reps)) {
                    let t0 = ref_time_ticks()
                    for (i in range64(ngen)) {
                        forward(t, s, synth_id(i, int64(rep)), i)
                    }
                    let us = get_time_usec(t0)
                    tps |> push(double(ngen) * 1.0e6lf / double(us))
                }
                report("tg{ngen}{dtag}", tps)
                delete tps
            }
        }
        delete s
    }
    delete t
}
