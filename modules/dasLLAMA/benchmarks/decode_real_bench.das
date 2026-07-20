options gen2
options stack = 65536   // dasLLAMA program-root convention (options stack does not unify up from libs)
options _jit_fast_math = true   // ggml-parity FP laxity — matches how llama.cpp is benched

require dasllama/dasllama_transformer   // nolint:STYLE029 — umbrella fires each arch [init] registration; requiring dasllama_common directly would drop them
require dasllama/dasllama_math   // setup_dasllama_jobque_ + apply_box_profile_runtime (engine-level, not re-exported)
require dasllama/dasllama_image   // load_model_cached — the .dlim rail, without the full facade's compile weight
require dasllama/dasllama_tokenizer
require ../performance/profile_common.das   // profile_threads + affinity_on — the standing per-box methodology
require daslib/jobque_boost
require daslib/module_path
require daslib/strings_boost
require daslib/clargs
require daslib/fio
require math

// The decode-arc judge: REAL greedy continuation over a prompt corpus, timed per rep. NOT a
// llama-bench mirror and NOT comparable to lcpp_bench tg rows — synthetic-id decode overstates
// expert reuse (measured 91% vs 81% LRU-64 on the 35B), and the greedy argmax runs inside the
// timed loop (A/B arms share it, so it cancels for judging).
//   bin/daslang -jit modules/dasLLAMA/benchmarks/decode_real_bench.das -- -m <model.gguf>

[CommandLineArgs]
struct Args {
    @clarg_required
    @clarg_short = "m"
    @clarg_doc = "Model GGUF path"
    model : string

    @clarg_doc = "Prompt corpus (segments separated by lines of two percent signs; default: data/wikitext2_prompts.txt beside this bench)"
    prompts : string = ""

    @clarg_short = "n"
    @clarg_doc = "Greedy tokens per prompt (default: 256)"
    ngen : int = 256

    @clarg_short = "r"
    @clarg_doc = "Timed repetitions over the whole corpus (default: 3)"
    reps : int = 3

    @clarg_short = "q"
    @clarg_doc = "Weight quant mode (default: q8)"
    quant : QuantMode = QuantMode.q8

    @clarg_doc = "After the timed reps, run one untimed decode window with forward_profile buckets (the per-token time split)"
    prof : bool

    @clarg_doc = "Arm the expert-reuse counter and dump its report at the end (adds <1% counting overhead to timed rows)"
    reuse : bool

    @clarg_doc = "Run the timed reps twice in-process — deltanet decode step OFF then ON (needs DASLLAMA_GPU_DND=1 for the uploads; the warmup pre-builds the ON arm's step chains)"
    dnd_ab : bool

    @clarg_doc = "Run the timed reps twice in-process — heat-pinned expert cache OFF then ON (needs DASLLAMA_GPU_HEAT=<slots> + DASLLAMA_GPU_MOE_STREAM covering the layers: uploads DMA from the pinned mirrors)"
    heat_ab : bool

    @clarg_short = "?"
    @clarg_doc = "Show this help and exit"
    help : bool
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

def private argmax_logit(s : Session; vocab : int64) : int64 {
    var best = 0l
    var bv = s.logits[0]
    for (v in range64(vocab)) {
        if (s.logits[v] > bv) {
            bv = s.logits[v]
            best = v
        }
    }
    return best
}

def private load_corpus(path : string) : array<string> {
    var texts : array<string>
    fopen(path, "rb") $(f) {
        if (f == null) {
            panic("can't open prompt corpus")
        }
        for (p in split(fread(f), "%%")) {
            let tp = strip(p)
            if (!empty(tp) && !starts_with(tp, "#")) {
                texts |> push(clone_string(tp))
            }
        }
    }
    return <- texts
}

[export]
def main() {
    allow_cpu_prefill()   // CPU-protocol bench by intent — disarm the Metal-build tripwire
    if (!jit_enabled()) {
        print("This benchmark is JIT-only — run with: daslang -jit <file>\n")
        return
    }
    var r <- parse_args(type<Args>)
    if (r |> is_err) {
        print("error: {r |> unwrap_err}\n\n")
        print_help(get_command_info(type<Args>), "decode_real_bench")
        return
    }
    let cfg <- r |> move_unwrap
    if (cfg.help) {
        print_help(get_command_info(type<Args>), "decode_real_bench")
        return
    }
    let ngen = max(int64(cfg.ngen), 1l)
    let reps = max(cfg.reps, 1)
    let corpus = empty(cfg.prompts) ? path_join(get_this_module_dir(), "data/wikitext2_prompts.txt") : cfg.prompts
    var texts <- load_corpus(corpus)
    if (empty(texts)) {
        print("error: no prompts in {corpus}\n")
        return
    }
    // the standing per-box methodology (lcpp_bench's block): cap the future que's workers and
    // hard-pin lanes on the boxes that pin; explicit DAS_JOBQUE_THREADS/AFFINITY envs still win
    let threads = has_env_variable("DAS_JOBQUE_THREADS") ? to_int(get_env_variable("DAS_JOBQUE_THREADS")) : profile_threads()
    if (!has_env_variable("DAS_JOBQUE_THREADS")) {
        set_jobque_threads_cap(max(threads - 1, 1))
    }
    let pinned = affinity_on() && !has_env_variable("DAS_JOBQUE_AFFINITY")
    if (pinned) {
        set_jobque_affinity(2)
    }
    apply_box_profile_runtime()
    if (cfg.reuse) {
        set_expert_reuse(true)
    }
    var tk <- load_tokenizer_auto(cfg.model)
    print("loading {cfg.model}\n")
    with_job_que() {
        setup_dasllama_jobque_()
        var t <- load_model_cached(cfg.model, cfg.quant)
        var toks : array<array<int64>>
        var maxlen = 0l
        for (tx in texts) {
            var ids <- encode(tk, tx, false)
            maxlen = max(maxlen, int64(length(ids)))
            toks |> emplace(ids)
        }
        t.config.seq_len = min(t.config.seq_len, maxlen + ngen + 8l)
        let c = t.config
        print("config: dim={c.dim} layers={c.n_layers} experts={c.n_expert} k={c.n_expert_used} | backend {active_kernel_backend()} | t{threads} affinity {pinned ? "hard" : "env/off"} | {length(toks)} prompts x {ngen} tok x {reps} reps\n")
        // loud tier line — a silent GPU no-engage (absent DLL, failed probe) must never pass as a row
        let gs = gpu_tier_status()
        let gname = empty(gs.backend) ? "OFF" : "{gs.backend} {gs.device}"
        let gcls = gs.cls_resident ? "on" : "off"
        let gwhy = gs.supported ? "" : " | not riding: {gs.reason}"
        print("gpu: {gname} | resident {gs.resident_layers}L streamed {gs.streamed_layers}L dn {gs.dn_layers} attn {gs.attn_layers} dense {gs.dense_planes} cls {gcls} vram {gs.vram_bytes / 1000000l}MB{gwhy}\n")
        {
            // warmup — one untimed prompt window, as lcpp_bench does per row
            var s <- make_run_state(c)
            let plen = int64(length(toks[0]))
            forward_prefill(t, s, toks[0], plen, 0l)
            for (g in range64(min(ngen, 32l))) {
                forward(t, s, argmax_logit(s, c.vocab_size), plen + g)
            }
            delete s
        }
        if (cfg.dnd_ab && cfg.heat_ab) {
            panic("decode_real_bench: --dnd-ab and --heat-ab are mutually exclusive (one route flip per run)")
        }
        let narms = cfg.dnd_ab || cfg.heat_ab ? 2 : 1
        for (ai in range(narms)) {
            if (cfg.dnd_ab) {
                set_moe_gpu_dnd_route(ai == 1)   // route flip only — uploads and residency stay
            } elif (cfg.heat_ab) {
                // OFF first so the ON arm's counters/pins warm during its own reps (rep 1 runs
                // partly cold — the mean understates ON slightly; the delta stays the judge)
                set_moe_gpu_heat_route(ai == 1)
            }
            var tps : array<double>
            for (_rep in range(reps)) {
                var total_us = 0l
                for (pi in range(length(toks))) {
                    var s <- make_run_state(c)   // fresh session per prompt — no KV/DN-state bleed
                    let plen = int64(length(toks[pi]))
                    forward_prefill(t, s, toks[pi], plen, 0l)   // untimed, as llama-bench excludes the prompt
                    let t0 = ref_time_ticks()
                    var pos = plen
                    for (_g in range64(ngen)) {
                        forward(t, s, argmax_logit(s, c.vocab_size), pos)
                        pos++
                    }
                    total_us += int64(get_time_usec(t0))
                    delete s
                }
                tps |> push(double(int64(length(toks)) * ngen) * 1.0e6lf / double(total_us))
            }
            report(cfg.dnd_ab ? "tg-real{ngen} dnd-{ai == 1 ? "on" : "off"}" : (cfg.heat_ab ? "tg-real{ngen} heat-{ai == 1 ? "on" : "off"}" : "tg-real{ngen}"), tps)
            delete tps
        }
        if (cfg.prof) {
            // decode-only bucket window: prefill lands before the reset, so the report is pure decode
            var s <- make_run_state(c)
            let plen = int64(length(toks[0]))
            forward_prefill(t, s, toks[0], plen, 0l)
            forward_profile_reset()
            var pos = plen
            for (_g in range64(ngen)) {
                forward(t, s, argmax_logit(s, c.vocab_size), pos)
                pos++
            }
            forward_profile_report()
            delete s
        }
        if (cfg.reuse) {
            expert_reuse_report(false)
        }
        heat_report(cfg.prof)   // silent when the cache never ran
        delete toks
        delete t
    }
    delete tk
    delete texts
}
