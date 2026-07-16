options gen2
options stack = 65536   // dasLLAMA program-root convention (options stack does not unify up from libs)
options _jit_fast_math = true   // ggml-parity FP laxity (non-bit-exact, ~+10%); tests stay bit-exact

require dasllama/dasllama_transformer   // nolint:STYLE029 — umbrella fires each arch [init] registration; requiring dasllama_common directly would drop them
require daslib/jobque_boost
require daslib/fio
require strings
require math
require dasllama/dasllama_math  // setup_dasllama_jobque_ + the runtime knob setters (engine-level, not re-exported)

// Prompt-processing (TTFT) profile for a Llama-3 GGUF (Q8): the OLD per-token path (N× forward,
// decode-style gemv per token) vs ONE batched forward_prefill. Token content doesn't affect speed,
// so synthetic ids are fine. One run state is reused — each run overwrites the cache positions it
// reads, so no reset is needed between runs.
//   bin/daslang -jit modules/dasLLAMA/dasllama_prefill_perf.das -- <model.gguf>
let DEFAULT_MODEL = "/Users/borisbatkin/Work/llama.cpp/models/Llama-3.2-1B-Instruct-Q8_0.gguf"

def build_prompt(n : int64) : array<int64> {
    return <- [for (i in range64(n)); 1000l + (i % 5000l)]
}

[export]
def main {
    if (!jit_enabled()) {
        print("This benchmark is JIT-only — run with: daslang -jit <file>\n")
        return
    }
    var path = DEFAULT_MODEL
    let args = get_command_line_arguments()
    if (!empty(args) && ends_with(args[length(args) - 1], ".gguf")) {
        path = args[length(args) - 1]
    }
    apply_box_profile_runtime()   // the tuned per-box stack first; the env pins below override it
    // backend pin for A/B on one binary — must land BEFORE load_gguf:
    // select_matmul_backend_for_load honors the pin at load, which decides repack
    if (has_env_variable("DASLLAMA_PIN_BACKEND")) {
        pin_kernel_backend(get_env_variable("DASLLAMA_PIN_BACKEND"))
    }
    if (has_env_variable("DASLLAMA_PIN_BATCH_BACKEND")) {
        select_batch_backend(get_env_variable("DASLLAMA_PIN_BATCH_BACKEND"))
    }
    // A/B the deltanet prefill algorithm on hybrid (qwen35) models: recurrent vs chunked (default)
    if (has_env_variable("DASLLAMA_DN_PREFILL")) {
        let dnm = (get_env_variable("DASLLAMA_DN_PREFILL") == "recurrent"
            ? DnPrefillMode.recurrent : DnPrefillMode.chunked)
        set_deltanet_prefill_mode(dnm)
        print("deltanet prefill mode override: {dnm}\n")   // the EFFECTIVE mode, not the raw env value
    }
    // the metal batch donor's rail: keep the f32 scale plane (the batch override swaps only the
    // f32 batch slot, and a wscale16 load would route prefill through batch_s16 instead)
    if (has_env_variable("DASLLAMA_WSCALE_F16")) {
        set_wscale_f16(get_env_variable("DASLLAMA_WSCALE_F16") != "0")
    }
    // the resident whole-prefill override (dasMetal Phase 6): claims the ENTIRE layer stack;
    // needs the f32 scale plane like the batch donor (DASLLAMA_WSCALE_F16=0)
    if (has_env_variable("DASLLAMA_PIN_PREFILL")) {
        let pf = get_env_variable("DASLLAMA_PIN_PREFILL")
        if (pf == "metal") {
            // measurement runs are MetalMode.required — a CPU fallback fails the run
            set_metal_mode(MetalMode.required)
            print("prefill override: metal, MetalMode.required — CPU fallback fails the run\n")
        } elif (!select_prefill_override(pf)) {
            print("prefill override `{pf}` not registered on this build\n")
        } else {
            print("prefill override: {pf}\n")
        }
    }
    print("loading {path}\n")
    var t <- load_gguf(path, QuantMode.q8)
    t.config.seq_len = min(t.config.seq_len, 8192l)   // cap KV cache (Llama-3 context length is 131072)
    let c = t.config
    print("config: dim={c.dim} hidden={c.hidden_dim} layers={c.n_layers} heads={c.n_heads}/{c.n_kv_heads} vocab={c.vocab_size}\n")
    print("active backend: {active_kernel_backend()}\n")
    print("prompt processing: per-token forward (decode-style) vs one batched forward_prefill\n\n")

    var s <- make_run_state(c)
    with_job_que() {
        setup_dasllama_jobque_()
        // join poll-level override (ggml denomination 0..100; setup default 50, 0 = park at join)
        if (has_env_variable("DAS_JOBQUE_JOIN_SPIN")) {
            let join_spin = to_int(get_env_variable("DAS_JOBQUE_JOIN_SPIN"))
            set_jobque_join_spin(join_spin)
            print("join spin override: level {join_spin}\n")
        }
        // A/B the matmul min-grains (rows per chunk floor); 0 = clamp off (pre-grain behavior)
        if (has_env_variable("DASLLAMA_BATCH_CHUNKS")) {   // batch-GEMM oversplit (chunks per hw job)
            let bc = to_int(get_env_variable("DASLLAMA_BATCH_CHUNKS"))
            set_q8_batch_chunks_per_job(bc)
            print("q8_batch_chunks_per_job override: {bc}\n")
        }
        if (has_env_variable("DASLLAMA_MIN_CHUNK_ROWS")) {
            let mg = to_int(get_env_variable("DASLLAMA_MIN_CHUNK_ROWS"))
            set_matmul_min_chunk_rows(mg)
            print("min chunk rows override: {mg}\n")
        }
        if (has_env_variable("DASLLAMA_MIN_CHUNK_ROWS_GEMV")) {
            let mgv = to_int(get_env_variable("DASLLAMA_MIN_CHUNK_ROWS_GEMV"))
            set_matmul_min_chunk_rows_gemv(mgv)
            print("gemv min chunk rows override: {mgv}\n")
        }
        // A/B fused (region-list, default) vs per-expert expert GEMMs in the grouped MoE prefill
        if (has_env_variable("DASLLAMA_MOE_FUSED")) {
            let mf = to_int(get_env_variable("DASLLAMA_MOE_FUSED"))
            set_moe_fused_expert_gemms(mf != 0)
            print("moe_fused_expert_gemms override: {mf}\n")
        }
        // A/B the grouped-MoE mx4 route: native mx4 batch GEMMs (backend default) vs expand-to-Q8
        if (has_env_variable("DASLLAMA_MX4_NATIVE")) {
            let mn = to_int(get_env_variable("DASLLAMA_MX4_NATIVE"))
            set_mx4_native_batch(mn != 0)
            print("mx4_native_batch override: {mn}\n")
        }
        // A/B the 2-D batch chunk grid (0 = 1-D, 1 = fine grid, 2 = wave-aligned)
        if (has_env_variable("DASLLAMA_BATCH_GRID_2D")) {
            let bg2 = to_int(get_env_variable("DASLLAMA_BATCH_GRID_2D"))
            set_batch_grid_2d(bg2)
            print("batch grid 2d override: {bg2}\n")
        }
        // team dispatch profile over each batched prefill (per-op publish/serve/join costs +
        // caller/solo shares — the starved-shape diagnostic)
        let teamProf = has_env_variable("DASLLAMA_TEAM_PROF")
        // warmup — JIT both paths
        let warm <- build_prompt(8l)
        for (p in range64(8l)) {
            forward(t, s, warm[p], p)
        }
        forward_prefill(t, s, warm, 8l, 0l)

        // DASLLAMA_PREFILL_ONLY=1 skips the per-token (decode-style) arm — the fast A/B loop
        // for side-by-side runs against llama-bench -p N -n 0
        let prefillOnly = has_env_variable("DASLLAMA_PREFILL_ONLY")
        let sizes <- [64l, 256l, 512l]
        for (n in sizes) {
            let prompt <- build_prompt(n)
            var us_old = 1
            if (!prefillOnly) {
                let t0 = ref_time_ticks()
                for (p in range64(n)) {
                    forward(t, s, prompt[p], p)
                }
                us_old = get_time_usec(t0)
            }
            forward_profile_reset()   // per-section table over the LAST batched prefill only
            if (teamProf) {
                set_jobque_team_prof(true)
                reset_jobque_team_prof()
            }
            // DASLLAMA_PREFILL_TRACE=<path> saves a per-lane jobque trace of the N=512 batched
            // prefill (the lane-timeline viewer's input; same rig as decode_prof --trace)
            let doTrace = n == 512l && has_env_variable("DASLLAMA_PREFILL_TRACE")
            if (doTrace) {
                jobque_trace_start(1 << 20)   // prefill windows are ~50x a decode token — bigger ring
                set_trace_tags(true)
            }
            let t1 = ref_time_ticks()
            forward_prefill(t, s, prompt, n, 0l)
            let us_new = get_time_usec(t1)
            if (doTrace) {
                set_trace_tags(false)
                jobque_trace_stop()
                let tpath = get_env_variable("DASLLAMA_PREFILL_TRACE")
                if (jobque_trace_save(tpath)) {
                    print("jobque trace saved: {tpath}\n")
                } else {
                    print("jobque trace: nothing recorded\n")
                }
            }
            if (teamProf) {
                let prof = get_jobque_team_prof()
                let counts = get_jobque_team_prof_counts()
                let react = get_jobque_team_prof_react()
                set_jobque_team_prof(false)
                let callerShare = counts.y > 0 ? 100 * counts.z / counts.y : 0
                let soloShare = counts.x > 0 ? 100 * counts.w / counts.x : 0
                print("team prof N={n}: avg ns/op publish {prof.x} serve {prof.y} join-tail {prof.z} total {prof.w} | ops {counts.x} chunks {counts.y} caller-share {callerShare}% solo {soloShare}% | claims first +{react.x}ns last +{react.y}ns\n")
            }
            let tps_new = double(n) / (double(us_new) * 1.0e-6lf)
            if (prefillOnly) {
                print("N={n}: prefill {us_new}us {tps_new} tok/s\n")
            } else {
                let tps_old = double(n) / (double(us_old) * 1.0e-6lf)
                let speedup = double(us_old) / double(us_new)
                print("N={n}: per-token {us_old}us {tps_old} tok/s  |  prefill {us_new}us {tps_new} tok/s  |  {speedup}x\n")
            }
        }
        forward_profile_report()   // buckets for the N=512 batched prefill (set against GGML_OP_PROFILE pp512)
    }
}
