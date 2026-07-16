options gen2
options persistent_heap   // + explicit deletes: the model frees before the process exits
options stack = 131072   // by-value Sessions in the kq + quantized-KV arms (test_batch_decode sizing)

require dastest/testing_boost public
require dasllama/dasllama   // the public facade — pulls the engine + arch registrations + (on Apple) the metal prefill override
require dasllama/dasllama_math   // set_wscale_f16 — the resident path reads the f32 scale plane
require ?das_metal dasllama/dasllama_metal_prefill  // nolint:STYLE030 — min-npos setter + stats + the shutdown/leak rail
require ?das_metal dasllama/dasllama_metal_llama  // nolint:STYLE030 — decode shutdown (the kq/quantized-KV arms touch the decode driver)
require ?das_metal metal/das_metal_boost  // nolint:STYLE030 — metal_live_object_count for the leak gate
require daslib/jobque_boost
require daslib/fio  // nolint:STYLE030 — used only inside the Apple static_if half; off-Apple lint sees the branch compiled out
require strings
require _model_tier

// End-to-end gate for the dasMetal whole-prefill override (Phase 6): greedy continuation of a
// confident counting prompt, GPU-resident prefill vs the CPU control ON THE SAME BINARY — decode
// is the identical CPU path in both runs, so any prefill-induced logit shift beyond the shared
// tolerance envelope surfaces as a token flip. The counting prompt is the kernel-robust fixture
// family test_parity.das established (high-entropy prompts flip on genuine near-ties, which are
// not bugs). Skips cleanly without the model, off-Apple, or with no Metal device.

def private same(a, b : array<int64>) : bool {
    if (length(a) != length(b)) {
        return false
    }
    for (i in range(length(a))) {
        if (a[i] != b[i]) {
            return false
        }
    }
    return true
}

// "1, 2, …, 120," — long enough to engage the resident path with margin (~300 llama-3 BPE tokens)
def private counting_prompt() : string {
    return build_string() $(var w) {
        for (i in range(1, 121)) {
            w |> write(i)
            w |> write(i < 120 ? ", " : ",")
        }
    }
}

def private continuation(t : Model; prompt : array<int64>; n_gen : int) : array<int64> {
    var got : array<int64>
    with_job_que() {
        setup_dasllama_jobque()
        var s = make_run_state(t.config)
        let steps = length(prompt) + n_gen
        let ids <- generate(t, s, prompt, steps)
        let echo = length(prompt) - 1
        got <- [for (i in range(n_gen)); ids[echo + i]; where echo + i < length(ids)]
        delete s
    }
    return <- got
}

[test]
def test_metal_prefill_parity(t : T?) {
    t |> run("GPU-resident prefill == CPU control, 40-token greedy continuation (Llama-3.2-1B Q8)") <| @(t : T?) {
        static_if (typeinfo builtin_module_exists(das_metal)) {
            if (!jit_enabled()) {
                t |> skip("interpreted (dasLLAMA model tests are JIT-only)")
                return
            }
            let path = path_join(models_dir(), "Llama-3.2-1B-Instruct-Q8_0.gguf")
            if (!model_available(t, path)) {
                return
            }
            let prior_wscale = get_wscale_f16()             // process-wide knobs — restored below
            let prior_backend = active_kernel_backend()     // (both exit paths run through the restore)
            set_wscale_f16(false)       // the resident path reads the f32 scale plane (declines on s16)
            pin_kernel_backend("portable")  // ...and ROW-MAJOR weights — a repack backend's layout is unreadable (5c donor contract)
            var tr <- load_model(path, QuantMode.q8)
            if (tr.config.seq_len > 2048l) {
                tr.config.seq_len = 2048l
            }
            let prompt <- encode(tr, counting_prompt(), true, false)
            if (arm_on(t, "base")) {
                let cpu <- continuation(tr, prompt, 40)
                t |> equal(length(cpu), 40)
                t |> success(select_prefill_override("metal"), "override registered")
                set_metal_prefill_min_npos(1l)
                let before = metal_prefill_stats().prefills
                let gpu <- continuation(tr, prompt, 40)         // default = lcpp mul_mm (f32 activations)
                let served = metal_prefill_stats().prefills - before
                set_metal_prefill_mulmm_legacy(true)            // the quantized path — bit-identical to CPU
                let before2 = metal_prefill_stats().prefills
                let gpu_legacy <- continuation(tr, prompt, 40)
                let served2 = metal_prefill_stats().prefills - before2
                set_metal_prefill_mulmm_legacy(false)
                select_prefill_override("")
                set_metal_prefill_min_npos(64l)
                clear_kernel_backend_pin()
                select_kernel_backend(prior_backend)
                set_wscale_f16(prior_wscale)
                if (served == 0l || served2 == 0l) {
                    metal_prefill_shutdown()
                    delete tr
                    feint("metal prefill declined (no Metal device?); GPU parity skipped\n")
                    return
                }
                t |> success(same(cpu, gpu_legacy), "40-token continuation matches the CPU control (legacy quantized GEMM)")
                t |> success(same(cpu, gpu), "40-token continuation matches the CPU control (lcpp mul_mm, f32 activations)")
            }

            // s16 arm (W2): the DEFAULT-load scale plane — blobs read raw f16 scales, the
            // classifier GEMV takes its halfword twin; a fresh load owns fresh weight regions.
            // The restores above un-pinned the backend — re-pin row-major (the donor contract)
            if (arm_on(t, "s16")) {
                set_wscale_f16(true)
                pin_kernel_backend("portable")
                var tr16 <- load_model(path, QuantMode.q8)
                if (tr16.config.seq_len > 2048l) {
                    tr16.config.seq_len = 2048l
                }
                let cpu16 <- continuation(tr16, prompt, 40)
                t |> success(select_prefill_override("metal"), "override registered (s16)")
                set_metal_prefill_min_npos(1l)
                let before16 = metal_prefill_stats().prefills
                let gpu16 <- continuation(tr16, prompt, 40)
                t |> success(metal_prefill_stats().prefills - before16 > 0l, "wscale16 prefill served on the GPU")
                select_prefill_override("")
                set_metal_prefill_min_npos(64l)
                clear_kernel_backend_pin()
                select_kernel_backend(prior_backend)
                set_wscale_f16(prior_wscale)
                t |> success(same(cpu16, gpu16), "40-token wscale16 continuation matches the CPU control (s16 mul_mm + s16 classifier)")
                delete tr16
            }

            // kq arm (M2.3): K-quant prefill — the kq mul_mm twins (mixed k4/k6 tensors per
            // layer) + the kq classifier GEMV. The CPU kq prefill is the control; the envelope
            // is the kq one (f32-x GPU vs Q8_K CPU activations), tokens exact on the fixture.
            let kpath = path_join(models_dir(), "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
            if (!arm_on(t, "kq")) {
                // filtered — the skip is already registered
            } elif (stat(kpath).is_valid) {
                set_wscale_f16(false)
                pin_kernel_backend("portable")
                var trk <- load_model(kpath, QuantMode.q8)
                if (trk.config.seq_len > 2048l) {
                    trk.config.seq_len = 2048l
                }
                let cpuk <- continuation(trk, prompt, 40)
                t |> success(select_prefill_override("metal"), "override registered (kq)")
                set_metal_prefill_min_npos(1l)
                let beforek = metal_prefill_stats().prefills
                let gpuk <- continuation(trk, prompt, 40)
                t |> success(metal_prefill_stats().prefills - beforek > 0l, "kq prefill served on the GPU")
                select_prefill_override("")
                set_metal_prefill_min_npos(64l)
                clear_kernel_backend_pin()
                select_kernel_backend(prior_backend)
                set_wscale_f16(prior_wscale)
                t |> success(same(cpuk, gpuk), "40-token kq continuation matches the CPU control (kq mul_mm + kq classifier)")
                delete trk
            } else {
                feint("Llama-3.2-1B Q4_K_M not present (llama-quantize the Q8 source); kq prefill arm skipped\n")
            }

            // dim-4096 arm (M2.6): past the fused add+rms+quant slab (dim <= 3072) the prefill
            // driver runs the unfused dispatch set — this is that rail's first real-model
            // parity. The 8B is large-tier: it runs only under DASLLAMA_PARITY_FULL.
            if (arm_on(t, "dim")) {
                let path8 = path_join(models_dir(), "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
                if (model_available(t, path8)) {
                    set_wscale_f16(false)
                    pin_kernel_backend("portable")
                    var tr8 <- load_model(path8, QuantMode.q8)
                    if (tr8.config.seq_len > 2048l) {
                        tr8.config.seq_len = 2048l
                    }
                    let cpu8 <- continuation(tr8, prompt, 40)
                    t |> success(select_prefill_override("metal"), "override registered (dim4096)")
                    set_metal_prefill_min_npos(1l)
                    let before8 = metal_prefill_stats().prefills
                    let gpu8 <- continuation(tr8, prompt, 40)
                    t |> success(metal_prefill_stats().prefills - before8 > 0l, "dim-4096 prefill served on the GPU")
                    select_prefill_override("")
                    set_metal_prefill_min_npos(64l)
                    clear_kernel_backend_pin()
                    select_kernel_backend(prior_backend)
                    set_wscale_f16(prior_wscale)
                    t |> success(same(cpu8, gpu8), "40-token dim-4096 continuation matches the CPU control (unfused prefill set)")
                    delete tr8
                }
            }

            // GPU-prefill x quantized-KV arm: the prefill KV writeback into q8_0/tq4 SESSION
            // codecs — this combo engaged since W3 but its output was never explicitly verified
            // (the decode parity arms use CPU prefill). GPU prefill + GPU decode vs the all-CPU
            // control on the same codec, token-exact.
            if (arm_on(t, "qkv")) {
                set_wscale_f16(false)
                pin_kernel_backend("portable")
                var trq <- load_model(path, QuantMode.q8)
                if (trq.config.seq_len > 2048l) {
                    trq.config.seq_len = 2048l
                }
                for (kdt in [KVDtype.q8_0, KVDtype.tq4]) {
                    var cpuq : array<int64>
                    var gpuq : array<int64>
                    with_job_que() {
                        setup_dasllama_jobque()
                        var sa = make_run_state(trq.config, kdt, kdt)
                        let ids <- generate(trq, sa, prompt, length(prompt) + 24)
                        cpuq := ids
                        delete sa
                        select_prefill_override("metal")
                        select_decode_override("metal")
                        set_metal_prefill_min_npos(1l)
                        var sb = make_run_state(trq.config, kdt, kdt)
                        let ids2 <- generate(trq, sb, prompt, length(prompt) + 24)
                        gpuq := ids2
                        delete sb
                        select_prefill_override("")
                        select_decode_override("")
                        set_metal_prefill_min_npos(64l)
                    }
                    t |> success(same(cpuq, gpuq), "GPU prefill x {kdt} KV: 24-token continuation matches the all-CPU control")
                    delete cpuq
                    delete gpuq
                }
                clear_kernel_backend_pin()
                select_kernel_backend(prior_backend)
                set_wscale_f16(prior_wscale)
                delete trq
            }

            metal_prefill_shutdown()
            metal_decode_shutdown()   // the kq uploads + the quantized-KV arm's decode steps live in the decode driver's caches
            t |> equal(metal_live_object_count(), 0l)
            delete tr
        } else {
            feint("das_metal is not built on this platform; nothing to compare\n")
        }
    }
}
