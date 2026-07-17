options gen2
options persistent_heap   // + explicit deletes: the models free before the process exits
options stack = 131072   // several by-value Sessions per cell (test_batch_decode sizing)

require dastest/testing_boost public
require dasllama/dasllama   // nolint:STYLE029 — the public facade (engine + arch registrations + the metal overrides); off-Apple the static_if halves compile out and only its dasllama_common re-export stays referenced
require dasllama/dasllama_math   // set_wscale_f16 — the wscale matrix axis
require ?das_metal dasllama/dasllama_metal_llama  // nolint:STYLE030 — stats/declines/shutdown rails
require ?das_metal dasllama/dasllama_metal_prefill  // nolint:STYLE030 — prefill stats/declines rails
require ?das_metal metal/das_metal_boost  // nolint:STYLE030 — device probe + metal_live_object_count
require daslib/jobque_boost
require daslib/fio  // nolint:STYLE030 — used only inside the Apple static_if half
require math  // nolint:STYLE030 — used only inside the Apple static_if half
require strings
require _model_tier

// The (weights x wscale x kv-dtype x path) SUPPORT MATRIX for the Metal drivers (format-matrix
// arc W0). Each cell asserts ENGAGE (steps served on the GPU) or DECLINE with the EXPECTED
// first-failing-clause reason from the declines-by-reason tables — a coverage regression (a gate
// declining a config that used to engage) turns into a red cell here instead of a silent CPU
// fallback (the wscale16-prefill inertness went unnoticed for months exactly that way). This
// expectation table IS the support documentation; each later wave flips its cells to ENGAGE as
// coverage widens (W2: s16 prefill; W3: q8_0/tq4 caches; W4: K-quant weights).
//
// Expected matrix TODAY (llama family, portable backend, Llama-3.2-1B):
//   weights   wscale  kv     prefill              decode               batch decode
//   q8        f32     f16    ENGAGE               ENGAGE               ENGAGE
//   q8        f32     f32    ENGAGE               ENGAGE               ENGAGE
//   q8        f32     q8_0   ENGAGE               ENGAGE (W3)          ENGAGE (W3)
//   q8        f32     tq4    ENGAGE               ENGAGE (W3)          ENGAGE (W3)
//   q8        f16     f16    ENGAGE (W2)          ENGAGE               ENGAGE
//   Q4_K_M    f32     f16    ENGAGE (M2.3)        ENGAGE (W4)          ENGAGE (M2.1/W4)
//   Q4_K_M    f32     q8_0   ENGAGE (M2.3)        ENGAGE (W4)          ENGAGE (M2.1/W4)
// (W4: single-stream decode serves K-quant weights via the per-site kq plane GEMVs; M2.1: the
// batched step serves kq at B <= 8 via nrows per-stream plane GEMVs, B >= 9 via the kq mul_mm
// twins (%64 shape gates); M2.3: prefill serves kq via the mul_mm twins, same gates.)
// The dim ladder (M2.6) rides two extra models when present: an 8B (dim 4096, q8) engages at
// every B — past B=4 the GEMM's fixed K-tiles take any width — and a 70B-class kq model
// (dim 8192) engages all three paths with the batched add+rms running as the unfused pair.
// The MetalMode knob (M1) rides the same matrix: off = overrides deselected, when_available =
// engage/fallback per the cells above, required = a CAPABILITY decline panics (policy declines
// — the prefill npos window — stay quiet CPU fallbacks). Batch kq: B <= 8 = per-stream GEMVs,
// B >= 9 = the kq mul_mm twins (%64 shapes gated).
// plus the prefill-only window clause: npos < min_npos declines npos_window. A continuation
// chunk (start_pos > 0) SERVES since M2.5 (existing rows gather into the K/V panels, the
// causal diagonal shifts).

def private counting_prompt(start, last : int) : string {
    return build_string() $(var w) {
        for (i in range(start, last + 1)) {
            w |> write(i)
            w |> write(i < last ? ", " : ",")
        }
    }
}

def private greedy_top(lg : array<float>; vocab : int64) : int64 {
    var best = 0l
    for (v in range64(1l, vocab)) {
        if (lg[v] > lg[best]) {
            best = v
        }
    }
    return best
}

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

[unused_argument(t, tr, kdt, tag, exp_pf, exp_dec, exp_bat)]   // off-Apple the gated body is empty
def private matrix_cell(t : T?; tr : Model; kdt : KVDtype; tag : string;
                        exp_pf, exp_dec, exp_bat : string) {
    // one prefill + two decode steps + one B=2 batch step under the metal overrides; each path
    // asserts its cell: "" = ENGAGE (served > 0), else DECLINE with that reason recorded
    static_if (typeinfo builtin_module_exists(das_metal)) {
        let c = tr.config
        let vocab = c.vocab_size
        let pf0 = metal_prefill_stats().prefills
        let de0 = metal_decode_stats().decodes
        let ba0 = metal_batch_decode_stats().steps
        var pfd0 <- metal_prefill_declines()
        var ded0 <- metal_decode_declines()
        var bad0 <- metal_batch_decode_declines()
        with_job_que() {
            setup_dasllama_jobque()
            var sa = make_run_state(c, kdt, kdt)
            var sb = make_run_state(c, kdt, kdt)
            let pa <- encode(tr, counting_prompt(1, 40), true, false)
            let pb <- encode(tr, counting_prompt(5, 44), true, false)
            t |> success(select_prefill_override("metal"), "{tag}: prefill override registered")
            t |> success(select_decode_override("metal"), "{tag}: decode override registered")
            set_metal_prefill_min_npos(1l)
            eval_(tr, sa, pa)   // the prefill cell
            eval_(tr, sb, pb)
            var cura = greedy_top(sa.logits, vocab)
            let curb = greedy_top(sb.logits, vocab)
            for (_i in range(2)) {   // the single-stream decode cell
                var one <- [cura]
                eval_(tr, sa, one)
                delete one
                cura = greedy_top(sa.logits, vocab)
            }
            var ws <- create_batch_workspace_(tr)   // the batched decode cell
            var sess : array<Session?>
            unsafe {
                sess |> push(addr(sa))
                sess |> push(addr(sb))
            }
            var cur <- [cura, curb]
            eval_batch_(tr, ws, sess, cur)
            select_decode_override("")
            select_prefill_override("")
            set_metal_prefill_min_npos(64l)
            sess |> clear()   // drop the borrowed handles: delete of array<T?> cascades to the pointees
            unsafe {
                delete ws
                delete sess
            }
            delete cur
            delete sa
            delete sb
        }
        var pfd <- metal_prefill_declines()
        var ded <- metal_decode_declines()
        var bad <- metal_batch_decode_declines()
        check_cell(t, "{tag} prefill", exp_pf, metal_prefill_stats().prefills - pf0, pfd0, pfd)
        check_cell(t, "{tag} decode", exp_dec, metal_decode_stats().decodes - de0, ded0, ded)
        check_cell(t, "{tag} batch", exp_bat, metal_batch_decode_stats().steps - ba0, bad0, bad)
        delete pfd0
        delete ded0
        delete bad0
        delete pfd
        delete ded
        delete bad
    }
}

[unused_argument(t, tr, nb, tag)]   // off-Apple the gated body is empty
def private dim_ladder_batch_cell(t : T?; tr : Model; nb : int; tag : string) {
    // one B=nb batched step under the metal decode override — the M2.6 coverage past B=4
    // (GPU prefill fills the sessions so the wide models stay cheap)
    static_if (typeinfo builtin_module_exists(das_metal)) {
        let c = tr.config
        let vocab = c.vocab_size
        var served = 0l
        with_job_que() {
            setup_dasllama_jobque()
            var ss : array<Session>
            ss |> reserve(nb)
            var sess : array<Session?>
            var cur : array<int64>
            select_prefill_override("metal")
            set_metal_prefill_min_npos(1l)
            for (i in range(nb)) {
                ss |> emplace <| make_run_state(c, KVDtype.f16, KVDtype.f16)
                let p <- encode(tr, counting_prompt(1 + i, 14 + i), true, false)
                eval_(tr, ss[i], p)
                unsafe {
                    sess |> push(addr(ss[i]))
                }
                cur |> push(greedy_top(ss[i].logits, vocab))
            }
            set_metal_prefill_min_npos(64l)
            select_prefill_override("")
            var ws <- create_batch_workspace_(tr)
            let ba0 = metal_batch_decode_stats().steps
            select_decode_override("metal")
            eval_batch_(tr, ws, sess, cur)
            select_decode_override("")
            served = metal_batch_decode_stats().steps - ba0
            sess |> clear()   // borrowed handles — delete of array<T?> cascades to the pointees
            unsafe {
                delete ws
                delete sess
            }
            delete cur
            delete ss
        }
        t |> success(served > 0l, "{tag}: B={nb} batched step ENGAGES on the GPU ({served} served)")
    }
}

// "" = the path must ENGAGE (served > 0); otherwise it must sit out (served == 0) with `want`
// recorded as a first-failing clause in the window
def private check_cell(t : T?; cell : string; want : string; served : int64;
                       before, after : table<string; int64>) {
    if (want == "") {
        t |> success(served > 0l, "{cell}: ENGAGES on the GPU ({served} served)")
    } else {
        let hits = (after?[want] ?? 0l) - (before?[want] ?? 0l)
        t |> success(served == 0l, "{cell}: stays on the CPU (expected decline {want})")
        t |> success(hits > 0l, "{cell}: declines with reason '{want}' ({hits} recorded)")
    }
}

[test]
def test_metal_support_matrix(t : T?) {
    t |> run("Metal support matrix: weights x wscale x kv x path engage/decline (Llama-3.2-1B)") <| @(t : T?) {
        if (!family_on(t, "llama")) return
        static_if (typeinfo builtin_module_exists(das_metal)) {
            if (!jit_enabled()) {
                t |> skip("interpreted (dasLLAMA model tests are JIT-only)")
                return
            }
            let path = path_join(models_dir(), "Llama-3.2-1B-Instruct-Q8_0.gguf")
            if (!model_available(t, path)) {
                return
            }
            var dev = metal_create_system_default_device()
            if (dev == null) {
                feint("no Metal device; support matrix skipped\n")
                return
            }
            metal_release(dev)
            let prior_wscale = get_wscale_f16()             // process-wide knobs — restored below
            let prior_backend = active_kernel_backend()
            pin_kernel_backend("portable")   // row-major weights (the resident drivers' donor contract)

            // ===== q8 weights, f32 scale plane =====
            set_wscale_f16(false)
            var tr <- load_model(path, QuantMode.q8)
            tr.config.seq_len = min(tr.config.seq_len, 512l)
            if (arm_on(t, "cells-q8")) {
                matrix_cell(t, tr, KVDtype.f16, "q8/s32/f16", "", "", "")
                matrix_cell(t, tr, KVDtype.f32, "q8/s32/f32", "", "", "")
                matrix_cell(t, tr, KVDtype.q8_0, "q8/s32/q8_0", "", "", "")
                matrix_cell(t, tr, KVDtype.tq4, "q8/s32/tq4", "", "", "")
            }

            // the prefill window clause rides the same model: npos below the default min_npos
            // declines npos_window; a continuation chunk (start_pos > 0) SERVES since M2.5
            if (arm_on(t, "window")) {
                var pfd0 <- metal_prefill_declines()
                let pf0 = metal_prefill_stats().prefills
                with_job_que() {
                    setup_dasllama_jobque()
                    var s = make_run_state(tr.config, KVDtype.f16, KVDtype.f16)
                    let short_p <- encode(tr, counting_prompt(1, 12), true, false)
                    let cont_p <- encode(tr, counting_prompt(13, 26), false, false)
                    select_prefill_override("metal")   // min_npos stays at the 64 default
                    eval_(tr, s, short_p)              // npos < 64 -> npos_window
                    set_metal_prefill_min_npos(1l)
                    eval_(tr, s, cont_p)               // start_pos = n_past > 0 -> SERVES (M2.5)
                    set_metal_prefill_min_npos(64l)
                    select_prefill_override("")
                    delete s
                }
                var pfd <- metal_prefill_declines()
                t |> success((pfd?["npos_window"] ?? 0l) > (pfd0?["npos_window"] ?? 0l),
                    "short prompt below min_npos declines npos_window")
                t |> success(metal_prefill_stats().prefills - pf0 > 0l,
                    "continuation prefill (start_pos > 0) ENGAGES on the GPU (M2.5)")
                t |> success((pfd?["start_pos"] ?? 0l) == (pfd0?["start_pos"] ?? 0l),
                    "the retired start_pos clause records nothing")
                delete pfd0
                delete pfd
            }

            // ===== q8 weights, f16 scale plane (the load DEFAULT since wscale16) =====
            if (arm_on(t, "cells-s16")) {
                set_wscale_f16(true)
                var tr16 <- load_model(path, QuantMode.q8)
                tr16.config.seq_len = min(tr16.config.seq_len, 512l)
                matrix_cell(t, tr16, KVDtype.f16, "q8/s16/f16", "", "", "")
                set_wscale_f16(false)
                delete tr16
            }

            // ===== the MetalMode knob (M1) on a fully-servable q8 config =====
            if (arm_on(t, "mode")) {
                let de_off = metal_decode_stats().decodes
                set_metal_mode(MetalMode.off)
                t |> success(active_decode_override() == "", "mode off: overrides deselected")
                with_job_que() {
                    setup_dasllama_jobque()
                    var s = make_run_state(tr.config, KVDtype.f16, KVDtype.f16)
                    let pr <- encode(tr, counting_prompt(1, 12), true, false)
                    eval_(tr, s, pr)
                    var one <- [greedy_top(s.logits, tr.config.vocab_size)]
                    eval_(tr, s, one)
                    delete one
                    delete s
                }
                t |> success(metal_decode_stats().decodes == de_off, "mode off: nothing served on the GPU")
                set_metal_mode(MetalMode.when_available)
                t |> success(active_decode_override() == "metal", "mode when_available: metal overrides selected")
                let de_avail = metal_decode_stats().decodes
                with_job_que() {
                    setup_dasllama_jobque()
                    var s = make_run_state(tr.config, KVDtype.f16, KVDtype.f16)
                    let pr <- encode(tr, counting_prompt(1, 12), true, false)
                    eval_(tr, s, pr)
                    var one <- [greedy_top(s.logits, tr.config.vocab_size)]
                    eval_(tr, s, one)
                    delete one
                    delete s
                }
                t |> success(metal_decode_stats().decodes > de_avail, "mode when_available: decode serves on the GPU")
                // required on a servable config: decode serves, and the SHORT prompt's
                // npos_window POLICY decline stays a quiet CPU prefill — no panic
                set_metal_mode(MetalMode.required)
                let de_req = metal_decode_stats().decodes
                var pfd_r0 <- metal_prefill_declines()
                with_job_que() {
                    setup_dasllama_jobque()
                    var s = make_run_state(tr.config, KVDtype.f16, KVDtype.f16)
                    let pr <- encode(tr, counting_prompt(1, 12), true, false)
                    eval_(tr, s, pr)
                    var one <- [greedy_top(s.logits, tr.config.vocab_size)]
                    eval_(tr, s, one)
                    delete one
                    delete s
                }
                var pfd_r1 <- metal_prefill_declines()
                t |> success(metal_decode_stats().decodes > de_req, "mode required: decode serves on the GPU")
                t |> success((pfd_r1?["npos_window"] ?? 0l) > (pfd_r0?["npos_window"] ?? 0l),
                    "mode required: the npos_window POLICY decline stays a quiet CPU fallback")
                set_metal_mode(MetalMode.off)
                delete pfd_r0
                delete pfd_r1
            }

            // ===== native K-quant weights (Q4_K_M) =====
            // All three paths ENGAGE (W4 decode, M2.1 batch B<=8 + W4 mul_mm twins past that,
            // M2.3 prefill mul_mm twins) — Q4_K_M mixes k4 and k6 per tensor, incl. the tied
            // Q6_K classifier on the GPU-logits paths
            let kpath = path_join(models_dir(), "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
            if (!arm_on(t, "kq")) {
                // filtered — the skip is already registered
            } elif (stat(kpath).is_valid) {
                var trk <- load_model(kpath, QuantMode.q8)
                trk.config.seq_len = min(trk.config.seq_len, 512l)
                t |> success(trk.kquant_native, "Q4_K_M under a q8-mode load keeps native K-quant planes")
                matrix_cell(t, trk, KVDtype.f16, "k4/s32/f16", "", "", "")
                matrix_cell(t, trk, KVDtype.q8_0, "k4/s32/q8_0", "", "", "")
                // the output half: GPU kq decode tokens must be the CPU truth
                let kprompt <- encode(trk, counting_prompt(1, 30), true, false)
                var cpu_ids : array<int64>
                var gpu_ids : array<int64>
                with_job_que() {
                    setup_dasllama_jobque()
                    var s = make_run_state(trk.config, KVDtype.f16, KVDtype.f16)
                    let ids <- generate(trk, s, kprompt, length(kprompt) + 8)
                    cpu_ids := ids
                    delete s
                    select_decode_override("metal")
                    select_prefill_override("metal")
                    var s2 = make_run_state(trk.config, KVDtype.f16, KVDtype.f16)
                    let ids2 <- generate(trk, s2, kprompt, length(kprompt) + 8)
                    gpu_ids := ids2
                    delete s2
                    select_decode_override("")
                    select_prefill_override("")
                }
                t |> success(same(cpu_ids, gpu_ids),
                    "K-quant continuation with metal overrides ON matches the CPU truth (guard verified)")
                delete cpu_ids
                delete gpu_ids

                // ===== MetalMode.required on the kq model =====
                // every implemented path serves (prefill mul_mm, single decode, batched steps);
                // the remaining capability gap — a continuation prefill (start_pos > 0, M2.5) —
                // fails hard instead of falling back
                with_job_que() {
                    setup_dasllama_jobque()
                    var sa = make_run_state(trk.config, KVDtype.f16, KVDtype.f16)
                    var sb = make_run_state(trk.config, KVDtype.f16, KVDtype.f16)
                    set_metal_mode(MetalMode.required)
                    let pf0 = metal_prefill_stats().prefills
                    set_metal_prefill_min_npos(1l)
                    eval_(trk, sa, kprompt)   // required: the kq prefill must serve on the GPU
                    eval_(trk, sb, kprompt)
                    set_metal_prefill_min_npos(64l)
                    t |> success(metal_prefill_stats().prefills - pf0 > 0l,
                        "required: kq prefill serves on the GPU (M2.3 mul_mm twins)")
                    let de0 = metal_decode_stats().decodes
                    var one <- [greedy_top(sa.logits, trk.config.vocab_size)]
                    eval_(trk, sa, one)
                    t |> success(metal_decode_stats().decodes - de0 > 0l,
                        "required: kq single-stream decode serves on the GPU")
                    var ws <- create_batch_workspace_(trk)
                    var sess : array<Session?>
                    unsafe {
                        sess |> push(addr(sa))
                        sess |> push(addr(sb))
                    }
                    var cur <- [greedy_top(sa.logits, trk.config.vocab_size), greedy_top(sb.logits, trk.config.vocab_size)]
                    let ba0 = metal_batch_decode_stats().steps
                    eval_batch_(trk, ws, sess, cur)
                    t |> success(metal_batch_decode_stats().steps - ba0 > 0l,
                        "required: kq batched step serves on the GPU (M2.1 per-stream form)")
                    // a continuation chunk under required SERVES since M2.5...
                    var sc = make_run_state(trk.config, KVDtype.f16, KVDtype.f16)
                    set_metal_prefill_min_npos(1l)
                    let pfc0 = metal_prefill_stats().prefills
                    eval_(trk, sc, kprompt)      // chunk 1: fresh-start, GPU-serves
                    eval_(trk, sc, kprompt)      // chunk 2: start_pos = n_past > 0 — GPU-serves too
                    t |> success(metal_prefill_stats().prefills - pfc0 == 2l,
                        "required: continuation prefill (start_pos > 0) serves on the GPU (M2.5)")
                    // ... and a CAPABILITY decline still panics — pin the legacy quantized GEMM,
                    // which has no kq twins (the kquant_native clause)
                    set_metal_prefill_mulmm_legacy(true)
                    var sd = make_run_state(trk.config, KVDtype.f16, KVDtype.f16)
                    var pf2_panicked = false
                    try {
                        eval_(trk, sd, kprompt)
                    } recover {
                        pf2_panicked = true
                    }
                    set_metal_prefill_mulmm_legacy(false)
                    set_metal_prefill_min_npos(64l)
                    t |> success(pf2_panicked, "required: a kq load on the legacy GEMM (kquant_native capability decline) panics instead of CPU fallback")
                    delete sd
                    set_metal_mode(MetalMode.off)
                    sess |> clear()   // borrowed handles — delete of array<T?> cascades to the pointees
                    unsafe {
                        delete ws
                        delete sess
                    }
                    delete cur
                    delete one
                    delete sa
                    delete sb
                    delete sc
                }
                delete trk
            } else {
                feint("Llama-3.2-1B Q4_K_M not present (llama-quantize the Q8 source); K-quant cells skipped\n")
            }

            // ===== the dim ladder (M2.6): batch dim caps retired =====
            // past ADD_RMS_MAX_DIM the batched add+rms runs as the unfused pair, and past the
            // mv8 probe's dim 3072 the B > 4 GEMM tier serves any width — an 8B (dim 4096)
            // batches at every B and a 70B-class kq model (dim 8192) serves all three paths.
            // Both are large-tier: they run only under DASLLAMA_PARITY_FULL (the pre-commit gate).
            if (arm_on(t, "dim8b")) {
                let path8 = path_join(models_dir(), "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")
                if (model_available(t, path8)) {
                    var tr8 <- load_model(path8, QuantMode.q8)
                    tr8.config.seq_len = min(tr8.config.seq_len, 512l)
                    matrix_cell(t, tr8, KVDtype.f16, "8B-q8/s32/f16", "", "", "")
                    dim_ladder_batch_cell(t, tr8, 8, "8B dim4096")   // the q8 GEMM tier past B=4
                    delete tr8
                }
            }
            if (arm_on(t, "dim70b")) {
                let path70 = path_join(models_dir(), "Llama-3.3-70B-Instruct-Q4_K_M.gguf")
                if (model_available(t, path70)) {
                    var tr70 <- load_model(path70, QuantMode.q8)
                    tr70.config.seq_len = min(tr70.config.seq_len, 128l)
                    matrix_cell(t, tr70, KVDtype.f16, "70B-kq/s32/f16", "", "", "")
                    dim_ladder_batch_cell(t, tr70, 12, "70B dim8192")   // the kq mul_mm twin at dim 8192
                    delete tr70
                }
            }

            // the decline tables surface in the human report too, and the leak gate holds
            metal_decode_stage_report()
            metal_decode_shutdown()
            metal_prefill_shutdown()
            t |> equal(metal_live_object_count(), 0l)
            clear_kernel_backend_pin()
            select_kernel_backend(prior_backend)
            set_wscale_f16(prior_wscale)
            delete tr
        } else {
            feint("das_metal is not built on this platform; nothing to assert\n")
        }
    }
}

[unused_argument(t, path, family, tag, expect_needs, want, kv2)]   // off-Apple the gated body is empty
def private family_cells(t : T?; path, family, tag : string; expect_needs : MetalNeed; want : string;
                         kv2 : KVDtype = KVDtype.f16) {
    // one family's row: pin the metal_needs derivation bits, then the three path cells — each
    // asserts ENGAGE ("") or DECLINE with the named need_* reason. Wave stages flip these.
    // An ENGAGE row ("" cells) ALSO proves token parity: a metal-overrides generate must equal
    // the CPU truth. kv2 != f16 adds a second cell row on that KV codec (its rope-store twin).
    static_if (typeinfo builtin_module_exists(das_metal)) {
        if (!family_on(t, family) || !arm_on(t, "fam-{family}") || !model_available(t, path)) {
            return
        }
        var tr <- load_model(path, QuantMode.q8)
        tr.config.seq_len = min(tr.config.seq_len, 512l)
        let derived = metal_needs(tr)
        t |> success((uint(derived) & uint(expect_needs)) == uint(expect_needs),
            "{tag}: metal_needs derives {expect_needs} (got {derived})")
        matrix_cell(t, tr, KVDtype.f16, tag, want, want, want)
        if (kv2 != KVDtype.f16) {
            matrix_cell(t, tr, kv2, "{tag}+{kv2}", want, want, want)
        }
        if (want == "") {
            let prompt <- encode(tr, counting_prompt(1, 30), true, false)
            var cpu_ids : array<int64>
            var gpu_ids : array<int64>
            with_job_que() {
                setup_dasllama_jobque()
                var s = make_run_state(tr.config, KVDtype.f16, KVDtype.f16)
                let ids <- generate(tr, s, prompt, length(prompt) + 8)
                cpu_ids := ids
                delete s
                select_decode_override("metal")
                select_prefill_override("metal")
                set_metal_prefill_min_npos(1l)
                var s2 = make_run_state(tr.config, KVDtype.f16, KVDtype.f16)
                let ids2 <- generate(tr, s2, prompt, length(prompt) + 8)
                gpu_ids := ids2
                delete s2
                set_metal_prefill_min_npos(64l)
                select_decode_override("")
                select_prefill_override("")
            }
            t |> success(same(cpu_ids, gpu_ids),
                "{tag}: metal-overrides generate matches the CPU truth token-for-token")
            delete cpu_ids
            delete gpu_ids
        }
        delete tr
    }
}

// The family axis (family waves A): each non-llama dense family loads its small carrier and
// asserts per-path ENGAGE/DECLINE with the exact need_* reason — the descriptor refactor's
// living expectation table. mistral3 (empty config = the llama shape) has no <= 35B fixture;
// its coverage is the derivation pins on the families above plus the llama cells.
[test]
def test_metal_family_matrix(t : T?) {
    t |> run("Metal family matrix: per-family needs derivation + path cells") <| @(t : T?) {
        static_if (typeinfo builtin_module_exists(das_metal)) {
            if (!jit_enabled()) {
                t |> skip("interpreted (dasLLAMA model tests are JIT-only)")
                return
            }
            var dev = metal_create_system_default_device()
            if (dev == null) {
                feint("no Metal device; family matrix skipped\n")
                return
            }
            metal_release(dev)
            let prior_wscale = get_wscale_f16()
            let prior_backend = active_kernel_backend()
            pin_kernel_backend("portable")
            set_wscale_f16(false)
            family_cells(t, path_join(models_dir(), "Qwen3-0.6B-Q8_0.gguf"), "qwen3",
                "qwen3-0.6b/q8/f16", MetalNeed.neox | MetalNeed.qk_norm | MetalNeed.qd_neq_dim,
                "need_qk_norm")
            family_cells(t, path_join(models_dir(), "Qwen2.5-0.5B-Instruct-Q8_0.gguf"), "qwen2",
                "qwen2.5-0.5b/q8/f16", MetalNeed.neox | MetalNeed.qkv_bias,
                "need_qkv_bias")
            // phi3 = NEOX only — ENGAGE since the wave A rope twins; the Q4 row also proves the
            // q8_0-KV rope-store's NEOX block loop, the Q8/s16 row the fused qkv_rs NEOX mapping
            family_cells(t, path_join(models_dir(), "Phi-3.5-mini-instruct-Q4_K_M.gguf"), "phi3",
                "phi3.5-mini/k4/f16", MetalNeed.neox,
                "", KVDtype.q8_0)
            set_wscale_f16(true)
            family_cells(t, path_join(models_dir(), "Phi-3.5-mini-instruct-Q8_0.gguf"), "phi3",
                "phi3.5-mini/q8/s16", MetalNeed.neox,
                "")
            set_wscale_f16(false)
            metal_decode_shutdown()
            metal_prefill_shutdown()
            t |> equal(metal_live_object_count(), 0l)
            clear_kernel_backend_pin()
            select_kernel_backend(prior_backend)
            set_wscale_f16(prior_wscale)
        } else {
            feint("das_metal is not built on this platform; nothing to assert\n")
        }
    }
}
