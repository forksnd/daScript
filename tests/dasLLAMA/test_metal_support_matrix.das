options gen2
options persistent_heap   // + explicit deletes: the models free before the process exits
options stack = 131072   // several by-value Sessions per cell (test_batch_decode sizing)

require dastest/testing_boost public
require dasllama/dasllama   // the public facade — engine + arch registrations + (on Apple) the metal overrides
require dasllama/dasllama_math   // set_wscale_f16 — the wscale matrix axis
require ?das_metal dasllama/dasllama_metal_decode  // nolint:STYLE030 — stats/declines/shutdown rails
require ?das_metal dasllama/dasllama_metal_prefill  // nolint:STYLE030 — prefill stats/declines rails
require ?das_metal metal/das_metal_boost  // nolint:STYLE030 — device probe + metal_live_object_count
require daslib/jobque_boost
require daslib/fio  // nolint:STYLE030 — used only inside the Apple static_if half
require math
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
//   q8        f32     q8_0   ENGAGE               decline kv_dtype     decline kv_dtype
//   q8        f32     tq4    ENGAGE               decline kv_dtype     decline kv_dtype
//   q8        f16     f16    decline wscale_f16   ENGAGE               ENGAGE
//   Q4_K_M    f32     f16    decline kquant_native x3 (the silent-misread guard — W4 lifts it)
// plus the prefill-only window clauses: npos < min_npos declines npos_window, a continuation
// chunk (start_pos > 0) declines start_pos.

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
            matrix_cell(t, tr, KVDtype.f16, "q8/s32/f16", "", "", "")
            matrix_cell(t, tr, KVDtype.f32, "q8/s32/f32", "", "", "")
            matrix_cell(t, tr, KVDtype.q8_0, "q8/s32/q8_0", "", "kv_dtype", "kv_dtype")
            matrix_cell(t, tr, KVDtype.tq4, "q8/s32/tq4", "", "kv_dtype", "kv_dtype")

            // the prefill window clauses ride the same model: npos below the default min_npos
            // declines npos_window; a continuation chunk (start_pos > 0) declines start_pos
            {
                var pfd0 <- metal_prefill_declines()
                with_job_que() {
                    setup_dasllama_jobque()
                    var s = make_run_state(tr.config, KVDtype.f16, KVDtype.f16)
                    let short_p <- encode(tr, counting_prompt(1, 12), true, false)
                    let cont_p <- encode(tr, counting_prompt(13, 26), false, false)
                    select_prefill_override("metal")   // min_npos stays at the 64 default
                    eval_(tr, s, short_p)              // npos < 64 -> npos_window
                    set_metal_prefill_min_npos(1l)
                    eval_(tr, s, cont_p)               // start_pos = n_past > 0 -> start_pos
                    set_metal_prefill_min_npos(64l)
                    select_prefill_override("")
                    delete s
                }
                var pfd <- metal_prefill_declines()
                t |> success((pfd?["npos_window"] ?? 0l) > (pfd0?["npos_window"] ?? 0l),
                    "short prompt below min_npos declines npos_window")
                t |> success((pfd?["start_pos"] ?? 0l) > (pfd0?["start_pos"] ?? 0l),
                    "continuation prefill (start_pos > 0) declines start_pos")
                delete pfd0
                delete pfd
            }

            // ===== q8 weights, f16 scale plane (the load DEFAULT since wscale16) =====
            set_wscale_f16(true)
            var tr16 <- load_model(path, QuantMode.q8)
            tr16.config.seq_len = min(tr16.config.seq_len, 512l)
            matrix_cell(t, tr16, KVDtype.f16, "q8/s16/f16", "wscale_f16", "", "")
            set_wscale_f16(false)
            delete tr16

            // ===== native K-quant weights (Q4_K_M) — the silent-misread guard, runtime-verified =====
            // a K-quant file PASSES the q8-mode check (t.quant == q8) but its per-tensor planes are
            // not qblob offsets; without the kquant_native decline the GPU reads the wrong plane
            let kpath = path_join(models_dir(), "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
            if (stat(kpath).is_valid) {
                var trk <- load_model(kpath, QuantMode.q8)
                trk.config.seq_len = min(trk.config.seq_len, 512l)
                t |> success(trk.kquant_native, "Q4_K_M under a q8-mode load keeps native K-quant planes")
                matrix_cell(t, trk, KVDtype.f16, "k4/s32/f16", "kquant_native", "kquant_native", "kquant_native")
                // the guard's output half: with the overrides ON the tokens must be the CPU truth
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
                delete trk
            } else {
                feint("Llama-3.2-1B Q4_K_M not present (llama-quantize the Q8 source); K-quant cells skipped\n")
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
