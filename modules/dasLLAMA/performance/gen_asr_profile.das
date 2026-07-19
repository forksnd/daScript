options gen2
options stack = 131072   // the ASR model structs (decoder + tower) need a deep stack (asr_bench parity)

// The das-side half of the ASR profiling suite — the audio sibling of gen_profile.das. For each
// catalog model it times transcribe (best-of-reps) over its corpus buckets under the tuned per-box
// runtime, pulls the matching reference tok/ms from the committed baseline TSV (never re-run here — see
// establish_asr_baselines.das), and MERGES the row into the persistent per-box ASR profile JSON.
// The das side is uniform: load_asr_model + transcribe drives every model (single-file pure ASR and
// two-file audio-chat alike). JIT-only, Parsec off.
//
// NOTE: unlike gen_profile.das (LLM), this does NOT set _jit_fast_math. LLM emission is a fixed token
// count (content-independent), so fast-math only shifts speed; ASR transcription length is
// content-dependent, so a fast-math token flip would change the decode loop length and the timing.
//   bin/daslang -jit modules/dasLLAMA/performance/gen_asr_profile.das -- -o Parakeet
//   bin/daslang -jit modules/dasLLAMA/performance/gen_asr_profile.das -- --libri

require dasllama/dasllama        // nolint:STYLE029 — umbrella pulls the ASR API + every arch [init]
require dasllama/dasllama_math   // setup_dasllama_jobque_ + apply_box_profile_runtime (engine-level, not re-exported)
require dasllama/dasllama_tune   // box_profile_path (the applied per-box manifest, recorded in the platform block)
require llvm/daslib/llvm_tune    // [tune_policy] + log_tune_status — the box tunes itself before measuring
require audio
require daslib/jobque_boost
require profile_common
require daslib/clargs
require daslib/fio
require strings
require math

[CommandLineArgs]
struct Args {
    @clarg_short = "o"
    @clarg_doc = "Only models whose display name contains this substring (default: all)"
    only : string = ""

    @clarg_doc = "Threads recorded in the platform block (default 8 — the perf-core count on both boxes)"
    threads : int = 8

    @clarg_short = "r"
    @clarg_doc = "Timed repetitions per clip; the profile keeps the best (default 2)"
    reps : int = 2

    @clarg_doc = "Also measure the 25-clip LibriSpeech dictation aggregate (default off)"
    libri : bool

    @clarg_doc = "Parsec was off for this run — recorded in the platform block (default true)"
    parsec_off : bool = true

    @clarg_short = "?"
    @clarg_doc = "Show this help and exit"
    help : bool
}

// Decode any file miniaudio understands to 16 kHz mono f32 (transcribe.das / asr_bench rail).
def load_audio_16k_mono(path : string) : array<float> {
    var decoder = new ma_decoder
    var config <- ma_decoder_config_init(ma_format.ma_format_f32, 1u, uint(AUDIO_SAMPLE_RATE))
    if (ma_decoder_init_file(path, unsafe(addr(config)), decoder) != ma_result.MA_SUCCESS) {
        panic("cannot open audio file '{path}'")
    }
    var out : array<float>
    let chunk = 65536
    var buf : array<float>
    buf |> resize(chunk)
    while (true) {
        let got = int(ma_decoder_read_pcm_frames(decoder, unsafe(addr(buf[0])), uint64(chunk)))
        break if (got <= 0)
        out |> reserve(length(out) + got)
        for (i in range(got)) {
            out |> push(buf[i])
        }
        break if (got < chunk)
    }
    ma_decoder_uninit(decoder)
    unsafe {
        delete decoder
    }
    delete buf
    return <- out
}

// Best-of-`reps` transcribe ms for one loaded model over `samples`, from a warmed JIT + live jobque.
def measure_das(var m : AsrModel; samples : array<float>; reps : int; lang : string) : double {
    var best = 1.0e30lf
    for (_rep in range(reps)) {
        var s <- create_session(m, lang)
        let t0 = ref_time_ticks()
        var ntok = 0
        transcribe(m, s, samples) $(seg) {
            ntok += length(seg.tokens)
        }
        let ms = double(get_time_usec(t0)) / 1000.0lf
        best = min(best, ms)
        delete s
    }
    return best
}

// The reference times measured for `display` on `wav` (kind "corpus"), one per tool that ran.
def ref_times_for(baselines : array<AsrBaseRow>; display, wav : string) : array<AsrRefTime> {
    var out : array<AsrRefTime>
    for (b in baselines) {
        if (b.model == display && b.kind == "corpus" && b.wav == wav && b.ms > 0.0lf) {
            var rt = AsrRefTime(tool = b.tool, ms = b.ms)
            if (b.encode_ms > 0.0lf) {
                rt.encode_ms = b.encode_ms
            }
            out |> push(rt)
        }
    }
    return <- out
}

// mean / p50 / p95 of a latency sample (ms). (-1,-1,-1) when empty.
def aggregate(ms_list : array<double>) : tuple<mean : double; p50 : double; p95 : double> {
    if (empty(ms_list)) {
        return (mean = -1.0lf, p50 = -1.0lf, p95 = -1.0lf)
    }
    var xs := ms_list
    sort(xs)
    var sum = 0.0lf
    for (x in xs) {
        sum += x
    }
    let n = length(xs)
    return (mean = sum / double(n), p50 = xs[n / 2], p95 = xs[min(n - 1, int(0.95lf * double(n)))])
}

// The applied per-box profile filename (or "default" when none is on disk) for the platform block.
def box_profile_tag() : string {
    let bp = box_profile_path()
    return stat(bp).is_valid ? base_name(bp) : "default"
}

// The 25 LibriSpeech dictation clips (sorted NN_* order), full paths.
def libri_clips() : array<string> {
    var out : array<string>
    let d = "{asr_corpus_dir()}librispeech/"
    dir(d) $(fn) {
        if (ends_with(fn, ".wav")) {
            out |> push("{d}{fn}")
        }
    }
    sort(out)
    return <- out
}

// Replace-or-append a dictation side by its label, so a re-run (or a second model's --libri pass)
// refreshes its rows instead of duplicating them.
def upsert_side(var sides : array<LibriStat>; s : LibriStat) {
    for (x in sides) {
        if (x.side == s.side) {
            x = s
            return
        }
    }
    sides |> push(s)
}

// Add the das + per-tool reference LibriSpeech dictation stats for one model into `sides`.
def add_libri_sides(var sides : array<LibriStat>; var m : AsrModel; spec : AsrModelSpec;
                    baselines : array<AsrBaseRow>; reps : int) {
    let clips <- libri_clips()
    var das_ms : array<double>
    das_ms |> reserve(length(clips))
    for (path in clips) {
        var samples <- load_audio_16k_mono(path)
        das_ms |> push(measure_das(m, samples, reps, spec.lang))
        delete samples
    }
    let da = aggregate(das_ms)
    upsert_side(sides, LibriStat(side = "das {spec.libri_short}", mean_ms = da.mean, p50_ms = da.p50, p95_ms = da.p95))
    // reference sides from the stored libri baseline rows (per tool, aggregated across clips)
    for (ref in spec.refs) {
        let rms <- [for (b in baselines);
            b.ms;
            where b.model == spec.display && b.kind == "libri" && b.tool == ref.tool && b.ms > 0.0lf]
        if (!empty(rms)) {
            let ra = aggregate(rms)
            upsert_side(sides, LibriStat(side = "{ref.tool} {spec.libri_short}", mean_ms = ra.mean, p50_ms = ra.p50, p95_ms = ra.p95))
        }
    }
}

// tune_policy: an untuned box silently publishes fallback-tier numbers — see gen_profile.das
[export, tune_policy(missing = "auto")]
def main() {
    allow_cpu_prefill()   // CPU rows are this tool's product — disarm the Metal-build tripwire
    if (!jit_enabled()) {
        print("gen_asr_profile is JIT-only — run with: daslang -jit <file>\n")
        return
    }
    var r <- parse_args(type<Args>)
    if (r |> is_err) {
        print("error: {r |> unwrap_err}\n\n")
        print_help(get_command_info(type<Args>), "gen_asr_profile")
        return
    }
    let cfg <- r |> move_unwrap
    if (cfg.help) {
        print_help(get_command_info(type<Args>), "gen_asr_profile")
        return
    }

    let baselines <- read_asr_baselines()
    let lcpp_commit = ""   // ASR references are heterogeneous; the platform llama_cpp field is N/A here

    var prof : AsrProfile
    read_asr_profile(prof)
    let date = format_time(get_clock(), "%Y-%m-%d")   // portable (no shell) — Windows cmd `date` prompts and hangs
    prof.platform = gather_platform(cfg.threads, lcpp_commit, date, box_profile_tag(), cfg.parsec_off)

    apply_box_profile_runtime()
    log_tune_status("gen_asr_profile")

    logger_set_name(prof_log_name())
    print("gen_asr_profile: box={box_name()} threads={cfg.threads} reps={cfg.reps} libri={cfg.libri}\n")
    print("log: {logger_get_path()}  (tail for live, timestamped progress)\n")
    print("baselines: {length(baselines)} row(s) from {baseline_asr_path()}\n\n")

    with_job_que() {
        setup_dasllama_jobque_()
        let lanes = get_dispatch_lanes()
        prof.platform.threads = lanes   // record the ACTUAL das lane count, not the advisory flag
        logger_info("profile.asr.gen", "run start", JV({"box" => JV(box_name()), "reps" => JV(cfg.reps),
            "libri" => JV(cfg.libri), "requested_threads" => JV(cfg.threads), "dispatch_lanes" => JV(lanes)}))
        if (lanes != cfg.threads) {
            logger_warning("profile.asr.gen", "dispatch lanes != requested threads — set DAS_JOBQUE_THREADS",
                JV({"dispatch_lanes" => JV(lanes), "requested_threads" => JV(cfg.threads)}))
            print("WARNING: das dispatch lanes = {lanes}, requested {cfg.threads}. Set DAS_JOBQUE_THREADS={cfg.threads}.\n")
        }
        for (spec in asr_catalog()) {
            if (!empty(cfg.only) && find(spec.display, cfg.only) < 0) {
                continue
            }
            let model_path = "{spec.root}{spec.das_model}"
            if (!stat(model_path).is_valid) {
                logger_warning("profile.asr.gen", "model absent — skipped", JV({"model" => JV(spec.display), "path" => JV(model_path)}))
                continue
            }
            logger_info("profile.asr.gen", "model start", JV({"model" => JV(spec.display)}))
            let t0 = ref_time_ticks()
            var m <- (empty(spec.das_mmproj)
                ? load_asr_model(model_path)
                : load_asr_model(model_path, "{spec.root}{spec.das_mmproj}"))
            // warmup (JIT compile + first-touch) on the shortest bucket
            var warm <- load_audio_16k_mono("{asr_corpus_dir()}{spec.buckets[0]}")
            measure_das(m, warm, 1, spec.lang)
            delete warm

            var buckets : array<AsrBucket>
            buckets |> reserve(length(spec.buckets))
            for (wav in spec.buckets) {
                var samples <- load_audio_16k_mono("{asr_corpus_dir()}{wav}")
                let audio_s = double(length(samples)) / double(AUDIO_SAMPLE_RATE)
                let das_ms = measure_das(m, samples, cfg.reps, spec.lang)
                buckets |> emplace(AsrBucket(wav = wav, audio_s = audio_s, das_ms = das_ms,
                    refs <- ref_times_for(baselines, spec.display, wav)))
                logger_info("profile.asr.gen", "bucket done", JV({"model" => JV(spec.display), "wav" => JV(wav),
                    "das_ms" => JV(das_ms), "audio_s" => JV(audio_s)}))
                delete samples
            }
            // NB: under -jit, a comprehension used INLINE as a `<-` move-init inside a struct
            // constructor silently yields an EMPTY array (interp is fine) — hoist it to a local
            // first. Verified JIT miscompile; the plain-local `<-` (buckets) beside it is unaffected.
            var reftools <- [for (rf in spec.refs); rf.tool]
            var rec = AsrRec(model = spec.display, das_model = spec.das_model,
                references <- reftools, buckets <- buckets)
            upsert_asr(prof.tests, rec)

            if (cfg.libri && !empty(spec.libri_short)) {
                add_libri_sides(prof.librispeech.sides, m, spec, baselines, cfg.reps)
                prof.librispeech.n_clips = 25
                logger_info("profile.asr.gen", "librispeech aggregate updated", JV({"model" => JV(spec.display)}))
            }
            write_asr_profile(prof)   // persist now — a crash on a later model never costs this one
            let ms = double(get_time_usec(t0)) / 1000.0lf
            logger_info("profile.asr.gen", "model done", JV({"model" => JV(spec.display), "elapsed_ms" => JV(ms), "buckets" => JV(length(spec.buckets))}))
            delete m   // free the loaded model's weights before the next model (else 6 models stay resident)
        }
    }
    logger_info("profile.asr.gen", "run done", JV({"profiled" => JV(length(prof.tests)), "path" => JV(profile_asr_path())}))
    print("profiled {length(prof.tests)} model row(s) -> {profile_asr_path()}\n")
}
