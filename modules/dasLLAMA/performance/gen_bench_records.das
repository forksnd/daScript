options gen2
options stack = 65536   // dasLLAMA program-root convention (options stack does not unify up from libs)

require profile_common
require daslib/clargs
require daslib/fio
require math

// The official record driver (modules/dasLLAMA/PUBLIC_BENCH_PLAN.md, step 3): loops the public
// catalog x backends, runs lcpp_bench as a CHILD per cell (one model process at a time), and
// upserts the --json-path records into the per-box store — written after EVERY cell, so a crash
// never costs a measured cell. Ref rows ride along: the cpu cell carries the clean-cpu ref, the
// metal cell the stock ref (-ngl 99), and a direct stock -ngl 0 pass adds the stock CPU row.
//   bin/daslang modules/dasLLAMA/performance/gen_bench_records.das -- [-o substr] [--backend cpu|metal|all]
// Ref binaries come from setup_lcpp_ref: --ref-clean/--ref-stock or LLAMA_BENCH_CLEAN/LLAMA_BENCH_STOCK.

[CommandLineArgs]
struct Args {
    @clarg_short = "o"
    @clarg_doc = "Only catalog entries whose filename contains this substring"
    only : string

    @clarg_doc = "Backends to measure: cpu | metal | all (default: all — metal on Apple boxes only)"
    backend : string = "all"

    @clarg_name = "ref-clean"
    @clarg_doc = "clean-cpu llama-bench binary (default: env LLAMA_BENCH_CLEAN)"
    ref_clean : string

    @clarg_name = "ref-stock"
    @clarg_doc = "stock llama-bench binary (default: env LLAMA_BENCH_STOCK)"
    ref_stock : string

    @clarg_short = "t"
    @clarg_doc = "Worker threads (default: the box profile)"
    threads : int = 0

    @clarg_short = "p"
    @clarg_doc = "Prompt tokens (default: 512)"
    plen : int = 512

    @clarg_short = "n"
    @clarg_doc = "Generated tokens (default: 128)"
    ngen : int = 128

    @clarg_short = "r"
    @clarg_doc = "Timed repetitions per row (default: 5)"
    reps : int = 5

    @clarg_short = "?"
    @clarg_doc = "Show this help and exit"
    help : bool
}

// The public catalog: community-popular GGUFs, one popular quant each — NOT the internal quant
// matrix. Absent files skip with a warning, so one list serves every box.
def private pub_catalog() : array<string> {
    return <- [
        "SmolLM2-135M-Instruct-Q8_0.gguf",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "Mistral-7B-Instruct-v0.3-Q8_0.gguf",
        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "gemma-3-4b-it-Q8_0.gguf",
        "gpt-oss-20b-mxfp4.gguf",
        "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
    ]
}

def private daslang_bin() : string {
    let env = get_env_variable("DASLANG_BIN")
    return empty(env) ? "bin/daslang" : env
}

def private elapsed_ms(t0 : int64) : int {
    return get_time_usec(t0) / 1000
}

// Stamp source and upsert every run of `cell` into the store, then persist.
def private absorb_cell(var models : array<BenchModel>; var cell : array<BenchModel>; store : string) {
    for (c in cell) {
        for (r in c.runs) {
            r.source = "official"
        }
        upsert_bench_model(models, c)
    }
    write_bench_records(store, models)
}

[export]
def main : int {
    var r <- parse_args(type<Args>)
    if (r |> is_err) {
        print("error: {r |> unwrap_err}\n\n")
        print_help(get_command_info(type<Args>), "gen_bench_records")
        return 1
    }
    let cfg <- r |> move_unwrap
    if (cfg.help) {
        print_help(get_command_info(type<Args>), "gen_bench_records")
        return 0
    }
    let apple = is_apple_box()
    var backends : array<string>
    if (cfg.backend == "cpu" || cfg.backend == "all") {
        backends |> push("cpu")
    }
    if (cfg.backend == "metal" || (cfg.backend == "all" && apple)) {
        backends |> push("metal")
    }
    if (empty(backends)) {
        to_log(LOG_ERROR, "gen_bench_records: no backends selected (--backend {cfg.backend})\n")
        return 1
    }
    var refClean = empty(cfg.ref_clean) ? get_env_variable("LLAMA_BENCH_CLEAN") : cfg.ref_clean
    var refStock = empty(cfg.ref_stock) ? get_env_variable("LLAMA_BENCH_STOCK") : cfg.ref_stock
    if (!empty(refClean) && !stat(refClean).is_valid) {
        to_log(LOG_WARNING, "gen_bench_records: clean-cpu ref `{refClean}` not found — cpu cells run das-only\n")
        refClean = ""
    }
    if (!empty(refStock) && !stat(refStock).is_valid) {
        to_log(LOG_WARNING, "gen_bench_records: stock ref `{refStock}` not found — no stock rows\n")
        refStock = ""
    }
    mkdir_rec("modules/dasLLAMA/performance/records")
    let store = bench_records_path()
    var models : array<BenchModel>
    read_bench_records(store, models)
    let hw = gather_hardware()
    let date = format_time(get_clock(), "%Y-%m-%d")
    let box = box_name()
    let threads = cfg.threads > 0 ? cfg.threads : profile_threads()
    let cellPath = "modules/dasLLAMA/performance/records/_cell_{box}.json"
    var failed : array<string>
    var cat <- pub_catalog()
    for (file in cat) {
        if (!empty(cfg.only) && find(file, cfg.only) < 0) {
            continue
        }
        let path = "{models_dir()}{file}"
        if (!stat(path).is_valid) {
            to_log(LOG_WARNING, "gen_bench_records: SKIP {file} — not on this box\n")
            continue
        }
        for (be in backends) {
            let t0 = ref_time_ticks()
            to_log(LOG_INFO, "gen_bench_records: START {file} {be}\n")
            var args <- [daslang_bin(), "-jit", "modules/dasLLAMA/benchmarks/lcpp_bench.das", "--",
                         "-m", path, "-p", "{cfg.plen}", "-n", "{cfg.ngen}", "-r", "{cfg.reps}",
                         "-t", "{threads}", "-o", "json", "--json-path", cellPath]
            if (be == "metal") {
                args |> push("--ngl")
                args |> push("99")
            }
            let ref = be == "metal" ? refStock : refClean
            if (!empty(ref)) {
                args |> push("--ref")
                args |> push(ref)
                args |> push("--ref-flavor")
                args |> push(be == "metal" ? "stock" : "clean-cpu")
            }
            var out = ""
            let rc = run_and_capture(args, out)
            var cell : array<BenchModel>
            if (rc != 0 || !read_bench_records(cellPath, cell)) {
                let tail = slice(out, max(0, length(out) - 400))
                to_log(LOG_ERROR, "gen_bench_records: FAILED {file} {be} rc={rc} ({elapsed_ms(t0)} ms) — tail: {tail}\n")
                failed |> push("{file} {be}")
                continue
            }
            absorb_cell(models, cell, store)
            remove(cellPath)
            to_log(LOG_INFO, "gen_bench_records: DONE {file} {be} ({elapsed_ms(t0)} ms)\n")
        }
        // the stock CPU row — llama.cpp as shipped (Accelerate on macOS), -ngl 0, direct
        if (!empty(refStock)) {
            let t0 = ref_time_ticks()
            let aff = affinity_on() ? lcpp_affinity_args(threads) : ""
            let cmd = "{refStock} -m {path} -ngl 0 -t {threads}{aff} -p {cfg.plen} -n {cfg.ngen} -r {cfg.reps} -o json"
            to_log(LOG_INFO, "gen_bench_records: START {file} cpu stock ref\n")
            var rows : array<LlamaBenchRow>
            if (!run_llama_bench(cmd, rows)) {
                to_log(LOG_ERROR, "gen_bench_records: FAILED {file} stock cpu ref ({elapsed_ms(t0)} ms)\n")
                failed |> push("{file} cpu stock-ref")
                continue
            }
            var rr = BenchRun(
                engine = "llama.cpp",
                backend = "cpu",
                flavor = "stock",
                box = box,
                threads = threads,
                date = date,
                cmd = cmd,
                sha = rows[0].build_commit,
                source = "official",
                hardware = hw)
            for (row in rows) {
                let tag = row.n_gen == 0 ? "pp{row.n_prompt}" : "tg{row.n_gen}"
                rr.tests |> insert(tag, BenchTest(tok_s = row.avg_ts, stddev = row.stddev_ts))
            }
            var bm = BenchModel(gguf = file, params_b = double(rows[0].model_n_params) / 1.0e9lf)
            bm.runs |> emplace(rr)
            upsert_bench_model(models, bm)
            write_bench_records(store, models)
            to_log(LOG_INFO, "gen_bench_records: DONE {file} cpu stock ref ({elapsed_ms(t0)} ms)\n")
        }
    }
    if (!empty(failed)) {
        let list = join(failed, ", ")
        to_log(LOG_ERROR, "gen_bench_records: {length(failed)} FAILED cell(s): {list}\n")
        return 1
    }
    return 0
}
