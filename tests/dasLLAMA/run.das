options gen2
options indenting = 4

// The ONLY sanctioned way to run the dasLLAMA Metal suites (see tests/dasLLAMA/CLAUDE.md).
// Enforces the iteration discipline that keeps suite walls honest: exactly ONE of --arm
// (filtered iteration, minutes) or --full (the pre-commit gate: every arm + the large-tier
// models, run ONCE after batching fixes). Every run's COMPLETE output lands in a log file
// whose path is printed — never pipe suite output through a grep at capture time.

require daslib/clargs
require daslib/fio
require strings
require math

[CommandLineArgs]
struct RunArgs {
    @clarg_short = "a"
    @clarg_doc = "Arm filter — comma-separated substrings matched against arm names (e.g. arm12,batchB8); sets DASLLAMA_TEST_ARMS"
    arm : string

    @clarg_short = "f"
    @clarg_doc = "The pre-commit gate: every arm + large-tier models (DASLLAMA_PARITY_FULL=1), raised dastest timeout. Run ONCE after batching fixes"
    full : bool

    @clarg_short = "s"
    @clarg_doc = "Suite: decode | prefill | matrix | kernels | all (default all)"
    suite : string = "all"

    @clarg_short = "n"
    @clarg_doc = "Repeat count — explicit only (default 1, never best-of-N)"
    nreps : int = 1

    @clarg_doc = "daslang binary (default ./bin/daslang — run from the repo root)"
    das : string = "./bin/daslang"

    @clarg_short = "?"
    @clarg_name = "show-help"
    @clarg_doc = "Show this help and exit"
    help : bool
}

def private suite_files(name : string) : array<string> {
    if (name == "decode") {
        return <- [ "tests/dasLLAMA/test_metal_decode_parity.das" ]
    } elif (name == "prefill") {
        return <- [ "tests/dasLLAMA/test_metal_prefill_parity.das" ]
    } elif (name == "matrix") {
        return <- [ "tests/dasLLAMA/test_metal_support_matrix.das" ]
    } elif (name == "kernels") {
        return <- [ "tests/dasLLAMA/test_metal_prefill_kernels.das" ]
    } elif (name == "all") {
        return <- [ "tests/dasLLAMA/test_metal_decode_parity.das",
                    "tests/dasLLAMA/test_metal_prefill_parity.das",
                    "tests/dasLLAMA/test_metal_support_matrix.das",
                    "tests/dasLLAMA/test_metal_prefill_kernels.das" ]
    }
    var none : array<string>
    return <- none
}

// the trailing "N tests, ..." summary line of a completed dastest log ("" when the run died
// before printing one — crash, kill, or compile error)
def private log_summary(log_path : string) : string {
    var summary = ""
    fopen(log_path, "rb") $(f) {
        if (f != null) {
            let text = fread(f)
            peek_data(text) $(bytes) {
                var at = 0
                var line_start = 0
                while (at < length(bytes)) {
                    if (bytes[at] == uint8('\n')) {
                        let line = slice(text, line_start, at)
                        if (find(line, " tests, ") >= 0) {
                            summary = clone_string(line)
                        }
                        line_start = at + 1
                    }
                    at++
                }
            }
        }
    }
    return summary
}

[export]
def main() : int {
    var r <- parse_args(type<RunArgs>)
    if (r |> is_err) {
        to_log(LOG_ERROR, "error: {r |> unwrap_err}\n\n")
        print_help(get_command_info(type<RunArgs>), "tests/dasLLAMA/run.das")
        return 2
    }
    let cfg <- r |> move_unwrap
    if (cfg.help) {
        print_help(get_command_info(type<RunArgs>), "tests/dasLLAMA/run.das")
        return 0
    }
    if (cfg.full == !empty(cfg.arm)) {   // both set, or neither
        to_log(LOG_ERROR, "pass exactly ONE of --arm <filter> (iteration) or --full (the pre-commit gate)\n")
        return 2
    }
    var files <- suite_files(cfg.suite)
    if (empty(files)) {
        to_log(LOG_ERROR, "unknown --suite '{cfg.suite}' (decode | prefill | matrix | kernels | all)\n")
        return 2
    }
    let tmp = empty(get_env_variable("TMPDIR")) ? "/tmp" : get_env_variable("TMPDIR")
    let envs = cfg.full ? "DASLLAMA_PARITY_FULL=1" : "DASLLAMA_TEST_ARMS='{cfg.arm}'"
    // the full gate holds the 8B/70B tails — the stock 1200s dastest wall is an arm-mode budget
    let tmo = cfg.full ? 3600 : 1200
    let reps = max(cfg.nreps, 1)
    var failures = 0
    for (rep in range(reps)) {
        for (file in files) {
            let tag = base_name(file)
            let log_path = path_join(tmp, "dasllama_run_{tag}_{rep}_{ref_time_ticks()}.log")
            let cmd = "{envs} {cfg.das} -jit dastest/dastest.das -- --test {file} --timeout {tmo} > \"{log_path}\" 2>&1"
            let rep_tag = reps > 1 ? " rep {rep + 1}/{reps}" : ""
            to_log(LOG_INFO, "START {tag}{rep_tag}: {cmd}\n")
            let t0 = ref_time_ticks()
            var rc = 0
            unsafe {
                rc = popen(cmd) $(f) {
                    let _drain = fread_to_eof(f)   // output is redirected; drain keeps the pipe contract
                }
            }
            let secs = double(get_time_usec(t0)) / 1000000.0lf
            let summary = log_summary(log_path)
            let verdict = !empty(summary) ? summary : "no dastest summary — the run died early (read the log)"
            to_log(rc == 0 ? LOG_INFO : LOG_ERROR, "DONE {tag}: exit={rc} {verdict} elapsed={secs}s log={log_path}\n")
            if (rc != 0) {
                failures++
            }
        }
    }
    return failures
}
