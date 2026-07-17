options gen2
options persistent_heap

require bench_metal_gemv_kernels
require daslib/clargs
require daslib/strings_boost
require daslib/fio

// MSL dump rail for the GEMV kernel lab — writes selected `<arm>_msl` globals to .metal files
// for the applegpu instruction-comparison step (the AGX discipline: disasm BEFORE timing):
//   bin/daslang modules/dasLLAMA/benchmarks/matmul/dump_msl.das -- --arms k4_ext4,k4_b1c --out <dir>
//   python3 ~/Work/applegpu/compiler_explorer.py <dir>/k4_ext4.metal   (one kernel per file)
// Add new arms to the table as the lab grows.

[CommandLineArgs]
struct Args {
    @clarg_required
    @clarg_doc = "Comma-separated arm names (see the table in main)"
    arms : string

    @clarg_doc = "Output directory for the .metal files (default: current dir)"
    out : string = "."

    @clarg_short = "?"
    @clarg_doc = "Show this help and exit"
    help : bool
}

[export]
def main {
    var r <- parse_args(type<Args>)
    if (r |> is_err) {
        print("error: {r |> unwrap_err}\n\n")
        print_help(get_command_info(type<Args>), "dump_msl")
        return
    }
    let cfg <- r |> move_unwrap
    if (cfg.help) {
        print_help(get_command_info(type<Args>), "dump_msl")
        return
    }
    let srcs <- {
        "k4_b1c" => lab_k4_b1c_msl,
        "k4_mvb2" => lab_k4_mvb2_msl,
        "k4_ext2" => lab_k4_ext2_msl,
        "k4_ext4" => lab_k4_ext4_msl,
        "k5_b2r" => lab_k5_b2r_msl,
        "k6_b1d" => lab_k6_b1d_msl,
        "k6_ext2" => lab_k6_ext2_msl,
        "k6_ext4" => lab_k6_ext4_msl
    }
    for (nm in split(cfg.arms, ",")) {
        let arm = strip(nm)
        if (!key_exists(srcs, arm)) {
            print("unknown arm `{arm}`\n")
            continue
        }
        let path = path_join(cfg.out, "{arm}.metal")
        let src = srcs?[arm] ?? ""
        if (fwrite(path, src)) {
            print("wrote {path}\n")
        } else {
            print("FAILED to write {path}\n")
        }
    }
}
