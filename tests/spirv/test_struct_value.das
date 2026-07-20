options gen2
options indenting = 4
options no_aot     // requires the no_aot _spirv_xmod fixture (optimizer-folded shape; AOT hash unstable)

// Struct VALUES: a plain (undecorated) struct as a function parameter and as a function result, plus
// the local whose initializer the optimizer dropped as dead. Both are what porting a GLSL helper that
// returns a struct needs -- glslang emits `OpFunction %SvPhys` over an OpTypeStruct with no Block /
// Offset decoration, and so must dasSpirv.
//
// The fixtures live in the REQUIRED module _spirv_xmod on purpose: that is the real-world shape (a
// shared CPU/GPU core module), and the dead-initializer drop happens ONLY there -- a required module
// finishes optimization before the requiring module's macros run, so a same-file copy would keep its
// initializer and never exercise the path. spirv-val is the oracle for the composite/type correctness
// the counts cannot see.

require dastest/testing_boost public
require _spirv_common              // validate_spirv
require _spirv_xmod                // struct-by-value + dropped-init helpers, optimized
require spirv/spirv_shader         // [compute_shader] annotation
require spirv/spirv_builtins public // gl_GlobalInvocationID
require spirv/spirv_dis            // has_op / count_op
require spirv/spirv_grammar        // SpvOp

var @ssbo @binding = 0 sv_data : array<float>

[compute_shader(local_size_x=64, name="sv_spv"), marker(no_coverage)]
def sv {
    let i = int(gl_GlobalInvocationID.x)
    let d = xm_sv_seed(sv_data[i])
    let p = xm_sv_phys(sv_data[i + 1], d)
    sv_data[i] = p.temp + p.radius + xm_sv_dropped_init(sv_data[i + 2])
}

[test]
def test_struct_value(t : T?) {
    t |> run("struct values: struct param + struct result, plain OpTypeStruct, dropped init") <| @(t : T?) {
        var words <- clone_to_move(sv_spv)
        // entry + xm_sv_seed + xm_sv_phys + xm_sv_dropped_init = 4 OpFunctions, each called once.
        t |> equal(count_op(words, SpvOp.Function), 4)
        t |> equal(count_op(words, SpvOp.FunctionCall), 3)
        // seed(s) = 1 + phys(m, d) = 2 + dropped_init(m) = 1 -> 4 OpFunctionParameter. The struct
        // param `d` is ONE by-value parameter, not a flattened field list.
        t |> equal(count_op(words, SpvOp.FunctionParameter), 4)
        // SvDraws + SvPhys + the sv_data SSBO's Block = 3 OpTypeStruct. SvDraws is named by the seed's
        // result, the phys's parameter AND the construction inside the seed; without dedup by Structure
        // identity each mention would mint its own, giving 5.
        t |> equal(count_op(words, SpvOp.TypeStruct), 3)
        // exactly ONE OpMemberDecorate -- the SSBO Block's Offset. A struct VALUE is not an interface
        // block, so neither SvDraws nor SvPhys may carry a member decoration.
        t |> equal(count_op(words, SpvOp.MemberDecorate), 1)
        // the two constructions -> one OpCompositeConstruct each (no per-field store into a temp).
        t |> equal(count_op(words, SpvOp.CompositeConstruct), 2)
        // d.scat_temp / d.scat_r / d.giant_r / p.temp / p.radius -> 5 reads off a struct VALUE,
        // each an OpCompositeExtract (glslang would AccessChain a Function-storage copy instead).
        t |> equal(count_op(words, SpvOp.CompositeExtract), 5)
        // the dropped initializer lowers to the zero das guarantees such a local: one OpConstantNull.
        t |> equal(count_op(words, SpvOp.ConstantNull), 1)
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "struct_value: spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
