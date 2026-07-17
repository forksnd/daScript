options gen2
options indenting = 4
options no_unused_block_arguments = false

// Spec constants: @constant_id globals -> OpSpecConstant (SpecConstantTrue/False for bools)
// decorated SpecId, never pooled with plain constants; local_size_x_id -> a per-axis OpSpecConstant
// folded into an OpSpecConstantComposite decorated BuiltIn WorkgroupSize, with the LocalSize
// execution mode kept alongside (glslang parity; the builtin takes precedence per the spec).

require dastest/testing_boost public
require _spirv_common
require spirv/spirv_dis
require spirv/spirv_grammar

[test]
def test_spec_constants(t : T?) {
    t |> run("OpSpecConstant shapes + SpecIds + WorkgroupSize composite; spirv-val clean") <| @(t : T?) {
        var words <- specc_words()
        // 4 OpSpecConstant: uint(8) + int(-3) + float(1.5) + the local_size_x axis (64)
        t |> equal(count_op(words, SpvOp.SpecConstant), 4)
        t |> equal(count_op(words, SpvOp.SpecConstantTrue), 1)
        t |> equal(count_op(words, SpvOp.SpecConstantFalse), 1)
        t |> equal(count_op(words, SpvOp.SpecConstantComposite), 1)
        // SpecIds 0..5 all decorated (0 = the workgroup-x axis, 1..5 = the @constant_id globals)
        for (cid in range(6)) {
            t |> success(op_has_operand_pair(words, SpvOp.Decorate, uint(SpvDecoration.SpecId), uint(cid)), "SpecId {cid} present")
        }
        t |> success(op_has_operand_pair(words, SpvOp.Decorate, uint(SpvDecoration.BuiltIn), uint(SpvBuiltIn.WorkgroupSize)), "WorkgroupSize builtin on the composite")
        t |> success(op_has_operand_at(words, SpvOp.ExecutionMode, 1, uint(SpvExecutionMode.LocalSize)), "LocalSize execution mode kept")
        // default values ride the OpSpecConstant literal at operand index 2
        t |> success(op_has_operand_at(words, SpvOp.SpecConstant, 2, 8u), "uint default 8")
        t |> success(op_has_operand_at(words, SpvOp.SpecConstant, 2, uint(-3)), "int default -3 (raw bits)")
        t |> success(op_has_operand_at(words, SpvOp.SpecConstant, 2, 0x3FC00000), "float default 1.5 (IEEE bits)")
        t |> success(op_has_operand_at(words, SpvOp.SpecConstant, 2, 64u), "workgroup-x axis default 64")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
