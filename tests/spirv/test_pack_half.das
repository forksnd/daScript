options gen2
options indenting = 4

// packHalf2x16/unpackHalf2x16 on the SPIR-V rail: GLSL.std.450 PackHalf2x16/UnpackHalf2x16
// ext-insts (numbers 58/62). Ext-inst NUMBERS don't enter the opcode census (only ExtInst
// itself, already declared); spirv-val is the correctness oracle.

require dastest/testing_boost public
require _spirv_common
require spirv/spirv_shader
require spirv/spirv_dis
require spirv/spirv_grammar

var @ssbo @binding = 0 ph_data : array<uint>

[compute_shader(local_size_x=64, name="pack_half_spv"), marker(no_coverage)]
def pack_half_kernel {
    let gi = gl_GlobalInvocationID.x
    let back = unpackHalf2x16(ph_data[gi])
    ph_data[gi] = packHalf2x16(float2(back.y, back.x))
}

[test]
def test_pack_half_spirv(t : T?) {
    t |> run("pack/unpack lower to GLSL.std.450 ext-insts and validate") <| @(t : T?) {
        var words <- clone_to_move(pack_half_spv)
        t |> equal(words[0], SPV_MAGIC)
        t |> equal(count_op(words, SpvOp.ExtInst), 2)       // one pack + one unpack
        t |> success(has_op(words, SpvOp.ExtInstImport), "GLSL.std.450 imported")
        // OpExtInst operand index 3 is the GLSLstd450 instruction literal, so the census above is
        // pinned to the two specific ext-insts rather than merely counting ExtInst opcodes.
        t |> success(op_has_operand_at(words, SpvOp.ExtInst, 3, uint(GLSLstd450.PackHalf2x16)),
            "the pack ext-inst is GLSLstd450 PackHalf2x16 (58)")
        t |> success(op_has_operand_at(words, SpvOp.ExtInst, 3, uint(GLSLstd450.UnpackHalf2x16)),
            "the unpack ext-inst is GLSLstd450 UnpackHalf2x16 (62)")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
