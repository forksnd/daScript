options gen2
options indenting = 4

// Uniform texel buffer gate.
//
// A textureBuffer is a buffer-backed sampled image: OpTypeImage with Dim=Buffer (Sampled=1, format
// Unknown) in UniformConstant storage, plus the SampledBuffer capability. It carries no sampler, so
// the fetch loads the image variable directly instead of extracting one from a sampled image (no
// OpImage), and it carries no mips, so the OpImageFetch has NO Lod operand -- SPIR-V forbids one on
// Dim=Buffer, which is what makes the two-argument texelFetch a distinct lowering rather than a
// spelling. The index is read off an SSBO, so it is a runtime value and not a folded constant.

require dastest/testing_boost public
require _spirv_common               // validate_spirv
require spirv/spirv_shader          // [compute_shader] annotation
require spirv/spirv_builtins public // gl_GlobalInvocationID + texelFetch(textureBuffer)
require spirv/spirv_dis             // count_op / op_has_operand_at / foreach_inst
require spirv/spirv_grammar         // SpvOp / SpvDim / SpvCapability
require spirv/spirv_reflect         // decode_reflection / SpirvDescriptorKind

var @binding = 3 tb_src : textureBuffer
var @ssbo @binding = 4 tb_idx : array<int>
var @ssbo @binding = 5 tb_out : array<float4>

[compute_shader(local_size_x=64, name="tb_spv")]
def tb_comp {
    let i = int(gl_GlobalInvocationID.x)
    tb_out[i] = texelFetch(tb_src, tb_idx[i])
}

// Operand count of the first instruction with `op` (the opcode word excluded). OpImageFetch's
// mandatory operands are result-type / result-id / image / coordinate; anything beyond them is an
// image-operands mask, so 4 is the proof that no Lod rode along.
def private operand_count(words : array<uint>; op : SpvOp) : int {
    var n = -1
    foreach_inst(words) $(opcode : uint; _w : array<uint>; _s, c : int) {
        if (opcode == uint(op) && n < 0) {
            n = c
        }
    }
    return n
}

[test]
def test_texel_buffer(t : T?) {
    t |> run("texelFetch(textureBuffer, index) -> OpImageFetch on Dim Buffer, no OpImage, no Lod") <| @(t : T?) {
        var words <- clone_to_move(tb_spv)
        t |> success(count_op(words, SpvOp.TypeImage) == 1, "one OpTypeImage (the texel buffer)")
        // OpTypeImage operands: [result_id, sampled_type, dim, depth, arrayed, ms, sampled, format].
        t |> success(op_has_operand_at(words, SpvOp.TypeImage, 2, uint(SpvDim.Buffer)), "Dim Buffer")
        t |> success(op_has_operand_at(words, SpvOp.TypeImage, 6, 1u), "Sampled 1 (a sampled image, not a storage one)")
        t |> success(count_op(words, SpvOp.Image) == 0, "no OpImage (the global IS the image)")
        t |> success(count_op(words, SpvOp.ImageFetch) == 1, "one OpImageFetch")
        t |> success(operand_count(words, SpvOp.ImageFetch) == 4, "OpImageFetch carries no image operands (no Lod)")
        // The capability is operand 0 of OpCapability, so an id equal to the enum cannot match.
        t |> success(op_has_operand_at(words, SpvOp.Capability, 0, uint(SpvCapability.SampledBuffer)),
            "OpCapability SampledBuffer")
        // The host binds a buffer view, not an image view, so the descriptor kind is its own.
        let refl = decode_reflection(tb_spv_reflect)
        var found = false
        for (b in refl.bindings) {
            if (b.binding == 3) {
                found = true
                t |> success(b.kind == SpirvDescriptorKind.uniform_texel_buffer,
                    "reflection: binding kind is uniform_texel_buffer (got {b.kind})")
            }
        }
        t |> success(found, "reflection: the texel buffer is reported at its binding")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
