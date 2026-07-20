options gen2
options indenting = 4

// A single-struct SSBO block: @ssbo on a STRUCT (not array<T>) lowers to a StorageBuffer,
// Block-decorated struct wrapping the payload -- GLSL's `buffer Block { S s; }`, the shape a small
// per-dispatch control buffer needs.  It differs from the array<T> SSBO (whose block member is a
// runtime array) and from the @uniform block (Uniform storage instead of StorageBuffer).

require dastest/testing_boost public
require _spirv_common               // validate_spirv
require spirv/spirv_shader          // [compute_shader]
require spirv/spirv_builtins public // image types + imageStore
require spirv/spirv_dis             // count_op / has_op / count_op_operand_at
require spirv/spirv_grammar         // SpvOp / SpvStorageClass / SpvDecoration

struct SsbPayload {
    a : int
    b : int
    c : float
}

struct SsbBlock {
    payload : SsbPayload
}

var @ssbo @set = 0 @binding = 3 g_ssb : SsbBlock
var @binding = 4 @format = "r32f" @writeonly g_out : image2D

[compute_shader(local_size_x=1, name="ssbo_struct_spv"), marker(no_coverage)]
def ssbo_struct {
    let s = float(g_ssb.payload.a + g_ssb.payload.b) + g_ssb.payload.c
    imageStore(g_out, int2(0, 0), float4(s, 0.0f, 0.0f, 0.0f))
}

[test]
def test_ssbo_struct(t : T?) {
    t |> run("@ssbo on a struct -> single-struct StorageBuffer Block, member reads lower") <| @(t : T?) {
        var words <- clone_to_move(ssbo_struct_spv)
        let vr = validate_spirv(words)
        if (vr.ran) {
            t |> success(vr.ok, "single-struct ssbo is valid SPIR-V: {vr.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        // NOT a runtime-array SSBO: the block wraps the payload struct directly, no OpTypeRuntimeArray.
        t |> success(!has_op(words, SpvOp.TypeRuntimeArray),
            "single-struct ssbo has no runtime array (distinct from array<T> ssbo)")
        // StorageBuffer storage (distinct from a @uniform block's Uniform storage): exactly g_ssb.
        t |> success(count_op_operand_at(words, SpvOp.Variable, 2, uint(SpvStorageClass.StorageBuffer)) == 1,
            "the struct ssbo is a StorageBuffer OpVariable")
        // one Block-decorated struct (the wrapper); the nested payload is non-Block.
        t |> success(count_op_operand_at(words, SpvOp.Decorate, 1, uint(SpvDecoration.Block)) == 1,
            "the wrapper struct is Block-decorated, the nested payload is not")
        // the two-level member read (g_ssb.payload.a) lowers to an OpAccessChain in StorageBuffer.
        t |> success(has_op(words, SpvOp.AccessChain), "g_ssb.payload.a lowers to an OpAccessChain")
    }
}
