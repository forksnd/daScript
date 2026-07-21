options gen2
options indenting = 4

// texelFetch on a layered sampler gate.
//
// A sampler2DArray indexes (x, y, layer), and the layer is a plain array index rather than a
// normalized coordinate, so the fetch coordinate is int3 where the sampler2D form takes int2. The
// lowering is the 2D one unchanged -- OpImage to extract the image from the sampled image, then
// OpImageFetch with a Lod operand -- against an OpTypeImage carrying Dim 2D and Arrayed 1. The
// coordinate is built from gl_FragCoord, so it is a runtime value and not a folded constant.

require dastest/testing_boost public
require _spirv_common               // validate_spirv
require spirv/spirv_shader          // [fragment_shader] annotation
require spirv/spirv_builtins public // gl_FragCoord + texelFetch(sampler2DArray)
require spirv/spirv_dis             // count_op / op_has_operand_at
require spirv/spirv_grammar         // SpvOp

var @binding = 0 tfa_tex : sampler2DArray
var @out @location = 0 tfa_out : float4

[fragment_shader(name="tfa_spv"), marker(no_coverage)]
def tfa_frag {
    let p = int2(gl_FragCoord.xy)
    tfa_out = texelFetch(tfa_tex, int3(p.x, p.y, 2), 0)
}

[test]
def test_texel_fetch_array(t : T?) {
    t |> run("texelFetch(sampler2DArray, int3, lod) -> OpImage + OpImageFetch on Dim 2D Arrayed 1") <| @(t : T?) {
        var words <- clone_to_move(tfa_spv)
        t |> success(count_op(words, SpvOp.TypeImage) == 1, "one OpTypeImage (the layered sampler)")
        // OpTypeImage operands: [result_id, sampled_type, dim, depth, arrayed, ...]. Dim at index 2,
        // Arrayed at index 4 -- the pair that distinguishes sampler2DArray from a plain sampler2D.
        t |> success(op_has_operand_at(words, SpvOp.TypeImage, 2, uint(SpvDim._2D)), "Dim 2D")
        t |> success(op_has_operand_at(words, SpvOp.TypeImage, 4, 1u), "Arrayed 1")
        t |> success(count_op(words, SpvOp.Image) == 1, "one OpImage (image extracted from the sampled image)")
        t |> success(count_op(words, SpvOp.ImageFetch) == 1, "one OpImageFetch")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
