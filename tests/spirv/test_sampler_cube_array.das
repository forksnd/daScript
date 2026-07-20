options gen2
options indenting = 4

// samplerCubeArray gate -- the sampled counterpart of imageCubeArray.
//
// A LAYERED cubemap sampled through a RUNTIME-SIZED descriptor array (GLSL `uniform samplerCubeArray
// pool[]`), which is how a pool of cube textures is bound: the host sizes the pool, so the array is an
// OpTypeRuntimeArray of OpTypeSampledImage in UniformConstant storage (no ArrayStride -- a descriptor
// slot has no byte layout) and the whole read is OpAccessChain + OpLoad + OpImageSampleExplicitLod.
//
// Two capabilities ride along and are BOTH pinned: SampledCubeArray for the Arrayed=1 cube image, and
// RuntimeDescriptorArray for the unsized array. The coordinate is a float4 -- .xyz the direction that
// picks the face, .w the cube layer -- and the Lod operand is what makes the sample stage-agnostic
// (an implicit-lod texture() would be fragment-only, and is deliberately not declared).

require dastest/testing_boost public
require _spirv_common               // validate_spirv
require spirv/spirv_shader          // [compute_shader] annotation
require spirv/spirv_builtins public // gl_GlobalInvocationID + samplerCubeArray / textureLod
require spirv/spirv_dis             // count_op / op_has_operand / op_has_operand_at
require spirv/spirv_grammar         // SpvOp / SpvDim / SpvCapability / SpvImageOperands
require spirv/spirv_reflect         // decode_reflection / SpirvDescriptorKind

var @binding = 6 sca_cubes : array<samplerCubeArray>
var @ssbo @binding = 7 sca_slot : array<int>
var @ssbo @binding = 8 sca_out : array<float4>

[compute_shader(local_size_x=64, name="sca_spv")]
def sca_comp {
    let i = int(gl_GlobalInvocationID.x)
    let dir = float3(0.0, 1.0, 0.0)
    sca_out[i] = textureLod(sca_cubes[sca_slot[i]], float4(dir, float(i)), 0.0)
}

[test]
def test_sampler_cube_array(t : T?) {
    t |> run("textureLod(samplerCubeArray[idx], float4(dir, layer), lod) -> OpImageSampleExplicitLod") <| @(t : T?) {
        var words <- clone_to_move(sca_spv)
        t |> success(count_op(words, SpvOp.TypeImage) == 1, "one OpTypeImage (the cube array)")
        // OpTypeImage operands: [result_id, sampled_type, dim, depth, arrayed, ms, sampled, format].
        t |> success(op_has_operand_at(words, SpvOp.TypeImage, 2, uint(SpvDim.Cube)), "Dim Cube")
        t |> success(op_has_operand_at(words, SpvOp.TypeImage, 4, 1u), "Arrayed 1 (a LAYERED cubemap)")
        t |> success(op_has_operand_at(words, SpvOp.TypeImage, 6, 1u), "Sampled 1 (sampled, not a storage image)")
        t |> success(count_op(words, SpvOp.TypeSampledImage) == 1, "one OpTypeSampledImage (combined)")
        // The descriptor array is UNSIZED: a runtime array, never an OpTypeArray with a count.
        t |> success(count_op(words, SpvOp.TypeRuntimeArray) >= 1, "the descriptor array is an OpTypeRuntimeArray")
        // The read: chain one slot off the array, load the sampled image, sample it with an explicit Lod.
        t |> success(count_op(words, SpvOp.ImageSampleExplicitLod) == 1, "one OpImageSampleExplicitLod")
        t |> success(op_has_operand(words, SpvOp.ImageSampleExplicitLod, uint(SpvImageOperands.Lod)),
            "the sample carries an explicit Lod operand")
        // An implicit-lod sample would be fragment-only and is not declared, so none may appear.
        t |> success(count_op(words, SpvOp.ImageSampleImplicitLod) == 0, "no implicit-lod sample")
        t |> success(op_has_operand(words, SpvOp.Capability, uint(SpvCapability.SampledCubeArray)),
            "OpCapability SampledCubeArray")
        t |> success(op_has_operand(words, SpvOp.Capability, uint(SpvCapability.RuntimeDescriptorArray)),
            "OpCapability RuntimeDescriptorArray")
        // The host sizes the pool, so the binding reports a variable descriptor count (0).
        let refl = decode_reflection(sca_spv_reflect)
        var found = false
        for (b in refl.bindings) {
            if (b.binding == 6) {
                found = true
                t |> success(b.kind == SpirvDescriptorKind.combined_image_sampler,
                    "reflection: binding kind is combined_image_sampler (got {b.kind})")
                t |> success(b.count == 0, "reflection: variable descriptor count (got {b.count})")
            }
        }
        t |> success(found, "reflection: the cube-array pool is reported at its binding")
        // RuntimeDescriptorArray raises the module to SPIR-V 1.5, so validate against the env that
        // speaks it (Vulkan 1.2) rather than the 1.3 default.
        let r = validate_spirv(words, "vulkan1.2")
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
