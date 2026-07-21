// Fail-closed fixture: a buffer image has no mips, and SPIR-V rejects a Lod image operand on
// Dim=Buffer.  So textureBuffer declares only the two-argument texelFetch(buffer, index): the
// sampler-shaped three-argument call must be rejected at the CALL, rather than lowered to an
// OpImageFetch whose Lod operand only spirv-val or the driver would catch.
// `_`-prefix + `expect 30341` keep dastest/lint off the failing shader.
expect 30341

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @binding = 0 fc_tb_src : textureBuffer
var @ssbo @binding = 1 fc_tb_out : array<float4>

[compute_shader(local_size_x=64, name="fc_texel_buffer_lod_spv")]
def fc_texel_buffer_lod {
    let i = int(gl_GlobalInvocationID.x)
    fc_tb_out[i] = texelFetch(fc_tb_src, i, 0)
}
