// a spec constant must be a 32-bit scalar (uint/int/float/bool): a float2 fails closed.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins public

var @constant_id = 6 fc_sc_v : float2 = float2(1.0, 2.0)
var @ssbo @binding = 0 fc_sct_d : array<float>

[compute_shader(local_size_x=64, name="fc_sc_type_spv")]
def fc_sc_type {
    let gi = gl_GlobalInvocationID.x
    fc_sct_d[gi] = fc_sc_v.x
}
