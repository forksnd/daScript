// a local_size_*_id axis colliding with a @constant_id global's SpecId fails closed.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins public

var @constant_id = 0 fc_sc_x : uint = 16u
var @ssbo @binding = 0 fc_sca_d : array<float>

[compute_shader(local_size_x=64, local_size_x_id=0, name="fc_sc_axis_spv")]
def fc_sc_axis {
    let gi = gl_GlobalInvocationID.x
    fc_sca_d[gi] = float(fc_sc_x)
}
