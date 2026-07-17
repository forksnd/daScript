// two spec constants sharing one @constant_id fail closed (SpecIds must be unique per module).
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins public

var @constant_id = 8 fc_sc_a : uint = 1u
var @constant_id = 8 fc_sc_b : uint = 2u
var @ssbo @binding = 0 fc_scd_d : array<float>

[compute_shader(local_size_x=64, name="fc_sc_dupid_spv")]
def fc_sc_dupid {
    let gi = gl_GlobalInvocationID.x
    fc_scd_d[gi] = float(fc_sc_a + fc_sc_b)
}
