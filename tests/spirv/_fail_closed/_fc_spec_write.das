// a spec constant is read-only in the shader (no OpVariable to store through): a write fails closed.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins public

var @constant_id = 9 fc_sc_w : uint = 4u
var @ssbo @binding = 0 fc_scw_d : array<uint>

[compute_shader(local_size_x=64, name="fc_sc_write_spv")]
def fc_sc_write {
    let gi = gl_GlobalInvocationID.x
    fc_sc_w = 5u
    fc_scw_d[gi] = fc_sc_w
}
