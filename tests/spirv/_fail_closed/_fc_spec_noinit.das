// a spec constant needs a literal initializer (its default value): a bare decl fails closed.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins public

var @constant_id = 7 fc_sc_n : uint
var @ssbo @binding = 0 fc_scn_d : array<float>

[compute_shader(local_size_x=64, name="fc_sc_noinit_spv")]
def fc_sc_noinit {
    let gi = gl_GlobalInvocationID.x
    fc_scn_d[gi] = float(fc_sc_n)
}
