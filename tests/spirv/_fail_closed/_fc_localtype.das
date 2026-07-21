// Fail-closed fixture (Phase 6.4): an unsupported LOCAL TYPE (an `int64` local -- a 64-bit INT lowers
// as an interface-block member only, where the 64-bit float does carry a value) must be rejected
// cleanly by declare_local_var. `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @ssbo @binding = 0 data : array<uint>

[compute_shader(local_size_x=64, name="fc_localtype_spv")]
def fc_localtype {
    let i = gl_GlobalInvocationID.x
    var d = 1l              // int64 local -> unsupported scalar class at this width
    d += 2l
    data[i] = data[i] + uint(d)
}
