// Fail-closed fixture: a user shader function with an unsupported parameter type (int64 -- a 64-bit
// INT lowers as an interface-block member only, where the 64-bit float does carry a value) must be
// rejected at registration with a clear per-parameter message.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @ssbo @binding = 0 data : array<float>

def scale(x : int64) : float {
    return float(x)
}

[compute_shader(local_size_x=64, name="fc_badparam_spv")]
def fc_badparam {
    let i = gl_GlobalInvocationID.x
    data[i] = scale(2l)
}
