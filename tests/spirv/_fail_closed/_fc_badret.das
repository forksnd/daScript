// Fail-closed fixture: a user shader function with an unsupported result type (int64 -- a 64-bit INT
// lowers as an interface-block member only, where the 64-bit float does carry a value) must be
// rejected at registration with a clear message.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @ssbo @binding = 0 data : array<float>

def widen(x : float) : int64 {
    return int64(x)
}

[compute_shader(local_size_x=64, name="fc_badret_spv")]
def fc_badret {
    let i = gl_GlobalInvocationID.x
    data[i] = float(widen(data[i]))
}
