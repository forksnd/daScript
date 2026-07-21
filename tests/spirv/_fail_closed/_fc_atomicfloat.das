// Fail-closed fixture: atomicAdd is the ONLY float-capable atomic (OpAtomicFAddEXT).  The float
// min/max opcodes sit behind a capability nothing here declares (AtomicFloat32MinMaxEXT), so the
// float overload is deliberately absent from the whole rest of the family -- a float atomicMax is
// rejected at the CALL, never lowered to the integer OpAtomicSMax that would reinterpret the float
// bits as a signed int.  (The emitter carries the same rule as a backstop, which this declaration
// surface keeps unreachable.)
// `_`-prefix + `expect 30341` keep dastest/lint off the failing shader.
expect 30341

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @ssbo @binding = 0 fc_af_data : array<float>

[compute_shader(local_size_x=64, name="fc_atomicfloat_spv")]
def fc_atomicfloat {
    let i = int(gl_GlobalInvocationID.x)
    let o = atomicMax(fc_af_data[i], 1.0f)
    fc_af_data[0] = o
}
