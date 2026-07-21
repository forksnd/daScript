// Fail-closed fixture: `array<array<T>>` under @ssbo is a descriptor array of storage buffers, so
// the INNER array is the buffer's own std430 element array -- its element must be a layable type.
// A third level (array<array<array<uint>>>) has no descriptor meaning and no byte layout, and must
// be rejected at classification rather than emitted as a runtime array with a made-up stride.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @ssbo @binding = 0 fc_sa_bufs : array<array<array<uint>>>

[compute_shader(local_size_x=64, name="fc_ssbo_array_nested_spv")]
def fc_ssbo_array_nested {
    let i = int(gl_GlobalInvocationID.x)
    fc_sa_bufs[i][i][i] = 0u
}
