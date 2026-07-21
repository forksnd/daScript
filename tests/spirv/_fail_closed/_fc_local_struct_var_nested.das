// Fail-closed fixture: a `var` struct LOCAL whose field is itself a struct. The local's OpTypeStruct is
// undecorated (Function storage has no Offset cascade), so a nested field would need a recursive plain
// struct that build_plain_struct does not build -- it must be refused, never guessed at. Distinct from
// _fc_local_struct_nested, which refuses the same shape on the VALUE path (a `let` copy + extract).
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

struct VnInner {
    x : float4
}
struct VnOuter {
    inner : VnInner
    tail  : float
}
var @ssbo @binding = 0 vn_out : array<float>

[compute_shader(local_size_x=64, name="fc_lsv_var_nested_spv")]
def fc_lsv_var_nested {
    let i = int(gl_GlobalInvocationID.x)
    var s : VnOuter               // nested-struct field of a struct LOCAL -> fail-closed
    s.tail = float(i)
    vn_out[i] = s.tail
}
