// subgroup ops are compute-only (v1): subgroupAdd in a fragment shader must fail closed.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins public

var @out @location = 0 fc_sg_color : float4

[fragment_shader(name="fc_sg_stage_spv")]
def fc_sg_stage {
    fc_sg_color = float4(subgroupAdd(1.0), 0.0, 0.0, 1.0)
}
