// Fail-closed fixture: gl_SubgroupID names the subgroup's index within a WORKGROUP, so SPIR-V allows
// it only in the workgroup-dispatched stages (compute / mesh / task).  Reading it in a fragment shader
// must be rejected at classification.  (gl_SubgroupSize / gl_SubgroupInvocationID carry no such
// restriction and stay legal here -- only the workgroup-scoped pair is gated.)
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @out @location = 0 fc_sg_color : float4

[fragment_shader(name="fc_subgroup_id_stage_spv")]
def fc_subgroup_id_stage {
    fc_sg_color = float4(float(gl_SubgroupID), 0.0f, 0.0f, 1.0f)
}
