// Fail-closed fixture: gl_WorkGroupSize is the declared LocalSize, which only the workgroup-dispatched
// stages (compute / mesh / task) have.  Reading it in a fragment shader must be rejected at
// classification -- there is no size to fold it to.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @out @location = 0 fc_wgs_color : float4

[fragment_shader(name="fc_wgsize_stage_spv")]
def fc_wgsize_stage {
    fc_wgs_color = float4(float(gl_WorkGroupSize.x), 0.0f, 0.0f, 1.0f)
}
