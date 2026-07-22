// Fail-closed fixture: cooperative-matrix ops are subgroup-scoped and only lowered in a COMPUTE shader.
// Using coopmatLoad in a fragment shader must be rejected at emit time with a clean diagnostic, not an
// invalid blob. `expect 50501` keeps dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @ssbo @binding = 0 fc_cm_a : array<float16>
var @out @location = 0 fc_cm_color : float4

[fragment_shader(name="fc_coopmat_stage_spv")]
def fc_coopmat_stage {
    var a : coopmatA_f16_16x16
    coopmatLoad(a, fc_cm_a, 0, 16, 0)
    fc_cm_color = float4(1.0f, 0.0f, 0.0f, 1.0f)
}
