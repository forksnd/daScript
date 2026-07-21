// Fail-closed fixture: a matrix column read (m[i]) lowers to OpCompositeExtract, whose column index is
// a LITERAL, not an id.  A matrix is a value here, so there is no OpVariable to OpAccessChain a dynamic
// index through either.  A non-constant column is therefore rejected at the emit, never lowered into a
// blob only spirv-val would catch.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins
require math

var @ssbo @binding = 0 fc_mdc_cols : array<float4>

[compute_shader(local_size_x=64, name="fc_matrix_dyn_col_spv")]
def fc_matrix_dyn_col {
    let i = int(gl_GlobalInvocationID.x)
    let m = float3x3(fc_mdc_cols[0].xyz, fc_mdc_cols[1].xyz, fc_mdc_cols[2].xyz)
    fc_mdc_cols[0] = float4(m[i], 0.0f)
}
