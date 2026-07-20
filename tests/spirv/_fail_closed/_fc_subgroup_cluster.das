// Fail-closed fixture: SPIR-V requires the ClusterSize operand of a clustered reduction to be a
// CONSTANT instruction whose value is a power of two.  A non-power-of-two literal (3u) must be
// rejected by the subgroup handler, never emitted for spirv-val (or the driver) to find.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @ssbo @binding = 0 data : array<float>

[compute_shader(local_size_x=64, name="fc_subgroup_cluster_spv")]
def fc_subgroup_cluster {
    let i = gl_GlobalInvocationID.x
    data[i] = subgroupClusteredAdd(data[i], 3u)
}
