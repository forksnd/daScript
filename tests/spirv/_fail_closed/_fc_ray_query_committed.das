// Fail-closed fixture (ray query): the `committed` argument of a rayQueryGetIntersection* accessor must
// be a bool LITERAL, so the emitter can fold it to the 32-bit int operand SPIR-V requires (glslang
// lowers GLSL's `true` to an int constant). A runtime bool has no compile-time value to fold, so it is
// rejected cleanly rather than reaching an invalid blob. `_`-prefix + `expect 50501` keep dastest/lint
// off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @set = 0 @binding = 0 fc_rq_tlas : accelerationStructureEXT
var @ssbo @binding = 1 fc_rq_out : array<uint>
[compute_shader(local_size_x=64, name="fc_ray_query_committed_spv")]
def fc_ray_query_committed {
    let i = gl_GlobalInvocationID.x
    var rq : rayQueryEXT
    rayQueryInitializeEXT(rq, fc_rq_tlas, 0x01u, 0xFFu,
        float3(0.0, 0.0, 0.0), 0.001, float3(0.0, 0.0, 1.0), 100.0)
    while (rayQueryProceedEXT(rq)) {}
    let flag = i > 0u   // a RUNTIME bool, not a compile-time literal
    fc_rq_out[i] = rayQueryGetIntersectionTypeEXT(rq, flag)
}
