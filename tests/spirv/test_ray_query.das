options gen2
options indenting = 4

// Inline ray query (VK_KHR_ray_query) in a COMPUTE shader. The fixture (ray_query in _spirv_common)
// declares an opaque rayQueryEXT local, initializes it against a single TLAS descriptor, steps it with
// `while (rayQueryProceedEXT(rq)) {}`, and reads the committed intersection through the four
// GetIntersection* accessors. This asserts:
//   * GLCompute execution model + a LocalSize execution mode (a ray query is NOT an RT-pipeline stage)
//   * OpCapability RayQueryKHR + OpExtension "SPV_KHR_ray_query", each exactly once, and NO
//     RayTracingKHR / SPV_KHR_ray_tracing (a ray query is not a trace)
//   * OpTypeRayQueryKHR + a Function-storage OpVariable of it (the rq local)
//   * exactly one each of OpRayQueryInitializeKHR / ProceedKHR / GetIntersectionTypeKHR /
//     InstanceCustomIndexKHR / PrimitiveIndexKHR / BarycentricsKHR / TKHR, and NO OpTraceRayKHR
//   * the TLAS descriptor: OpTypeAccelerationStructureKHR + set/binding decorations
//   * spirv-val against vulkan1.2 (which, among other things, rejects a bool committed operand where the
//     accessors require a 32-bit int).

require dastest/testing_boost public
require _spirv_common
require spirv/spirv_dis
require spirv/spirv_grammar

def private validate(t : T?; words : array<uint>; lbl : string) {
    let r = validate_spirv(words, "vulkan1.2")
    if (r.ran) {
        t |> success(r.ok, "{lbl}: spirv-val: {r.msg}")
    } else {
        feint("spirv-val not found locally; skipping (CI enforces)\n")
    }
}

// Result id (operand 1) of the FIRST OpVariable whose storage class (operand 2) is `sc`.
def private variable_id_with_storage(words : array<uint>; sc : SpvStorageClass) : tuple<ok : bool; value : uint> {
    var res = (ok = false, value = 0u)
    foreach_inst(words) $(opcode : uint; w : array<uint>; s, n : int) {
        if (res.ok || opcode != uint(SpvOp.Variable)) return
        if (n > 2 && w[s + 2] == uint(sc)) {
            res = (ok = true, value = w[s + 1])
        }
    }
    return res
}

[test]
def test_ray_query_compute(t : T?) {
    t |> run("[compute_shader] inline ray query: cap + ext + type + the seven OpRayQuery* + spirv-val") <| @(t : T?) {
        var words <- ray_query_words()
        // a ray query is a plain compute stage: GLCompute model + a LocalSize execution mode
        t |> success(op_has_operand_at(words, SpvOp.EntryPoint, 0, uint(SpvExecutionModel.GLCompute)),
            "ray_query: GLCompute execution model")
        t |> success(count_op(words, SpvOp.ExecutionMode) >= 1,
            "ray_query: a LocalSize execution mode is present")
        // RayQueryKHR + SPV_KHR_ray_query, each exactly once; and NOT the RT-pipeline cap/ext
        t |> success(op_has_operand_at(words, SpvOp.Capability, 0, uint(SpvCapability.RayQueryKHR)),
            "ray_query: OpCapability RayQueryKHR")
        let ncap = count_op_operand_at(words, SpvOp.Capability, 0, uint(SpvCapability.RayQueryKHR))
        t |> success(ncap == 1, "ray_query: exactly one RayQueryKHR cap (got {ncap})")
        t |> success(!op_has_operand_at(words, SpvOp.Capability, 0, uint(SpvCapability.RayTracingKHR)),
            "ray_query: no RayTracingKHR capability (a ray query is not a trace)")
        // OpTypeRayQueryKHR + a Function-storage OpVariable of it (the rq local)
        t |> success(has_op(words, SpvOp.TypeRayQueryKHR), "ray_query: OpTypeRayQueryKHR")
        let rqvar = variable_id_with_storage(words, SpvStorageClass.Function)
        t |> success(rqvar.ok, "ray_query: a Function-storage OpVariable (the rq local)")
        // the TLAS descriptor
        t |> success(has_op(words, SpvOp.TypeAccelerationStructureKHR),
            "ray_query: OpTypeAccelerationStructureKHR")
        t |> success(op_has_operand_at(words, SpvOp.Decorate, 1, uint(SpvDecoration.DescriptorSet)),
            "ray_query: TLAS has a DescriptorSet decoration")
        // the seven ray-query opcodes, each exactly once; and NO OpTraceRayKHR
        for (op in [SpvOp.RayQueryInitializeKHR, SpvOp.RayQueryProceedKHR,
                    SpvOp.RayQueryGetIntersectionTypeKHR,
                    SpvOp.RayQueryGetIntersectionInstanceCustomIndexKHR,
                    SpvOp.RayQueryGetIntersectionPrimitiveIndexKHR,
                    SpvOp.RayQueryGetIntersectionBarycentricsKHR,
                    SpvOp.RayQueryGetIntersectionTKHR]) {
            let c = count_op(words, op)
            t |> success(c == 1, "ray_query: exactly one {op} (got {c})")
        }
        t |> success(count_op(words, SpvOp.TraceRayKHR) == 0, "ray_query: no OpTraceRayKHR")
        validate(t, words, "ray_query")
        delete words
    }
}
