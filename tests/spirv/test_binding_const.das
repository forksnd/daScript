options gen2
options indenting = 4

// Named integer annotation args: `@set = NAME` / `@binding = NAME` / `local_size_x = NAME`, where NAME
// is a module-scope `let` int constant, must resolve to that constant's value. The annotation stores
// the identifier as a string arg, so a `?as tInt` read silently yields the fallback -- 0 for a binding,
// 1 for a workgroup size -- and the blob still compiles and still passes spirv-val. Only the emitted
// decoration / execution mode reveals it. This guards the const-name resolution in spirv_emit's
// annotation_int, for both the layout indices and the [compute_shader] local_size.

require dastest/testing_boost public
require _spirv_common
require spirv/spirv_dis
require spirv/spirv_grammar

[test]
def test_binding_name_resolves(t : T?) {
    t |> run("named @binding / @set resolve to the constant's value") <| @(t : T?) {
        var words <- bindconst_words()
        // @binding = BIND_CONST_SLOT (7) -> the Binding decoration must carry 7, not the 0 fallback.
        t |> success(op_has_operand_pair(words, SpvOp.Decorate, uint(SpvDecoration.Binding), 7u),
            "Binding decoration == 7 (from BIND_CONST_SLOT)")
        // @set = BIND_CONST_SET (2) -> the DescriptorSet decoration must carry 2.
        t |> success(op_has_operand_pair(words, SpvOp.Decorate, uint(SpvDecoration.DescriptorSet), 2u),
            "DescriptorSet decoration == 2 (from BIND_CONST_SET)")
        // The single descriptor is at binding 7 -- a stray Binding 0 means the name resolved to the fallback.
        t |> success(!op_has_operand_pair(words, SpvOp.Decorate, uint(SpvDecoration.Binding), 0u),
            "no fallback Binding 0")
        delete words
    }
    t |> run("named local_size_x resolves to the constant's value") <| @(t : T?) {
        var words <- bindconst_words()
        // OpExecutionMode operands: entry-point id(0), mode(1), then the mode's literals -- LocalSize
        // x(2), y(3), z(4). x must be 32 (from BIND_CONST_LOCAL_SIZE), never the silent 1 fallback.
        t |> success(op_has_operand_at(words, SpvOp.ExecutionMode, 1, uint(SpvExecutionMode.LocalSize)),
            "an ExecutionMode LocalSize is emitted")
        t |> success(op_has_operand_at(words, SpvOp.ExecutionMode, 2, 32u),
            "LocalSize x == 32 (from BIND_CONST_LOCAL_SIZE), not the 1 fallback")
        delete words
    }
}
