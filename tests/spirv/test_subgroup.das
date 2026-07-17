options gen2
options indenting = 4
options no_unused_block_arguments = false

// Subgroup ops (SPIR-V 1.3 GroupNonUniform*): opcode shapes — the Execution scope is a constant id
// (Subgroup = 3) and the reduce family's GroupOperation (Reduce = 0) is a LITERAL at operand index
// 3 — plus the per-kind capability pulls, the four builtin decorations, and spirv-val.

require dastest/testing_boost public
require _spirv_common
require spirv/spirv_dis
require spirv/spirv_grammar

[test]
def test_subgroup_ops(t : T?) {
    t |> run("reduce/shuffle/broadcast/elect shapes + caps + builtins; spirv-val clean") <| @(t : T?) {
        var words <- subgroup_words()
        // 8 reduce opcodes: IAdd twice (subgroupAdd int + uint share it), the rest exactly once
        t |> equal(count_op(words, SpvOp.GroupNonUniformIAdd), 2)
        for (op in [SpvOp.GroupNonUniformFAdd, SpvOp.GroupNonUniformFMin, SpvOp.GroupNonUniformFMax,
                    SpvOp.GroupNonUniformUMin, SpvOp.GroupNonUniformUMax,
                    SpvOp.GroupNonUniformSMin, SpvOp.GroupNonUniformSMax]) {
            t |> equal(count_op(words, op), 1)
        }
        // every reduce carries the Reduce(0) GroupOperation LITERAL at operand index 3
        for (op in [SpvOp.GroupNonUniformIAdd, SpvOp.GroupNonUniformFAdd,
                    SpvOp.GroupNonUniformFMin, SpvOp.GroupNonUniformFMax,
                    SpvOp.GroupNonUniformUMin, SpvOp.GroupNonUniformUMax,
                    SpvOp.GroupNonUniformSMin, SpvOp.GroupNonUniformSMax]) {
            t |> success(op_has_operand_at(words, op, 3, uint(SpvGroupOperation.Reduce)), "Reduce literal at operand 3")
        }
        // shuffle-xor + broadcast-first appear per operand class (f/u/i); elect once
        t |> equal(count_op(words, SpvOp.GroupNonUniformShuffleXor), 3)
        t |> equal(count_op(words, SpvOp.GroupNonUniformBroadcastFirst), 3)
        t |> equal(count_op(words, SpvOp.GroupNonUniformElect), 1)
        // the Subgroup(3) execution scope exists as an OpConstant (value at operand index 2)
        t |> success(op_has_operand_at(words, SpvOp.Constant, 2, uint(SpvScope.Subgroup)), "Subgroup scope constant present")
        // per-kind capability pulls: base + Arithmetic (reduces) + Shuffle (shuffleXor) + Ballot (broadcastFirst)
        for (cap in [SpvCapability.GroupNonUniform, SpvCapability.GroupNonUniformArithmetic,
                     SpvCapability.GroupNonUniformShuffle, SpvCapability.GroupNonUniformBallot]) {
            t |> success(op_has_operand_at(words, SpvOp.Capability, 0, uint(cap)), "capability {uint(cap)} present")
        }
        // the four subgroup builtins are BuiltIn-decorated Inputs
        for (bi in [SpvBuiltIn.SubgroupSize, SpvBuiltIn.SubgroupLocalInvocationId,
                    SpvBuiltIn.NumSubgroups, SpvBuiltIn.SubgroupId]) {
            t |> success(op_has_operand_pair(words, SpvOp.Decorate, uint(SpvDecoration.BuiltIn), uint(bi)), "BuiltIn {uint(bi)} decorated")
        }
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
