options gen2
options indenting = 4

// gl_WorkGroupSize gate: the one workgroup builtin SPIR-V spells as a CONSTANT rather than a
// dispatch Input. The emitter folds it to a uvec3 OpConstantComposite of the shader's own
// local_size and decorates THAT with BuiltIn WorkgroupSize -- so it takes no OpVariable and no
// entry-point interface entry, matching glslang. What it buys: a module shared by several shaders,
// each compiled with a different local_size, can read the size of whichever one is compiling it.

require dastest/testing_boost public
require _spirv_common
require spirv/spirv_dis
require spirv/spirv_grammar

// The id OpDecorate ... BuiltIn WorkgroupSize targets (0 = no such decoration).
def private workgroup_size_id(words : array<uint>) : uint {
    var target = 0u
    foreach_inst(words) $(opcode : uint; w : array<uint>; s, n : int) {
        if (opcode == uint(SpvOp.Decorate) && n >= 3 && w[s + 1] == uint(SpvDecoration.BuiltIn) &&
            w[s + 2] == uint(SpvBuiltIn.WorkgroupSize)) {
            target = w[s]
        }
    }
    return target
}

// Is `id` the result of an instruction with `op` whose result-id sits at operand `idx`?
def private is_result_of(words : array<uint>; op : SpvOp; idx : int; id : uint) : bool {
    var found = false
    foreach_inst(words) $(opcode : uint; w : array<uint>; s, n : int) {
        if (opcode == uint(op) && idx < n && w[s + idx] == id) {
            found = true
        }
    }
    return found
}

// Is `id` one of the entry point's interface ids? OpEntryPoint operands: model(0), entry-id(1),
// the name (a NUL-padded literal string -- "main" is exactly 2 words), then the interface ids.
def private entry_interface_has(words : array<uint>; id : uint) : bool {
    var found = false
    foreach_inst(words) $(opcode : uint; w : array<uint>; s, n : int) {
        if (opcode == uint(SpvOp.EntryPoint)) {
            for (k in range(4, n)) {
                if (w[s + k] == id) {
                    found = true
                }
            }
        }
    }
    return found
}

[test]
def test_workgroup_size_is_a_decorated_constant(t : T?) {
    t |> run("compute: gl_WorkGroupSize -> BuiltIn WorkgroupSize on an OpConstantComposite") <| @(t : T?) {
        var words <- wgsize_words()
        let model = op_first_operand(words, SpvOp.EntryPoint)
        t |> success(model.ok && model.value == uint(SpvExecutionModel.GLCompute),
            "wgsize: EntryPoint model is GLCompute")
        // ExecutionMode operands: entry-point(0), mode(1), x(2), y(3), z(4).
        t |> success(op_has_operand_pair(words, SpvOp.ExecutionMode,
            uint(SpvExecutionMode.LocalSize), 64u), "wgsize: LocalSize is the declared 64")
        let wgs = workgroup_size_id(words)
        t |> success(wgs != 0u, "wgsize: a BuiltIn WorkgroupSize decoration is emitted")
        // OpConstantComposite operands: result-type(0), result-id(1), constituents(2..).
        t |> success(is_result_of(words, SpvOp.ConstantComposite, 1, wgs),
            "wgsize: the decorated id is an OpConstantComposite")
        // OpVariable operands: result-type(0), result-id(1), storage-class(2). A constant is not a
        // variable, so it must appear in NO OpVariable and in no entry-point interface entry.
        t |> success(!is_result_of(words, SpvOp.Variable, 1, wgs),
            "wgsize: the decorated id is not an OpVariable")
        t |> success(!entry_interface_has(words, wgs),
            "wgsize: the constant is not listed in the entry-point interface")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "wgsize: spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
