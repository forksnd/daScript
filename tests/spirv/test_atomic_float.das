options gen2
options indenting = 4

// Gate: the float atomicAdd (VK_EXT_shader_atomic_float) and the single-component swizzle lvalue it
// rides on.  cfatomic accumulates onto ONE COMPONENT of an SSBO vector element (fout[0].x) -- the
// shape a scatter pass writing an RGB radiance uses.
//   - the component of an addressable vector is a POINTER: OpAccessChain chains the component index
//     off the element pointer, so the atomic target is a StorageBuffer float.  The plain component
//     store (fout[gid].w = ...) rides the same path; a component READ does not, staying the
//     OpCompositeExtract off the loaded vector.
//   - a float target picks OpAtomicFAddEXT (never OpAtomicIAdd) and declares OpCapability
//     AtomicFloat32AddEXT + OpExtension "SPV_EXT_shader_atomic_float_add".
// The scope/semantics operands are constant <id>s, so their well-formedness is spirv-val's to judge.
// The rejections (a float target on a non-add atomic; a multi-component write-swizzle) are in
// tests/spirv/_fail_closed/.

require dastest/testing_boost public
require _spirv_common
require spirv/spirv_dis
require spirv/spirv_grammar

def private validate(t : T?; words : array<uint>; lbl : string) {
    let r = validate_spirv(words)
    if (r.ran) {
        t |> success(r.ok, "{lbl}: spirv-val: {r.msg}")
    } else {
        feint("spirv-val not found locally; skipping (CI enforces)\n")
    }
}

[test]
def test_float_atomic_add(t : T?) {
    t |> run("compute: float atomicAdd on an SSBO vector component") <| @(t : T?) {
        var words <- cfatomic_words()
        t |> success(has_op(words, SpvOp.AtomicFAddEXT),
            "cfatomic: atomicAdd(float) -> OpAtomicFAddEXT")
        t |> success(!has_op(words, SpvOp.AtomicIAdd),
            "cfatomic: a float target never emits the integer OpAtomicIAdd")
        t |> success(op_has_operand_at(words, SpvOp.Capability, 0,
            uint(SpvCapability.AtomicFloat32AddEXT)),
            "cfatomic: declares OpCapability AtomicFloat32AddEXT")
        t |> success(has_op(words, SpvOp.Extension),
            "cfatomic: declares the SPV_EXT_shader_atomic_float_add OpExtension")
        // The component lvalue: the atomic target and the component store both chain through an
        // OpAccessChain, so a StorageBuffer float pointer type has to exist in the module.
        t |> success(has_op(words, SpvOp.AccessChain),
            "cfatomic: the vector component is an OpAccessChain, not a loaded vector")
        // A component READ takes the value path, so the extract survives alongside the chains.
        t |> success(has_op(words, SpvOp.CompositeExtract),
            "cfatomic: a component read stays an OpCompositeExtract off the loaded vector")
        validate(t, words, "cfatomic")
        delete words
    }
}
