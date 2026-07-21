options gen2
options indenting = 4

// Descriptor-array-of-SSBOs gate.
//
// `array<array<T>>` under @ssbo is a runtime-sized DESCRIPTOR array of storage buffers (GLSL
// `buffer Block { T d[]; } g[];`): one buffer per slot, indexed g[slot][i]. The outer runtime array
// is over the Block struct and carries NO ArrayStride -- a descriptor slot has no byte layout --
// while the inner one strides std430 like any SSBO, and the whole access lowers to ONE
// OpAccessChain (var, slot, member 0, i), exactly as glslang chains it. RuntimeDescriptorArray rides
// along. Both the slot and the element index are runtime values, so nothing folds away.

require dastest/testing_boost public
require _spirv_common               // validate_spirv
require spirv/spirv_shader          // [compute_shader] annotation
require spirv/spirv_builtins public // gl_GlobalInvocationID
require spirv/spirv_dis             // count_op / op_has_operand / foreach_inst
require spirv/spirv_grammar         // SpvOp / SpvDecoration / SpvCapability
require spirv/spirv_reflect         // decode_reflection / SpirvDescriptorKind

// sa_out's element type differs from the buffers' on purpose: an identical one would DEDUP its
// runtime array (and its Block struct) with theirs, and the type counts below would prove nothing.
var @ssbo @binding = 0 sa_bufs : array<array<uint>>
var @ssbo @binding = 1 sa_out : array<float>

[compute_shader(local_size_x=64, name="sa_spv")]
def sa_comp {
    let i = int(gl_GlobalInvocationID.x)
    sa_bufs[i][i + 1] = 7u
    sa_out[i] = float(sa_bufs[i + 2][i])
}

// How many instructions with `op` carry exactly `n` operands (the opcode word excluded). An
// OpAccessChain into a descriptor array of SSBOs spends 6: result type, result id, the variable,
// the slot, the member index, the element index -- two more than a plain SSBO's (var, 0, i).
def private count_op_with_operands(words : array<uint>; op : SpvOp; n : int) : int {
    var c = 0
    foreach_inst(words) $(opcode : uint; _w : array<uint>; _s, oc : int) {
        if (opcode == uint(op) && oc == n) {
            c++
        }
    }
    return c
}

[test]
def test_ssbo_array(t : T?) {
    t |> run("array<array<T>> @ssbo -> descriptor array of SSBOs, one OpAccessChain per access") <| @(t : T?) {
        var words <- clone_to_move(sa_spv)
        // three runtime arrays: the uint element array inside the Block, the outer array OVER the
        // Block (the descriptor array), and the plain output SSBO's.
        t |> success(count_op(words, SpvOp.TypeRuntimeArray) == 3, "three OpTypeRuntimeArray")
        // ...but only the two ELEMENT arrays are laid out; the descriptor array has no byte stride.
        t |> success(count_op_operand_at(words, SpvOp.Decorate, 1, uint(SpvDecoration.ArrayStride)) == 2,
            "ArrayStride on the two element arrays only (none on the descriptor array)")
        t |> success(count_op_with_operands(words, SpvOp.AccessChain, 6) == 2,
            "both descriptor-array accesses are a single 4-index OpAccessChain")
        // The capability is operand 0 of OpCapability, so an id equal to the enum cannot match.
        t |> success(op_has_operand_at(words, SpvOp.Capability, 0, uint(SpvCapability.RuntimeDescriptorArray)),
            "OpCapability RuntimeDescriptorArray")
        // The host sizes the pool, so the binding reflects a variable descriptor count (0).
        let refl = decode_reflection(sa_spv_reflect)
        var found = false
        for (b in refl.bindings) {
            if (b.binding == 0) {
                found = true
                t |> success(b.kind == SpirvDescriptorKind.storage_buffer, "reflection: storage_buffer kind")
                t |> success(b.count == 0, "reflection: variable descriptor count")
            }
        }
        t |> success(found, "reflection: the descriptor array is reported at its binding")
        // the descriptor array raised the module to 1.5, so vulkan1.1 (SPIR-V 1.3) would reject the
        // binary on its VERSION alone -- validate against the environment it actually targets.
        let r = validate_spirv(words, "vulkan1.2")
        if (r.ran) {
            t |> success(r.ok, "spirv-val (vulkan1.2): {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
