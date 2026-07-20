options gen2
options indenting = 4

// Struct LOCALS: a mutable `var s : S` gets a Function-storage OpVariable, an OpConstantNull zero-init
// and one OpAccessChain per field touched -- glslang's shape for a local struct, and the exact opposite
// of the value path (test_local_struct_member, where a `let` copy of an SSBO element extracts instead).
// A struct written through by a callee rides the same rail: das gives a struct parameter no flags.ref,
// so `var o : SlOut&` differs from a read-only `c : SlCell` in the CONST flag alone, and it lowers to a
// Function POINTER parameter fed by the call site's copy-in/copy-out temp.
//
// spirv-val is the load-bearing oracle here, not the counts: an OpAccessChain whose result storage class
// disagrees with its base is well-formed to the emitter and rejected only by the validator.

require dastest/testing_boost public
require _spirv_common              // validate_spirv
require spirv/spirv_shader         // [compute_shader] annotation
require spirv/spirv_builtins public // gl_GlobalInvocationID
require spirv/spirv_dis            // has_op / count_op
require spirv/spirv_grammar        // SpvOp

struct SlCell {
    n : float4
    k : float
}

struct SlOut {
    v : float4
    w : float
}

struct SlArr {
    arr : float[4]
    tail : float
}

var @ssbo @binding = 0 sl_data : array<float4>

def private sl_fill(c : SlCell; var o : SlOut&) {
    o.v = c.n * c.k
    o.w = c.k
}

[compute_shader(local_size_x=64, name="slocal_spv"), marker(no_coverage)]
def slocal {
    let i = int(gl_GlobalInvocationID.x)
    var c : SlCell // nolint:STYLE013 -- the zero-init is the point (asserts OpConstantNull)
    c.n = sl_data[i]
    c.k = 2.0f
    var o : SlOut
    sl_fill(c, o)
    sl_data[i] = o.v + float4(o.w, 0.0f, 0.0f, 0.0f)
}

// A fixed-array field of a struct local: the array is chained off the local's Function pointer, so its
// OpTypeArray carries NO ArrayStride (a Block's would) and the element pointer stays Function-storage.
[compute_shader(local_size_x=64, name="slarr_spv"), marker(no_coverage)]
def slarr {
    let i = int(gl_GlobalInvocationID.x)
    var s : SlArr
    s.arr[2] = float(i)
    s.tail = 3.0f
    sl_data[i] = float4(s.arr[2], s.tail, 0.0f, 0.0f)
}

def private validate(t : T?; words : array<uint>; lbl : string) {
    let r = validate_spirv(words)
    if (r.ran) {
        t |> success(r.ok, "{lbl}: spirv-val: {r.msg}")
    } else {
        feint("spirv-val not found locally; skipping (CI enforces)\n")
    }
}

[test]
def test_struct_local(t : T?) {
    t |> run("struct local: Function OpVariable + zero-init + access-chained fields, and an out-param") <| @(t : T?) {
        var words <- clone_to_move(slocal_spv)
        // entry + sl_fill = 2 OpFunctions, called once.
        t |> equal(count_op(words, SpvOp.Function), 2)
        t |> equal(count_op(words, SpvOp.FunctionCall), 1)
        // sl_fill(c, o) -> 2 OpFunctionParameter: `c` by value, `o` as a Function pointer.
        t |> equal(count_op(words, SpvOp.FunctionParameter), 2)
        // SlCell + SlOut + the sl_data SSBO's Block = 3 OpTypeStruct.
        t |> equal(count_op(words, SpvOp.TypeStruct), 3)
        // THE discriminating assertion: exactly ONE OpMemberDecorate -- the SSBO Block's Offset. A
        // struct local is not an interface block, so neither SlCell nor SlOut may carry a member
        // decoration; a layout-decorated type here would mean the local took the Block path.
        t |> equal(count_op(words, SpvOp.MemberDecorate), 1)
        // both `var` struct locals zero-init from the das-guaranteed zero -> one OpConstantNull each.
        t |> equal(count_op(words, SpvOp.ConstantNull), 2)
        // the locals are MEMORY: fields are chained, not extracted off a loaded composite.
        t |> success(has_op(words, SpvOp.AccessChain), "slocal: struct-local fields are access-chained")
        validate(t, words, "slocal")
        delete words
    }
    t |> run("struct local: a fixed-array field chains in Function storage (no ArrayStride)") <| @(t : T?) {
        var words <- clone_to_move(slarr_spv)
        // SlArr + the SSBO Block = 2 OpTypeStruct; the Block's Offset is again the only member decoration.
        t |> equal(count_op(words, SpvOp.TypeStruct), 2)
        t |> equal(count_op(words, SpvOp.MemberDecorate), 1)
        t |> equal(count_op(words, SpvOp.ConstantNull), 1)
        t |> success(has_op(words, SpvOp.TypeArray), "slarr: the fixed-array field is an OpTypeArray")
        validate(t, words, "slarr")
        delete words
    }
}
