options gen2
options indenting = 4

// Vector modulo gate: `uvec3 % uvec3` lowers to ONE native OpUMod on v3uint.
//
// GLSL's `%` is closed over the integer vector families; daslang declares it only on scalars, so
// shader_lingua_franca declares operator %(uint3, uint3). The emitter needed nothing -- binop_code
// already keys `%` off the operand's class -- but the lowering has to be PINNED BY RESULT TYPE, not
// by opcode count alone: three scalar OpUMods and one vector OpUMod are told apart by the width of
// the result, and only the vector form matches what glslang gives a uvec3 modulo.
//
// The signed twin is deliberately NOT declared: daslang's scalar `%` takes the dividend's sign
// (OpSRem) where GLSL's takes the divisor's (OpSMod), so a signed vector `%` would lower to an
// opcode that disagrees with the glslang reference on a negative dividend. This gate pins zero of both.

require dastest/testing_boost public
require _spirv_common              // validate_spirv
require spirv/spirv_shader         // [compute_shader] annotation
require spirv/spirv_builtins public // gl_GlobalInvocationID
require spirv/spirv_dis            // count_op, foreach_inst
require spirv/spirv_grammar        // SpvOp

var @ssbo @binding = 0 vmod_uv : array<uint4>
[compute_shader(local_size_x=64, name="vmod_spv"), marker(no_coverage)]
def vmod {
    let i = gl_GlobalInvocationID.x
    let a = vmod_uv[i].xyz
    let m = uint3(vmod_uv[i].w)     // the vector-scalar form, splatted as glslang splats it
    vmod_uv[i] = uint4(a % m, 0u)
}

// Every OpUMod's result must be a 3-component vector of 32-bit UNSIGNED int. A lane-by-lane
// lowering would report a scalar result here, which is exactly what this rules out.
def private all_umods_are_v3uint(words : array<uint>) : bool {
    var int_info : table<uint; uint2>       // id -> (width, signedness)
    var vec_info : table<uint; uint2>       // id -> (component type, count)
    foreach_inst(words) $(opcode : uint; w : array<uint>; s, n : int) {
        if (opcode == uint(SpvOp.TypeInt) && n >= 3) {
            int_info[w[s]] = uint2(w[s + 1], w[s + 2])
        } elif (opcode == uint(SpvOp.TypeVector) && n >= 3) {
            vec_info[w[s]] = uint2(w[s + 1], w[s + 2])
        }
    }
    var all_v3uint = true
    var seen = 0
    foreach_inst(words) $(opcode : uint; w : array<uint>; s, n : int) {
        if (opcode == uint(SpvOp.UMod) && n >= 1) {
            seen++
            let v = vec_info?[w[s]] ?? uint2(0u, 0u)
            let c = int_info?[v.x] ?? uint2(0u, 9u)
            if (v.y != 3u || c.x != 32u || c.y != 0u) {
                all_v3uint = false
            }
        }
    }
    return all_v3uint && seen == 1
}

[test]
def test_vec_mod(t : T?) {
    t |> run("uvec3 % uvec3 is one native OpUMod on v3uint; spirv-val clean") <| @(t : T?) {
        var words <- clone_to_move(vmod_spv)
        t |> success(count_op(words, SpvOp.UMod) == 1, "one OpUMod (not three scalar ones)")
        t |> success(all_umods_are_v3uint(words), "the OpUMod's result type is v3uint")
        t |> success(count_op(words, SpvOp.SRem) == 0, "no OpSRem (unsigned operands)")
        t |> success(count_op(words, SpvOp.SMod) == 0, "no OpSMod (unsigned operands)")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
