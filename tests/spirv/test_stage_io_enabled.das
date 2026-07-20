options gen2
options indenting = 4

// Stage-I/O @enabled gate.
//
// A @in/@out global carrying @enabled = <compile-time int/bool const> is EXCLUDED from the SPIR-V
// interface when the const is 0/false -- the das mirror of GLSL wrapping a `layout(location = N)
// in/out ...` in `#if COND`.  The matching write is gated by `static_if (COND)` in the body (dropped
// before name resolution), so no reference to the excluded global survives.  Here v_off (@enabled =
// false, location 3) is excluded while v_main (location 0) and v_on (@enabled = true, location 4)
// remain; the blob is spirv-val clean either way.

require dastest/testing_boost public
require _spirv_common               // validate_spirv
require spirv/spirv_shader          // [vertex_shader] annotation
require spirv/spirv_builtins public // gl_Position
require spirv/spirv_dis             // op_has_operand_pair
require spirv/spirv_grammar         // SpvOp / SpvDecoration

let ENABLED_ON  = true
let ENABLED_OFF = false

var @out @location = 0 v_main : float3
var @out @location = 3 @enabled = ENABLED_OFF v_off : float3
var @out @location = 4 @enabled = ENABLED_ON  v_on  : float3

[vertex_shader(name="enabled_spv"), marker(no_coverage)]
def enabled_io {
    gl_Position = float4(1.0f, 2.0f, 3.0f, 1.0f)
    v_main = float3(1.0f, 1.0f, 1.0f)
    static_if (ENABLED_OFF) { v_off = float3(0.5f, 0.0f, 0.0f) }
    static_if (ENABLED_ON)  { v_on  = float3(0.0f, 0.5f, 0.0f) }
}

def private has_location(words : array<uint>; loc : uint) : bool {
    return op_has_operand_pair(words, SpvOp.Decorate,
        uint(SpvDecoration.Location), loc)
}

[test]
def test_stage_io_enabled(t : T?) {
    t |> run("@enabled = false excludes a stage output from the interface; spirv-val clean") <| @(t : T?) {
        var words <- clone_to_move(enabled_spv)
        t |> success(has_location(words, 0u), "location 0 (default-enabled) present")
        t |> success(has_location(words, 4u), "location 4 (@enabled = true) present")
        t |> success(!has_location(words, 3u), "location 3 (@enabled = false) EXCLUDED")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
