// Fail-closed fixture: a write-swizzle whose BASE is not addressable. `fc_wswz_out.xyz` is itself a
// swizzle, so it lowers to a loaded VALUE with no OpVariable to chain a component pointer off -- there
// is nothing to store through. daslang's typer accepts the nested form (both masks are contiguous, so
// both are references), which is why the emitter has to reject it. The write-swizzle of an addressable
// base -- a global, a `var` local, an SSBO element -- ships. `_`-prefix + `expect 50501` keep
// dastest/lint off.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @out @location = 0 fc_wswz_out : float4

[fragment_shader(name="fc_wswz_spv")]
def fc_wswz {
    fc_wswz_out = float4(0.0, 0.0, 0.0, 1.0)
    fc_wswz_out.xyz.xy = float2(1.0, 0.0)   // swizzle of a swizzle -> no base pointer
}
