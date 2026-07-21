// Fail-closed fixture: a @readonly storage image is NonWritable, so imageStore into it must be
// rejected at the call site -- the mirror of the @writeonly / imageLoad rejection.  Never drop a GLSL
// access qualifier on a port, and never emit an OpImageWrite against a NonWritable variable.
// `_`-prefix + `expect 50501` keep dastest/lint off the failing shader.
expect 50501

options gen2

require spirv/spirv_shader
require spirv/spirv_builtins

var @binding = 0 @format = "rgba32f" @readonly fc_ro_img : image2DArray

[compute_shader(local_size_x=64, name="fc_image_readonly_store_spv")]
def fc_image_readonly_store {
    let i = gl_GlobalInvocationID.x
    imageStore(fc_ro_img, int3(int(i), 0, 0), float4(1.0f, 0.0f, 0.0f, 1.0f))
}
