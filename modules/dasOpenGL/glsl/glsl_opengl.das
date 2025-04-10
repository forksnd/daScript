options gen2
options indenting = 4
options no_unused_block_arguments = false
options no_unused_function_arguments = false
options strict_smart_pointers

module glsl_opengl shared public

require daslib/ast_boost
require daslib/strings_boost

require glsl/glsl_internal
require glsl/glsl_common public

class GlslOpenGLShader : GlslShader {
    def override generate_bind_uniform_dummy(var func : FunctionPtr) {
        generate_bind_uniform_dummy(func)
    }
    def override generate_bind_uniform(fnMain : FunctionPtr; var fn : FunctionPtr) {
        generate_bind_uniform(fnMain, fn)
    }
}

[function_macro(name="vertex_program")]
class GlslVertexProgram : GlslOpenGLShader {
    override shaderType = ShaderType.vertex
    override output_shader_lines = false
}

[function_macro(name="fragment_program")]
class GlslPixelProgram : GlslOpenGLShader {
    override shaderType = ShaderType.fragment
    override output_shader_lines = false
}

[function_macro(name="compute_program")]
class GlslComputeProgram : GlslOpenGLShader {
    override shaderType = ShaderType.compute
    override output_shader_lines = false
}

[macro_function]
def private generate_bind_uniform_dummy(fnMain : FunctionPtr) {
    var inscope fn <- new Function(at = fnMain.at, atDecl = fnMain.at, name := bind_uniform_function_name(fnMain))
    fn.flags |= FunctionFlags.generated
    fn.flags |= FunctionFlags.exports   // note: this is temporary, until we are done with dependency collecting etc
    move_new(fn.result) <| new TypeDecl(baseType = Type.tVoid, at = fnMain.at)
    fn.arguments |> emplace_new() <| new Variable(at = fnMain.at,
        name := "__program",
        _type <- new TypeDecl(at = fnMain.at, baseType = Type.tUInt)
    )
    move_new(fn.body) <| new ExprBlock(at = fnMain.at)
    if (!(compiling_module() |> add_function(fn))) {
        panic("can't setup")
    }
    return true
}

[macro_function]
def private generate_bind_uniform(fnMain : FunctionPtr; var fn : FunctionPtr) {
    var inscope blk <- new ExprBlock(at = fn.at)
    collect_dependencies(fnMain) <| $(vfun, vvar) {
        for (vv in vvar) {
            if (vv.annotation |> find_arg("uniform") is tBool) {
                if (is_glsl_structure(vv, "sampler2D")) {
                    // bind_sampler_2d(binding, varName.texture2D)
                    var binding = vv.annotation |> find_arg("stage") ?as tInt ?? 0
                    var inscope vtex <- new ExprField(at = vv.at,
                        value <- new ExprVar(at = vv.at, name := vv.name),
                        name := "texture2D"
                    )
                    var inscope cll_bind <- new ExprCall(at = vv.at, name := "bind_sampler_2d")
                    cll_bind.arguments |> emplace_new() <| new ExprConstInt(at = vv.at, value = binding)
                    cll_bind.arguments |> emplace(vtex)
                    blk.list |> emplace(cll_bind)
                    // glUniformAny(gl_GetUniformLocation("varName"),binding)
                    var inscope cll_gul <- new ExprCall(at = vv.at, name := "glGetUniformLocation")
                    cll_gul.arguments |> emplace_new() <| new ExprVar(at = vv.at, name := "__program")
                    cll_gul.arguments |> emplace_new() <| new ExprConstString(at = vv.at, value := vv.name)
                    var inscope cll_uni <- new ExprCall(at = vv.at, name := "glUniformAny")
                    cll_uni.arguments |> emplace(cll_gul)
                    cll_uni.arguments |> emplace_new() <| new ExprConstInt(at = vv.at, value = binding)
                    blk.list |> emplace(cll_uni)
                } elif (is_glsl_structure(vv, "sampler3D")) {
                    // bind_sampler_2d(binding, varName.texture2D)
                    var binding = vv.annotation |> find_arg("stage") ?as tInt ?? 0
                    var inscope vtex <- new ExprField(at = vv.at,
                        value <- new ExprVar(at = vv.at, name := vv.name),
                        name := "texture3D"
                    )
                    var inscope cll_bind <- new ExprCall(at = vv.at, name := "bind_sampler_3d")
                    cll_bind.arguments |> emplace_new() <| new ExprConstInt(at = vv.at, value = binding)
                    cll_bind.arguments |> emplace(vtex)
                    blk.list |> emplace(cll_bind)
                    // glUniformAny(gl_GetUniformLocation("varName"),binding)
                    var inscope cll_gul <- new ExprCall(at = vv.at, name := "glGetUniformLocation")
                    cll_gul.arguments |> emplace_new() <| new ExprVar(at = vv.at, name := "__program")
                    cll_gul.arguments |> emplace_new() <| new ExprConstString(at = vv.at, value := vv.name)
                    var inscope cll_uni <- new ExprCall(at = vv.at, name := "glUniformAny")
                    cll_uni.arguments |> emplace(cll_gul)
                    cll_uni.arguments |> emplace_new() <| new ExprConstInt(at = vv.at, value = binding)
                    blk.list |> emplace(cll_uni)
                } elif (is_glsl_structure(vv, "image2D") || is_glsl_structure(vv, "image3D")) {
                    // bind_sampler_any(binding, varName.texture?D, format, access)
                    let is2D = is_glsl_structure(vv, "image2D")
                    var binding = vv.annotation |> find_arg("stage") ?as tInt ?? 0
                    var inscope vtex <- new ExprField(at = vv.at,
                        value <- new ExprVar(at = vv.at, name := vv.name),
                        name := is2D ? "texture2D" : "texture3D"
                    )
                    var format = vv.annotation |> find_arg("format") ?as tString ?? "rgba8"
                    format = "GL_{to_upper(format)}"
                    var access = "GL_READ_WRITE"
                    if (vv.annotation |> find_arg("writeonly") is tBool) {
                        access = "GL_WRITE_ONLY"
                    }
                    if (vv.annotation |> find_arg("readonly") is tBool) {
                        access = "GL_READ_ONLY"
                    }
                    var inscope cll_uni <- new ExprCall(at = vv.at, name := "bind_image_any")
                    cll_uni.arguments |> emplace_new() <| new ExprConstInt(at = vv.at, value = binding)
                    cll_uni.arguments |> emplace(vtex)
                    cll_uni.arguments |> emplace_new() <| new ExprVar(at = vv.at, name := format)
                    cll_uni.arguments |> emplace_new() <| new ExprVar(at = vv.at, name := access)
                    blk.list |> emplace(cll_uni)
                } else {
                    // glUniformAny(gl_GetUniformLocation("varName"),varName)
                    var inscope cll_gul <- new ExprCall(at = vv.at, name := "glGetUniformLocation")
                    cll_gul.arguments |> emplace_new() <| new ExprVar(at = vv.at, name := "__program")
                    cll_gul.arguments |> emplace_new() <| new ExprConstString(at = vv.at, value := vv.name)
                    var inscope cll_uni <- new ExprCall(at = vv.at, name := "glUniformAny")
                    cll_uni.arguments |> emplace(cll_gul)
                    cll_uni.arguments |> emplace_new() <| new ExprVar(at = vv.at, name := vv.name)
                    blk.list |> emplace(cll_uni)
                }
            }
        }
    }
    fn.body := blk
    if (DEBUG_SHADER_BINDINGS) {
        print("{describe(fn)}\n")
    }
}
