options gen2
options indenting = 4

// Matrix column constructor gate.
//
// GLSL's `mat4(col0, col1, col2, col3)` builds a matrix from its columns. daslang matrices are
// column-major (m[i] is column i), so float4x4(c0, c1, c2, c3) lowers to a single OpCompositeConstruct
// whose operands are the four column vectors -- matching glslang's `mat4(cols)`. The columns are
// LOADED from an SSBO (not constructed), so the ONLY OpCompositeConstruct in the blob is the matrix
// itself; the count is exact. The matrix is then multiplied by a vector (OpMatrixTimesVector), and
// the whole module is spirv-val clean.

require dastest/testing_boost public
require _spirv_common               // validate_spirv
require spirv/spirv_shader          // [vertex_shader] annotation
require spirv/spirv_builtins public // gl_Position + the float4x4 column ctor
require spirv/spirv_dis             // count_op
require spirv/spirv_grammar         // SpvOp
require math

var @ssbo @binding = 0 mc_cols : array<float4>

[vertex_shader(name="mctor_spv"), marker(no_coverage)]
def mctor {
    let m = float4x4(mc_cols[0], mc_cols[1], mc_cols[2], mc_cols[3])
    gl_Position = m * float4(1.0f, 2.0f, 3.0f, 1.0f)
}

// float3x3(col0..col2) is the same column form one width down (GLSL's `mat3(cols)`), and the columns
// are read back out with m[i] -- a matrix is an emittable VALUE, so a column read is an
// OpCompositeExtract off it, not an OpAccessChain through a variable that does not exist. mat3 * vec3
// is the width-3 OpMatrixTimesVector.
[vertex_shader(name="mctor3_spv"), marker(no_coverage)]
def mctor3 {
    let m = float3x3(mc_cols[0].xyz, mc_cols[1].xyz, mc_cols[2].xyz)
    let v = m * mc_cols[3].xyz
    gl_Position = float4(m[0] + m[1] + m[2] + v, 1.0f)
}

// A MUTABLE matrix local: a matrix is a fine Function-storage pointee, so `var m : float4x4` gets its
// own OpVariable (OpLoad/OpStore on assignment) exactly as a local `mat4` does in glslang -- which is
// what an accumulate-a-product chain needs. daslang declares no `*=` over matrices, so the
// accumulation is spelled `tm = tm * m` (one OpMatrixTimesMatrix).
[vertex_shader(name="mvarloc_spv"), marker(no_coverage)]
def mvarloc {
    var tm = float4x4(mc_cols[0], mc_cols[1], mc_cols[2], mc_cols[3])
    tm = tm * float4x4(mc_cols[3], mc_cols[2], mc_cols[1], mc_cols[0])
    gl_Position = tm[3]
}

// A module-scope `let` matrix of constant columns folds to an OpConstantComposite of the four column
// constants -- no OpVariable, no OpLoad -- which is what glslang folds a `const mat4` global to. The
// row-vector form `v * m` is GLSL's `vec4 * mat4`, lowering to the one OpVectorTimesMatrix; the
// column form `m * v` beside it stays OpMatrixTimesVector, so the two spellings are not conflated.
let MC_BASIS = float4x4(
    float4(-0.16666667f,  0.5f, -0.5f, 0.16666667f),
    float4(0.5f, -1.0f,  0.0f, 0.6666667f),
    float4(-0.5f,  0.5f,  0.5f, 0.16666667f),
    float4(0.16666667f,  0.0f,  0.0f, 0.0f))

[vertex_shader(name="mvtm_spv"), marker(no_coverage)]
def mvtm {
    gl_Position = (mc_cols[0] * MC_BASIS) + (MC_BASIS * mc_cols[1])
}

// The width-3 row-vector form (GLSL's `vec3 * mat3`) -- same opcode, one width down.
[vertex_shader(name="mvtm3_spv"), marker(no_coverage)]
def mvtm3 {
    let m = float3x3(mc_cols[0].xyz, mc_cols[1].xyz, mc_cols[2].xyz)
    gl_Position = float4(mc_cols[3].xyz * m, 1.0f)
}

[test]
def test_matrix_ctor(t : T?) {
    t |> run("float4x4(col0..col3) -> one OpCompositeConstruct (matrix) + OpMatrixTimesVector; spirv-val clean") <| @(t : T?) {
        var words <- clone_to_move(mctor_spv)
        t |> success(count_op(words, SpvOp.TypeMatrix) == 1, "one OpTypeMatrix (float4x4)")
        // Columns are loaded from the SSBO, so the matrix is the only OpCompositeConstruct.
        t |> success(count_op(words, SpvOp.CompositeConstruct) == 1, "one OpCompositeConstruct == the matrix from its columns")
        t |> success(count_op(words, SpvOp.MatrixTimesVector) == 1, "one OpMatrixTimesVector (m * v)")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}

[test]
def test_matrix_var_local(t : T?) {
    t |> run("`var` matrix local -> Function-storage OpVariable + OpMatrixTimesMatrix on `*=`; spirv-val clean") <| @(t : T?) {
        var words <- clone_to_move(mvarloc_spv)
        t |> success(op_has_operand_at(words, SpvOp.Variable, 2, uint(SpvStorageClass.Function)),
            "the matrix local gets a Function-storage OpVariable")
        t |> success(count_op(words, SpvOp.MatrixTimesMatrix) == 1, "one OpMatrixTimesMatrix (tm *= m)")
        t |> success(count_op(words, SpvOp.Store) >= 2, "the local is stored into (init + compound assign)")
        t |> success(count_op(words, SpvOp.CompositeExtract) == 1, "one OpCompositeExtract == the tm[3] column read")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}

[test]
def test_matrix_ctor3(t : T?) {
    t |> run("float3x3(col0..col2) + m[i] column reads -> OpCompositeConstruct + OpCompositeExtract; spirv-val clean") <| @(t : T?) {
        var words <- clone_to_move(mctor3_spv)
        t |> success(count_op(words, SpvOp.TypeMatrix) == 1, "one OpTypeMatrix (float3x3)")
        // the three columns are swizzled out of the SSBO (OpVectorShuffle), so the only composite
        // CONSTRUCTs are the matrix itself and the float4 fed to gl_Position
        t |> success(count_op(words, SpvOp.CompositeConstruct) == 2, "two OpCompositeConstruct == the matrix from its columns + the gl_Position float4")
        t |> success(count_op(words, SpvOp.MatrixTimesVector) == 1, "one OpMatrixTimesVector (m * v)")
        // m[0], m[1], m[2] -- one extract per column read, off the matrix value
        t |> success(count_op(words, SpvOp.CompositeExtract) == 3, "three OpCompositeExtract == the three m[i] column reads")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}

[test]
def test_vector_times_matrix(t : T?) {
    t |> run("v * m -> OpVectorTimesMatrix; a const-column module `let` matrix -> OpConstantComposite; spirv-val clean") <| @(t : T?) {
        var words <- clone_to_move(mvtm_spv)
        t |> success(count_op(words, SpvOp.VectorTimesMatrix) == 1, "one OpVectorTimesMatrix (v * m)")
        t |> success(count_op(words, SpvOp.MatrixTimesVector) == 1, "one OpMatrixTimesVector (m * v) -- the two forms are distinct")
        // the constant matrix is folded, so the only OpVariables are the SSBO and gl_Position --
        // MC_BASIS gets no storage of its own
        t |> success(count_op(words, SpvOp.Variable) == 2, "the const matrix is a constant, not a third OpVariable")
        // its four constant columns plus the matrix itself
        t |> success(count_op(words, SpvOp.ConstantComposite) == 5, "five OpConstantComposite == the four columns + the matrix")
        t |> success(count_op(words, SpvOp.CompositeConstruct) == 0, "the matrix is folded, not constructed at run time")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}

[test]
def test_vector_times_matrix3(t : T?) {
    t |> run("float3 * float3x3 -> the width-3 OpVectorTimesMatrix; spirv-val clean") <| @(t : T?) {
        var words <- clone_to_move(mvtm3_spv)
        t |> success(count_op(words, SpvOp.VectorTimesMatrix) == 1, "one OpVectorTimesMatrix (v * m)")
        t |> success(count_op(words, SpvOp.MatrixTimesVector) == 0, "no OpMatrixTimesVector")
        let r = validate_spirv(words)
        if (r.ran) {
            t |> success(r.ok, "spirv-val: {r.msg}")
        } else {
            feint("spirv-val not found locally; skipping (CI enforces)\n")
        }
        delete words
    }
}
