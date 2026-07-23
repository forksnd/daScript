options gen2
options indenting = 4

module dasllama_metal_lens shared private

require daslib/ast
require daslib/ast_boost
require daslib/templates_boost
require daslib/rtti
require strings
require daslib/strings_boost   // split / strip — the grid & params micro-format tokenizers
require das_metal   // nolint:STYLE030 — MetalBuffer/MetalComputeEncoder are referenced only inside qmacro_type(type<...>), which STYLE030 can't see; the require is load-bearing for macro-time type resolution

// The dasLLAMA Metal kernel-dispatch lens (v1). [metal_dispatch] reads a [metal_kernel] class's
// @ssbo/@uniform/@role field annotations and generates the enc_* dispatch builder that the hand-
// written twin used to be. v1 covers exactly one shape (single buffers at offset 0, constant tg,
// grid=="1") — the seed the lens grows from; anything else fails loudly.

struct private BufField {
    name : string
    binding : int
    role : string
}

def private mk_kn_pipeline(at : LineInfo; pso : string) : ExpressionPtr {
    var c = new ExprCall(at = at, name := "kn_pipeline")
    c.arguments |> emplace_new <| new ExprVar(at = at, name := "enc")
    c.arguments |> emplace_new <| new ExprVar(at = at, name := pso)
    return c
}

def private mk_kn_tgmem(at : LineInfo; tgmem : string) : ExpressionPtr {
    var c = new ExprCall(at = at, name := "kn_tgmem")
    c.arguments |> emplace_new <| new ExprVar(at = at, name := "enc")
    c.arguments |> emplace_new <| new ExprVar(at = at, name := tgmem)
    c.arguments |> emplace_new <| new ExprConstInt(at = at, value = 0)
    return c
}

def private mk_kn_buffer(at : LineInfo; pname : string; binding : int) : ExpressionPtr {
    var c = new ExprCall(at = at, name := "kn_buffer")
    c.arguments |> emplace_new <| new ExprVar(at = at, name := "enc")
    c.arguments |> emplace_new <| new ExprVar(at = at, name := pname)
    c.arguments |> emplace_new <| new ExprConstUInt64(at = at, value = 0ul)
    c.arguments |> emplace_new <| new ExprConstInt(at = at, value = binding)
    return c
}

def private mk_var_call(at : LineInfo; call_name, arg0 : string) : ExpressionPtr {
    var c = new ExprCall(at = at, name := call_name)
    c.arguments |> emplace_new <| new ExprVar(at = at, name := arg0)
    return c
}

def private mk_uint_cast(at : LineInfo; var inner : Expression?) : Expression? {
    var c = new ExprCall(at = at, name := "uint")
    c.arguments |> emplace_new <| inner
    return c
}

def private is_digit_tok(s : string) : bool {
    if (empty(s)) {
        return false
    }
    return is_number(first_character(s))
}

// A coverage token is either an int literal (e.g. "32") or a global/param name (e.g. "g_gemv_rows").
def private mk_cov(at : LineInfo; tok : string) : Expression? {
    if (is_digit_tok(tok)) {
        return new ExprConstInt64(at = at, value = int64(to_int(tok)))
    }
    return new ExprVar(at = at, name := tok)
}

// One grid dim: "1" -> 1u ; "name" -> uint(name) ; "name/cov" -> uint((name + cov - 1) / cov) (ceil).
def private mk_grid_dim(at : LineInfo; spec : string) : Expression? {
    let s = strip(spec)
    if (s == "1") {
        return new ExprConstUInt(at = at, value = 1u)
    }
    let parts <- split(s, "/")
    if (length(parts) == 1) {
        return mk_uint_cast(at, new ExprVar(at = at, name := s))
    }
    let name = strip(parts[0])
    let cov = strip(parts[1])
    // ceil((name + cov - 1) / cov) — cov appears twice, so build two independent nodes
    var add = new ExprOp2(at = at, op := "+", left = new ExprVar(at = at, name := name), right = mk_cov(at, cov))
    var sub = new ExprOp2(at = at, op := "-", left = add, right = new ExprConstInt64(at = at, value = 1l))
    var div = new ExprOp2(at = at, op := "/", left = sub, right = mk_cov(at, cov))
    return mk_uint_cast(at, div)
}

// grid = comma-separated per WorkGroupID dim; padded to uint3 with trailing 1u.
def private mk_grid(at : LineInfo; grid : string) : Expression? {
    let dims <- split(grid, ",")
    var c = new ExprCall(at = at, name := "uint3")
    for (i in range(3)) {
        if (i < length(dims)) {
            c.arguments |> emplace_new <| mk_grid_dim(at, dims[i])
        } else {
            c.arguments |> emplace_new <| new ExprConstUInt(at = at, value = 1u)
        }
    }
    return c
}

// tg = threads-per-threadgroup (a const, all dims): uint3(tg, 1, 1).
def private mk_tg(at : LineInfo; tg : int) : Expression? {
    var c = new ExprCall(at = at, name := "uint3")
    c.arguments |> emplace_new <| new ExprConstUInt(at = at, value = uint(tg))
    c.arguments |> emplace_new <| new ExprConstUInt(at = at, value = 1u)
    c.arguments |> emplace_new <| new ExprConstUInt(at = at, value = 1u)
    return c
}

def private mk_kn_dispatch(at : LineInfo; grid : string; tg : int) : Expression? {
    var c = new ExprCall(at = at, name := "kn_dispatch")
    c.arguments |> emplace_new <| new ExprVar(at = at, name := "enc")
    c.arguments |> emplace_new <| mk_grid(at, grid)
    c.arguments |> emplace_new <| mk_tg(at, tg)
    return c
}

// map a params-string type token to the param TypeDecl (the size/offset scalars the grid references)
// a generated builder param is non-`var`, so its type carries the trailing const (the const model)
def private const_t(var t : TypeDecl?) : TypeDecl? {
    t.flags.constant = true
    return t
}

def private param_type(tok : string; var ok : bool&) : TypeDecl? {
    if (tok == "int64") { return qmacro_type(type<int64>) }
    if (tok == "uint64") { return qmacro_type(type<uint64>) }
    if (tok == "int") { return qmacro_type(type<int>) }
    if (tok == "uint") { return qmacro_type(type<uint>) }
    ok = false
    return qmacro_type(type<int64>)
}

[structure_macro(name = metal_dispatch)]
class MetalDispatch : AstStructureAnnotation {
    def override apply(var st : StructurePtr; var group : ModuleGroup; args : AnnotationArgumentList; var errors : das_string) : bool {
        let at = st.at
        let fn_name = find_arg(args, "name") ?as tString ?? ""
        let pso = find_arg(args, "pso") ?as tString ?? ""
        let tgmem = find_arg(args, "tgmem") ?as tString ?? ""
        let tg = find_arg(args, "tg") ?as tInt ?? -1
        let grid = find_arg(args, "grid") ?as tString ?? ""

        if (empty(fn_name)) {
            errors := "[metal_dispatch]: 'name' (the enc_ builder name) is required"
            return false
        }
        if (empty(pso)) {
            errors := "[metal_dispatch]: 'pso' (the pipeline global name) is required"
            return false
        }
        if (tg <= 0) {
            errors := "[metal_dispatch] {fn_name}: 'tg' must be a positive int constant (v1 only supports a constant threadgroup size)"
            return false
        }
        if (empty(grid)) {
            errors := "[metal_dispatch] {fn_name}: 'grid' is required (per WorkGroupID dim: 1 | param | param/coverage)"
            return false
        }

        // Collect @ssbo/@uniform buffer fields in @binding order. @workgroup + hidden/method
        // fields are skipped; field-annotation validity is msl_emit's job, not the lens's.
        var bufs : array<BufField>
        bufs |> reserve(length(st.fields))
        for (field in st.fields) {
            var is_buf = false
            var is_wg = false
            for (annDecl in field.annotation) {
                if (annDecl.name == "ssbo" || annDecl.name == "uniform") {
                    is_buf = true
                } elif (annDecl.name == "workgroup") {
                    is_wg = true
                }
            }
            if (is_wg || !is_buf) continue
            let binding = find_arg(field.annotation, "binding") ?as tInt ?? -1
            if (binding < 0) {
                errors := "[metal_dispatch] {fn_name}: field '{field.name}' is @ssbo/@uniform but has no @binding"
                return false
            }
            let role = find_arg(field.annotation, "role") ?as tString ?? ""
            bufs |> push(BufField(name = "{field.name}", binding = binding, role = role))
        }
        sort(bufs) $(a, b) => a.binding < b.binding

        // Emit the dispatch sequence, mirroring the hand-written enc_ builder exactly.
        var body : array<ExpressionPtr>
        body |> reserve(length(bufs) * 3 + 4)
        body |> push(mk_kn_pipeline(at, pso))
        if (!empty(tgmem)) {
            body |> push(mk_kn_tgmem(at, tgmem))
        }
        for (bf in bufs) {
            body |> push(mk_kn_buffer(at, "b{bf.name}", bf.binding))
        }
        for (bf in bufs) {
            if (bf.role == "read" || bf.role == "readwrite") {
                body |> push(mk_var_call(at, "hz_read", "b{bf.name}"))
            }
            if (bf.role == "write" || bf.role == "readwrite") {
                body |> push(mk_var_call(at, "hz_write", "b{bf.name}"))
            }
        }
        body |> push(mk_var_call(at, "hz_gate", "enc"))
        body |> push(mk_kn_dispatch(at, grid, tg))

        // Build the builder: void return, enc + one MetalBuffer? per @ssbo/@uniform field + the
        // declared size/offset params (int64/uint64/... — the scalars the grid references).
        var fn = new Function(at = at, atDecl = at, name := fn_name,
            flags = FunctionFlags.generated,
            result = new TypeDecl(at = at, baseType = Type.tVoid))
        fn.arguments |> emplace_new <| new Variable(at = at, name := "enc", _type = const_t(qmacro_type(type<MetalComputeEncoder?>)))
        for (bf in bufs) {
            fn.arguments |> emplace_new <| new Variable(at = at, name := "b{bf.name}", _type = const_t(qmacro_type(type<MetalBuffer?>)))
        }
        let params = find_arg(args, "params") ?as tString ?? ""
        if (!empty(strip(params))) {
            for (p in split(params, ",")) {
                let nt <- split(p, ":")
                if (length(nt) != 2) {
                    errors := "[metal_dispatch] {fn_name}: param '{strip(p)}' must be 'name : type'"
                    return false
                }
                var ok = true
                var pt = param_type(strip(nt[1]), ok)
                if (!ok) {
                    errors := "[metal_dispatch] {fn_name}: param '{strip(p)}' has unsupported type '{strip(nt[1])}' (int/uint/int64/uint64)"
                    return false
                }
                fn.arguments |> emplace_new <| new Variable(at = at, name := "{strip(nt[0])}", _type = const_t(pt))
            }
        }
        fn.body = make_expression_block(body)
        fn.body |> force_at(at)
        if (!(compiling_module() |> add_function(fn))) {
            errors := "[metal_dispatch]: failed to add generated function '{fn_name}'"
            return false
        }
        return true
    }
}
