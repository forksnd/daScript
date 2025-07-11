options gen2
options indenting = 4
options no_unused_block_arguments = false
options no_unused_function_arguments = false
options strict_smart_pointers

module apply shared private

require ast
require daslib/ast_boost
require daslib/defer
require daslib/templates_boost
require daslib/macro_boost
require math

[macro_function]
def for_each_subrange(total : int; blk : block<(r : range) : void>) {
    let MAX_ARGUMENTS = DAS_MAX_FUNCTION_ARGUMENTS - 2  // we need one for the data itself
    var i = 0
    while (i < total) {
        let fromI = i
        let toI = min(i + MAX_ARGUMENTS, total)
        blk |> invoke(range(fromI, toI))
        i = toI
    }
}

/*
    def apply`Foo(self:Foo;arg_field1:block<(name:string,value:field1-type):void>;arg_field2:...)
        invoke(arg_field1,self.field1)
        invoke(arg_field2,self.field2)
        ...
*/
[macro_function]
def generateApplyVisitStruct(stype : TypeDeclPtr; frange : range; fnname : string; at : LineInfo; var names : array<string>) {
    assert(stype.baseType == Type.tStructure)
    assert(stype.dim |> length == 0)
    var inscope selfT <- clone_type(stype)
    selfT.flags |= TypeDeclFlags.isExplicit | TypeDeclFlags.explicitConst
    var inscope blkList : array<ExpressionPtr>
    var inscope func_args : array<VariablePtr>
    // for fld in stype.structType.fields
    for (fldi in frange) {
        if (true) {
            let fld & = unsafe(stype.structType.fields[fldi])
            names |> push(string(fld.name))
            blkList |> emplace_new <|  qmacro(invoke($i("__arg_{fld.name}"), _`_self.$f(fld.name)))
            var inscope argT <- new TypeDecl(baseType = Type.tBlock, at = at)                      // block<(x:field1type)>
            var inscope fldT <- clone_type(fld._type)
            if (stype.flags.constant) {
                fldT.flags |= TypeDeclFlags.constant
            }
            if (!fldT.isRefType) {
                fldT.flags |= TypeDeclFlags.ref
            }
            emplace(argT.argTypes, fldT)
            move_new(argT.firstType) <| new TypeDecl(baseType = Type.tVoid, at = at)
            argT.flags = TypeDeclFlags.constant
            func_args |> emplace_new <| new Variable(name := "__arg_{fld.name}", _type <- argT)
        }
    }
    var inscope fn <- qmacro_function(fnname) <| $ [unused_argument(_`_self)] (_`_self : $t(selfT); $a(func_args)) {
        $b(blkList)
    }
    fn.flags |= FunctionFlags.privateFunction
    compiling_module() |> add_function(fn)
}

[macro_function]
def generateApplyVisitStruct(stype : TypeDeclPtr; fnname : string; at : LineInfo) : array<string> {
    var names : array<string>
    for_each_subrange(length(stype.structType.fields)) <| $(frange) {
        generateApplyVisitStruct(stype, frange, "{fnname}`{frange.x}`{frange.y}", at, names)
    }
    return <- names
}

/*
    def apply`Foo(self:Foo;arg_field1:block<(name:string,value:field1-type):void>;arg_field2:...)
        invoke(arg_field1,self.field1)
        invoke(arg_field2,self.field2)
        ...
*/
[macro_function]
def generateApplyVisitTuple(stype : TypeDeclPtr; frange : range; fnname : string; at : LineInfo; var names : array<string>) {
    assert(stype.baseType == Type.tTuple)
    assert(stype.dim |> length == 0)
    var inscope selfT <- clone_type(stype)
    selfT.flags |= TypeDeclFlags.isExplicit | TypeDeclFlags.explicitConst
    var inscope blkList : array<ExpressionPtr>
    var inscope func_args : array<VariablePtr>
    for (fldi in frange) {
        if (true) {
            assume flda = stype.argTypes[fldi]
            let fldname = length(stype.argNames) == length(stype.argTypes) ? string(stype.argNames[fldi]) : "_{fldi}"
            names |> push(fldname)
            blkList |> emplace_new <|  qmacro(invoke($i("__arg_{fldname}"), _`_self.$f(fldname)))
            var inscope argT <- new TypeDecl(baseType = Type.tBlock, at = at)                      // block<(x:field1type)>
            var inscope fldT <- clone_type(flda)
            if (stype.flags.constant) {
                fldT.flags |= TypeDeclFlags.constant
            }
            if (!fldT.isRefType) {
                fldT.flags |= TypeDeclFlags.ref
            }
            emplace(argT.argTypes, fldT)
            move_new(argT.firstType) <| new TypeDecl(baseType = Type.tVoid, at = at)
            argT.flags = TypeDeclFlags.constant
            func_args |> emplace_new <| new Variable(name := "__arg_{fldname}", _type <- argT)
        }
    }
    var inscope fn <- qmacro_function(fnname) <| $ [unused_argument(_`_self)] (_`_self : $t(selfT); $a(func_args)) {
        $b(blkList)
    }
    fn.flags |= FunctionFlags.privateFunction
    compiling_module() |> add_function(fn)
}

[macro_function]
def generateApplyVisitTuple(stype : TypeDeclPtr; fnname : string; at : LineInfo) : array<string> {
    var names : array<string>
    for_each_subrange(length(stype.argTypes)) <| $(frange) {
        generateApplyVisitTuple(stype, frange, "{fnname}`{frange.x}`{frange.y}", at, names)
    }
    return <- names
}

/*
    def apply`Foo(self:Foo;arg_field1:block<(name:string,value:field1-type):void>;arg_field2:...)
        if variant_index(self)==0
            invoke(arg_field1,self.field1)
            return
        if variant_idnex(self)==2
            invoke(arg_field2,self.field2)
            return
        ...
*/
[macro_function]
def generateApplyVisitVariant(stype : TypeDeclPtr; frange : range; fnname : string; at : LineInfo; var names : array<string>) {
    assert(stype.baseType == Type.tVariant)
    assert(stype.dim |> length == 0)
    var inscope selfT <- clone_type(stype)
    selfT.flags |= TypeDeclFlags.isExplicit | TypeDeclFlags.explicitConst
    var inscope blkList : array<ExpressionPtr>
    var inscope func_args : array<VariablePtr>
    for (fldi in frange) {
        if (true) {
            assume flda = stype.argTypes[fldi]
            let fldname = length(stype.argNames) == length(stype.argTypes) ? string(stype.argNames[fldi]) : "_{fldi}"
            names |> push(fldname)
            var inscope vexpr <- qmacro_block <| $() {
                if (variant_index(_`_self) == $v(fldi)) {
                    invoke($i("__arg_{fldname}"), _`_self as $f(fldname))
                    return
                }
            }
            blkList |> emplace(vexpr)
            var inscope argT <- new TypeDecl(baseType = Type.tBlock, at = at)                      // block<(x:field1type)>
            var inscope fldT <- clone_type(flda)
            if (stype.flags.constant) {
                fldT.flags |= TypeDeclFlags.constant
            }
            if (!fldT.isRefType) {
                fldT.flags |= TypeDeclFlags.ref
            }
            emplace(argT.argTypes, fldT)
            move_new(argT.firstType) <| new TypeDecl(baseType = Type.tVoid, at = at)
            argT.flags = TypeDeclFlags.constant
            func_args |> emplace_new <| new Variable(name := "__arg_{fldname}", _type <- argT)
        }
    }
    var inscope fn <- qmacro_function(fnname) <| $ [unused_argument(_`_self)] (_`_self : $t(selfT); $a(func_args)) {
        $b(blkList)
    }
    fn.flags |= FunctionFlags.privateFunction
    compiling_module() |> add_function(fn)
}

[macro_function]
def generateApplyVisitVariant(stype : TypeDeclPtr; fnname : string; at : LineInfo) : array<string> {
    var names : array<string>
    for_each_subrange(length(stype.argTypes)) <| $(frange) {
        generateApplyVisitVariant(stype, frange, "{fnname}`{frange.x}`{frange.y}", at, names)
    }
    return <- names
}

[macro_function]
def clone_block_with_name(name : string; blk : ExpressionPtr) {
    macro_verify(blk is ExprMakeBlock, this_program(), blk.at, "expecting make block, i.e. $(..)")
    var inscope res <- clone_expression(blk)
    var imb = res as ExprMakeBlock
    var rblk = imb._block as ExprBlock
    macro_verify(rblk != null, this_program(), blk.at, "expecting block")
    macro_verify(rblk.arguments |> length == 2, this_program(), blk.at, "expecting block with 2 arguments")
    var inscope eassume <- new ExprAssume(at = blk.at,
        alias := rblk.arguments[0].name,
        subexpr <- new ExprConstString(at = blk.at, value := name)
    )
    rblk.arguments |> erase(0)
    rblk.list |> emplace(eassume, 0)
    return <- res
}

[call_macro(name="apply")]  // apply(value, block)
class ApplyMacro : AstCallMacro {
    //! This macro implements the apply() pattern. The idea is that for each entry in the structure, variant, or tuple,
    //! the block will be invoked. Both element name, and element value are passed to the block.
    //! For example
    //!
    //!     struct Bar
    //!         x, y : float
    //!     apply([[Bar x=1.,y=2.]]) <| $ ( name:string; field )
    //!         print("{name} = {field} ")
    //!
    //! Would print x = 1.0 y = 2.0
    def override visit(prog : ProgramPtr; mod : Module?; var expr : smart_ptr<ExprCallMacro>) : ExpressionPtr {
        macro_verify(expr.arguments |> length == 2, prog, expr.at, "expecting apply(value, block)")
        if (expr.arguments[0]._type != null) {// need value inferred
            var inscope argT := expr.arguments[0]._type
            macro_verify(argT.dim |> length == 0, prog, expr.at, "can't apply to dim")
            macro_verify(argT.baseType == Type.tStructure || argT.baseType == Type.tTuple || argT.baseType == Type.tVariant,
                prog, expr.at, "can only apply to {describe(expr.arguments[0]._type)}")
            macro_verify(expr.arguments[1] is ExprMakeBlock, prog, expr.at, "expecting make block, i.e. $(..)")
            var callName : string
            var nfields : int
            var names : array<string>
            if (argT.baseType == Type.tStructure) {
                callName = "apply`struct`{argT.structType.name}`{expr.at.line}"
                nfields = argT.structType.fields |> length
                names <- generateApplyVisitStruct(expr.arguments[0]._type, callName, expr.at)
            } elif (argT.baseType == Type.tTuple) {
                callName = "apply`tuple`{expr.arguments[0]._type.get_mnh}`{expr.at.line}"
                nfields = length(argT.argTypes)
                names <- generateApplyVisitTuple(expr.arguments[0]._type, callName, expr.at)
            } elif (argT.baseType == Type.tVariant) {
                callName = "apply`variant`{expr.arguments[0]._type.get_mnh}`{expr.at.line}"
                nfields = length(argT.argTypes)
                names <- generateApplyVisitVariant(expr.arguments[0]._type, callName, expr.at)
            } else {
                macro_error(prog, expr.at, "internal error. can't apply to {describe(argT)}")
                return <- default<ExpressionPtr>
            }
            // make a collection of calls
            var inscope calls : array<ExpressionPtr>
            for_each_subrange(nfields) <| $(frange) {
                var inscope call <- new ExprCall(name := "_::{callName}`{frange.x}`{frange.y}", at = expr.at)
                emplace_new(call.arguments, clone_expression(expr.arguments[0]))
                for (idx in frange) {
                    emplace_new(call.arguments, clone_block_with_name(names[idx], expr.arguments[1]))
                }
                calls |> emplace(call)
            }
            if (length(calls) == 1) {
                return <- calls[0]
            }
            var inscope callblock <- new ExprBlock(at = expr.at)
            unsafe {
                callblock.list := calls
            }
            return <- callblock
        }
        return <- default<ExpressionPtr>
    }
}
