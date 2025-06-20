#include "daScript/misc/platform.h"

#include "daScript/simulate/simulate.h"
#include "daScript/simulate/aot.h"
#include "daScript/simulate/aot_library.h"

 // require builtin
 // require rtti
#include "daScript/simulate/aot_builtin_rtti.h"
#include "daScript/ast/ast.h"
#include "daScript/ast/ast_handle.h"
 // require strings
#include "daScript/simulate/aot_builtin_string.h"
 // require ast
#include "daScript/ast/ast.h"
#include "daScript/simulate/aot_builtin_ast.h"
#include "daScript/ast/ast_generate.h"
 // require math
#include "daScript/simulate/aot_builtin_math.h"
 // require strings_boost
 // require ast_boost
 // require templates
 // require contracts
 // require array_boost
 // require algorithm
 // require templates_boost
 // require macro_boost
 // require defer
 // require apply

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4100)   // unreferenced formal parameter
#pragma warning(disable:4189)   // local variable is initialized but not referenced
#pragma warning(disable:4244)   // conversion from 'int32_t' to 'float', possible loss of data
#pragma warning(disable:4114)   // same qualifier more than once
#pragma warning(disable:4623)   // default constructor was implicitly defined as deleted
#pragma warning(disable:4946)   // reinterpret_cast used between related classes
#pragma warning(disable:4269)   // 'const' automatic data initialized with compiler generated default constructor produces unreliable results
#pragma warning(disable:4555)   // result of expression not used
#endif
#if defined(__EDG__)
#pragma diag_suppress 826
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wwrite-strings"
#pragma GCC diagnostic ignored "-Wreturn-local-addr"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wwritable-strings"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wunsequenced"
#pragma clang diagnostic ignored "-Wunused-function"
#endif

namespace das {
namespace _anon_7436603895564257385 {

namespace apply { struct ApplyMacro; };
namespace ast { struct AstFunctionAnnotation; };
namespace ast { struct AstBlockAnnotation; };
namespace ast { struct AstStructureAnnotation; };
namespace ast { struct AstPassMacro; };
namespace ast { struct AstVariantMacro; };
namespace ast { struct AstForLoopMacro; };
namespace ast { struct AstCaptureMacro; };
namespace ast { struct AstTypeMacro; };
namespace ast { struct AstSimulateMacro; };
namespace ast { struct AstReaderMacro; };
namespace ast { struct AstCommentReader; };
namespace ast { struct AstCallMacro; };
namespace ast { struct AstTypeInfoMacro; };
namespace ast { struct AstEnumerationAnnotation; };
namespace ast { struct AstVisitor; };
namespace ast_boost { struct MacroMacro; };
namespace ast_boost { struct TagFunctionAnnotation; };
namespace ast_boost { struct TagStructureAnnotation; };
namespace ast_boost { struct SetupAnyAnnotation; };
namespace ast_boost { struct SetupFunctionAnnotation; };
namespace ast_boost { struct SetupBlockAnnotation; };
namespace ast_boost { struct SetupStructureAnnotation; };
namespace ast_boost { struct SetupEnumerationAnnotation; };
namespace ast_boost { struct SetupContractAnnotation; };
namespace ast_boost { struct SetupReaderMacro; };
namespace ast_boost { struct SetupCommentReader; };
namespace ast_boost { struct SetupVariantMacro; };
namespace ast_boost { struct SetupForLoopMacro; };
namespace ast_boost { struct SetupCaptureMacro; };
namespace ast_boost { struct SetupTypeMacro; };
namespace ast_boost { struct SetupSimulateMacro; };
namespace ast_boost { struct SetupCallMacro; };
namespace ast_boost { struct SetupTypeInfoMacro; };
namespace ast_boost { struct SetupInferMacro; };
namespace ast_boost { struct SetupDirtyInferMacro; };
namespace ast_boost { struct SetupLintMacro; };
namespace ast_boost { struct SetupGlobalLintMacro; };
namespace ast_boost { struct SetupOptimizationMacro; };
namespace ast_boost { struct TagFunctionMacro; };
namespace ast_boost { struct BetterRttiVisitor; };
namespace templates { struct DecltypeMacro; };
namespace templates { struct DecltypeNoRefMacro; };
namespace templates { struct TemplateMacro; };
namespace contracts { struct IsAnyType; };
namespace contracts { struct IsAnyArrayMacro; };
namespace contracts { struct IsAnyEnumMacro; };
namespace contracts { struct IsAnyBitfieldMacro; };
namespace contracts { struct IsAnyVectorType; };
namespace contracts { struct IsAnyStructMacro; };
namespace contracts { struct IsAnyNumericMacro; };
namespace contracts { struct IsAnyWorkhorse; };
namespace contracts { struct IsAnyWorkhorseNonPtrMacro; };
namespace contracts { struct IsAnyTupleNonPtrMacro; };
namespace contracts { struct IsAnyVariantNonPtrMacro; };
namespace contracts { struct IsAnyFunctionNonPtrMacro; };
namespace contracts { struct IsAnyLambdaMacro; };
namespace contracts { struct IsRefMacro; };
namespace contracts { struct IsPointer; };
namespace contracts { struct IsClass; };
namespace contracts { struct IsValueHandle; };
namespace templates_boost { struct Template; };
namespace templates_boost { struct TemplateVisitor; };
namespace templates_boost { struct RemoveDerefVisitor; };
namespace templates_boost { struct QRulesVisitor; };
namespace templates_boost { struct AstQCallMacro; };
namespace templates_boost { struct QMacro; };
namespace templates_boost { struct QBlockMacro; };
namespace templates_boost { struct QBlockToArrayMacro; };
namespace templates_boost { struct QBlockExprMacro; };
namespace templates_boost { struct QTypeMacro; };
namespace templates_boost { struct AstQNamedMacro; };
namespace templates_boost { struct AstQFunctionMacro; };
namespace templates_boost { struct AstQVariableMacro; };
namespace templates_boost { struct AstQNamedClassMacro; };
namespace templates_boost { struct AstQMethodMacro; };
namespace macro_boost { struct MacroVerifyMacro; };
namespace macro_boost { struct CaptureBlock; };
namespace macro_boost { struct CapturedVariable; };
namespace macro_boost { struct ColletFinally; };
namespace macro_boost { struct ColletLabels; };
namespace macro_boost { struct ReturnSkipLockcheck; };
namespace defer { struct DeferMacro; };
namespace defer { struct DeferDeleteMacro; };
// unused enumeration CompilationError
// unused enumeration ConstMatters
// unused enumeration RefMatters
// unused enumeration TemporaryMatters
#if 0 // external enum
namespace rtti {

enum class Type : int32_t {
    none = int32_t(INT64_C(0)),
    autoinfer = int32_t(INT64_C(1)),
    alias = int32_t(INT64_C(2)),
    option = int32_t(INT64_C(3)),
    typeDecl = int32_t(INT64_C(4)),
    typeMacro = int32_t(INT64_C(5)),
    fakeContext = int32_t(INT64_C(6)),
    fakeLineInfo = int32_t(INT64_C(7)),
    anyArgument = int32_t(INT64_C(8)),
    tVoid = int32_t(INT64_C(9)),
    tBool = int32_t(INT64_C(10)),
    tInt8 = int32_t(INT64_C(11)),
    tUInt8 = int32_t(INT64_C(12)),
    tInt16 = int32_t(INT64_C(13)),
    tUInt16 = int32_t(INT64_C(14)),
    tInt64 = int32_t(INT64_C(15)),
    tUInt64 = int32_t(INT64_C(16)),
    tInt = int32_t(INT64_C(17)),
    tInt2 = int32_t(INT64_C(18)),
    tInt3 = int32_t(INT64_C(19)),
    tInt4 = int32_t(INT64_C(20)),
    tUInt = int32_t(INT64_C(21)),
    tUInt2 = int32_t(INT64_C(22)),
    tUInt3 = int32_t(INT64_C(23)),
    tUInt4 = int32_t(INT64_C(24)),
    tFloat = int32_t(INT64_C(25)),
    tFloat2 = int32_t(INT64_C(26)),
    tFloat3 = int32_t(INT64_C(27)),
    tFloat4 = int32_t(INT64_C(28)),
    tDouble = int32_t(INT64_C(29)),
    tRange = int32_t(INT64_C(30)),
    tURange = int32_t(INT64_C(31)),
    tRange64 = int32_t(INT64_C(32)),
    tURange64 = int32_t(INT64_C(33)),
    tString = int32_t(INT64_C(34)),
    tStructure = int32_t(INT64_C(35)),
    tHandle = int32_t(INT64_C(36)),
    tEnumeration = int32_t(INT64_C(37)),
    tEnumeration8 = int32_t(INT64_C(38)),
    tEnumeration16 = int32_t(INT64_C(39)),
    tEnumeration64 = int32_t(INT64_C(40)),
    tBitfield = int32_t(INT64_C(41)),
    tPointer = int32_t(INT64_C(42)),
    tFunction = int32_t(INT64_C(43)),
    tLambda = int32_t(INT64_C(44)),
    tIterator = int32_t(INT64_C(45)),
    tArray = int32_t(INT64_C(46)),
    tTable = int32_t(INT64_C(47)),
    tBlock = int32_t(INT64_C(48)),
    tTuple = int32_t(INT64_C(49)),
    tVariant = int32_t(INT64_C(50)),
};
}
#endif // external enum
// unused enumeration ConversionResult
// unused enumeration CaptureMode
// unused enumeration SideEffects
// unused structure AstFunctionAnnotation
// unused structure AstBlockAnnotation
// unused structure AstStructureAnnotation
// unused structure AstPassMacro
// unused structure AstVariantMacro
// unused structure AstForLoopMacro
// unused structure AstCaptureMacro
// unused structure AstTypeMacro
// unused structure AstSimulateMacro
// unused structure AstReaderMacro
// unused structure AstCommentReader
namespace ast {

struct AstCallMacro {
    void * __rtti;
    Func DAS_COMMENT((void,ast::AstCallMacro)) __finalize;
    Func DAS_COMMENT((void,ast::AstCallMacro,smart_ptr_raw<Program> const ,Module * const ,smart_ptr_raw<ExprCallMacro> const )) preVisit;
    Func DAS_COMMENT((smart_ptr_raw<Expression>,ast::AstCallMacro,smart_ptr_raw<Program> const ,Module * const ,smart_ptr_raw<ExprCallMacro> const )) visit;
    Func DAS_COMMENT((bool,ast::AstCallMacro,smart_ptr_raw<ExprCallMacro> const ,int32_t)) canVisitArgument;
    Func DAS_COMMENT((bool,ast::AstCallMacro,smart_ptr_raw<ExprCallMacro> const )) canFoldReturnResult;
};
}
// unused structure AstTypeInfoMacro
// unused structure AstEnumerationAnnotation
// unused structure AstVisitor
// unused structure MacroMacro
// unused structure TagFunctionAnnotation
// unused structure TagStructureAnnotation
// unused structure SetupAnyAnnotation
// unused structure SetupFunctionAnnotation
// unused structure SetupBlockAnnotation
// unused structure SetupStructureAnnotation
// unused structure SetupEnumerationAnnotation
// unused structure SetupContractAnnotation
// unused structure SetupReaderMacro
// unused structure SetupCommentReader
// unused structure SetupVariantMacro
// unused structure SetupForLoopMacro
// unused structure SetupCaptureMacro
// unused structure SetupTypeMacro
// unused structure SetupSimulateMacro
// unused structure SetupCallMacro
// unused structure SetupTypeInfoMacro
// unused structure SetupInferMacro
// unused structure SetupDirtyInferMacro
// unused structure SetupLintMacro
// unused structure SetupGlobalLintMacro
// unused structure SetupOptimizationMacro
// unused structure TagFunctionMacro
// unused structure BetterRttiVisitor
// unused structure DecltypeMacro
// unused structure DecltypeNoRefMacro
// unused structure TemplateMacro
// unused structure IsAnyType
// unused structure IsAnyArrayMacro
// unused structure IsAnyEnumMacro
// unused structure IsAnyBitfieldMacro
// unused structure IsAnyVectorType
// unused structure IsAnyStructMacro
// unused structure IsAnyNumericMacro
// unused structure IsAnyWorkhorse
// unused structure IsAnyWorkhorseNonPtrMacro
// unused structure IsAnyTupleNonPtrMacro
// unused structure IsAnyVariantNonPtrMacro
// unused structure IsAnyFunctionNonPtrMacro
// unused structure IsAnyLambdaMacro
// unused structure IsRefMacro
// unused structure IsPointer
// unused structure IsClass
// unused structure IsValueHandle
// unused structure Template
// unused structure TemplateVisitor
// unused structure RemoveDerefVisitor
// unused structure QRulesVisitor
// unused structure AstQCallMacro
// unused structure QMacro
// unused structure QBlockMacro
// unused structure QBlockToArrayMacro
// unused structure QBlockExprMacro
// unused structure QTypeMacro
// unused structure AstQNamedMacro
// unused structure AstQFunctionMacro
// unused structure AstQVariableMacro
// unused structure AstQNamedClassMacro
// unused structure AstQMethodMacro
// unused structure MacroVerifyMacro
// unused structure CaptureBlock
// unused structure CapturedVariable
// unused structure ColletFinally
// unused structure ColletLabels
// unused structure ReturnSkipLockcheck
// unused structure DeferMacro
// unused structure DeferDeleteMacro
namespace apply {

struct ApplyMacro {
    void * __rtti;
    Func DAS_COMMENT((void,ast::AstCallMacro)) __finalize;
    Func DAS_COMMENT((void,ast::AstCallMacro,smart_ptr_raw<Program> const ,Module * const ,smart_ptr_raw<ExprCallMacro> const )) preVisit;
    Func DAS_COMMENT((smart_ptr_raw<Expression>,ast::AstCallMacro,smart_ptr_raw<Program> const ,Module * const ,smart_ptr_raw<ExprCallMacro> const )) visit;
    Func DAS_COMMENT((bool,ast::AstCallMacro,smart_ptr_raw<ExprCallMacro> const ,int32_t)) canVisitArgument;
    Func DAS_COMMENT((bool,ast::AstCallMacro,smart_ptr_raw<ExprCallMacro> const )) canFoldReturnResult;
};
}
extern TypeInfo __type_info__5eddda99b960ff80;
extern TypeInfo __type_info__2dc797bf0765183e;
extern TypeInfo __type_info__e6e3914a5426853e;
extern TypeInfo __type_info__af63ee4c86020b22;

TypeInfo __type_info__5eddda99b960ff80 = { Type::tArray, nullptr, nullptr, nullptr, &__type_info__af63ee4c86020b22, nullptr, nullptr, nullptr, 0, 0, nullptr, 57346, TypeSize<TArray<char *>>::size, UINT64_C(0x5eddda99b960ff80) };
TypeInfo __type_info__2dc797bf0765183e = { Type::tArray, nullptr, nullptr, nullptr, &__type_info__af63ee4c86020b22, nullptr, nullptr, nullptr, 0, 0, nullptr, 57346, TypeSize<TArray<char *>>::size, UINT64_C(0x2dc797bf0765183e) };
TypeInfo __type_info__e6e3914a5426853e = { Type::tArray, nullptr, nullptr, nullptr, &__type_info__af63ee4c86020b22, nullptr, nullptr, nullptr, 0, 0, nullptr, 57346, TypeSize<TArray<char *>>::size, UINT64_C(0xe6e3914a5426853e) };
TypeInfo __type_info__af63ee4c86020b22 = { Type::tString, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, nullptr, 16388, TypeSize<char *>::size, UINT64_C(0xaf63ee4c86020b22) };

static void resolveTypeInfoAnnotations()
{
    vector<TypeInfo> annotations = {};
    for (auto& ann : annotations) {
        ann.resolveAnnotation();
    }
}

TypeInfo * __tinfo_0[1] = { &__type_info__5eddda99b960ff80 };
TypeInfo * __tinfo_1[1] = { &__type_info__2dc797bf0765183e };
TypeInfo * __tinfo_2[1] = { &__type_info__e6e3914a5426853e };

inline void _FuncbuiltinTickpushTick10769833213962245646_ecf3600ad4420fc3 ( Context * __context__, TArray<ast::AstCallMacro *> & __Arr_rename_at_181_0, ast::AstCallMacro * __value_rename_at_181_1 );
inline StructInfo const  * _FuncrttiTickclass_infoTick15801393167907430156_8e51e7776ee5b334 ( Context * __context__, apply::ApplyMacro const  & __cl_rename_at_116_2 );
inline void clone_4cd0086ead03b6d6 ( Context * __context__, smart_ptr_raw<TypeDecl> & __dest_rename_at_1092_3, smart_ptr_raw<TypeDecl> const  __src_rename_at_1092_4 );
inline smart_ptr_raw<TypeDecl> _FuncbuiltinTickclone_to_moveTick2007252383599261567_8819185055f20046 ( Context * __context__, smart_ptr_raw<TypeDecl> const  __clone_src_rename_at_1089_9 );
inline void _FuncbuiltinTickemplace_newTick990257600204377963_65f613aacb85635 ( Context * __context__, TArray<smart_ptr_raw<Expression>> & __Arr_rename_at_1036_11, smart_ptr_raw<Expression> __value_rename_at_1036_12 );
inline void _FuncbuiltinTickfinalizeTick13836114024949725080_1f1a10a486be953f ( Context * __context__, TArray<smart_ptr_raw<Variable>> & __a_rename_at_1234_13 );
inline void _FuncbuiltinTickfinalizeTick13836114024949725080_12907c65976ea98b ( Context * __context__, TArray<smart_ptr_raw<Expression>> & __a_rename_at_1234_15 );
inline TArray<char *> & _FuncbuiltinTick_return_with_lockcheckTick2939372000839727345_218b97dd24b7eccd ( Context * __context__, TArray<char *> & __a_rename_at_50_17 );
inline smart_ptr_raw<Expression> _Funcast_boostTickconvert_to_expressionTick11707039267614988023_9deacd1af62e7af5 ( Context * __context__, int32_t __value_rename_at_849_18 );
inline void _FuncbuiltinTick_move_with_lockcheckTick3467971516570048129_76cb559e350b5007 ( Context * __context__, TArray<char *> & __a_rename_at_32_19, TArray<char *> & __b_rename_at_32_20 );
inline void _FuncbuiltinTickpushTick14133213201864676143_147a6dd0baa7ab92 ( Context * __context__, TArray<char *> & __Arr_rename_at_165_21, char * const  __value_rename_at_165_22 );
inline void _FuncbuiltinTickemplace_newTick990257600204377963_b88a35d64549eac0 ( Context * __context__, TArray<smart_ptr_raw<Variable>> & __Arr_rename_at_1036_23, smart_ptr_raw<Variable> __value_rename_at_1036_24 );
inline char * _FuncastTickdescribeTick2562845734617055679_af1bd1c0c428d037 ( Context * __context__, smart_ptr_raw<TypeDecl> const  __decl_rename_at_38_25, bool __extra_rename_at_38_26, bool __contracts_rename_at_38_27, bool __modules_rename_at_38_28 );
inline void _FuncbuiltinTickemplaceTick13923705686753630697_b3d3044d3eccbbb7 ( Context * __context__, TArray<smart_ptr_raw<Expression>> & __Arr_rename_at_287_29, smart_ptr_raw<Expression> & __value_rename_at_287_30 );
inline void _FuncbuiltinTickcloneTick9409548443506319159_e97a0907d62c3fa4 ( Context * __context__, das::vector<smart_ptr<Expression>> & __args_rename_at_1171_31, TArray<smart_ptr_raw<Expression>> & __nargs_rename_at_1171_32 );

void __init_script ( Context * __context__, bool __init_shared )
{
}

inline void _FuncbuiltinTickpushTick10769833213962245646_ecf3600ad4420fc3 ( Context * __context__, TArray<ast::AstCallMacro *> &  __Arr_rename_at_181_0, ast::AstCallMacro * __value_rename_at_181_1 )
{
    das_copy(__Arr_rename_at_181_0(builtin_array_push_back(das_arg<TArray<ast::AstCallMacro *>>::pass(__Arr_rename_at_181_0),8,__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL))),__context__),__value_rename_at_181_1);
}

inline StructInfo const  * _FuncrttiTickclass_infoTick15801393167907430156_8e51e7776ee5b334 ( Context * __context__, apply::ApplyMacro const  &  __cl_rename_at_116_2 )
{
    return das_auto_cast<StructInfo const  *>::cast(das_cast<StructInfo const  *>::cast(((das_deref(__context__,das_cast<TypeInfo const  *>::cast(__cl_rename_at_116_2.__rtti))).getStructType())));
}

inline void clone_4cd0086ead03b6d6 ( Context * __context__, smart_ptr_raw<TypeDecl> & __dest_rename_at_1092_3, smart_ptr_raw<TypeDecl> const  __src_rename_at_1092_4 )
{
    builtin_smart_ptr_clone(das_auto_cast<smart_ptr_raw<void> &>::cast(__dest_rename_at_1092_3),das_auto_cast<smart_ptr_raw<void> const >::cast(__src_rename_at_1092_4),__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL)));
}

inline smart_ptr_raw<TypeDecl> _FuncbuiltinTickclone_to_moveTick2007252383599261567_8819185055f20046 ( Context * __context__, smart_ptr_raw<TypeDecl> const  __clone_src_rename_at_1089_9 )
{
    smart_ptr_raw<TypeDecl> __clone_dest_rename_at_1091_10;das_zero(__clone_dest_rename_at_1091_10);
    clone_4cd0086ead03b6d6(__context__,__clone_dest_rename_at_1091_10,__clone_src_rename_at_1089_9);
    return /* <- */ das_auto_cast_move<smart_ptr_raw<TypeDecl>>::cast(__clone_dest_rename_at_1091_10);
}

inline void _FuncbuiltinTickemplace_newTick990257600204377963_65f613aacb85635 ( Context * __context__, TArray<smart_ptr_raw<Expression>> &  __Arr_rename_at_1036_11, smart_ptr_raw<Expression> __value_rename_at_1036_12 )
{
    das_move(__Arr_rename_at_1036_11(builtin_array_push_back(das_arg<TArray<smart_ptr_raw<Expression>>>::pass(__Arr_rename_at_1036_11),8,__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL))),__context__),__value_rename_at_1036_12);
}

inline void _FuncbuiltinTickfinalizeTick13836114024949725080_1f1a10a486be953f ( Context * __context__, TArray<smart_ptr_raw<Variable>> &  __a_rename_at_1234_13 )
{
    {
        bool __need_loop_1236 = true;
        // aV: smart_ptr<ast::Variable> aka TT&
        das_iterator<TArray<smart_ptr_raw<Variable>>> __aV_iterator(__a_rename_at_1234_13);
        smart_ptr_raw<Variable> * __aV_rename_at_1236_14;
        __need_loop_1236 = __aV_iterator.first(__context__,(__aV_rename_at_1236_14)) && __need_loop_1236;
        for ( ; __need_loop_1236 ; __need_loop_1236 = __aV_iterator.next(__context__,(__aV_rename_at_1236_14)) )
        {
            das_delete_handle<smart_ptr_raw<Variable>>::clear(__context__,(*__aV_rename_at_1236_14));
        }
        __aV_iterator.close(__context__,(__aV_rename_at_1236_14));
    };
    builtin_array_free(das_arg<TArray<smart_ptr_raw<Variable>>>::pass(__a_rename_at_1234_13),8,__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL)));
}

inline void _FuncbuiltinTickfinalizeTick13836114024949725080_12907c65976ea98b ( Context * __context__, TArray<smart_ptr_raw<Expression>> &  __a_rename_at_1234_15 )
{
    {
        bool __need_loop_1236 = true;
        // aV: smart_ptr<ast::Expression> aka TT&
        das_iterator<TArray<smart_ptr_raw<Expression>>> __aV_iterator(__a_rename_at_1234_15);
        smart_ptr_raw<Expression> * __aV_rename_at_1236_16;
        __need_loop_1236 = __aV_iterator.first(__context__,(__aV_rename_at_1236_16)) && __need_loop_1236;
        for ( ; __need_loop_1236 ; __need_loop_1236 = __aV_iterator.next(__context__,(__aV_rename_at_1236_16)) )
        {
            das_delete_handle<smart_ptr_raw<Expression>>::clear(__context__,(*__aV_rename_at_1236_16));
        }
        __aV_iterator.close(__context__,(__aV_rename_at_1236_16));
    };
    builtin_array_free(das_arg<TArray<smart_ptr_raw<Expression>>>::pass(__a_rename_at_1234_15),8,__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL)));
}

inline TArray<char *> & _FuncbuiltinTick_return_with_lockcheckTick2939372000839727345_218b97dd24b7eccd ( Context * __context__, TArray<char *> &  __a_rename_at_50_17 )
{
    das_call_interop<void>::call(&builtin_verify_locks,__context__,SimNode_AotInterop<1>(__tinfo_0,cast<TArray<char *> &>::from(__a_rename_at_50_17)));
    return das_auto_cast_ref<TArray<char *> &>::cast(__a_rename_at_50_17);
}

inline smart_ptr_raw<Expression> _Funcast_boostTickconvert_to_expressionTick11707039267614988023_9deacd1af62e7af5 ( Context * __context__, int32_t __value_rename_at_849_18 )
{
    LineInfo _temp_make_local_850_43_0; _temp_make_local_850_43_0;
    return /* <- */ das_auto_cast_move<smart_ptr_raw<Expression>>::cast(das_invoke_function<smart_ptr_raw<Expression>>::invoke<int32_t,LineInfo const  &>(__context__,nullptr,Func(__context__->fnByMangledName(/*@apply::ast_boost`convert_to_expression`16483834305137942954 C=Xi CH<rtti::LineInfo>*/ 0xff72eba5ac8b0825)),__value_rename_at_849_18,das_arg<LineInfo>::pass((_temp_make_local_850_43_0 = (/*c-tor*/ LineInfo(/*end-c-tor*/))))));
}

inline void _FuncbuiltinTick_move_with_lockcheckTick3467971516570048129_76cb559e350b5007 ( Context * __context__, TArray<char *> &  __a_rename_at_32_19, TArray<char *> &  __b_rename_at_32_20 )
{
    das_call_interop<void>::call(&builtin_verify_locks,__context__,SimNode_AotInterop<1>(__tinfo_1,cast<TArray<char *> &>::from(__a_rename_at_32_19)));
    das_call_interop<void>::call(&builtin_verify_locks,__context__,SimNode_AotInterop<1>(__tinfo_2,cast<TArray<char *> &>::from(__b_rename_at_32_20)));
    das_move(__a_rename_at_32_19,__b_rename_at_32_20);
}

inline void _FuncbuiltinTickpushTick14133213201864676143_147a6dd0baa7ab92 ( Context * __context__, TArray<char *> &  __Arr_rename_at_165_21, char * const  __value_rename_at_165_22 )
{
    das_copy(__Arr_rename_at_165_21(builtin_array_push_back(das_arg<TArray<char *>>::pass(__Arr_rename_at_165_21),8,__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL))),__context__),__value_rename_at_165_22);
}

inline void _FuncbuiltinTickemplace_newTick990257600204377963_b88a35d64549eac0 ( Context * __context__, TArray<smart_ptr_raw<Variable>> &  __Arr_rename_at_1036_23, smart_ptr_raw<Variable> __value_rename_at_1036_24 )
{
    das_move(__Arr_rename_at_1036_23(builtin_array_push_back(das_arg<TArray<smart_ptr_raw<Variable>>>::pass(__Arr_rename_at_1036_23),8,__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL))),__context__),__value_rename_at_1036_24);
}

inline char * _FuncastTickdescribeTick2562845734617055679_af1bd1c0c428d037 ( Context * __context__, smart_ptr_raw<TypeDecl> const  __decl_rename_at_38_25, bool __extra_rename_at_38_26, bool __contracts_rename_at_38_27, bool __modules_rename_at_38_28 )
{
    return das_auto_cast<char *>::cast(((char * const )(ast_describe_typedecl(__decl_rename_at_38_25,__extra_rename_at_38_26,__contracts_rename_at_38_27,__modules_rename_at_38_28,__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL))))));
}

inline void _FuncbuiltinTickemplaceTick13923705686753630697_b3d3044d3eccbbb7 ( Context * __context__, TArray<smart_ptr_raw<Expression>> &  __Arr_rename_at_287_29, smart_ptr_raw<Expression> & __value_rename_at_287_30 )
{
    das_move(__Arr_rename_at_287_29(builtin_array_push_back(das_arg<TArray<smart_ptr_raw<Expression>>>::pass(__Arr_rename_at_287_29),8,__context__,((LineInfoArg *)(&LineInfo::g_LineInfoNULL))),__context__),__value_rename_at_287_30);
}

inline void _FuncbuiltinTickcloneTick9409548443506319159_e97a0907d62c3fa4 ( Context * __context__, das::vector<smart_ptr<Expression>> &  __args_rename_at_1171_31, TArray<smart_ptr_raw<Expression>> &  __nargs_rename_at_1171_32 )
{
    das_vector_clear(das_arg<das::vector<smart_ptr<Expression>>>::pass(__args_rename_at_1171_31));
    {
        bool __need_loop_1173 = true;
        // narg: smart_ptr<ast::Expression aka TT>&
        das_iterator<TArray<smart_ptr_raw<Expression>>> __narg_iterator(__nargs_rename_at_1171_32);
        smart_ptr_raw<Expression> * __narg_rename_at_1173_33;
        __need_loop_1173 = __narg_iterator.first(__context__,(__narg_rename_at_1173_33)) && __need_loop_1173;
        for ( ; __need_loop_1173 ; __need_loop_1173 = __narg_iterator.next(__context__,(__narg_rename_at_1173_33)) )
        {
            das_vector_emplace_back(das_arg<das::vector<smart_ptr<Expression>>>::pass(__args_rename_at_1171_31),(*__narg_rename_at_1173_33));
        }
        __narg_iterator.close(__context__,(__narg_rename_at_1173_33));
    };
    _FuncbuiltinTickfinalizeTick13836114024949725080_12907c65976ea98b(__context__,das_arg<TArray<smart_ptr_raw<Expression>>>::pass(__nargs_rename_at_1171_32));
}

static void registerAotFunctions ( AotLibrary & aotLib ) {
    aotLib[0xb6c97724d854791f] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickpushTick10769833213962245646_ecf3600ad4420fc3>>();
    };
    aotLib[0xdea04d6a477ccf6a] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncrttiTickclass_infoTick15801393167907430156_8e51e7776ee5b334>>();
    };
    aotLib[0x3fd64833e872b68d] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&clone_4cd0086ead03b6d6>>();
    };
    aotLib[0x1e02b0a331897e5b] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickclone_to_moveTick2007252383599261567_8819185055f20046>>();
    };
    aotLib[0xacc5639dfe2fa351] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickemplace_newTick990257600204377963_65f613aacb85635>>();
    };
    aotLib[0x2d0a519d221df31f] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickfinalizeTick13836114024949725080_1f1a10a486be953f>>();
    };
    aotLib[0x9a4e250937532a7e] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickfinalizeTick13836114024949725080_12907c65976ea98b>>();
    };
    aotLib[0xa1118398b6a2b59d] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTick_return_with_lockcheckTick2939372000839727345_218b97dd24b7eccd>>();
    };
    aotLib[0x3022a96f39dae377] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_Funcast_boostTickconvert_to_expressionTick11707039267614988023_9deacd1af62e7af5>>();
    };
    aotLib[0xc5c4c782f77b0308] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTick_move_with_lockcheckTick3467971516570048129_76cb559e350b5007>>();
    };
    aotLib[0x6470abb50680944f] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickpushTick14133213201864676143_147a6dd0baa7ab92>>();
    };
    aotLib[0x89cc69e91131efea] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickemplace_newTick990257600204377963_b88a35d64549eac0>>();
    };
    aotLib[0x77fc06d975793138] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncastTickdescribeTick2562845734617055679_af1bd1c0c428d037>>();
    };
    aotLib[0x3fb2ff171baaa093] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickemplaceTick13923705686753630697_b3d3044d3eccbbb7>>();
    };
    aotLib[0x39947241dd456d71] = +[](Context & ctx) -> SimNode* {
        return ctx.code->makeNode<AutoSimNode_Aot<&_FuncbuiltinTickcloneTick9409548443506319159_e97a0907d62c3fa4>>();
    };
    resolveTypeInfoAnnotations();
}

} // namespace _anon_7436603895564257385
AotListBase impl_aot_apply(_anon_7436603895564257385::registerAotFunctions);
} // namespace das

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#if defined(__EDG__)
#pragma diag_default 826
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
