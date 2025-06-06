cmake_minimum_required (VERSION 3.16)
project (DAS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DAS_FLEX_BISON_DISABLED "Disable FLEX/BISON stage" ON)
option(DAS_CLANG_BIND_DISABLED "Disable dasClangBind (libclang bindings, C/C++ parsing)" ON)
option(DAS_LLVM_DISABLED "Disable dasLLVM (llvm bindings)" ON)
option(DAS_QUIRREL_DISABLED "Disable dasQuirrel (quirrel bindings)" ON)
option(DAS_HV_DISABLED "Disable dasHV (websokets,http server and client)" ON)
option(DAS_GLFW_DISABLED "Disable dasGLFW (GLFW window for graphics apps)" ON)
option(DAS_IMGUI_DISABLED "Disable dasIMGUI (IMGUI, IMNODES, IMGUI-NODE-EDITOR gui libraries)" ON)
option(DAS_BGFX_DISABLED "Disable dasBGFX (BGFX graphics API)" ON)
option(DAS_XBYAK_DISABLED "Disable dasXbyak (XBYAK and ZYDIS, x86 assembly, jit)" ON)
option(DAS_MINFFT_DISABLED "Disable dasMinfft (Minimal FFT library)" ON)
option(DAS_AUDIO_DISABLED "Disable dasAudio (Miniaudio sound library)" ON)
option(DAS_STDDLG_DISABLED "Disable dasStdDlg (File new,open,save etc dialogs)" ON)
option(DAS_STBIMAGE_DISABLED "Disable dasStbImage (StbImage bindings, image loading and saving)" ON)
option(DAS_STBTRUETYPE_DISABLED "Disable dasStbTrueType (StbTrueType bindings, ttf rasterization)" ON)
option(DAS_SFML_DISABLED "Disable dasSFML (SFML multimedia library)" ON)
option(DAS_PUGIXML_DISABLED "Disable dasPUGIXML (xml parsing library)" ON)
option(DAS_SQLITE_DISABLED "Disable dasSQLITE (sqlite3 library)" ON)

add_compile_options(-msimd128)
add_compile_options(-msse2)
add_compile_options(-mnontrapping-fptoint)

add_compile_options(-fwasm-exceptions)
add_link_options(-fwasm-exceptions)

add_compile_definitions(DAS_ENABLE_EXCEPTIONS)
add_compile_definitions(_EMSCRIPTEN_VER)

set(CMAKE_CXX_FLAGS "-s TOTAL_MEMORY=128MB")

INCLUDE(../CMakeCommon.txt)

IF(DEFINED DAS_CONFIG_INCLUDE_DIR)
    INCLUDE_DIRECTORIES(${DAS_CONFIG_INCLUDE_DIR})
ENDIF()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/output/)

SET(LINUX_UUID FALSE)

IF(NOT DAS_FLEX_BISON_DISABLED)
    FIND_FLEX_AND_BISON()
ENDIF()

SETUP_COMPILER()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

MACRO(ADD_STANDALONE_FILE genList input)
    get_filename_component(input_src ${input} ABSOLUTE)
    get_filename_component(input_dir ${input_src} DIRECTORY)
    get_filename_component(input_name ${input} NAME)
    set(out_dir ${input_dir}/_standalone_ctx_generated)
    set(out_inc ${out_dir}/${input_name}.h)
    set(out_src ${out_dir}/${input_name}.cpp)
    set_source_files_properties(${out_inc} ${out_src} PROPERTIES GENERATED TRUE)
    list(APPEND ${genList} ${out_inc} ${out_src})
ENDMACRO()

MACRO (ADD_AOT_EXT_FILE genList mainTarget input)
    get_filename_component(input_src ${input} ABSOLUTE)
    get_filename_component(input_dir ${input_src} DIRECTORY)
    get_filename_component(input_name ${input} NAME)
    set(out_dir ${input_dir}/_aot_generated)
    set(out_src "${out_dir}/${mainTarget}_${input_name}.cpp")
    set_source_files_properties(${out_src} PROPERTIES GENERATED TRUE)
    list(APPEND ${genList} ${out_src})
ENDMACRO()


MACRO(UNITIZE_BUILD input_dir genList)
    set(unitList)
    set(files ${${genList}})
    set(out_dir "${PROJECT_SOURCE_DIR}/${input_dir}/_aot_generated")
    set_source_files_properties(${files} PROPERTIES HEADER_FILE_ONLY true)
    set(unit_build_file "")
    set(file_index 0)
    set(batch_index 0)
    foreach(source_file ${files} )
        if(file_index EQUAL "0")
            set(unit_build_file "${out_dir}/unity_build_${batch_index}.cpp")
            FILE(WRITE ${unit_build_file} "// unity build batch ${batch_index}\n")
            list(APPEND unitList ${unit_build_file})
        endif()
        get_filename_component(input_name ${source_file} NAME)
        FILE(APPEND ${unit_build_file} "#include \"${input_name}\"\n")
        math(EXPR file_index "${file_index} + 1")
        if(file_index EQUAL ${UNITZE_BUILD_BATCH_SIZE})
            set(file_index 0)
            math(EXPR batch_index "${batch_index} + 1")
        endif()
    endforeach()
    list(APPEND ${genList} ${unitList})
ENDMACRO()

FIND_PROGRAM(CLANG_BIN_EXE clang)
IF(CLANG_BIN_EXE)
    MACRO(DAS_CPP_BIND_AST tgt generate_das header_from prefix src_dir inc_dirs extras)
        MESSAGE(STATUS "Generating C++ bindings target ${tgt} from ${header_from}")
        ADD_CUSTOM_TARGET(${tgt})
        ADD_DEPENDENCIES(${tgt} daScript)
        ADD_CUSTOM_COMMAND(
            TARGET ${tgt}
            WORKING_DIRECTORY ${src_dir}
            VERBATIM
            COMMAND ${CLANG_BIN_EXE} -Xclang -ast-dump=json -x c++ -c ${header_from} -I${inc_dirs} > ${header_from}.json 2> ${header_from}.json.log || cmd /c exit /b 0
            COMMENT "Generating JSON AST for ${header_from} to ${header_from}.json\ninclude directoryes are ${inc_dirs}\n"
        )
        ADD_CUSTOM_COMMAND(
            TARGET ${tgt}
            WORKING_DIRECTORY ${src_dir}
            VERBATIM
            COMMAND daScript ${generate_das} -args ${header_from}.json ${prefix} "${extras}"
            COMMENT "Generating C++ bindings from ${header_from}.json\n"
        )
    ENDMACRO()

    MACRO(DAS_OUTPUT_AST tgt header_from src_dir inc_dirs)
        MESSAGE(STATUS "Generating C++ bindings target ${tgt} from ${header_from}")
        ADD_CUSTOM_TARGET(${tgt})
        ADD_DEPENDENCIES(${tgt} daScript)
        ADD_CUSTOM_COMMAND(
            TARGET ${tgt}
            WORKING_DIRECTORY ${src_dir}
            VERBATIM
            COMMAND ${CLANG_BIN_EXE} -Xclang -ast-dump -x c++ -c ${header_from} -I${inc_dirs} > ${header_from}.ast 2> ${header_from}.ast.log || cmd /c exit /b 0
            COMMENT "Generating JSON AST for ${header_from} to ${header_from}.json\ninclude directoryes are ${inc_dirs}\n"
        )
    ENDMACRO()

    MACRO(DAS_OUTPUT_JSON tgt header_from src_dir inc_dirs)
        MESSAGE(STATUS "Generating C++ bindings target ${tgt} from ${header_from}")
        ADD_CUSTOM_TARGET(${tgt})
        ADD_DEPENDENCIES(${tgt} daScript)
        ADD_CUSTOM_COMMAND(
            TARGET ${tgt}
            WORKING_DIRECTORY ${src_dir}
            VERBATIM
            COMMAND ${CLANG_BIN_EXE} -Xclang -ast-dump=json -x c++ -c ${header_from} -I${inc_dirs} > ${header_from}.ast 2> ${header_from}.ast.log || cmd /c exit /b 0
            COMMENT "Generating JSON AST for ${header_from} to ${header_from}.json\ninclude directoryes are ${inc_dirs}\n"
        )
    ENDMACRO()


ELSE()
    MACRO(DAS_CPP_BIND_AST tgt generate_das header_from prefix src_dir inc_dirs)
        MESSAGE(STATUS "Skipping C++ bindings from ${header_from}")
    ENDMACRO()

    MACRO(DAS_OUTPUT_AST tgt header_from src_dir inc_dirs)
        MESSAGE(STATUS "Skipping ast dump from from ${header_from}")
    ENDMACRO()

ENDIF()

SET(DAS_MODULES_RESOLVE_INC ${PROJECT_SOURCE_DIR}/../include/modules/external_resolve.inc)
SET(DAS_MODULES_NEED_INC ${PROJECT_SOURCE_DIR}/../include/modules/external_need.inc)
FILE(WRITE ${DAS_MODULES_RESOLVE_INC}.temp "")
FILE(WRITE ${DAS_MODULES_NEED_INC}.temp "")

SET(DAS_MODULES_LIBS)
SET(DAS_ALL_EXAMPLES)
FILE(GLOB _modules RELATIVE "${PROJECT_SOURCE_DIR}/modules" "modules/*")
FOREACH(_module ${_modules})
    MACRO(ADD_MODULE_LIB lib)
        MESSAGE("REGISTER DAS MODULE ${lib}")
        LIST(APPEND DAS_MODULES_LIBS ${lib})
    ENDMACRO()

    MACRO(ADD_EXAMPLE tgt)
        MESSAGE("EXAMPLE ${tgt}")
        LIST(APPEND DAS_ALL_EXAMPLES ${tgt})
    ENDMACRO()

    FUNCTION(ADD_MODULE_CPP cpp)
        FILE(APPEND ${DAS_MODULES_NEED_INC}.temp "NEED_MODULE(Module_${cpp});\n")
    ENDFUNCTION()

    FUNCTION(ADD_MODULE_NATIVE native)
        FILE(APPEND ${DAS_MODULES_RESOLVE_INC}.temp "NATIVE_MODULE(\"daslib\",\"daslib\", ${_module}, ${native});\n")
    ENDFUNCTION()

    FUNCTION(ADD_MODULE_DAS category subfolder native)
        FILE(APPEND ${DAS_MODULES_RESOLVE_INC}.temp "NATIVE_MODULE(${category}, ${subfolder}, ${_module}, ${native});\n")
    ENDFUNCTION()


    #INCLUDE(modules/${_module}/CMakeLists.txt OPTIONAL)
ENDFOREACH()

FOREACH(_example ${DAS_ALL_EXAMPLES})
    MESSAGE("adding to ${_example} ${DAS_MODULES_LIBS}")
    TARGET_LINK_LIBRARIES(${_example} ${DAS_MODULES_LIBS})
    ADD_DEPENDENCIES(${_example} ${DAS_MODULES_LIBS})
ENDFOREACH()

ADD_CUSTOM_COMMAND(
    DEPENDS ${DAS_MODULES_NEED_INC}.temp
    OUTPUT ${DAS_MODULES_NEED_INC}
    VERBATIM
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DAS_MODULES_NEED_INC}.temp  ${DAS_MODULES_NEED_INC}
)

ADD_CUSTOM_COMMAND(
    DEPENDS ${DAS_MODULES_RESOLVE_INC}.temp
    OUTPUT ${DAS_MODULES_RESOLVE_INC}
    VERBATIM
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DAS_MODULES_RESOLVE_INC}.temp  ${DAS_MODULES_RESOLVE_INC}
)

ADD_CUSTOM_TARGET(need_and_resolve ALL DEPENDS ${DAS_MODULES_NEED_INC} ${DAS_MODULES_RESOLVE_INC})

# libUriParser

SET(URIPARSER_SRCS
../3rdparty/uriparser/src/UriCommon.c
../3rdparty/uriparser/src/UriCompare.c
../3rdparty/uriparser/src/UriEscape.c
../3rdparty/uriparser/src/UriFile.c
../3rdparty/uriparser/src/UriIp4.c
../3rdparty/uriparser/src/UriIp4Base.c
../3rdparty/uriparser/src/UriNormalize.c
../3rdparty/uriparser/src/UriNormalizeBase.c
../3rdparty/uriparser/src/UriParse.c
../3rdparty/uriparser/src/UriParseBase.c
../3rdparty/uriparser/src/UriQuery.c
../3rdparty/uriparser/src/UriRecompose.c
../3rdparty/uriparser/src/UriResolve.c
../3rdparty/uriparser/src/UriShorten.c
../3rdparty/uriparser/src/UriMemory.c
)

ADD_LIBRARY(libUriParser STATIC ${URIPARSER_SRCS})
SETUP_CPP11(libUriParser)
target_compile_definitions(libUriParser PUBLIC URIPARSER_BUILD_CHAR)
target_compile_definitions(libUriParser PUBLIC URI_STATIC_BUILD)
target_include_directories(libUriParser PUBLIC
  ${PROJECT_SOURCE_DIR}/../3rdparty/uriparser/include
)

# libDaScript

FLEX(../src/parser/ds_lexer.lpp)
BISON(../src/parser/ds_parser.ypp)

SET(PARSER_GENERATED_SRC
# parser 1
../src/parser/ds_parser.hpp
../src/parser/ds_parser.cpp
../src/parser/ds_parser.output
../src/parser/ds_lexer.cpp
../src/parser/lex.yy.h
# parser 2
../src/parser/ds2_parser.hpp
../src/parser/ds2_parser.cpp
../src/parser/ds2_parser.output
../src/parser/ds2_lexer.cpp
../src/parser/lex2.yy.h
)

SET(PARSER_SRC
../src/parser/ds_parser.ypp
../src/parser/ds_lexer.lpp
../src/parser/ds2_parser.ypp
../src/parser/ds2_lexer.lpp
../src/parser/parser_state.h
../src/parser/parser_impl.cpp
../src/parser/parser_impl.h
)


list(SORT PARSER_SRC)
SOURCE_GROUP_FILES("parser" PARSER_SRC)
SOURCE_GROUP_FILES("parser/generated" PARSER_GENERATED_SRC)

IF(MSVC)
    # Don't use precompiled headers with flex/bison generated files
    FOREACH(file ${PARSER_GENERATED_SRC})
        SET_SOURCE_FILES_PROPERTIES(${file} PROPERTIES COMPILE_FLAGS "/Y-" )
    ENDFOREACH()
ENDIF()

SET(VECMATH_SRC
../include/vecmath/dag_vecMath.h
../include/vecmath/dag_vecMathDecl.h
../include/vecmath/dag_vecMath_common.h
../include/vecmath/dag_vecMath_const.h
../include/vecmath/dag_vecMath_neon.h
../include/vecmath/dag_vecMath_pc_sse.h
)
list(SORT VECMATH_SRC)
SOURCE_GROUP_FILES("vecmath" VECMATH_SRC)

SET(AST_SRC
../src/ast/ast.cpp
../src/ast/ast_tls.cpp
../src/ast/ast_visitor.cpp
../src/ast/ast_generate.cpp
../src/ast/ast_simulate.cpp
../src/ast/ast_typedecl.cpp
../src/ast/ast_match.cpp
../src/ast/ast_module.cpp
../src/ast/ast_print.cpp
../src/ast/ast_aot_cpp.cpp
../src/ast/ast_infer_type.cpp
../src/ast/ast_lint.cpp
../src/ast/ast_allocate_stack.cpp
../src/ast/ast_derive_alias.cpp
../src/ast/ast_const_folding.cpp
../src/ast/ast_block_folding.cpp
../src/ast/ast_unused.cpp
../src/ast/ast_annotations.cpp
../src/ast/ast_export.cpp
../src/ast/ast_parse.cpp
../src/ast/ast_debug_info_helper.cpp
../src/ast/ast_handle.cpp
../include/daScript/ast/compilation_errors.h
../include/daScript/ast/ast_typedecl.h
../include/daScript/ast/ast_typefactory.h
../include/daScript/ast/ast.h
../include/daScript/ast/ast_expressions.h
../include/daScript/ast/ast_visitor.h
../include/daScript/ast/ast_generate.h
../include/daScript/ast/ast_match.h
../include/daScript/ast/ast_interop.h
../include/daScript/ast/ast_handle.h
../include/daScript/ast/ast_policy_types.h
)
list(SORT AST_SRC)
SOURCE_GROUP_FILES("ast" AST_SRC)

SET(BUILTIN_SRC
../include/daScript/builtin/ast_gen.inc
../include/daScript/builtin/debugapi_gen.inc


../src/builtin/module_builtin.h
../src/builtin/module_builtin.cpp
../src/builtin/module_builtin_misc_types.cpp
../src/builtin/module_builtin_runtime.cpp
../src/builtin/module_builtin_runtime_sort.cpp
../src/builtin/module_builtin_runtime_lockcheck.cpp
../src/builtin/module_builtin_vector.cpp
../src/builtin/module_builtin_vector_ctor.cpp
../src/builtin/module_builtin_array.cpp
../src/builtin/module_builtin_das.cpp
../src/builtin/module_builtin_math.cpp
../src/builtin/module_builtin_raster.cpp
../src/builtin/module_builtin_string.cpp
../src/builtin/module_builtin_rtti.h
../src/builtin/module_builtin_rtti.cpp
../src/builtin/module_builtin_ast.cpp
../src/builtin/module_builtin_ast_serialize.cpp
../src/builtin/module_builtin_ast_flags.cpp
../src/builtin/module_builtin_ast_annotations.cpp
../src/builtin/module_builtin_ast_annotations_1.cpp
../src/builtin/module_builtin_ast_annotations_2.cpp
../src/builtin/module_builtin_ast_annotations_3.cpp
../src/builtin/module_builtin_ast_adapters.cpp
../src/builtin/module_builtin_ast.h
../src/builtin/module_builtin_uriparser.h
../src/builtin/module_builtin_uriparser.cpp
../src/builtin/module_jit.cpp
../src/builtin/module_builtin_fio.cpp
../src/builtin/module_builtin_dasbind.cpp
../src/builtin/module_builtin_network.cpp
../src/builtin/module_builtin_debugger.cpp
../src/builtin/module_builtin_jobque.cpp
../src/builtin/module_file_access.cpp
../src/builtin/builtin.das.inc
../src/builtin/builtin.das
../src/builtin/fio.das.inc
../src/builtin/fio.das
../src/builtin/rtti.das.inc
../src/builtin/rtti.das
../src/builtin/ast.das.inc
../src/builtin/ast.das
../src/builtin/network.das.inc
../src/builtin/network.das
../src/builtin/debugger.das.inc
../src/builtin/debugger.das
)
list(SORT BUILTIN_SRC)
SOURCE_GROUP_FILES("module builtin" BUILTIN_SRC)
CMAKE_TEXT_XXD(libDaScript ../src/builtin/builtin.das)
CMAKE_TEXT_XXD(libDaScript ../src/builtin/fio.das)
CMAKE_TEXT_XXD(libDaScript ../src/builtin/rtti.das)
CMAKE_TEXT_XXD(libDaScript ../src/builtin/ast.das)
CMAKE_TEXT_XXD(libDaScript ../src/builtin/network.das)
CMAKE_TEXT_XXD(libDaScript ../src/builtin/debugger.das)

SET(MISC_SRC
../include/daScript/misc/enums.h
../include/daScript/misc/hal.h
../include/daScript/misc/fpe.h
../include/daScript/misc/copy_bytes.h
../include/daScript/misc/platform.h
../include/daScript/misc/vectypes.h
../include/daScript/misc/arraytype.h
../include/daScript/misc/rangetype.h
../include/daScript/misc/string_writer.h
../include/daScript/misc/type_name.h
../include/daScript/misc/memory_model.h
../include/daScript/misc/wyhash.h
../include/daScript/misc/anyhash.h
../include/daScript/misc/safebox.h
../include/daScript/misc/smart_ptr.h
../include/daScript/misc/free_list.h
../include/daScript/misc/sysos.h
../include/daScript/misc/callable.h
../include/daScript/misc/debug_break.h
../include/daScript/misc/instance_debugger.h
../include/daScript/misc/job_que.h
../include/daScript/misc/uric.h
../src/misc/sysos.cpp
../src/misc/string_writer.cpp
../src/misc/memory_model.cpp
../src/misc/job_que.cpp
../src/misc/free_list.cpp
../src/misc/daScriptC.cpp
../src/misc/uric.cpp
../src/misc/format.cpp
)
list(SORT MISC_SRC)
SOURCE_GROUP_FILES("misc" MISC_SRC)

SET(SIMULATE_FUSION_SRC
../src/simulate/simulate_fusion.cpp
../src/simulate/simulate_fusion_op1.cpp
../src/simulate/simulate_fusion_op1_return.cpp
../src/simulate/simulate_fusion_ptrfdr.cpp
../src/simulate/simulate_fusion_op2.cpp
../src/simulate/simulate_fusion_op2_set.cpp
../src/simulate/simulate_fusion_op2_bool.cpp
../src/simulate/simulate_fusion_op2_bin.cpp
../src/simulate/simulate_fusion_op2_vec.cpp
../src/simulate/simulate_fusion_op2_set_vec.cpp
../src/simulate/simulate_fusion_op2_bool_vec.cpp
../src/simulate/simulate_fusion_op2_bin_vec.cpp
../src/simulate/simulate_fusion_op2_scalar_vec.cpp
../src/simulate/simulate_fusion_at.cpp
../src/simulate/simulate_fusion_at_array.cpp
../src/simulate/simulate_fusion_tableindex.cpp
../src/simulate/simulate_fusion_misc_copy.cpp
../src/simulate/simulate_fusion_call1.cpp
../src/simulate/simulate_fusion_call2.cpp
../src/simulate/simulate_fusion_if.cpp
../include/daScript/simulate/simulate_fusion.h
../include/daScript/simulate/simulate_fusion_op1.h
../include/daScript/simulate/simulate_fusion_op1_impl.h
../include/daScript/simulate/simulate_fusion_op1_perm.h
../include/daScript/simulate/simulate_fusion_op1_set_impl.h
../include/daScript/simulate/simulate_fusion_op1_set_perm.h
../include/daScript/simulate/simulate_fusion_op1_reg.h
../include/daScript/simulate/simulate_fusion_op2.h
../include/daScript/simulate/simulate_fusion_op2_impl.h
../include/daScript/simulate/simulate_fusion_op2_perm.h
../include/daScript/simulate/simulate_fusion_op2_set_impl.h
../include/daScript/simulate/simulate_fusion_op2_set_perm.h
../include/daScript/simulate/simulate_fusion_op2_vec_settings.h
)
list(SORT SIMULATE_FUSION_SRC)
SOURCE_GROUP_FILES("fusion" SIMULATE_FUSION_SRC)

SET(SIMULATE_SRC
../src/hal/performance_time.cpp
../include/daScript/misc/performance_time.h
../src/hal/debug_break.cpp
../src/hal/project_specific.cpp
../src/hal/project_specific_file_info.cpp
../include/daScript/misc/network.h
../src/misc/network.cpp
../src/simulate/hash.cpp
../src/simulate/debug_info.cpp
../src/simulate/runtime_string.cpp
../src/simulate/runtime_array.cpp
../src/simulate/runtime_table.cpp
../src/simulate/runtime_profile.cpp
../src/simulate/standalone_ctx_utils.cpp
../src/simulate/simulate.cpp
../src/simulate/simulate_exceptions.cpp
../src/simulate/simulate_gc.cpp
../src/simulate/simulate_tracking.cpp
../src/simulate/simulate_visit.cpp
../src/simulate/simulate_print.cpp
../src/simulate/simulate_fn_hash.cpp
../src/simulate/simulate_instrument.cpp
../include/daScript/simulate/cast.h
../include/daScript/simulate/hash.h
../include/daScript/simulate/heap.h
../src/simulate/heap.cpp
../include/daScript/simulate/debug_info.h
../include/daScript/simulate/interop.h
../include/daScript/simulate/runtime_string.h
../include/daScript/simulate/runtime_string_delete.h
../include/daScript/simulate/runtime_array.h
../include/daScript/simulate/runtime_table.h
../include/daScript/simulate/runtime_table_nodes.h
../include/daScript/simulate/runtime_range.h
../include/daScript/simulate/runtime_profile.h
../include/daScript/simulate/runtime_matrices.h
../include/daScript/simulate/simulate.h
../include/daScript/simulate/simulate_nodes.h
../include/daScript/simulate/simulate_visit.h
../include/daScript/simulate/simulate_visit_op.h
../include/daScript/simulate/simulate_visit_op_undef.h
../include/daScript/simulate/sim_policy.h
../src/simulate/data_walker.cpp
../include/daScript/simulate/data_walker.h
../src/simulate/debug_print.cpp
../src/simulate/json_print.cpp
../include/daScript/simulate/debug_print.h
../include/daScript/simulate/for_each.h
../include/daScript/simulate/bind_enum.h
../include/daScript/simulate/bin_serializer.h
../src/simulate/bin_serializer.cpp
../include/daScript/simulate/aot.h
../include/daScript/simulate/aot_library.h
../include/daScript/simulate/aot_builtin.h
../include/daScript/simulate/aot_builtin_math.h
../include/daScript/simulate/aot_builtin_raster.h
../include/daScript/simulate/aot_builtin_matrix.h
../include/daScript/simulate/aot_builtin_time.h
../include/daScript/simulate/aot_builtin_string.h
../include/daScript/simulate/aot_builtin_fio.h
../include/daScript/simulate/aot_builtin_rtti.h
../include/daScript/simulate/aot_builtin_ast.h
../include/daScript/simulate/aot_builtin_debugger.h
../include/daScript/simulate/aot_builtin_jobque.h
../include/daScript/simulate/aot_builtin_dasbind.h
../include/daScript/simulate/aot_builtin_uriparser.h
../include/daScript/simulate/aot_builtin_jit.h
../include/daScript/simulate/fs_file_info.h
../src/simulate/fs_file_info.cpp
${DAS_MODULES_RESOLVE_INC}
)

SET(DASCRIPT_FMT_SRC
../utils/dasFormatter/fmt.cpp
../utils/dasFormatter/formatter.cpp
../utils/dasFormatter/helpers.cpp
../utils/dasFormatter/ds_parser.cpp
CACHE INTERNAL "DAS_DASCRIPT_FMT_SRC"
)

SET(AOT_STUB_SRC
../src/misc/aot_stub.cpp
)


list(SORT SIMULATE_SRC)
SOURCE_GROUP_FILES("simulate" SIMULATE_SRC)

SET(DAGOR_NOISE_SRC
../include/dag_noise/dag_uint_noise.h
)
SOURCE_GROUP_FILES("dag_noise" DAGOR_NOISE_SRC)

SET(FLAT_HASH_MAP_SRC
../include/ska/flat_hash_map.hpp
)
SOURCE_GROUP_FILES("flat_hash_map" FLAT_HASH_MAP_SRC)

SET(MAIN_SRC
../include/daScript/daScript.h
../include/daScript/daScriptC.h
../include/daScript/daScriptModule.h
../include/daScript/das_config.h
../include/daScript/das_project_specific.h
)
SOURCE_GROUP_FILES("main" MAIN_SRC)

file(GLOB DAS_LIB_SRC
"daslib/*.das"
)
list(SORT DAS_LIB_SRC)
SOURCE_GROUP_FILES("daslib" DSA_LIB_SRC)
list(SORT DAS_LIB_SRC)

add_custom_target(dasAotStub)
SET(AOT_GENERATED_SRC)
set(AotDaslibList
        ../daslib/functional.das
        ../daslib/json.das
        ../daslib/json_boost.das
        ../daslib/ast_boost.das
        ../daslib/templates_boost.das
        ../daslib/utf8_utils.das
        ../daslib/regex.das
        ../daslib/regex_boost.das
        ../daslib/strings_boost.das
        ../daslib/random.das
        ../daslib/math_boost.das

        ../src/das/ast/printer_flags_visitor.das
)
SET(AOT_GENERATED_SRC)
FOREACH(inF IN LISTS AotDaslibList)
    ADD_AOT_EXT_FILE(AOT_GENERATED_SRC dasAotStub ${inF}) # No need to do aot, since these files already aot'ed
ENDFOREACH()

set(StandaloneFilesList
        ../src/das/ast/ast_print.das
        ../src/das/ast/ast_aot_cpp.das
        ../src/das/ast/aot_constants.das
        ../src/das/ast/standalone_contexts.das
)

FOREACH(inF IN LISTS StandaloneFilesList)
    ADD_STANDALONE_FILE(AOT_GENERATED_SRC ${inF})
ENDFOREACH()


SOURCE_GROUP_FILES("aot stub" AOT_GENERATED_SRC)
#UNITIZE_BUILD("daslib" AOT_GENERATED_SRC)


include_directories(../include)
include_directories(${PROJECT_SOURCE_DIR}/../3rdparty/fmt/include)
ADD_LIBRARY(libDaScript ${PARSER_GENERATED_SRC} ${PARSER_SRC} ${VECMATH_SRC} ${AST_SRC} ${BUILTIN_SRC}
    ${MISC_SRC} ${SIMULATE_SRC} ${SIMULATE_FUSION_SRC} ${TEST_SRC} ${MAIN_SRC}
	${DAGOR_NOISE_SRC} ${FLAT_HASH_MAP_SRC} ${DAS_LIB_SRC} ${DASCRIPT_FMT_SRC} ${AOT_STUB_SRC} ${AOT_GENERATED_SRC})
ADD_DEPENDENCIES(libDaScript need_and_resolve)
ADD_PROJECT_XXD_DEPENDS(libDaScript)
target_include_directories(libDaScript PUBLIC
    ${DAS_SMMALLOC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/uriparser/include
)
target_link_libraries(libDaScript libUriParser)
IF(LINUX_UUID)
target_link_libraries(libDaScript uuid)
ENDIF()
IF(UNIX AND NOT APPLE)
    TARGET_LINK_LIBRARIES(libDaScript ${CMAKE_DL_LIBS})
ENDIF()
SETUP_CPP11(libDaScript)
#target_precompile_headers(libDaScript PUBLIC ../include/daScript/misc/platform.h)


if (NOT ${DAS_BUILD_TOOLS} MATCHES NO)
# Stand alone command line compiler

  SET(DASCRIPT_MAIN_SRC
  ../utils/daScript/main.cpp
  ${DAS_MODULES_NEED_INC}
  )
  SOURCE_GROUP_FILES("main" DASCRIPT_MAIN_SRC)

  add_executable(daScript ${DASCRIPT_MAIN_SRC})
  TARGET_LINK_LIBRARIES(daScript libDaScript Threads::Threads ${DAS_MODULES_LIBS})
  SETUP_CPP11(daScript)
  SETUP_LTO(daScript)


endif()


SET(PROFILE_MAIN_LIB_SRC
../examples/profile/test_profile.cpp
../examples/profile/test_profile.h
)
SOURCE_GROUP_FILES("source" PROFILE_MAIN_LIB_SRC)

ADD_LIBRARY(libDaScriptProfile ${PROFILE_MAIN_LIB_SRC})
SETUP_CPP11(libDaScriptProfile)
