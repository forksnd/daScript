cmake_minimum_required (VERSION 3.16)
project (DAS)
include(GNUInstallDirs)

# for now we gonna run with exceptions enabled
#add_compile_definitions(DAS_ENABLE_EXCEPTIONS=1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

option(DAS_FLEX_BISON_DISABLED "Disable FLEX/BISON stage" OFF)
option(DAS_CLANG_BIND_DISABLED "Disable dasClangBind (libclang bindings, C/C++ parsing)" ON)
option(DAS_LLVM_DISABLED "Disable dasLLVM (llvm bindings)" ON)
option(DAS_QUIRREL_DISABLED "Disable dasQuirrel (quirrel bindings)" ON)
option(DAS_HV_DISABLED "Disable dasHV (websokets,http server and client)" ON)
option(DAS_GLFW_DISABLED "Disable dasGLFW (GLFW window for graphics apps)" OFF)
option(DAS_IMGUI_DISABLED "Disable dasIMGUI (IMGUI, IMNODES, IMGUI-NODE-EDITOR gui libraries)" ON)
option(DAS_BGFX_DISABLED "Disable dasBGFX (BGFX graphics API)" ON)
option(DAS_XBYAK_DISABLED "Disable dasXbyak (XBYAK and ZYDIS, x86 assembly, jit)" ON)
option(DAS_MINFFT_DISABLED "Disable dasMinfft (Minimal FFT library)" ON)
option(DAS_AUDIO_DISABLED "Disable dasAudio (Miniaudio sound library)" ON)
option(DAS_STDDLG_DISABLED "Disable dasStdDlg (File new,open,save etc dialogs)" OFF)
option(DAS_STBIMAGE_DISABLED "Disable dasStbImage (StbImage bindings, image loading and saving)" OFF)
option(DAS_STBTRUETYPE_DISABLED "Disable dasStbTrueType (StbTrueType bindings, ttf rasterization)" OFF)
option(DAS_SFML_DISABLED "Disable dasSFML (SFML multimedia library)" ON)
option(DAS_PUGIXML_DISABLED "Disable dasPUGIXML (xml parsing library)" ON)
option(DAS_SQLITE_DISABLED "Disable dasSQLITE (sqlite3 library)" ON)
option(DAS_TOOLS_DISABLED "Disable dasTools" OFF)
option(DAS_AOT_EXAMPLES_DISABLED "Disable dasAOT examples" OFF)
option(DAS_PROFILE_DISABLED "Disable dasProfile" OFF)
option(DAS_TUTORIAL_DISABLED "Disable dasTutorial" OFF)
option(DAS_TESTS_DISABLED "Disable dasTests" OFF)
set(DAS_INSTALL_BINDIR bin CACHE STRING "Directory where to install binaries")
set(DAS_INSTALL_DOCDIR . CACHE STRING "Directory where to install documentation")
set(DAS_INSTALL_DASLIBDIR daslib CACHE STRING "Directory where to install daslib")
set(DAS_INSTALL_EXAMPLESDIR examples CACHE STRING "Directory where to install examples")
set(DAS_INSTALL_MODULESDIR modules CACHE STRING "Directory where to install modules")


IF(WIN32)
    IF(DAS_LLVM_DISABLED)
        # FOR NOW DISABLE EXCEPTIONS AND USE SETJMP/LONGJUMP, IF JIT IS ENABLED
        add_compile_definitions(DAS_ENABLE_EXCEPTIONS=1)
    ENDIF()
ENDIF()

INCLUDE(./CMakeCommon.txt)

IF(DEFINED DAS_CONFIG_INCLUDE_DIR)
    INCLUDE_DIRECTORIES(${DAS_CONFIG_INCLUDE_DIR})
ENDIF()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin/)

SET(LINUX_UUID FALSE)

IF(NOT DAS_FLEX_BISON_DISABLED)

# On macOS, search Homebrew for keg-only versions of Bison and Flex. Xcode does
# not provide new enough versions for us to use.
if (CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    execute_process(
        COMMAND brew --prefix bison
        RESULT_VARIABLE BREW_BISON
        OUTPUT_VARIABLE BREW_BISON_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (BREW_BISON EQUAL 0 AND EXISTS "${BREW_BISON_PREFIX}")
        message(STATUS "Found Bison keg installed by Homebrew at ${BREW_BISON_PREFIX}")
        set(BISON_EXECUTABLE "${BREW_BISON_PREFIX}/bin/bison")
    endif()

    execute_process(
        COMMAND brew --prefix flex
        RESULT_VARIABLE BREW_FLEX
        OUTPUT_VARIABLE BREW_FLEX_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (BREW_FLEX EQUAL 0 AND EXISTS "${BREW_FLEX_PREFIX}")
        message(STATUS "Found Flex keg installed by Homebrew at ${BREW_FLEX_PREFIX}")
        set(FLEX_EXECUTABLE "${BREW_FLEX_PREFIX}/bin/flex")
    endif()
endif()

    FIND_FLEX_AND_BISON()
ENDIF()

SETUP_COMPILER()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)


MACRO (ADD_AOT_EXT_FILE genList mainTarget input)
    get_filename_component(input_src ${input} ABSOLUTE)
    get_filename_component(input_dir ${input_src} DIRECTORY)
    get_filename_component(input_name ${input} NAME)
    set(out_dir ${input_dir}/_aot_generated)
    set(out_src "${out_dir}/${mainTarget}_${input_name}.cpp")
    set_source_files_properties(${out_src} PROPERTIES GENERATED TRUE)
    list(APPEND ${genList} ${out_src})
ENDMACRO()

MACRO(DAS_AOT_EXT input genList mainTarget dasAotTool dasAotToolArg)
    ADD_AOT_EXT_FILE(${genList} ${mainTarget} ${input})
    list(GET ${genList} -1 out_src)
    get_filename_component(input_src ${input} ABSOLUTE)
    get_filename_component(out_dir ${out_src} DIRECTORY)
    file(MAKE_DIRECTORY ${out_dir})
    if("${dasAotToolArg}" STREQUAL "-aotlib")
        set(CROSS_PLATFORM "-cross-platform")
    endif()
    ADD_CUSTOM_COMMAND(
        DEPENDS ${input_src}
        OUTPUT  ${out_src}
        COMMENT "AOT precompiling ${input_src} -> ${out_src}"
        COMMAND ${dasAotTool} ${dasAotToolArg} ${input_src} ${out_src} ${CROSS_PLATFORM}
    )
    set(custom_name ${mainTarget}_${input_name}_aot)
    ADD_CUSTOM_TARGET(${custom_name} DEPENDS ${out_src})
    set_source_files_properties(${out_src} PROPERTIES GENERATED TRUE)
    SET_TARGET_PROPERTIES(${custom_name} PROPERTIES FOLDER _${mainTarget}_aot)
    ADD_DEPENDENCIES(${mainTarget} ${custom_name})
ENDMACRO()

MACRO(DAS_AOT input genList mainTarget dasAotTool)
    DAS_AOT_EXT(${input} ${genList} ${mainTarget} ${dasAotTool} -aot)
ENDMACRO()

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

MACRO(DAS_AOT_STANDALONE input genList mainTarget dasAotTool dasAotToolArg)
    get_filename_component(ctx_name ${input} NAME_WE)
    ADD_STANDALONE_FILE(${genList} ${input})
    list(GET ${genList} -1 out_src)
    get_filename_component(out_dir ${out_src} DIRECTORY)
    file(MAKE_DIRECTORY ${out_dir})
    ADD_CUSTOM_COMMAND(
            DEPENDS ${input_src}
            OUTPUT  ${out_src}
            COMMENT "AOT precompiling ${input_src} -> ${out_src}"
            COMMAND ${dasAotTool} ${dasAotToolArg} ${input_src} ${out_dir} -cross-platform -standalone-context ${ctx_name} -standalone-class Standalone
    )
    set(custom_name ${mainTarget}_${input_name}_standalone)
    ADD_CUSTOM_TARGET(${custom_name} DEPENDS ${out_src})
    set_source_files_properties(${out_src} PROPERTIES GENERATED TRUE)
    SET_TARGET_PROPERTIES(${custom_name} PROPERTIES FOLDER _${mainTarget}_aot)
    ADD_DEPENDENCIES(${mainTarget} ${custom_name})
ENDMACRO()

MACRO(DAS_AOT_LIB input genList mainTarget dasAotTool)
    DAS_AOT_EXT(${input} ${genList} ${mainTarget} ${dasAotTool} -aotlib)
ENDMACRO()

MACRO(DAS_AOT_D input genList mainTarget dasAotTool)
    get_filename_component(input_src ${input} ABSOLUTE)
    get_filename_component(input_dir ${input_src} DIRECTORY)
    get_filename_component(input_name ${input} NAME)
    set(out_dir ${input_dir}/_aot_generated)
    set(out_src ${out_dir}/${input_name}.cpp)
    file(MAKE_DIRECTORY ${out_dir})
	ADD_CUSTOM_COMMAND(
		DEPENDS ${input_src}
        DEPENDS ${dasAotTool}
        OUTPUT  ${out_src}
        COMMENT "AOT precompiling ${input_src} -> ${out_src}"
        COMMAND ${dasAotTool} -aot ${input_src} ${out_src}
    )
    list(APPEND ${genList} ${out_src})
ENDMACRO()

SET(AOT_BATCH_SIZE 10)

MACRO(DAS_AOT_DIR inFiles genList mainTarget dasAotTool)
    set(file_index 0)
    set(batch_index 0)
    set(batch_list)
    FOREACH(inF ${inFiles})
        DAS_AOT_D(${inF} batch_list ${mainTarget} ${dasAotTool})
        math(EXPR file_index "${file_index} + 1")
        if(file_index EQUAL ${AOT_BATCH_SIZE})
            set(custom_name ${mainTarget}_${batch_index}_aot)
            ADD_CUSTOM_TARGET(${custom_name} ALL DEPENDS ${batch_list})
            ADD_DEPENDENCIES(${mainTarget} ${custom_name})
            list(APPEND ${genList} ${batch_list})
            set(file_index 0)
            math(EXPR batch_index "${batch_index} + 1")
            set(batch_list)
        endif()
    ENDFOREACH()
    if(NOT(file_index EQUAL "0"))
        set(custom_name ${mainTarget}_${batch_index}_aot)
        ADD_CUSTOM_TARGET(${custom_name} ALL DEPENDS ${batch_list})
        ADD_DEPENDENCIES(${mainTarget} ${custom_name})
        list(APPEND ${genList} ${batch_list})
    endif()
ENDMACRO()

SET(UNITZE_BUILD_BATCH_SIZE 10)

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
        ADD_DEPENDENCIES(${tgt} daslang)
        ADD_CUSTOM_COMMAND(
            TARGET ${tgt}
            WORKING_DIRECTORY ${src_dir}
            VERBATIM
            COMMAND ${CLANG_BIN_EXE} -Xclang -ast-dump=json -x c++ -c ${header_from} -I${inc_dirs} > ${header_from}.json 2> ${header_from}.json.log || cmd /c exit /b 0
            COMMENT "Generating JSON AST for ${header_from} to ${header_from}.json\ninclude directories are ${inc_dirs}\n"
        )
        ADD_CUSTOM_COMMAND(
            TARGET ${tgt}
            WORKING_DIRECTORY ${src_dir}
            VERBATIM
            COMMAND daslang ${generate_das} -args ${header_from}.json ${prefix} "${extras}"
            COMMENT "Generating C++ bindings from ${header_from}.json\n"
        )
    ENDMACRO()

    MACRO(DAS_OUTPUT_AST tgt header_from src_dir inc_dirs)
        MESSAGE(STATUS "Generating C++ bindings target ${tgt} from ${header_from}")
        ADD_CUSTOM_TARGET(${tgt})
        ADD_DEPENDENCIES(${tgt} daslang)
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
        ADD_DEPENDENCIES(${tgt} daslang)
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

SET(DAS_MODULES_RESOLVE_INC ${PROJECT_SOURCE_DIR}/include/modules/external_resolve.inc)
SET(DAS_MODULES_NEED_INC ${PROJECT_SOURCE_DIR}/include/modules/external_need.inc)
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


    INCLUDE(modules/${_module}/CMakeLists.txt OPTIONAL)
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
3rdparty/uriparser/src/UriCommon.c
3rdparty/uriparser/src/UriCompare.c
3rdparty/uriparser/src/UriEscape.c
3rdparty/uriparser/src/UriFile.c
3rdparty/uriparser/src/UriIp4.c
3rdparty/uriparser/src/UriIp4Base.c
3rdparty/uriparser/src/UriNormalize.c
3rdparty/uriparser/src/UriNormalizeBase.c
3rdparty/uriparser/src/UriParse.c
3rdparty/uriparser/src/UriParseBase.c
3rdparty/uriparser/src/UriQuery.c
3rdparty/uriparser/src/UriRecompose.c
3rdparty/uriparser/src/UriResolve.c
3rdparty/uriparser/src/UriShorten.c
3rdparty/uriparser/src/UriMemory.c
)

ADD_LIBRARY(libUriParser STATIC ${URIPARSER_SRCS})
SETUP_CPP11(libUriParser)
target_compile_definitions(libUriParser PUBLIC URIPARSER_BUILD_CHAR)
target_compile_definitions(libUriParser PUBLIC URI_STATIC_BUILD)
target_include_directories(libUriParser PUBLIC
  ${PROJECT_SOURCE_DIR}/3rdparty/uriparser/include
)

# libDaScript

FLEX(src/parser/ds_lexer.lpp)
FLEX(src/parser/ds2_lexer.lpp)
BISON(src/parser/ds_parser.ypp)
BISON(src/parser/ds2_parser.ypp)
FLEX(utils/dasFsormatter/ds_lexer.lpp)
BISON(utils/dasFormatter/ds_parser.ypp)

SET(PARSER_GENERATED_SRC
# 1st parser
src/parser/ds_parser.hpp
src/parser/ds_parser.cpp
src/parser/ds_parser.output
src/parser/ds_lexer.cpp
src/parser/lex.yy.h
# 2nd parser
src/parser/ds2_parser.hpp
src/parser/ds2_parser.cpp
src/parser/ds2_parser.output
src/parser/ds2_lexer.cpp
src/parser/lex2.yy.h
)

SET(PARSER_SRC
src/parser/ds_parser.ypp
src/parser/ds_lexer.lpp
src/parser/ds2_parser.ypp
src/parser/ds2_lexer.lpp
src/parser/parser_state.h
src/parser/parser_impl.cpp
src/parser/parser_impl.h
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
include/vecmath/dag_vecMath.h
include/vecmath/dag_vecMathDecl.h
include/vecmath/dag_vecMath_common.h
include/vecmath/dag_vecMath_const.h
include/vecmath/dag_vecMath_neon.h
include/vecmath/dag_vecMath_pc_sse.h
)
list(SORT VECMATH_SRC)
SOURCE_GROUP_FILES("vecmath" VECMATH_SRC)

SET(AST_SRC
src/ast/ast.cpp
src/ast/ast_tls.cpp
src/ast/ast_visitor.cpp
src/ast/ast_generate.cpp
src/ast/ast_simulate.cpp
src/ast/ast_typedecl.cpp
src/ast/ast_match.cpp
src/ast/ast_module.cpp
src/ast/ast_print.cpp
src/ast/ast_aot_cpp.cpp
src/ast/ast_infer_type.cpp
src/ast/ast_lint.cpp
src/ast/ast_allocate_stack.cpp
src/ast/ast_derive_alias.cpp
src/ast/ast_const_folding.cpp
src/ast/ast_block_folding.cpp
src/ast/ast_unused.cpp
src/ast/ast_annotations.cpp
src/ast/ast_export.cpp
src/ast/ast_parse.cpp
src/ast/ast_debug_info_helper.cpp
src/ast/ast_handle.cpp
include/daScript/ast/compilation_errors.h
include/daScript/ast/ast_typedecl.h
include/daScript/ast/ast_typefactory.h
include/daScript/ast/ast.h
include/daScript/ast/ast_expressions.h
include/daScript/ast/ast_visitor.h
include/daScript/ast/ast_generate.h
include/daScript/ast/ast_match.h
include/daScript/ast/ast_interop.h
include/daScript/ast/ast_handle.h
include/daScript/ast/ast_policy_types.h
)
list(SORT AST_SRC)
SOURCE_GROUP_FILES("ast" AST_SRC)

SET(BUILTIN_SRC
include/daScript/builtin/ast_gen.inc
include/daScript/builtin/debugapi_gen.inc

src/builtin/module_builtin.h
src/builtin/module_builtin.cpp
src/builtin/module_builtin_misc_types.cpp
src/builtin/module_builtin_runtime.cpp
src/builtin/module_builtin_runtime_sort.cpp
src/builtin/module_builtin_runtime_lockcheck.cpp
src/builtin/module_builtin_vector.cpp
src/builtin/module_builtin_vector_ctor.cpp
src/builtin/module_builtin_array.cpp
src/builtin/module_builtin_das.cpp
src/builtin/module_builtin_math.cpp
src/builtin/module_builtin_raster.cpp
src/builtin/module_builtin_string.cpp
src/builtin/module_builtin_rtti.h
src/builtin/module_builtin_rtti.cpp
src/builtin/module_builtin_ast.cpp
src/builtin/module_builtin_ast_serialize.cpp
src/builtin/module_builtin_ast_flags.cpp
src/builtin/module_builtin_ast_annotations.cpp
src/builtin/module_builtin_ast_annotations_1.cpp
src/builtin/module_builtin_ast_annotations_2.cpp
src/builtin/module_builtin_ast_annotations_3.cpp
src/builtin/module_builtin_ast_adapters.cpp
src/builtin/module_builtin_ast.h
src/builtin/module_builtin_uriparser.h
src/builtin/module_builtin_uriparser.cpp
src/builtin/module_jit.cpp
src/builtin/module_builtin_fio.cpp
src/builtin/module_builtin_dasbind.cpp
src/builtin/module_builtin_network.cpp
src/builtin/module_builtin_debugger.cpp
src/builtin/module_builtin_jobque.cpp
src/builtin/module_file_access.cpp
src/builtin/builtin.das.inc
src/builtin/builtin.das
src/builtin/fio.das.inc
src/builtin/fio.das
src/builtin/rtti.das.inc
src/builtin/rtti.das
src/builtin/ast.das.inc
src/builtin/ast.das
src/builtin/network.das.inc
src/builtin/network.das
src/builtin/debugger.das.inc
src/builtin/debugger.das
)
list(SORT BUILTIN_SRC)
SOURCE_GROUP_FILES("module builtin" BUILTIN_SRC)
CMAKE_TEXT_XXD(libDaScript src/builtin/builtin.das)
CMAKE_TEXT_XXD(libDaScript src/builtin/fio.das)
CMAKE_TEXT_XXD(libDaScript src/builtin/rtti.das)
CMAKE_TEXT_XXD(libDaScript src/builtin/ast.das)
CMAKE_TEXT_XXD(libDaScript src/builtin/network.das)
CMAKE_TEXT_XXD(libDaScript src/builtin/debugger.das)

SET(MISC_SRC
include/daScript/misc/enums.h
include/daScript/misc/hal.h
include/daScript/misc/fpe.h
include/daScript/misc/copy_bytes.h
include/daScript/misc/platform.h
include/daScript/misc/vectypes.h
include/daScript/misc/arraytype.h
include/daScript/misc/rangetype.h
include/daScript/misc/string_writer.h
include/daScript/misc/type_name.h
include/daScript/misc/memory_model.h
include/daScript/misc/wyhash.h
include/daScript/misc/anyhash.h
include/daScript/misc/safebox.h
include/daScript/misc/smart_ptr.h
include/daScript/misc/free_list.h
include/daScript/misc/sysos.h
include/daScript/misc/callable.h
include/daScript/misc/debug_break.h
include/daScript/misc/instance_debugger.h
include/daScript/misc/job_que.h
include/daScript/misc/uric.h
src/misc/sysos.cpp
src/misc/string_writer.cpp
src/misc/memory_model.cpp
src/misc/job_que.cpp
src/misc/free_list.cpp
src/misc/daScriptC.cpp
src/misc/uric.cpp
src/misc/format.cpp
)
list(SORT MISC_SRC)
SOURCE_GROUP_FILES("misc" MISC_SRC)

SET(SIMULATE_FUSION_SRC
src/simulate/simulate_fusion.cpp
src/simulate/simulate_fusion_op1.cpp
src/simulate/simulate_fusion_op1_return.cpp
src/simulate/simulate_fusion_ptrfdr.cpp
src/simulate/simulate_fusion_op2.cpp
src/simulate/simulate_fusion_op2_set.cpp
src/simulate/simulate_fusion_op2_bool.cpp
src/simulate/simulate_fusion_op2_bin.cpp
src/simulate/simulate_fusion_op2_vec.cpp
src/simulate/simulate_fusion_op2_set_vec.cpp
src/simulate/simulate_fusion_op2_bool_vec.cpp
src/simulate/simulate_fusion_op2_bin_vec.cpp
src/simulate/simulate_fusion_op2_scalar_vec.cpp
src/simulate/simulate_fusion_at.cpp
src/simulate/simulate_fusion_at_array.cpp
src/simulate/simulate_fusion_tableindex.cpp
src/simulate/simulate_fusion_misc_copy.cpp
src/simulate/simulate_fusion_call1.cpp
src/simulate/simulate_fusion_call2.cpp
src/simulate/simulate_fusion_if.cpp
include/daScript/simulate/simulate_fusion.h
include/daScript/simulate/simulate_fusion_op1.h
include/daScript/simulate/simulate_fusion_op1_impl.h
include/daScript/simulate/simulate_fusion_op1_perm.h
include/daScript/simulate/simulate_fusion_op1_set_impl.h
include/daScript/simulate/simulate_fusion_op1_set_perm.h
include/daScript/simulate/simulate_fusion_op1_reg.h
include/daScript/simulate/simulate_fusion_op2.h
include/daScript/simulate/simulate_fusion_op2_impl.h
include/daScript/simulate/simulate_fusion_op2_perm.h
include/daScript/simulate/simulate_fusion_op2_set_impl.h
include/daScript/simulate/simulate_fusion_op2_set_perm.h
include/daScript/simulate/simulate_fusion_op2_vec_settings.h
)
list(SORT SIMULATE_FUSION_SRC)
SOURCE_GROUP_FILES("fusion" SIMULATE_FUSION_SRC)

SET(SIMULATE_SRC
src/hal/performance_time.cpp
include/daScript/misc/performance_time.h
src/hal/debug_break.cpp
src/hal/project_specific.cpp
src/hal/project_specific_file_info.cpp
include/daScript/misc/network.h
src/misc/network.cpp
src/simulate/hash.cpp
src/simulate/debug_info.cpp
src/simulate/runtime_string.cpp
src/simulate/runtime_array.cpp
src/simulate/runtime_table.cpp
src/simulate/runtime_profile.cpp
src/simulate/standalone_ctx_utils.cpp
src/simulate/simulate.cpp
src/simulate/simulate_exceptions.cpp
src/simulate/simulate_gc.cpp
src/simulate/simulate_tracking.cpp
src/simulate/simulate_visit.cpp
src/simulate/simulate_print.cpp
src/simulate/simulate_fn_hash.cpp
src/simulate/simulate_instrument.cpp
include/daScript/simulate/cast.h
include/daScript/simulate/hash.h
include/daScript/simulate/heap.h
src/simulate/heap.cpp
include/daScript/simulate/debug_info.h
include/daScript/simulate/interop.h
include/daScript/simulate/runtime_string.h
include/daScript/simulate/runtime_string_delete.h
include/daScript/simulate/runtime_array.h
include/daScript/simulate/runtime_table.h
include/daScript/simulate/runtime_table_nodes.h
include/daScript/simulate/runtime_range.h
include/daScript/simulate/runtime_profile.h
include/daScript/simulate/runtime_matrices.h
include/daScript/simulate/simulate.h
include/daScript/simulate/simulate_nodes.h
include/daScript/simulate/simulate_visit.h
include/daScript/simulate/simulate_visit_op.h
include/daScript/simulate/simulate_visit_op_undef.h
include/daScript/simulate/sim_policy.h
src/simulate/data_walker.cpp
include/daScript/simulate/data_walker.h
src/simulate/debug_print.cpp
src/simulate/json_print.cpp
include/daScript/simulate/debug_print.h
include/daScript/simulate/for_each.h
include/daScript/simulate/bind_enum.h
include/daScript/simulate/bin_serializer.h
src/simulate/bin_serializer.cpp
include/daScript/simulate/aot.h
include/daScript/simulate/aot_library.h
include/daScript/simulate/aot_builtin.h
include/daScript/simulate/aot_builtin_math.h
include/daScript/simulate/aot_builtin_raster.h
include/daScript/simulate/aot_builtin_matrix.h
include/daScript/simulate/aot_builtin_time.h
include/daScript/simulate/aot_builtin_string.h
include/daScript/simulate/aot_builtin_fio.h
include/daScript/simulate/aot_builtin_rtti.h
include/daScript/simulate/aot_builtin_ast.h
include/daScript/simulate/aot_builtin_debugger.h
include/daScript/simulate/aot_builtin_jobque.h
include/daScript/simulate/aot_builtin_dasbind.h
include/daScript/simulate/aot_builtin_uriparser.h
include/daScript/simulate/aot_builtin_jit.h
include/daScript/simulate/fs_file_info.h
src/simulate/fs_file_info.cpp
${DAS_MODULES_RESOLVE_INC}
)

# das formatter
SET(DASCRIPT_FMT_SRC
        ${PROJECT_SOURCE_DIR}/utils/dasFormatter/fmt.cpp
        ${PROJECT_SOURCE_DIR}/utils/dasFormatter/formatter.cpp
        ${PROJECT_SOURCE_DIR}/utils/dasFormatter/helpers.cpp
        ${PROJECT_SOURCE_DIR}/utils/dasFormatter/ds_parser.cpp
        CACHE INTERNAL "DAS_DASCRIPT_FMT_SRC"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/builtin/module_builtin_ast_serialize.cpp PROPERTIES
    COMPILE_FLAGS "-fexceptions"
  )
elseif(MSVC)
  set_source_files_properties(src/builtin/module_builtin_ast_serialize.cpp PROPERTIES
    COMPILE_FLAGS "/EHsc"
  )
endif()

list(SORT SIMULATE_SRC)
SOURCE_GROUP_FILES("simulate" SIMULATE_SRC)

SET(DAGOR_NOISE_SRC
include/dag_noise/dag_uint_noise.h
)
SOURCE_GROUP_FILES("dag_noise" DAGOR_NOISE_SRC)

SET(FLAT_HASH_MAP_SRC
include/ska/flat_hash_map.hpp
)
SOURCE_GROUP_FILES("flat_hash_map" FLAT_HASH_MAP_SRC)

SET(FAST_FLOAT_SRC
include/fast_float/fast_float.h
)

SET(MAIN_SRC
include/daScript/daScript.h
include/daScript/daScriptC.h
include/daScript/daScriptModule.h
include/daScript/das_config.h
include/daScript/das_project_specific.h
)
SOURCE_GROUP_FILES("main" MAIN_SRC)

file(GLOB DAS_LIB_SRC
"daslib/*.das"
)
list(SORT DAS_LIB_SRC)
SOURCE_GROUP_FILES("daslib" DSA_LIB_SRC)
list(SORT DAS_LIB_SRC)

include_directories(include)
include_directories(3rdparty/fmt/include)
set(AotDaslibList
        # daslib/algorithm.das
        # daslib/apply.das
        # daslib/apply_in_context.das
        # daslib/archive.das
        # daslib/array_boost.das
        # daslib/assert_once.das
        # daslib/ast_block_to_loop.das
        daslib/ast_boost.das
        # daslib/ast_debug.das
        # daslib/ast_used.das
        # daslib/async_boost.das
        # daslib/base64.das
        # daslib/bitfield_trait.das
        # daslib/constant_expression.das
        # daslib/contracts.das
        # daslib/coroutines.das
        # daslib/cpp_bind.das
        # daslib/cpp_gen.das
        # daslib/dap.das
        # daslib/das_source_formatter.das
        # daslib/das_source_formatter_fio.das
        # daslib/debug.das
        # daslib/debug_eval.das
        # daslib/decs_boost.das
        # daslib/decs.das
        # daslib/decs_state.das
        # daslib/defer.das
        # daslib/dynamic_cast_rtti.das
        # daslib/enum_trait.das
        # daslib/export_constructor.das
        # daslib/faker.das
        daslib/functional.das
        # daslib/fuzzer.das
        # daslib/generic_return.das
        # daslib/hash_map.das
        # daslib/heartbeat.das
        # daslib/if_not_null.das
        # daslib/instance_function.das
        # daslib/interfaces.das
        # daslib/is_local.das
        # daslib/jobque_boost.das # We need to add it in a stub. Will do it later for all files
        daslib/json_boost.das
        daslib/json.das
        # daslib/just_in_time.das # jit disabled by default
        # daslib/linked_list.das
        # daslib/lint.das
        # daslib/lint_everything.das
        # daslib/live.das # depends on filesystem
        # daslib/lpipe.das
        # daslib/macro_boost.das
        # daslib/match.das
        # daslib/math_bits.das
        daslib/math_boost.das
        # daslib/profiler_boost.das
        # daslib/profiler.das
        daslib/random.das
        # daslib/raster_boost.das # fails on windows
        # daslib/refactor.das
        daslib/regex_boost.das
        daslib/regex.das
        # daslib/remove_call_args.das
        # daslib/rst_comment.das
        # daslib/rst.das # depends on filesystem
        # daslib/safe_addr.das
        # daslib/soa.das
        # daslib/sort_boost.das
        # daslib/spoof.das
        # daslib/static_let.das
        # daslib/stringify.das
        daslib/strings_boost.das
        # daslib/stub.das
        daslib/templates_boost.das
        # daslib/templates.das
        # daslib/temp_strings.das
        # daslib/typemacro_boost.das
        # daslib/type_traits.das
        # daslib/unroll.das
        # daslib/uriparser_boost.das
        daslib/utf8_utils.das
        # daslib/validate_code.das

        # src/das/ast/printer_flags_visitor.das # aot das-mode temporary disabled
)

SET(AOT_GENERATED_SRC)
FOREACH(inF IN LISTS AotDaslibList)
    ADD_AOT_EXT_FILE(AOT_GENERATED_SRC dasAotStub ${inF})
ENDFOREACH()

# one-shot target
set(AOT_STAGE0_PATH ${CMAKE_CURRENT_BINARY_DIR}/_daslib_gen_stage0)
foreach(FILE ${AOT_GENERATED_SRC})
    get_filename_component(FILENAME ${FILE} NAME)
    set(DST_FILE ${AOT_STAGE0_PATH}/${FILENAME})
    if (EXISTS ${FILE}) # if file doesn't exist - ignore it. It will be generated on next iteration
        add_custom_command(
                OUTPUT ${DST_FILE}
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FILE}
                ${DST_FILE}
        )
        list(APPEND AOT_GENERATED_SRC_STAGE0 "${DST_FILE}")
    endif()
endforeach()

add_custom_target(dasAotStub)
SOURCE_GROUP_FILES("aot stub" AOT_GENERATED_SRC)

set(StandaloneFilesList
    # # aot das-mode temporary disabled
    # src/das/ast/ast_print.das
    # src/das/ast/ast_aot_cpp.das
    # src/das/ast/aot_constants.das
    # src/das/ast/standalone_contexts.das
)

SET(STANDALONE_SRC)
FOREACH(inF IN LISTS StandaloneFilesList)
    ADD_STANDALONE_FILE(STANDALONE_SRC ${inF})
ENDFOREACH()

set(STANDALONE_STAGE0_PATH ${CMAKE_CURRENT_BINARY_DIR}/_standalone_gen_stage0)
foreach(FILE ${STANDALONE_SRC})
    get_filename_component(FILENAME ${FILE} NAME)
    if (EXISTS ${FILE}) # if file doesn't exist - ignore it. It will be generated on next iteration
        set(DST_FILE ${STANDALONE_STAGE0_PATH}/${FILENAME})
        add_custom_command(
                OUTPUT ${DST_FILE}
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${FILE}
                ${DST_FILE}
        )
        list(APPEND STANDALONE_SRC_STAGE0 "${DST_FILE}")
    endif()
endforeach()

SET(AOT_STUB_SRC
        src/misc/aot_stub.cpp
)

MACRO(SETUP_LIBDASCRIPT library aotFiles standaloneFiles)
    ADD_LIBRARY(${library} STATIC ${PARSER_GENERATED_SRC} ${PARSER_SRC} ${VECMATH_SRC} ${AST_SRC} ${BUILTIN_SRC}
            ${MISC_SRC} ${SIMULATE_SRC} ${SIMULATE_FUSION_SRC} ${TEST_SRC} ${MAIN_SRC}
            ${DAGOR_NOISE_SRC} ${FLAT_HASH_MAP_SRC} ${FAST_FLOAT_SRC} ${DASCRIPT_FMT_SRC}
            ${standaloneFiles} ${aotFiles})
    ADD_DEPENDENCIES(${library} need_and_resolve)
    ADD_TARGET_PROJECT_XXD_DEPENDS(${library} libDaScript)
    target_include_directories(${library} PUBLIC
            ${DAS_SMMALLOC_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/uriparser/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(${library} libUriParser)
    #target_link_libraries(${library} fmt::fmt)
    IF(LINUX_UUID)
        target_link_libraries(${library} uuid)
    ENDIF()
    IF(UNIX AND NOT APPLE)
        TARGET_LINK_LIBRARIES(${library} ${CMAKE_DL_LIBS})
    ENDIF()
    IF(HAIKU)
        TARGET_LINK_LIBRARIES(${library} network uuid)
    ENDIF()
    SETUP_CPP11(${library})
    #target_precompile_headers(${library} PUBLIC include/daScript/misc/platform.h)
ENDMACRO()

SETUP_LIBDASCRIPT(libDaScriptStage0 "${AOT_GENERATED_SRC_STAGE0}" "${STANDALONE_SRC_STAGE0}")
SETUP_LIBDASCRIPT(libDaScript "${AOT_GENERATED_SRC}" "${STANDALONE_SRC}")

if(WIN32)
    if (CMAKE_CONFIGURATION_TYPES)
        set(DASLANG_STAGE0_PATH "${PROJECT_SOURCE_DIR}/${DAS_INSTALL_BINDIR}/$<CONFIG>/daslang.exe")
    else()
        set(DASLANG_STAGE0_PATH "${PROJECT_SOURCE_DIR}/${DAS_INSTALL_BINDIR}/daslang.exe")
    endif()
else()
    set(DASLANG_STAGE0_PATH "${PROJECT_SOURCE_DIR}/${DAS_INSTALL_BINDIR}/daslang")
endif()
add_custom_target(standaloneGeneratedSrc)

FOREACH(inF IN LISTS StandaloneFilesList)
    DAS_AOT_STANDALONE(${inF} unused_standaloneSrc standaloneGeneratedSrc ${DASLANG_STAGE0_PATH} "-aotlib")
ENDFOREACH()


FOREACH(inF IN LISTS AotDaslibList)
    DAS_AOT_LIB("${inF}" unused_aot_src dasAotStub ${DASLANG_STAGE0_PATH})
ENDFOREACH()

ADD_DEPENDENCIES(libDaScript standaloneGeneratedSrc dasAotStub)

if(NOT ${DAS_AOT_EXAMPLES_DISABLED})

    if(${DAS_TOOLS_DISABLED})
        # error, we need tools to generate aot
        message(FATAL_ERROR "DAS_AOT_EXAMPLES_DISABLED requires DAS_TOOLS_DISABLED to be OFF")
    endif()

    #UNITIZE_BUILD("daslib" AOT_GENERATED_SRC)

    add_library(libDaScriptAot STATIC ${AOT_GENERATED_SRC} ${AOT_STUB_SRC})
    ADD_DEPENDENCIES(libDaScriptAot dasAotStub libDaScript)
    TARGET_LINK_LIBRARIES(libDaScriptAot libDaScript)
    SETUP_CPP11(libDaScriptAot)
endif()

SET(DAS_DASCRIPT_MAIN_SRC
    ${PROJECT_SOURCE_DIR}/utils/daScript/main.cpp
    ${DAS_MODULES_NEED_INC}
    CACHE INTERNAL "DAS_DASCRIPT_MAIN_SRC"
)

if (NOT ${DAS_TOOLS_DISABLED})
# Stand alone command line compiler

    SOURCE_GROUP_FILES("main" DAS_DASCRIPT_MAIN_SRC)

    SET(SRC_LIBRARIES libDaScriptTest Threads::Threads ${DAS_MODULES_LIBS})

    MACRO(SETUP_COMPILER_BINARY target_name library_name)
        add_executable(${target_name} ${DAS_DASCRIPT_MAIN_SRC})
        TARGET_LINK_LIBRARIES(${target_name} ${library_name} ${SRC_LIBRARIES})
        ADD_DEPENDENCIES(${target_name} ${library_name} libDaScriptTest ${DAS_MODULES_LIBS})
        SETUP_CPP11(${target_name})
        SETUP_LTO(${target_name})
    ENDMACRO()

    SETUP_COMPILER_BINARY(daslang libDaScriptStage0)
    INSTALL(TARGETS daslang
            RUNTIME DESTINATION ${DAS_INSTALL_BINDIR}
    )
    set_property(TARGET daslang PROPERTY GENERATED TRUE)
    SETUP_COMPILER_BINARY(daslang-dev libDaScript)

    file(GLOB DASLIB_SOURCES
        ${PROJECT_SOURCE_DIR}/daslib/*.das
    )
    install(FILES ${DASLIB_SOURCES}
        DESTINATION ${DAS_INSTALL_DASLIBDIR}
    )

    add_executable(das-fmt ${PROJECT_SOURCE_DIR}/utils/dasFormatter/main.cpp)
    TARGET_LINK_LIBRARIES(das-fmt libDaScript ${SRC_LIBRARIES})

    SETUP_CPP11(das-fmt)
    SETUP_LTO(das-fmt)

    #IF(APPLE)
    #  add_executable(daslangOsx MACOSX_BUNDLE ${DAS_DASCRIPT_MAIN_SRC})
    #  TARGET_LINK_LIBRARIES(daslangOsx libDaScript libDaScriptTest Threads::Threads ${DAS_MODULES_LIBS})
    #  ADD_DEPENDENCIES(daslangOsx libDaScript libDaScriptTest  need_and_resolve ${DAS_MODULES_LIBS})
    #  SETUP_CPP11(daslangOsx)
    #  SETUP_LTO(daslangOsx)
    #  set_target_properties(daslangOsx PROPERTIES MACOSX_BUNDLE_BUNDLE_NAME "daslang")
    #  set_target_properties(daslangOsx PROPERTIES
    #      MACOSX_BUNDLE_SHORT_VERSION_STRING "0.5"
    #      MACOSX_BUNDLE_LONG_VERSION_STRING  "daslang 0.5"
    #      MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/utils/daScript/MacOSXBundleInfo.plist.in)
    #ENDIF()

endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/dastest DESTINATION .) # It's temporary, daslib should not depend on dastest

install(FILES ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${DAS_INSTALL_DOCDIR})
install(FILES ${PROJECT_SOURCE_DIR}/3rdparty/uriparser/COPYING DESTINATION ${DAS_INSTALL_DOCDIR} RENAME URIPARSER.LICENSE)
install(FILES ${PROJECT_SOURCE_DIR}/include/dag_noise/LICENSE DESTINATION ${DAS_INSTALL_DOCDIR} RENAME DAG_NOISE.LICENSE)
install(FILES ${PROJECT_SOURCE_DIR}/include/ska/LICENSE_1_0.txt DESTINATION ${DAS_INSTALL_DOCDIR} RENAME SKA.LICENSE)
install(FILES ${PROJECT_SOURCE_DIR}/include/vecmath/LICENSE DESTINATION ${DAS_INSTALL_DOCDIR} RENAME VEC_MATH.LICENSE)


# Test module

SET(TEST_MAIN_LIB_SRC
examples/test/bytecode.cpp
examples/test/bytecode.h
examples/test/test_handles.cpp
examples/test/test_enum.cpp
examples/test/module_unitTest.h
examples/test/unitTest.h
examples/test/unit_test.das.inc
examples/test/unit_test.das
)
SOURCE_GROUP_FILES("source" TEST_MAIN_LIB_SRC)



# Tests
if (NOT ${DAS_TESTS_DISABLED})
    include(examples/test/CMakeLists.txt)
    include(examples/pathTracer/CMakeLists.txt)
endif()

# Profile module

if (NOT ${DAS_PROFILE_DISABLED})
    include(examples/profile/CMakeLists.txt)
endif()

if (NOT ${DAS_TUTORIAL_DISABLED})
    include(examples/tutorial/CMakeLists.txt)
endif()
