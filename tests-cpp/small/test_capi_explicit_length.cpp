#include <doctest/doctest.h>

#include "daScript/daScriptC.h"

#include <cstring>

namespace {

struct CApiProgram {
    das_text_writer * output = nullptr;
    das_module_group * modules = nullptr;
    das_file_access * files = nullptr;
    das_policies * policies = nullptr;
    das_program * program = nullptr;
    das_context * context = nullptr;

    ~CApiProgram() {
        if ( context ) das_context_release(context);
        if ( program ) das_program_release(program);
        if ( policies ) das_policies_release(policies);
        if ( files ) das_fileaccess_release(files);
        if ( modules ) das_modulegroup_release(modules);
        if ( output ) das_text_release(output);
    }
};

}
TEST_CASE("C API explicit-length strings accept non-null-terminated ranges") {
    const char file_name[] = {'m','a','i','n','.','d','a','s'};
    const char source[] = {
        'o','p','t','i','o','n','s',' ','g','e','n','2','\n',
        '[','e','x','p','o','r','t',']',' ','d','e','f',' ','m','a','i','n',' ','{','}','\n'
    };

    CApiProgram capi;
    capi.output = das_text_make_writer();
    capi.modules = das_modulegroup_make();
    capi.files = das_fileaccess_make_default();
    capi.policies = das_policies_make();

    das_fileaccess_introduce_file_n(capi.files,
        file_name, sizeof(file_name), source, sizeof(source), 1);
    capi.program = das_program_compile_policies_n(file_name, sizeof(file_name),
        capi.files, capi.output, capi.modules, capi.policies);
    REQUIRE(capi.program != nullptr);
    REQUIRE(das_program_err_count(capi.program) == 0);

    capi.context = das_context_make(das_program_context_stack_size(capi.program));
    REQUIRE(capi.context != nullptr);
    REQUIRE(das_program_simulate(capi.program, capi.context, capi.output) == 1);

    const char function_name[] = {'m','a','i','n'};
    CHECK(das_context_find_function_n(capi.context, function_name, sizeof(function_name)) != nullptr);
    const char function_name_terminated[] = "main";
    CHECK(das_context_find_function_n(capi.context, function_name_terminated, sizeof(function_name_terminated)) != nullptr);

    const char value[] = {'D','-','s','l','i','c','e'};
    char * allocated = das_allocate_string_n(capi.context, value, sizeof(value));
    REQUIRE(allocated != nullptr);
    CHECK(std::memcmp(allocated, value, sizeof(value)) == 0);
    CHECK(allocated[sizeof(value)] == 0);

    const char builtin_name[] = {'$'};
    CHECK(das_module_find_n(builtin_name, sizeof(builtin_name)) != nullptr);
    const char builtin_name_with_suffix[] = {'$','\0','x'};
    CHECK(das_module_find_n(builtin_name_with_suffix, sizeof(builtin_name_with_suffix)) == nullptr);

    char one_byte_root[1] = {'x'};
    das_get_root_n(one_byte_root, sizeof(one_byte_root));
    CHECK(one_byte_root[0] == 0);
    das_get_root_n(nullptr, 0);
}
