#include "daScript/daScript.h"

// These modules live in separate libraries and expose C-linkage registration
// entry points. Their declarations must therefore live at file scope.
DECLARE_MODULE(Module_Raster);
DECLARE_MODULE(Module_StbImage);

namespace host {

void register_static_modules() {
    NEED_ALL_DEFAULT_MODULES;
    NEED_MODULE(Module_Raster);
    NEED_MODULE(Module_StbImage);
}

}

int main() {
    host::register_static_modules();
    das::Module::Initialize();

    const bool found = das::Module::require("raster") != nullptr
                    && das::Module::require("stbimage") != nullptr;

    das::Module::Shutdown();
    return found ? 0 : 1;
}
