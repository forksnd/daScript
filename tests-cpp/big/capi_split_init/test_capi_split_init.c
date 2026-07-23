#include "daScript/daScriptC.h"

// Static modules intentionally expose these exact unmangled C entry points.
extern das_module * register_Module_Raster(void);
extern das_module * register_Module_StbImage(void);

int main(void) {
    const char raster_name[] = {'r','a','s','t','e','r'};
    const char stbimage_name[] = {'s','t','b','i','m','a','g','e'};

    das_initialize_modules();
    if (!register_Module_Raster()) return 1;
    if (!register_Module_StbImage()) return 2;
    das_initialize_finalize();

    if (!das_module_find_n(raster_name, sizeof(raster_name))) return 3;
    if (!das_module_find_n(stbimage_name, sizeof(stbimage_name))) return 4;

    das_shutdown();
    return 0;
}
