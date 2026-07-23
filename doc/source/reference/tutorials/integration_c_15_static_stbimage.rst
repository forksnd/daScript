.. _tutorial_integration_c_static_stbimage:

.. index::
   single: Tutorial; C Integration; Static Modules
   single: Tutorial; C Integration; StbImage
   single: Tutorial; C Integration; Split Initialization
   single: Tutorial; C Integration; Explicit-Length Strings
   single: register_Module_StbImage
   single: das_initialize_modules
   single: das_initialize_finalize

================================================
 C Integration: Static StbImage Module
================================================

This tutorial demonstrates how a C host statically links an additional native
module and registers it before daScript finalizes module dependencies. It uses
StbImage and Raster as a concrete end-to-end example: an embedded script creates
an RGBA image through ``stbimage_boost`` and resizes it through Raster.

The host also uses the explicit-length ``*_n`` C APIs throughout. These APIs map
directly to languages such as D, whose strings are pointer/length slices and do
not need to be null-terminated.


Why initialization is split
===========================

``das_initialize()`` remains the convenient one-call API for hosts which only
need the built-in modules. A host with statically linked application modules
needs an insertion point between registering the built-ins and resolving module
dependencies:

.. code-block:: c

   das_initialize_modules();

   register_Module_Raster();
   register_Module_StbImage();

   das_initialize_finalize();

The three phases are:

1. ``das_initialize_modules()`` registers daScript's built-in modules.
2. The host calls every statically linked ``register_Module_*`` entry point.
3. ``das_initialize_finalize()`` initializes the complete dependency graph.

Do not call ``das_initialize()`` as well: it performs phases 1 and 3 itself.


C module entry points
=====================

Module registration functions have C linkage and therefore expose undecorated
symbol names. A C host declares only the opaque C API return type:

.. code-block:: c

   extern das_module * register_Module_Raster(void);
   extern das_module * register_Module_StbImage(void);

The equivalent declarations in D use ``extern(C)``:

.. code-block:: d

   extern(C) das_module* register_Module_Raster();
   extern(C) das_module* register_Module_StbImage();

The application must link ``libDasModuleStbImage``. That library contains the
Raster, StbImage, and StbTrueType registration functions.


Native module versus its daslang wrapper
========================================

Calling ``register_Module_StbImage`` makes the native functions available to
the compiler. The higher-level ``stbimage/stbimage_boost.das`` wrapper is still
a source module and must be visible through the compilation's
``das_file_access``:

.. code-block:: c

   static const char BOOST_MODULE[] = "stbimage/stbimage_boost";

   das_file_access * files = das_fileaccess_make_default();
   if (!das_fileaccess_introduce_native_module_n(
           files, BOOST_MODULE, sizeof(BOOST_MODULE) - 1)) {
       /* The SDK's StbImage source module was not found. */
   }

This distinction is important:

* ``register_Module_StbImage`` registers the statically linked C++ module.
* ``das_fileaccess_introduce_native_module_n`` introduces its installed
  daslang wrapper into this file access.


The embedded daslang script
===========================

The script creates a 2x2 four-channel image, resizes it to 1x1, and returns a
value encoding the resulting dimensions and channel count:

.. code-block:: das

   options gen2
   require stbimage/stbimage_boost

   [export]
   def main : int {
       var image = make_image(2, 2, 4)
       var resized <- image.resize(1, 1)
       return resized.width * 100 + resized.height * 10 + resized.channels
   }

``make_image`` exercises the StbImage wrapper. ``Image.resize`` reaches the
native Raster resize implementation. The expected return value is ``114``.


Explicit-length source and names
================================

The host passes every string as a pointer and a ``size_t`` length. No
``strlen`` call and no narrowing to ``uint32_t`` is required:

.. code-block:: c

   static const char SCRIPT_NAME[] = "stbimage_from_c.das";
   static const char SCRIPT_SOURCE[] =
       "options gen2\n"
       "require stbimage/stbimage_boost\n"
       /* ... */;

   das_fileaccess_introduce_file_n(
       files,
       SCRIPT_NAME, sizeof(SCRIPT_NAME) - 1,
       SCRIPT_SOURCE, sizeof(SCRIPT_SOURCE) - 1,
       0);

   das_program * program = das_program_compile_n(
       SCRIPT_NAME, sizeof(SCRIPT_NAME) - 1,
       files, output, modules);

The same pattern is used for ``das_context_find_function_n``. Existing
null-terminated APIs remain available for compatibility.


Calling the result
==================

After compilation and simulation, the host locates and invokes the exported
``main`` function through the regular C evaluation API:

.. code-block:: c

   static const char MAIN_FUNCTION[] = "main";

   das_function * main_function = das_context_find_function_n(
       context, MAIN_FUNCTION, sizeof(MAIN_FUNCTION) - 1);
   vec4f result = das_context_eval_with_catch(context, main_function, NULL);

   if (!das_context_get_exception(context)) {
       int value = das_argument_int(result);  /* 114 */
   }

Release the context, program, file access, module group, and output writer
before calling ``das_shutdown()``.


Static CMake linkage
====================

The in-tree tutorial links the static module before the static compiler
library. CMake carries the module's transitive runtime dependencies:

.. code-block:: cmake

   target_link_libraries(integration_c_15 PRIVATE
       libDasModuleStbImage libDaScript Threads::Threads)

An installed SDK exposes the corresponding imported targets as
``DAS::libDasModuleStbImage`` and ``DAS::libDaScript``. The standalone tutorial
build enables this target when the SDK was installed with StbImage support.


Build & run
===========

Build::

   cmake --build build --config Release --target integration_c_15

Run from the project root::

   bin/Release/integration_c_15

Expected output::

   static StbImage integration succeeded: 114


.. seealso::

   Full source:
   :download:`15_static_stbimage.c <../../../../tutorials/integration/c/15_static_stbimage.c>`

   Previous tutorial: :ref:`tutorial_integration_c_passing_arrays`

   C API reference: :ref:`embedding_c_api`

   Native and shared module details: :ref:`embedding_external_modules`

   daScriptC.h API header: ``include/daScript/daScriptC.h``
