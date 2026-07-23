.. _tutorial_integration_cpp_namespace_integration:

.. index::
   single: Tutorial; C++ Integration; Namespace Integration

==========================================
 C++ Integration: Namespace Integration
==========================================

This tutorial shows how to initialize daslang modules when your code lives
inside a C++ namespace.

Topics covered:

* Why module registration entry points use C linkage
* Using ``NEED_MODULE`` safely inside a namespace
* ``DECLARE_MODULE`` / ``PULL_MODULE`` for separately compiled modules


Prerequisites
=============

* Tutorial 1 completed (:ref:`tutorial_integration_cpp_hello_world`) —
  basic compile → simulate → eval cycle.


Global C-linkage entry points
=============================

``REGISTER_MODULE`` exposes an unmangled global C entry point.  For example,
``REGISTER_MODULE(Module_BuiltIn)`` provides:

.. code-block:: cpp

   extern "C" das::Module * register_Module_BuiltIn();

The built-in declarations are provided by ``daScriptModule.h``.
``NEED_MODULE`` calls the global entry point explicitly, so it is safe inside
a namespace:

.. code-block:: cpp

   namespace MyApp {
       void init() {
           NEED_ALL_DEFAULT_MODULES;
           das::Module::Initialize();
       }
   }

``DECLARE_MODULE`` / ``PULL_MODULE``
====================================

Separately compiled custom modules are not known to ``daScriptModule.h``.
``DECLARE_MODULE(ClassName)`` forward-declares their global C-linkage
``register_*`` function.  It must be placed at file scope, outside any
namespace.

``PULL_MODULE(ClassName)`` — performs the registration call using the
``::`` prefix to explicitly reference the global-scope function.  Safe
inside any namespace, class, or function body.  Once a declaration is
visible, ``NEED_MODULE`` and ``PULL_MODULE`` have the same pull behavior.

Convenience wrappers ``DECLARE_ALL_DEFAULT_MODULES`` and
``PULL_ALL_DEFAULT_MODULES`` remain available, although the built-in
declarations are already supplied by the header.


The tutorial
============

The tutorial runs the same ``01_hello_world.das`` script as Tutorial 1,
but all daslang calls happen inside ``namespace MyApp``.

.. literalinclude:: ../../../../tutorials/integration/cpp/22_namespace_integration.cpp
   :language: cpp
   :caption: tutorials/integration/cpp/22_namespace_integration.cpp


How it works
============

1. ``daScriptModule.h`` declares the default module entry points globally
   with C linkage.

2. Inside ``MyApp::initialize()``, ``NEED_ALL_DEFAULT_MODULES`` calls each
   ``::register_Module_*()`` function explicitly.

3. ``Module::Initialize()`` and the rest of the daslang API work normally
   inside the namespace.


Custom modules
==============

For a separately compiled custom module, declare it at file scope before
pulling it from a function or namespace:

.. code-block:: cpp

   DECLARE_MODULE(Module_MyMod);

   namespace MyApp {
       void init() {
           NEED_ALL_DEFAULT_MODULES;
           PULL_MODULE(Module_MyMod);
           das::Module::Initialize();
       }
   }


External (plugin) modules
=========================

CMake generates three ``.inc`` files for every module registered with
``ADD_MODULE_CPP``:

``external_need.inc``
   Contains ``NEED_MODULE(...)`` calls.  The corresponding declarations
   must already be visible.

``external_declare.inc``
   Contains ``DECLARE_MODULE(...)`` — include at file scope.

``external_pull.inc``
   Contains ``PULL_MODULE(...)`` — include inside any namespace or
   function body.

The explicit declaration/pull pair is recommended:

.. code-block:: cpp

   // file scope
   #include "modules/external_declare.inc"

   namespace MyApp {
       void init() {
           // inside namespace
           #include "modules/external_pull.inc"
       }
   }


Build and run
=============

.. code-block:: bash

   cmake --build build --config Release --target integration_cpp_22

.. code-block::

   $ bin/Release/integration_cpp_22
   Hello, World!
