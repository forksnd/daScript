.. _classes:

=====
Class
=====

In Daslang, classes are an extension of structures designed to provide OOP capabilities.
Classes provides single parent inheritance, abstract and virtual methods, initializers, and finalizers.

The basic class declaration is similar to that of a structure, but with the ``class`` keyword::

    class Foo {
        x, y : int = 0
        def Foo {                           // custom initializer
            Foo`set(self,1,1)
        }
        def set(X,Y:int) {                  // inline method
            x = X
            y = Y
        }
    }

The initializer is a function with a name matching that of a class.
Classes can have multiple initializer with different arguments::

    class Foo {
        ...
        def Foo(T:int) {                    // custom initializer
            self->set(T,T)
        }
        def Foo(X,Y:int) {                  // custom initializer
            Foo`set(self,X,Y)
        }
    }

.. _classes_finalizer:

Finalizers can be defined explicitly as void functions named ``finalize``::

    class Foo {
        ...
        def finalize {                      // custom finalizer
            delFoo ++
        }
    }

Alternative syntax is::

    class Foo {
        ...
        def operator delete {              // custom finalizer
            delFoo ++
        }
    }

There are no guarantees that a finalizer is called implicitly (see :ref:`Finalizers <finalizers>`).

Derived classes need to override methods explicitly, using the ``override`` keyword::

    class Foo3D : Foo {
        z : int = 13
        def Foo3D {                         // overriding default initializer
            Foo`Foo(self)                   // call parents initializer explicitly
            z = 3
        }
        def override set(X,Y:int) {         // overriding method variable
            Foo`set(self,X,Y)               // calling generated method function directly
            z = 0
        }
    }

Classes can define abstract methods using the ``abstract`` keyword::

    class FooAbstract {
        def abstract set(X,Y:int) : void             // inline method
    }

Abstract functions need to be fully qualified, including their return type.
Class member functions are inferred in the same manner as regular functions.

Sealed functions cannot be overridden. The ``sealed`` keyword is used to prevent overriding::

    class Foo3D : Foo {
        def sealed set(X,Y:int ) { // subclasses of Foo3D can no longer override this method
            xyz = X + Y
        }
    }

Sealed classes can not be inherited from. The ``sealed`` keyword is used to prevent inheritance::

    class sealed Foo3D : Foo        // Foo3D can no longer be inherited from
        ...

A pointer named ``self`` is available inside any class method.

Classes can be created via the ``new`` operator::

    var f = new Foo()

Local class variables are unsafe::

    unsafe {
        var f = Foo()       // unsafe
    }

Class methods can be invoked using ``->`` syntax::

    f->set(1,2)

Alternatively, the . operator can be used (in most cases, short of some macros)::

    f.set(1,2)

A specific version of the method can also be called explicitly::

    Foo`set(*f,1,2)

Class methods can be constant::

    class Foo {
        dir : float3
        def const length {
            return length(dir)  // dir is const float3 here
        }
    }

Class methods can be operators::

    class Foo {
        dir : float3
        def Foo ( x,y,z:float ) {
            dir = float3(x,y,z)
        }
        def Foo ( d:float3 ) {
            dir = d
        }
        def const operator . length {
            return length(dir)
        }
        def operator . length := ( value:float ) {
            dir = normalize(dir) * value
        }
        def const operator + ( other:Foo ) {
            return Foo(dir + other.dir)
        }
    }

Class fields can be declared static, i.e. shared between all instances of the class::

    class Foo {
        static count : int = 0
        def Foo {
            count ++
        }
        def finalize {
            count --
        }
    }

Class methods can be declared static. Static methods don't have access to 'self' but can access static fields::

        class Foo {
            static count : int = 0
            def static getCount : int {
                return count
            }
        }

	    let count = Foo`getCount()  // they can be accessed outside of class

----------------------
Implementation details
----------------------

Class initializers are generated by adding a local ``self`` variable with `construct` syntax.
The body of the method is prefixed via a ``with self`` expression.
The final expression is a ``return <- self``::

    def Foo ( X:int const; Y:int const ) : Foo {
        var self:Foo <- Foo(uninitialized)
        with ( self ) {
            Foo`Foo(self,X,Y)
        }
        return <- self
    }

Class methods and finalizers are generated by providing the extra argument ``self``.
The body of the method is prefixed with a ``with self`` expression::

    def Foo3D`set ( var self:Foo3D; X:int const; Y:int const ) {
        with ( self ) {
            Foo`set(self,X,Y)
            z = 0
        }
    }

Calling virtual methods is implemented via invoke::

    invoke(f3d.set,cast<Foo> f3d,1,2)

Every base class gets an ``__rtti`` pointer, and a ``__finalize`` function pointer.
Additionally, a function pointer is added for each member function::

    class Foo {
        __rtti : void? = typeinfo(rtti_classinfo type<Foo>)
        __finalize : function<(self:Foo):void> = @@_::Foo'__finalize
        x : int = 0
        y : int = 0
        set : function<(self:Foo;X:int const;Y:int const):void> = @@_::Foo`set
    }

``__rtti`` contains rtti::TypeInfo for the specific class instance.
There is helper function in the rtti module to access class_info safely::

    def class_info ( cl ) : StructInfo const?

The ``finalize`` pointer is invoked when the finalizer is called for the class pointer.
That way, when delete is called on the base class pointer, the correct version of the derived finalizer is called.


