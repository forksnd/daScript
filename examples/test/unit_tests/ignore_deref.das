options gen2

class A {
    def test() {
        print("test\n")
    }
}

[export]
def test {
    var arr <- [ new A(), new A() ]
    var a = arr[0] // A?&
    a->test()
    var b : A? = arr[0] // explicit A?
    b->test()
    return true
}
