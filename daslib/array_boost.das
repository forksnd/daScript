options indenting = 4
options no_unused_block_arguments = false
options no_unused_function_arguments = false
options no_aot = true

module array_boost shared public

require daslib/templates

[unsafe_operation, template (a), unused_argument(a)]
def private array_helper ( var arr : auto implicit =const; a : auto(TT) ) : array<TT -const-#>
    var res : array<TT -const-#>
    let lenA = length(arr)
    if lenA>=1
        unsafe
            _builtin_make_temp_array(res, addr(arr[0]), lenA)
    return <- res

[unsafe_operation, template (a), unused_argument(a)]
def private array_helper ( arr : auto implicit =const; a : auto(TT) ) : array<TT -const-#>
    var res : array<TT -const-#>
    let lenA = length(arr)
    if lenA>=1
        unsafe
            _builtin_make_temp_array(res, addr(arr[0]), lenA)
    return <- res

[unsafe_operation]
def public temp_array ( var arr : auto implicit =const )
    unsafe
        return <- array_helper ( arr, decltype(arr[0]) )

[unsafe_operation]
def public temp_array ( arr : auto implicit =const)
    unsafe
        return <- array_helper ( arr, decltype(arr[0]) )