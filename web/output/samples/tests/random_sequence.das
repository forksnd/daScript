options gen2
require math

// enable for test purposes only. this is slower
// options intern_strings = true

def makeRandomSequence(var src : array<string>) {
    let n = 500000
    let mod = uint(n)
    resize(src, n)
    for (i in range(n)) {
        let num = (271828183u ^ uint(i * 119)) % mod
        src[i] = "{num}"
    }
}

def dict(var tab : table<string; int>; src : array<string>) {
    clear(tab)
    var maxOcc = 0
    for (s in src) {
        maxOcc = max(++tab[s], maxOcc)
    }
    return maxOcc
}

[export]
def main {
    var tab : table<string; int>
    var src : array<string>
    makeRandomSequence(src)
    profile(20, "dictionary") <| $() {
        dict(tab, src)
    }
    return true
}