options gen2
require daslib/safe_addr

let shared num2name <- {
        1 => ["one", "один"],
        2 => ["two", "два"]
    }

[export]
def main() {
    let t = shared_addr(num2name, 1)
    print("t = {t}\n")
    for (k, v in keys(num2name), values(num2name)) {
        let pA = shared_addr(v)
        print("k = {k}, v = {v}, pA = {pA}\n")
    }
}

