# Tree-sitter syntax highlighting

This document demonstrates language selection through ordinary Markdown fence
labels. The viewer is in **Auto** mode: each code block chooses its own grammar.

## daScript

```das
options gen2

require strings

struct Greeting {
    recipient : string
    punctuation : string = "!"
}

def render(greeting : Greeting const) : string {
    return "Hello, {greeting.recipient}{greeting.punctuation}"
}

[export]
def main() {
    let greeting = Greeting(recipient = "Tree-sitter")
    print("{render(greeting)}\n")
}
```

## C++

```cpp
#include <iostream>
#include <string>
#include <string_view>

struct Greeting {
    std::string recipient;
    char punctuation = '!';
};

std::string render(const Greeting& greeting) {
    return "Hello, " + greeting.recipient + greeting.punctuation;
}

int main() {
    const Greeting greeting{std::string{"Tree-sitter"}};
    std::cout << render(greeting) << '\n';
    return 0;
}
```

## Neutral code

An unlabelled fence intentionally stays parser-neutral.

```
Auto mode leaves this block unhighlighted.
```
