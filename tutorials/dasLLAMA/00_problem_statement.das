options gen2
options stack = 65536

require daslib/fio
require daslib/array_boost
require math
require strings

// Tutorial dasLLAMA-00: Problem Statement
//
// This file does not use dasLLAMA. It is a complete, if deliberately tiny,
// Llama-2 inference engine: map a real checkpoint, run the transformer, greedily
// choose the next token, and print a story.
//
// It also tells another story. The first dasLLAMA commit was authored on
// 2026-06-27. It began with the naive matmul below; about a day later PR #3297
// contained Llama-2/3, GGUF, quantization, threading, prefill, tests, and chat.
// That was possible because all the steps from readable script to real machine
// code could stay in one programming environment.
//
// This is the seed, before it became a library.
//
// Get the two llama2.c files:
//   https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
//   https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
//
// Run:
//   daslang.exe -jit tutorials/dasLLAMA/00_problem_statement.das -- stories15M.bin tokenizer.bin
//
// JIT is much faster, but even the interpreter tells this story at about five
// tokens per second. Try both: the same source runs unchanged in either tier.

// -----------------------------------------------------------------------------
// The five operations used by a Llama transformer
// -----------------------------------------------------------------------------

// y = W*x. W has d rows and n columns. Most inference time is spent here.
def matmul(var y : array<float>; w : array<float>#; x : array<float>; n, d : int) {
    for (row in range(d)) {
        var sum = 0.0
        let offset = row * n
        for (col in range(n)) {
            sum += w[offset + col] * x[col]
        }
        y[row] = sum
    }
}

// RMSNorm keeps the residual stream at a useful scale.
def rmsnorm(var out : array<float>; x : array<float>; weight : array<float>#; size : int) {
    var sum = 0.0
    for (i in range(size)) {
        let xi = x[i]
        sum += xi * xi
    }
    let scale = 1.0 / sqrt(sum / float(size) + 1e-5)
    for (i in range(size)) {
        out[i] = weight[i] * x[i] * scale
    }
}

// Softmax turns attention scores into weights which sum to one.
def softmax(var x : array<float>; size : int) {
    var largest = x[0]
    for (i in range(1, size)) {
        largest = max(largest, x[i])
    }
    var sum = 0.0
    for (i in range(size)) {
        x[i] = exp(x[i] - largest)
        sum += x[i]
    }
    for (i in range(size)) {
        x[i] /= sum
    }
}

// SiLU is the non-linearity in the feed-forward gate.
def silu(var x : array<float>; size : int) {
    for (i in range(size)) {
        x[i] /= 1.0 + exp(-x[i])
    }
}

// RoPE encodes position by rotating pairs of query/key coordinates.
def rope(var x : array<float>; position, head_size, size : int) {
    for (pair in range(size / 2)) {
        let i = pair * 2
        let frequency = 1.0 / pow(10000.0, float(i % head_size) / float(head_size))
        let angle = float(position) * frequency
        let a = x[i]
        let b = x[i + 1]
        let sin_angle = sin(angle)
        let cos_angle = cos(angle)
        x[i] = a * cos_angle - b * sin_angle
        x[i + 1] = a * sin_angle + b * cos_angle
    }
}

// -----------------------------------------------------------------------------
// What is in a llama2.c checkpoint
// -----------------------------------------------------------------------------

struct Config {
    dim : int
    hidden_dim : int
    layers : int
    heads : int
    kv_heads : int
    vocab : int
    sequence : int
    head_size : int
    kv_dim : int
    shared_classifier : bool
}

// The checkpoint is one flat fp32 array. These are element offsets, not copies.
struct Weights {
    token_embedding : int
    attention_norm : int
    query : int
    key : int
    value : int
    attention_output : int
    ffn_norm : int
    ffn_gate : int
    ffn_down : int
    ffn_up : int
    final_norm : int
    classifier : int
    end : int
}

def take(var cursor : int&; count : int) : int {
    let result = cursor
    cursor += count
    return result
}

def weight_layout(c : Config) : Weights {
    var at = 0
    var w = Weights(
        token_embedding = take(at, c.vocab * c.dim),
        attention_norm = take(at, c.layers * c.dim),
        query = take(at, c.layers * c.dim * c.dim),
        key = take(at, c.layers * c.dim * c.kv_dim),
        value = take(at, c.layers * c.dim * c.kv_dim),
        attention_output = take(at, c.layers * c.dim * c.dim),
        ffn_norm = take(at, c.layers * c.dim),
        ffn_gate = take(at, c.layers * c.dim * c.hidden_dim),
        ffn_down = take(at, c.layers * c.hidden_dim * c.dim),
        ffn_up = take(at, c.layers * c.dim * c.hidden_dim),
        final_norm = take(at, c.dim)
    )
    at += c.sequence * c.head_size   // old precomputed RoPE tables; this program computes them
    w.classifier = c.shared_classifier ? w.token_embedding : take(at, c.vocab * c.dim)
    w.end = at
    return w
}

// Scratch memory is allocated once and reused for every generated token.
struct State {
    x : array<float>
    xb : array<float>
    xb2 : array<float>
    hidden : array<float>
    gate : array<float>
    query : array<float>
    key : array<float>
    value : array<float>
    attention : array<float>
    logits : array<float>
    key_cache : array<float>
    value_cache : array<float>
}

def zeros(size : int) : array<float> {
    var result : array<float>
    result |> resize(size)
    return <- result
}

def make_state(c : Config) : State {
    return <- State(
        x <- zeros(c.dim),
        xb <- zeros(c.dim),
        xb2 <- zeros(c.dim),
        hidden <- zeros(c.hidden_dim),
        gate <- zeros(c.hidden_dim),
        query <- zeros(c.dim),
        key <- zeros(c.kv_dim),
        value <- zeros(c.kv_dim),
        attention <- zeros(c.sequence),
        logits <- zeros(c.vocab),
        key_cache <- zeros(c.layers * c.sequence * c.kv_dim),
        value_cache <- zeros(c.layers * c.sequence * c.kv_dim)
    )
}

// Present a matrix or norm vector inside the flat checkpoint as a borrowed view.
def project(
            var out : array<float>;
            model : array<float>#;
            offset : int;
            input : array<float>;
            n, d : int
            ) {
    array_view(model, offset, n * d) $(matrix : array<float>#) {
        matmul(out, matrix, input, n, d)
    }
}

def normalize(
              var out : array<float>;
              model : array<float>#;
              offset : int;
              input : array<float>;
              size : int
              ) {
    array_view(model, offset, size) $(weight : array<float>#) {
        rmsnorm(out, input, weight, size)
    }
}

// -----------------------------------------------------------------------------
// One token through the entire transformer
// -----------------------------------------------------------------------------

def forward(
            c : Config;
            w : Weights;
            model : array<float>#;
            var s : State;
            token, position : int
            ) {
    let dim = c.dim
    let kv_multiple = c.heads / c.kv_heads
    let attention_scale = 1.0 / sqrt(float(c.head_size))

    array_view(model, w.token_embedding + token * dim, dim) $(embedding : array<float>#) {
        for (i in range(dim)) {
            s.x[i] = embedding[i]
        }
    }

    for (layer in range(c.layers)) {
        let cache_layer = layer * c.sequence * c.kv_dim

        normalize(s.xb, model, w.attention_norm + layer * dim, s.x, dim)
        project(s.query, model, w.query + layer * dim * dim, s.xb, dim, dim)
        project(s.key, model, w.key + layer * dim * c.kv_dim, s.xb, dim, c.kv_dim)
        project(s.value, model, w.value + layer * dim * c.kv_dim, s.xb, dim, c.kv_dim)

        rope(s.query, position, c.head_size, dim)
        rope(s.key, position, c.head_size, c.kv_dim)

        let cache_row = cache_layer + position * c.kv_dim
        for (i in range(c.kv_dim)) {
            s.key_cache[cache_row + i] = s.key[i]
            s.value_cache[cache_row + i] = s.value[i]
        }

        for (head in range(c.heads)) {
            let query_row = head * c.head_size
            let cache_head = (head / kv_multiple) * c.head_size

            for (past in range(position + 1)) {
                let key_row = cache_layer + past * c.kv_dim + cache_head
                var score = 0.0
                for (i in range(c.head_size)) {
                    score += s.query[query_row + i] * s.key_cache[key_row + i]
                }
                s.attention[past] = score * attention_scale
            }
            softmax(s.attention, position + 1)

            for (i in range(c.head_size)) {
                s.xb[query_row + i] = 0.0
            }
            for (past in range(position + 1)) {
                let value_row = cache_layer + past * c.kv_dim + cache_head
                let score = s.attention[past]
                for (i in range(c.head_size)) {
                    s.xb[query_row + i] += score * s.value_cache[value_row + i]
                }
            }
        }

        project(s.xb2, model, w.attention_output + layer * dim * dim, s.xb, dim, dim)
        for (i in range(dim)) {
            s.x[i] += s.xb2[i]
        }

        normalize(s.xb, model, w.ffn_norm + layer * dim, s.x, dim)
        project(s.hidden, model, w.ffn_gate + layer * dim * c.hidden_dim, s.xb, dim, c.hidden_dim)
        project(s.gate, model, w.ffn_up + layer * dim * c.hidden_dim, s.xb, dim, c.hidden_dim)
        silu(s.hidden, c.hidden_dim)
        for (i in range(c.hidden_dim)) {
            s.hidden[i] *= s.gate[i]
        }
        project(s.xb, model, w.ffn_down + layer * c.hidden_dim * dim, s.hidden, c.hidden_dim, dim)
        for (i in range(dim)) {
            s.x[i] += s.xb[i]
        }
    }

    normalize(s.xb, model, w.final_norm, s.x, dim)
    project(s.logits, model, w.classifier, s.xb, dim, c.vocab)
}

def most_likely(logits : array<float>) : int {
    var best = 0
    for (i in range(1, length(logits))) {
        if (logits[i] > logits[best]) {
            best = i
        }
    }
    return best
}

// -----------------------------------------------------------------------------
// Just enough tokenizer to turn generated token ids back into text
// -----------------------------------------------------------------------------

def read_i32_le(data : array<uint8>#; offset : int) : int {
    if (offset < 0 || offset + 4 > length(data)) {
        panic("tokenizer ends inside an int32")
    }
    return (
        int(data[offset])
        | (int(data[offset + 1]) << 8)
        | (int(data[offset + 2]) << 16)
        | (int(data[offset + 3]) << 24)
    )
}

def load_pieces(path : string; count : int) : array<string> {
    var pieces : array<string>
    let file = fopen(path, "rb")
    if (file == null) {
        panic("cannot open tokenizer '{path}'")
    }
    fmap(file) $(var data : array<uint8>#) {
        var cursor = 4   // max_token_length, unused by decoding
        for (_ in range(count)) {
            cursor += 4 // merge score, used by encoding but not decoding
            let size = read_i32_le(data, cursor)
            cursor += 4
            array_view(data, cursor, size) $(piece : array<uint8>#) {
                pieces |> push(string(piece))
            }
            cursor += size
        }
    }
    fclose(file)
    return <- pieces
}

def token_text(pieces : array<string>; previous, token : int) : string {
    let text = pieces[token]
    if (length(text) == 6 && starts_with(text, "<0x") && ends_with(text, ">")) {
        return string([uint8(to_int(slice(text, 1, 5), true))])
    }
    if (previous == 1 && !empty(text) && first_character(text) == ' ') {
        return slice(text, 1)
    }
    return text
}

// -----------------------------------------------------------------------------
// Map the checkpoint and let token 1 (BOS and sequence delimiter) become a story
// -----------------------------------------------------------------------------

def tell_story(model_path, tokenizer_path : string) {
    let file = fopen(model_path, "rb")
    if (file == null) {
        panic("cannot open model '{model_path}'")
    }

    fmap(file) $(var bytes : array<uint8>#) {
        if (length(bytes) < 28) {
            panic("checkpoint is smaller than its seven-int header")
        }
        array_view(bytes, 0, 7, type<int>) $(header : array<int>#) {
            let signed_vocab = header[5]
            let c = Config(
                dim = header[0],
                hidden_dim = header[1],
                layers = header[2],
                heads = header[3],
                kv_heads = header[4],
                vocab = abs(signed_vocab),
                sequence = header[6],
                head_size = header[0] / header[3],
                kv_dim = header[0] * header[4] / header[3],
                shared_classifier = signed_vocab > 0
            )
            let w = weight_layout(c)
            if (length(bytes) < 28 + w.end * 4) {
                panic("checkpoint ends before its declared weights")
            }
            array_view(bytes, 28, (length(bytes) - 28) / 4, type<float>) $(model : array<float>#) {
                let pieces <- load_pieces(tokenizer_path, c.vocab)
                var state <- make_state(c)

                print("dim={c.dim}, layers={c.layers}, heads={c.heads}, parameters≈{w.end}\n\n")
                let delimiter = 1
                var token = delimiter
                for (position in range(c.sequence)) {
                    forward(c, w, model, state, token, position)
                    let next = most_likely(state.logits)
                    if (next == delimiter) {
                        break
                    }
                    fprint(fstdout(), token_text(pieces, token, next))
                    fflush(fstdout())
                    token = next
                }
                print("\n")
            }
        }
    }
    fclose(file)
}

[export]
def main() {
    let args <- get_command_line_arguments()
    let paths <- [for (arg in args); arg; where ends_with(arg, ".bin")]
    if (length(paths) < 2) {
        print("usage: daslang -jit tutorials/dasLLAMA/00_problem_statement.das -- stories15M.bin tokenizer.bin\n")
        return
    }
    tell_story(paths[length(paths) - 2], paths[length(paths) - 1])
}
