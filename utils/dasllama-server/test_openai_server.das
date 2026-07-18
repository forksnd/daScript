options gen2
options persistent_heap   // the server thread loads/frees a multi-GB model once; explicit deletes below
options stack = 262144    // the forward chain runs under the HTTP tick + C++->das bridge (matches main.das)
options fast_math = true  // ggml-parity FP laxity — this is an inference tool test, not the bit-exact suite

require dastest/testing_boost public
require openai_server                                   // the server: OpenAiServer + set_model + run_server (sibling)
require dasllama/dasllama                               // load_model / QuantMode / embed
require daslib/jobque_boost                             // with_job_status (server-thread lifecycle)
require daslib/json_boost                               // read_json / from_JV
require daslib/fio                                      // get_env_variable / stat / path_join / sleep / base_name
require math                                            // sqrt

// End-to-end conformance for the OpenAI-compatible server: spin it up on its own thread (a separate
// context — so the model loads INSIDE that thread, since g_model is a per-context module global),
// then drive /v1/models, /v1/embeddings and /v1/chat/completions over the real dashv HTTP client.
// Model-gated + JIT-only, so it's a real regression locally and a clean skip in CI (no GGUF).
// Lives here (not modules/dasLLAMA/tests/) because a require path can't contain the '-' in "dasllama-server":
// a require_module_name is NAME tokens joined by / . % .. — so hyphenated-dir tools keep tests in-dir
// and require the sibling module by BARE name (same convention as detect-dupe / find-dupe / mcp).

let TEST_PORT = 18099

def private models_dir() : string {
    return get_env_variable("DASLLAMA_MODELS_DIR")   // empty -> MODEL_PATH is relative -> the test skips
}
let MODEL_PATH = path_join(models_dir(), "tinyllama-1.1b-chat-v1.0.Q8_0.gguf")

// Response shapes we assert on (from_JV ignores the object/model/usage fields we don't check).
struct private EmbItem {
    embedding : array<float>
    index : int
}
struct private EmbResp {
    data : array<EmbItem>
}

def private l2(v : array<float>) : float {
    var s = 0.0
    for (x in v) {
        s += x * x
    }
    return sqrt(s)
}

def private vecs_equal(a, b : array<float>) : bool {
    if (length(a) != length(b)) {
        return false
    }
    for (i in range(length(a))) {
        if (a[i] != b[i]) {
            return false
        }
    }
    return true
}

// Start the server on a dedicated thread (model loaded in-thread), wait until it answers /v1/models,
// run the block, then POST /shutdown (flips the server context's stop flag) and join. `mode`
// arms the Metal pipeline in the server thread's context BEFORE the load (the --metal flag's
// engine effect — the != off backend pin must precede load_model).
def private with_llama_server(model_path : string; port : int; mode : MetalMode = MetalMode.off; blk : block<(base_url : string) : void>) {
    with_job_status(1) $(finished) {
        new_thread() @() {
            set_metal_mode(mode)
            var m <- load_model(model_path, QuantMode.q8)
            set_model(m, "test-llm")
            set_default_max_tokens(4l)   // per-context global — exercised by the default-budget completion test
            run_server(port)   // blocks (with_job_que + tick loop) until POST /shutdown
            finished |> notify_and_release
        }
        let base_url = "http://127.0.0.1:{port}"
        var ready = false
        for (_attempt in range(300)) {   // up to ~30s for load + JIT
            GET("{base_url}/v1/models") $(resp) {
                if (resp != null && resp.status_code == http_status.OK) {
                    ready = true
                }
            }
            if (ready) {
                break
            }
            sleep(100u)
        }
        if (ready) {
            invoke(blk, base_url)
        }
        POST("{base_url}/shutdown", "") $(_resp) {}
        finished |> join
        assert(ready, "dasllama server became ready")
    }
}

[test]
def test_openai_server(t : T?) {
    t |> run("OpenAI server: models + embeddings + chat (needs tinyllama q8)") @(t : T?) {
        if (!jit_enabled()) {
            t |> success(true, "skipped: interpreted (dasLLAMA server test is JIT-only)")
            return
        }
        if (!stat(MODEL_PATH).is_valid) {
            t |> success(true, "skipped: {base_name(MODEL_PATH)} not present (set DASLLAMA_MODELS_DIR)")
            return
        }
        with_llama_server(MODEL_PATH, TEST_PORT) $(base_url) {
            // /v1/models
            GET("{base_url}/v1/models") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.OK))
                var err : string
                var js = read_json(string(resp.body), err)
                t |> success(js != null, "models JSON parses")
                if (js != null) {
                    t |> equal(string(js?["object"] ?? ""), "list")
                    unsafe {
                        delete js
                    }
                }
            }
            // /v1/embeddings — single input: unit-normalized vector of the model's hidden dim
            POST("{base_url}/v1/embeddings", "\{\"input\":\"the quick brown fox\"}") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.OK))
                var err : string
                var js = read_json(string(resp.body), err)
                t |> success(js != null, "embeddings JSON parses")
                if (js != null) {
                    t |> equal(string(js?["object"] ?? ""), "list")
                    let r = from_JV(js, type<EmbResp>)
                    t |> success(length(r.data) == 1, "one embedding returned")
                    if (length(r.data) == 1) {
                        t |> success(length(r.data[0].embedding) > 0, "embedding is non-empty")
                        t |> success(abs(l2(r.data[0].embedding) - 1.0) < 1.0e-4, "embedding is L2-normalized")
                    }
                    unsafe {
                        delete js
                    }
                }
            }
            // /v1/embeddings — batch: identical inputs give identical vectors (deterministic), distinct differ
            POST("{base_url}/v1/embeddings",
                 "\{\"input\":[\"hello world\",\"a different sentence\",\"hello world\"]}") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.OK))
                var err : string
                var js = read_json(string(resp.body), err)
                if (js != null) {
                    let r = from_JV(js, type<EmbResp>)
                    t |> success(length(r.data) == 3, "three embeddings returned")
                    if (length(r.data) == 3) {
                        t |> success(vecs_equal(r.data[0].embedding, r.data[2].embedding),
                                     "identical inputs -> identical vectors")
                        t |> success(!vecs_equal(r.data[0].embedding, r.data[1].embedding),
                                     "distinct inputs -> distinct vectors")
                    }
                    unsafe {
                        delete js
                    }
                }
            }
            // /v1/chat/completions (non-streaming) — end-to-end generation smoke
            POST("{base_url}/v1/chat/completions",
                 "\{\"messages\":[\{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8}") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.OK))
                var err : string
                var js = read_json(string(resp.body), err)
                t |> success(js != null, "chat JSON parses")
                if (js != null) {
                    t |> equal(string(js?["object"] ?? ""), "chat.completion")
                    unsafe {
                        delete js
                    }
                }
            }
            // /v1/completions WITHOUT max_tokens — the configured default budget applies (4 in this
            // harness; llm-cli-style clients omit the field, raw completions run to the budget)
            POST("{base_url}/v1/completions", "\{\"prompt\":\"Once\"}") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.OK))
                var err : string
                var js = read_json(string(resp.body), err)
                t |> success(js != null, "default-budget completion JSON parses")
                if (js != null) {
                    t |> equal(int(js?["usage"]?["completion_tokens"] ?? 0l), 4)
                    unsafe {
                        delete js
                    }
                }
            }
            // content-parts array form (what pi/opencode send even for pure text): the text
            // must reach the prompt — an empty-parse would render a near-empty user turn
            POST("{base_url}/v1/chat/completions",
                 "\{\"messages\":[\{\"role\":\"user\",\"content\":[\{\"type\":\"text\",\"text\":\"Please repeat the word strawberry twenty times, then count the letters in it, then explain why.\"}]}],\"max_tokens\":4}") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.OK))
                var err : string
                var js = read_json(string(resp.body), err)
                t |> success(js != null, "parts-form chat JSON parses")
                if (js != null) {
                    t |> success(int(js?["usage"]?["prompt_tokens"] ?? 0l) > 40, "parts text reached the prompt")
                    unsafe {
                        delete js
                    }
                }
            }
            // tools on a template with no tool format (TinyLlama resolves Zephyr) — honest 400
            POST("{base_url}/v1/chat/completions",
                 "\{\"messages\":[\{\"role\":\"user\",\"content\":\"hi\"}],\"tools\":[\{\"type\":\"function\",\"function\":\{\"name\":\"f\"}}]}") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.BAD_REQUEST))
            }
            // unknown endpoint hits the ANY("*") catch-all: logged server-side, OpenAI-shaped 404 back
            POST("{base_url}/v1/responses", "\{\"input\":\"hi\"}") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.NOT_FOUND))
                var err : string
                var js = read_json(string(resp.body), err)
                t |> success(js != null && js?["error"] != null, "404 body is an OpenAI error object")
                if (js != null) {
                    unsafe {
                        delete js
                    }
                }
            }
        }
    }
}

[test]
def test_asr_worker_panic_boundary(t : T?) {
    t |> run("ASR worker boundary returns normally after a panic") @(t : T?) {
        var failure = ""
        var failed = false
        with (module openai_server) {
            failed = catch_asr_panic(failure) {
                panic("forced ASR worker failure")
            }
        }
        t |> success(failed, "panic was caught")
        t |> equal(failure, "forced ASR worker failure")
    }
    t |> run("ASR worker boundary preserves successful execution") @(t : T?) {
        var invoked = false
        var failure = ""
        var failed = true
        with (module openai_server) {
            failed = catch_asr_panic(failure) {
                invoked = true
            }
        }
        t |> success(invoked, "work block ran")
        t |> success(!failed, "successful work was not reported as failed")
        t |> equal(failure, "")
    }
}

// ===== parse_tool_calls (model-free) =====

[test]
def test_openai_server_metal_mode(t : T?) {
    t |> run("OpenAI server: metal mode when_available serves (the --metal flag's engine effect)") @(t : T?) {
        if (!jit_enabled()) {
            t |> success(true, "skipped: interpreted (dasLLAMA server test is JIT-only)")
            return
        }
        if (!stat(MODEL_PATH).is_valid) {
            t |> success(true, "skipped: {base_name(MODEL_PATH)} not present (set DASLLAMA_MODELS_DIR)")
            return
        }
        // when_available: engages the GPU where the gates pass (Apple), quietly stays CPU
        // elsewhere — serving must work identically either way
        with_llama_server(MODEL_PATH, TEST_PORT + 2, MetalMode.when_available) $(base_url) {
            POST("{base_url}/v1/chat/completions",
                 "\{\"messages\":[\{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":8}") $(resp) {
                t |> equal(int(resp.status_code), int(http_status.OK))
                var err : string
                var js = read_json(string(resp.body), err)
                t |> success(js != null, "chat JSON parses under metal mode")
                if (js != null) {
                    t |> success(!empty(string(js?["choices"]?[0]?["message"]?["content"] ?? "")),
                        "chat completion returns content under metal mode")
                    unsafe {
                        delete js
                    }
                }
            }
        }
    }
}

[test]
def test_parse_tool_calls(t : T?) {
    t |> run("single call: content prefix, name, compact arguments, id/type") @(t : T?) {
        var calls : array<OaiToolCall>
        let pre = parse_tool_calls("Let me check.\n<tool_call>\n\{\"name\": \"get_weather\", \"arguments\": \{\"city\": \"Paris\"}}\n</tool_call>",
                                   "<tool_call>", "</tool_call>", "call_7", calls)
        t |> equal(pre, "Let me check.")
        t |> equal(length(calls), 1)
        if (length(calls) == 1) {
            t |> equal(calls[0]._function.name, "get_weather")
            t |> equal(calls[0]._function.arguments, "\{\"city\": \"Paris\"}")
            t |> equal(calls[0].id, "call_7_0")
            t |> equal(calls[0]._type, "function")
        }
        delete calls
    }
    t |> run("two calls parse in order; empty content strips clean") @(t : T?) {
        var calls : array<OaiToolCall>
        let pre = parse_tool_calls("<tool_call>\n\{\"name\": \"a\", \"arguments\": \{}}\n</tool_call>\n<tool_call>\n\{\"name\": \"b\", \"arguments\": \{\"n\": 1}}\n</tool_call>",
                                   "<tool_call>", "</tool_call>", "call_1", calls)
        t |> equal(pre, "")
        t |> equal(length(calls), 2)
        if (length(calls) == 2) {
            t |> equal(calls[0]._function.name, "a")
            t |> equal(calls[0]._function.arguments, "\{}")
            t |> equal(calls[1]._function.name, "b")
            t |> equal(calls[1].id, "call_1_1")
        }
        delete calls
    }
    t |> run("malformed spans keep the reply as content") @(t : T?) {
        var calls : array<OaiToolCall>
        // unterminated span (token budget cut): no calls, full text back
        let full1 = "text <tool_call>\n\{\"name\": \"a\""
        t |> equal(parse_tool_calls(full1, "<tool_call>", "</tool_call>", "c", calls), full1)
        t |> equal(length(calls), 0)
        // unparseable JSON inside a well-formed span: no calls, full text back
        let full2 = "<tool_call>\nnot json\n</tool_call>"
        t |> equal(parse_tool_calls(full2, "<tool_call>", "</tool_call>", "c", calls), full2)
        t |> equal(length(calls), 0)
        delete calls
    }
    t |> run("arguments emitted as a JSON string pass through verbatim") @(t : T?) {
        var calls : array<OaiToolCall>
        parse_tool_calls("<tool_call>\n\{\"name\": \"a\", \"arguments\": \"\{\\\"x\\\": 1}\"}\n</tool_call>",
                         "<tool_call>", "</tool_call>", "c", calls)
        t |> equal(length(calls), 1)
        if (length(calls) == 1) {
            t |> equal(calls[0]._function.arguments, "\{\"x\": 1}")
        }
        delete calls
    }
}

[test]
def test_parse_tool_calls_auto(t : T?) {
    t |> run("Mistral TOOL_CALLS array normalizes to OpenAI calls") @(t : T?) {
        var calls : array<OaiToolCall>
        let pre = parse_tool_calls_auto("checking\n[TOOL_CALLS][\{\"name\":\"search_messages\",\"arguments\":\{\"query\":\"zephyr\"}}]",
                                        "", "", "m", calls)
        t |> equal(pre, "checking")
        t |> equal(length(calls), 1)
        if (length(calls) == 1) {
            t |> equal(calls[0]._function.name, "search_messages")
            t |> equal(calls[0]._function.arguments, "\{\"query\": \"zephyr\"}")
        }
        delete calls
    }
    t |> run("Llama python tag maps parameters to arguments") @(t : T?) {
        var calls : array<OaiToolCall>
        let pre = parse_tool_calls_auto("<|python_tag|>\{\"name\":\"message_timeline\",\"parameters\":\{\"query\":\"x\"}}",
                                        "", "", "l", calls)
        t |> equal(pre, "")
        t |> equal(length(calls), 1)
        if (length(calls) == 1) {
            t |> equal(calls[0]._function.name, "message_timeline")
            t |> equal(calls[0]._function.arguments, "\{\"query\": \"x\"}")
        }
        delete calls
    }
    t |> run("strict bare OpenAI call object is accepted") @(t : T?) {
        var calls : array<OaiToolCall>
        let pre = parse_tool_calls_auto("\{\"function\":\{\"name\":\"get_message_context\",\"arguments\":\{\"message_id\":7}}}",
                                        "", "", "b", calls)
        t |> equal(pre, "")
        t |> equal(length(calls), 1)
        delete calls
    }
    t |> run("ordinary JSON remains assistant content") @(t : T?) {
        var calls : array<OaiToolCall>
        let full = "\{\"answer\":42}"
        t |> equal(parse_tool_calls_auto(full, "", "", "n", calls), full)
        t |> equal(length(calls), 0)
        delete calls
    }
}
