# Thinking-model support across architectures

Working brief for the per-model thinking/parameters session. Today thinking works properly
only on the `<think>` family (Qwen, GLM); everything else either leaks chain-of-thought as
content or never gets its markers recognized. Live evidence: gemma-4-E4B on the control-page
chat emitted bare chain-of-thought and leaked a `<channel|>` closer as text (its opener never
recognized).

## The two mechanisms today

1. **Prompt side — `think_suppress`** (`dasllama_chat.das::render_think_suppress`). Turns a
   hybrid thinking model into an instruct model by prefilling an *empty, closed* thought block
   on the generation prompt (Qwen3's `enable_thinking=false`). Declared per-arch as
   `d.chat.think_suppress.parts`; renders only when `!enable_thinking` AND the vocab has the
   specials (else silently skips). Replayed history carries no think blocks. This part is
   already per-arch data and extensible.
   - Server wiring: request `enable_thinking` (and `chat_template_kwargs.enable_thinking`) →
     `set_thinking(chat, false)` → the suppress block. Default is thinking-ON.

2. **Reply side — `strip_think`** (`dasllama_chat.das::strip_think`). Removes the thought
   section from the *stored* assistant turn so history holds only the final answer. **This is
   the piece that only handles Qwen**: hardcoded to `<think>` / `</think>`. Every other thinking
   family uses different delimiters (below), so their thoughts leak into history or never strip.

**The gap in one line:** `think_suppress` is per-arch data (extensible) but `strip_think` is a
single hardcoded literal pair. The core of the arc is making the reply-side delimiters per-arch
too, and exposing the reasoning span to the server as `reasoning_content` (OpenAI-compatible)
instead of dropping it.

## Per-family wire formats

| family | thinking default | opener | closer | delimiters | status |
|---|---|---|---|---|---|
| **Qwen 2/2.5/3/3.5, qwen*moe** | on (hybrid) | `<think>` | `</think>` | symmetric | **fully correct today** — both mechanisms match |
| **GLM-4 MoE** | on | `<think>` | `</think>` | symmetric | `think_suppress` declared; shares Qwen's literal — should strip, VERIFY |
| **gpt-oss (Harmony-lite)** | on | `<\|channel\|>analysis` | `<\|channel\|>final` (switch) | channel-switch | `strip_think` does NOT handle it |
| **gemma-4 E-series** | **off** (gate `<\|think\|>` in system turn) | `<\|channel>thought` | `<channel\|>` | ASYMMETRIC (`<\|X>` / `<X\|>`) | arch prefills empty thought = its instruct default; reply-strip missing |
| llama, phi3, gemma2/3, mistral3 | none | — | — | — | non-thinking; no work |

### gemma-4 E-series specifics (2026-07-19 research pass)

- Thinking is **off by default**, enabled by injecting `<|think|>` at the start of the SYSTEM
  turn. Our arch's `assistant_open` already prefills `<|channel>thought\n<channel|>` (empty
  closed block) — that IS the non-thinking default (instruct mode). Thinking-ON means NOT
  prefilling the closer and letting the model emit thought text.
- Generated (thinking on): after `<|turn>model\n` the MODEL emits
  `<|channel>thought\n … reasoning … <channel|>` then the answer then `<turn|>`. Opener is
  model-emitted in the normal case; three exceptions prefill it (tool-response continuation,
  non-E sizes with thinking off, llama.cpp's bundled template) — so the reply parser must accept
  the opener from either side.
- Asymmetric brackets are the DESIGN, uniform across the family: `<|turn>`/`<turn|>`,
  `<|channel>`/`<channel|>`, `<|tool_call>`/`<tool_call|>`. `thought` is the only reasoning
  channel; tool calls have their own tags.

## Target reply-side rules

Generalize `strip_think` into a per-arch matcher carrying `(opener, closer, mode)`:
- **symmetric** (`<think>`/`</think>`): current logic.
- **channel-switch** (gpt-oss): reasoning = `analysis` open → `final` switch; content = after.
- **asymmetric** (gemma-4): reasoning = `<|channel>thought` (trailing `\n` optional) →
  `<channel|>`; content = after `<channel|>` (trim leading `\n`) until stop.
- **truncated** (no closer before EOG / limit): classify the whole tail as reasoning.
- **history**: strip prior-turn thoughts EXCEPT inside an in-flight tool-call chain (thoughts
  between calls survive). gemma-4's canonical template does exactly this (`strip_thinking`
  macro + a per-turn gate keyed on the last-user index).
- **server**: expose the reasoning span as `reasoning_content` (streaming: buffer partial tags
  across SSE chunks), not just drop it. llama.cpp's merged gemma-4 PEG parser
  (`common_peg_gemma4_builder`, PR #21326) is the reference implementation.

## Where the work lands

- `dasllama_chat.das` — generalize `strip_think` (per-arch delimiters + mode); keep `<think>`
  as the symmetric case; the reasoning span becomes a return value, not a drop.
- `dasllama_arch_*.das` — each thinking arch declares its reply delimiters beside its existing
  `think_suppress` (gpt-oss channel-switch, gemma-4 asymmetric; Qwen/GLM already symmetric).
- `utils/dasllama-server/openai_server.das` — carry `reasoning_content` through the
  chat-completion + streaming surface; `enable_thinking` plumbing already exists.
- Tests — a render + strip round-trip per family (`render_turn_` already covers the prompt side).

## Sources (gemma-4 pass, 2026-07-19)

- Canonical template: https://huggingface.co/google/gemma-4-E4B-it/raw/main/chat_template.jinja
- Google thinking docs: https://ai.google.dev/gemma/docs/capabilities/thinking
- llama.cpp gemma-4 parser (merged): https://github.com/ggml-org/llama.cpp/pull/21326
- disable-thinking discussion: https://github.com/ggml-org/llama.cpp/discussions/21338
