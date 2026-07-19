# dasllama-server control page — build plan

**Framing (Boris):** "this is as much of a server as it is a technology demo." Polish target:
HackerNews + r/LocalLLaMA. Styling comes LAST via the Claude Design "daslang.io Forge" project
(id `4a6f4f1c-61cc-414d-b50e-98dae7f218b3`) — every slice below builds *contents*; the design
pass restyles once contents are final.

**Already done (2026-07-18, branch `bbatkin/dasllama-control-page`):** `GET /` serves
`control.html` beside the sources; `/v1/stats` + `model`/`ctx`/`uptime_s`/`draining`; tiles +
tok/s & streams charts + gc/drain buttons. Verified against a python mock (playwright, incl.
confirm-drain) and live `test_openai_server` 17/17 with tinyllama.

**S1 + S9 DONE (2026-07-19):** `write_toml` in daslib (4 tests); authoritative-TOML precedence
+ per-key source tracking in `apply_config`; `GET/POST /config` (validated save, model picker,
active rail); `POST /restart` + exit code 4 + watchdog relaunch; `gpu` selector + vulkan detail
keys with programmatic `set_gpu_tier_want`/`moe_gpu_tier_arm` engine setters (env stays
override); capability guard (`supported` + `reason` recorded at load); `gpu_tier_status` getter
+ page GPU badge; typed config form with provenance chips + save→confirm→restart flow.
Verified: toml 108/108, server 17/17 live, vulkan tier 12/12, e2e restart loop (CLI-beats-toml
→ save authoritative → exit 4 → relaunch → toml-beats-CLI), page behavior via mock+playwright.

**S5 DONE (2026-07-19):** page chat over the existing `/v1/chat/completions` SSE (fetch reader,
frame-split, keep-alive tolerant) — conversation client-held, assistant history stripped of the
think trace; `<think>` spans render inline in a serif think font; all sampling knobs
(temperature / top_k / top_p / min_p / presence / frequency / max_tokens / enable_thinking);
mic button (shown when `asr_workers > 0`) records via ScriptProcessor → 16-bit WAV → POST
`/v1/audio/transcriptions` → transcript into the input. Verified: mock (knob/history/think
assertions on the recorded request) + LIVE tinyllama end-to-end (real SSE chat bubble, real
hardware line, history card, cache hint) via playwright.

**S4 DONE (2026-07-19):** prefix cache grows a chain registry (`PrefixChain` per donation with
≥1 new page: page-covered tokens, live pages, hits — bumped when an attach's deepest matched
page belongs to it — born/last-hit ticks, preview); scheduler passes `st.prompt_head` as the
preview at donation; `prefix_chain_list` facade; `/v1/streams` gains `cache`; page: prefix-cache
table (preview / tok / pages / hits / age / idle). Verified: kv_prefix 7/7+1skip, scheduler
12/12, server 17/17 (cache round trip: >1-page prompt donates → re-send hits → chain surfaced
with preview), page via mock+playwright.

**S3 DONE (2026-07-19):** Stream carries `prompt_head` (~160 chars, whole pieces) + `gen_tail`
(240..480 chars, UTF-8-safe front trim); finished streams linger ~10 s in
`Scheduler.recent_done` (bounded 32); `GET /v1/streams` serves state / counters / TTFT / tails
for active + queued + finished. Page: slot swimlane replaces the streams chart (prefill/decode
shades per lane, queued depth as the baseline strip), live stream cards with client-side
transcript accumulation (overlap-merge of capped tails — verified exact against a scripted
mock), History view (last 20). Verified: scheduler 12/12, server 17/17 live (finished-linger +
tail assertions), page via mock+playwright.

**S2 DONE (2026-07-19):** scheduler `prefill_tokens` (computed tokens only, cache hits
excluded) + TTFT aggregate (sum/count/last at reap; per-stream `ttft_us` predated); /v1/stats
grows prefill/ttft/memory (`model_weights_bytes` + `kv_pool_bytes` engine helpers, das heaps)
+ `hardware` line (CPU brand via argv `run_and_capture`, worker lanes, GPU device name via new
`set_moe_gpu_device_report` hook feeding `GpuTierStatus.device`); page: separate prefill and
generation tok/s charts, TTFT + memory tiles, hardware disclosure under the header. Verified:
scheduler 12/12, server 17/17 live (post-work stats assertions), page via mock+playwright.

---

## Slices (order = build order)

### S1 — Config editor (the TOML round-trip)
- **daslib**: new `write_toml(JsonValue?) : string` in `daslib/toml.das` (parse-only today).
  Canonical emit, comments lost — accepted. Doc + tests (round-trip read_toml∘write_toml).
- **Precedence flip**: page-saved TOML carries `authoritative = true`; `apply_config` gains:
  authoritative beside-program TOML **overrides** CLI flags (defaults < CLI < authoritative).
  No watchdog surgery — it keeps re-running the same command line.
- **Save**: `POST /config` writes the FULL effective config to `dasllama-server.toml` beside
  the program (created if the server started config-less). `GET /config` returns effective
  config + per-key source (default / cli / toml) for the form.
- **Restart**: `POST /restart` = drain like `/shutdown`, then exit with a **distinct** exit
  code (3 stays tune-restart; pick 4 = config-restart; `watchdog.py` treats both as relaunch,
  logs them apart).
- **Model picker**: server enumerates `*.gguf` / `*.dlim` in the configured model's directory
  (`load_model` already auto-caches gguf→`.dlim` beside the file in q8 mode — show which rail
  is active).
- **GPU backend switching (gap confirmed 2026-07-18)**: metal has `--metal off|auto|required`;
  the vulkan tier is ENV-ONLY (`DASLLAMA_GPU_MOE_LAYERS/MOE_STREAM/DN/ATTN/DENSE/CLS/COMBINE/
  VRAM_MB`, read inside the load walk). Add first-class config: a `gpu = off | metal |
  metal-required | vulkan` selector plus vulkan detail keys (`gpu_layers` K, `gpu_stream` S,
  `gpu_dn`, `gpu_attn`, `gpu_dense`, `gpu_vram_mb`; defaults = the blessed K=2+S=35+DN+ATTN
  shape when `gpu = vulkan`). Engine side: programmatic setters in `dasllama_math` mirroring
  the env reads (env stays as override), set before load — pairs naturally with S9's status
  getter (config in, status out through the same surface). Config editor + page badge then
  cover backend switching end-to-end.
  **Capability guards (Boris 2026-07-18)**: the vulkan tier today serves MoE only (qwen-family
  class models). When `gpu = vulkan` is configured but the loaded model can't ride the tier
  (dense-only arch, unsupported quant/plane class), the server must say so — status surface
  carries `supported : bool` + human `reason`; the config form warns "not supported for this
  model" and the badge shows the reason instead of silently arming nothing. Hook deeper as the
  tier grows arch coverage.
- **Page**: typed form — dropdowns for `quant`/`kv_dtype`/`metal`/`team_dispatch`, numbers,
  path fields with the picker; save → confirm → restart flow with the draining badge.

### S2 — Stats depth (the numbers HN asks about)
- **Scheduler**: cumulative `prefill_tokens` counter beside `gen_tokens`; per-stream TTFT
  (first-token timestamp − admit timestamp) recorded on each stream.
- **/v1/stats grows**: `prefill_tokens`, memory footprint (weights bytes, KV pages ×
  page bytes, das heap/string heap), hardware line (CPU brand string, worker lanes, GPU name
  when the vulkan tier is up).
- **Page**: prefill tok/s and emission tok/s as **two separate charts**; memory tile (GB);
  one-line hardware+config disclosure under the header — screenshots self-document.

### S3 — Prompt state + swimlane (the parallelization money shot)
- **New poll payload** (`/v1/streams` or folded into stats): per stream — id, state
  (`queued` / `prefilling x/y` / `decoding` / `finished`), age, token counts, TTFT, and a
  capped live text tail (prompt head + generated tail). Server keeps tails only; the PAGE
  accumulates transcripts client-side (reload loses history — accepted). Finished streams
  linger ~10 s flagged `finished`, then drop; the page moves them to a **History tab**.
- **Swimlane**: the streams line chart is REPLACED by a slot swimlane — one row per stream
  slot up to `max_streams`, prefill/decode in two shades, idle gaps, sharing the time axis
  with the tok/s charts. Built from poll samples (per-slot state per tick), no event stream.
- **Stream cards**: live text flowing per active stream ("this is literally what's generated").

### S4 — Cache contents
- Prefix cache entries surfaced: token count, pages held, age, hit count, plus a ~100-char
  **detokenized preview** stored once at insert.
- Page: cache tab/panel. Demo moment: re-send a long conversation → TTFT collapses, cache row
  hits tick up (pairs with S2's TTFT).

### S5 — Chat mode
- Client-held conversation over the existing `/v1/chat/completions` SSE — no new server
  surface. `<think>` traces shown inline in a **different font** (no collapse UI — "not a
  working tool replacement").
- **All** sampling knobs exposed: temperature, top_k, top_p, min_p, presence/frequency
  penalties, max_tokens, thinking toggle — a tuning panel.
- **Mic input** (when `--asr`): record in browser → `POST /v1/audio/transcriptions` →
  transcript into the input box.

### S6 — ASR previews + stats
- Waveform rendered client-side (the page holds the mic/upload audio); **VAD spans** overlaid —
  from the serving path if silero already runs there, else one tiny preview endpoint.
- **Segment timeline** from verbose transcription under the waveform; click a segment → play
  that span in-browser.
- **RTF** (audio-seconds per wall-second) per job + rolling chart; ASR job cards
  (queued/running/done, text appearing) beside the LLM stream cards; cumulative ASR stats
  (jobs, audio seconds).
- Log-mel spectrogram explicitly PARKED → future standalone wasm `.das` demo.

### S7 — Demo load generator
- `demo_load.das` beside the server (dogfoods the dashv client): `new_thread` workers looping
  varied chat completions. Default = **staged ramp** 1→2→4→8 workers, a few requests per
  stage — the swimlane fills and total tok/s stair-steps. `-n` flag for constant load.

### S8 — Benchmark button (the llama.cpp A/B)
- Optional config key `lcpp_bin` (path to llama.cpp build); button disabled without it.
- Run **quiesced only**: refuse (or offer drain-first) while streams are active — contended
  numbers are misleading. Sequential: our bench (lcpp_bench methodology, same threads/model)
  then `llama-bench` on the same GGUF; both rows + delta into a bench tab, kept with hardware
  line + timestamp. Progress shown (runs are minutes-class).

### S9 — GPU-tier badge (SAME PR — Boris confirmed)
- Small public tier-status getter in `dasllama_math` (state is internal today): active
  backend, resident expert stacks K / streamed S, DN on/off, ATTN on/off, VRAM bytes.
- Page badge: which rails are on + VRAM. Hybrid-by-layer-*type* is the differentiator vs
  `-ngl` — say it.

### S10 — The reveal footer
- "How this works": every kernel is daslang, JIT-compiled via LLVM at startup, auto-tuned per
  box; line counts; repo link. HN's click-through.

### Finish
- Tests per slice ride `test_openai_server.das` (model-gated) + mock/playwright for page
  behavior; `write_toml` gets daslib tests; README route table + demo/bench sections.
- **Claude Design pass** over the final contents in the Forge system, then apply.
- One PR (server + daslib/toml writer + scheduler counters + dasllama_math getter + page +
  demo script + tests + README), per `skills/make_pr.md`.

---

## Deferred — recorded gates

- **V2 (GATED on the 0.6.4 release):** the 30-second try-it — copy-paste serve + demo
  commands on the page/README. Needs **shippable daslang for `[tune]`** (per-box tuning from a
  release binary). Add when 0.6.4 is out.
- **V3 (if ever):** multi-LLM serving (jobque = one model at a time today; easier over
  `.dlim`); per-model stats restructure rides it.
