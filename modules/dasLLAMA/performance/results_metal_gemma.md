# Metal gemma-line scoreboard (living doc — CURRENT numbers only; history in git)

The wave B stages 1+2 board: every gemma model that serves on Metal today — gemma2-2b
(NEOX + attn/final softcaps + sliding even layers + GeGLU + 4 norms, hs 256, dim 2304 /
8q/4kv / hidden 9216, 26 layers) and gemma3-1b/4b (drops the softcaps, adds QK-norm +
5:1 SWA with dual-theta per-layer rope; 1b: dim 1152 / 4q/1kv hs 256; 4b: dim 2560 /
8q/4kv hs 256). The gemma4-12B roster follows with wave B stage 3. Metal vs Metal, one
rep per cell, the 3B methodology verbatim (`results_metal_3b.md`): das = dasLLAMA
(`-jit`, mapped `.dlim` image, `MetalMode.required`); llama.cpp = stock flags, Metal
default, `-fa auto`; single-stream from `llama-bench -r 1`, batch ladder from
`llama-batched-bench -c 12288 -npp 512 -ntg 128`. das tg128 B=1 rows are GREEDY; batch
rows use `--fixed-token 100` with `DASLLAMA_METAL_BATCH_PIPE=1`; zero-copy logits on
everywhere. lcpp baselines pinned in `baseline_metal_gemma_m1.tsv`, NOT re-measured.
Ratio = das / llama.cpp, >1 = das faster; winners bold.

## Apple M1 Max — daslang branch `bbatkin/dasllama-metal-wave-a`, llama.cpp 7642f1c, 2026-07-17 (Parsec off)

das rails: `batch_decode_perf.das --prefill 512 --ngen 128 --bmax 16 --kv f16
--fixed-token 100` (pins: decode=metal, backend=portable, mirror 3072MB; B=1 row from a
separate greedy `--bmax 1` run, no pipe) + `prefill_perf.das` N=512 row (prefill=metal).

### gemma2-2b Q4_K_M (native kq planes)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1635.8 | 1654.1 | 0.99x |
| tg128 B=1 | 112.6 | 112.9 | **1.00x** |
| tg128 B=2 | 166.1 | 121.9 | **1.36x** |
| tg128 B=4 | 203.5 | 205.0 | 0.99x |
| tg128 B=8 | 231.6 | 229.5 | **1.01x** |
| tg128 B=16 | 331.0 | 348.1 | 0.95x |

Single-stream parity on the first wide-head model out of the gate; the B=2 win mirrors
the qwen3-4B Q4 board (1.37x — the mv-shape batch kernels' sweet spot). B=16 -5% is the
chase candidate.

### gemma2-2b Q8_0 (q8 repack, s16 scale plane)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1863.8 | 1896.9 | 0.98x |
| tg128 B=1 | 90.0 | 98.6 | **0.91x RED** |
| tg128 B=2 | 161.3 | 153.0 | **1.05x** |
| tg128 B=4 | 263.3 | 254.0 | **1.04x** |
| tg128 B=8 | 286.2 | 293.6 | 0.97x |
| tg128 B=16 | 422.4 | 394.4 | **1.07x** |

B=1 0.91x = the Q8 sequential-phase anomaly class (3B 0.97x, qwen3-4B 0.95x — same
signature, batch rows green on the same forward), deferred to the end of the wave chain.

### gemma3-1b Q8_0

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 4777.5 | 4974.6 | 0.96x |
| tg128 B=1 | 187.0 | 155.8 | **1.20x** |
| tg128 B=2 | 336.7 | 314.5 | **1.07x** |
| tg128 B=4 | 485.5 | 494.3 | 0.98x |
| tg128 B=8 | 675.8 | 664.1 | **1.02x** |
| tg128 B=16 | 883.3 | 767.7 | **1.15x** |

The small-model single-stream win (+20%): the resident driver's per-step overhead is a
fraction of lcpp's graph launch at this size. Dual-theta rope + QK-norm + 5:1 SWA all on
the serving path.

### gemma3-4b Q8_0

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1194.7 | 1217.9 | 0.98x |
| tg128 B=1 | 60.8 | 68.8 | **0.88x RED** |
| tg128 B=2 | 95.0 | 99.5 | 0.95x |
| tg128 B=4 | 157.2 | 164.3 | 0.96x |
| tg128 B=8 | 179.2 | 175.3 | **1.02x** |
| tg128 B=16 | 276.1 | 290.1 | 0.95x |

B=1 0.88x = the Q8 anomaly class again, amplified by a gemma-specific term: the greedy
run's own batched-B1 form collapsed to 45.3 tok/s (0.75x of its sequential twin) — a
~5.6 ms/step CPU tail. Gemma's sqrt(dim) embed scale stands the kq spec chain DOWN
(the chain requires embed_scale == 1.0), so every greedy step pays the CPU argmax +
embed poke over the 262k vocab — the exact tail the spec chain kills on qwen. The
embed-scale uniform on the GPU gather (ledger item 4 in "Gemma stage-1 Metal
deferrals", API_REWORK.md) is THE gemma B=1 lever; revisit at the gemma4-12B board.
The mild B=2/4/16 red (-4..-5%) rides the same per-step tail at small B.

## Reading the board

Prefill 0.96-0.99x everywhere and the batch ladder within ±7% with scattered das wins —
the wave B capability kernels (softcap epilogues, sliding chunk skips, GeGLU ew, sandwich
norms, dual-theta rope picks) land at effective parity with lcpp's fused Metal graph on
first measurement, before any gemma-shaped perf pass. The two structural gaps are named:
the Q8 sequential B=1 class (deferred, one combined deep-dive at the end of the wave
chain) and the spec-chain embed-scale stand-down (ledger; bites every gemma greedy row).
