# Metal gemma-line scoreboard (living doc — CURRENT numbers only; history in git)

The wave B stages 1+2 board: every gemma model that serves on Metal today — gemma2-2b
(NEOX + attn/final softcaps + sliding even layers + GeGLU + 4 norms, hs 256, dim 2304 /
8q/4kv / hidden 9216, 26 layers) and gemma3-1b/4b (drops the softcaps, adds QK-norm +
5:1 SWA with dual-theta per-layer rope; 1b: dim 1152 / 4q/1kv hs 256; 4b: dim 2560 /
8q/4kv hs 256) and the gemma4-12B roster (wave B stage 3: the heterogeneous line — two
attention classes 16q/8kv/256 sliding vs 16q/1kv/512 MQA global, V-from-K, v_norm,
out_scale, final softcap + suppress; dim 3840, 48 layers, vocab 262k). Metal vs Metal, one
rep per cell, the 3B methodology verbatim (`results_metal_3b.md`): das = dasLLAMA
(`-jit`, mapped `.dlim` image, `MetalMode.required`); llama.cpp = stock flags, Metal
default, `-fa auto`; single-stream from `llama-bench -r 1`, batch ladder from
`llama-batched-bench -c 12288 -npp 512 -ntg 128`. das tg128 B=1 rows are GREEDY; batch
rows use `--fixed-token 100` with `DASLLAMA_METAL_BATCH_PIPE=1`; zero-copy logits on
everywhere. lcpp baselines pinned in `baseline_metal_gemma_m1.tsv`, NOT re-measured.
Ratio = das / llama.cpp, >1 = das faster; winners bold.

## Apple M1 Max — daslang branch `bbatkin/dasllama-metal-wave-a`, llama.cpp 7642f1c, 2026-07-17/18 (quiet box)

das rails: `batch_decode_perf.das --prefill 512 --ngen 128 --bmax 16 --kv f16
--fixed-token 100 --seq once` (pins: decode=metal, backend=portable; B=1 rows from a
separate greedy `--bmax 1` run, no pipe) + `prefill_perf.das` N=512 row (prefill=metal).
Greedy B=1 rows include the spec chain + the GPU embed-scale gather (every gemma is
spec-capable since the escale uniform landed). gemma4 rows measured 2026-07-18 AFTER the
batch V-copy/v_norm race fix + the hetero mul_mv un-gate (80fab16b6); small-model batch
rows are 2026-07-17 (that code is untouched by both fixes), their greedy B=1 rows
re-measured post-embed-scale. gemma4 K-quant B=8 rows re-measured 2026-07-18 with the
single-pass kq b8 twins (`DASLLAMA_METAL_KQ_B8`, cd88fd1ea; metal-mode rail — GPU prefill;
decode cells are prefill-method-independent). Q8 batch cells re-measured after scope C
(V-copy fusion d6a9a7a4b + hetero split-K cf991ccdf); Q4 b8/b16 re-verified unchanged.

### gemma2-2b Q4_K_M (native kq planes)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1635.8 | 1654.1 | 0.99x |
| tg128 B=1 | 118.8 | 112.9 | **1.05x** |
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
| tg128 B=1 | 87.6 | 98.6 | **0.89x RED** |
| tg128 B=2 | 161.3 | 153.0 | **1.05x** |
| tg128 B=4 | 263.3 | 254.0 | **1.04x** |
| tg128 B=8 | 286.2 | 293.6 | 0.97x |
| tg128 B=16 | 422.4 | 394.4 | **1.07x** |

B=1 0.89x = the Q8 sequential-phase anomaly class (3B 0.97x, qwen3-4B 0.95x — same
signature, batch rows green on the same forward; the spec chain mildly hurts this
mid-size Q8 cell — 90.0 with it adaptive-off), deferred to the end of the wave chain.

### gemma3-1b Q8_0

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 4777.5 | 4974.6 | 0.96x |
| tg128 B=1 | 204.7 | 155.8 | **1.31x** |
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
| tg128 B=1 | 62.4 | 68.8 | **0.91x RED** |
| tg128 B=2 | 95.0 | 99.5 | 0.95x |
| tg128 B=4 | 157.2 | 164.3 | 0.96x |
| tg128 B=8 | 179.2 | 175.3 | **1.02x** |
| tg128 B=16 | 276.1 | 290.1 | 0.95x |

B=1 0.91x = the Q8 sequential class (the embed-scale uniform on the GPU gather landed —
the spec chain now serves every gemma greedy row; it lifted this cell 60.8 -> 62.4 and
gemma3-1b to 1.31x, but mid-size Q8 stays red on the per-step dispatch tail). The mild
B=2/4/16 red (-4..-5%) rides the same per-step tail at small B.

### gemma4-12B Q4_K_M (native kq planes)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 306.7 | 309.0 | 0.99x |
| tg128 B=1 | 32.7 | 29.0 | **1.13x** |
| tg128 B=2 | 41.9 | 31.2 | **1.34x** |
| tg128 B=4 | 49.0 | 51.2 | 0.96x |
| tg128 B=8 | 57.1 | 53.5 | **1.07x** |
| tg128 B=16 | 91.6 | 100.5 | 0.91x |

The daily-driver headline: +13% single-stream greedy (the spec chain + GPU embed-scale
gather over the 262k vocab) and the B=2 mv-kernel win, on the first dense model where
the batch loop runs per-class uniform twins every layer.

### gemma4-12B Q5_K_M

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 276.5 | 277.0 | **1.00x** |
| tg128 B=1 | 23.3 | 22.5 | **1.04x** |
| tg128 B=2 | 36.0 | 23.9 | **1.51x** |
| tg128 B=4 | 45.9 | 48.3 | 0.95x |
| tg128 B=8 | 53.8 | 49.7 | **1.08x** |
| tg128 B=16 | 85.0 | 93.0 | 0.91x |

B=2 1.51x is the board's widest win: lcpp's K-quant ext kernels only batch from B=4
(their B=2/3 runs the per-token mul_mv), while the kq mvb twins reuse the weight pass.

### gemma4-12B Q6_K (pure k6)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 298.8 | 304.5 | 0.98x |
| tg128 B=1 | 22.4 | 27.7 | **0.81x RED** |
| tg128 B=2 | 42.0 | 29.8 | **1.41x** |
| tg128 B=4 | 46.1 | 49.6 | 0.93x |
| tg128 B=8 | 56.3 | 50.6 | **1.11x** |
| tg128 B=16 | 86.6 | 98.4 | 0.88x |

The B=1 red is greedy-specific and NEW information: the fixed-token sequential twin runs
26.2 (0.95x) while greedy runs 22.4 — the spec chain that gains +5..13% on q4/q5/q8
COSTS ~14% on the pure-k6 file at 12B (it gained +16% at 4B). Spec-chain engagement on
big pure-k6 = open item; `DASLLAMA_METAL_SPEC=0` recovers the 0.95x class today.

### gemma4-12B Q8_0 (q8 repack, s16 scale plane)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 362.7 | 356.0 | **1.02x** |
| tg128 B=1 | 20.5 | 25.2 | **0.81x RED** |
| tg128 B=2 | 36.0 | 38.0 | 0.95x |
| tg128 B=4 | 58.8 | 63.8 | 0.92x |
| tg128 B=8 | 54.3 | 62.0 | **0.88x RED** |
| tg128 B=16 | 96.3 | 114.9 | **0.84x RED** |

B=2/4 were 0.90/0.59 before the hetero mul_mv un-gate (the fixed-B forms streamed the
12.2GB blob once PER TOKEN under hetero); B=4/8/16 gained +2/+6.5/+6% from scope C (the
V-copy→ones-RMS fusion + split-K un-gated under hetero via the per-class twins). The
remaining column tail is structural and named: B=1 = the Q8 sequential class (the q8 GEMV
itself measures ~352 GB/s — lcpp bandwidth class; the gap is the ~4.6 ms/step elementwise
dispatch tail lcpp's norm-chain fusion avoids), and the same dispatch-tail class plus
GEMM-tier scheduling noise holds the B=8/16 residue (the K-quant ladders' B=8/16 are
rep-stable to 0.1%; Q8's wobble 8-11% between reps — a scope D datapoint). lcpp's Q8 is
its strongest Metal suite (ext kernels from B=2, highest bytes/weight), so this column's
denominator is the hardest.

## Reading the board

Prefill 0.96-1.02x everywhere, and the gemma4-12B daily driver serves at +13% greedy
single-stream (Q4) with B=2 wins of 1.34-1.51x and B=8 wins of 1.07-1.11x across the
K-quants (the single-pass kq b8 twins — ONE weight stream instead of two column-group
passes; the kernel lab could not see this win at SLC-resident shapes, only the in-graph
A/B could). The remaining reds are structural and each has a named owner: the Q8
sequential/dispatch-tail class (gemma2-q8 0.89, gemma3-4b 0.91, gemma4-q8 0.81 — the
~4.6 ms/step elementwise tail; one combined deep-dive), the Q8 B=8/16 GEMM-tier residue
(0.88/0.84 after scope C's fusion + hetero split-K — the rest is the dispatch-tail class
plus GEMM-tier scheduling noise), the K-quant B=16 cells (0.88-0.91 — the kq mul_mm tier
itself: K-quant files never dispatch split-K, which is a q8-blob kernel; earlier boards
misattributed this to the sk stand-down), the K-quant B=4 cells (0.93-0.96 — the round-3
kernel lab REFUTED further kq-GEMV headroom at B=4: issue-bound at the wall), and the
Q6-greedy spec inversion (open item; spec-off recovers 0.95x).
