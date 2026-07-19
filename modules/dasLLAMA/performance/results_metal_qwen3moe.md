# Metal Qwen3-30B-A3B (qwen3moe) format scoreboard (living doc — CURRENT numbers only; history in git)

Qwen3-30B-A3B-Instruct-2507 — the Wave C MoE showcase (GPU routing: router GEMV + top-k
select + expert-indexed GEMVs on decode/batch, CSR-gathered mul_mm on prefill; dim 2048 /
ne 128 / k 8 / nfe 768, 48 layers). Metal vs Metal, one rep per cell, the 3B methodology
(`results_metal_3b.md`): das = dasLLAMA (`-jit`, mapped metal-flavor `.dlim`,
`DASLLAMA_PIN_DECODE=metal` = MetalMode.required); llama.cpp = stock flags, Metal default,
`-fa auto`; single-stream from `llama-bench -r 1`, batch ladder from `llama-batched-bench
-npp 512 -ntg 128`. das tg128 B=1 rows are GREEDY; batch rows use `--fixed-token 100` with
`DASLLAMA_METAL_BATCH_PIPE=1`. lcpp baselines pinned in `baseline_metal_qwen3moe_m1.tsv`,
NOT re-measured. Ratio = das / llama.cpp, >1 = das faster; winners bold.

## Apple M1 Max — daslang branch `bbatkin/dasllama-metal-wave-c`, llama.cpp 7642f1c, 2026-07-19 (Parsec off)

das rails: `batch_decode_perf.das --prefill 512 --ngen 128 --bmax 16 --kv f16
--fixed-token 100 --seq skip` (pins: `DASLLAMA_PIN_DECODE=metal DASLLAMA_PIN_BACKEND=portable`,
default 4096MB mirror; B=1 row from a separate greedy `--bmax 1` run, no pipe) +
`prefill_perf.das` N=512 row (`DASLLAMA_PIN_PREFILL=metal`).

### Q4_K_M (native kq expert stacks: k4 gate/up + k6 down)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 786.6 | 842.0 | 0.93x |
| tg128 B=1 | 72.6 | 75.3 | 0.96x |
| tg128 B=2 | 98.9 | 87.6 | **1.13x** |
| tg128 B=4 | 114.9 | 111.6 | **1.03x** |
| tg128 B=8 | 127.2 | 128.4 | 0.99x |
| tg128 B=16 | 141.8 | 144.0 | 0.98x |

First MoE-on-Metal board: single-stream and the B>=4 ladder land within the parity class
on the very first measurement pass; B=2 leads 1.13x. pp512 0.93x — the gathered expert
mul_mm chain runs unfused with a serial CSR build per layer (perf ledger).

### Q5_K_M (local requant from Q8; k5 gate/up + k6 down — parity smoke 24/24 token-exact)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 707.0 | 783.8 | 0.90x |
| tg128 B=1 | 57.8 | 62.8 | 0.92x |
| tg128 B=2 | 83.9 | 70.6 | **1.19x** |
| tg128 B=4 | 99.7 | 91.5 | **1.09x** |
| tg128 B=8 | 109.1 | 103.1 | **1.06x** |
| tg128 B=16 | 120.6 | 113.2 | **1.07x** |

Batch ladder all green; B=1 0.92x is the k5 single-stream class (the 4B chase's b1c
select-form pick does not yet apply to the expert-indexed twin — perf ledger).

### Q6_K (local requant from Q8; pure-k6 stacks — parity smoke 24/24 token-exact)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 728.9 | 837.1 | 0.87x |
| tg128 B=1 | 68.1 | 68.3 | 1.00x |
| tg128 B=2 | 102.4 | 77.8 | **1.32x** |
| tg128 B=4 | 120.6 | 99.2 | **1.22x** |
| tg128 B=8 | 134.0 | 113.9 | **1.18x** |
| tg128 B=16 | 145.6 | 126.4 | **1.15x** |

B=1 dead-even, the whole batch ladder green 1.15-1.32x.

### Q8_0 (q8 expert stacks through the 34B blob rail)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 871.3 | 963.8 | 0.90x |
| tg128 B=1 | 67.1 | 54.9 | **1.22x** |
| tg128 B=2 | 104.9 | 68.6 | **1.53x** |
| tg128 B=4 | 134.5 | 81.5 | **1.65x** |
| tg128 B=8 | 150.2 | 89.7 | **1.67x** |
| tg128 B=16 | 159.3 | 93.5 | **1.70x** |

das sweeps the ENTIRE Q8 decode ladder — the expert-indexed q8 blob GEMVs (one weight
pass per (slot, stream) with GPU-side routing) against llama.cpp's per-token q8 MoE
mul_mv: the same weakness class as their kq B=2/3 hole, here across the whole ladder and
growing with B. pp512 0.90x (same gathered-chain ledger item as Q4).
