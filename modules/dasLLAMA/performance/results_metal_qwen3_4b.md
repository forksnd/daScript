# Metal Qwen3-4B format scoreboard (living doc — CURRENT numbers only; history in git)

Qwen3-4B-Instruct-2507 — the family-waves Wave A showcase (NEOX rope + QK-norm + qd != dim:
dim 2560 / qd 4096 / kv 1024 / hidden 9728, 36 layers). Metal vs Metal, one rep per cell,
the 3B methodology verbatim (`results_metal_3b.md`): das = dasLLAMA (`-jit`, mapped `.dlim`
image, `MetalMode.required`); llama.cpp = stock flags, Metal default, `-fa auto`;
single-stream from `llama-bench -r 1`, batch ladder from `llama-batched-bench -npp 512
-ntg 128`. das tg128 B=1 rows are GREEDY; batch rows use `--fixed-token 100` with
`DASLLAMA_METAL_BATCH_PIPE=1`; zero-copy logits on everywhere. lcpp baselines pinned in
`baseline_metal_qwen3_4b_m1.tsv`, NOT re-measured. Ratio = das / llama.cpp, >1 = das
faster; winners bold.

## Apple M1 Max — daslang branch `bbatkin/dasllama-metal-wave-a`, llama.cpp 7642f1c, 2026-07-17 (Parsec off)

das rails: `batch_decode_perf.das --prefill 512 --ngen 128 --bmax 16 --kv f16
--fixed-token 100` (pins: decode=metal, backend=portable, mirror 3072MB; B=1 row from a
separate greedy `--bmax 1` run, no pipe) + `prefill_perf.das` N=512 row (prefill=metal).

### Q4_K_M (native kq planes: k4 + k6)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 914.3 | 929.2 | 0.98x |
| tg128 B=1 | 82.5 | 82.1 | **1.00x** |
| tg128 B=2 | 122.2 | 89.2 | **1.37x** |
| tg128 B=4 | 151.5 | 127.7 | **1.19x** |
| tg128 B=8 | 163.0 | 152.2 | **1.07x** |
| tg128 B=16 | 247.6 | 234.5 | **1.06x** |

Decode all green; the daily-driver format holds parity single-stream and leads the whole
batch ladder. pp512 -1.6% — see the pp line note below.

### Q8_0 (q8 repack, s16 scale plane — the load default)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1041.7 | 1064.9 | 0.98x |
| tg128 B=1 | 64.5 | 67.7 | **0.95x RED** |
| tg128 B=2 | 112.7 | 103.0 | **1.09x** |
| tg128 B=4 | 182.7 | 159.6 | **1.14x** |
| tg128 B=8 | 186.1 | 190.4 | 0.98x |
| tg128 B=16 | 282.1 | 266.0 | **1.06x** |

B=1 0.95x = the 3B Q8 sequential-phase anomaly class (the 3B board sits at 0.97 with the
same signature: the batched-B1 form beats the sequential loop on the same forward()).

### Q6_K (pure k6)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 918.3 | 916.3 | **1.00x** |
| tg128 B=1 | 66.6 | 78.2 | **0.85x RED** |
| tg128 B=2 | 106.9 | 83.7 | **1.28x** |
| tg128 B=4 | 142.0 | 124.2 | **1.14x** |
| tg128 B=8 | 153.6 | 137.6 | **1.12x** |
| tg128 B=16 | 232.2 | 230.8 | **1.01x** |

FAILED CELL, twice over: (a) the k6 GEMV trails lcpp's mul_mv at the 4B shapes even
fixed-token (71.1 sequential = 0.91x — the 3B k6 B=1 is 1.01x, so this is shape-dependent:
wo is 2560x4096, w13 9728-wide); (b) a greedy-path inversion COSTS ~6% ON TOP (greedy 66.6
vs fixed-token 71.1 — on the 3B board greedy WINS by ~3.5%; reproduced twice, not
throttle). Both are the follow-up chase; neither is wobble.

### Q5_K_M (k5-dominant, k6 mixed in)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 820.1 | 831.2 | 0.99x |
| tg128 B=1 | 59.9 | 66.5 | **0.90x RED** |
| tg128 B=2 | 104.3 | 71.2 | **1.46x** |
| tg128 B=4 | 142.3 | 120.5 | **1.18x** |
| tg128 B=8 | 154.3 | 143.0 | **1.08x** |
| tg128 B=16 | 228.2 | 219.0 | **1.04x** |

B=1 0.90x: sequential phases were flat across every run (59.0-59.9 — honest, not
throttle); the k5 GEMV trails at the 4B shapes (its 3B cell is 0.99x). B=2 1.46x = the k5
ext twin vs lcpp's weakest K-quant mul_mv, same as the 3B board.

## Board summary + the follow-up chase

Batch B=2..16 is GREEN across every format (17 of 18 cells; B=2 leads 1.28-1.46x). pp512
sits AT the line (0.98-1.00x; the QK-norm prepass adds 2 dispatches/layer to prefill —
ledger). **The single-stream B=1 column is the red line: 1.00 / 0.95 / 0.85 / 0.90** vs
the 3B board's 1.11 / 0.97 / 1.01 / 0.99. Two distinct causes, both measured, neither
wobble:
1. **K-quant GEMV shape regression at 4B geometry** (k6 worst): the k5/k6 B=1 kernels were
   lab-tuned at 3B shapes (square 3072, hidden 8192); the 4B's qd != dim (wo 2560x4096) and
   9728-wide FFN change the walls. Chase = the gemv lab at 4B shapes.
2. **Greedy-path inversion (Q6 measured, ~6%)**: greedy B=1 trails fixed-token B=1 — the
   opposite of the 3B protocol note (greedy wins ~3.5% via the spec/argmax rail). Suspect
   the per-step logits consumption path at 4B; joins the parked 3B sequential-phase
   diagnostic.
3. The QK-norm prepass dispatch (~155us/token = ~1% of B=1 here) rides on top of every
   qwen3 decode cell — the fused rope-store H-form in the API_REWORK ledger recovers it.
