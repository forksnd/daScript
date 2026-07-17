# Metal 3B format scoreboard (living doc — CURRENT numbers only; history in git)

Llama-3.2-3B-Instruct, Metal vs Metal, one rep per cell. das = dasLLAMA (`-jit`, mapped `.dlim`
image, `MetalMode.required` — CPU fallback panics, so every cell is honestly GPU). llama.cpp =
stock flags, Metal default, `-fa auto`; single-stream from `llama-bench -r 1`, batch ladder from
`llama-batched-bench -npp 512 -ntg 128` (das twin: `batch_decode_perf --fixed-token`). lcpp
baselines are pinned in `baseline_metal_3b_m1.tsv` and are NOT re-measured — only the das side
reruns. Hand-maintained (not generated). Ratio = das / llama.cpp, >1 = das faster; winners bold.

## Apple M1 Max — daslang branch `bbatkin/dasllama-3b-format-profiling`, llama.cpp 7642f1c, 2026-07-17 (Parsec off)

das rails: `batch_decode_perf.das -p 512 -n 128 -b 16 --kv f16 --fixed-token 100` (pins:
decode=metal, backend=portable, mirror 3072MB, batch pipe on) + `prefill_perf.das` N=512 row.

### Q4_K_M (native kq planes: k4 + k6, tied Q6_K classifier)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1197.9 | 1213.5 | 0.99x |
| tg128 B=1 | 108.6 | 101.5 | **1.07x** |
| tg128 B=2 | 146.1 | 112.8 | **1.30x** |
| tg128 B=4 | 188.3 | 167.1 | **1.13x** |
| tg128 B=8 | 204.7 | 200.3 | **1.02x** |
| tg128 B=16 | 323.1 | 315.7 | **1.02x** |

B=2..8 ride the kq ext small-batch mv twins (one weight pass per 4-column group — the round-2
gemv-lab winners; the round-1 per-stream form lost B=4/8 at 0.91x/0.75x). B>=9 = the kq mul_mm
twins. pp512 at 0.99x (-1.3%, stable): APPROVED DEFERRAL — revisit after the format roster.
lcpp's B=1 batched-bench row is 93.3 (its batch machinery costs ~8% vs llama-bench's 101.5
single loop); das beats both.

### Q8_0 (q8 repack, s16 scale plane — the load default)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1380.5 | 1395.9 | 0.99x |
| tg128 B=1 | 71.9 | 86.4 | **0.83x RED** |
| tg128 B=2 | 140.4 | 127.9 | **1.10x** |
| tg128 B=4 | 232.9 | 218.7 | **1.07x** |
| tg128 B=8 | 250.4 | 252.7 | 0.99x |
| tg128 B=16 | 401.2 | 362.7 | **1.11x** |

OPEN RED CELLS: tg128 B=1 at 0.83x — and the tell: das's OWN batched path at B=1 does 79.9
(both pipe modes), beating the dedicated single-stream driver's 71.9-72.9. The single-stream
rail is leaving ~10% on the table on q8 specifically (Q4_K_M shows no such split: 108.4 both
forms). Under diagnosis. B=8 at 0.99x (-0.9%) — the emission-arc soft cell, improved from
0.94x but still a hair short; candidate fix = the ext-form twin with the q8 A-stage replacing
mv8 at B=5..8. pp512 0.99x (-1.1%) — same class as the Q4_K_M deferral.
