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
| pp512 | 1198.3 | 1213.5 | 0.99x |
| tg128 B=1 | 108.2 | 101.5 | **1.07x** |
| tg128 B=2 | 140.1 | 112.8 | **1.24x** |
| tg128 B=4 | 151.8 | 167.1 | 0.91x |
| tg128 B=8 | 150.5 | 200.3 | 0.75x |
| tg128 B=16 | 324.1 | 315.7 | **1.03x** |

Known gap: B=4/B=8 ride the M2.1 interim per-stream kq GEMVs (B separate B1 weight passes per
site — no scaling B=2→8); the kq mul_mm twins take over at B>=9 and win again. lcpp's B=1
batched-bench row is 93.3 (its batch machinery costs ~8% vs llama-bench's 101.5 single loop);
das --fixed-token twin beats both.
