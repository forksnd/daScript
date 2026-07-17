# Metal 3B format scoreboard (living doc — CURRENT numbers only; history in git)

Llama-3.2-3B-Instruct, Metal vs Metal, one rep per cell. das = dasLLAMA (`-jit`, mapped `.dlim`
image, `MetalMode.required` — CPU fallback panics, so every cell is honestly GPU). llama.cpp =
stock flags, Metal default, `-fa auto`; single-stream from `llama-bench -r 1`, batch ladder from
`llama-batched-bench -npp 512 -ntg 128`. das protocol: tg128 B=1 rows are GREEDY (the real
single-stream workload — the speculative decode chain is a product feature and fixed-token
feeding defeats it, costing q8 14%); batch rows use `--fixed-token` (batched-bench parity). lcpp
baselines are pinned in `baseline_metal_3b_m1.tsv` and are NOT re-measured — only the das side
reruns. Hand-maintained (not generated). Ratio = das / llama.cpp, >1 = das faster; winners bold.

## Apple M1 Max — daslang branch `bbatkin/dasllama-3b-format-profiling`, llama.cpp 7642f1c, 2026-07-17 (Parsec off)

das rails: `batch_decode_perf.das -p 512 -n 128 -b 16 --kv f16 --fixed-token 100` (pins:
decode=metal, backend=portable, mirror 3072MB, batch pipe on) + `prefill_perf.das` N=512 row.

High-B cells (B=8/B=16) carry ±2-3% window-to-window variance (more CPU-side tail and
scheduling in the step) — a cell at 0.98-1.02x is AT the line, not a trend; judge it across
windows before calling regression or victory.

### Q4_K_M (native kq planes: k4 + k6, tied Q6_K classifier)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1215.6 | 1213.5 | **1.00x** |
| tg128 B=1 | 111.6 | 101.5 | **1.10x** |
| tg128 B=2 | 149.1 | 112.8 | **1.32x** |
| tg128 B=4 | 189.0 | 167.1 | **1.13x** |
| tg128 B=8 | 196.8 | 200.3 | 0.98x |
| tg128 B=16 | 314.6 | 315.7 | 1.00x |

B=2..8 ride the kq ext small-batch mv twins (one weight pass per 4-column group — the round-2
gemv-lab winners; the round-1 per-stream form lost B=4/8 at 0.91x/0.75x). B>=9 = the kq mul_mm
twins. The k6 round-1 kernels (see Q6_K) lift the k6 sites here too — pp512's -1.3% deferral
closed to parity. B=8 measured 1.02x the previous window, 0.98x this one — the wobble class.
lcpp's B=1 batched-bench row is 93.3 (its batch machinery costs ~8% vs llama-bench's 101.5
single loop); das beats both.

### Q8_0 (q8 repack, s16 scale plane — the load default)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1380.5 | 1395.9 | 0.99x |
| tg128 B=1 | 82.0 | 86.4 | **0.95x RED** |
| tg128 B=2 | 140.4 | 127.9 | **1.10x** |
| tg128 B=4 | 232.9 | 218.7 | **1.07x** |
| tg128 B=8 | 250.4 | 252.7 | 0.99x |
| tg128 B=16 | 401.2 | 362.7 | **1.11x** |

OPEN RED CELLS: tg128 B=1 at 0.95x (greedy, warmed bench — the 256-step warmup confirms the
phase split is NOT clock ramp): the sequential phase does 82.0 while the batched-B1 form does
85.8 = 0.99x on the same forward() — the per-process-phase anomaly is still open (per-32-step
bucket split is the parked diagnostic); the converged gap to lcpp is ~1% at a
bandwidth-saturated shape (3.37 GB/token). B=8 at 0.99x (-0.9%) — the emission-arc soft cell,
improved from 0.94x; candidate fix = the ext-form twin with the q8 A-stage replacing mv8 at
B=5..8. pp512 0.99x (-1.1%) — same class as the Q4_K_M deferral. Q4_K_M follow-up lever:
its spec chain is gated OFF (tied cls = k6, not cls_q8) — extending speculation to the kq
classifier stacks on its 1.11x.

### Q6_K (pure k6 — every weight site rides the k6 kernels; round-1 kernels live)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1214.7 | 1199.2 | **1.01x** |
| tg128 B=1 | 97.5 | 97.3 | **1.00x** |
| tg128 B=2 | 152.4 | 107.5 | **1.42x** |
| tg128 B=4 | 184.0 | 164.9 | **1.12x** |
| tg128 B=8 | 192.6 | 195.6 | 0.98x |
| tg128 B=16 | 300.7 | 310.6 | **0.97x RED** |

The k6 round-1 kernels flipped the broad red (was pp 0.92 / B=2 1.09 / B=4 0.99 / B=8 0.90 /
B=16 0.93): the mul_mm A-stage got the fma fold (-32 into a dmn term) + the vectorized 6-bit
compose (one masked-or per uint), and the ext twins swapped 32 scattered byte loads per chunk
for 4+4 uint loads with the same compose (gemv lab: ext2u -22..28%, ext4u -13..16%; das q6
prefill now equals das q4 prefill, mirroring lcpp's own q4~q6 behavior). REMAINING: B=16
0.97x — per-step GPU 50.9ms vs lcpp's 51.5ms TOTAL; the gap is the non-overlapped CPU tail
(logits readback 1.24ms = 16 x vocab f32, setup 0.57ms, sched 0.91ms). A batch-driver overlap
item (deferred landing / async readback), same family as the Q8 B=1 phase split — NOT a
kernel cell. B=8 0.98x = the wobble class. An ext-twin probe at B=16 lost to the mul_mm
(128 vs 294 tok/s — 4 dequant passes vs one dequant-to-tgmem); the B>8 crossover stands.

### Q5_K_M (k5-dominant, k6 mixed in; B=2..8 ride the new k5 ext mv twins)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1090.4 | 1090.8 | **1.00x** |
| tg128 B=1 | 82.3 | 84.0 | **0.98x RED** |
| tg128 B=2 | 128.3 | 90.5 | **1.42x** |
| tg128 B=4 | 177.9 | 159.1 | **1.12x** |
| tg128 B=8 | 189.2 | 187.3 | **1.01x** |
| tg128 B=16 | 288.6 | 294.7 | 0.98x |

The k5 ext twins carry the batch ladder — B=2 1.42x (lcpp has no small-batch ext form at ne11
2-3 and its k5 mul_mv is its weakest K-quant kernel), B=4 1.12x — and the k6 round-1 kernels
lift the k6 subset (B=8 flipped to 1.01x, pp to parity). OPEN: B=1 0.98x (-2.1%, greedy) —
the lab has our k5 b1d/b2r walls ABOVE lcpp's per-kernel, so the residual is the same
sequential-phase class as Q8 B=1, scaled by k5's lower tok/s; B=16 0.98x = the wobble class.
