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

### Q4_K_M (native kq planes: k4 + k6, tied Q6_K classifier)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1197.9 | 1213.5 | 0.99x |
| tg128 B=1 | 112.5 | 101.5 | **1.11x** |
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

### Q6_K (pure k6 — every weight site rides the k6 kernels)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1107.8 | 1199.2 | **0.92x RED** |
| tg128 B=1 | 97.5 | 97.3 | **1.00x** |
| tg128 B=2 | 117.4 | 107.5 | **1.09x** |
| tg128 B=4 | 162.5 | 164.9 | **0.99x RED** |
| tg128 B=8 | 175.8 | 195.6 | **0.90x RED** |
| tg128 B=16 | 289.0 | 310.6 | **0.93x RED** |

FAILED CELLS — the roster's broad red. Q4_K_M only rode the k6 kernels on attn_v/ffn_down/cls;
a pure Q6_K model puts them on EVERY site and the weight triples: B=8 0.90x = the k6 ext4 twin
(the known round-2 levers — byte-fragmented quant loads, 53 scattered byte loads a superblock
that want uint32 views, and ext4 waits=11); B=16 0.93x and pp512 0.92x = the k6 mul_mm twin
(prefill and B>=9 share it). B=1 1.00x (greedy) — the k6 b1d GEMV holds at every-site weight.

### Q5_K_M (k5-dominant, k6 mixed in; B=2..8 ride the new k5 ext mv twins)

| shape | das tok/s | lcpp tok/s | das/lcpp |
| :--- | ---: | ---: | ---: |
| pp512 | 1075.1 | 1090.8 | 0.99x |
| tg128 B=1 | 82.0 | 84.0 | **0.98x RED** |
| tg128 B=2 | 123.1 | 90.5 | **1.36x** |
| tg128 B=4 | 171.9 | 159.1 | **1.08x** |
| tg128 B=8 | 186.2 | 187.3 | 0.99x |
| tg128 B=16 | 292.0 | 294.7 | 0.99x |

The k5 ext twins carry the batch ladder: B=2 1.36x (lcpp has no small-batch ext form at ne11
2-3 and its k5 mul_mv is its weakest K-quant kernel), B=4 1.08x. OPEN: B=1 0.98x (-2.4%,
greedy) — the lab has our k5 b1d/b2r walls ABOVE lcpp's per-kernel, so the residual is the
same sequential-phase class as Q8 B=1, scaled by k5's lower tok/s; B=8 -0.6% and B=16 -0.9%
thin; pp512 -1.4% = the standing pp deferral class. lcpp's B=1 batched-bench row is 80.7 —
das batched-B1 does 80.1 on the fixed-token protocol.
