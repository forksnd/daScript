# tests/dasLLAMA — testing discipline

The Metal suites here are wall-time-expensive (model loads dominate; a full pass holds 40GB
GGUFs). The rules below exist because one session spent 5.75 of 6 hours re-running full suites
to verify one-arm fixes. They are enforcement, not advice.

## Run suites ONLY through the runner

```
./bin/daslang -jit tests/dasLLAMA/run.das -- --arm <filter> [--suite decode|prefill|matrix|all]
./bin/daslang -jit tests/dasLLAMA/run.das -- --full [--suite ...]
```

Never invoke `dastest/dastest.das --test tests/dasLLAMA/...` directly for the metal suites.
The runner refuses to run without exactly one of `--arm` / `--full`, tees the COMPLETE output
to a log file (path printed on the DONE line), owns the dastest timeout (arm 1200s, full
3600s — the 70B tails), and repeats only when `--nreps` is passed explicitly (default 1,
never best-of-N).

## The iteration loop

1. Fixing/adding one arm → run exactly that arm: `--arm arm12 --suite decode` (~minutes).
   Scratchpad probes that replicate one arm in isolation are encouraged for kernel/driver
   fixes — cheaper still.
2. Batch every pending fix. Do NOT re-run a full suite per fix.
3. The pre-commit gate is ONE `--full --suite all` pass (large-tier models included via
   `DASLLAMA_PARITY_FULL=1`, which the runner sets). Green gate ⇒ commit.
4. Before launching ANY suite, state what the change can affect; a default-off knob or a
   comment edit does not need a rerun.

## Arm filter mechanics

`DASLLAMA_TEST_ARMS` (set by `--arm`) is a comma list of substrings matched against arm names
by `arm_on(t, name)` in `_model_tier.das`. Filtered arms register a LOUD `t |> skip`, so a
filtered run reports SKIPPED, never PASS — partial coverage cannot masquerade as full. A
failed assert still FAILS the test (dastest: `failed` beats `skipped`), so filtering can
never mask a red.

Arm names — decode parity: `arm1-basic arm2-hybrid arm3-step arm4-paged arm5-rewind
arm6-churn arm7-q8kv arm7b-tq4kv arm8-s16 arm9-reload arm10-kq arm11-depth arm12-dim`,
batch test: `batch` (whole test), `batchB7-partd`, `batchB8-kq`. Prefill parity: `base s16
kq dim qkv`. Support matrix: `cells-q8 window cells-s16 mode kq dim8b dim70b`.

## Model tiers

`model_available(t, path)` gates by size: models over 6GiB (the 8B, the 70B) run only under
`DASLLAMA_PARITY_FULL=1` — i.e. only in the `--full` gate. A default run reporting SKIPPED
for those arms is correct, not a failure. The 70B is further restricted: no CPU-control
batch parity against it (streaming 40GB on the CPU while the GPU has the same bytes wired
OOM-kills a 64GB box) — 70B batch coverage is ENGAGE-only in the support matrix; the
batched code paths get their parity on small models via pins (e.g.
`set_metal_batch_addrms_unfused`).

## Log discipline

Always capture COMPLETE logs (the runner does this); grep afterwards, never at capture time —
a capture-time filter once hid the exact proof line a verification run existed to produce.
When a fixture claims a size/depth property ("2030 tokens", "crosses 2048"), assert the
actual number in the test; a resize cap is not evidence.
