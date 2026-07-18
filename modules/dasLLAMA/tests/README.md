# dasLLAMA tests

These module-owned suites are intentionally outside the core `tests/` tree. Run them explicitly:

```text
bin/daslang -jit dastest/dastest.das -- --test modules/dasLLAMA/tests/
```

Model-driven tests self-skip when their fixture models are absent. For the expensive Metal suites,
follow [CLAUDE.md](CLAUDE.md) and use the scoped runner instead of launching the whole directory.

The suite covers architecture registration, audio and Whisper, batching and dispatch, the public
facade, forward and parity fixtures, quantization and KV codecs, CPU and Metal kernels, model images,
MTP, tokenization, sampling, Unicode, and the tune/tuned integration surfaces.
