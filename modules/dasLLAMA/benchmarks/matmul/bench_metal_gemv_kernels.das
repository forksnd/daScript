options gen2
options persistent_heap

require dastest/testing_boost
require metal/msl_shader
require ?das_metal metal/das_metal_boost  // nolint:STYLE030 — GPU driver half, Apple builds only
require ?das_metal dasllama/dasllama_math_metal  // nolint:STYLE030 — the production 32x32 GEMM MSL (the batch-crossover row)
require dasllama/dasllama_metal_prefill  // nolint:STYLE030 — the production GEMV MSL (v0 baseline)
require llvm/daslib/f16_cvt  // f16 bit converts for the K-quant superblock fills
require daslib/fio
require math  // nolint:STYLE030 — used only inside the Apple static_if half
require strings

// The Metal Q8 GEMV kernel lab — the emission (decode) arc's weight-stream detector. Decode is
// GEMV-bandwidth-bound: one token reads every weight once (3B q8 ~3.4GB/token), so the metric
// that matters is the WEIGHT GB/s each variant sustains, not GMAC/s. The single-stream done gate
// (llama.cpp Metal tg128 = 87 t/s on the 3B) implies ~340 GB/s effective on M1 Max.
//
// Batched decode (B independent streams, one weight pass) turns the GEMV into a skinny GEMM:
// variants here stream the weight row ONCE and dot it against B activation vectors — the
// arithmetic-intensity amplifier the eval_batch override will ride. B in {1,2,4,8,16}.
//
// Correctness is bit-exact by construction (same contract as bench_metal_gemm_kernels): quants
// confined to [-8,7], scales to powers of two, so every product and partial sum is an exactly
// representable integer times a power of two — any accumulation order yields identical bits.
// v0 = the production metal_q8_gemv MSL text (zero copy drift); every variant must match the
// CPU reference exactly on the sampled region, and same-B variants must match each other.
//
//   bin/daslang -jit dastest/dastest.das -- --bench --test modules/dasLLAMA/benchmarks/matmul/bench_metal_gemv_kernels.das
//
// DASMETAL_LAB_ROUNDS: interleaved timing rounds (default 3, best-of)
// DASMETAL_LAB_SHAPES / DASMETAL_LAB_VARIANTS: substring filters for fast iteration
//
// READING THE NUMBERS — SLC recycling: back-to-back reps re-read the same weight buffer, so
// shapes whose weights fit the system-level cache (q3b 9.4MB, kv3b 3.1MB, w13/w2 25MB partly)
// report SLC bandwidth, not DRAM (q3b can exceed the 400GB/s DRAM peak). Real decode streams
// all ~3.6GB once per token — cls3b (394MB) is the DRAM-honest proxy shape.
//
// Round-2 verdicts (M1 Max, 2026-07-14): v0/B1 ~352 wGB/s on cls3b (the bar is ~340) — the
// single-stream GEMV is at target already; gemv_b1's plain loop matches it (the 4-in-flight
// unroll is neutral at decode shapes). Stream-major gemv_bN saturates ALU ~340-540 GMAC/s (a
// B=8 step costs 8 single GEMVs — no batch win); the transposed-X forms (b4t/b8t/b16t) are
// PATHOLOGICAL (6-24 wGB/s, 64B-strided lane loads wreck coalescing) — kept as negative
// controls. gemm32 (production 32x32 GEMM, M-pad-32) is B-independent: it beats B sequential
// GEMVs from B~4-6 on big shapes and cls, but only from B~10 on kv3b (d=1024 = 32 tgs,
// under-occupied) — the batched driver picks per (B, shape), like gemm_use64.
//
// Round-3 verdicts (M1 Max, 2026-07-14 late): the X-STAGED forms (bNx — tgmem panel chunk,
// cooperative 128-thread load) are the fixed-B winners: b2x +40-88% over b2 (326-362 wGB/s;
// kv3b neutral), b4x ~2x over b4 (188-217 — ABOVE lcpp's ~186 effective at ne11=4); the
// per-lane device x loads were the limiter, NOT fma. b8x/b16x double their baselines but stay
// UNDER the per-site GEMM policy (96-107 vs 109-124 wGB/s at B=8) — the B>=5 GEMM boundary
// holds. The chunked-prefetch forms (bNc/bNc8 — per-lane-CONSECUTIVE byte4 chunks, one scale
// load per chunk) are 30-45% WORSE everywhere: inter-lane coalescing beats per-lane locality
// on AGX (lcpp's mul_mv_ext chunks are strided per-thread too — its prefetch win is the load
// BATCHING, which [unroll_count=4] already banks). Kept as negative controls.
//
// K-quant round verdicts (M1 Max, 2026-07-15): k4_b1c (16B compact scale blocks, whole block in
// ONE uint4 load) beats the lcpp mul_mv_q4_K wall at every shape — parity on the SLC-resident
// small shapes (q3b/kv3b), -16..-25% on qkv/w13/w2/cls (cls3b 0.677 vs 0.872 ms, 327 wGB/s,
// 1.87x the q8 GEMV row; the 144/288-byte ceiling is 2.0x). k4_b1 (reading the production 20B
// scale blocks) is the negative control: the 20B stride's varying mod-16 phase keeps LLVM from
// merging the scale-triple into a vector load (3 scalar loads/row, +4 issues/iter) — 7-8%
// behind lcpp everywhere. Production uploads therefore strip the pads GPU-side: quant plane
// byte-identical (zero-copy-able), scale plane repacked 20B->16B at upload (2.7% of tensor bytes).

// ===== kernel variants =====
// Shared contract (f32-X planar): b0 wq [d x n] Q8 byte4 view, b1 ws [d x n/32] f32 scales,
// b2 x [B x n] f32 float4 view, b3 y [B x d] f32, b4 ndim = n, b5 ddim = d. One simdgroup per
// output row, 128 threads = 4 rows per threadgroup (the production gemv geometry). The MSL
// emitter is pure das, so the classes compile on every platform — only the driver is Apple-gated.

// Fixed-B batched gemv: the weight row (and its scale) is loaded ONCE per k step and dotted
// against B activation vectors held in registers — B accumulators per lane, unrolled compile-time.
// The scale folds into the weight ONCE (w4 * s), saving B-1 scale multiplies per byte4 — exact
// under the lab fill (int x pow2).
class MetalGemvLabB2 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b2_msl")]
    def lab_gemv_b2 {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[2]
        var k = lane
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            for [unroll_full] (b in range(2)) {
                let xv = x[uint(b) * nb4 + k]
                acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
            }
            k += gl_SubgroupSize
        }
        for [unroll_full] (b in range(2)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

class MetalGemvLabB4 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b4_msl")]
    def lab_gemv_b4 {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[4]
        // unroll_count=4 mirrors the production MetalGemvB4 (landed 3a1c1b6e9: +3.7% at B=4)
        var k = lane
        while [unroll_count=4] (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            for [unroll_full] (b in range(4)) {
                let xv = x[uint(b) * nb4 + k]
                acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
            }
            k += gl_SubgroupSize
        }
        for [unroll_full] (b in range(4)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

class MetalGemvLabB8 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b8_msl")]
    def lab_gemv_b8 {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[8]
        var k = lane
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            for [unroll_full] (b in range(8)) {
                let xv = x[uint(b) * nb4 + k]
                acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
            }
            k += gl_SubgroupSize
        }
        for [unroll_full] (b in range(8)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

class MetalGemvLabB16 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b16_msl")]
    def lab_gemv_b16 {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[16]
        var k = lane
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            for [unroll_full] (b in range(16)) {
                let xv = x[uint(b) * nb4 + k]
                acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
            }
            k += gl_SubgroupSize
        }
        for [unroll_full] (b in range(16)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

// B=1 control with the batched loop structure (single block in flight, no 4-deep ILP): isolates
// how much of v0's rate is the 4-blocks-in-flight unroll vs the plain stream.
class MetalGemvLabB1 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b1_msl")]
    def lab_gemv_b1 {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc = 0.0
        var k = lane
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            let xv = x[k]
            acc += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
            k += gl_SubgroupSize
        }
        acc = simd_sum(acc)
        if (lane == 0u) {
            y[row] = acc
        }
    }
}

// Chunked-prefetch twins (lcpp mul_mv_ext's chpt=4 discipline, plus per-lane-CONSECUTIVE chunks):
// each lane owns 4 adjacent byte4s (16 elements) per k step, so the whole chunk sits inside ONE
// scale block — 1 scale load per 16 elements instead of 4 — and the quant loads share a base
// register (base+immediate addressing, merge-friendly). W dequants into registers up front, the
// dot chains follow: ~4 waits per 16 elements where the strided forms wait every 4. Requires
// n % 512 == 0 (the driver adds a strided tail; every lab shape tiles exactly).
class MetalGemvLabB2C {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b2c_msl")]
    def lab_gemv_b2c {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[2]
        let step = gl_SubgroupSize * 4u
        var c = lane * 4u
        while (c < nb4) {
            let sv = ws[sbase + c / 8u]
            var wf : float4[4]
            for [unroll_full] (g in range(4)) {
                wf[g] = float4(int4(wq[wbase + c + uint(g)])) * sv
            }
            for [unroll_full] (b in range(2)) {
                let xb = uint(b) * nb4 + c
                for [unroll_full] (g in range(4)) {
                    let xv = x[xb + uint(g)]
                    acc[b] += wf[g].x * xv.x + wf[g].y * xv.y + wf[g].z * xv.z + wf[g].w * xv.w
                }
            }
            c += step
        }
        for [unroll_full] (b in range(2)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

class MetalGemvLabB4C {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b4c_msl")]
    def lab_gemv_b4c {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[4]
        let step = gl_SubgroupSize * 4u
        var c = lane * 4u
        while (c < nb4) {
            let sv = ws[sbase + c / 8u]
            var wf : float4[4]
            for [unroll_full] (g in range(4)) {
                wf[g] = float4(int4(wq[wbase + c + uint(g)])) * sv
            }
            for [unroll_full] (b in range(4)) {
                let xb = uint(b) * nb4 + c
                for [unroll_full] (g in range(4)) {
                    let xv = x[xb + uint(g)]
                    acc[b] += wf[g].x * xv.x + wf[g].y * xv.y + wf[g].z * xv.z + wf[g].w * xv.w
                }
            }
            c += step
        }
        for [unroll_full] (b in range(4)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

// chunk-depth probes: 8 byte4s per lane (a FULL scale block per chunk — one scale, one select);
// doubles the staged registers, requires n % 1024 == 0
class MetalGemvLabB2C8 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b2c8_msl")]
    def lab_gemv_b2c8 {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[2]
        let step = gl_SubgroupSize * 8u
        var c = lane * 8u
        while (c < nb4) {
            let sv = ws[sbase + c / 8u]
            var wf : float4[8]
            for [unroll_full] (g in range(8)) {
                wf[g] = float4(int4(wq[wbase + c + uint(g)])) * sv
            }
            for [unroll_full] (b in range(2)) {
                let xb = uint(b) * nb4 + c
                for [unroll_full] (g in range(8)) {
                    let xv = x[xb + uint(g)]
                    acc[b] += wf[g].x * xv.x + wf[g].y * xv.y + wf[g].z * xv.z + wf[g].w * xv.w
                }
            }
            c += step
        }
        for [unroll_full] (b in range(2)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

class MetalGemvLabB4C8 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b4c8_msl")]
    def lab_gemv_b4c8 {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[4]
        let step = gl_SubgroupSize * 8u
        var c = lane * 8u
        while (c < nb4) {
            let sv = ws[sbase + c / 8u]
            var wf : float4[8]
            for [unroll_full] (g in range(8)) {
                wf[g] = float4(int4(wq[wbase + c + uint(g)])) * sv
            }
            for [unroll_full] (b in range(4)) {
                let xb = uint(b) * nb4 + c
                for [unroll_full] (g in range(8)) {
                    let xv = x[xb + uint(g)]
                    acc[b] += wf[g].x * xv.x + wf[g].y * xv.y + wf[g].z * xv.z + wf[g].w * xv.w
                }
            }
            c += step
        }
        for [unroll_full] (b in range(4)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

// X-staged batched gemv (the API_REWORK ledger's candidate (a)): the tg's 4 rows all dot the SAME
// B x-vectors, so stage a k-chunk of the X panel in threadgroup memory ONCE per tg (cooperative
// 128-thread load, 4x fewer device x loads) and the per-lane inner-loop x reads become tgmem hits.
// W stays strided/coalesced. Chunk = 64 float4s per stream (4KB tgmem at B=4). Barrier-safe by
// construction only when every simdgroup owns a real row (d % 4 == 0 — all decode shapes) and
// every thread iterates the same chunk count (n % 1024 == 0... n=3072/8192 both tile: nb4 % 64).
class MetalGemvLabB4X {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint
    @workgroup txp : float4[256]            // [4 streams x 64 float4] X panel chunk

    [metal_kernel(name="lab_gemv_b4x_msl")]
    def lab_gemv_b4x {
        let lid = gl_LocalInvocationID.x
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[4]
        var cb = 0u
        while (cb < nb4) {
            barrier()                       // previous round done reading the panel
            for [unroll_full] (j in range(2)) {
                let slot = uint(j) * 128u + lid
                txp[slot] = x[(slot / 64u) * nb4 + cb + slot % 64u]
            }
            barrier()
            for [unroll_full] (i in range(2)) {
                let k = cb + uint(i) * 32u + lane
                let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
                let kt = uint(i) * 32u + lane
                for [unroll_full] (b in range(4)) {
                    let xv = txp[uint(b) * 64u + kt]
                    acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
                }
            }
            cb += 64u
        }
        for [unroll_full] (b in range(4)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

class MetalGemvLabB2X {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint
    @workgroup txp : float4[128]            // [2 streams x 64 float4] X panel chunk

    [metal_kernel(name="lab_gemv_b2x_msl")]
    def lab_gemv_b2x {
        let lid = gl_LocalInvocationID.x
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[2]
        var cb = 0u
        while (cb < nb4) {
            barrier()
            let slot = lid
            txp[slot] = x[(slot / 64u) * nb4 + cb + slot % 64u]
            barrier()
            for [unroll_full] (i in range(2)) {
                let k = cb + uint(i) * 32u + lane
                let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
                let kt = uint(i) * 32u + lane
                for [unroll_full] (b in range(2)) {
                    let xv = txp[uint(b) * 64u + kt]
                    acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
                }
            }
            cb += 64u
        }
        for [unroll_full] (b in range(2)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

class MetalGemvLabB8X {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint
    @workgroup txp : float4[512]            // [8 streams x 64 float4] X panel chunk (8KB)

    [metal_kernel(name="lab_gemv_b8x_msl")]
    def lab_gemv_b8x {
        let lid = gl_LocalInvocationID.x
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[8]
        var cb = 0u
        while (cb < nb4) {
            barrier()
            for [unroll_full] (j in range(4)) {
                let slot = uint(j) * 128u + lid
                txp[slot] = x[(slot / 64u) * nb4 + cb + slot % 64u]
            }
            barrier()
            for [unroll_full] (i in range(2)) {
                let k = cb + uint(i) * 32u + lane
                let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
                let kt = uint(i) * 32u + lane
                for [unroll_full] (b in range(8)) {
                    let xv = txp[uint(b) * 64u + kt]
                    acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
                }
            }
            cb += 64u
        }
        for [unroll_full] (b in range(8)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

class MetalGemvLabB16X {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint
    @workgroup txp : float4[1024]           // [16 streams x 64 float4] X panel chunk (16KB)

    [metal_kernel(name="lab_gemv_b16x_msl")]
    def lab_gemv_b16x {
        let lid = gl_LocalInvocationID.x
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[16]
        var cb = 0u
        while (cb < nb4) {
            barrier()
            for [unroll_full] (j in range(8)) {
                let slot = uint(j) * 128u + lid
                txp[slot] = x[(slot / 64u) * nb4 + cb + slot % 64u]
            }
            barrier()
            for [unroll_full] (i in range(2)) {
                let k = cb + uint(i) * 32u + lane
                let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
                let kt = uint(i) * 32u + lane
                for [unroll_full] (b in range(16)) {
                    let xv = txp[uint(b) * 64u + kt]
                    acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
                }
            }
            cb += 64u
        }
        for [unroll_full] (b in range(16)) {
            let s = simd_sum(acc[b])
            if (lane == 0u) {
                y[uint(b) * ddim + row] = s
            }
        }
    }
}

// Transposed-X batched gemv: X stored stream-minor [n][16] so ONE float4 load covers 4 streams
// at one k — the B accumulator update vectorizes to B/4 vec4-fmas per k-element instead of B
// scalar chains (4x fewer ALU ops than the stream-major forms). Stride is baked at 16 streams;
// unused tail streams are zero-filled.
class MetalGemvLabB4T {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 xt : array<float4>   // [n][16] stream-minor, float4 = 4 adjacent streams
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b4t_msl")]
    def lab_gemv_b4t {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc = float4(0.0)
        var k = lane
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            let kk = k * 16u                // float4 slots: 4 per k-element, 16 per byte4
            acc += w4.x * xt[kk]
            acc += w4.y * xt[kk + 4u]
            acc += w4.z * xt[kk + 8u]
            acc += w4.w * xt[kk + 12u]
            k += gl_SubgroupSize
        }
        let s0 = simd_sum(acc.x)
        let s1 = simd_sum(acc.y)
        let s2 = simd_sum(acc.z)
        let s3 = simd_sum(acc.w)
        if (lane == 0u) {
            y[row] = s0
            y[ddim + row] = s1
            y[2u * ddim + row] = s2
            y[3u * ddim + row] = s3
        }
    }
}

class MetalGemvLabB8T {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 xt : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b8t_msl")]
    def lab_gemv_b8t {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc0 = float4(0.0)
        var acc1 = float4(0.0)
        var k = lane
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            let kk = k * 16u
            acc0 += w4.x * xt[kk]
            acc1 += w4.x * xt[kk + 1u]
            acc0 += w4.y * xt[kk + 4u]
            acc1 += w4.y * xt[kk + 5u]
            acc0 += w4.z * xt[kk + 8u]
            acc1 += w4.z * xt[kk + 9u]
            acc0 += w4.w * xt[kk + 12u]
            acc1 += w4.w * xt[kk + 13u]
            k += gl_SubgroupSize
        }
        for [unroll_full] (c in range(4)) {
            let s0 = simd_sum(acc0[c])
            let s1 = simd_sum(acc1[c])
            if (lane == 0u) {
                y[uint(c) * ddim + row] = s0
                y[uint(c + 4) * ddim + row] = s1
            }
        }
    }
}

class MetalGemvLabB16T {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 xt : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_gemv_b16t_msl")]
    def lab_gemv_b16t {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float4[4]
        var k = lane
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            let kk = k * 16u
            for [unroll_full] (g in range(4)) {
                acc[g] += w4.x * xt[kk + uint(g)]
                acc[g] += w4.y * xt[kk + 4u + uint(g)]
                acc[g] += w4.z * xt[kk + 8u + uint(g)]
                acc[g] += w4.w * xt[kk + 12u + uint(g)]
            }
            k += gl_SubgroupSize
        }
        for [unroll_full] (g in range(4)) {
            for [unroll_full] (c in range(4)) {
                let s = simd_sum(acc[g][c])
                if (lane == 0u) {
                    y[uint(g * 4 + c) * ddim + row] = s
                }
            }
        }
    }
}

// Runtime-B probe: same math with B as a uniform and a dynamic accumulator loop — measures what
// ONE PSO serving every batch size costs vs the compile-time-B set (the dynamic index likely
// spills acc[] to thread memory).
class MetalGemvLabBDyn {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint
    @uniform @binding = 6 bcount : uint

    [metal_kernel(name="lab_gemv_bdyn_msl")]
    def lab_gemv_bdyn {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[16]
        var k = lane
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k])) * ws[sbase + k / 8u]
            var b = 0u
            while (b < bcount) {
                let xv = x[b * nb4 + k]
                acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
                b++
            }
            k += gl_SubgroupSize
        }
        var bo = 0u
        while (bo < bcount) {
            let s = simd_sum(acc[bo])
            if (lane == 0u) {
                y[bo * ddim + row] = s
            }
            bo++
        }
    }
}

// q8-X probe (B=1): the activation arrives PRE-QUANTIZED (the CPU decode contract — xq/xs planes);
// integer byte4 dot, one combined scale per byte4. Reads n bytes of X instead of 4n — X is noise
// either way; this measures whether the int-dot pipeline beats the float one on the weight stream.
class MetalGemvLabQ8X {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 xq : array<byte4>
    @ssbo @binding = 3 xs : array<float>
    @ssbo @binding = 4 y : array<float>
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 ddim : uint

    [metal_kernel(name="lab_gemv_q8x_msl")]
    def lab_gemv_q8x {
        let row = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc = 0.0
        var k = lane
        while (k < nb4) {
            let w4 = int4(wq[wbase + k])
            let x4 = int4(xq[k])
            let idot = w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w
            acc += float(idot) * (ws[sbase + k / 8u] * xs[k / 8u])
            k += gl_SubgroupSize
        }
        acc = simd_sum(acc)
        if (lane == 0u) {
            y[row] = acc
        }
    }
}

// The 64-wide twin — the SAME v2 discipline on MulMm's 32(M) x 64(N) tile (its k-loop measured
// 156 instr vs lcpp's 165, per-element 29% under the 32-wide v2). Half the threadgroups of v2
// (d/64), so the per-shape winner is an occupancy-vs-ISA trade: wide-d sites (w13, cls) keep a
// healthy grid and should capture the per-element edge; 48/16-tg sites should stay 32-wide.
class MetalGemm64V2Lab {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>    // [32 x n] f32 rows (M-pad-32)
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint       // n (reduction dim)
    @uniform @binding = 5 ddim : uint       // d (output rows)
    @workgroup tw : float16[2048]           // W: 4 k-tiles x 8 row-tiles of 8x8, (k row, wrow col)
    @workgroup tx : float16[1024]           // X: 4 k-tiles x 4 token-tiles of 8x8, (token row, k col)

    [metal_kernel(name="lab_gemm64_v2_msl")]
    def lab_gemm64_v2 {
        let lid = gl_LocalInvocationID.x
        let sg = gl_SubgroupID
        let ntileN = ddim / 64u
        let mBase = (gl_WorkGroupID.x / ntileN) * 32u
        let nBase = (gl_WorkGroupID.x % ntileN) * 64u
        var mc : simdgroup_float8x8[8]      // this simdgroup's 16x32 quadrant: 2 token x 4 wrow tiles
        var ma : simdgroup_half8x8[4]
        var mb : simdgroup_half8x8[2]
        var va : float16[16]                // dequantized W staging regs (this thread's k-half)
        let kdim4 = ndim / 4u
        let nkb = ndim / 32u
        // W split: thread stages 16 elems — row lr0 (0..63), k-half il0 (0..1)
        let lr0 = lid / 2u
        let il0 = lid % 2u
        let tW0 = (il0 * 2u * 8u + lr0 / 8u) * 64u + (lr0 % 8u)
        // X split: thread stages 8 elems — token lr1 (0..31), k-eighth il1 (0..3)
        let lr1 = lid / 4u
        let il1 = lid % 4u
        let tX0 = (il1 * 4u + lr1 / 8u) * 64u + (lr1 % 8u) * 8u
        let aQ = (sg % 2u) * 256u           // wrow tile-quad base within a k-tile slab
        let bQ = (sg / 2u) * 128u           // token tile-pair base
        var wp = unsafe(addr(wq[(nBase + lr0) * kdim4 + il0 * 4u]))
        var sp = unsafe(addr(ws[(nBase + lr0) * nkb]))
        var xp = unsafe(addr(x[(mBase + lr1) * kdim4 + il1 * 2u]))
        var kb = 0u
        while (kb < nkb) {
            let sA = unsafe(sp[0])
            let b0 = unsafe(xp[0])
            let b1 = unsafe(xp[1])
            for [unroll_full] (g in range(4)) {
                let w4 = float4(int4(unsafe(wp[uint(g)]))) * sA
                va[g * 4] = float16(w4.x)
                va[g * 4 + 1] = float16(w4.y)
                va[g * 4 + 2] = float16(w4.z)
                va[g * 4 + 3] = float16(w4.w)
            }
            barrier()                       // previous math done reading the tiles
            for [unroll_full] (i in range(16)) {
                tw[tW0 + uint(i / 8) * 512u + uint(i % 8) * 8u] = va[i]
            }
            tx[tX0] = float16(b0.x)
            tx[tX0 + 1u] = float16(b0.y)
            tx[tX0 + 2u] = float16(b0.z)
            tx[tX0 + 3u] = float16(b0.w)
            tx[tX0 + 4u] = float16(b1.x)
            tx[tX0 + 5u] = float16(b1.y)
            tx[tX0 + 6u] = float16(b1.z)
            tx[tX0 + 7u] = float16(b1.w)
            barrier()
            var acur = unsafe(addr(tw[aQ]))
            var bcur = unsafe(addr(tx[bQ]))
            for [unroll_full] (_kt in range(4)) {
                for [unroll_full] (i in range(4)) {
                    simdgroup_load(ma[i], acur, uint(i) * 64u, 8u)
                }
                for [unroll_full] (i in range(2)) {
                    simdgroup_load(mb[i], bcur, uint(i) * 64u, 8u)
                }
                for [unroll_full] (i in range(8)) {
                    simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i])
                }
                acur = unsafe(acur + 512)
                bcur = unsafe(bcur + 256)
            }
            wp = unsafe(wp + 8u)
            sp = unsafe(sp + 1u)
            xp = unsafe(xp + 8u)
            kb++
        }
        let row0 = mBase + (sg / 2u) * 16u
        let col0 = nBase + (sg % 2u) * 32u
        for [unroll_full] (i in range(8)) {
            simdgroup_store(mc[i], y, (row0 + uint(i / 4) * 8u) * ddim + col0 + uint(i % 4) * 8u, ddim)
        }
    }
}

// The GemmB rework candidate — MulMm's staging discipline on the occupancy-friendly 32x32 tile:
// planar W + raw f32 X (the quant tails disappear), per-thread staging indexes hoisted out of the
// k-loop, device reads + dequant into registers BEFORE the barrier (they fly while the previous
// math drains), pointer-walked streams, k-major 8x8 W tiles so the mma reads straight.
class MetalGemmV2Lab {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>    // [32 x n] f32 rows (M-pad-32)
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint       // n (reduction dim)
    @uniform @binding = 5 ddim : uint       // d (output rows)
    @workgroup tw : float16[1024]           // W: 4 k-tiles x 4 row-tiles of 8x8, (k row, wrow col)
    @workgroup tx : float16[1024]           // X: 4 k-tiles x 4 token-tiles of 8x8, (token row, k col)

    [metal_kernel(name="lab_gemm_v2_msl")]
    def lab_gemm_v2 {
        let lid = gl_LocalInvocationID.x
        let sg = gl_SubgroupID
        let ntileN = ddim / 32u
        let mBase = (gl_WorkGroupID.x / ntileN) * 32u
        let nBase = (gl_WorkGroupID.x % ntileN) * 32u
        var mc : simdgroup_float8x8[4]      // this simdgroup's 16x16 quadrant: 2 token x 2 wrow tiles
        var ma : simdgroup_half8x8[2]
        var mb : simdgroup_half8x8[2]
        let kdim4 = ndim / 4u
        let nkb = ndim / 32u
        // staging split: thread handles row lr (0..31) x k-eighth il (0..3) — 8 elems of W AND X
        let lr = lid / 4u
        let il = lid % 4u
        let tW0 = (il * 4u + lr / 8u) * 64u + (lr % 8u)
        let tX0 = (il * 4u + lr / 8u) * 64u + (lr % 8u) * 8u
        let aQ = (sg % 2u) * 128u           // wrow tile-pair base within a k-tile slab
        let bQ = (sg / 2u) * 128u           // token tile-pair base
        var wp = unsafe(addr(wq[(nBase + lr) * kdim4 + il * 2u]))
        var sp = unsafe(addr(ws[(nBase + lr) * nkb]))
        var xp = unsafe(addr(x[(mBase + lr) * kdim4 + il * 2u]))
        var kb = 0u
        while (kb < nkb) {
            let sA = unsafe(sp[0])
            let q0 = unsafe(wp[0])
            let q1 = unsafe(wp[1])
            let b0 = unsafe(xp[0])
            let b1 = unsafe(xp[1])
            let w40 = float4(int4(q0)) * sA
            let w41 = float4(int4(q1)) * sA
            barrier()                       // previous math done reading the tiles
            tw[tW0] = float16(w40.x)
            tw[tW0 + 8u] = float16(w40.y)
            tw[tW0 + 16u] = float16(w40.z)
            tw[tW0 + 24u] = float16(w40.w)
            tw[tW0 + 32u] = float16(w41.x)
            tw[tW0 + 40u] = float16(w41.y)
            tw[tW0 + 48u] = float16(w41.z)
            tw[tW0 + 56u] = float16(w41.w)
            tx[tX0] = float16(b0.x)
            tx[tX0 + 1u] = float16(b0.y)
            tx[tX0 + 2u] = float16(b0.z)
            tx[tX0 + 3u] = float16(b0.w)
            tx[tX0 + 4u] = float16(b1.x)
            tx[tX0 + 5u] = float16(b1.y)
            tx[tX0 + 6u] = float16(b1.z)
            tx[tX0 + 7u] = float16(b1.w)
            barrier()
            var acur = unsafe(addr(tw[aQ]))
            var bcur = unsafe(addr(tx[bQ]))
            for [unroll_full] (_kt in range(4)) {
                simdgroup_load(ma[0], acur, 0u, 8u)
                simdgroup_load(ma[1], acur, 64u, 8u)
                simdgroup_load(mb[0], bcur, 0u, 8u)
                simdgroup_load(mb[1], bcur, 64u, 8u)
                simdgroup_multiply_accumulate(mc[0], mb[0], ma[0], mc[0])
                simdgroup_multiply_accumulate(mc[1], mb[0], ma[1], mc[1])
                simdgroup_multiply_accumulate(mc[2], mb[1], ma[0], mc[2])
                simdgroup_multiply_accumulate(mc[3], mb[1], ma[1], mc[3])
                acur = unsafe(acur + 256)
                bcur = unsafe(bcur + 256)
            }
            wp = unsafe(wp + 8u)
            sp = unsafe(sp + 1u)
            xp = unsafe(xp + 8u)
            kb++
        }
        let row0 = mBase + (sg / 2u) * 16u
        let col0 = nBase + (sg % 2u) * 16u
        simdgroup_store(mc[0], y, row0 * ddim + col0, ddim)
        simdgroup_store(mc[1], y, row0 * ddim + (col0 + 8u), ddim)
        simdgroup_store(mc[2], y, (row0 + 8u) * ddim + col0, ddim)
        simdgroup_store(mc[3], y, (row0 + 8u) * ddim + (col0 + 8u), ddim)
    }
}

// The split-K (vertical) twin of gemm_v2: grid gains a k-slice axis (grid.y = slice), each
// threadgroup walks nkb/ksplit k-blocks into its own partial-C plane, and a tiny reduce pass
// sums the planes — deterministic (no float atomics). At mp=32 the M axis is one tile, so
// k-slicing is the only way to grow the grid past d/32 threadgroups on the starved sites.
class MetalGemmV2SkLab {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float>
    @ssbo @binding = 2 x : array<float4>    // [32 x n] f32 rows (M-pad-32)
    @ssbo @binding = 3 part : array<float>  // [ksplit x 32 x d] partial-C planes
    @uniform @binding = 4 ndim : uint       // n (reduction dim)
    @uniform @binding = 5 ddim : uint       // d (output rows)
    @uniform @binding = 6 ksplit : uint
    @workgroup tw : float16[1024]
    @workgroup tx : float16[1024]

    [metal_kernel(name="lab_gemm_sk_msl")]
    def lab_gemm_sk {
        let lid = gl_LocalInvocationID.x
        let sg = gl_SubgroupID
        let ntileN = ddim / 32u
        let mBase = (gl_WorkGroupID.x / ntileN) * 32u
        let nBase = (gl_WorkGroupID.x % ntileN) * 32u
        let sl = gl_WorkGroupID.y
        var mc : simdgroup_float8x8[4]
        var ma : simdgroup_half8x8[2]
        var mb : simdgroup_half8x8[2]
        let kdim4 = ndim / 4u
        let kbn = (ndim / 32u) / ksplit
        let kb0 = sl * kbn
        let lr = lid / 4u
        let il = lid % 4u
        let tW0 = (il * 4u + lr / 8u) * 64u + (lr % 8u)
        let tX0 = (il * 4u + lr / 8u) * 64u + (lr % 8u) * 8u
        let aQ = (sg % 2u) * 128u
        let bQ = (sg / 2u) * 128u
        var wp = unsafe(addr(wq[(nBase + lr) * kdim4 + kb0 * 8u + il * 2u]))
        var sp = unsafe(addr(ws[(nBase + lr) * (ndim / 32u) + kb0]))
        var xp = unsafe(addr(x[(mBase + lr) * kdim4 + kb0 * 8u + il * 2u]))
        var kb = 0u
        while (kb < kbn) {
            let sA = unsafe(sp[0])
            let q0 = unsafe(wp[0])
            let q1 = unsafe(wp[1])
            let b0 = unsafe(xp[0])
            let b1 = unsafe(xp[1])
            let w40 = float4(int4(q0)) * sA
            let w41 = float4(int4(q1)) * sA
            barrier()
            tw[tW0] = float16(w40.x)
            tw[tW0 + 8u] = float16(w40.y)
            tw[tW0 + 16u] = float16(w40.z)
            tw[tW0 + 24u] = float16(w40.w)
            tw[tW0 + 32u] = float16(w41.x)
            tw[tW0 + 40u] = float16(w41.y)
            tw[tW0 + 48u] = float16(w41.z)
            tw[tW0 + 56u] = float16(w41.w)
            tx[tX0] = float16(b0.x)
            tx[tX0 + 1u] = float16(b0.y)
            tx[tX0 + 2u] = float16(b0.z)
            tx[tX0 + 3u] = float16(b0.w)
            tx[tX0 + 4u] = float16(b1.x)
            tx[tX0 + 5u] = float16(b1.y)
            tx[tX0 + 6u] = float16(b1.z)
            tx[tX0 + 7u] = float16(b1.w)
            barrier()
            var acur = unsafe(addr(tw[aQ]))
            var bcur = unsafe(addr(tx[bQ]))
            for [unroll_full] (_kt in range(4)) {
                simdgroup_load(ma[0], acur, 0u, 8u)
                simdgroup_load(ma[1], acur, 64u, 8u)
                simdgroup_load(mb[0], bcur, 0u, 8u)
                simdgroup_load(mb[1], bcur, 64u, 8u)
                simdgroup_multiply_accumulate(mc[0], mb[0], ma[0], mc[0])
                simdgroup_multiply_accumulate(mc[1], mb[0], ma[1], mc[1])
                simdgroup_multiply_accumulate(mc[2], mb[1], ma[0], mc[2])
                simdgroup_multiply_accumulate(mc[3], mb[1], ma[1], mc[3])
                acur = unsafe(acur + 256)
                bcur = unsafe(bcur + 256)
            }
            wp = unsafe(wp + 8u)
            sp = unsafe(sp + 1u)
            xp = unsafe(xp + 8u)
            kb++
        }
        let pbase = sl * 32u * ddim
        let row0 = mBase + (sg / 2u) * 16u
        let col0 = nBase + (sg % 2u) * 16u
        simdgroup_store(mc[0], part, pbase + row0 * ddim + col0, ddim)
        simdgroup_store(mc[1], part, pbase + row0 * ddim + (col0 + 8u), ddim)
        simdgroup_store(mc[2], part, pbase + (row0 + 8u) * ddim + col0, ddim)
        simdgroup_store(mc[3], part, pbase + (row0 + 8u) * ddim + (col0 + 8u), ddim)
    }
}

// the split-K reduce: y = sum of the ksplit partial planes (one thread per output element)
class MetalSkReduceLab {
    @ssbo @binding = 0 part : array<float>
    @ssbo @binding = 1 y : array<float>
    @uniform @binding = 2 total : uint      // 32 * d
    @uniform @binding = 3 ksplit : uint

    [metal_kernel(name="lab_sk_reduce_msl")]
    def lab_sk_reduce {
        let gid = gl_GlobalInvocationID.x
        if (gid >= total) {
            return
        }
        var acc = 0.0
        var s = 0u
        while (s < ksplit) {
            acc += part[s * total + gid]
            s++
        }
        y[gid] = acc
    }
}

// ===== K-quant (Q4_K) arms — the format-matrix W4 lab gate =====
// Weight contract: f32-X, in-kernel superblock dequant. Two weight layouts per format:
// the ggml interleaved block_q4_K stream (144B/sb: d f16, dmin f16, scales[12], qs[128]) for the
// lcpp wall arm, and the production plane pair (k4q 128B qs + k4s 20B [d][dmin][scales[12]][pad4],
// gguf_transcode_q4k's exact output) for ours. Geometry both: the lcpp mul_mv shape — 64-thread
// tg, 2 simdgroups x 2 rows (N_SG_Q4_K=2, N_R0_Q4_K=2), lane owns 32 elems of every 4th
// superblock (ix = lane/8 block-stream, iq/ir sub-position), y loaded once per superblock and
// reused across both rows. K-quant traffic/row: 148B (planes) / 144B (ggml) per 256 elems vs
// q8's 288B — the ~1.95x decode payoff W4 is after.

// the lcpp kernel_mul_mv_q4_K_f32 wall: their lane map, scale-halfword decode (kmask trick) and
// nibble-mask acc folds (x16/x256 groups folded by pow2 constants), on the ggml interleaved
// stream. Verbatim math; only the kargs plumbing is baked (dense strides from ndim/ddim).
class MetalKqLabLcppK4 {
    @ssbo @binding = 0 gsh : array<float16> // interleaved block_q4_K stream, half view (d at 72*blk)
    @ssbo @binding = 1 gqu : array<uint>    // the same stream, uint view (scales at 36*blk+1, qs at 36*blk+4)
    @ssbo @binding = 2 x : array<float>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint

    [metal_kernel(name="lab_k4_lcpp_msl")]
    def lab_k4_lcpp {
        let lane = gl_SubgroupInvocationID
        let ix = lane / 8u
        let it = lane % 8u
        let iq = it / 4u
        let ir = it % 4u
        let nb = ndim / 256u
        let first_row = (gl_WorkGroupID.x * 2u + gl_SubgroupID) * 2u
        var yl : float[16]
        var yh : float[16]
        var sumf : float[2]
        var ib = ix
        while (ib < nb) {
            let yb = ib * 256u + iq * 64u + ir * 8u
            var sumy : float[4]
            for [unroll_full] (i in range(8)) {
                yl[i] = x[yb + uint(i)]
                sumy[0] += yl[i]
                yl[i + 8] = x[yb + 32u + uint(i)]
                sumy[1] += yl[i + 8]
                yh[i] = x[yb + 128u + uint(i)]
                sumy[2] += yh[i]
                yh[i + 8] = x[yb + 160u + uint(i)]
                sumy[3] += yh[i + 8]
            }
            for [unroll_full] (r in range(2)) {
                let blk = (first_row + uint(r)) * nb + ib
                let so = blk * 36u + 1u
                let hw0 = (gqu[so] >> (16u * iq)) & 0xFFFF
                let hw2 = (gqu[so + 1u] >> (16u * iq)) & 0xFFFF
                let hw4 = (gqu[so + 2u] >> (16u * iq)) & 0xFFFF
                let sc0 = hw0 & 0x3F3F
                let sc1 = hw2 & 0x3F3F
                let sc2 = (hw4 & 0x0F0F) | ((hw0 & 0xC0C0) >> 2u)
                let sc3 = ((hw4 >> 4u) & 0x0F0F) | ((hw2 & 0xC0C0) >> 2u)
                let qb = blk * 36u + 4u + iq * 8u + ir * 2u
                let q1a = gqu[qb]
                let q1b = gqu[qb + 1u]
                let q2a = gqu[qb + 16u]
                let q2b = gqu[qb + 17u]
                var acc1 : float[4]
                var acc2 : float[4]
                for [unroll_full] (i in range(4)) {
                    let sh = 16u * (uint(i) & 1u)
                    let q1 = ((i < 2 ? q1a : q1b) >> sh) & 0xFFFF
                    let q2 = ((i < 2 ? q2a : q2b) >> sh) & 0xFFFF
                    acc1[0] += yl[2 * i] * float(q1 & 0x000F)
                    acc1[1] += yl[2 * i + 1] * float(q1 & 0x0F00)
                    acc1[2] += yl[2 * i + 8] * float(q1 & 0x00F0)
                    acc1[3] += yl[2 * i + 9] * float(q1 & 0xF000)
                    acc2[0] += yh[2 * i] * float(q2 & 0x000F)
                    acc2[1] += yh[2 * i + 1] * float(q2 & 0x0F00)
                    acc2[2] += yh[2 * i + 8] * float(q2 & 0x00F0)
                    acc2[3] += yh[2 * i + 9] * float(q2 & 0xF000)
                }
                let dd = float(gsh[blk * 72u])
                let dm = float(gsh[blk * 72u + 1u])
                sumf[r] += dd * ((acc1[0] + 0.00390625 * acc1[1]) * float(sc0 & 255u)
                    + (acc1[2] + 0.00390625 * acc1[3]) * float(sc0 >> 8u) * 0.0625
                    + (acc2[0] + 0.00390625 * acc2[1]) * float(sc2 & 255u)
                    + (acc2[2] + 0.00390625 * acc2[3]) * float(sc2 >> 8u) * 0.0625)
                sumf[r] -= dm * (sumy[0] * float(sc1 & 255u) + sumy[1] * float(sc1 >> 8u)
                    + sumy[2] * float(sc3 & 255u) + sumy[3] * float(sc3 >> 8u))
            }
            ib += 4u
        }
        for [unroll_full] (r in range(2)) {
            let s = simd_sum(sumf[r])
            if (lane == 0u && first_row + uint(r) < ddim) {
                y[first_row + uint(r)] = s
            }
        }
    }
}

// ours v1 — NEGATIVE CONTROL: the same lcpp shape reading the production 20B scale blocks
// (k4s dual f16/uint views). The 20B stride's varying mod-16 phase blocks LLVM's vector-load
// merge, so the scale triple compiles to 3 scalar loads per row — 7-8% behind the lcpp arm.
class MetalKqLabK4B1 {
    @ssbo @binding = 0 ksh : array<float16> // k4s plane, half view (d at 10*blk, dmin at 10*blk+1)
    @ssbo @binding = 1 ksu : array<uint>    // the same plane, uint view (scales at 5*blk+1)
    @ssbo @binding = 2 kqu : array<uint>    // k4q plane, uint view (qs of blk at 32*blk)
    @ssbo @binding = 3 x : array<float>
    @ssbo @binding = 4 y : array<float>
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 ddim : uint

    [metal_kernel(name="lab_k4_b1_msl")]
    def lab_k4_b1 {
        let lane = gl_SubgroupInvocationID
        let ix = lane / 8u
        let it = lane % 8u
        let iq = it / 4u
        let ir = it % 4u
        let nb = ndim / 256u
        let first_row = (gl_WorkGroupID.x * 2u + gl_SubgroupID) * 2u
        var yl : float[16]
        var yh : float[16]
        var sumf : float[2]
        var ib = ix
        while (ib < nb) {
            let yb = ib * 256u + iq * 64u + ir * 8u
            var sumy : float[4]
            for [unroll_full] (i in range(8)) {
                yl[i] = x[yb + uint(i)]
                sumy[0] += yl[i]
                yl[i + 8] = x[yb + 32u + uint(i)]
                sumy[1] += yl[i + 8]
                yh[i] = x[yb + 128u + uint(i)]
                sumy[2] += yh[i]
                yh[i + 8] = x[yb + 160u + uint(i)]
                sumy[3] += yh[i + 8]
            }
            for [unroll_full] (r in range(2)) {
                let blk = (first_row + uint(r)) * nb + ib
                let so = blk * 5u + 1u
                let hw0 = (ksu[so] >> (16u * iq)) & 0xFFFF
                let hw2 = (ksu[so + 1u] >> (16u * iq)) & 0xFFFF
                let hw4 = (ksu[so + 2u] >> (16u * iq)) & 0xFFFF
                let sc0 = hw0 & 0x3F3F
                let sc1 = hw2 & 0x3F3F
                let sc2 = (hw4 & 0x0F0F) | ((hw0 & 0xC0C0) >> 2u)
                let sc3 = ((hw4 >> 4u) & 0x0F0F) | ((hw2 & 0xC0C0) >> 2u)
                let qb = blk * 32u + iq * 8u + ir * 2u
                let q1a = kqu[qb]
                let q1b = kqu[qb + 1u]
                let q2a = kqu[qb + 16u]
                let q2b = kqu[qb + 17u]
                var acc1 : float[4]
                var acc2 : float[4]
                for [unroll_full] (i in range(4)) {
                    let sh = 16u * (uint(i) & 1u)
                    let q1 = ((i < 2 ? q1a : q1b) >> sh) & 0xFFFF
                    let q2 = ((i < 2 ? q2a : q2b) >> sh) & 0xFFFF
                    acc1[0] += yl[2 * i] * float(q1 & 0x000F)
                    acc1[1] += yl[2 * i + 1] * float(q1 & 0x0F00)
                    acc1[2] += yl[2 * i + 8] * float(q1 & 0x00F0)
                    acc1[3] += yl[2 * i + 9] * float(q1 & 0xF000)
                    acc2[0] += yh[2 * i] * float(q2 & 0x000F)
                    acc2[1] += yh[2 * i + 1] * float(q2 & 0x0F00)
                    acc2[2] += yh[2 * i + 8] * float(q2 & 0x00F0)
                    acc2[3] += yh[2 * i + 9] * float(q2 & 0xF000)
                }
                let dd = float(ksh[blk * 10u])
                let dm = float(ksh[blk * 10u + 1u])
                sumf[r] += dd * ((acc1[0] + 0.00390625 * acc1[1]) * float(sc0 & 255u)
                    + (acc1[2] + 0.00390625 * acc1[3]) * float(sc0 >> 8u) * 0.0625
                    + (acc2[0] + 0.00390625 * acc2[1]) * float(sc2 & 255u)
                    + (acc2[2] + 0.00390625 * acc2[3]) * float(sc2 >> 8u) * 0.0625)
                sumf[r] -= dm * (sumy[0] * float(sc1 & 255u) + sumy[1] * float(sc1 >> 8u)
                    + sumy[2] * float(sc3 & 255u) + sumy[3] * float(sc3 >> 8u))
            }
            ib += 4u
        }
        for [unroll_full] (r in range(2)) {
            let s = simd_sum(sumf[r])
            if (lane == 0u && first_row + uint(r) < ddim) {
                y[first_row + uint(r)] = s
            }
        }
    }
}

// ours v2: 16B scale blocks — the 20B transcode block minus its 4 pad bytes, built at upload.
// The 20B stride's varying mod-16 phase blocks LLVM's vector-load merge (the v1 scale triple
// compiles to 3 scalar loads; lcpp's 144B-stride triple gets one xyz), and 16B blocks land the
// whole [d|dmin][s0][s1][s2] block in ONE uint4 load at phase 0 — fewer issues than lcpp's
// xyz+halfpair AND 148->144B per 256 elems, byte-parity with ggml. Quant plane unchanged.
class MetalKqLabK4B1C {
    @ssbo @binding = 0 ksh : array<float16> // 16B scale blocks, half view (d at 8*blk, dmin at 8*blk+1)
    @ssbo @binding = 1 ks4 : array<uint4>   // the same plane, uint4 view (whole block at blk)
    @ssbo @binding = 2 kqu : array<uint>    // k4q plane, uint view (qs of blk at 32*blk)
    @ssbo @binding = 3 x : array<float>
    @ssbo @binding = 4 y : array<float>
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 ddim : uint

    [metal_kernel(name="lab_k4_b1c_msl")]
    def lab_k4_b1c {
        let lane = gl_SubgroupInvocationID
        let ix = lane / 8u
        let it = lane % 8u
        let iq = it / 4u
        let ir = it % 4u
        let nb = ndim / 256u
        let first_row = (gl_WorkGroupID.x * 2u + gl_SubgroupID) * 2u
        var yl : float[16]
        var yh : float[16]
        var sumf : float[2]
        var ib = ix
        while (ib < nb) {
            let yb = ib * 256u + iq * 64u + ir * 8u
            var sumy : float[4]
            for [unroll_full] (i in range(8)) {
                yl[i] = x[yb + uint(i)]
                sumy[0] += yl[i]
                yl[i + 8] = x[yb + 32u + uint(i)]
                sumy[1] += yl[i + 8]
                yh[i] = x[yb + 128u + uint(i)]
                sumy[2] += yh[i]
                yh[i + 8] = x[yb + 160u + uint(i)]
                sumy[3] += yh[i + 8]
            }
            for [unroll_full] (r in range(2)) {
                let blk = (first_row + uint(r)) * nb + ib
                let sv = ks4[blk]
                let hw0 = (sv.y >> (16u * iq)) & 0xFFFF
                let hw2 = (sv.z >> (16u * iq)) & 0xFFFF
                let hw4 = (sv.w >> (16u * iq)) & 0xFFFF
                let sc0 = hw0 & 0x3F3F
                let sc1 = hw2 & 0x3F3F
                let sc2 = (hw4 & 0x0F0F) | ((hw0 & 0xC0C0) >> 2u)
                let sc3 = ((hw4 >> 4u) & 0x0F0F) | ((hw2 & 0xC0C0) >> 2u)
                let qb = blk * 32u + iq * 8u + ir * 2u
                let q1a = kqu[qb]
                let q1b = kqu[qb + 1u]
                let q2a = kqu[qb + 16u]
                let q2b = kqu[qb + 17u]
                var acc1 : float[4]
                var acc2 : float[4]
                for [unroll_full] (i in range(4)) {
                    let sh = 16u * (uint(i) & 1u)
                    let q1 = ((i < 2 ? q1a : q1b) >> sh) & 0xFFFF
                    let q2 = ((i < 2 ? q2a : q2b) >> sh) & 0xFFFF
                    acc1[0] += yl[2 * i] * float(q1 & 0x000F)
                    acc1[1] += yl[2 * i + 1] * float(q1 & 0x0F00)
                    acc1[2] += yl[2 * i + 8] * float(q1 & 0x00F0)
                    acc1[3] += yl[2 * i + 9] * float(q1 & 0xF000)
                    acc2[0] += yh[2 * i] * float(q2 & 0x000F)
                    acc2[1] += yh[2 * i + 1] * float(q2 & 0x0F00)
                    acc2[2] += yh[2 * i + 8] * float(q2 & 0x00F0)
                    acc2[3] += yh[2 * i + 9] * float(q2 & 0xF000)
                }
                let dd = float(ksh[blk * 8u])
                let dm = float(ksh[blk * 8u + 1u])
                sumf[r] += dd * ((acc1[0] + 0.00390625 * acc1[1]) * float(sc0 & 255u)
                    + (acc1[2] + 0.00390625 * acc1[3]) * float(sc0 >> 8u) * 0.0625
                    + (acc2[0] + 0.00390625 * acc2[1]) * float(sc2 & 255u)
                    + (acc2[2] + 0.00390625 * acc2[3]) * float(sc2 >> 8u) * 0.0625)
                sumf[r] -= dm * (sumy[0] * float(sc1 & 255u) + sumy[1] * float(sc1 >> 8u)
                    + sumy[2] * float(sc3 & 255u) + sumy[3] * float(sc3 >> 8u))
            }
            ib += 4u
        }
        for [unroll_full] (r in range(2)) {
            let s = simd_sum(sumf[r])
            if (lane == 0u && first_row + uint(r) < ddim) {
                y[first_row + uint(r)] = s
            }
        }
    }
}

// ===== driver (Apple-only; every Metal-typed helper is a generic so non-Apple never compiles it) =====

struct private LabVariant {
    name : string
    src : string
    entry : string
    b_list : array<int>                     // batch sizes this kernel serves
    q8x : bool = false                      // b2/b3 get the quantized activation planes
    dyn_b : bool = false                    // b6 gets the bcount uniform
    xt : bool = false                       // b2 gets the stream-minor [n][16] f32 panel
    gemm : bool = false                     // the production 32x32 GEMM, M padded to 32 (q8 X)
    gemm_fx : bool = false                  // f32-X GEMM (default buffer list, gemm dispatch)
    gemm64_fx : bool = false                // f32-X GEMM, 64-wide N tile (d/64 grid)
    gemm_sk : bool = false                  // split-K f32-X GEMM (partial planes + reduce pass)
    sk : int = 0                            // slice count for gemm_sk
    tgmem : uint64 = 0ul                    // dynamic [[threadgroup(0)]] bytes (@workgroup kernels)
    kq : int = 0                            // K-quant plane arm: 4 = k4 (b0/b1 k4s dual view, b2 k4q)
    kq_ggml : bool = false                  // K-quant lcpp wall arm (b0/b1 = ggml interleaved stream)
    kq16 : bool = false                     // kq arm reads the 16B compact scale blocks instead
    tgsz : int = 128                        // threads per threadgroup (kq arms ride lcpp's 64)
}

// exact-arithmetic Q8 fill: quants in [-8, 7], pow2 scales (same contract as the gemm lab)
def private fill_exact_q8(var q : array<int8>; var s : array<float>; rows, k : int64; seed : int64) {
    let nb = k / 32l
    q |> resize(rows * k)
    s |> resize(rows * nb)
    let lut = fixed_array(0.25, 0.5, 1.0, 2.0)
    for (i in range64(rows * k)) {
        q[i] = int8(int((i * seed + i / 31l) % 16l) - 8)
    }
    for (i in range64(rows * nb)) {
        s[i] = lut[int((i / 7l) % 4l)]
    }
}

// CPU reference, sampled: first `refd` output rows of every stream — every partial sum exact
def private cpu_ref_sample(wq : array<int8>; ws : array<float>; xq : array<int8>; xs : array<float>;
                           n, nb_streams, refd : int64) : array<float> {
    var y : array<float>
    y |> resize(nb_streams * refd)
    let nb = n / 32l
    for (b in range64(nb_streams)) {
        for (j in range64(refd)) {
            var acc = 0.0
            for (kb in range64(nb)) {
                var blk = 0.0
                for (c in range64(32l)) {
                    blk += float(wq[j * n + kb * 32l + c]) * float(xq[b * n + kb * 32l + c])
                }
                acc += blk * (ws[j * nb + kb] * xs[b * nb + kb])
            }
            y[b * refd + j] = acc
        }
    }
    return <- y
}

// exact-arithmetic Q4_K fill: sub-scales/mins in [0,7], pow2 d/dmin, nibbles 0..15 — every
// product is an exact int x pow2 (the fill_exact_q8 contract). Emits BOTH weight layouts from
// one logical stream: the production plane pair (kq 128B qs + ks 20B scale blocks) and the ggml
// interleaved block_q4_K stream (gg 144B) for the lcpp wall arm.
def private fill_exact_k4(var kq : array<uint8>; var ks : array<uint8>; var gg : array<uint8>; rows, k : int64; seed : int64) {
    let nsb = k / 256l
    let total = rows * nsb
    kq |> resize(total * 128l)
    ks |> resize(total * 20l)
    gg |> resize(total * 144l)
    let lut = fixed_array(0.25, 0.5, 1.0, 2.0)
    for (sb in range64(total)) {
        let db = f32_to_f16(lut[int((sb / 5l) % 4l)])
        let mb = f32_to_f16(lut[int((sb / 3l + 1l) % 4l)])
        let ko = sb * 20l
        let go = sb * 144l
        ks[ko] = uint8(db & 255u)
        ks[ko + 1l] = uint8(db >> 8u)
        ks[ko + 2l] = uint8(mb & 255u)
        ks[ko + 3l] = uint8(mb >> 8u)
        gg[go] = ks[ko]
        gg[go + 1l] = ks[ko + 1l]
        gg[go + 2l] = ks[ko + 2l]
        gg[go + 3l] = ks[ko + 3l]
        for (j in range64(4l)) {   // ggml scales[12]: [0,7] values keep the packed high bits zero
            let scl = int((sb + j * 5l + seed) % 8l)
            let mnl = int((sb * 3l + j * 7l + 1l) % 8l)
            let sch = int((sb + (j + 4l) * 5l + seed) % 8l)
            let mnh = int((sb * 3l + (j + 4l) * 7l + 1l) % 8l)
            ks[ko + 4l + j] = uint8(scl)
            ks[ko + 8l + j] = uint8(mnl)
            ks[ko + 12l + j] = uint8(sch | (mnh << 4))
        }
        for (j in range64(4l)) {
            gg[go + 4l + j] = ks[ko + 4l + j]
            gg[go + 8l + j] = ks[ko + 8l + j]
            gg[go + 12l + j] = ks[ko + 12l + j]
        }
        ks[ko + 16l] = uint8(0)   // the transcode zeroes the 4 pad bytes
        ks[ko + 17l] = uint8(0)
        ks[ko + 18l] = uint8(0)
        ks[ko + 19l] = uint8(0)
        let qo = sb * 128l
        for (b in range64(128l)) {
            let lo = int((b * 7l + sb * 13l + seed) % 16l)
            let hi = int((b * 11l + sb * 5l + 3l) % 16l)
            kq[qo + b] = uint8(lo | (hi << 4))
            gg[go + 16l + b] = kq[qo + b]
        }
    }
}

// 6-bit sub-scale/min j of a 12-byte packed Q4_K scale block (ggml get_scale_min_k4, array form)
def private k4_ref_sc_mn(ks : array<uint8>; base, j : int64) : int2 {
    if (j < 4l) {
        return int2(int(ks[base + j]) & 63, int(ks[base + j + 4l]) & 63)
    }
    let sc = (int(ks[base + j + 4l]) & 15) | ((int(ks[base + j - 4l]) >> 6) << 4)
    let mn = (int(ks[base + j + 4l]) >> 4) | ((int(ks[base + j]) >> 6) << 4)
    return int2(sc, mn)
}

// CPU reference for the k4 arms, sampled like cpu_ref_sample: per-element w = d*sc*q - dmin*mn
// against the f32 panel — algebraically the kernels' grouped folds, identical bits since every
// intermediate is exact under the fill
def private cpu_ref_k4(kq : array<uint8>; ks : array<uint8>; xf : array<float>; n, nb_streams, refd : int64) : array<float> {
    var y : array<float>
    y |> resize(nb_streams * refd)
    let nsb = n / 256l
    for (b in range64(nb_streams)) {
        for (j in range64(refd)) {
            var acc = 0.0
            for (sb in range64(nsb)) {
                let ko = (j * nsb + sb) * 20l
                let qo = (j * nsb + sb) * 128l
                let dd = f16_to_f32(uint(ks[ko]) | (uint(ks[ko + 1l]) << 8u))
                let dm = f16_to_f32(uint(ks[ko + 2l]) | (uint(ks[ko + 3l]) << 8u))
                for (js in range64(8l)) {
                    let s = k4_ref_sc_mn(ks, ko + 4l, js)
                    let ds = dd * float(s.x)
                    let dmn = dm * float(s.y)
                    for (l in range64(32l)) {
                        let qv = int(kq[qo + (js / 2l) * 32l + l])
                        let q = (js & 1l) == 0l ? (qv & 15) : (qv >> 4)
                        acc += (ds * float(q) - dmn) * xf[b * n + sb * 256l + js * 32l + l]
                    }
                }
            }
            y[b * refd + j] = acc
        }
    }
    return <- y
}

def private upload_array(dev; arr; bytes : int64) {
    var buf = metal_new_buffer(dev, uint64(bytes))
    unsafe {
        memcpy(metal_buffer_contents(buf), addr < void ? > (arr[0]), int(bytes))
    }
    return buf
}

def private upload_u32(dev; v : uint) {
    var buf = metal_new_buffer(dev, 4ul)
    var p = unsafe(reinterpret<uint?>(metal_buffer_contents(buf)))
    unsafe(p[0]) = v
    return buf
}

def private download_floats(buf; var dst : array<float>; count : int64) {
    dst |> resize(count)
    unsafe {
        memcpy(addr < void ? > (dst[0]), metal_buffer_contents(buf), int(count * 4l))
    }
}

def private compile_lab_pso(dev; v : LabVariant) {
    var err : string
    var pso = pipeline_from_source(dev, v.src, v.entry, true, err)
    if (pso == null) {
        to_log(LOG_ERROR, "{v.name}: kernel compile failed — {err}\n")
    }
    return pso
}

// one timed encoder: set state once, dispatch `reps` back-to-back (hazard tracking on the shared
// y buffer serializes them); returns per-dispatch GPU milliseconds, or -1 on failure
def private time_reps(queue; pso; bufs; groups : uint3; reps : int; tgmem : uint64; tgsz : int = 128) : double {
    var err : string
    var gpu_ms = 0.0lf
    let ok = with_compute_encoder_timed(queue, err, gpu_ms) $(enc) {
        metal_set_pipeline(enc, pso)
        for (bi in range(length(bufs))) {
            metal_set_buffer(enc, bufs[bi], 0ul, bi)
        }
        if (tgmem > 0ul) {
            metal_set_threadgroup_memory_length(enc, tgmem, 0)
        }
        for (_i in range(reps)) {
            metal_dispatch_threadgroups(enc, groups, uint3(uint(tgsz), 1u, 1u))
        }
    }
    if (!ok) {
        to_log(LOG_ERROR, "dispatch failed: {err}\n")
        return -1.0lf
    }
    return gpu_ms / double(reps)
}

// split-K timing: each rep = the partial GEMM then the reduce, same serial encoder
def private time_reps_sk(queue; pso_g; bufs_g; groups_g : uint3; tgmem : uint64;
                         pso_r; bufs_r; groups_r : uint3; reps : int) : double {
    var err : string
    var gpu_ms = 0.0lf
    let ok = with_compute_encoder_timed(queue, err, gpu_ms) $(enc) {
        for (_i in range(reps)) {
            metal_set_pipeline(enc, pso_g)
            for (bi in range(length(bufs_g))) {
                metal_set_buffer(enc, bufs_g[bi], 0ul, bi)
            }
            if (tgmem > 0ul) {
                metal_set_threadgroup_memory_length(enc, tgmem, 0)
            }
            metal_dispatch_threadgroups(enc, groups_g, uint3(128u, 1u, 1u))
            metal_set_pipeline(enc, pso_r)
            for (bi in range(length(bufs_r))) {
                metal_set_buffer(enc, bufs_r[bi], 0ul, bi)
            }
            metal_dispatch_threadgroups(enc, groups_r, uint3(128u, 1u, 1u))
        }
    }
    if (!ok) {
        to_log(LOG_ERROR, "sk dispatch failed: {err}\n")
        return -1.0lf
    }
    return gpu_ms / double(reps)
}

def private env_or(name : string; dflt : int) : int {
    return has_env_variable(name) ? to_int(get_env_variable(name)) : dflt
}

def private want(filter_env, name : string) : bool {
    return !has_env_variable(filter_env) || find(get_env_variable(filter_env), name) >= 0
}

def private sk_idx(sv : int) : int => sv == 2 ? 0 : (sv == 4 ? 1 : 2)

// per-variant buffer list in binding order — generic (untyped params) so non-Apple never compiles it
def private make_bufs(v : LabVariant; var wbuf; var sbuf; var xfbuf; var xtbuf; var xqbuf; var xsbuf; var ybuf; var ndimbuf; var ddimbuf; var bbuf; var partbuf; var skbuf; var kqbuf; var ksbuf; var kgbuf; var kcbuf) {
    if (v.kq_ggml) {   // interleaved stream bound twice: f16 view + uint view
        return <- [kgbuf, kgbuf, xfbuf, ybuf, ndimbuf, ddimbuf]
    }
    if (v.kq16) {      // compact scale blocks bound twice (f16 + uint4 views) + the quant plane
        return <- [kcbuf, kcbuf, kqbuf, xfbuf, ybuf, ndimbuf, ddimbuf]
    }
    if (v.kq != 0) {   // scale plane bound twice (f16 + uint views) + the quant plane
        return <- [ksbuf, ksbuf, kqbuf, xfbuf, ybuf, ndimbuf, ddimbuf]
    }
    if (v.gemm_sk) {
        return <- [wbuf, sbuf, xfbuf, partbuf, ndimbuf, ddimbuf, skbuf]
    }
    if (v.q8x || v.gemm) {
        return <- [wbuf, sbuf, xqbuf, xsbuf, ybuf, ndimbuf, ddimbuf]
    }
    if (v.dyn_b) {
        return <- [wbuf, sbuf, xfbuf, ybuf, ndimbuf, ddimbuf, bbuf]
    }
    if (v.xt) {
        return <- [wbuf, sbuf, xtbuf, ybuf, ndimbuf, ddimbuf]
    }
    return <- [wbuf, sbuf, xfbuf, ybuf, ndimbuf, ddimbuf]
}

def private count_mismatch_sampled(got : array<float>; ref_v : array<float>; d, nb_streams, refd : int64;
                                   var firstBad : int64&) : int64 {
    var bad = 0l
    firstBad = -1l
    for (b in range64(nb_streams)) {
        for (j in range64(refd)) {
            if (got[b * d + j] != ref_v[b * refd + j]) {
                if (firstBad < 0l) {
                    firstBad = b * d + j
                }
                bad++
            }
        }
    }
    return bad
}

def private run_lab(b : B?; variants) {
    let rounds = env_or("DASMETAL_LAB_ROUNDS", 3)
    let refd = 128l
    let bmax = 16l
    var dev = metal_create_system_default_device()
    if (dev == null) {
        to_log(LOG_WARNING, "skipped: no Metal device\n")
        return
    }
    to_log(LOG_INFO, "device={metal_device_name(dev)} rounds={rounds}\n")
    var queue = metal_new_command_queue(dev)
    var psos <- [for (v in variants); compile_lab_pso(dev, v)]
    var err_red : string
    var pso_red = pipeline_from_source(dev, lab_sk_reduce_msl, lab_sk_reduce_msl_entry, lab_sk_reduce_msl_fastmath, err_red)
    for (v, pso in variants, psos) {
        b |> success(pso != null, "pipeline compiles: {v.name}")
        if (pso != null && want("DASMETAL_LAB_VARIANTS", v.name)) {
            to_log(LOG_INFO, "{v.name}: max_threads={metal_pipeline_max_total_threads(pso)} simd_width={metal_pipeline_thread_execution_width(pso)}\n")
        }
    }
    // pre-built B uniforms for the dynamic-B variant
    var bbufs <- [for (bi in range(5)); upload_u32(dev, uint(1 << bi))]
    // the llama-3B decode GEMV geometries (n = reduction dim, d = output rows)
    var shapes <- [("q3b", 3072l, 3072l), ("kv3b", 3072l, 1024l), ("qkv3b", 3072l, 5120l), ("w13_3b", 3072l, 8192l),
                   ("w2_3b", 8192l, 3072l), ("cls3b", 3072l, 128256l)]
    for (sh in shapes) {
        if (!want("DASMETAL_LAB_SHAPES", sh._0)) {
            continue
        }
        let n = sh._1
        let d = sh._2
        let nb = n / 32l
        let xrows = 32l                      // the GEMM row is M-padded to 32; streams 16-31 are pad
        let t_shape0 = ref_time_ticks()
        var wq : array<int8>
        var ws : array<float>
        var xq : array<int8>
        var xs : array<float>
        fill_exact_q8(wq, ws, d, n, 7l)
        fill_exact_q8(xq, xs, xrows, n, 11l)
        var yref <- cpu_ref_sample(wq, ws, xq, xs, n, bmax, refd)
        // the f32 activation panel — exact dequant of xq/xs (int x pow2); full 32 M-pad rows
        // (the f32-X GEMM reads all of them — pad outputs land in unchecked rows)
        var xf : array<float>
        xf |> resize(int(xrows * n))
        for (i in range64(xrows * n)) {
            xf[i] = float(xq[i]) * xs[(i / n) * nb + (i % n) / 32l]
        }
        // the stream-minor transpose (stride 16) for the xt variants
        var xft : array<float>
        xft |> resize(int(n * bmax))
        for (i in range64(bmax * n)) {
            xft[(i % n) * bmax + i / n] = xf[i]
        }
        // the k4 arms' weight streams (built only when a kq arm survives the filter — the cls
        // shape's three extra layouts are ~450MB of host fill otherwise wasted)
        var anykq = false
        for (v in variants) {
            anykq ||= (v.kq != 0 || v.kq_ggml || v.kq16) && want("DASMETAL_LAB_VARIANTS", v.name)
        }
        var k4q_h : array<uint8>
        var k4s_h : array<uint8>
        var k4g_h : array<uint8>
        var k4c_h : array<uint8>
        var yref_k4 : array<float>
        if (anykq) {
            fill_exact_k4(k4q_h, k4s_h, k4g_h, d, n, 7l)
            yref_k4 <- cpu_ref_k4(k4q_h, k4s_h, xf, n, bmax, refd)
            // the 16B compact scale plane = the 20B blocks with the pads stripped
            let nsb_all = (d * n) / 256l
            k4c_h |> resize(nsb_all * 16l)
            for (sb in range64(nsb_all)) {
                for (i in range64(16l)) {
                    k4c_h[sb * 16l + i] = k4s_h[sb * 20l + i]
                }
            }
        }
        var wbuf = upload_array(dev, wq, d * n)
        var sbuf = upload_array(dev, ws, d * nb * 4l)
        var kqbuf = anykq ? upload_array(dev, k4q_h, (d * n) / 2l) : null
        var ksbuf = anykq ? upload_array(dev, k4s_h, (d * n / 256l) * 20l) : null
        var kgbuf = anykq ? upload_array(dev, k4g_h, (d * n / 256l) * 144l) : null
        var kcbuf = anykq ? upload_array(dev, k4c_h, (d * n / 256l) * 16l) : null
        var xfbuf = upload_array(dev, xf, xrows * n * 4l)
        var xtbuf = upload_array(dev, xft, n * bmax * 4l)
        var xqbuf = upload_array(dev, xq, xrows * n)
        var xsbuf = upload_array(dev, xs, xrows * nb * 4l)
        var ybuf = metal_new_buffer(dev, uint64(xrows * d * 4l))
        var partbuf = metal_new_buffer(dev, uint64(8l * 32l * d * 4l))   // up to 8 split-K planes
        var totalbuf = upload_u32(dev, uint(32l * d))
        var skbufs <- [for (sv in [2, 4, 8]); upload_u32(dev, uint(sv))]
        var ndimbuf = upload_u32(dev, uint(n))
        var ddimbuf = upload_u32(dev, uint(d))
        let setup_ms = double(ref_time_ticks() - t_shape0) / 1000000.0lf
        to_log(LOG_INFO, "=== {sh._0} n={n} d={d} === (setup {setup_ms} ms)\n")
        let groups = uint3(uint((d + 3l) / 4l), 1u, 1u)
        let groups_gemm = uint3(uint(d / 32l), 1u, 1u)   // (32/32 token tiles) x (d/32 output tiles)
        let groups_gemm64 = uint3(uint(d / 64l), 1u, 1u)
        // flattened (variant, B) runs — interleaved timing across ALL runs neutralizes drift
        var run_v : array<int>
        var run_b : array<int>
        run_v |> reserve(length(variants) * 5)
        run_b |> reserve(length(variants) * 5)
        for (v, idx in variants, count()) {
            if (psos[idx] == null || !want("DASMETAL_LAB_VARIANTS", v.name)) {
                continue
            }
            for (bn in v.b_list) {
                run_v |> push(idx)
                run_b |> push(bn)
            }
        }
        var alive : array<bool>
        var est_ms : array<double>
        var best : array<double>
        alive |> resize(length(run_v))
        est_ms |> resize(length(run_v))
        best |> resize(length(run_v))
        var ygot : array<float>
        for (ri in range(length(run_v))) {
            alive[ri] = false
            best[ri] = 1.0e30lf
            let vi = run_v[ri]
            let nbatch = run_b[ri]
            var bufs <- make_bufs(variants[vi], wbuf, sbuf, xfbuf, xtbuf, xqbuf, xsbuf, ybuf, ndimbuf, ddimbuf, bbufs[find_bbuf(nbatch)], partbuf, skbufs[sk_idx(variants[vi].sk)], kqbuf, ksbuf, kgbuf, kcbuf)
            var ms = 0.0lf   // warmup + correctness dispatch
            if (variants[vi].gemm_sk) {
                var rbufs <- [partbuf, ybuf, totalbuf, skbufs[sk_idx(variants[vi].sk)]]
                ms = time_reps_sk(queue, psos[vi], bufs, uint3(uint(d / 32l), uint(variants[vi].sk), 1u), variants[vi].tgmem,
                    pso_red, rbufs, uint3(uint((32l * d + 127l) / 128l), 1u, 1u), 1)
                rbufs |> clear()
                unsafe {
                    delete rbufs
                }
            } else {
                ms = time_reps(queue, psos[vi], bufs, variants[vi].gemm64_fx ? groups_gemm64 : ((variants[vi].gemm || variants[vi].gemm_fx) ? groups_gemm : groups), 1, variants[vi].tgmem, variants[vi].tgsz)
            }
            bufs |> clear()   // borrowed handles — delete of array<T?> would free the pointees
            unsafe {
                delete bufs
            }
            if (ms < 0.0lf) {
                b |> success(false, "{variants[vi].name}/B{nbatch} {sh._0}: dispatch")
                continue
            }
            est_ms[ri] = ms
            download_floats(ybuf, ygot, int64(nbatch) * d)
            var firstBad = -1l
            let iskq = variants[vi].kq != 0 || variants[vi].kq_ggml || variants[vi].kq16
            let bad = (iskq ? count_mismatch_sampled(ygot, yref_k4, d, int64(nbatch), refd, firstBad)
                : count_mismatch_sampled(ygot, yref, d, int64(nbatch), refd, firstBad))
            b |> success(bad == 0l, "{variants[vi].name}/B{nbatch} {sh._0}: bit-exact vs CPU reference")
            if (bad != 0l) {
                to_log(LOG_ERROR, "{variants[vi].name}/B{nbatch} {sh._0}: {bad} mismatches, first at {firstBad}: got {ygot[firstBad]}\n")
            } else {
                alive[ri] = true
            }
        }
        // interleaved best-of-N timing rounds
        for (_round in range(rounds)) {
            for (ri in range(length(run_v))) {
                if (!alive[ri]) {
                    continue
                }
                let vi = run_v[ri]
                var bufs <- make_bufs(variants[vi], wbuf, sbuf, xfbuf, xtbuf, xqbuf, xsbuf, ybuf, ndimbuf, ddimbuf, bbufs[find_bbuf(run_b[ri])], partbuf, skbufs[sk_idx(variants[vi].sk)], kqbuf, ksbuf, kgbuf, kcbuf)
                let reps = clamp(int(25.0lf / max(est_ms[ri], 0.01lf)) + 1, 1, 128)
                var ms = 0.0lf
                if (variants[vi].gemm_sk) {
                    var rbufs <- [partbuf, ybuf, totalbuf, skbufs[sk_idx(variants[vi].sk)]]
                    ms = time_reps_sk(queue, psos[vi], bufs, uint3(uint(d / 32l), uint(variants[vi].sk), 1u), variants[vi].tgmem,
                        pso_red, rbufs, uint3(uint((32l * d + 127l) / 128l), 1u, 1u), reps)
                    rbufs |> clear()
                    unsafe {
                        delete rbufs
                    }
                } else {
                    ms = time_reps(queue, psos[vi], bufs, variants[vi].gemm64_fx ? groups_gemm64 : ((variants[vi].gemm || variants[vi].gemm_fx) ? groups_gemm : groups), reps, variants[vi].tgmem, variants[vi].tgsz)
                }
                if (ms > 0.0lf && ms < best[ri]) {
                    best[ri] = ms
                }
                bufs |> clear()
                unsafe {
                    delete bufs
                }
            }
        }
        // weight traffic: q8 planar = n quant + n/8 scale bytes per output row; k4 = 148B
        // (planes, 128+20) or 144B (ggml interleaved) per 256 elems
        for (ri in range(length(run_v))) {
            if (!alive[ri]) {
                continue
            }
            let vi = run_v[ri]
            let wbytes = double(variants[vi].kq_ggml || variants[vi].kq16 ? d * (n / 256l) * 144l
                : (variants[vi].kq != 0 ? d * (n / 256l) * 148l : d * (n + n / 8l)))
            let nbatch = run_b[ri]
            let gbs = wbytes / (best[ri] * 1.0e6lf)
            let gmacs = double(n * d * int64(nbatch)) / (best[ri] * 1.0e6lf)
            to_log(LOG_INFO, "{variants[vi].name}/B{nbatch}  {sh._0} {best[ri]} ms  {gbs} wGB/s  {gmacs} GMAC/s\n")
        }
        metal_release(wbuf)
        metal_release(sbuf)
        if (anykq) {
            metal_release(kqbuf)
            metal_release(ksbuf)
            metal_release(kgbuf)
            metal_release(kcbuf)
        }
        metal_release(partbuf)
        metal_release(totalbuf)
        for (skb in skbufs) {
            metal_release(skb)
        }
        skbufs |> clear()   // released above — delete of array<T?> would double-free the pointees
        unsafe {
            delete skbufs
        }
        metal_release(xfbuf)
        metal_release(xtbuf)
        metal_release(xqbuf)
        metal_release(xsbuf)
        metal_release(ybuf)
        metal_release(ndimbuf)
        metal_release(ddimbuf)
        delete run_v
        delete run_b
        delete alive
        delete est_ms
        delete best
        delete ygot
        delete wq
        delete ws
        delete xq
        delete xs
        delete xf
        delete xft
        delete yref
        delete k4q_h
        delete k4s_h
        delete k4g_h
        delete k4c_h
        delete yref_k4
    }
    delete shapes
    for (bbuf in bbufs) {
        metal_release(bbuf)
    }
    bbufs |> clear()
    unsafe {
        delete bbufs
    }
    for (pso in psos) {
        if (pso != null) {
            metal_release(pso)
        }
    }
    if (pso_red != null) {
        metal_release(pso_red)
    }
    psos |> clear()
    unsafe {
        delete psos
    }
    metal_release(queue)
    metal_release(dev)
    assert_no_metal_leaks()
}

def private find_bbuf(nbatch : int) : int {
    var bi = 0
    var v = nbatch
    while (v > 1) {
        v /= 2
        bi++
    }
    return bi
}

[benchmark, unused_argument(b)]   // b is consumed only inside the Apple static_if half
def bench_metal_gemv_kernels(b : B?) {
    if (!jit_enabled()) {
        to_log(LOG_WARNING, "skipped: interpreted (dasLLAMA benchmarks are JIT-only)\n")
        return
    }
    static_if (typeinfo builtin_module_exists(das_metal)) {
        var variants <- [
            LabVariant(name = "v0_gemv", src = metal_q8_gemv_msl, entry = metal_q8_gemv_msl_entry,
                       b_list <- [1]),
            LabVariant(name = "gemv_b2", src = lab_gemv_b2_msl, entry = lab_gemv_b2_msl_entry,
                       b_list <- [2]),
            LabVariant(name = "gemv_b4", src = lab_gemv_b4_msl, entry = lab_gemv_b4_msl_entry,
                       b_list <- [4]),
            LabVariant(name = "gemv_b8", src = lab_gemv_b8_msl, entry = lab_gemv_b8_msl_entry,
                       b_list <- [8]),
            LabVariant(name = "gemv_b16", src = lab_gemv_b16_msl, entry = lab_gemv_b16_msl_entry,
                       b_list <- [16]),
            LabVariant(name = "gemv_b2c", src = lab_gemv_b2c_msl, entry = lab_gemv_b2c_msl_entry,
                       b_list <- [2]),
            LabVariant(name = "gemv_b4c", src = lab_gemv_b4c_msl, entry = lab_gemv_b4c_msl_entry,
                       b_list <- [4]),
            LabVariant(name = "gemv_b2c8", src = lab_gemv_b2c8_msl, entry = lab_gemv_b2c8_msl_entry,
                       b_list <- [2]),
            LabVariant(name = "gemv_b4c8", src = lab_gemv_b4c8_msl, entry = lab_gemv_b4c8_msl_entry,
                       b_list <- [4]),
            LabVariant(name = "gemv_b4x", src = lab_gemv_b4x_msl, entry = lab_gemv_b4x_msl_entry,
                       b_list <- [4], tgmem = lab_gemv_b4x_msl_tgmem),
            LabVariant(name = "gemv_b2x", src = lab_gemv_b2x_msl, entry = lab_gemv_b2x_msl_entry,
                       b_list <- [2], tgmem = lab_gemv_b2x_msl_tgmem),
            LabVariant(name = "gemv_b8x", src = lab_gemv_b8x_msl, entry = lab_gemv_b8x_msl_entry,
                       b_list <- [8], tgmem = lab_gemv_b8x_msl_tgmem),
            LabVariant(name = "gemv_b16x", src = lab_gemv_b16x_msl, entry = lab_gemv_b16x_msl_entry,
                       b_list <- [16], tgmem = lab_gemv_b16x_msl_tgmem),
            LabVariant(name = "gemv_bdyn", src = lab_gemv_bdyn_msl, entry = lab_gemv_bdyn_msl_entry,
                       b_list <- [1, 2, 4, 8, 16], dyn_b = true),
            LabVariant(name = "gemv_q8x", src = lab_gemv_q8x_msl, entry = lab_gemv_q8x_msl_entry,
                       b_list <- [1], q8x = true),
            LabVariant(name = "gemv_b1", src = lab_gemv_b1_msl, entry = lab_gemv_b1_msl_entry,
                       b_list <- [1]),
            LabVariant(name = "gemv_b4t", src = lab_gemv_b4t_msl, entry = lab_gemv_b4t_msl_entry,
                       b_list <- [4], xt = true),
            LabVariant(name = "gemv_b8t", src = lab_gemv_b8t_msl, entry = lab_gemv_b8t_msl_entry,
                       b_list <- [8], xt = true),
            LabVariant(name = "gemv_b16t", src = lab_gemv_b16t_msl, entry = lab_gemv_b16t_msl_entry,
                       b_list <- [16], xt = true),
            LabVariant(name = "gemm32", src = metal_q8_gemm_msl, entry = metal_q8_gemm_msl_entry,
                       b_list <- [1, 2, 4, 8, 16], gemm = true, tgmem = metal_q8_gemm_msl_tgmem),
            LabVariant(name = "gemm_v2", src = lab_gemm_v2_msl, entry = lab_gemm_v2_msl_entry,
                       b_list <- [8, 16], gemm_fx = true, tgmem = lab_gemm_v2_msl_tgmem),
            LabVariant(name = "gemm64_v2", src = lab_gemm64_v2_msl, entry = lab_gemm64_v2_msl_entry,
                       b_list <- [8, 16], gemm64_fx = true, tgmem = lab_gemm64_v2_msl_tgmem),
            LabVariant(name = "gemm_sk2", src = lab_gemm_sk_msl, entry = lab_gemm_sk_msl_entry,
                       b_list <- [8], gemm_sk = true, sk = 2, tgmem = lab_gemm_sk_msl_tgmem),
            LabVariant(name = "gemm_sk4", src = lab_gemm_sk_msl, entry = lab_gemm_sk_msl_entry,
                       b_list <- [8], gemm_sk = true, sk = 4, tgmem = lab_gemm_sk_msl_tgmem),
            LabVariant(name = "gemm_sk8", src = lab_gemm_sk_msl, entry = lab_gemm_sk_msl_entry,
                       b_list <- [8], gemm_sk = true, sk = 8, tgmem = lab_gemm_sk_msl_tgmem),
            LabVariant(name = "k4_lcpp", src = lab_k4_lcpp_msl, entry = lab_k4_lcpp_msl_entry,
                       b_list <- [1], kq_ggml = true, tgsz = 64),
            LabVariant(name = "k4_b1", src = lab_k4_b1_msl, entry = lab_k4_b1_msl_entry,
                       b_list <- [1], kq = 4, tgsz = 64),
            LabVariant(name = "k4_b1c", src = lab_k4_b1c_msl, entry = lab_k4_b1c_msl_entry,
                       b_list <- [1], kq16 = true, tgsz = 64)
        ]
        run_lab(b, variants)
        delete variants
    }
}
