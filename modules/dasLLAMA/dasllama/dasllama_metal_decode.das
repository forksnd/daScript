options gen2

module dasllama_metal_decode shared public

require math
require strings
require daslib/fio
require metal/msl_shader
require metal/das_metal_boost
require dasllama/dasllama_common
require dasllama/dasllama_math
require dasllama/dasllama_metal_prefill

// The dasMetal GPU-resident DECODE path (the emission arc's P2 driver + P3 chase shape) — the
// mirror image of dasllama_metal_prefill: one command buffer per token step runs the whole llama
// layer stack on the GPU. Decode is GEMV-bandwidth-bound (one token streams every weight once)
// AND launch-gap-bound (~4.3us per serial dispatch), so the layer runs 9 fused dispatches:
//   fused QKV / W1|W3 GEMVs   the P1-lab-proven single-simdgroup-per-row Q8 GEMV
//                             (metal_q8_gemv_msl, f32 activations) over CONCATENATED weight
//                             regions (upload_region_cat3) into one output buffer each —
//                             downstream dispatches bind at byte offsets
//   metal_sq_attn16/32        single-query attention over the KV mirror, one threadgroup per
//                             (stream, head), simdgroup-cooperative coalesced K scoring; depth
//                             > 128 rows takes metal_sq_attn_part16/32 + metal_sq_attn_comb
//                             (chunked split-K: occupancy scales with depth)
//   metal_rope_store16/32     fused rope + KV-row store: q ropes in place, k ropes straight
//                             into the mirror with cvt_f32_to_f16's exact ±65504 clamp (CPU
//                             fallback steps read byte-identical KV), v copies alongside
//   metal_add_rms             fused residual add + rmsnorm (the trailing one feeds the NEXT
//                             layer's attention norm; layer 0 gets a prologue rms)
// rmsnorm / add / swiglu / the classifier GEMV reuse the prefill module's MSL sources verbatim
// (own PSOs, zero copy drift).
//
// GPU-resident KV = a lazy LOGICAL mirror per session, keyed by Session.uid (address keying is
// unsound: scheduler churn ABAs same-geometry sessions, paged sessions have no flat base,
// kv_pool_grow reallocs blobs in place, Stream erase memmoves Sessions). The CPU session cache
// stays the source of truth: uploads gather logical rows [watermark, pos) through kv_runs,
// pos < watermark = rewind detected (the ASR release_kv_pages + n_past = 0 pattern) → re-upload
// from 0, and after each step the GPU's row bytes memcpy VERBATIM back into the CPU cache (the
// CPU never re-encodes — mirror dtype == session dtype by the f16/f32 gate). Flat and paged
// sessions are the same driver code: the GPU kernels only ever see a dense logical slab.

// ===== the new kernels =====

// One threadgroup per (stream, head), head_size threads — lane owns one V column. Pass 1 scores
// the context (<= 128 rows — deeper contexts take the chunked twins below) simdgroup-
// cooperatively: each simdgroup owns row j, its 32 lanes cover a contiguous 64B run of the K row
// and reduce the dot via shuffles, so every load instruction coalesces (the per-lane-per-row form
// gathered 32 cache lines 2KB apart per step — measured 27us/layer at depth 24). Pass 2 softmax
// (simd butterfly + cross-simdgroup partials), pass 3 per-lane V-column accumulate. GQA: head h
// reads kv head h/kv_mul. The oracle is the CPU decode path itself (tolerance — exp() and
// accumulation order differ legitimately).
class MetalSqAttnF16 {
    @ssbo @binding = 0 kc : array<float16>  // this layer's K mirror slab (bound at the layer's byte offset)
    @ssbo @binding = 1 vc : array<float16>
    @ssbo @binding = 2 q : array<float>     // [B x qd] roped queries
    @ssbo @binding = 3 yo : array<float>    // [B x qd] attention out
    @uniform @binding = 4 cnt : uint        // context depth (pos + 1)
    @uniform @binding = 5 kv_dim : uint
    @uniform @binding = 6 nheads : uint
    @uniform @binding = 7 kv_mul : uint
    @uniform @binding = 8 hs : uint         // head_size == threads per group
    @uniform @binding = 9 scale : float
    @workgroup scores : float[128]  // >= ATTN_CHUNK_ROWS — the single-tg depth ceiling
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn16_msl")]
    def metal_sq_attn16 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let bb = wg / nheads
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let khoff = (h / kv_mul) * hs
        let qoff = bb * (nheads * hs) + h * hs
        // pass 1: simdgroup-cooperative score fill — simdgroup sg dots q[head] against K row j;
        // the lane's q elements are loop-invariant, hoisted to registers (device loads can't be
        // hoisted by the compiler — buffers may alias)
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u              // elements per lane in the row run
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        var j = sg
        while (j < cnt) {
            let kb = j * kv_dim + khoff + sl * epl
            var s = q0 * float(kc[kb])
            if (epl > 1u) {
                s += q1 * float(kc[kb + 1u])
            }
            if (epl > 2u) {
                s += q2 * float(kc[kb + 2u])
            }
            if (epl > 3u) {
                s += q3 * float(kc[kb + 3u])
            }
            s = simd_sum(s)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        // pass 2a: max — strided local, simd butterfly, cross-simdgroup partials
        var m = -3.0e38
        j = lid
        while (j < cnt) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()                           // red[] reused for the sum below
        // pass 2b: exp in place + sum
        var l = 0.0
        j = lid
        while (j < cnt) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        // pass 3: lane owns V column `lid` — 4 rows in flight so device-load latency overlaps
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= cnt) {
            acc0 += scores[j] * float(vc[j * kv_dim + khoff + lid])
            acc1 += scores[j + 1u] * float(vc[(j + 1u) * kv_dim + khoff + lid])
            acc2 += scores[j + 2u] * float(vc[(j + 2u) * kv_dim + khoff + lid])
            acc3 += scores[j + 3u] * float(vc[(j + 3u) * kv_dim + khoff + lid])
            j += 4u
        }
        while (j < cnt) {
            acc0 += scores[j] * float(vc[j * kv_dim + khoff + lid])
            j++
        }
        yo[qoff + lid] = (acc0 + acc1 + acc2 + acc3) / tot
    }
}

// f32-mirror twin (sessions created with f32 KV) — same math, f32 loads.
class MetalSqAttnF32 {
    @ssbo @binding = 0 kc : array<float>
    @ssbo @binding = 1 vc : array<float>
    @ssbo @binding = 2 q : array<float>
    @ssbo @binding = 3 yo : array<float>
    @uniform @binding = 4 cnt : uint
    @uniform @binding = 5 kv_dim : uint
    @uniform @binding = 6 nheads : uint
    @uniform @binding = 7 kv_mul : uint
    @uniform @binding = 8 hs : uint
    @uniform @binding = 9 scale : float
    @workgroup scores : float[128]
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn32_msl")]
    def metal_sq_attn32 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let bb = wg / nheads
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let khoff = (h / kv_mul) * hs
        let qoff = bb * (nheads * hs) + h * hs
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        var j = sg
        while (j < cnt) {
            let kb = j * kv_dim + khoff + sl * epl
            var s = q0 * kc[kb]
            if (epl > 1u) {
                s += q1 * kc[kb + 1u]
            }
            if (epl > 2u) {
                s += q2 * kc[kb + 2u]
            }
            if (epl > 3u) {
                s += q3 * kc[kb + 3u]
            }
            s = simd_sum(s)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        var m = -3.0e38
        j = lid
        while (j < cnt) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()
        var l = 0.0
        j = lid
        while (j < cnt) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= cnt) {
            acc0 += scores[j] * vc[j * kv_dim + khoff + lid]
            acc1 += scores[j + 1u] * vc[(j + 1u) * kv_dim + khoff + lid]
            acc2 += scores[j + 2u] * vc[(j + 2u) * kv_dim + khoff + lid]
            acc3 += scores[j + 3u] * vc[(j + 3u) * kv_dim + khoff + lid]
            j += 4u
        }
        while (j < cnt) {
            acc0 += scores[j] * vc[j * kv_dim + khoff + lid]
            j++
        }
        yo[qoff + lid] = (acc0 + acc1 + acc2 + acc3) / tot
    }
}

// q8_0-mirror twin — K/V rows are ggml block_q8_0 (34B per 32 elements, {f16 d; int8 qs[32]});
// the mirror slab binds TWICE (half view: scale of block b at halfword 17*b; byte view: quants
// at 34*b + 2 — the MetalQ8MulMm dual-view pattern) and dequant rides the stream. Lane sl's epl
// score elements share one block, so the block scale factors out of the lane dot; the V column
// walk reads quant + scale off the same 34B block (~one cacheline per row, half the f16 bytes).
// Lab: benchmarks/attn/bench_metal_sq_attn.das — 1.1-1.7x over the f16 form at d512-2048.
class MetalSqAttnQ8 {
    @ssbo @binding = 0 ksh : array<float16> // K mirror slab, HALF-SCALE view (bound at the layer's byte offset)
    @ssbo @binding = 1 kqb : array<int8>    // the SAME K slab, BYTE view
    @ssbo @binding = 2 vsh : array<float16>
    @ssbo @binding = 3 vqb : array<int8>
    @ssbo @binding = 4 q : array<float>     // [B x qd] roped queries
    @ssbo @binding = 5 yo : array<float>    // [B x qd] attention out
    @uniform @binding = 6 cnt : uint        // context depth (pos + 1)
    @uniform @binding = 7 nblk : uint       // blocks per row = kv_dim / 32
    @uniform @binding = 8 nheads : uint
    @uniform @binding = 9 kv_mul : uint
    @uniform @binding = 10 hs : uint        // head_size == threads per group
    @uniform @binding = 11 scale : float
    @workgroup scores : float[128]  // >= ATTN_CHUNK_ROWS — the single-tg depth ceiling
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_q8_msl")]
    def metal_sq_attn_q8 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let bb = wg / nheads
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let khblk = (h / kv_mul) * (hs / 32u)   // kv head offset in BLOCKS
        let qoff = bb * (nheads * hs) + h * hs
        // pass 1: simdgroup-cooperative score fill — lane sl's epl elements sit in ONE block
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        let hb = (sl * epl) / 32u               // lane's block within the head
        let ib = (sl * epl) % 32u               // lane's quant offset within the block
        var j = sg
        while (j < cnt) {
            let blk = j * nblk + khblk + hb
            let d = float(ksh[blk * 17u])
            let kb = blk * 34u + 2u + ib
            var s = q0 * float(kqb[kb])
            if (epl > 1u) {
                s += q1 * float(kqb[kb + 1u])
            }
            if (epl > 2u) {
                s += q2 * float(kqb[kb + 2u])
            }
            if (epl > 3u) {
                s += q3 * float(kqb[kb + 3u])
            }
            s = simd_sum(s * d)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        // pass 2a: max — strided local, simd butterfly, cross-simdgroup partials
        var m = -3.0e38
        j = lid
        while (j < cnt) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()                           // red[] reused for the sum below
        // pass 2b: exp in place + sum
        var l = 0.0
        j = lid
        while (j < cnt) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        // pass 3: lane owns V column `lid` — 4 rows in flight, dequant rides the weighted sum
        let vb0 = khblk + lid / 32u
        let vin = lid % 32u
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= cnt) {
            let b0 = j * nblk + vb0
            let b1 = (j + 1u) * nblk + vb0
            let b2 = (j + 2u) * nblk + vb0
            let b3 = (j + 3u) * nblk + vb0
            acc0 += scores[j] * float(vqb[b0 * 34u + 2u + vin]) * float(vsh[b0 * 17u])
            acc1 += scores[j + 1u] * float(vqb[b1 * 34u + 2u + vin]) * float(vsh[b1 * 17u])
            acc2 += scores[j + 2u] * float(vqb[b2 * 34u + 2u + vin]) * float(vsh[b2 * 17u])
            acc3 += scores[j + 3u] * float(vqb[b3 * 34u + 2u + vin]) * float(vsh[b3 * 17u])
            j += 4u
        }
        while (j < cnt) {
            let b0 = j * nblk + vb0
            acc0 += scores[j] * float(vqb[b0 * 34u + 2u + vin]) * float(vsh[b0 * 17u])
            j++
        }
        yo[qoff + lid] = (acc0 + acc1 + acc2 + acc3) / tot
    }
}

// Chunked split-K twin for deeper contexts (cnt > ATTN_CHUNK_ROWS): grid (stream, head,
// context-chunk) — each threadgroup owns one 128-row chunk of the context, runs the same
// score/softmax/V-accumulate over just its rows, and writes an UNNORMALIZED partial (acc[hs] +
// chunk max + chunk expsum); the tiny comb kernel rescales across chunks. The single-tg kernel
// above collapses into one serial per-lane loop over the whole context (24 threadgroups on a
// 32-core GPU — latency-bound, measured 7.5ms/token at depth 512); chunking scales occupancy
// with depth and divides the serial pass length by nchunks.
class MetalSqAttnPartF16 {
    @ssbo @binding = 0 kc : array<float16>
    @ssbo @binding = 1 vc : array<float16>
    @ssbo @binding = 2 q : array<float>
    @ssbo @binding = 3 part : array<float>  // [B x nheads x nchunks x (hs+2)]: acc[hs], chunk max, chunk expsum
    @uniform @binding = 4 cnt : uint
    @uniform @binding = 5 kv_dim : uint
    @uniform @binding = 6 nheads : uint
    @uniform @binding = 7 kv_mul : uint
    @uniform @binding = 8 hs : uint
    @uniform @binding = 9 scale : float
    @uniform @binding = 10 nchunks : uint
    @workgroup scores : float[64]   // == ATTN_CHUNK_ROWS
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_part16_msl")]
    def metal_sq_attn_part16 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let ci = (wg / nheads) % nchunks
        let bb = wg / (nheads * nchunks)
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let khoff = (h / kv_mul) * hs
        let qoff = bb * (nheads * hs) + h * hs
        let c0 = ci * 64u
        let c1 = min(c0 + 64u, cnt)
        let nrows = c1 - c0
        // pass 1: simdgroup-cooperative score fill over this chunk's rows (coalesced K loads,
        // loop-invariant q hoisted to registers)
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        var j = sg
        while (j < nrows) {
            let kb = (c0 + j) * kv_dim + khoff + sl * epl
            var s = q0 * float(kc[kb])
            if (epl > 1u) {
                s += q1 * float(kc[kb + 1u])
            }
            if (epl > 2u) {
                s += q2 * float(kc[kb + 2u])
            }
            if (epl > 3u) {
                s += q3 * float(kc[kb + 3u])
            }
            s = simd_sum(s)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        // pass 2a: chunk max
        var m = -3.0e38
        j = lid
        while (j < nrows) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()                           // red[] reused for the sum below
        // pass 2b: exp in place + chunk expsum
        var l = 0.0
        j = lid
        while (j < nrows) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        // pass 3: lane owns V column `lid` — 4 rows in flight over this chunk's rows
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= nrows) {
            acc0 += scores[j] * float(vc[(c0 + j) * kv_dim + khoff + lid])
            acc1 += scores[j + 1u] * float(vc[(c0 + j + 1u) * kv_dim + khoff + lid])
            acc2 += scores[j + 2u] * float(vc[(c0 + j + 2u) * kv_dim + khoff + lid])
            acc3 += scores[j + 3u] * float(vc[(c0 + j + 3u) * kv_dim + khoff + lid])
            j += 4u
        }
        while (j < nrows) {
            acc0 += scores[j] * float(vc[(c0 + j) * kv_dim + khoff + lid])
            j++
        }
        let pbase = ((bb * nheads + h) * nchunks + ci) * (hs + 2u)
        part[pbase + lid] = acc0 + acc1 + acc2 + acc3
        if (lid == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// f32-mirror twin — same math, f32 K/V loads.
class MetalSqAttnPartF32 {
    @ssbo @binding = 0 kc : array<float>
    @ssbo @binding = 1 vc : array<float>
    @ssbo @binding = 2 q : array<float>
    @ssbo @binding = 3 part : array<float>
    @uniform @binding = 4 cnt : uint
    @uniform @binding = 5 kv_dim : uint
    @uniform @binding = 6 nheads : uint
    @uniform @binding = 7 kv_mul : uint
    @uniform @binding = 8 hs : uint
    @uniform @binding = 9 scale : float
    @uniform @binding = 10 nchunks : uint
    @workgroup scores : float[64]
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_part32_msl")]
    def metal_sq_attn_part32 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let ci = (wg / nheads) % nchunks
        let bb = wg / (nheads * nchunks)
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let khoff = (h / kv_mul) * hs
        let qoff = bb * (nheads * hs) + h * hs
        let c0 = ci * 64u
        let c1 = min(c0 + 64u, cnt)
        let nrows = c1 - c0
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        var j = sg
        while (j < nrows) {
            let kb = (c0 + j) * kv_dim + khoff + sl * epl
            var s = q0 * kc[kb]
            if (epl > 1u) {
                s += q1 * kc[kb + 1u]
            }
            if (epl > 2u) {
                s += q2 * kc[kb + 2u]
            }
            if (epl > 3u) {
                s += q3 * kc[kb + 3u]
            }
            s = simd_sum(s)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        var m = -3.0e38
        j = lid
        while (j < nrows) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()
        var l = 0.0
        j = lid
        while (j < nrows) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= nrows) {
            acc0 += scores[j] * vc[(c0 + j) * kv_dim + khoff + lid]
            acc1 += scores[j + 1u] * vc[(c0 + j + 1u) * kv_dim + khoff + lid]
            acc2 += scores[j + 2u] * vc[(c0 + j + 2u) * kv_dim + khoff + lid]
            acc3 += scores[j + 3u] * vc[(c0 + j + 3u) * kv_dim + khoff + lid]
            j += 4u
        }
        while (j < nrows) {
            acc0 += scores[j] * vc[(c0 + j) * kv_dim + khoff + lid]
            j++
        }
        let pbase = ((bb * nheads + h) * nchunks + ci) * (hs + 2u)
        part[pbase + lid] = acc0 + acc1 + acc2 + acc3
        if (lid == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// q8_0-mirror chunked twin — the Part kernel's block-codec form (dual-view reads, see
// MetalSqAttnQ8); partials stay f32, so the comb below serves this form unchanged.
class MetalSqAttnPartQ8 {
    @ssbo @binding = 0 ksh : array<float16>
    @ssbo @binding = 1 kqb : array<int8>
    @ssbo @binding = 2 vsh : array<float16>
    @ssbo @binding = 3 vqb : array<int8>
    @ssbo @binding = 4 q : array<float>
    @ssbo @binding = 5 part : array<float>  // [B x nheads x nchunks x (hs+2)]: acc[hs], chunk max, chunk expsum
    @uniform @binding = 6 cnt : uint
    @uniform @binding = 7 nblk : uint       // blocks per row = kv_dim / 32
    @uniform @binding = 8 nheads : uint
    @uniform @binding = 9 kv_mul : uint
    @uniform @binding = 10 hs : uint
    @uniform @binding = 11 scale : float
    @uniform @binding = 12 nchunks : uint
    @workgroup scores : float[64]   // == ATTN_CHUNK_ROWS
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_part_q8_msl")]
    def metal_sq_attn_part_q8 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let ci = (wg / nheads) % nchunks
        let bb = wg / (nheads * nchunks)
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let khblk = (h / kv_mul) * (hs / 32u)
        let qoff = bb * (nheads * hs) + h * hs
        let c0 = ci * 64u
        let c1 = min(c0 + 64u, cnt)
        let nrows = c1 - c0
        // pass 1: simdgroup-cooperative score fill over this chunk's rows
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        let hb = (sl * epl) / 32u
        let ib = (sl * epl) % 32u
        var j = sg
        while (j < nrows) {
            let blk = (c0 + j) * nblk + khblk + hb
            let d = float(ksh[blk * 17u])
            let kb = blk * 34u + 2u + ib
            var s = q0 * float(kqb[kb])
            if (epl > 1u) {
                s += q1 * float(kqb[kb + 1u])
            }
            if (epl > 2u) {
                s += q2 * float(kqb[kb + 2u])
            }
            if (epl > 3u) {
                s += q3 * float(kqb[kb + 3u])
            }
            s = simd_sum(s * d)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        // pass 2a: chunk max
        var m = -3.0e38
        j = lid
        while (j < nrows) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()                           // red[] reused for the sum below
        // pass 2b: exp in place + chunk expsum
        var l = 0.0
        j = lid
        while (j < nrows) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        // pass 3: lane owns V column `lid` — 4 rows in flight over this chunk's rows
        let vb0 = khblk + lid / 32u
        let vin = lid % 32u
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= nrows) {
            let b0 = (c0 + j) * nblk + vb0
            let b1 = (c0 + j + 1u) * nblk + vb0
            let b2 = (c0 + j + 2u) * nblk + vb0
            let b3 = (c0 + j + 3u) * nblk + vb0
            acc0 += scores[j] * float(vqb[b0 * 34u + 2u + vin]) * float(vsh[b0 * 17u])
            acc1 += scores[j + 1u] * float(vqb[b1 * 34u + 2u + vin]) * float(vsh[b1 * 17u])
            acc2 += scores[j + 2u] * float(vqb[b2 * 34u + 2u + vin]) * float(vsh[b2 * 17u])
            acc3 += scores[j + 3u] * float(vqb[b3 * 34u + 2u + vin]) * float(vsh[b3 * 17u])
            j += 4u
        }
        while (j < nrows) {
            let b0 = (c0 + j) * nblk + vb0
            acc0 += scores[j] * float(vqb[b0 * 34u + 2u + vin]) * float(vsh[b0 * 17u])
            j++
        }
        let pbase = ((bb * nheads + h) * nchunks + ci) * (hs + 2u)
        part[pbase + lid] = acc0 + acc1 + acc2 + acc3
        if (lid == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// Cross-chunk combine: one threadgroup per (stream, head), hs threads — rescale each chunk's
// unnormalized partial to the global max, sum, normalize. Dtype-agnostic (partials are f32).
class MetalSqAttnComb {
    @ssbo @binding = 0 part : array<float>
    @ssbo @binding = 1 yo : array<float>
    @uniform @binding = 2 nchunks : uint
    @uniform @binding = 3 nheads : uint
    @uniform @binding = 4 hs : uint

    [metal_kernel(name="metal_sq_attn_comb_msl")]
    def metal_sq_attn_comb {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let bb = wg / nheads
        let lid = gl_LocalInvocationID.x
        let base = (bb * nheads + h) * nchunks
        var mm = -3.0e38
        var ci = 0u
        while (ci < nchunks) {
            mm = max(mm, part[(base + ci) * (hs + 2u) + hs])
            ci++
        }
        var acc = 0.0
        var l = 0.0
        ci = 0u
        while (ci < nchunks) {
            let pb = (base + ci) * (hs + 2u)
            let w = exp(part[pb + hs] - mm)
            acc += part[pb + lid] * w
            l += part[pb + hs + 1u] * w
            ci++
        }
        yo[bb * (nheads * hs) + h * hs + lid] = acc / l
    }
}

// The seed GEMV's f16-SCALE twin (wscale_f16 models — the qscales16 halfword plane, llama.cpp
// q8_0 byte parity): one simdgroup per output row, lanes stride byte4s, 4 blocks in flight.
// 1.0625 vs 1.125 bytes/weight — ~5.6% less stream on a bandwidth-bound decode. (A uint4
// wide-load + shift-unpack variant measured 0.5ms/token SLOWER — the char4 convert path wins
// on M1; don't re-try without new silicon.)
class MetalQ8GemvS16 {
    @ssbo @binding = 0 wq : array<byte4>    // [d x n] Q8 rows, byte4 view
    @ssbo @binding = 1 ws : array<float16>  // [d x n/32] scales, raw binary16
    @ssbo @binding = 2 x : array<float4>    // [n] f32 vector, float4 view
    @ssbo @binding = 3 y : array<float>     // [d]
    @uniform @binding = 4 ndim : uint       // n (reduction dim, % 32)
    @uniform @binding = 5 ddim : uint       // d (output rows)

    [metal_kernel(name="metal_q8_gemv_s16_msl")]
    def metal_q8_gemv_s16 {
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
        while (k + 96u < nb4) {
            let wa = float4(int4(wq[wbase + k]))
            let xa = x[k]
            let wb = float4(int4(wq[wbase + (k + 32u)]))
            let xb = x[k + 32u]
            let wc = float4(int4(wq[wbase + (k + 64u)]))
            let xc = x[k + 64u]
            let wd = float4(int4(wq[wbase + (k + 96u)]))
            let xd = x[k + 96u]
            acc += (wa.x * xa.x + wa.y * xa.y + wa.z * xa.z + wa.w * xa.w) * float(ws[sbase + k / 8u])
            acc += (wb.x * xb.x + wb.y * xb.y + wb.z * xb.z + wb.w * xb.w) * float(ws[sbase + (k + 32u) / 8u])
            acc += (wc.x * xc.x + wc.y * xc.y + wc.z * xc.z + wc.w * xc.w) * float(ws[sbase + (k + 64u) / 8u])
            acc += (wd.x * xd.x + wd.y * xd.y + wd.z * xd.z + wd.w * xd.w) * float(ws[sbase + (k + 96u) / 8u])
            k += 128u
        }
        while (k < nb4) {
            let w4 = float4(int4(wq[wbase + k]))
            let x4 = x[k]
            acc += (w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w) * float(ws[sbase + k / 8u])
            k += gl_SubgroupSize
        }
        acc = simd_sum(acc)
        if (lane == 0u) {
            y[row] = acc
        }
    }
}

// The 2-rows-per-simdgroup fused QKV GEMV + rope + KV-row store (the s16 perf path): element
// pairs (2r, 2r+1) of the q/k vectors ARE weight rows 2r, 2r+1 of the fused QKV region, so one
// simdgroup holds a whole rope pair in registers after its two row dots — q pairs rope into the
// QKV buffer (attention reads them), k pairs rope straight into the mirror K row with the
// ±65504 clamp, v pairs store raw. Deletes the rope_store dispatch AND its serialization level
// (range boundaries are pair-aligned: qd and kv_dim are even).
class MetalQ8GemvQkvRsF16 {
    @ssbo @binding = 0 wq : array<byte4>    // fused QKV rows (concat region)
    @ssbo @binding = 1 ws : array<float16>  // fused scale planes, halfwords
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 qo : array<float>    // q out, roped ([0, qd) of the QKV buffer)
    @ssbo @binding = 4 kd : array<float16>  // mirror K row (bound at the row offset)
    @ssbo @binding = 5 vd : array<float16>
    @ssbo @binding = 6 tcos : array<float>  // [hs/2] this position's rope row
    @ssbo @binding = 7 tsin : array<float>
    @uniform @binding = 8 ndim : uint
    @uniform @binding = 9 ddim : uint       // qd + 2*kvd
    @uniform @binding = 10 qd : uint
    @uniform @binding = 11 kvd : uint
    @uniform @binding = 12 hs : uint

    [metal_kernel(name="metal_q8_gemv_qkv_rs16_msl")]
    def metal_q8_gemv_qkv_rs16 {
        let row = (gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID) * 2u
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let nb = ndim / 32u
        var acc0 = 0.0
        var acc1 = 0.0
        var k = lane
        while (k < nb4) {
            let x4 = x[k]
            let w0 = float4(int4(wq[row * nb4 + k]))
            let w1 = float4(int4(wq[(row + 1u) * nb4 + k]))
            acc0 += (w0.x * x4.x + w0.y * x4.y + w0.z * x4.z + w0.w * x4.w) * float(ws[row * nb + k / 8u])
            acc1 += (w1.x * x4.x + w1.y * x4.y + w1.z * x4.z + w1.w * x4.w) * float(ws[(row + 1u) * nb + k / 8u])
            k += gl_SubgroupSize
        }
        acc0 = simd_sum(acc0)
        acc1 = simd_sum(acc1)
        if (lane != 0u) {
            return
        }
        if (row < qd) {
            let j = (row % hs) / 2u
            let fcr = tcos[j]
            let fci = tsin[j]
            qo[row] = acc0 * fcr - acc1 * fci
            qo[row + 1u] = acc0 * fci + acc1 * fcr
        } elif (row < qd + kvd) {
            let ik = row - qd
            let j = (ik % hs) / 2u
            let fcr = tcos[j]
            let fci = tsin[j]
            kd[ik] = float16(min(max(acc0 * fcr - acc1 * fci, -65504.0), 65504.0))   // nolint:STYLE019 — the MSL emitter lowers min/max only, no clamp
            kd[ik + 1u] = float16(min(max(acc0 * fci + acc1 * fcr, -65504.0), 65504.0))   // nolint:STYLE019
        } else {
            let iv = row - qd - kvd
            vd[iv] = float16(min(max(acc0, -65504.0), 65504.0))   // nolint:STYLE019
            vd[iv + 1u] = float16(min(max(acc1, -65504.0), 65504.0))   // nolint:STYLE019
        }
    }
}

// f32-mirror twin — straight stores, no clamp.
class MetalQ8GemvQkvRsF32 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 ws : array<float16>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 qo : array<float>
    @ssbo @binding = 4 kd : array<float>
    @ssbo @binding = 5 vd : array<float>
    @ssbo @binding = 6 tcos : array<float>
    @ssbo @binding = 7 tsin : array<float>
    @uniform @binding = 8 ndim : uint
    @uniform @binding = 9 ddim : uint
    @uniform @binding = 10 qd : uint
    @uniform @binding = 11 kvd : uint
    @uniform @binding = 12 hs : uint

    [metal_kernel(name="metal_q8_gemv_qkv_rs32_msl")]
    def metal_q8_gemv_qkv_rs32 {
        let row = (gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID) * 2u
        if (row >= ddim) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let nb = ndim / 32u
        var acc0 = 0.0
        var acc1 = 0.0
        var k = lane
        while (k < nb4) {
            let x4 = x[k]
            let w0 = float4(int4(wq[row * nb4 + k]))
            let w1 = float4(int4(wq[(row + 1u) * nb4 + k]))
            acc0 += (w0.x * x4.x + w0.y * x4.y + w0.z * x4.z + w0.w * x4.w) * float(ws[row * nb + k / 8u])
            acc1 += (w1.x * x4.x + w1.y * x4.y + w1.z * x4.z + w1.w * x4.w) * float(ws[(row + 1u) * nb + k / 8u])
            k += gl_SubgroupSize
        }
        acc0 = simd_sum(acc0)
        acc1 = simd_sum(acc1)
        if (lane != 0u) {
            return
        }
        if (row < qd) {
            let j = (row % hs) / 2u
            let fcr = tcos[j]
            let fci = tsin[j]
            qo[row] = acc0 * fcr - acc1 * fci
            qo[row + 1u] = acc0 * fci + acc1 * fcr
        } elif (row < qd + kvd) {
            let ik = row - qd
            let j = (ik % hs) / 2u
            let fcr = tcos[j]
            let fci = tsin[j]
            kd[ik] = acc0 * fcr - acc1 * fci
            kd[ik + 1u] = acc0 * fci + acc1 * fcr
        } else {
            let iv = row - qd - kvd
            vd[iv] = acc0
            vd[iv + 1u] = acc1
        }
    }
}

// The W1|W3 twin of the same shape: one simdgroup dots BOTH ffn rows of an output element —
// gate row r and up row r + hidden of the fused region — and writes silu(g)·u directly.
// Deletes the swiglu dispatch AND its serialization level.
class MetalQ8GemvW13Sw {
    @ssbo @binding = 0 wq : array<byte4>    // fused W1|W3 rows (concat region)
    @ssbo @binding = 1 ws : array<float16>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 ho : array<float>    // [hidden] silu(gate)·up
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 hidden : uint

    [metal_kernel(name="metal_q8_gemv_w13sw16_msl")]
    def metal_q8_gemv_w13sw16 {
        let r = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
        if (r >= hidden) {
            return
        }
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let nb = ndim / 32u
        let ubase = (r + hidden) * nb4
        let usb = (r + hidden) * nb
        var acc0 = 0.0
        var acc1 = 0.0
        var k = lane
        while (k < nb4) {
            let x4 = x[k]
            let w0 = float4(int4(wq[r * nb4 + k]))
            let w1 = float4(int4(wq[ubase + k]))
            acc0 += (w0.x * x4.x + w0.y * x4.y + w0.z * x4.z + w0.w * x4.w) * float(ws[r * nb + k / 8u])
            acc1 += (w1.x * x4.x + w1.y * x4.y + w1.z * x4.z + w1.w * x4.w) * float(ws[usb + k / 8u])
            k += gl_SubgroupSize
        }
        acc0 = simd_sum(acc0)
        acc1 = simd_sum(acc1)
        if (lane == 0u) {
            ho[r] = (acc0 / (1.0 + exp(-acc0))) * acc1
        }
    }
}

// GPU-side greedy sampling — the speculative chain's token source: one threadgroup, strided
// argmax over the logits with the CPU sampler's exact tie-break (lowest index wins; per-lane the
// strict > keeps the earliest, cross-lane the (==, <) rule picks the lower index).
class MetalArgmax {
    @ssbo @binding = 0 lg : array<float>
    @ssbo @binding = 1 tok : array<uint>    // [1] winner index out
    @uniform @binding = 2 n : uint
    @workgroup redv : float[32]
    @workgroup redi : uint[32]

    [metal_kernel(name="metal_argmax_msl")]
    def metal_argmax {
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        var bv = -3.0e38
        var bi = 0u
        var j = lid
        while (j < n) {
            let v = lg[j]
            if (v > bv) {
                bv = v
                bi = j
            }
            j += nt
        }
        var m = 16u
        while (m >= 1u) {
            let ov = simd_shuffle_xor(bv, m)
            let oi = simd_shuffle_xor(bi, m)
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov
                bi = oi
            }
            m /= 2u
        }
        if (gl_SubgroupInvocationID == 0u) {
            redv[gl_SubgroupID] = bv
            redi[gl_SubgroupID] = bi
        }
        barrier()
        if (lid == 0u) {
            var wv = redv[0]
            var wi = redi[0]
            var sgi = 1u
            while (sgi < gl_NumSubgroups) {
                if (redv[sgi] > wv || (redv[sgi] == wv && redi[sgi] < wi)) {
                    wv = redv[sgi]
                    wi = redi[sgi]
                }
                sgi++
            }
            tok[0] = wi
        }
    }
}

// Speculative embed gather: dequant the argmax winner's Q8 embedding row (the cls_q8 linear copy
// at emb_q8_off) into the next step's residual input — the GPU chain's embed_row.
class MetalEmbedQ8 {
    @ssbo @binding = 0 q : array<int8>      // the whole linear embed plane
    @ssbo @binding = 1 sc : array<float>    // its scale plane
    @ssbo @binding = 2 tok : array<uint>
    @ssbo @binding = 3 xo : array<float>    // [dim] residual input out
    @uniform @binding = 4 dim : uint

    [metal_kernel(name="metal_embed_q8_msl")]
    def metal_embed_q8 {
        let gid = gl_GlobalInvocationID.x
        if (gid >= dim) {
            return
        }
        let base = tok[0] * dim + gid
        xo[gid] = float(q[base]) * sc[base / 32u]
    }
}

// f16-scale twin (wscale_f16 models — the qscales16 halfword plane).
class MetalEmbedQ8S16 {
    @ssbo @binding = 0 q : array<int8>
    @ssbo @binding = 1 sc : array<float16>
    @ssbo @binding = 2 tok : array<uint>
    @ssbo @binding = 3 xo : array<float>
    @uniform @binding = 4 dim : uint

    [metal_kernel(name="metal_embed_q8_s16_msl")]
    def metal_embed_q8_s16 {
        let gid = gl_GlobalInvocationID.x
        if (gid >= dim) {
            return
        }
        let base = tok[0] * dim + gid
        xo[gid] = float(q[base]) * float(sc[base / 32u])
    }
}

// Fused residual-add + rmsnorm, one threadgroup per step: x += a in place, then y = rms(x)·w —
// one dispatch instead of the add + rmsnorm pair (two of these per layer). The added row stages
// through the threadgroup slab because barrier() is a threadgroup fence only — this dispatch's
// own device x writes are not ordered for re-reads. dim caps at the slab; wider models take the
// unfused pair (enc_add_rms_site).
class MetalAddRms {
    @ssbo @binding = 0 x : array<float>     // [dim] residual stream, add target
    @ssbo @binding = 1 a : array<float>     // [dim] addend (fully read before y is written — y may alias a)
    @ssbo @binding = 2 w : array<float>     // [dim] norm weight
    @ssbo @binding = 3 y : array<float>     // [dim] normed out
    @uniform @binding = 4 dim : uint
    @uniform @binding = 5 eps : float
    @workgroup xrow : float[4096]           // == ADD_RMS_MAX_DIM
    @workgroup partial : float[32]          // one per simdgroup at the 1024-thread dispatch
    @workgroup inv_sh : float

    [metal_kernel(name="metal_add_rms_msl")]
    def metal_add_rms {
        let lid = gl_LocalInvocationID.x
        var ss = 0.0
        var k = lid
        while (k < dim) {
            let vv = x[k] + a[k]
            x[k] = vv
            xrow[k] = vv
            ss += vv * vv
            k += gl_WorkGroupSize.x
        }
        ss = simd_sum(ss)
        if (gl_SubgroupInvocationID == 0u) {
            partial[gl_SubgroupID] = ss
        }
        barrier()
        if (lid == 0u) {
            var tot = 0.0
            var sgi = 0u
            while (sgi < gl_NumSubgroups) {
                tot += partial[sgi]
                sgi++
            }
            inv_sh = rsqrt(tot / float(dim) + eps)
        }
        barrier()
        let inv = inv_sh
        k = lid
        while (k < dim) {
            y[k] = w[k] * (xrow[k] * inv)
            k += gl_WorkGroupSize.x
        }
    }
}

// Fused rope + KV-row store, one thread per rotation pair (ntok == 1): q pairs rope in place;
// k pairs rope straight into the mirror K row (nothing reads the roped k buffer — attention
// reads the mirror) with cvt_f32_to_f16's exact ±65504 clamp, and the same thread copies its V
// pair. Replaces the rope_qk + kv_store dispatch pair. The v source binds the same fused QKV
// buffer at its v byte offset (disjoint from the q/k range).
class MetalRopeStoreF16 {
    @ssbo @binding = 0 qk : array<float>    // q at [0, qd) roped in place, k at [qd, qd + kvd)
    @ssbo @binding = 1 vs : array<float>    // raw v (the QKV buffer at the v offset)
    @ssbo @binding = 2 kd : array<float16>  // mirror K row (bound at the (layer, pos) row offset)
    @ssbo @binding = 3 vd : array<float16>
    @ssbo @binding = 4 tcos : array<float>  // [head_size/2] this position's rope row
    @ssbo @binding = 5 tsin : array<float>
    @uniform @binding = 6 qd : uint
    @uniform @binding = 7 head_size : uint
    @uniform @binding = 8 npairs_q : uint   // qd/2 — the q half of the grid
    @uniform @binding = 9 npairs : uint     // + kvd/2 — grid guard

    [metal_kernel(name="metal_rope_store16_msl")]
    def metal_rope_store16 {
        let gid = gl_GlobalInvocationID.x
        if (gid >= npairs) {
            return
        }
        let isK = gid >= npairs_q
        let lgid = isK ? gid - npairs_q : gid
        let half = head_size / 2u
        let j = lgid % half
        let i = (lgid / half) * head_size + 2u * j
        let fcr = tcos[j]
        let fci = tsin[j]
        if (isK) {
            let v0 = qk[qd + i]
            let v1 = qk[qd + i + 1u]
            let r0 = v0 * fcr - v1 * fci
            let r1 = v0 * fci + v1 * fcr
            kd[i] = float16(min(max(r0, -65504.0), 65504.0))   // nolint:STYLE019 — the MSL emitter lowers min/max only, no clamp
            kd[i + 1u] = float16(min(max(r1, -65504.0), 65504.0))   // nolint:STYLE019
            vd[i] = float16(min(max(vs[i], -65504.0), 65504.0))   // nolint:STYLE019
            vd[i + 1u] = float16(min(max(vs[i + 1u], -65504.0), 65504.0))   // nolint:STYLE019
        } else {
            let v0 = qk[i]
            let v1 = qk[i + 1u]
            qk[i] = v0 * fcr - v1 * fci
            qk[i + 1u] = v0 * fci + v1 * fcr
        }
    }
}

// f32-mirror twin — straight stores, no clamp.
class MetalRopeStoreF32 {
    @ssbo @binding = 0 qk : array<float>
    @ssbo @binding = 1 vs : array<float>
    @ssbo @binding = 2 kd : array<float>
    @ssbo @binding = 3 vd : array<float>
    @ssbo @binding = 4 tcos : array<float>
    @ssbo @binding = 5 tsin : array<float>
    @uniform @binding = 6 qd : uint
    @uniform @binding = 7 head_size : uint
    @uniform @binding = 8 npairs_q : uint
    @uniform @binding = 9 npairs : uint

    [metal_kernel(name="metal_rope_store32_msl")]
    def metal_rope_store32 {
        let gid = gl_GlobalInvocationID.x
        if (gid >= npairs) {
            return
        }
        let isK = gid >= npairs_q
        let lgid = isK ? gid - npairs_q : gid
        let half = head_size / 2u
        let j = lgid % half
        let i = (lgid / half) * head_size + 2u * j
        let fcr = tcos[j]
        let fci = tsin[j]
        if (isK) {
            let v0 = qk[qd + i]
            let v1 = qk[qd + i + 1u]
            kd[i] = v0 * fcr - v1 * fci
            kd[i + 1u] = v0 * fci + v1 * fcr
            vd[i] = vs[i]
            vd[i + 1u] = vs[i + 1u]
        } else {
            let v0 = qk[i]
            let v1 = qk[i + 1u]
            qk[i] = v0 * fcr - v1 * fci
            qk[i + 1u] = v0 * fci + v1 * fcr
        }
    }
}

// q8_0-mirror rope + KV-row store: q pairs rope in place exactly like the f16 form (one thread
// per pair); K and V rows quantize per 32-block — one thread per block, MetalQ8Quant's shape —
// with quantize_q8kv_row's exact semantics (d = amax/127 stored f16, quants round with the
// unrounded f32 1/d), so CPU fallback steps read the same codec the CPU store would have
// written. K blocks rope their 16 pairs first (rope pairs never straddle a block: head_size %
// 32 == 0), V blocks copy raw. Grid: npairs_q pair threads + nblk K-block + nblk V-block threads.
class MetalRopeStoreQ8 {
    @ssbo @binding = 0 qk : array<float>    // q at [0, qd) roped in place, k at [qd, qd + kvd)
    @ssbo @binding = 1 vs : array<float>    // raw v (the QKV buffer at the v offset)
    @ssbo @binding = 2 ksh : array<float16> // mirror K row, HALF-SCALE view (bound at the (layer, pos) row offset)
    @ssbo @binding = 3 kqb : array<int8>    // the SAME K row, BYTE view
    @ssbo @binding = 4 vsh : array<float16>
    @ssbo @binding = 5 vqb : array<int8>
    @ssbo @binding = 6 tcos : array<float>  // [head_size/2] this position's rope row
    @ssbo @binding = 7 tsin : array<float>
    @uniform @binding = 8 qd : uint
    @uniform @binding = 9 head_size : uint
    @uniform @binding = 10 npairs_q : uint  // qd/2 — the q-rope half of the grid
    @uniform @binding = 11 nblk : uint      // kv_dim/32 — K blocks (V adds the same again)

    [metal_kernel(name="metal_rope_store_q8_msl")]
    def metal_rope_store_q8 {
        let gid = gl_GlobalInvocationID.x
        if (gid >= npairs_q + 2u * nblk) {
            return
        }
        let half = head_size / 2u
        if (gid < npairs_q) {
            let j = gid % half
            let i = (gid / half) * head_size + 2u * j
            let fcr = tcos[j]
            let fci = tsin[j]
            let v0 = qk[i]
            let v1 = qk[i + 1u]
            qk[i] = v0 * fcr - v1 * fci
            qk[i + 1u] = v0 * fci + v1 * fcr
            return
        }
        let bi = gid - npairs_q
        let isV = bi >= nblk
        let blk = isV ? bi - nblk : bi
        let base = blk * 32u                // element base within the K/V row
        var vals : float[32]
        if (isV) {
            var e = 0u
            while (e < 32u) {
                vals[e] = vs[base + e]
                e++
            }
        } else {
            var p = 0u
            while (p < 16u) {
                let ge = base + 2u * p
                let j = (ge % head_size) / 2u
                let fcr = tcos[j]
                let fci = tsin[j]
                let v0 = qk[qd + ge]
                let v1 = qk[qd + ge + 1u]
                vals[2u * p] = v0 * fcr - v1 * fci
                vals[2u * p + 1u] = v0 * fci + v1 * fcr
                p++
            }
        }
        var amax = 0.0
        var k = 0u
        while (k < 32u) {
            amax = max(amax, abs(vals[k]))
            k++
        }
        let d = amax / 127.0
        let id = d != 0.0 ? 1.0 / d : 0.0
        if (isV) {
            vsh[blk * 17u] = float16(d)
            k = 0u
            while (k < 32u) {
                vqb[blk * 34u + 2u + k] = int8(round(vals[k] * id))
                k++
            }
        } else {
            ksh[blk * 17u] = float16(d)
            k = 0u
            while (k < 32u) {
                kqb[blk * 34u + 2u + k] = int8(round(vals[k] * id))
                k++
            }
        }
    }
}

// ===== the batched-step kernels (P4) =====
// One eval_batch step = B independent sessions through one weight pass. The GEMV sites become
// either fixed-B batched GEMVs (B <= 4 — the P1a lab's winners) or the M-pad-32 GEMM (B >= 5,
// B-independent cost), and the per-row kernels (attention, rope+store) reach each row's KV slice
// through the mirror-arena row table: rt[b] = (k base, v base, cap_rows, cnt) in arena ELEMENTS,
// with the layer index a uniform — per-row capacities differ, so the layer stride is per row.

// Fixed-B batched GEMV twins, X-STAGED (lab round 3, 2026-07-14): the tg's 4 rows all dot the
// SAME B activation vectors, so a 64-float4-per-stream chunk of the X panel stages in threadgroup
// memory once per tg (cooperative 128-thread load) and the inner-loop x reads become tgmem hits —
// the per-lane device x loads were the ALU-bound form's limiter (lab: B=2 +40-88%, B=4 ~2x on
// every decode shape; per-lane-CONSECUTIVE chunking measured 30-45% WORSE — inter-lane coalescing
// beats per-lane locality on AGX). W stays strided/coalesced; the strided tail (device x, no
// barriers) covers n % 256 != 0; row clamps instead of early-returning so barriers stay converged
// in a partial last tg. Geometry is pinned at 128 threads = 4 rows (the slot math bakes it in).
// Scales bind through BOTH views (f32 plane / wscale_f16 halfword plane — the same buffer twice,
// MulMm's dual-view trick); `s16` picks per model.
class MetalGemvB2 {
    @ssbo @binding = 0 wq : array<byte4>    // [d x n] Q8 rows, byte4 view
    @ssbo @binding = 1 wsf : array<float>   // [d x n/32] scales, f32 view
    @ssbo @binding = 2 wsh : array<float16> // ... the same buffer, halfword view (wscale_f16)
    @ssbo @binding = 3 x : array<float4>    // [B x n] f32 rows, float4 view
    @ssbo @binding = 4 y : array<float>     // [B x d]
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 ddim : uint
    @uniform @binding = 7 s16 : uint
    @workgroup txp : float4[128]            // [2 streams x 64 float4] X panel chunk

    [metal_kernel(name="metal_gemv_b2_msl")]
    def metal_gemv_b2 {
        let lid = gl_LocalInvocationID.x
        let row = min(gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID, ddim - 1u)
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[2]
        var cb = 0u
        while (cb + 64u <= nb4) {
            barrier()                       // previous round done reading the panel
            txp[lid] = x[(lid / 64u) * nb4 + cb + lid % 64u]
            barrier()
            for [unroll_full] (i in range(2)) {
                let k = cb + uint(i) * 32u + lane
                let sv = s16 != 0u ? float(wsh[sbase + k / 8u]) : wsf[sbase + k / 8u]
                let w4 = float4(int4(wq[wbase + k])) * sv
                let kt = uint(i) * 32u + lane
                for [unroll_full] (b in range(2)) {
                    let xv = txp[uint(b) * 64u + kt]
                    acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
                }
            }
            cb += 64u
        }
        var k = cb + lane
        while (k < nb4) {
            let sv = s16 != 0u ? float(wsh[sbase + k / 8u]) : wsf[sbase + k / 8u]
            let w4 = float4(int4(wq[wbase + k])) * sv
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

class MetalGemvB4 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 wsf : array<float>
    @ssbo @binding = 2 wsh : array<float16>
    @ssbo @binding = 3 x : array<float4>
    @ssbo @binding = 4 y : array<float>
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 ddim : uint
    @uniform @binding = 7 s16 : uint
    @workgroup txp : float4[256]            // [4 streams x 64 float4] X panel chunk

    [metal_kernel(name="metal_gemv_b4_msl")]
    def metal_gemv_b4 {
        let lid = gl_LocalInvocationID.x
        let row = min(gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID, ddim - 1u)
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let wbase = row * nb4
        let sbase = row * (ndim / 32u)
        var acc : float[4]
        var cb = 0u
        while (cb + 64u <= nb4) {
            barrier()                       // previous round done reading the panel
            for [unroll_full] (j in range(2)) {
                let slot = uint(j) * 128u + lid
                txp[slot] = x[(slot / 64u) * nb4 + cb + slot % 64u]
            }
            barrier()
            for [unroll_full] (i in range(2)) {
                let k = cb + uint(i) * 32u + lane
                let sv = s16 != 0u ? float(wsh[sbase + k / 8u]) : wsf[sbase + k / 8u]
                let w4 = float4(int4(wq[wbase + k])) * sv
                let kt = uint(i) * 32u + lane
                for [unroll_full] (b in range(4)) {
                    let xv = txp[uint(b) * 64u + kt]
                    acc[b] += w4.x * xv.x + w4.y * xv.y + w4.z * xv.z + w4.w * xv.w
                }
            }
            cb += 64u
        }
        var k = cb + lane
        while (k < nb4) {
            let sv = s16 != 0u ? float(wsh[sbase + k / 8u]) : wsf[sbase + k / 8u]
            let w4 = float4(int4(wq[wbase + k])) * sv
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

// The winning skinny-batch GEMV shape (the Part-3 in-graph verdict: this streamed form beats the
// x-staged panel kernels ~12% at B <= 4): 64 threads = 2 simdgroups x 2 row-halves; 16 lanes
// stream ONE output row's interleaved 34B q8 blocks (each lane owns chunk tx%8 of every 2nd
// block — 4 independent chunk loads per step, the scale rides the quants' cacheline), the batch
// columns share the dequantized weight stream, shuffle-down folds the 16-lane partials.
// Requires n % 256 == 0 (the 4-chunk stripe reads unguarded — the same constraint the reference
// PSO bakes). W = the SAME dual-view blob the prefill mul_mm reads.
class MetalQ8MvB2 {
    @ssbo @binding = 0 wsh : array<float16> // 34B-block W blob, half-scale view (block b at 17*b)
    @ssbo @binding = 1 wqb : array<int8>    // the same blob buffer, byte view (quants at 34*b + 2)
    @ssbo @binding = 2 x : array<float4>    // [B rows x xs4 float4-stride] f32 rows
    @ssbo @binding = 3 y : array<float>     // [B x ddim]
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint
    @uniform @binding = 6 xs4 : uint        // x row stride in float4s (ndim/4 = dense)

    [metal_kernel(name="metal_q8_mv_b2_msl")]
    def metal_q8_mv_b2 {
        let lane = gl_SubgroupInvocationID
        let tx = lane % 16u
        let row0 = gl_WorkGroupID.x * 4u + gl_SubgroupID * 2u + lane / 16u
        let row = min(row0, ddim - 1u)      // clamp, not return — keeps the shuffle groups full
        let nb = ndim / 32u
        let blk = row * nb + tx / 8u
        let cq = 4u * (tx % 4u) + 16u * ((tx % 8u) / 4u)   // quant offset of chunk tx%8 in its block
        var sp = unsafe(addr(wsh[blk * 17u]))
        var qp = unsafe(addr(wqb[blk * 34u + 2u + cq]))
        var x0 = unsafe(addr(x[tx]))
        var x1 = unsafe(addr(x[xs4 + tx]))
        var acc : float[2]
        var lw : float4[4]
        var ich = tx
        while (4u * ich < ndim) {
            // all 4 chunk dequants batch AHEAD of the fma chains — the load batching that makes
            // this shape win (interleaved dequant+fma measured +0.09 GPU-ms/step)
            for [unroll_full] (c in range(4)) {
                let d = float(unsafe(sp[uint(c) * 34u]))   // chunk c sits 2 blocks (34 halves) on
                let qb = uint(c) * 68u                     // ... and 68 quant bytes
                lw[c] = float4(float(unsafe(qp[qb])), float(unsafe(qp[qb + 1u])),
                    float(unsafe(qp[qb + 2u])), float(unsafe(qp[qb + 3u]))) * d
            }
            for [unroll_full] (c in range(4)) {
                let xv0 = unsafe(x0[uint(c) * 16u])
                let xv1 = unsafe(x1[uint(c) * 16u])
                acc[0] += lw[c].x * xv0.x + lw[c].y * xv0.y + lw[c].z * xv0.z + lw[c].w * xv0.w
                acc[1] += lw[c].x * xv1.x + lw[c].y * xv1.y + lw[c].z * xv1.z + lw[c].w * xv1.w
            }
            sp = unsafe(sp + 136)           // 8 blocks x 17 halves per stripe
            qp = unsafe(qp + 272)           // 8 blocks x 34 bytes
            x0 = unsafe(x0 + 64)
            x1 = unsafe(x1 + 64)
            ich += 64u
        }
        for [unroll_full] (b in range(2)) {
            var s = acc[b]
            s += simd_shuffle_down(s, 8u)
            s += simd_shuffle_down(s, 4u)
            s += simd_shuffle_down(s, 2u)
            s += simd_shuffle_down(s, 1u)
            if (tx == 0u && row0 < ddim) {
                y[uint(b) * ddim + row0] = s
            }
        }
    }
}

// the 4-column twin (serves B = 3..4; pad column reads clamp to row 0, stores gate on nrows).
// The 4-column geometry drops to 8 lanes per output row (4 rows/simdgroup, 8 rows/tg — the
// reference bakes nxpsg=8 for its 4-column form): with 4 fma chains per weight load the stream
// needs fewer lanes, and doubling the rows in flight per tg hides the block-load latency instead.
// Each lane owns chunk tx of EVERY block (8 chunks/block == 8 lanes), hopping 1 block per step.
class MetalQ8MvB4 {
    @ssbo @binding = 0 wsh : array<float16>
    @ssbo @binding = 1 wqb : array<int8>
    @ssbo @binding = 2 x : array<float4>
    @ssbo @binding = 3 y : array<float>
    @uniform @binding = 4 ndim : uint
    @uniform @binding = 5 ddim : uint
    @uniform @binding = 6 nrows : uint
    @uniform @binding = 7 xs4 : uint

    [metal_kernel(name="metal_q8_mv_b4_msl")]
    def metal_q8_mv_b4 {
        let lane = gl_SubgroupInvocationID
        let tx = lane % 8u
        let row0 = gl_WorkGroupID.x * 8u + gl_SubgroupID * 4u + lane / 8u
        let row = min(row0, ddim - 1u)
        let nb = ndim / 32u
        let blk = row * nb
        let cq = 4u * (tx % 4u) + 16u * (tx / 4u)   // quant offset of chunk tx in its block
        var sp = unsafe(addr(wsh[blk * 17u]))
        var qp = unsafe(addr(wqb[blk * 34u + 2u + cq]))
        var x0 = unsafe(addr(x[tx]))
        var x1 = unsafe(addr(x[xs4 + tx]))
        var x2 = unsafe(addr(x[min(2u, nrows - 1u) * xs4 + tx]))
        var x3 = unsafe(addr(x[min(3u, nrows - 1u) * xs4 + tx]))
        var acc : float[4]
        var lw : float4[4]
        var ich = tx
        while (4u * ich < ndim) {
            for [unroll_full] (c in range(4)) {
                let d = float(unsafe(sp[uint(c) * 17u]))   // chunk c sits 1 block (17 halves) on
                let qb = uint(c) * 34u
                lw[c] = float4(float(unsafe(qp[qb])), float(unsafe(qp[qb + 1u])),
                    float(unsafe(qp[qb + 2u])), float(unsafe(qp[qb + 3u]))) * d
            }
            for [unroll_full] (c in range(4)) {
                let o = uint(c) * 8u
                let xv0 = unsafe(x0[o])
                let xv1 = unsafe(x1[o])
                let xv2 = unsafe(x2[o])
                let xv3 = unsafe(x3[o])
                acc[0] += lw[c].x * xv0.x + lw[c].y * xv0.y + lw[c].z * xv0.z + lw[c].w * xv0.w
                acc[1] += lw[c].x * xv1.x + lw[c].y * xv1.y + lw[c].z * xv1.z + lw[c].w * xv1.w
                acc[2] += lw[c].x * xv2.x + lw[c].y * xv2.y + lw[c].z * xv2.z + lw[c].w * xv2.w
                acc[3] += lw[c].x * xv3.x + lw[c].y * xv3.y + lw[c].z * xv3.z + lw[c].w * xv3.w
            }
            sp = unsafe(sp + 68)            // 4 blocks x 17 halves per stripe
            qp = unsafe(qp + 136)           // 4 blocks x 34 bytes
            x0 = unsafe(x0 + 32)
            x1 = unsafe(x1 + 32)
            x2 = unsafe(x2 + 32)
            x3 = unsafe(x3 + 32)
            ich += 32u
        }
        for [unroll_full] (b in range(4)) {
            var s = acc[b]
            s += simd_shuffle_down(s, 4u)
            s += simd_shuffle_down(s, 2u)
            s += simd_shuffle_down(s, 1u)
            if (tx == 0u && row0 < ddim && uint(b) < nrows) {
                y[uint(b) * ddim + row0] = s
            }
        }
    }
}

// Fused fixed-B W1|W3 GEMV + swiglu (the batch twin of MetalQ8GemvW13Sw): each subgroup owns one
// hidden row, dots the gate row r AND the up row r+hidden against the shared x panel for every
// batch row, and writes silu(gate)*up directly — replacing the w1 GEMV + w3 GEMV + swiglu triple
// with one dispatch. The x panel is staged once and reused for both weight rows. Dual-view scales
// (s16 uniform picks the halfword plane). B2 form serves B <= 2; B4 serves 3..4.
class MetalGemvW13SwB2 {
    @ssbo @binding = 0 wq : array<byte4>    // cat W1|W3 rows, byte4 view
    @ssbo @binding = 1 wsf : array<float>   // scales, f32 view
    @ssbo @binding = 2 wsh : array<float16> // ... same buffer, halfword view (wscale_f16)
    @ssbo @binding = 3 x : array<float4>    // [B x ndim] f32 rows, float4 view
    @ssbo @binding = 4 y : array<float>     // [B x hidden] silu(gate)*up
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 hidden : uint     // output rows AND the gate/up row split
    @uniform @binding = 7 s16 : uint
    @workgroup txp : float4[128]            // [2 streams x 64 float4] X panel chunk

    [metal_kernel(name="metal_gemv_w13sw_b2_msl")]
    def metal_gemv_w13sw_b2 {
        let lid = gl_LocalInvocationID.x
        let row = min(gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID, hidden - 1u)
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let ns = ndim / 32u
        let wbg = row * nb4              // gate weight row (W1)
        let wbu = (row + hidden) * nb4   // up weight row (W3)
        let sbg = row * ns
        let sbu = (row + hidden) * ns
        var accg : float[2]
        var accu : float[2]
        var cb = 0u
        while (cb + 64u <= nb4) {
            barrier()                       // previous round done reading the panel
            txp[lid] = x[(lid / 64u) * nb4 + cb + lid % 64u]
            barrier()
            for [unroll_full] (i in range(2)) {
                let k = cb + uint(i) * 32u + lane
                let svg = s16 != 0u ? float(wsh[sbg + k / 8u]) : wsf[sbg + k / 8u]
                let svu = s16 != 0u ? float(wsh[sbu + k / 8u]) : wsf[sbu + k / 8u]
                let wg = float4(int4(wq[wbg + k])) * svg
                let wu = float4(int4(wq[wbu + k])) * svu
                let kt = uint(i) * 32u + lane
                for [unroll_full] (b in range(2)) {
                    let xv = txp[uint(b) * 64u + kt]
                    accg[b] += wg.x * xv.x + wg.y * xv.y + wg.z * xv.z + wg.w * xv.w
                    accu[b] += wu.x * xv.x + wu.y * xv.y + wu.z * xv.z + wu.w * xv.w
                }
            }
            cb += 64u
        }
        var k = cb + lane
        while (k < nb4) {
            let svg = s16 != 0u ? float(wsh[sbg + k / 8u]) : wsf[sbg + k / 8u]
            let svu = s16 != 0u ? float(wsh[sbu + k / 8u]) : wsf[sbu + k / 8u]
            let wg = float4(int4(wq[wbg + k])) * svg
            let wu = float4(int4(wq[wbu + k])) * svu
            for [unroll_full] (b in range(2)) {
                let xv = x[uint(b) * nb4 + k]
                accg[b] += wg.x * xv.x + wg.y * xv.y + wg.z * xv.z + wg.w * xv.w
                accu[b] += wu.x * xv.x + wu.y * xv.y + wu.z * xv.z + wu.w * xv.w
            }
            k += gl_SubgroupSize
        }
        for [unroll_full] (b in range(2)) {
            let g = simd_sum(accg[b])
            let u = simd_sum(accu[b])
            if (lane == 0u) {
                y[uint(b) * hidden + row] = (g / (1.0 + exp(-g))) * u
            }
        }
    }
}

class MetalGemvW13SwB4 {
    @ssbo @binding = 0 wq : array<byte4>
    @ssbo @binding = 1 wsf : array<float>
    @ssbo @binding = 2 wsh : array<float16>
    @ssbo @binding = 3 x : array<float4>
    @ssbo @binding = 4 y : array<float>
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 hidden : uint
    @uniform @binding = 7 s16 : uint
    @workgroup txp : float4[256]            // [4 streams x 64 float4] X panel chunk

    [metal_kernel(name="metal_gemv_w13sw_b4_msl")]
    def metal_gemv_w13sw_b4 {
        let lid = gl_LocalInvocationID.x
        let row = min(gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID, hidden - 1u)
        let lane = gl_SubgroupInvocationID
        let nb4 = ndim / 4u
        let ns = ndim / 32u
        let wbg = row * nb4
        let wbu = (row + hidden) * nb4
        let sbg = row * ns
        let sbu = (row + hidden) * ns
        var accg : float[4]
        var accu : float[4]
        var cb = 0u
        while (cb + 64u <= nb4) {
            barrier()
            for [unroll_full] (j in range(2)) {
                let slot = uint(j) * 128u + lid
                txp[slot] = x[(slot / 64u) * nb4 + cb + slot % 64u]
            }
            barrier()
            for [unroll_full] (i in range(2)) {
                let k = cb + uint(i) * 32u + lane
                let svg = s16 != 0u ? float(wsh[sbg + k / 8u]) : wsf[sbg + k / 8u]
                let svu = s16 != 0u ? float(wsh[sbu + k / 8u]) : wsf[sbu + k / 8u]
                let wg = float4(int4(wq[wbg + k])) * svg
                let wu = float4(int4(wq[wbu + k])) * svu
                let kt = uint(i) * 32u + lane
                for [unroll_full] (b in range(4)) {
                    let xv = txp[uint(b) * 64u + kt]
                    accg[b] += wg.x * xv.x + wg.y * xv.y + wg.z * xv.z + wg.w * xv.w
                    accu[b] += wu.x * xv.x + wu.y * xv.y + wu.z * xv.z + wu.w * xv.w
                }
            }
            cb += 64u
        }
        var k = cb + lane
        while (k < nb4) {
            let svg = s16 != 0u ? float(wsh[sbg + k / 8u]) : wsf[sbg + k / 8u]
            let svu = s16 != 0u ? float(wsh[sbu + k / 8u]) : wsf[sbu + k / 8u]
            let wg = float4(int4(wq[wbg + k])) * svg
            let wu = float4(int4(wq[wbu + k])) * svu
            for [unroll_full] (b in range(4)) {
                let xv = x[uint(b) * nb4 + k]
                accg[b] += wg.x * xv.x + wg.y * xv.y + wg.z * xv.z + wg.w * xv.w
                accu[b] += wu.x * xv.x + wu.y * xv.y + wu.z * xv.z + wu.w * xv.w
            }
            k += gl_SubgroupSize
        }
        for [unroll_full] (b in range(4)) {
            let g = simd_sum(accg[b])
            let u = simd_sum(accu[b])
            if (lane == 0u) {
                y[uint(b) * hidden + row] = (g / (1.0 + exp(-g))) * u
            }
        }
    }
}

// The M-pad-32 batch GEMM (B >= 5, cost B-independent to 32 rows) — MulMm's staging discipline
// on the occupancy-friendly 32x32 tile (lab: 110 instr/k-block vs the old 235; 1.16-1.33x on
// every decode shape). Raw f32 X — the quant tails are gone; dual-view WEIGHT scales (s16 picks).
// Every thread stages 8 W elems AND 8 X elems (indexes hoisted; device reads + dequant into
// registers BEFORE the barrier so they fly while the previous math drains); k-major 8x8 W tiles
// feed the mma straight. Pad token rows carry stale bytes by design: C rows are per-token
// independent, and pad Y rows are never read back.
class MetalQ8GemmB {
    @ssbo @binding = 0 wq : array<byte4>    // W [d x n] row-major Q8 quants, byte4 view
    @ssbo @binding = 1 wsf : array<float>   // W scales [d x n/32], f32 view
    @ssbo @binding = 2 wsh : array<float16> // ... the same buffer, halfword view (wscale_f16)
    @ssbo @binding = 3 x : array<float4>    // X [M' x n] f32 rows (M' = 32-padded B)
    @ssbo @binding = 4 y : array<float>     // Y [M' x d]
    @uniform @binding = 5 kdim : uint       // n (K)
    @uniform @binding = 6 ndim : uint       // d (N)
    @uniform @binding = 7 s16 : uint
    @workgroup tw : float16[1024]           // W: 4 k-tiles x 4 row-tiles of 8x8, (k row, wrow col)
    @workgroup tx : float16[1024]           // X: 4 k-tiles x 4 token-tiles of 8x8, (token row, k col)

    [metal_kernel(name="metal_q8_gemm_b_msl")]
    def metal_q8_gemm_b {
        let lid = gl_LocalInvocationID.x
        let sg = gl_SubgroupID
        let ntileN = ndim / 32u
        let mBase = (gl_WorkGroupID.x / ntileN) * 32u
        let nBase = (gl_WorkGroupID.x % ntileN) * 32u
        var mc : simdgroup_float8x8[4]      // this simdgroup's 16x16 quadrant: 2 token x 2 wrow tiles
        var ma : simdgroup_half8x8[2]
        var mb : simdgroup_half8x8[2]
        let kdim4 = kdim / 4u
        let nkb = kdim / 32u
        // staging split: thread handles row lr (0..31) x k-eighth il (0..3) — 8 elems of W AND X
        let lr = lid / 4u
        let il = lid % 4u
        let tW0 = (il * 4u + lr / 8u) * 64u + (lr % 8u)
        let tX0 = (il * 4u + lr / 8u) * 64u + (lr % 8u) * 8u
        let aQ = (sg % 2u) * 128u           // wrow tile-pair base within a k-tile slab
        let bQ = (sg / 2u) * 128u           // token tile-pair base
        var wp = unsafe(addr(wq[(nBase + lr) * kdim4 + il * 2u]))
        var spf = unsafe(addr(wsf[(nBase + lr) * nkb]))
        var sph = unsafe(addr(wsh[(nBase + lr) * nkb]))
        var xp = unsafe(addr(x[(mBase + lr) * kdim4 + il * 2u]))
        var kb = 0u
        while (kb < nkb) {
            let sA = s16 != 0u ? float(unsafe(sph[0])) : unsafe(spf[0])
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
            spf = unsafe(spf + 1u)
            sph = unsafe(sph + 1u)
            xp = unsafe(xp + 8u)
            kb++
        }
        let row0 = mBase + (sg / 2u) * 16u
        let col0 = nBase + (sg % 2u) * 16u
        simdgroup_store(mc[0], y, row0 * ndim + col0, ndim)
        simdgroup_store(mc[1], y, row0 * ndim + (col0 + 8u), ndim)
        simdgroup_store(mc[2], y, (row0 + 8u) * ndim + col0, ndim)
        simdgroup_store(mc[3], y, (row0 + 8u) * ndim + (col0 + 8u), ndim)
    }
}

// The split-K twin of MetalQ8GemmB (the GEMM lab's per-site verdicts): grid.y gains a k-slice
// axis — each threadgroup walks nkb/ksplit k-blocks into its own partial-C plane, and the
// MetalSkReduce pass sums the planes (separate dispatch — no float atomics, deterministic).
// At M-pad-32 the M axis is one tile, so k-slicing is the only way to grow the grid past d/32
// threadgroups on the starved sites (qkv, wo, w2). Planes are 32 x d: the driver gates split-K
// to mp == 32.
class MetalQ8GemmBSk {
    @ssbo @binding = 0 wq : array<byte4>    // W [d x n] row-major Q8 quants, byte4 view
    @ssbo @binding = 1 wsf : array<float>   // W scales [d x n/32], f32 view
    @ssbo @binding = 2 wsh : array<float16> // ... the same buffer, halfword view (wscale_f16)
    @ssbo @binding = 3 x : array<float4>    // X [32 x n] f32 rows
    @ssbo @binding = 4 part : array<float>  // [ksplit x 32 x d] partial-C planes
    @uniform @binding = 5 kdim : uint       // n (K)
    @uniform @binding = 6 ndim : uint       // d (N)
    @uniform @binding = 7 s16 : uint
    @uniform @binding = 8 ksplit : uint
    @workgroup tw : float16[1024]           // W: 4 k-tiles x 4 row-tiles of 8x8, (k row, wrow col)
    @workgroup tx : float16[1024]           // X: 4 k-tiles x 4 token-tiles of 8x8, (token row, k col)

    [metal_kernel(name="metal_q8_gemm_b_sk_msl")]
    def metal_q8_gemm_b_sk {
        let lid = gl_LocalInvocationID.x
        let sg = gl_SubgroupID
        let ntileN = ndim / 32u
        let mBase = (gl_WorkGroupID.x / ntileN) * 32u
        let nBase = (gl_WorkGroupID.x % ntileN) * 32u
        let sl = gl_WorkGroupID.y
        var mc : simdgroup_float8x8[4]      // this simdgroup's 16x16 quadrant: 2 token x 2 wrow tiles
        var ma : simdgroup_half8x8[2]
        var mb : simdgroup_half8x8[2]
        let kdim4 = kdim / 4u
        let nkb = kdim / 32u
        let kbn = nkb / ksplit
        let kb0 = sl * kbn
        // staging split: thread handles row lr (0..31) x k-eighth il (0..3) — 8 elems of W AND X
        let lr = lid / 4u
        let il = lid % 4u
        let tW0 = (il * 4u + lr / 8u) * 64u + (lr % 8u)
        let tX0 = (il * 4u + lr / 8u) * 64u + (lr % 8u) * 8u
        let aQ = (sg % 2u) * 128u           // wrow tile-pair base within a k-tile slab
        let bQ = (sg / 2u) * 128u           // token tile-pair base
        var wp = unsafe(addr(wq[(nBase + lr) * kdim4 + kb0 * 8u + il * 2u]))
        var spf = unsafe(addr(wsf[(nBase + lr) * nkb + kb0]))
        var sph = unsafe(addr(wsh[(nBase + lr) * nkb + kb0]))
        var xp = unsafe(addr(x[(mBase + lr) * kdim4 + kb0 * 8u + il * 2u]))
        var kb = 0u
        while (kb < kbn) {
            let sA = s16 != 0u ? float(unsafe(sph[0])) : unsafe(spf[0])
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
            spf = unsafe(spf + 1u)
            sph = unsafe(sph + 1u)
            xp = unsafe(xp + 8u)
            kb++
        }
        let pbase = sl * 32u * ndim
        let row0 = mBase + (sg / 2u) * 16u
        let col0 = nBase + (sg % 2u) * 16u
        simdgroup_store(mc[0], part, pbase + row0 * ndim + col0, ndim)
        simdgroup_store(mc[1], part, pbase + row0 * ndim + (col0 + 8u), ndim)
        simdgroup_store(mc[2], part, pbase + (row0 + 8u) * ndim + col0, ndim)
        simdgroup_store(mc[3], part, pbase + (row0 + 8u) * ndim + (col0 + 8u), ndim)
    }
}

// the split-K reduce: y = sum of the ksplit partial planes (one thread per output element);
// total = 32 * d is both the element count and the plane stride
class MetalSkReduce {
    @ssbo @binding = 0 part : array<float>
    @ssbo @binding = 1 y : array<float>
    @uniform @binding = 2 total : uint
    @uniform @binding = 3 ksplit : uint

    [metal_kernel(name="metal_sk_reduce_msl")]
    def metal_sk_reduce {
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

// The 64-wide twin — the same v2 staging discipline on MulMm's 32(M) x 64(N) tile: half the
// threadgroups (d/64) at ~29% fewer instructions per element, so it wins exactly where the grid
// stays fat regardless — the classifier (vocab/64 threadgroups). Each simdgroup owns a 16x32
// quadrant (2 token x 4 wrow tiles); a thread stages 16 W elems (row x k-half) and 8 X elems.
class MetalQ8Gemm64B {
    @ssbo @binding = 0 wq : array<byte4>    // W [d x n] row-major Q8 quants, byte4 view
    @ssbo @binding = 1 wsf : array<float>   // W scales [d x n/32], f32 view
    @ssbo @binding = 2 wsh : array<float16> // ... the same buffer, halfword view (wscale_f16)
    @ssbo @binding = 3 x : array<float4>    // X [M' x n] f32 rows (M' = 32-padded B)
    @ssbo @binding = 4 y : array<float>     // Y [M' x d]
    @uniform @binding = 5 kdim : uint       // n (K)
    @uniform @binding = 6 ndim : uint       // d (N)
    @uniform @binding = 7 s16 : uint
    @workgroup tw : float16[2048]           // W: 4 k-tiles x 8 row-tiles of 8x8, (k row, wrow col)
    @workgroup tx : float16[1024]           // X: 4 k-tiles x 4 token-tiles of 8x8, (token row, k col)

    [metal_kernel(name="metal_q8_gemm64_b_msl")]
    def metal_q8_gemm64_b {
        let lid = gl_LocalInvocationID.x
        let sg = gl_SubgroupID
        let ntileN = ndim / 64u
        let mBase = (gl_WorkGroupID.x / ntileN) * 32u
        let nBase = (gl_WorkGroupID.x % ntileN) * 64u
        var mc : simdgroup_float8x8[8]      // this simdgroup's 16x32 quadrant: 2 token x 4 wrow tiles
        var ma : simdgroup_half8x8[4]
        var mb : simdgroup_half8x8[2]
        var va : float16[16]                // dequantized W staging regs (this thread's k-half)
        let kdim4 = kdim / 4u
        let nkb = kdim / 32u
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
        var spf = unsafe(addr(wsf[(nBase + lr0) * nkb]))
        var sph = unsafe(addr(wsh[(nBase + lr0) * nkb]))
        var xp = unsafe(addr(x[(mBase + lr1) * kdim4 + il1 * 2u]))
        var kb = 0u
        while (kb < nkb) {
            let sA = s16 != 0u ? float(unsafe(sph[0])) : unsafe(spf[0])
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
            spf = unsafe(spf + 1u)
            sph = unsafe(sph + 1u)
            xp = unsafe(xp + 8u)
            kb++
        }
        let row0 = mBase + (sg / 2u) * 16u
        let col0 = nBase + (sg % 2u) * 32u
        for [unroll_full] (i in range(8)) {
            simdgroup_store(mc[i], y, (row0 + uint(i / 4) * 8u) * ndim + col0 + uint(i % 4) * 8u, ndim)
        }
    }
}

// Batched single-query attention, single-tg form: grid (nheads x B) threadgroups — the same
// simdgroup-cooperative math as MetalSqAttnF16, but each row reads ITS mirror slice through the
// row table (per-row base + capacity + depth) and its roped q lives in the fused QKV buffer at
// stride qstride (= qkvd).
class MetalSqAttnBF16 {
    @ssbo @binding = 0 kc : array<float16>  // the whole mirror arena, f16 view
    @ssbo @binding = 1 vc : array<float16>
    @ssbo @binding = 2 q : array<float>     // [B x qstride] roped queries (the fused QKV rows)
    @ssbo @binding = 3 yo : array<float>    // [B x nheads*hs] attention out
    @ssbo @binding = 4 rt : array<uint4>    // per-row (k base, v base, cap_rows, cnt) — arena elems
    @uniform @binding = 5 lay : uint
    @uniform @binding = 6 kv_dim : uint
    @uniform @binding = 7 nheads : uint
    @uniform @binding = 8 kv_mul : uint
    @uniform @binding = 9 hs : uint
    @uniform @binding = 10 scale : float
    @uniform @binding = 11 qstride : uint
    @workgroup scores : float[128]
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_b16_msl")]
    def metal_sq_attn_b16 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let bb = wg / nheads
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let rrow = rt[bb]
        let cnt = rrow.w
        let kb0 = rrow.x + lay * rrow.z * kv_dim
        let vb0 = rrow.y + lay * rrow.z * kv_dim
        let khoff = (h / kv_mul) * hs
        let qoff = bb * qstride + h * hs
        let yoff = bb * (nheads * hs) + h * hs
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        var j = sg
        while (j < cnt) {
            let kb = kb0 + j * kv_dim + khoff + sl * epl
            var s = q0 * float(kc[kb])
            if (epl > 1u) {
                s += q1 * float(kc[kb + 1u])
            }
            if (epl > 2u) {
                s += q2 * float(kc[kb + 2u])
            }
            if (epl > 3u) {
                s += q3 * float(kc[kb + 3u])
            }
            s = simd_sum(s)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        var m = -3.0e38
        j = lid
        while (j < cnt) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()                           // red[] reused for the sum below
        var l = 0.0
        j = lid
        while (j < cnt) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= cnt) {
            acc0 += scores[j] * float(vc[vb0 + j * kv_dim + khoff + lid])
            acc1 += scores[j + 1u] * float(vc[vb0 + (j + 1u) * kv_dim + khoff + lid])
            acc2 += scores[j + 2u] * float(vc[vb0 + (j + 2u) * kv_dim + khoff + lid])
            acc3 += scores[j + 3u] * float(vc[vb0 + (j + 3u) * kv_dim + khoff + lid])
            j += 4u
        }
        while (j < cnt) {
            acc0 += scores[j] * float(vc[vb0 + j * kv_dim + khoff + lid])
            j++
        }
        yo[yoff + lid] = (acc0 + acc1 + acc2 + acc3) / tot
    }
}

// f32-mirror twin.
class MetalSqAttnBF32 {
    @ssbo @binding = 0 kc : array<float>
    @ssbo @binding = 1 vc : array<float>
    @ssbo @binding = 2 q : array<float>
    @ssbo @binding = 3 yo : array<float>
    @ssbo @binding = 4 rt : array<uint4>
    @uniform @binding = 5 lay : uint
    @uniform @binding = 6 kv_dim : uint
    @uniform @binding = 7 nheads : uint
    @uniform @binding = 8 kv_mul : uint
    @uniform @binding = 9 hs : uint
    @uniform @binding = 10 scale : float
    @uniform @binding = 11 qstride : uint
    @workgroup scores : float[128]
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_b32_msl")]
    def metal_sq_attn_b32 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let bb = wg / nheads
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let rrow = rt[bb]
        let cnt = rrow.w
        let kb0 = rrow.x + lay * rrow.z * kv_dim
        let vb0 = rrow.y + lay * rrow.z * kv_dim
        let khoff = (h / kv_mul) * hs
        let qoff = bb * qstride + h * hs
        let yoff = bb * (nheads * hs) + h * hs
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        var j = sg
        while (j < cnt) {
            let kb = kb0 + j * kv_dim + khoff + sl * epl
            var s = q0 * kc[kb]
            if (epl > 1u) {
                s += q1 * kc[kb + 1u]
            }
            if (epl > 2u) {
                s += q2 * kc[kb + 2u]
            }
            if (epl > 3u) {
                s += q3 * kc[kb + 3u]
            }
            s = simd_sum(s)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        var m = -3.0e38
        j = lid
        while (j < cnt) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()
        var l = 0.0
        j = lid
        while (j < cnt) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= cnt) {
            acc0 += scores[j] * vc[vb0 + j * kv_dim + khoff + lid]
            acc1 += scores[j + 1u] * vc[vb0 + (j + 1u) * kv_dim + khoff + lid]
            acc2 += scores[j + 2u] * vc[vb0 + (j + 2u) * kv_dim + khoff + lid]
            acc3 += scores[j + 3u] * vc[vb0 + (j + 3u) * kv_dim + khoff + lid]
            j += 4u
        }
        while (j < cnt) {
            acc0 += scores[j] * vc[vb0 + j * kv_dim + khoff + lid]
            j++
        }
        yo[yoff + lid] = (acc0 + acc1 + acc2 + acc3) / tot
    }
}

// Batched chunked split-K twin: grid (nheads x nchmax x B) — rows shallower than a chunk's start
// early-out whole-threadgroup (uniform per tg), and the comb below reads each row's own chunk
// count from the table, so ragged depths share one dispatch.
// q8_0-mirror batch twin (dual-view block reads — see MetalSqAttnQ8). Its row-table entries
// carry arena BYTE offsets (the driver fills per dtype); rowb = nblk*34 and every offset is
// even (16B-aligned slices), so the half-scale view indexes at byte/2 exactly.
class MetalSqAttnBQ8 {
    @ssbo @binding = 0 ksh : array<float16> // the whole mirror arena, half-scale view
    @ssbo @binding = 1 kqb : array<int8>    // the SAME arena, byte view
    @ssbo @binding = 2 vsh : array<float16>
    @ssbo @binding = 3 vqb : array<int8>
    @ssbo @binding = 4 q : array<float>     // [B x qstride] roped queries (the fused QKV rows)
    @ssbo @binding = 5 yo : array<float>    // [B x nheads*hs] attention out
    @ssbo @binding = 6 rt : array<uint4>    // per-row (k base BYTES, v base BYTES, cap_rows, cnt)
    @uniform @binding = 7 lay : uint
    @uniform @binding = 8 nblk : uint       // blocks per row = kv_dim / 32
    @uniform @binding = 9 nheads : uint
    @uniform @binding = 10 kv_mul : uint
    @uniform @binding = 11 hs : uint
    @uniform @binding = 12 scale : float
    @uniform @binding = 13 qstride : uint
    @workgroup scores : float[128]
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_b_q8_msl")]
    def metal_sq_attn_b_q8 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let bb = wg / nheads
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let rrow = rt[bb]
        let cnt = rrow.w
        let rowb = nblk * 34u
        let kb0b = rrow.x + lay * rrow.z * rowb   // layer slab base, BYTES (even)
        let vb0b = rrow.y + lay * rrow.z * rowb
        let kb0h = kb0b / 2u                      // ... and HALFWORDS for the scale view
        let vb0h = vb0b / 2u
        let khblk = (h / kv_mul) * (hs / 32u)
        let qoff = bb * qstride + h * hs
        let yoff = bb * (nheads * hs) + h * hs
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        let hb = (sl * epl) / 32u
        let ib = (sl * epl) % 32u
        var j = sg
        while (j < cnt) {
            let blk = j * nblk + khblk + hb
            let d = float(ksh[kb0h + blk * 17u])
            let kb = kb0b + blk * 34u + 2u + ib
            var s = q0 * float(kqb[kb])
            if (epl > 1u) {
                s += q1 * float(kqb[kb + 1u])
            }
            if (epl > 2u) {
                s += q2 * float(kqb[kb + 2u])
            }
            if (epl > 3u) {
                s += q3 * float(kqb[kb + 3u])
            }
            s = simd_sum(s * d)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        var m = -3.0e38
        j = lid
        while (j < cnt) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()                           // red[] reused for the sum below
        var l = 0.0
        j = lid
        while (j < cnt) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        let vblk0 = khblk + lid / 32u
        let vin = lid % 32u
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= cnt) {
            let b0 = j * nblk + vblk0
            let b1 = (j + 1u) * nblk + vblk0
            let b2 = (j + 2u) * nblk + vblk0
            let b3 = (j + 3u) * nblk + vblk0
            acc0 += scores[j] * float(vqb[vb0b + b0 * 34u + 2u + vin]) * float(vsh[vb0h + b0 * 17u])
            acc1 += scores[j + 1u] * float(vqb[vb0b + b1 * 34u + 2u + vin]) * float(vsh[vb0h + b1 * 17u])
            acc2 += scores[j + 2u] * float(vqb[vb0b + b2 * 34u + 2u + vin]) * float(vsh[vb0h + b2 * 17u])
            acc3 += scores[j + 3u] * float(vqb[vb0b + b3 * 34u + 2u + vin]) * float(vsh[vb0h + b3 * 17u])
            j += 4u
        }
        while (j < cnt) {
            let b0 = j * nblk + vblk0
            acc0 += scores[j] * float(vqb[vb0b + b0 * 34u + 2u + vin]) * float(vsh[vb0h + b0 * 17u])
            j++
        }
        yo[yoff + lid] = (acc0 + acc1 + acc2 + acc3) / tot
    }
}

class MetalSqAttnPartBF16 {
    @ssbo @binding = 0 kc : array<float16>
    @ssbo @binding = 1 vc : array<float16>
    @ssbo @binding = 2 q : array<float>
    @ssbo @binding = 3 part : array<float>  // [B x nheads x nchmax x (hs+2)]
    @ssbo @binding = 4 rt : array<uint4>
    @uniform @binding = 5 lay : uint
    @uniform @binding = 6 kv_dim : uint
    @uniform @binding = 7 nheads : uint
    @uniform @binding = 8 kv_mul : uint
    @uniform @binding = 9 hs : uint
    @uniform @binding = 10 scale : float
    @uniform @binding = 11 qstride : uint
    @uniform @binding = 12 nchmax : uint
    @workgroup scores : float[64]   // == ATTN_CHUNK_ROWS
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_part_b16_msl")]
    def metal_sq_attn_part_b16 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let ci = (wg / nheads) % nchmax
        let bb = wg / (nheads * nchmax)
        let rrow = rt[bb]
        let cnt = rrow.w
        let c0 = ci * 64u
        if (c0 >= cnt) {   // this row is shallower than the chunk — whole tg exits together
            return
        }
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let kb0 = rrow.x + lay * rrow.z * kv_dim
        let vb0 = rrow.y + lay * rrow.z * kv_dim
        let khoff = (h / kv_mul) * hs
        let qoff = bb * qstride + h * hs
        let c1 = min(c0 + 64u, cnt)
        let nrows = c1 - c0
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        var j = sg
        while (j < nrows) {
            let kb = kb0 + (c0 + j) * kv_dim + khoff + sl * epl
            var s = q0 * float(kc[kb])
            if (epl > 1u) {
                s += q1 * float(kc[kb + 1u])
            }
            if (epl > 2u) {
                s += q2 * float(kc[kb + 2u])
            }
            if (epl > 3u) {
                s += q3 * float(kc[kb + 3u])
            }
            s = simd_sum(s)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        var m = -3.0e38
        j = lid
        while (j < nrows) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()                           // red[] reused for the sum below
        var l = 0.0
        j = lid
        while (j < nrows) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= nrows) {
            acc0 += scores[j] * float(vc[vb0 + (c0 + j) * kv_dim + khoff + lid])
            acc1 += scores[j + 1u] * float(vc[vb0 + (c0 + j + 1u) * kv_dim + khoff + lid])
            acc2 += scores[j + 2u] * float(vc[vb0 + (c0 + j + 2u) * kv_dim + khoff + lid])
            acc3 += scores[j + 3u] * float(vc[vb0 + (c0 + j + 3u) * kv_dim + khoff + lid])
            j += 4u
        }
        while (j < nrows) {
            acc0 += scores[j] * float(vc[vb0 + (c0 + j) * kv_dim + khoff + lid])
            j++
        }
        let pbase = ((bb * nheads + h) * nchmax + ci) * (hs + 2u)
        part[pbase + lid] = acc0 + acc1 + acc2 + acc3
        if (lid == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// f32-mirror twin.
class MetalSqAttnPartBF32 {
    @ssbo @binding = 0 kc : array<float>
    @ssbo @binding = 1 vc : array<float>
    @ssbo @binding = 2 q : array<float>
    @ssbo @binding = 3 part : array<float>
    @ssbo @binding = 4 rt : array<uint4>
    @uniform @binding = 5 lay : uint
    @uniform @binding = 6 kv_dim : uint
    @uniform @binding = 7 nheads : uint
    @uniform @binding = 8 kv_mul : uint
    @uniform @binding = 9 hs : uint
    @uniform @binding = 10 scale : float
    @uniform @binding = 11 qstride : uint
    @uniform @binding = 12 nchmax : uint
    @workgroup scores : float[64]
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_part_b32_msl")]
    def metal_sq_attn_part_b32 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let ci = (wg / nheads) % nchmax
        let bb = wg / (nheads * nchmax)
        let rrow = rt[bb]
        let cnt = rrow.w
        let c0 = ci * 64u
        if (c0 >= cnt) {
            return
        }
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let kb0 = rrow.x + lay * rrow.z * kv_dim
        let vb0 = rrow.y + lay * rrow.z * kv_dim
        let khoff = (h / kv_mul) * hs
        let qoff = bb * qstride + h * hs
        let c1 = min(c0 + 64u, cnt)
        let nrows = c1 - c0
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        var j = sg
        while (j < nrows) {
            let kb = kb0 + (c0 + j) * kv_dim + khoff + sl * epl
            var s = q0 * kc[kb]
            if (epl > 1u) {
                s += q1 * kc[kb + 1u]
            }
            if (epl > 2u) {
                s += q2 * kc[kb + 2u]
            }
            if (epl > 3u) {
                s += q3 * kc[kb + 3u]
            }
            s = simd_sum(s)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        var m = -3.0e38
        j = lid
        while (j < nrows) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()
        var l = 0.0
        j = lid
        while (j < nrows) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= nrows) {
            acc0 += scores[j] * vc[vb0 + (c0 + j) * kv_dim + khoff + lid]
            acc1 += scores[j + 1u] * vc[vb0 + (c0 + j + 1u) * kv_dim + khoff + lid]
            acc2 += scores[j + 2u] * vc[vb0 + (c0 + j + 2u) * kv_dim + khoff + lid]
            acc3 += scores[j + 3u] * vc[vb0 + (c0 + j + 3u) * kv_dim + khoff + lid]
            j += 4u
        }
        while (j < nrows) {
            acc0 += scores[j] * vc[vb0 + (c0 + j) * kv_dim + khoff + lid]
            j++
        }
        let pbase = ((bb * nheads + h) * nchmax + ci) * (hs + 2u)
        part[pbase + lid] = acc0 + acc1 + acc2 + acc3
        if (lid == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// q8_0-mirror batched chunked twin — byte-offset row table + dual-view block reads; partials
// stay f32, so MetalSqAttnCombB serves this form unchanged.
class MetalSqAttnPartBQ8 {
    @ssbo @binding = 0 ksh : array<float16>
    @ssbo @binding = 1 kqb : array<int8>
    @ssbo @binding = 2 vsh : array<float16>
    @ssbo @binding = 3 vqb : array<int8>
    @ssbo @binding = 4 q : array<float>
    @ssbo @binding = 5 part : array<float>  // [B x nheads x nchmax x (hs+2)]
    @ssbo @binding = 6 rt : array<uint4>    // per-row (k base BYTES, v base BYTES, cap_rows, cnt)
    @uniform @binding = 7 lay : uint
    @uniform @binding = 8 nblk : uint
    @uniform @binding = 9 nheads : uint
    @uniform @binding = 10 kv_mul : uint
    @uniform @binding = 11 hs : uint
    @uniform @binding = 12 scale : float
    @uniform @binding = 13 qstride : uint
    @uniform @binding = 14 nchmax : uint
    @workgroup scores : float[64]   // == ATTN_CHUNK_ROWS
    @workgroup red : float[4]

    [metal_kernel(name="metal_sq_attn_part_b_q8_msl")]
    def metal_sq_attn_part_b_q8 {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let ci = (wg / nheads) % nchmax
        let bb = wg / (nheads * nchmax)
        let rrow = rt[bb]
        let cnt = rrow.w
        let c0 = ci * 64u
        if (c0 >= cnt) {   // this row is shallower than the chunk — whole tg exits together
            return
        }
        let lid = gl_LocalInvocationID.x
        let nt = gl_WorkGroupSize.x
        let rowb = nblk * 34u
        let kb0b = rrow.x + lay * rrow.z * rowb
        let vb0b = rrow.y + lay * rrow.z * rowb
        let kb0h = kb0b / 2u
        let vb0h = vb0b / 2u
        let khblk = (h / kv_mul) * (hs / 32u)
        let qoff = bb * qstride + h * hs
        let c1 = min(c0 + 64u, cnt)
        let nrows = c1 - c0
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let epl = hs / 32u
        let qb = qoff + sl * epl
        let q0 = q[qb]
        let q1 = epl > 1u ? q[qb + 1u] : 0.0
        let q2 = epl > 2u ? q[qb + 2u] : 0.0
        let q3 = epl > 3u ? q[qb + 3u] : 0.0
        let hb = (sl * epl) / 32u
        let ib = (sl * epl) % 32u
        var j = sg
        while (j < nrows) {
            let blk = (c0 + j) * nblk + khblk + hb
            let d = float(ksh[kb0h + blk * 17u])
            let kb = kb0b + blk * 34u + 2u + ib
            var s = q0 * float(kqb[kb])
            if (epl > 1u) {
                s += q1 * float(kqb[kb + 1u])
            }
            if (epl > 2u) {
                s += q2 * float(kqb[kb + 2u])
            }
            if (epl > 3u) {
                s += q3 * float(kqb[kb + 3u])
            }
            s = simd_sum(s * d)
            if (sl == 0u) {
                scores[j] = s * scale
            }
            j += gl_NumSubgroups
        }
        barrier()
        var m = -3.0e38
        j = lid
        while (j < nrows) {
            m = max(m, scores[j])
            j += nt
        }
        m = max(m, simd_shuffle_xor(m, 16u))
        m = max(m, simd_shuffle_xor(m, 8u))
        m = max(m, simd_shuffle_xor(m, 4u))
        m = max(m, simd_shuffle_xor(m, 2u))
        m = max(m, simd_shuffle_xor(m, 1u))
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = m
        }
        barrier()
        var mm = red[0]
        var sgi = 1u
        while (sgi < gl_NumSubgroups) {
            mm = max(mm, red[sgi])
            sgi++
        }
        barrier()                           // red[] reused for the sum below
        var l = 0.0
        j = lid
        while (j < nrows) {
            let w = exp(scores[j] - mm)
            scores[j] = w
            l += w
            j += nt
        }
        l += simd_shuffle_xor(l, 16u)
        l += simd_shuffle_xor(l, 8u)
        l += simd_shuffle_xor(l, 4u)
        l += simd_shuffle_xor(l, 2u)
        l += simd_shuffle_xor(l, 1u)
        if (gl_SubgroupInvocationID == 0u) {
            red[gl_SubgroupID] = l
        }
        barrier()
        var tot = red[0]
        sgi = 1u
        while (sgi < gl_NumSubgroups) {
            tot += red[sgi]
            sgi++
        }
        let vblk0 = khblk + lid / 32u
        let vin = lid % 32u
        var acc0 = 0.0
        var acc1 = 0.0
        var acc2 = 0.0
        var acc3 = 0.0
        j = 0u
        while (j + 4u <= nrows) {
            let b0 = (c0 + j) * nblk + vblk0
            let b1 = (c0 + j + 1u) * nblk + vblk0
            let b2 = (c0 + j + 2u) * nblk + vblk0
            let b3 = (c0 + j + 3u) * nblk + vblk0
            acc0 += scores[j] * float(vqb[vb0b + b0 * 34u + 2u + vin]) * float(vsh[vb0h + b0 * 17u])
            acc1 += scores[j + 1u] * float(vqb[vb0b + b1 * 34u + 2u + vin]) * float(vsh[vb0h + b1 * 17u])
            acc2 += scores[j + 2u] * float(vqb[vb0b + b2 * 34u + 2u + vin]) * float(vsh[vb0h + b2 * 17u])
            acc3 += scores[j + 3u] * float(vqb[vb0b + b3 * 34u + 2u + vin]) * float(vsh[vb0h + b3 * 17u])
            j += 4u
        }
        while (j < nrows) {
            let b0 = (c0 + j) * nblk + vblk0
            acc0 += scores[j] * float(vqb[vb0b + b0 * 34u + 2u + vin]) * float(vsh[vb0h + b0 * 17u])
            j++
        }
        let pbase = ((bb * nheads + h) * nchmax + ci) * (hs + 2u)
        part[pbase + lid] = acc0 + acc1 + acc2 + acc3
        if (lid == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// Batched cross-chunk combine: grid (nheads x B) — each row combines its OWN chunk count
// (from the table), holes past a shallow row's last chunk are never read.
class MetalSqAttnCombB {
    @ssbo @binding = 0 part : array<float>
    @ssbo @binding = 1 yo : array<float>
    @ssbo @binding = 2 rt : array<uint4>
    @uniform @binding = 3 nchmax : uint
    @uniform @binding = 4 nheads : uint
    @uniform @binding = 5 hs : uint

    [metal_kernel(name="metal_sq_attn_comb_b_msl")]
    def metal_sq_attn_comb_b {
        let wg = gl_WorkGroupID.x
        let h = wg % nheads
        let bb = wg / nheads
        let lid = gl_LocalInvocationID.x
        let cnt = rt[bb].w
        let nch = (cnt + 63u) / 64u
        let base = (bb * nheads + h) * nchmax
        var mm = -3.0e38
        var ci = 0u
        while (ci < nch) {
            mm = max(mm, part[(base + ci) * (hs + 2u) + hs])
            ci++
        }
        var acc = 0.0
        var l = 0.0
        ci = 0u
        while (ci < nch) {
            let pb = (base + ci) * (hs + 2u)
            let w = exp(part[pb + hs] - mm)
            acc += part[pb + lid] * w
            l += part[pb + hs + 1u] * w
            ci++
        }
        yo[bb * (nheads * hs) + h * hs + lid] = acc / l
    }
}

// Fused single-pass batched attention (the walkthrough kernel-2 winner, <= llama.cpp fa_vec at
// every depth 512-16K in the lab): ONE simdgroup per (row, head, 32-position slice-group) — a
// budget of min(ceil(cnt/32), 16) sgs strides the slices; K keeps each position's score in the
// owning lane's REGISTER (no divergent store), the slice softmax is a register butterfly, V
// accumulates a half4 column group per lane with the O accumulator in a LANE-PRIVATE tgmem slot
// (their so4 discipline — zero barriers), and slices merge online (branch-free exp-rescale).
// hs == 128 only (32 lanes x half4 = one head); grid (nheads, ceil(nsgs/4), rows) HEAD-FASTEST
// (consecutive tgs share position rows — SLC locality). Entries are (hs+4) floats,
// float4-aligned, combined by MetalSqAttnCombD.
class MetalSqAttnDF16 {
    @ssbo @binding = 0 kc : array<half4>
    @ssbo @binding = 1 vc : array<half4>
    @ssbo @binding = 2 q : array<float4>
    @ssbo @binding = 3 part : array<float>
    @ssbo @binding = 4 rt : array<uint4>
    @uniform @binding = 5 lay : uint
    @uniform @binding = 6 kv_dim : uint
    @uniform @binding = 7 nheads : uint
    @uniform @binding = 8 kv_mul : uint
    @uniform @binding = 9 hs : uint
    @uniform @binding = 10 scale : float
    @uniform @binding = 11 qstride : uint
    @uniform @binding = 12 nsgs : uint
    @workgroup wacc : float[512]

    [metal_kernel(name="metal_sq_attn_d16_msl")]
    def metal_sq_attn_d16 {
        let h = gl_WorkGroupID.x
        let bb = gl_WorkGroupID.z
        let gsg = gl_WorkGroupID.y * 4u + gl_SubgroupID
        let rrow = rt[bb]
        let cnt = rrow.w
        if (gsg >= nsgs || gsg * 32u >= cnt) {
            return
        }
        let sl = gl_SubgroupInvocationID
        let kb0 = rrow.x + lay * rrow.z * kv_dim
        let vb0 = rrow.y + lay * rrow.z * kv_dim
        let khoff = (h / kv_mul) * hs
        let qoff = bb * qstride + h * hs
        let kv4 = kv_dim / 4u
        let q4 = float4(q[qoff / 4u + sl])
        let wo = gl_SubgroupID * 128u + sl * 4u
        wacc[wo] = 0.0
        wacc[wo + 1u] = 0.0
        wacc[wo + 2u] = 0.0
        wacc[wo + 3u] = 0.0
        var mm = -3.0e38
        var tot = 0.0
        var ss = gsg
        while (ss * 32u < cnt) {
            let c0 = ss * 32u
            let nrows = min(c0 + 32u, cnt) - c0
            var keep = -3.0e38
            var kp = unsafe(addr(kc[(kb0 + c0 * kv_dim + khoff) / 4u + sl]))
            var vp = unsafe(addr(vc[(vb0 + c0 * kv_dim + khoff) / 4u + sl]))
            var bacc = float4(0.0)
            if (nrows == 32u) {
                for [unroll_full] (i in range(32)) {
                    let p = q4 * float4(unsafe(kp[uint(i) * kv4]))
                    let sc = simd_sum(p.x + p.y + p.z + p.w)
                    keep = uint(i) == sl ? sc : keep
                }
            } else {
                var j = 0u
                while (j < nrows) {
                    let p = q4 * float4(unsafe(kp[0]))
                    let sc = simd_sum(p.x + p.y + p.z + p.w)
                    keep = j == sl ? sc : keep
                    kp = unsafe(kp + kv4)
                    j++
                }
            }
            keep *= scale   // invalid tail lanes stay hugely negative -> weight 0
            var bm = keep
            bm = max(bm, simd_shuffle_xor(bm, 16u))
            bm = max(bm, simd_shuffle_xor(bm, 8u))
            bm = max(bm, simd_shuffle_xor(bm, 4u))
            bm = max(bm, simd_shuffle_xor(bm, 2u))
            bm = max(bm, simd_shuffle_xor(bm, 1u))
            let w = exp(keep - bm)
            let btot = simd_sum(w)
            if (nrows == 32u) {
                for [unroll_full] (i in range(32)) {
                    bacc += simd_shuffle(w, uint(i)) * float4(unsafe(vp[uint(i) * kv4]))
                }
            } else {
                var j = 0u
                while (j < nrows) {
                    bacc += simd_shuffle(w, j) * float4(unsafe(vp[0]))
                    vp = unsafe(vp + kv4)
                    j++
                }
            }
            // online merge into the running slice-group state (first pass: r1 = exp(-inf) = 0)
            let nm = max(mm, bm)
            let r1 = exp(mm - nm)
            let r2 = exp(bm - nm)
            wacc[wo] = wacc[wo] * r1 + bacc.x * r2
            wacc[wo + 1u] = wacc[wo + 1u] * r1 + bacc.y * r2
            wacc[wo + 2u] = wacc[wo + 2u] * r1 + bacc.z * r2
            wacc[wo + 3u] = wacc[wo + 3u] * r1 + bacc.w * r2
            tot = tot * r1 + btot * r2
            mm = nm
            ss += nsgs
        }
        let pbase = ((bb * nheads + h) * nsgs + gsg) * (hs + 4u)
        let po = pbase + sl * 4u
        part[po] = wacc[wo]
        part[po + 1u] = wacc[wo + 1u]
        part[po + 2u] = wacc[wo + 2u]
        part[po + 3u] = wacc[wo + 3u]
        if (sl == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// f32-mirror twin — float4 view, no conversion.
class MetalSqAttnDF32 {
    @ssbo @binding = 0 kc : array<float4>
    @ssbo @binding = 1 vc : array<float4>
    @ssbo @binding = 2 q : array<float4>
    @ssbo @binding = 3 part : array<float>
    @ssbo @binding = 4 rt : array<uint4>
    @uniform @binding = 5 lay : uint
    @uniform @binding = 6 kv_dim : uint
    @uniform @binding = 7 nheads : uint
    @uniform @binding = 8 kv_mul : uint
    @uniform @binding = 9 hs : uint
    @uniform @binding = 10 scale : float
    @uniform @binding = 11 qstride : uint
    @uniform @binding = 12 nsgs : uint
    @workgroup wacc : float[512]

    [metal_kernel(name="metal_sq_attn_d32_msl")]
    def metal_sq_attn_d32 {
        let h = gl_WorkGroupID.x
        let bb = gl_WorkGroupID.z
        let gsg = gl_WorkGroupID.y * 4u + gl_SubgroupID
        let rrow = rt[bb]
        let cnt = rrow.w
        if (gsg >= nsgs || gsg * 32u >= cnt) {
            return
        }
        let sl = gl_SubgroupInvocationID
        let kb0 = rrow.x + lay * rrow.z * kv_dim
        let vb0 = rrow.y + lay * rrow.z * kv_dim
        let khoff = (h / kv_mul) * hs
        let qoff = bb * qstride + h * hs
        let kv4 = kv_dim / 4u
        let q4 = float4(q[qoff / 4u + sl])
        let wo = gl_SubgroupID * 128u + sl * 4u
        wacc[wo] = 0.0
        wacc[wo + 1u] = 0.0
        wacc[wo + 2u] = 0.0
        wacc[wo + 3u] = 0.0
        var mm = -3.0e38
        var tot = 0.0
        var ss = gsg
        while (ss * 32u < cnt) {
            let c0 = ss * 32u
            let nrows = min(c0 + 32u, cnt) - c0
            var keep = -3.0e38
            var kp = unsafe(addr(kc[(kb0 + c0 * kv_dim + khoff) / 4u + sl]))
            var vp = unsafe(addr(vc[(vb0 + c0 * kv_dim + khoff) / 4u + sl]))
            var bacc = float4(0.0)
            if (nrows == 32u) {
                for [unroll_full] (i in range(32)) {
                    let p = q4 * unsafe(kp[uint(i) * kv4])
                    let sc = simd_sum(p.x + p.y + p.z + p.w)
                    keep = uint(i) == sl ? sc : keep
                }
            } else {
                var j = 0u
                while (j < nrows) {
                    let p = q4 * unsafe(kp[0])
                    let sc = simd_sum(p.x + p.y + p.z + p.w)
                    keep = j == sl ? sc : keep
                    kp = unsafe(kp + kv4)
                    j++
                }
            }
            keep *= scale
            var bm = keep
            bm = max(bm, simd_shuffle_xor(bm, 16u))
            bm = max(bm, simd_shuffle_xor(bm, 8u))
            bm = max(bm, simd_shuffle_xor(bm, 4u))
            bm = max(bm, simd_shuffle_xor(bm, 2u))
            bm = max(bm, simd_shuffle_xor(bm, 1u))
            let w = exp(keep - bm)
            let btot = simd_sum(w)
            if (nrows == 32u) {
                for [unroll_full] (i in range(32)) {
                    bacc += simd_shuffle(w, uint(i)) * unsafe(vp[uint(i) * kv4])
                }
            } else {
                var j = 0u
                while (j < nrows) {
                    bacc += simd_shuffle(w, j) * unsafe(vp[0])
                    vp = unsafe(vp + kv4)
                    j++
                }
            }
            let nm = max(mm, bm)
            let r1 = exp(mm - nm)
            let r2 = exp(bm - nm)
            wacc[wo] = wacc[wo] * r1 + bacc.x * r2
            wacc[wo + 1u] = wacc[wo + 1u] * r1 + bacc.y * r2
            wacc[wo + 2u] = wacc[wo + 2u] * r1 + bacc.z * r2
            wacc[wo + 3u] = wacc[wo + 3u] * r1 + bacc.w * r2
            tot = tot * r1 + btot * r2
            mm = nm
            ss += nsgs
        }
        let pbase = ((bb * nheads + h) * nsgs + gsg) * (hs + 4u)
        let po = pbase + sl * 4u
        part[po] = wacc[wo]
        part[po + 1u] = wacc[wo + 1u]
        part[po + 2u] = wacc[wo + 2u]
        part[po + 3u] = wacc[wo + 3u]
        if (sl == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// Cross-slice combine for the D entries: PARALLEL over entries (4 sgs online-merge their
// subsets with float4 loads — m and l arrive in ONE float4), a single tgmem cross-sg merge.
class MetalSqAttnCombD {
    @ssbo @binding = 0 part : array<float4>
    @ssbo @binding = 1 yo : array<float>
    @ssbo @binding = 2 rt : array<uint4>
    @uniform @binding = 3 nchmax : uint
    @uniform @binding = 4 nheads : uint
    @uniform @binding = 5 hs : uint
    @workgroup wacc : float[512]
    @workgroup wml : float[8]

    [metal_kernel(name="metal_sq_attn_comb_d_msl")]
    def metal_sq_attn_comb_d {
        let h = gl_WorkGroupID.x
        let bb = gl_WorkGroupID.y
        let sg = gl_SubgroupID
        let sl = gl_SubgroupInvocationID
        let lid = gl_LocalInvocationID.x
        let cnt = rt[bb].w
        let nch = min((cnt + 31u) / 32u, nchmax)
        let s4 = (hs + 4u) / 4u
        let base4 = (bb * nheads + h) * nchmax * s4
        var mm = -3.0e38
        var tot = 0.0
        var acc4 = float4(0.0)
        var e = sg
        while (e < nch) {
            let eb = base4 + e * s4
            let ml = float4(part[eb + hs / 4u])
            let av = float4(part[eb + sl])
            let nm = max(mm, ml.x)
            let r1 = exp(mm - nm)
            let r2 = exp(ml.x - nm)
            acc4 = acc4 * r1 + av * r2
            tot = tot * r1 + ml.y * r2
            mm = nm
            e += gl_NumSubgroups
        }
        let wo = sg * 128u + sl * 4u
        wacc[wo] = acc4.x
        wacc[wo + 1u] = acc4.y
        wacc[wo + 2u] = acc4.z
        wacc[wo + 3u] = acc4.w
        if (sl == 0u) {
            wml[sg * 2u] = mm
            wml[sg * 2u + 1u] = tot
        }
        barrier()
        // final 4-way merge, one lane per output column
        let m0 = wml[0]
        let m1 = wml[2]
        let m2 = wml[4]
        let m3 = wml[6]
        let gm = max(max(m0, m1), max(m2, m3))
        let r0 = exp(m0 - gm)
        let r1 = exp(m1 - gm)
        let r2 = exp(m2 - gm)
        let r3 = exp(m3 - gm)
        let l = wml[1] * r0 + wml[3] * r1 + wml[5] * r2 + wml[7] * r3
        let acc = wacc[lid] * r0 + wacc[128u + lid] * r1 + wacc[256u + lid] * r2 + wacc[384u + lid] * r3
        yo[bb * (nheads * hs) + h * hs + lid] = acc / l
    }
}

// Batched fused rope + KV-row store: grid (pairs x B) — row b's q pairs rope in place in ITS
// fused QKV row (stride qstride), its k pairs rope into ITS mirror row (table base + capacity,
// depth cnt ⇒ the new row lands at cnt-1) with the ±65504 clamp, v pairs copy alongside. The
// rope row is row b's own (positions differ across the batch).
class MetalRopeStoreBF16 {
    @ssbo @binding = 0 qk : array<float>    // [B x qstride] fused QKV rows — q ropes in place
    @ssbo @binding = 1 kd : array<float16>  // the whole mirror arena, f16 view (k dest)
    @ssbo @binding = 2 vd : array<float16>
    @ssbo @binding = 3 tcos : array<float>  // [B x head_size/2] per-row rope rows
    @ssbo @binding = 4 tsin : array<float>
    @ssbo @binding = 5 rt : array<uint4>
    @uniform @binding = 6 lay : uint
    @uniform @binding = 7 kv_dim : uint
    @uniform @binding = 8 qd : uint
    @uniform @binding = 9 head_size : uint
    @uniform @binding = 10 npairs_q : uint  // qd/2 — the q half of the pair grid
    @uniform @binding = 11 npairs : uint    // + kvd/2 — grid guard
    @uniform @binding = 12 qstride : uint

    [metal_kernel(name="metal_rope_store_b16_msl")]
    def metal_rope_store_b16 {
        let gid = gl_GlobalInvocationID.x
        if (gid >= npairs) {
            return
        }
        let bb = gl_WorkGroupID.y
        let rrow = rt[bb]
        let qbase = bb * qstride
        let half = head_size / 2u
        let isK = gid >= npairs_q
        let lgid = isK ? gid - npairs_q : gid
        let j = lgid % half
        let i = (lgid / half) * head_size + 2u * j
        let fcr = tcos[bb * half + j]
        let fci = tsin[bb * half + j]
        if (isK) {
            let kdst = rrow.x + (lay * rrow.z + (rrow.w - 1u)) * kv_dim
            let vdst = rrow.y + (lay * rrow.z + (rrow.w - 1u)) * kv_dim
            let v0 = qk[qbase + qd + i]
            let v1 = qk[qbase + qd + i + 1u]
            let r0 = v0 * fcr - v1 * fci
            let r1 = v0 * fci + v1 * fcr
            kd[kdst + i] = float16(min(max(r0, -65504.0), 65504.0))   // nolint:STYLE019 — the MSL emitter lowers min/max only, no clamp
            kd[kdst + i + 1u] = float16(min(max(r1, -65504.0), 65504.0))   // nolint:STYLE019
            vd[vdst + i] = float16(min(max(qk[qbase + qd + kv_dim + i], -65504.0), 65504.0))   // nolint:STYLE019
            vd[vdst + i + 1u] = float16(min(max(qk[qbase + qd + kv_dim + i + 1u], -65504.0), 65504.0))   // nolint:STYLE019
        } else {
            let v0 = qk[qbase + i]
            let v1 = qk[qbase + i + 1u]
            qk[qbase + i] = v0 * fcr - v1 * fci
            qk[qbase + i + 1u] = v0 * fci + v1 * fcr
        }
    }
}

// f32-mirror twin — straight stores, no clamp.
class MetalRopeStoreBF32 {
    @ssbo @binding = 0 qk : array<float>
    @ssbo @binding = 1 kd : array<float>
    @ssbo @binding = 2 vd : array<float>
    @ssbo @binding = 3 tcos : array<float>
    @ssbo @binding = 4 tsin : array<float>
    @ssbo @binding = 5 rt : array<uint4>
    @uniform @binding = 6 lay : uint
    @uniform @binding = 7 kv_dim : uint
    @uniform @binding = 8 qd : uint
    @uniform @binding = 9 head_size : uint
    @uniform @binding = 10 npairs_q : uint
    @uniform @binding = 11 npairs : uint
    @uniform @binding = 12 qstride : uint

    [metal_kernel(name="metal_rope_store_b32_msl")]
    def metal_rope_store_b32 {
        let gid = gl_GlobalInvocationID.x
        if (gid >= npairs) {
            return
        }
        let bb = gl_WorkGroupID.y
        let rrow = rt[bb]
        let qbase = bb * qstride
        let half = head_size / 2u
        let isK = gid >= npairs_q
        let lgid = isK ? gid - npairs_q : gid
        let j = lgid % half
        let i = (lgid / half) * head_size + 2u * j
        let fcr = tcos[bb * half + j]
        let fci = tsin[bb * half + j]
        if (isK) {
            let kdst = rrow.x + (lay * rrow.z + (rrow.w - 1u)) * kv_dim
            let vdst = rrow.y + (lay * rrow.z + (rrow.w - 1u)) * kv_dim
            let v0 = qk[qbase + qd + i]
            let v1 = qk[qbase + qd + i + 1u]
            kd[kdst + i] = v0 * fcr - v1 * fci
            kd[kdst + i + 1u] = v0 * fci + v1 * fcr
            vd[vdst + i] = qk[qbase + qd + kv_dim + i]
            vd[vdst + i + 1u] = qk[qbase + qd + kv_dim + i + 1u]
        } else {
            let v0 = qk[qbase + i]
            let v1 = qk[qbase + i + 1u]
            qk[qbase + i] = v0 * fcr - v1 * fci
            qk[qbase + i + 1u] = v0 * fci + v1 * fcr
        }
    }
}

// q8_0-mirror partD twin (hs == 128 only, the DF16 shape with dual-view block reads): one sg per
// (row, head, 32-position slice), online slice merge, byte-offset row table. K/V loads are 4
// scalar int8s + the block scale per position (34B blocks break half4 alignment); the block
// scale folds into the lane product before the simd_sum.
class MetalSqAttnDQ8 {
    @ssbo @binding = 0 ksh : array<float16>
    @ssbo @binding = 1 kqb : array<int8>
    @ssbo @binding = 2 vsh : array<float16>
    @ssbo @binding = 3 vqb : array<int8>
    @ssbo @binding = 4 q : array<float4>
    @ssbo @binding = 5 part : array<float>
    @ssbo @binding = 6 rt : array<uint4>    // per-row (k base BYTES, v base BYTES, cap_rows, cnt)
    @uniform @binding = 7 lay : uint
    @uniform @binding = 8 nblk : uint
    @uniform @binding = 9 nheads : uint
    @uniform @binding = 10 kv_mul : uint
    @uniform @binding = 11 hs : uint
    @uniform @binding = 12 scale : float
    @uniform @binding = 13 qstride : uint
    @uniform @binding = 14 nsgs : uint
    @workgroup wacc : float[512]

    [metal_kernel(name="metal_sq_attn_d_q8_msl")]
    def metal_sq_attn_d_q8 {
        let h = gl_WorkGroupID.x
        let bb = gl_WorkGroupID.z
        let gsg = gl_WorkGroupID.y * 4u + gl_SubgroupID
        let rrow = rt[bb]
        let cnt = rrow.w
        if (gsg >= nsgs || gsg * 32u >= cnt) {
            return
        }
        let sl = gl_SubgroupInvocationID
        let rowb = nblk * 34u
        let kb0b = rrow.x + lay * rrow.z * rowb
        let vb0b = rrow.y + lay * rrow.z * rowb
        let kb0h = kb0b / 2u
        let vb0h = vb0b / 2u
        let khblk = (h / kv_mul) * (hs / 32u)
        let qoff = bb * qstride + h * hs
        let q4 = float4(q[qoff / 4u + sl])
        let blkoff = khblk + sl / 8u            // this lane's block within the row
        let inb = (sl % 8u) * 4u                // ... and its 4 quants' offset inside it
        let wo = gl_SubgroupID * 128u + sl * 4u
        wacc[wo] = 0.0
        wacc[wo + 1u] = 0.0
        wacc[wo + 2u] = 0.0
        wacc[wo + 3u] = 0.0
        var mm = -3.0e38
        var tot = 0.0
        var ss = gsg
        while (ss * 32u < cnt) {
            let c0 = ss * 32u
            let nrows = min(c0 + 32u, cnt) - c0
            var keep = -3.0e38
            var bacc = float4(0.0)
            if (nrows == 32u) {
                for [unroll_full] (i in range(32)) {
                    let blk = (c0 + uint(i)) * nblk + blkoff
                    let kb = kb0b + blk * 34u + 2u + inb
                    let dq = float4(float(kqb[kb]), float(kqb[kb + 1u]), float(kqb[kb + 2u]), float(kqb[kb + 3u]))
                    let p = q4 * dq
                    let sc = simd_sum((p.x + p.y + p.z + p.w) * float(ksh[kb0h + blk * 17u]))
                    keep = uint(i) == sl ? sc : keep
                }
            } else {
                var j = 0u
                while (j < nrows) {
                    let blk = (c0 + j) * nblk + blkoff
                    let kb = kb0b + blk * 34u + 2u + inb
                    let dq = float4(float(kqb[kb]), float(kqb[kb + 1u]), float(kqb[kb + 2u]), float(kqb[kb + 3u]))
                    let p = q4 * dq
                    let sc = simd_sum((p.x + p.y + p.z + p.w) * float(ksh[kb0h + blk * 17u]))
                    keep = j == sl ? sc : keep
                    j++
                }
            }
            keep *= scale   // invalid tail lanes stay hugely negative -> weight 0
            var bm = keep
            bm = max(bm, simd_shuffle_xor(bm, 16u))
            bm = max(bm, simd_shuffle_xor(bm, 8u))
            bm = max(bm, simd_shuffle_xor(bm, 4u))
            bm = max(bm, simd_shuffle_xor(bm, 2u))
            bm = max(bm, simd_shuffle_xor(bm, 1u))
            let w = exp(keep - bm)
            let btot = simd_sum(w)
            if (nrows == 32u) {
                for [unroll_full] (i in range(32)) {
                    let blk = (c0 + uint(i)) * nblk + blkoff
                    let vb = vb0b + blk * 34u + 2u + inb
                    let dq = float4(float(vqb[vb]), float(vqb[vb + 1u]), float(vqb[vb + 2u]), float(vqb[vb + 3u]))
                    bacc += (simd_shuffle(w, uint(i)) * float(vsh[vb0h + blk * 17u])) * dq
                }
            } else {
                var j = 0u
                while (j < nrows) {
                    let blk = (c0 + j) * nblk + blkoff
                    let vb = vb0b + blk * 34u + 2u + inb
                    let dq = float4(float(vqb[vb]), float(vqb[vb + 1u]), float(vqb[vb + 2u]), float(vqb[vb + 3u]))
                    bacc += (simd_shuffle(w, j) * float(vsh[vb0h + blk * 17u])) * dq
                    j++
                }
            }
            // online merge into the running slice-group state (first pass: r1 = exp(-inf) = 0)
            let nm = max(mm, bm)
            let r1 = exp(mm - nm)
            let r2 = exp(bm - nm)
            wacc[wo] = wacc[wo] * r1 + bacc.x * r2
            wacc[wo + 1u] = wacc[wo + 1u] * r1 + bacc.y * r2
            wacc[wo + 2u] = wacc[wo + 2u] * r1 + bacc.z * r2
            wacc[wo + 3u] = wacc[wo + 3u] * r1 + bacc.w * r2
            tot = tot * r1 + btot * r2
            mm = nm
            ss += nsgs
        }
        let pbase = ((bb * nheads + h) * nsgs + gsg) * (hs + 4u)
        let po = pbase + sl * 4u
        part[po] = wacc[wo]
        part[po + 1u] = wacc[wo + 1u]
        part[po + 2u] = wacc[wo + 2u]
        part[po + 3u] = wacc[wo + 3u]
        if (sl == 0u) {
            part[pbase + hs] = mm
            part[pbase + hs + 1u] = tot
        }
    }
}

// q8_0-mirror batched rope + KV-row store: the RopeStoreQ8 shape with the byte-offset row table
// (grid y = batch row; the new KV row lands at cnt-1). Same quantize_q8kv_row semantics.
class MetalRopeStoreBQ8 {
    @ssbo @binding = 0 qk : array<float>    // [B x qstride] fused QKV rows — q ropes in place
    @ssbo @binding = 1 ksh : array<float16> // the whole mirror arena, half-scale view (k dest)
    @ssbo @binding = 2 kqb : array<int8>    // the SAME arena, byte view
    @ssbo @binding = 3 vsh : array<float16>
    @ssbo @binding = 4 vqb : array<int8>
    @ssbo @binding = 5 tcos : array<float>  // [B x head_size/2] per-row rope rows
    @ssbo @binding = 6 tsin : array<float>
    @ssbo @binding = 7 rt : array<uint4>    // per-row (k base BYTES, v base BYTES, cap_rows, cnt)
    @uniform @binding = 8 lay : uint
    @uniform @binding = 9 nblk : uint       // kv_dim/32 — K blocks (V adds the same again)
    @uniform @binding = 10 qd : uint
    @uniform @binding = 11 head_size : uint
    @uniform @binding = 12 npairs_q : uint  // qd/2 — the q-rope half of the grid
    @uniform @binding = 13 qstride : uint

    [metal_kernel(name="metal_rope_store_b_q8_msl")]
    def metal_rope_store_b_q8 {
        let gid = gl_GlobalInvocationID.x
        if (gid >= npairs_q + 2u * nblk) {
            return
        }
        let bb = gl_WorkGroupID.y
        let rrow = rt[bb]
        let qbase = bb * qstride
        let half = head_size / 2u
        if (gid < npairs_q) {
            let j = gid % half
            let i = (gid / half) * head_size + 2u * j
            let fcr = tcos[bb * half + j]
            let fci = tsin[bb * half + j]
            let v0 = qk[qbase + i]
            let v1 = qk[qbase + i + 1u]
            qk[qbase + i] = v0 * fcr - v1 * fci
            qk[qbase + i + 1u] = v0 * fci + v1 * fcr
            return
        }
        let rowb = nblk * 34u
        let bi = gid - npairs_q
        let isV = bi >= nblk
        let blk = isV ? bi - nblk : bi
        let base = blk * 32u                // element base within the K/V row
        var vals : float[32]
        if (isV) {
            var e = 0u
            while (e < 32u) {
                vals[e] = qk[qbase + qd + nblk * 32u + base + e]   // the raw v third of the QKV row
                e++
            }
        } else {
            var p = 0u
            while (p < 16u) {
                let ge = base + 2u * p
                let j = (ge % head_size) / 2u
                let fcr = tcos[bb * half + j]
                let fci = tsin[bb * half + j]
                let v0 = qk[qbase + qd + ge]
                let v1 = qk[qbase + qd + ge + 1u]
                vals[2u * p] = v0 * fcr - v1 * fci
                vals[2u * p + 1u] = v0 * fci + v1 * fcr
                p++
            }
        }
        var amax = 0.0
        var k = 0u
        while (k < 32u) {
            amax = max(amax, abs(vals[k]))
            k++
        }
        let d = amax / 127.0
        let id = d != 0.0 ? 1.0 / d : 0.0
        if (isV) {
            let vb0b = rrow.y + (lay * rrow.z + (rrow.w - 1u)) * rowb
            vsh[vb0b / 2u + blk * 17u] = float16(d)
            k = 0u
            while (k < 32u) {
                vqb[vb0b + blk * 34u + 2u + k] = int8(round(vals[k] * id))
                k++
            }
        } else {
            let kb0b = rrow.x + (lay * rrow.z + (rrow.w - 1u)) * rowb
            ksh[kb0b / 2u + blk * 17u] = float16(d)
            k = 0u
            while (k < 32u) {
                kqb[kb0b + blk * 34u + 2u + k] = int8(round(vals[k] * id))
                k++
            }
        }
    }
}

// Batched fused residual add + rmsnorm (MetalAddRms with a row grid): one threadgroup per row,
// x += a in place, y = rms(x)·w. Same tgmem row slab — dim caps at ADD_RMS_MAX_DIM.
class MetalAddRmsB {
    @ssbo @binding = 0 x : array<float>     // [B x dim] residual stream, add target
    @ssbo @binding = 1 a : array<float>     // [B x dim] addend
    @ssbo @binding = 2 w : array<float>     // [dim] norm weight
    @ssbo @binding = 3 y : array<float>     // [B x dim] normed out (y may alias a)
    @uniform @binding = 4 dim : uint
    @uniform @binding = 5 eps : float
    @workgroup xrow : float[4096]           // == ADD_RMS_MAX_DIM
    @workgroup partial : float[32]
    @workgroup inv_sh : float

    [metal_kernel(name="metal_add_rms_b_msl")]
    def metal_add_rms_b {
        let row = gl_WorkGroupID.x
        let lid = gl_LocalInvocationID.x
        var ss = 0.0
        var k = lid
        while (k < dim) {
            let vv = x[row * dim + k] + a[row * dim + k]
            x[row * dim + k] = vv
            xrow[k] = vv
            ss += vv * vv
            k += gl_WorkGroupSize.x
        }
        ss = simd_sum(ss)
        if (gl_SubgroupInvocationID == 0u) {
            partial[gl_SubgroupID] = ss
        }
        barrier()
        if (lid == 0u) {
            var tot = 0.0
            var sgi = 0u
            while (sgi < gl_NumSubgroups) {
                tot += partial[sgi]
                sgi++
            }
            inv_sh = rsqrt(tot / float(dim) + eps)
        }
        barrier()
        let inv = inv_sh
        k = lid
        while (k < dim) {
            y[row * dim + k] = w[k] * (xrow[k] * inv)
            k += gl_WorkGroupSize.x
        }
    }
}

// ===== driver state =====

// the attention kernel's threadgroup score slab — contexts deeper than this decline per call (the
// CPU path serves the step; the watermark contract re-mirrors the missed rows if depth drops back)
let MAX_ATTN_ROWS = 2048l
// chunk width of the split-K attention pair (64 rows per threadgroup — must match the part
// kernels' slab + hardcoded chunk math; 64 beats 128 on M1: more threadgroups in flight + half
// the serial per-lane pass length)
let ATTN_CHUNK_ROWS = 64l
// contexts up to this depth take the one-dispatch single-tg kernel (no comb dispatch), deeper
// ones the chunked pair — DASLLAMA_METAL_ATTN_SINGLE overrides for A/B (64 measured better than
// 128 in the 65-128 band: occupancy beats the comb overhead)
var private g_attn_single_max = 64l
// the fused add+rms kernel's threadgroup row slab — wider models take the unfused pair
let ADD_RMS_MAX_DIM = 4096l

// why a step declined to the CPU path — every gate reports its FIRST failing clause, and the
// declines-by-reason tables make silent coverage regressions observable (the support-matrix test
// asserts these names as strings; renaming a value is a test-visible change)
enum MetalDecodeDecline {
    none                // supported — the step runs on the GPU
    quant_mode          // weights not loaded under QuantMode.q8
    kquant_native       // per-tensor K-quant planes: not readable as qblob offsets (kernels = W4)
    arch                // not the llama family graph
    backend_repack      // repack backend: interleaved weight layout unreadable by the GPU kernels
    rope_mode           // rope tables off
    config              // feature-flag config the kernels don't implement (neox/bias/qk-norm/softcap/...)
    shape               // head/dim alignment outside the kernel grids
    layers              // per-layer non-uniformity (kv_dim/head_size/hidden/sliding/missing wv)
    kv_dtype            // session KV codec not uniform f16/f32 (the mirror moves raw row bytes)
    depth               // context deeper than MAX_ATTN_ROWS
    no_uid              // session without a real uid (mirror keying)
    rope_rows           // the position's rope row not built
    batch_dim_cap       // batch: dim exceeds the add+rms/GEMM row slabs for this B
    device              // metal_decode_init failed (no device / PSO build failure)
    mirror              // the mirror arena cannot hold the KV slice
    gpu_error           // dispatch failed at execution time
}

var private g_dev : MetalDevice?
var private g_queue : MetalCommandQueue?
var private g_failed = false
var private g_regions : table<uint64; MetalBuffer?>  // base address -> resident weight/scale/norm region
var private g_pool : MetalBufferPool
var private g_upool : MetalBufferPool                // UNTRACKED pool: uniforms + rope rows (GPU-read-only)
var private g_decodes = 0l
var private g_declines = 0l
var private g_declines_by_reason : table<string; int64>
var private g_logits_cpu = false
var private g_skip = ""                              // DASLLAMA_METAL_DECODE_SKIP=gemv|attn|ew knockout rail
var private g_pso_gemv : MetalComputePipeline?
var private g_pso_gemv16 : MetalComputePipeline?   // f16-scale twin (wscale_f16 models)
var private g_gemv_rows = 4l                       // rows (simdgroups) per GEMV threadgroup
var private g_pso_rms : MetalComputePipeline?
var private g_pso_add : MetalComputePipeline?
var private g_pso_addrms : MetalComputePipeline?
var private g_pso_swiglu : MetalComputePipeline?
var private g_pso_attn16 : MetalComputePipeline?
var private g_pso_attn32 : MetalComputePipeline?
var private g_pso_attn_q8 : MetalComputePipeline?   // q8_0-mirror twin (dual-view block reads)
var private g_pso_attnp16 : MetalComputePipeline?
var private g_pso_attnp32 : MetalComputePipeline?
var private g_pso_attnp_q8 : MetalComputePipeline?
var private g_pso_attncomb : MetalComputePipeline?
var private g_pso_rpst16 : MetalComputePipeline?
var private g_pso_rpst32 : MetalComputePipeline?
var private g_pso_rpst_q8 : MetalComputePipeline?   // per-32-block quantizing store
var private g_pso_argmax : MetalComputePipeline?
var private g_pso_emb : MetalComputePipeline?
var private g_pso_emb16 : MetalComputePipeline?
var private g_pso_qkvrs16 : MetalComputePipeline?   // fused QKV+rope+store (s16 path)
var private g_pso_qkvrs32 : MetalComputePipeline?
var private g_pso_w13sw : MetalComputePipeline?     // fused W1|W3+swiglu (s16 path)
// the batched-step set (P4): fixed-B GEMVs + the M-pad-32 GEMM twin (f32 X) + row-table kernels
var private g_pso_gemv_b2 : MetalComputePipeline?
var private g_pso_gemv_b4 : MetalComputePipeline?
var private g_pso_mv_b2 : MetalComputePipeline?
var private g_pso_mv_b4 : MetalComputePipeline?
var private g_pso_w13sw_b2 : MetalComputePipeline?
var private g_pso_w13sw_b4 : MetalComputePipeline?
var private g_pso_gemm_b : MetalComputePipeline?
var private g_pso_gemm_b_sk : MetalComputePipeline? // split-K twin (per-site policy)
var private g_pso_skred : MetalComputePipeline?     // the split-K plane reduce
var private g_pso_gemm64_b : MetalComputePipeline?  // 64-wide N tile (classifier site)
var private g_pso_mm : MetalComputePipeline?        // the prefill module's MetalQ8MulMm (34B blob)
var private g_pso_attnb16 : MetalComputePipeline?
var private g_pso_attnb32 : MetalComputePipeline?
var private g_pso_attnb_q8 : MetalComputePipeline?
var private g_pso_attnpb16 : MetalComputePipeline?
var private g_pso_attnpb32 : MetalComputePipeline?
var private g_pso_attnpb_q8 : MetalComputePipeline?
var private g_pso_attncombb : MetalComputePipeline?
var private g_pso_attnd16 : MetalComputePipeline?   // fused single-pass batch attention (hs 128)
var private g_pso_attnd32 : MetalComputePipeline?
var private g_pso_attnd_q8 : MetalComputePipeline?
var private g_pso_attncombd : MetalComputePipeline?
var private g_pso_rpstb16 : MetalComputePipeline?
var private g_pso_rpstb32 : MetalComputePipeline?
var private g_pso_rpstb_q8 : MetalComputePipeline?
var private g_pso_addrmsb : MetalComputePipeline?
// cumulative per-stage wall (us) since load — decode is per-token, so logging is on-demand
// (metal_decode_stage_report), never per step
var private g_us_setup = 0l
var private g_us_encode = 0l
var private g_us_wait = 0l
var private g_us_readback = 0l
var private g_gpu_ms = 0.0lf
// the batched-step twins (per eval_batch step, not per token)
var private g_batch_steps = 0l
var private g_batch_rows = 0l
var private g_batch_declines = 0l
var private g_batch_declines_by_reason : table<string; int64>
var private g_us_bsetup = 0l
var private g_us_bencode = 0l
var private g_us_bwait = 0l
var private g_us_breadback = 0l
var private g_gpu_b_ms = 0.0lf
var private g_sched_b_ms = 0.0lf                     // driver scheduling span (kernelStart..kernelEnd)
var private g_handoff_b_ms = 0.0lf                   // kernelEnd -> GPUStart gap

// per-session logical KV mirror record (see the module header). Mirrors are SLICES of one shared
// arena buffer (g_arena) so the batched-attention kernel can reach every row's KV through one
// bind + a per-row element-offset table; the single-stream path binds the same arena at byte
// offsets. All fields copyable — the slice is returned to the free list (mirror_release) before
// any table erase.
struct private KVMirror {
    uid : uint64
    kdt : KVDtype
    vdt : KVDtype
    cap_rows : int64      // logical rows allocated per layer (grows by rebuild + re-upload)
    watermark : int64     // logical rows [0, watermark) are mirrored; pos < watermark = rewind
    tick : int64          // LRU stamp
    koff : uint64         // arena byte offset of the K plane (n_layers slabs of cap_rows rows)
    voff : uint64         // ... and the V plane (koff + bytes/2 — one slice carries both)
    bytes : uint64        // slice size (k + v planes)
}

var private g_mirrors : table<uint64; KVMirror>
var private g_mirror_bytes = 0ul
var private g_tick = 0l
var private g_mirror_cap_mb = -1   // <0 = unset: DASLLAMA_METAL_KV_MIRROR_MB (default 1024) reads at use

// the mirror arena: one buffer, first-fit free-list suballocation (slices are per-session and
// coarse, so fragmentation stays trivial; a failed fit evicts LRU mirrors and retries)
var private g_arena : MetalBuffer?
var private g_arena_bytes = 0ul
var private g_arena_free : array<tuple<off : uint64; size : uint64>>   // sorted by off, coalesced
// uids a batch step is actively encoding against — eviction must never free a batch row's slice
// out from under the step being assembled (the single path's `keep` protects only one uid)
var private g_pin : table<uint64>

def private arena_alloc(size : uint64) : int64 {
    let sz = (size + 15ul) & ~15ul   // 16B-align every slice (buffer binds at byte offsets)
    for (i in range(length(g_arena_free))) {
        if (g_arena_free[i].size >= sz) {
            let off = g_arena_free[i].off
            if (g_arena_free[i].size == sz) {
                g_arena_free |> erase(i)
            } else {
                g_arena_free[i].off += sz
                g_arena_free[i].size -= sz
            }
            return int64(off)
        }
    }
    return -1l
}

def private arena_free_slice(off : uint64; size : uint64) {
    let sz = (size + 15ul) & ~15ul
    var at = length(g_arena_free)
    for (i in range(length(g_arena_free))) {
        if (g_arena_free[i].off > off) {
            at = i
            break
        }
    }
    g_arena_free |> push((off = off, size = sz), at)
    // coalesce with the right neighbor, then the left
    if (at + 1 < length(g_arena_free) && g_arena_free[at].off + g_arena_free[at].size == g_arena_free[at + 1].off) {
        g_arena_free[at].size += g_arena_free[at + 1].size
        g_arena_free |> erase(at + 1)
    }
    if (at > 0 && g_arena_free[at - 1].off + g_arena_free[at - 1].size == g_arena_free[at].off) {
        g_arena_free[at - 1].size += g_arena_free[at].size
        g_arena_free |> erase(at)
    }
}

//! Cap the resident KV-mirror bytes (LRU eviction of idle sessions' mirrors past it) — the churn
//! gate's rail. Negative restores the env default (`DASLLAMA_METAL_KV_MIRROR_MB`, 1024).
def public set_metal_decode_mirror_cap_mb(v : int) {
    g_mirror_cap_mb = v
}

//! Pin the CPU final-norm/classifier step (logits-on-GPU off) — the A/B and fallback rail.
//! `DASLLAMA_METAL_LOGITS=0` pins it process-wide (shared with the prefill driver).
def public set_metal_decode_logits_cpu(v : bool) {
    g_logits_cpu = v
}

def private decline(why : MetalDecodeDecline) {
    g_declines++
    g_declines_by_reason["{why}"]++
}

def private batch_decline(why : MetalDecodeDecline) {
    g_batch_declines++
    g_batch_declines_by_reason["{why}"]++
}

//! (gpu decode steps served, declines to the CPU path) since load — diagnostics / tests.
def public metal_decode_stats() : tuple<decodes : int64; declines : int64> {
    return (decodes = g_decodes, declines = g_declines)
}

//! Single-stream declines-by-reason since load — reason name -> count (the first failing gate
//! clause per declined step). The support-matrix test asserts WHY a config declined.
def public metal_decode_declines() : table<string; int64> {
    return clone_to_move(g_declines_by_reason)
}

//! (batched eval_batch steps served, total rows across them, batch declines) since load.
def public metal_batch_decode_stats() : tuple<steps : int64; rows : int64; declines : int64> {
    return (steps = g_batch_steps, rows = g_batch_rows, declines = g_batch_declines)
}

//! Batched-step declines-by-reason since load — reason name -> count.
def public metal_batch_decode_declines() : table<string; int64> {
    return clone_to_move(g_batch_declines_by_reason)
}

// speculative-chain state: the pre-encoded next step may open with GPU argmax + embed gather off
// the CURRENT step's logits (no CPU poke — the chain never leaves the GPU). A greedy caller
// never misses (same bytes, same tie-break); forced-token phases (warmup loops, tiny prompts
// through per-token forward) and non-greedy samplers miss every step, so adaptive mode backs
// off exponentially (skip 1, 2, 4 … 256 steps) and re-arms fully on any hit.
var private g_spec_mode = -1        // -1 adaptive (default), 0 off, 1 forced
var private g_spec_cooldown = 0     // adaptive: served steps left before the next attempt
var private g_spec_backoff = 1      // the next cooldown length
var private g_spec_hits = 0l
var private g_spec_misses = 0l

//! Speculative greedy chaining rail: -1 adaptive (default), 0 off, 1 forced-on (tests/A-B).
def public set_metal_decode_speculate(v : int) {
    g_spec_mode = v
    g_spec_cooldown = 0
    g_spec_backoff = 1
}

//! (speculative steps verified correct, mispredicted-and-rerun) since load.
def public metal_decode_spec_stats() : tuple<hits : int64; misses : int64> {
    return (hits = g_spec_hits, misses = g_spec_misses)
}

def private spec_armed() : bool {
    if (g_spec_mode == 0) {
        return false
    }
    if (g_spec_mode == 1) {
        return true
    }
    if (g_spec_cooldown > 0) {
        g_spec_cooldown--
        return false
    }
    return true
}

//! The knockout rail as an in-process setter (timing attribution — served output is GARBAGE while
//! non-empty): `"gemv"` | `"attn"` | `"ew"`; `""` restores full math. Same rail as
//! `DASLLAMA_METAL_DECODE_SKIP`, switchable between measurement windows.
def public set_metal_decode_skip(v : string) {
    g_skip = v
}

//! Zero the cumulative per-stage timers + the served-step counter — windows a chase run's stage
//! stats to one measurement loop (declines keep accumulating; they gate support, not speed).
def public metal_decode_stage_reset {
    g_decodes = 0l
    g_us_setup = 0l
    g_us_encode = 0l
    g_us_wait = 0l
    g_us_readback = 0l
    g_gpu_ms = 0.0lf
    g_batch_steps = 0l
    g_batch_rows = 0l
    g_us_bsetup = 0l
    g_us_bencode = 0l
    g_us_bwait = 0l
    g_us_breadback = 0l
    g_gpu_b_ms = 0.0lf
    g_sched_b_ms = 0.0lf
    g_handoff_b_ms = 0.0lf
}

//! Cumulative batched-step stage wall since load/reset — the batch bench's numbers rail
//! (per STEP, one step = nrows tokens).
def public metal_batch_decode_stage_stats() : tuple<steps : int64; rows : int64; setup_us : int64; encode_us : int64; wait_us : int64; gpu_us : double; readback_us : int64> {
    return (steps = g_batch_steps, rows = g_batch_rows, setup_us = g_us_bsetup,
        encode_us = g_us_bencode, wait_us = g_us_bwait, gpu_us = g_gpu_b_ms * 1000.0lf,
        readback_us = g_us_breadback)
}

//! Cumulative per-stage wall since load/reset — (served steps, CPU setup us, encode us, commit+wait
//! us, GPU-timestamp us within the wait, readback+writeback us). The chase harness's numbers rail.
def public metal_decode_stage_stats() : tuple<steps : int64; setup_us : int64; encode_us : int64; wait_us : int64; gpu_us : double; readback_us : int64> {
    return (steps = g_decodes, setup_us = g_us_setup, encode_us = g_us_encode,
        wait_us = g_us_wait, gpu_us = g_gpu_ms * 1000.0lf, readback_us = g_us_readback)
}

//! (live KV mirrors, resident mirror bytes) — the churn / LRU-cap gates read this.
def public metal_decode_mirror_stats() : tuple<mirrors : int64; bytes : uint64> {
    return (mirrors = int64(length(g_mirrors)), bytes = g_mirror_bytes)
}

// " reason=count reason=count" — the declines-by-reason tail of a report line
def private declines_line(tab : table<string; int64>) : string {
    return build_string() $(var w) {
        for (k, v in keys(tab), values(tab)) {
            w |> write(" ")
            w |> write(k)
            w |> write("=")
            w |> write(v)
        }
    }
}

//! One cumulative per-stage timing line for the steps served since load (decode is per-token, so
//! the driver never logs per step) + declines-by-reason for all three GPU paths — the decline
//! tables surface HERE even when zero steps served (the silent-decline case this report exists for).
def public metal_decode_stage_report {
    if (g_decodes == 0l && g_batch_steps == 0l) {
        to_log(LOG_INFO, "dasLLAMA metal decode: no steps served\n")
    }
    if (g_decodes > 0l) {
        let n = float(g_decodes)
        to_log(LOG_INFO, "dasLLAMA metal decode: {g_decodes} steps  per-step setup={float(g_us_setup) / n}us encode={float(g_us_encode) / n}us wait={float(g_us_wait) / n}us (gpu {g_gpu_ms * 1000.0lf / double(g_decodes)}us) readback+writeback={float(g_us_readback) / n}us\n")
    }
    if (g_batch_steps > 0l) {
        let n = float(g_batch_steps)
        to_log(LOG_INFO, "dasLLAMA metal batch decode: {g_batch_steps} steps ({g_batch_rows} rows)  per-step setup={float(g_us_bsetup) / n}us encode={float(g_us_bencode) / n}us wait={float(g_us_bwait) / n}us (sched {g_sched_b_ms * 1000.0lf / double(g_batch_steps)}us handoff {g_handoff_b_ms * 1000.0lf / double(g_batch_steps)}us gpu {g_gpu_b_ms * 1000.0lf / double(g_batch_steps)}us) readback+writeback={float(g_us_breadback) / n}us\n")
    }
    if (g_declines > 0l) {
        to_log(LOG_INFO, "dasLLAMA metal decode declines: {g_declines} —{declines_line(g_declines_by_reason)}\n")
    }
    if (g_batch_declines > 0l) {
        to_log(LOG_INFO, "dasLLAMA metal batch decode declines: {g_batch_declines} —{declines_line(g_batch_declines_by_reason)}\n")
    }
    let pf = metal_prefill_stats()
    if (pf.declines > 0l) {
        var pfd <- metal_prefill_declines()
        to_log(LOG_INFO, "dasLLAMA metal prefill declines: {pf.declines} —{declines_line(pfd)}\n")
        delete pfd
    }
}

def private release_pso(var pso : MetalComputePipeline?&) {
    if (pso != null) {
        metal_release(pso)
        pso = null
    }
}

def private mirror_release(var mr : KVMirror) {
    if (mr.bytes > 0ul) {
        arena_free_slice(mr.koff, mr.bytes)
    }
    g_mirror_bytes -= mr.bytes
    mr.bytes = 0ul
}

//! Release every GPU object the driver holds — mirrors, weight regions, pools, PSOs, queue,
//! device (the tests' leak-gate rail, same contract as metal_prefill_shutdown). Lazily
//! re-initializes on the next served step; all mirrors rebuild from the CPU caches on demand.
def public metal_decode_shutdown {
    discard_pre()
    finish_pending_step()   // land any in-flight pipelined step before tearing down
    g_lp_err = false
    metal_blob_regions_release()   // the batch mm path creates blobs in the prefill module's cache
    for (buf in values(g_cat_blobs)) {
        if (buf != null) {
            metal_release(buf)
        }
        buf = null
    }
    unsafe {
        delete g_cat_blobs
    }
    for (mr in values(g_mirrors)) {
        mirror_release(mr)
    }
    unsafe {
        delete g_mirrors
    }
    g_mirror_bytes = 0ul
    if (g_arena != null) {
        metal_release(g_arena)
        g_arena = null
    }
    g_arena_bytes = 0ul
    g_arena_free |> clear()
    for (buf in values(g_regions)) {
        if (buf != null) {
            metal_release(buf)
        }
        buf = null
    }
    unsafe {
        delete g_regions
    }
    pool_drain(g_pool)
    pool_drain(g_upool)
    if (g_pso_gemv != null) {
        metal_release(g_pso_gemv)
        g_pso_gemv = null
    }
    if (g_pso_gemv16 != null) {
        metal_release(g_pso_gemv16)
        g_pso_gemv16 = null
    }
    if (g_pso_rms != null) {
        metal_release(g_pso_rms)
        g_pso_rms = null
    }
    if (g_pso_add != null) {
        metal_release(g_pso_add)
        g_pso_add = null
    }
    if (g_pso_addrms != null) {
        metal_release(g_pso_addrms)
        g_pso_addrms = null
    }
    if (g_pso_swiglu != null) {
        metal_release(g_pso_swiglu)
        g_pso_swiglu = null
    }
    if (g_pso_attn16 != null) {
        metal_release(g_pso_attn16)
        g_pso_attn16 = null
    }
    if (g_pso_attn32 != null) {
        metal_release(g_pso_attn32)
        g_pso_attn32 = null
    }
    if (g_pso_attnp16 != null) {
        metal_release(g_pso_attnp16)
        g_pso_attnp16 = null
    }
    if (g_pso_attnp32 != null) {
        metal_release(g_pso_attnp32)
        g_pso_attnp32 = null
    }
    release_pso(g_pso_attn_q8)
    release_pso(g_pso_attnp_q8)
    release_pso(g_pso_rpst_q8)
    if (g_pso_attncomb != null) {
        metal_release(g_pso_attncomb)
        g_pso_attncomb = null
    }
    if (g_pso_rpst16 != null) {
        metal_release(g_pso_rpst16)
        g_pso_rpst16 = null
    }
    if (g_pso_rpst32 != null) {
        metal_release(g_pso_rpst32)
        g_pso_rpst32 = null
    }
    if (g_pso_argmax != null) {
        metal_release(g_pso_argmax)
        g_pso_argmax = null
    }
    if (g_pso_emb != null) {
        metal_release(g_pso_emb)
        g_pso_emb = null
    }
    if (g_pso_emb16 != null) {
        metal_release(g_pso_emb16)
        g_pso_emb16 = null
    }
    if (g_pso_qkvrs16 != null) {
        metal_release(g_pso_qkvrs16)
        g_pso_qkvrs16 = null
    }
    if (g_pso_qkvrs32 != null) {
        metal_release(g_pso_qkvrs32)
        g_pso_qkvrs32 = null
    }
    if (g_pso_w13sw != null) {
        metal_release(g_pso_w13sw)
        g_pso_w13sw = null
    }
    release_pso(g_pso_gemv_b2)
    release_pso(g_pso_gemv_b4)
    release_pso(g_pso_mv_b2)
    release_pso(g_pso_mv_b4)
    release_pso(g_pso_w13sw_b2)
    release_pso(g_pso_w13sw_b4)
    release_pso(g_pso_gemm_b)
    release_pso(g_pso_gemm_b_sk)
    release_pso(g_pso_skred)
    release_pso(g_pso_gemm64_b)
    release_pso(g_pso_mm)
    release_pso(g_pso_attnb16)
    release_pso(g_pso_attnb32)
    release_pso(g_pso_attnb_q8)
    release_pso(g_pso_attnpb16)
    release_pso(g_pso_attnpb32)
    release_pso(g_pso_attnpb_q8)
    release_pso(g_pso_attncombb)
    release_pso(g_pso_attnd16)
    release_pso(g_pso_attnd32)
    release_pso(g_pso_attnd_q8)
    release_pso(g_pso_attncombd)
    release_pso(g_pso_rpstb16)
    release_pso(g_pso_rpstb32)
    release_pso(g_pso_rpstb_q8)
    release_pso(g_pso_addrmsb)
    if (g_queue != null) {
        metal_release(g_queue)
        g_queue = null
    }
    if (g_dev != null) {
        metal_release(g_dev)
        g_dev = null
    }
    g_failed = false
}

def private env_int(name : string; dflt : int) : int {
    return has_env_variable(name) ? to_int(get_env_variable(name)) : dflt
}

def private compile_pso(source, entry : string; fastmath : bool; var ok : bool&) : MetalComputePipeline? {
    var err : string
    var pso = pipeline_from_source(g_dev, source, entry, fastmath, err)
    if (pso == null) {
        to_log(LOG_ERROR, "dasLLAMA metal decode: `{entry}` compile failed — {err}\n")
        ok = false
    }
    return pso
}

def private metal_decode_init : bool {
    if (g_pso_add != null) {
        return true
    }
    if (g_failed) {
        return false
    }
    g_dev = metal_create_system_default_device()
    if (g_dev == null) {
        to_log(LOG_WARNING, "dasLLAMA metal decode: no Metal device — decode stays on the CPU\n")
        g_failed = true
        return false
    }
    g_queue = metal_new_command_queue(g_dev)
    var ok = true
    // the GEMV / rms / add / swiglu MSL sources are the prefill module's, verbatim (own PSOs)
    g_pso_gemv = compile_pso(metal_q8_gemv_msl, metal_q8_gemv_msl_entry, metal_q8_gemv_msl_fastmath, ok)
    g_pso_gemv16 = compile_pso(metal_q8_gemv_s16_msl, metal_q8_gemv_s16_msl_entry, metal_q8_gemv_s16_msl_fastmath, ok)
    g_gemv_rows = clamp(int64(env_int("DASLLAMA_METAL_GEMV_TG", 4)), 1l, 32l)
    g_attn_single_max = clamp(int64(env_int("DASLLAMA_METAL_ATTN_SINGLE", 64)), ATTN_CHUNK_ROWS, 128l)
    g_pso_rms = compile_pso(metal_rmsnorm_msl, metal_rmsnorm_msl_entry, metal_rmsnorm_msl_fastmath, ok)
    g_pso_add = compile_pso(metal_add_msl, metal_add_msl_entry, metal_add_msl_fastmath, ok)
    g_pso_addrms = compile_pso(metal_add_rms_msl, metal_add_rms_msl_entry, metal_add_rms_msl_fastmath, ok)
    g_pso_swiglu = compile_pso(metal_swiglu_msl, metal_swiglu_msl_entry, metal_swiglu_msl_fastmath, ok)
    g_pso_attn16 = compile_pso(metal_sq_attn16_msl, metal_sq_attn16_msl_entry, metal_sq_attn16_msl_fastmath, ok)
    g_pso_attn32 = compile_pso(metal_sq_attn32_msl, metal_sq_attn32_msl_entry, metal_sq_attn32_msl_fastmath, ok)
    g_pso_attnp16 = compile_pso(metal_sq_attn_part16_msl, metal_sq_attn_part16_msl_entry, metal_sq_attn_part16_msl_fastmath, ok)
    g_pso_attnp32 = compile_pso(metal_sq_attn_part32_msl, metal_sq_attn_part32_msl_entry, metal_sq_attn_part32_msl_fastmath, ok)
    g_pso_attncomb = compile_pso(metal_sq_attn_comb_msl, metal_sq_attn_comb_msl_entry, metal_sq_attn_comb_msl_fastmath, ok)
    g_pso_attn_q8 = compile_pso(metal_sq_attn_q8_msl, metal_sq_attn_q8_msl_entry, metal_sq_attn_q8_msl_fastmath, ok)
    g_pso_attnp_q8 = compile_pso(metal_sq_attn_part_q8_msl, metal_sq_attn_part_q8_msl_entry, metal_sq_attn_part_q8_msl_fastmath, ok)
    g_pso_rpst16 = compile_pso(metal_rope_store16_msl, metal_rope_store16_msl_entry, metal_rope_store16_msl_fastmath, ok)
    g_pso_rpst32 = compile_pso(metal_rope_store32_msl, metal_rope_store32_msl_entry, metal_rope_store32_msl_fastmath, ok)
    g_pso_rpst_q8 = compile_pso(metal_rope_store_q8_msl, metal_rope_store_q8_msl_entry, metal_rope_store_q8_msl_fastmath, ok)
    g_pso_argmax = compile_pso(metal_argmax_msl, metal_argmax_msl_entry, metal_argmax_msl_fastmath, ok)
    g_pso_emb = compile_pso(metal_embed_q8_msl, metal_embed_q8_msl_entry, metal_embed_q8_msl_fastmath, ok)
    g_pso_emb16 = compile_pso(metal_embed_q8_s16_msl, metal_embed_q8_s16_msl_entry, metal_embed_q8_s16_msl_fastmath, ok)
    g_pso_qkvrs16 = compile_pso(metal_q8_gemv_qkv_rs16_msl, metal_q8_gemv_qkv_rs16_msl_entry, metal_q8_gemv_qkv_rs16_msl_fastmath, ok)
    g_pso_qkvrs32 = compile_pso(metal_q8_gemv_qkv_rs32_msl, metal_q8_gemv_qkv_rs32_msl_entry, metal_q8_gemv_qkv_rs32_msl_fastmath, ok)
    g_pso_w13sw = compile_pso(metal_q8_gemv_w13sw16_msl, metal_q8_gemv_w13sw16_msl_entry, metal_q8_gemv_w13sw16_msl_fastmath, ok)
    // the batched-step set (P4)
    g_pso_gemv_b2 = compile_pso(metal_gemv_b2_msl, metal_gemv_b2_msl_entry, metal_gemv_b2_msl_fastmath, ok)
    g_pso_gemv_b4 = compile_pso(metal_gemv_b4_msl, metal_gemv_b4_msl_entry, metal_gemv_b4_msl_fastmath, ok)
    g_pso_mv_b2 = compile_pso(metal_q8_mv_b2_msl, metal_q8_mv_b2_msl_entry, metal_q8_mv_b2_msl_fastmath, ok)
    g_pso_mv_b4 = compile_pso(metal_q8_mv_b4_msl, metal_q8_mv_b4_msl_entry, metal_q8_mv_b4_msl_fastmath, ok)
    g_pso_w13sw_b2 = compile_pso(metal_gemv_w13sw_b2_msl, metal_gemv_w13sw_b2_msl_entry, metal_gemv_w13sw_b2_msl_fastmath, ok)
    g_pso_w13sw_b4 = compile_pso(metal_gemv_w13sw_b4_msl, metal_gemv_w13sw_b4_msl_entry, metal_gemv_w13sw_b4_msl_fastmath, ok)
    g_pso_gemm_b = compile_pso(metal_q8_gemm_b_msl, metal_q8_gemm_b_msl_entry, metal_q8_gemm_b_msl_fastmath, ok)
    g_pso_gemm_b_sk = compile_pso(metal_q8_gemm_b_sk_msl, metal_q8_gemm_b_sk_msl_entry, metal_q8_gemm_b_sk_msl_fastmath, ok)
    g_pso_skred = compile_pso(metal_sk_reduce_msl, metal_sk_reduce_msl_entry, metal_sk_reduce_msl_fastmath, ok)
    g_pso_gemm64_b = compile_pso(metal_q8_gemm64_b_msl, metal_q8_gemm64_b_msl_entry, metal_q8_gemm64_b_msl_fastmath, ok)
    g_pso_mm = compile_pso(metal_q8_mulmm_msl, metal_q8_mulmm_msl_entry, metal_q8_mulmm_msl_fastmath, ok)
    g_pso_attnb16 = compile_pso(metal_sq_attn_b16_msl, metal_sq_attn_b16_msl_entry, metal_sq_attn_b16_msl_fastmath, ok)
    g_pso_attnb32 = compile_pso(metal_sq_attn_b32_msl, metal_sq_attn_b32_msl_entry, metal_sq_attn_b32_msl_fastmath, ok)
    g_pso_attnb_q8 = compile_pso(metal_sq_attn_b_q8_msl, metal_sq_attn_b_q8_msl_entry, metal_sq_attn_b_q8_msl_fastmath, ok)
    g_pso_attnpb16 = compile_pso(metal_sq_attn_part_b16_msl, metal_sq_attn_part_b16_msl_entry, metal_sq_attn_part_b16_msl_fastmath, ok)
    g_pso_attnpb32 = compile_pso(metal_sq_attn_part_b32_msl, metal_sq_attn_part_b32_msl_entry, metal_sq_attn_part_b32_msl_fastmath, ok)
    g_pso_attnpb_q8 = compile_pso(metal_sq_attn_part_b_q8_msl, metal_sq_attn_part_b_q8_msl_entry, metal_sq_attn_part_b_q8_msl_fastmath, ok)
    g_pso_attncombb = compile_pso(metal_sq_attn_comb_b_msl, metal_sq_attn_comb_b_msl_entry, metal_sq_attn_comb_b_msl_fastmath, ok)
    g_pso_attnd16 = compile_pso(metal_sq_attn_d16_msl, metal_sq_attn_d16_msl_entry, metal_sq_attn_d16_msl_fastmath, ok)
    g_pso_attnd32 = compile_pso(metal_sq_attn_d32_msl, metal_sq_attn_d32_msl_entry, metal_sq_attn_d32_msl_fastmath, ok)
    g_pso_attnd_q8 = compile_pso(metal_sq_attn_d_q8_msl, metal_sq_attn_d_q8_msl_entry, metal_sq_attn_d_q8_msl_fastmath, ok)
    g_pso_attncombd = compile_pso(metal_sq_attn_comb_d_msl, metal_sq_attn_comb_d_msl_entry, metal_sq_attn_comb_d_msl_fastmath, ok)
    g_pso_rpstb16 = compile_pso(metal_rope_store_b16_msl, metal_rope_store_b16_msl_entry, metal_rope_store_b16_msl_fastmath, ok)
    g_pso_rpstb32 = compile_pso(metal_rope_store_b32_msl, metal_rope_store_b32_msl_entry, metal_rope_store_b32_msl_fastmath, ok)
    g_pso_rpstb_q8 = compile_pso(metal_rope_store_b_q8_msl, metal_rope_store_b_q8_msl_entry, metal_rope_store_b_q8_msl_fastmath, ok)
    g_pso_addrmsb = compile_pso(metal_add_rms_b_msl, metal_add_rms_b_msl_entry, metal_add_rms_b_msl_fastmath, ok)
    if (!ok) {
        metal_decode_shutdown()
        g_failed = true
        return false
    }
    to_log(LOG_INFO, "dasLLAMA metal decode: resident path live on {metal_device_name(g_dev)}\n")
    return true
}

// lazily upload one stable region (weight quants / scales / norm vectors), keyed by base address.
// UNTRACKED: weights are GPU-read-only, so they carry no hazards (the prefill driver measured
// tracked weight regions dominating the scheduler's dependency analysis)
def private upload_region(src : void?; bytes : uint64) : MetalBuffer? {
    let key = intptr(src)
    if (key_exists(g_regions, key)) {
        return g_regions[key]
    }
    var buf = metal_new_buffer_untracked(g_dev, bytes)
    unsafe {
        memcpy(metal_buffer_contents(buf), src, int(bytes))
    }
    g_regions[key] = buf
    return buf
}

// lazily upload up to three CONSECUTIVE-ROW regions as ONE resident buffer, keyed by the first
// piece's base address — the fused QKV / W1W3 GEMV streams (one dispatch, continuous row
// indexing). Load-once-stable like upload_region; b2 == 0 makes it a two-piece concat.
def private upload_region_cat3(p0 : void?; b0 : uint64; p1 : void?; b1 : uint64; p2 : void?; b2 : uint64) : MetalBuffer? {
    let key = intptr(p0)
    if (key_exists(g_regions, key)) {
        return g_regions[key]
    }
    var buf = metal_new_buffer_untracked(g_dev, b0 + b1 + b2)
    unsafe {
        var d = reinterpret<uint8?>(metal_buffer_contents(buf))
        memcpy(reinterpret<void?>(d), p0, int(b0))
        memcpy(reinterpret<void?>(d + b0), p1, int(b1))
        if (b2 > 0ul) {
            memcpy(reinterpret<void?>(d + (b0 + b1)), p2, int(b2))
        }
    }
    g_regions[key] = buf
    return buf
}

// one tensor's scale-plane region — the f32 plane or the wscale_f16 halfword plane, whichever
// the model carries (the GEMV PSO twin is picked to match)
def private upload_scales(t : Model; off, rows, n : int64) : MetalBuffer? {
    let cnt = rows * (n / 32l)
    unsafe {
        if (t.wscale_f16) {
            return upload_region(addr < void ? > (t.qscales16[off / 32l]), uint64(cnt * 2l))
        }
        return upload_region(addr < void ? > (t.qscales[off / 32l]), uint64(cnt * 4l))
    }
}

// one tensor's resident 34B q8-block blob (the prefill module's cache — shared, repacked once)
def private blob_of(t : Model; off, rows, n : int64) : MetalBuffer? {
    unsafe {
        let q = addr < void ? > (t.qblob[off])
        let s = t.wscale_f16 ? addr < void ? > (t.qscales16[off / 32l]) : addr < void ? > (t.qscales[off / 32l])
        return metal_blob_region(g_dev, q, s, rows, n, t.wscale_f16)
    }
}

// the concat twin for the fused QKV / W1|W3 scale planes (rows2 == 0 = two-piece)
def private upload_scales_cat3(t : Model; off0, rows0, off1, rows1, off2, rows2, n : int64) : MetalBuffer? {
    let ssz = t.wscale_f16 ? 2l : 4l
    let nb = n / 32l
    unsafe {
        let p0 = t.wscale_f16 ? addr < void ? > (t.qscales16[off0 / 32l]) : addr < void ? > (t.qscales[off0 / 32l])
        let p1 = t.wscale_f16 ? addr < void ? > (t.qscales16[off1 / 32l]) : addr < void ? > (t.qscales[off1 / 32l])
        var p2 : void?
        if (rows2 > 0l) {
            p2 = t.wscale_f16 ? addr < void ? > (t.qscales16[off2 / 32l]) : addr < void ? > (t.qscales[off2 / 32l])
        }
        return upload_region_cat3(p0, uint64(rows0 * nb * ssz), p1, uint64(rows1 * nb * ssz), p2, uint64(rows2 * nb * ssz))
    }
}

def private uniform_u32(v : uint) : MetalBuffer? {
    var buf = pool_acquire_untracked(g_upool, g_dev, 4ul)
    var p = unsafe(reinterpret<uint?>(metal_buffer_contents(buf)))
    unsafe(p[0]) = v
    return buf
}

def private uniform_f32(v : float) : MetalBuffer? {
    var buf = pool_acquire_untracked(g_upool, g_dev, 4ul)
    var p = unsafe(reinterpret<float?>(metal_buffer_contents(buf)))
    unsafe(p[0]) = v
    return buf
}

// only the llama-family std shape is wired — the same decline set as prefill_decline minus the
// npos window (the model/config half, shared by the single-stream and batched drivers). Returns
// the FIRST failing clause; .none = supported.
def private decode_shape_decline(t : Model) : MetalDecodeDecline {
    if (t.quant != QuantMode.q8) {
        return MetalDecodeDecline.quant_mode
    }
    // kquant_native: the per-tensor K-quant planes are NOT readable as qblob offsets — without
    // this decline a Q4_K_M file passes the q8-mode check and blob_of reads the wrong plane
    // (silent garbage). Real K-quant kernels are the format-matrix arc's W4.
    if (t.kquant_native) {
        return MetalDecodeDecline.kquant_native
    }
    if (t.arch != "llama") {
        return MetalDecodeDecline.arch
    }
    if (kernel_backend_needs_repack(active_kernel_backend())) {
        return MetalDecodeDecline.backend_repack
    }
    if (!get_rope_table_mode()) {
        return MetalDecodeDecline.rope_mode
    }
    let c = t.config
    if ((c.rope_neox || c.attn_qkv_bias || c.qk_norm || c.v_norm || c.pre_post_norm) ||
            (c.layer_out_scale || c.attn_sinks || c.attn_logit_softcap > 0.0 || c.sliding_window > 0l) ||
            (c.ffn_act != FfnAct.silu || c.attn_scale > 0.0)) {
        return MetalDecodeDecline.config
    }
    let head_size = c.head_size
    let qd = c.n_heads * head_size
    let kv_dim = layer_kv_dim(c, 0l)
    let hidden = layer_hidden(t, 0l)
    // head_size == the attention threadgroup width: whole simdgroups only, red[] holds <= 4
    if ((qd != c.dim || (head_size % 32l) != 0l || head_size > 128l) ||
            ((c.dim % 32l) != 0l || (kv_dim % 32l) != 0l || (hidden % 32l) != 0l)) {
        return MetalDecodeDecline.shape
    }
    for (l in range64(c.n_layers)) {
        if ((t.kv_src[l] != l || layer_kv_dim(c, l) != kv_dim || layer_head_size(c, l) != head_size) ||
                (layer_hidden(t, l) != hidden || layer_is_sliding(c, l) || t.wv_offs[l] < 0l)) {
            return MetalDecodeDecline.layers
        }
    }
    return MetalDecodeDecline.none
}

// ... plus the decode-specific gates (uniform f16/f32 KV, attention depth, a real session uid)
def private decode_decline(t : Model; s : Session; pos : int64) : MetalDecodeDecline {
    // the mirror + writeback move raw row bytes, so the session codec must equal the mirror
    // dtype; the attention kernel's threadgroup score slab caps the depth — deeper steps run on
    // the CPU and the watermark contract mirrors them back in if the driver is ever re-entered
    if (s.kv_dtype_k != s.kv_dtype_v ||
            (s.kv_dtype_k != KVDtype.f16 && s.kv_dtype_k != KVDtype.f32 && s.kv_dtype_k != KVDtype.q8_0)) {
        return MetalDecodeDecline.kv_dtype
    }
    if (pos + 1l > MAX_ATTN_ROWS) {
        return MetalDecodeDecline.depth
    }
    if (s.uid == 0ul) {
        return MetalDecodeDecline.no_uid
    }
    let shape = decode_shape_decline(t)
    if (shape != MetalDecodeDecline.none) {
        return shape
    }
    // forward() built this position's rope row right before the hook
    if (int64(length(s.rope_cos)) < t.config.head_size / 2l) {
        return MetalDecodeDecline.rope_rows
    }
    return MetalDecodeDecline.none
}

// Evict least-recently-used mirrors (never `keep`, never a pinned batch row) until the resident
// bytes fit the cap.
def private mirror_evict_to_cap(cap_bytes : uint64; keep : uint64) {
    while (g_mirror_bytes > cap_bytes) {
        var victim = 0ul
        var vtick = 0l
        for (uid, mr in keys(g_mirrors), values(g_mirrors)) {
            if (uid != keep && !key_exists(g_pin, uid) && (victim == 0ul || mr.tick < vtick)) {
                victim = uid
                vtick = mr.tick
            }
        }
        if (victim == 0ul) {   // only the live session's mirror remains
            return
        }
        mirror_release(g_mirrors[victim])
        g_mirrors |> erase(victim)
    }
}

// Ensure s.uid's mirror exists, is big enough, and holds logical rows [0, pos) — the watermark
// contract (module header). Returns (ok, koff, voff, cap_rows) — arena byte offsets; the caller
// writes row `pos` GPU-side and advances the watermark to pos + 1 only after the step + writeback
// complete. ok = false ⇒ the arena cannot hold the slice (decline the step to the CPU path).
def private mirror_prepare(t : Model; var s : Session; pos : int64) : tuple<ok : bool; koff : uint64; voff : uint64; cap : int64> {
    let c = t.config
    let kv_dim = layer_kv_dim(c, 0l)
    let krow = kv_row_bytes(s.kv_dtype_k, kv_dim)
    let vrow = kv_row_bytes(s.kv_dtype_v, kv_dim)
    let needed = pos + 1l
    if (key_exists(g_mirrors, s.uid)) {
        // uid is never reused, so dtype can't legitimately change — a mismatch means a stale
        // record; capacity growth rebuilds too (watermark 0 → full re-upload from the CPU cache)
        let mr0 = g_mirrors[s.uid]   // cheap copy — fields tested off it, one lookup per statement
        if (mr0.cap_rows < needed || mr0.kdt != s.kv_dtype_k || mr0.vdt != s.kv_dtype_v) {
            mirror_release(g_mirrors[s.uid])
            g_mirrors |> erase(s.uid)
        }
    }
    if (!key_exists(g_mirrors, s.uid)) {
        if (g_arena == null) {
            let mb = uint64(clamp(env_int("DASLLAMA_METAL_KV_MIRROR_MB", 1024), 64, 4096))
            g_arena_bytes = mb * 1024ul * 1024ul
            g_arena = metal_new_buffer(g_dev, g_arena_bytes)
            g_arena_free |> clear()
            g_arena_free |> push((off = 0ul, size = g_arena_bytes))
        }
        var cap = 256l
        while (cap < needed) {
            cap *= 2l
        }
        cap = clamp(cap, needed, c.seq_len)   // chunked high-water, never past the session's own cap
        let kbytes = uint64(c.n_layers * cap * krow)
        let vbytes = uint64(c.n_layers * cap * vrow)
        var off = arena_alloc(kbytes + vbytes)
        while (off < 0l) {   // arena full — evict LRU until the slice fits (or nothing evictable)
            var victim = 0ul
            var vtick = 0l
            for (uid, mr in keys(g_mirrors), values(g_mirrors)) {
                if (uid != s.uid && !key_exists(g_pin, uid) && (victim == 0ul || mr.tick < vtick)) {
                    victim = uid
                    vtick = mr.tick
                }
            }
            if (victim == 0ul) {
                return (ok = false, koff = 0ul, voff = 0ul, cap = 0l)
            }
            mirror_release(g_mirrors[victim])
            g_mirrors |> erase(victim)
            off = arena_alloc(kbytes + vbytes)
        }
        let mr = KVMirror(uid = s.uid, kdt = s.kv_dtype_k, vdt = s.kv_dtype_v, cap_rows = cap,
            koff = uint64(off), voff = uint64(off) + kbytes, bytes = kbytes + vbytes)
        g_mirror_bytes += mr.bytes
        g_mirrors[s.uid] = mr
        let cap_mb = uint64(g_mirror_cap_mb >= 0 ? g_mirror_cap_mb : env_int("DASLLAMA_METAL_KV_MIRROR_MB", 1024))
        mirror_evict_to_cap(cap_mb * 1024ul * 1024ul, s.uid)
    }
    g_tick++
    g_mirrors[s.uid].tick = g_tick
    if (pos < g_mirrors[s.uid].watermark) {   // rewind detected → re-mirror from scratch
        g_mirrors[s.uid].watermark = 0l
    }
    let wm = g_mirrors[s.uid].watermark
    let cap = g_mirrors[s.uid].cap_rows
    let koff = g_mirrors[s.uid].koff
    let voff = g_mirrors[s.uid].voff
    if (wm < pos) {
        // gather logical rows [wm, pos) from the CPU cache (source of truth) — flat sessions are
        // one run, paged sessions gather through kv_runs; bytes move verbatim (same dtype)
        let nrun = kv_runs(s, wm, pos, s.kv_runs)
        unsafe {
            var kdst = reinterpret<uint8?>(metal_buffer_contents(g_arena)) + koff
            var vdst = reinterpret<uint8?>(metal_buffer_contents(g_arena)) + voff
            for (l in range64(c.n_layers)) {
                let kbase = kv_layer_kbase(s, l, c.seq_len)
                let vbase = kv_layer_vbase(s, l, c.seq_len)
                for (r in range64(nrun)) {
                    let j0 = s.kv_runs[r].j0
                    let nrows = s.kv_runs[r].j1 - j0
                    let prow = s.kv_runs[r].row0
                    memcpy(reinterpret<void?>(kdst + (l * cap + j0) * krow),
                        reinterpret<void?>(kbase + prow * krow), int(nrows * krow))
                    memcpy(reinterpret<void?>(vdst + (l * cap + j0) * vrow),
                        reinterpret<void?>(vbase + prow * vrow), int(nrows * vrow))
                }
            }
        }
        g_mirrors[s.uid].watermark = pos
    }
    return (ok = true, koff = koff, voff = voff, cap = cap)
}

// ===== encoders =====

def private enc_gemv(enc : MetalComputeEncoder?; s16 : bool; bwq, bws, bx, by, bn, bd : MetalBuffer?; d : int64) {
    metal_set_pipeline(enc, s16 ? g_pso_gemv16 : g_pso_gemv)
    metal_set_buffer(enc, bwq, 0ul, 0)
    metal_set_buffer(enc, bws, 0ul, 1)
    metal_set_buffer(enc, bx, 0ul, 2)
    metal_set_buffer(enc, by, 0ul, 3)
    metal_set_buffer(enc, bn, 0ul, 4)
    metal_set_buffer(enc, bd, 0ul, 5)
    // the kernel is width-agnostic (one row per simdgroup) — rows/tg via DASLLAMA_METAL_GEMV_TG
    metal_dispatch_threadgroups(enc, uint3(uint((d + g_gemv_rows - 1l) / g_gemv_rows), 1u, 1u), uint3(uint(g_gemv_rows * 32l), 1u, 1u))
}

def private enc_rms(enc : MetalComputeEncoder?; bx, bw, by, bdim, beps : MetalBuffer?) {
    metal_set_pipeline(enc, g_pso_rms)
    metal_set_threadgroup_memory_length(enc, metal_rmsnorm_msl_tgmem, 0)
    metal_set_buffer(enc, bx, 0ul, 0)
    metal_set_buffer(enc, bw, 0ul, 1)
    metal_set_buffer(enc, by, 0ul, 2)
    metal_set_buffer(enc, bdim, 0ul, 3)
    metal_set_buffer(enc, beps, 0ul, 4)
    // one row, latency-bound — the widest tg shortens the strided-load rounds (1024 max on M1)
    metal_dispatch_threadgroups(enc, uint3(1u, 1u, 1u), uint3(1024u, 1u, 1u))
}

// fused rope + KV-row store: the v source binds the SAME QKV buffer at its v byte offset
def private enc_rope_store(enc : MetalComputeEncoder?; f16 : bool; bqkv : MetalBuffer?; vsoff : uint64;
                           kmb, vmb : MetalBuffer?; koff, voff : uint64;
                           bcos, bsin, bqd, bhs, bnpq, bnpall : MetalBuffer?; npairs : int64) {
    metal_set_pipeline(enc, f16 ? g_pso_rpst16 : g_pso_rpst32)
    metal_set_buffer(enc, bqkv, 0ul, 0)
    metal_set_buffer(enc, bqkv, vsoff, 1)
    metal_set_buffer(enc, kmb, koff, 2)
    metal_set_buffer(enc, vmb, voff, 3)
    metal_set_buffer(enc, bcos, 0ul, 4)
    metal_set_buffer(enc, bsin, 0ul, 5)
    metal_set_buffer(enc, bqd, 0ul, 6)
    metal_set_buffer(enc, bhs, 0ul, 7)
    metal_set_buffer(enc, bnpq, 0ul, 8)
    metal_set_buffer(enc, bnpall, 0ul, 9)
    metal_dispatch_threadgroups(enc, uint3(uint((npairs + 63l) / 64l), 1u, 1u), uint3(64u, 1u, 1u))
}

// q8_0-mirror rope+store: the K/V mirror rows bind TWICE each (half-scale + byte views at the
// same row offset); grid covers the q pairs + one thread per K/V 32-block
def private enc_rope_store_q8(enc : MetalComputeEncoder?; bqkv : MetalBuffer?; vsoff : uint64;
                              kmb, vmb : MetalBuffer?; koff, voff : uint64;
                              bcos, bsin, bqd, bhs, bnpq, bnblk : MetalBuffer?; nthreads : int64) {
    metal_set_pipeline(enc, g_pso_rpst_q8)
    metal_set_buffer(enc, bqkv, 0ul, 0)
    metal_set_buffer(enc, bqkv, vsoff, 1)
    metal_set_buffer(enc, kmb, koff, 2)
    metal_set_buffer(enc, kmb, koff, 3)
    metal_set_buffer(enc, vmb, voff, 4)
    metal_set_buffer(enc, vmb, voff, 5)
    metal_set_buffer(enc, bcos, 0ul, 6)
    metal_set_buffer(enc, bsin, 0ul, 7)
    metal_set_buffer(enc, bqd, 0ul, 8)
    metal_set_buffer(enc, bhs, 0ul, 9)
    metal_set_buffer(enc, bnpq, 0ul, 10)
    metal_set_buffer(enc, bnblk, 0ul, 11)
    metal_dispatch_threadgroups(enc, uint3(uint((nthreads + 63l) / 64l), 1u, 1u), uint3(64u, 1u, 1u))
}

// the fused QKV+rope+store dispatch (2 rows per simdgroup — grid covers ddim/2 pairs)
def private enc_gemv_qkv_rs(enc : MetalComputeEncoder?; f16 : bool; bwq, bws, bx, bq : MetalBuffer?;
                            kmb, vmb : MetalBuffer?; koff, voff : uint64;
                            bcos, bsin, bn, bd, bqd, bkvd, bhs : MetalBuffer?; d : int64) {
    metal_set_pipeline(enc, f16 ? g_pso_qkvrs16 : g_pso_qkvrs32)
    metal_set_buffer(enc, bwq, 0ul, 0)
    metal_set_buffer(enc, bws, 0ul, 1)
    metal_set_buffer(enc, bx, 0ul, 2)
    metal_set_buffer(enc, bq, 0ul, 3)
    metal_set_buffer(enc, kmb, koff, 4)
    metal_set_buffer(enc, vmb, voff, 5)
    metal_set_buffer(enc, bcos, 0ul, 6)
    metal_set_buffer(enc, bsin, 0ul, 7)
    metal_set_buffer(enc, bn, 0ul, 8)
    metal_set_buffer(enc, bd, 0ul, 9)
    metal_set_buffer(enc, bqd, 0ul, 10)
    metal_set_buffer(enc, bkvd, 0ul, 11)
    metal_set_buffer(enc, bhs, 0ul, 12)
    let pairs = d / 2l
    metal_dispatch_threadgroups(enc, uint3(uint((pairs + 3l) / 4l), 1u, 1u), uint3(128u, 1u, 1u))
}

// the fused W1|W3+swiglu dispatch (2 rows per simdgroup — gate + up of one output element)
def private enc_gemv_w13sw(enc : MetalComputeEncoder?; bwq, bws, bx, bh : MetalBuffer?; bn, bhid : MetalBuffer?; hidden : int64) {
    metal_set_pipeline(enc, g_pso_w13sw)
    metal_set_buffer(enc, bwq, 0ul, 0)
    metal_set_buffer(enc, bws, 0ul, 1)
    metal_set_buffer(enc, bx, 0ul, 2)
    metal_set_buffer(enc, bh, 0ul, 3)
    metal_set_buffer(enc, bn, 0ul, 4)
    metal_set_buffer(enc, bhid, 0ul, 5)
    metal_dispatch_threadgroups(enc, uint3(uint((hidden + 3l) / 4l), 1u, 1u), uint3(128u, 1u, 1u))
}

def private enc_argmax(enc : MetalComputeEncoder?; blg, btok, bn : MetalBuffer?) {
    metal_set_pipeline(enc, g_pso_argmax)
    metal_set_threadgroup_memory_length(enc, metal_argmax_msl_tgmem, 0)
    metal_set_buffer(enc, blg, 0ul, 0)
    metal_set_buffer(enc, btok, 0ul, 1)
    metal_set_buffer(enc, bn, 0ul, 2)
    metal_dispatch_threadgroups(enc, uint3(1u, 1u, 1u), uint3(1024u, 1u, 1u))
}

def private enc_embed(enc : MetalComputeEncoder?; s16 : bool; bq, bs, btok, bx, bdim : MetalBuffer?; dim : int64) {
    metal_set_pipeline(enc, s16 ? g_pso_emb16 : g_pso_emb)
    metal_set_buffer(enc, bq, 0ul, 0)
    metal_set_buffer(enc, bs, 0ul, 1)
    metal_set_buffer(enc, btok, 0ul, 2)
    metal_set_buffer(enc, bx, 0ul, 3)
    metal_set_buffer(enc, bdim, 0ul, 4)
    metal_dispatch_threadgroups(enc, uint3(uint((dim + 63l) / 64l), 1u, 1u), uint3(64u, 1u, 1u))
}

// fused add + rms when dim fits the kernel's row slab, the unfused pair otherwise
def private enc_add_rms_site(enc : MetalComputeEncoder?; bx, ba, bw, by, bdim, beps : MetalBuffer?; dim : int64) {
    if (dim <= ADD_RMS_MAX_DIM) {
        metal_set_pipeline(enc, g_pso_addrms)
        metal_set_threadgroup_memory_length(enc, metal_add_rms_msl_tgmem, 0)
        metal_set_buffer(enc, bx, 0ul, 0)
        metal_set_buffer(enc, ba, 0ul, 1)
        metal_set_buffer(enc, bw, 0ul, 2)
        metal_set_buffer(enc, by, 0ul, 3)
        metal_set_buffer(enc, bdim, 0ul, 4)
        metal_set_buffer(enc, beps, 0ul, 5)
        metal_dispatch_threadgroups(enc, uint3(1u, 1u, 1u), uint3(1024u, 1u, 1u))
    } else {
        enc_ew2(enc, g_pso_add, bx, 0ul, ba, 0ul, bdim, dim)
        enc_rms(enc, bx, bw, by, bdim, beps)
    }
}

def private enc_attn(enc : MetalComputeEncoder?; f16 : bool; kmb, vmb : MetalBuffer?; koff, voff : uint64;
                     bq, by, bcnt, bkvd, bnh, bkvm, bhs, bscale : MetalBuffer?; nheads, head_size : int64) {
    metal_set_pipeline(enc, f16 ? g_pso_attn16 : g_pso_attn32)
    metal_set_threadgroup_memory_length(enc, f16 ? metal_sq_attn16_msl_tgmem : metal_sq_attn32_msl_tgmem, 0)
    metal_set_buffer(enc, kmb, koff, 0)
    metal_set_buffer(enc, vmb, voff, 1)
    metal_set_buffer(enc, bq, 0ul, 2)
    metal_set_buffer(enc, by, 0ul, 3)
    metal_set_buffer(enc, bcnt, 0ul, 4)
    metal_set_buffer(enc, bkvd, 0ul, 5)
    metal_set_buffer(enc, bnh, 0ul, 6)
    metal_set_buffer(enc, bkvm, 0ul, 7)
    metal_set_buffer(enc, bhs, 0ul, 8)
    metal_set_buffer(enc, bscale, 0ul, 9)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

def private enc_attn_part(enc : MetalComputeEncoder?; f16 : bool; kmb, vmb : MetalBuffer?; koff, voff : uint64;
                          bq, bpart, bcnt, bkvd, bnh, bkvm, bhs, bscale, bnch : MetalBuffer?; ntg, head_size : int64) {
    metal_set_pipeline(enc, f16 ? g_pso_attnp16 : g_pso_attnp32)
    metal_set_threadgroup_memory_length(enc, f16 ? metal_sq_attn_part16_msl_tgmem : metal_sq_attn_part32_msl_tgmem, 0)
    metal_set_buffer(enc, kmb, koff, 0)
    metal_set_buffer(enc, vmb, voff, 1)
    metal_set_buffer(enc, bq, 0ul, 2)
    metal_set_buffer(enc, bpart, 0ul, 3)
    metal_set_buffer(enc, bcnt, 0ul, 4)
    metal_set_buffer(enc, bkvd, 0ul, 5)
    metal_set_buffer(enc, bnh, 0ul, 6)
    metal_set_buffer(enc, bkvm, 0ul, 7)
    metal_set_buffer(enc, bhs, 0ul, 8)
    metal_set_buffer(enc, bscale, 0ul, 9)
    metal_set_buffer(enc, bnch, 0ul, 10)
    metal_dispatch_threadgroups(enc, uint3(uint(ntg), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

// q8_0-mirror attention twins: the K/V slabs bind TWICE each (half-scale + byte views at the
// same layer offset); nblk (blocks per row) replaces the kv_dim uniform
def private enc_attn_q8(enc : MetalComputeEncoder?; kmb, vmb : MetalBuffer?; koff, voff : uint64;
                        bq, by, bcnt, bnblk, bnh, bkvm, bhs, bscale : MetalBuffer?; nheads, head_size : int64) {
    metal_set_pipeline(enc, g_pso_attn_q8)
    metal_set_threadgroup_memory_length(enc, metal_sq_attn_q8_msl_tgmem, 0)
    metal_set_buffer(enc, kmb, koff, 0)
    metal_set_buffer(enc, kmb, koff, 1)
    metal_set_buffer(enc, vmb, voff, 2)
    metal_set_buffer(enc, vmb, voff, 3)
    metal_set_buffer(enc, bq, 0ul, 4)
    metal_set_buffer(enc, by, 0ul, 5)
    metal_set_buffer(enc, bcnt, 0ul, 6)
    metal_set_buffer(enc, bnblk, 0ul, 7)
    metal_set_buffer(enc, bnh, 0ul, 8)
    metal_set_buffer(enc, bkvm, 0ul, 9)
    metal_set_buffer(enc, bhs, 0ul, 10)
    metal_set_buffer(enc, bscale, 0ul, 11)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

def private enc_attn_part_q8(enc : MetalComputeEncoder?; kmb, vmb : MetalBuffer?; koff, voff : uint64;
                             bq, bpart, bcnt, bnblk, bnh, bkvm, bhs, bscale, bnch : MetalBuffer?; ntg, head_size : int64) {
    metal_set_pipeline(enc, g_pso_attnp_q8)
    metal_set_threadgroup_memory_length(enc, metal_sq_attn_part_q8_msl_tgmem, 0)
    metal_set_buffer(enc, kmb, koff, 0)
    metal_set_buffer(enc, kmb, koff, 1)
    metal_set_buffer(enc, vmb, voff, 2)
    metal_set_buffer(enc, vmb, voff, 3)
    metal_set_buffer(enc, bq, 0ul, 4)
    metal_set_buffer(enc, bpart, 0ul, 5)
    metal_set_buffer(enc, bcnt, 0ul, 6)
    metal_set_buffer(enc, bnblk, 0ul, 7)
    metal_set_buffer(enc, bnh, 0ul, 8)
    metal_set_buffer(enc, bkvm, 0ul, 9)
    metal_set_buffer(enc, bhs, 0ul, 10)
    metal_set_buffer(enc, bscale, 0ul, 11)
    metal_set_buffer(enc, bnch, 0ul, 12)
    metal_dispatch_threadgroups(enc, uint3(uint(ntg), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

def private enc_attn_comb(enc : MetalComputeEncoder?; bpart, by, bnch, bnh, bhs : MetalBuffer?; nheads, head_size : int64) {
    metal_set_pipeline(enc, g_pso_attncomb)
    metal_set_buffer(enc, bpart, 0ul, 0)
    metal_set_buffer(enc, by, 0ul, 1)
    metal_set_buffer(enc, bnch, 0ul, 2)
    metal_set_buffer(enc, bnh, 0ul, 3)
    metal_set_buffer(enc, bhs, 0ul, 4)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

// swiglu and add share the (a, b, total) elementwise shape; offsets serve the fused-buffer binds
def private enc_ew2(enc : MetalComputeEncoder?; pso : MetalComputePipeline?; ba : MetalBuffer?; aoff : uint64; bb : MetalBuffer?; boff : uint64; btot : MetalBuffer?; total : int64) {
    metal_set_pipeline(enc, pso)
    metal_set_buffer(enc, ba, aoff, 0)
    metal_set_buffer(enc, bb, boff, 1)
    metal_set_buffer(enc, btot, 0ul, 2)
    metal_dispatch_threadgroups(enc, uint3(uint((total + 63l) / 64l), 1u, 1u), uint3(64u, 1u, 1u))
}

// ===== the batched-step encoders (P4) =====

// fixed-B batched GEMV (the B2 form serves B <= 2, B4 serves 3..4 — pad streams read stale rows,
// whose outputs are never consumed); the cat regions bind at byte offsets, scales through both
// views (the s16 uniform picks). Geometry pinned at 4 rows/tg — the x-staged cooperative-load
// slot math bakes 128 threads in (the g_gemv_rows knob stays single-stream-only).
def private enc_gemv_b(enc : MetalComputeEncoder?; four : bool; bwq : MetalBuffer?; wqoff : uint64;
                       bws : MetalBuffer?; wsoff : uint64; bx, by, bn, bd, bs16 : MetalBuffer?; d : int64) {
    metal_set_pipeline(enc, four ? g_pso_gemv_b4 : g_pso_gemv_b2)
    metal_set_threadgroup_memory_length(enc, four ? metal_gemv_b4_msl_tgmem : metal_gemv_b2_msl_tgmem, 0)
    metal_set_buffer(enc, bwq, wqoff, 0)
    metal_set_buffer(enc, bws, wsoff, 1)
    metal_set_buffer(enc, bws, wsoff, 2)
    metal_set_buffer(enc, bx, 0ul, 3)
    metal_set_buffer(enc, by, 0ul, 4)
    metal_set_buffer(enc, bn, 0ul, 5)
    metal_set_buffer(enc, bd, 0ul, 6)
    metal_set_buffer(enc, bs16, 0ul, 7)
    metal_dispatch_threadgroups(enc, uint3(uint((d + 3l) / 4l), 1u, 1u), uint3(128u, 1u, 1u))
}

// the mul_mv-shape fixed-B GEMV on an interleaved 34B blob (dual views of the SAME buffer);
// bxs = x row stride in float4s; bnr = the nrows uniform, four-form only; xoff/yoff = the
// column-tile byte offsets (B > 4 streams the weights once per 4-column tile)
def private enc_mv_b(enc : MetalComputeEncoder?; four : bool; bwblob, bx, by, bn, bd, bxs : MetalBuffer?; d : int64;
                     bnr : MetalBuffer? = null; xoff : uint64 = 0ul; yoff : uint64 = 0ul) {
    metal_set_pipeline(enc, four ? g_pso_mv_b4 : g_pso_mv_b2)
    metal_set_buffer(enc, bwblob, 0ul, 0)
    metal_set_buffer(enc, bwblob, 0ul, 1)
    metal_set_buffer(enc, bx, xoff, 2)
    metal_set_buffer(enc, by, yoff, 3)
    metal_set_buffer(enc, bn, 0ul, 4)
    metal_set_buffer(enc, bd, 0ul, 5)
    if (four) {
        metal_set_buffer(enc, bnr, 0ul, 6)
        metal_set_buffer(enc, bxs, 0ul, 7)
    } else {
        metal_set_buffer(enc, bxs, 0ul, 6)
    }
    // b2 = 4 rows/tg (16 lanes/row); b4 = 8 rows/tg (8 lanes/row)
    metal_dispatch_threadgroups(enc, uint3(uint(four ? (d + 7l) / 8l : (d + 3l) / 4l), 1u, 1u), uint3(64u, 1u, 1u))
}

// one mv GEMV site: a single form dispatch at B <= 4, two 4-column tiles at B = 5..8 (the
// reference's own shape — it never switches to GEMM below B = 9; the weights stream twice but
// stay the fast skinny form). xrowb = the x row stride in bytes; bnr/bnr2 = per-tile nrows.
def private enc_mv_site(enc : MetalComputeEncoder?; bwblob, bx, by, bn, bd, bxs : MetalBuffer?;
                        d, nrows : int64; xrowb : uint64; bnr, bnr2 : MetalBuffer?) {
    enc_mv_b(enc, nrows > 2l, bwblob, bx, by, bn, bd, bxs, d, bnr)
    if (nrows > 4l) {
        enc_mv_b(enc, true, bwblob, bx, by, bn, bd, bxs, d, bnr2, 4ul * xrowb, uint64(4l * d * 4l))
    }
}

// fused fixed-B W1|W3 GEMV + swiglu — bwq/bws are the cat W1|W3 region at offset 0 (up rows sit
// at +hidden internally), by is the [B x hidden] swiglu output. Replaces the w1 + w3 GEMVs + the
// swiglu ew2 with one dispatch. Same 4-rows/tg geometry, gridded over the hidden output rows.
def private enc_gemv_w13sw_b(enc : MetalComputeEncoder?; four : bool; bwq, bws, bx, by, bn, bhid, bs16 : MetalBuffer?; hidden : int64) {
    metal_set_pipeline(enc, four ? g_pso_w13sw_b4 : g_pso_w13sw_b2)
    metal_set_threadgroup_memory_length(enc, four ? metal_gemv_w13sw_b4_msl_tgmem : metal_gemv_w13sw_b2_msl_tgmem, 0)
    metal_set_buffer(enc, bwq, 0ul, 0)
    metal_set_buffer(enc, bws, 0ul, 1)
    metal_set_buffer(enc, bws, 0ul, 2)
    metal_set_buffer(enc, bx, 0ul, 3)
    metal_set_buffer(enc, by, 0ul, 4)
    metal_set_buffer(enc, bn, 0ul, 5)
    metal_set_buffer(enc, bhid, 0ul, 6)
    metal_set_buffer(enc, bs16, 0ul, 7)
    metal_dispatch_threadgroups(enc, uint3(uint((hidden + 3l) / 4l), 1u, 1u), uint3(128u, 1u, 1u))
}

// the production mul_mm dispatch (the prefill kernel: 34B-blob dual-view W, raw f32 X): y binds
// at a byte offset so per-tensor tiles land inside a wider fused row — the ndim uniform is the
// y ROW STRIDE while this tensor's output width d shapes the grid (d % 64 == 0, mp % 32 == 0)
def private enc_gemm_mm_b(enc : MetalComputeEncoder?; bwblob, bx, by : MetalBuffer?; yoff : uint64;
                          bk, bn : MetalBuffer?; mp, d : int64) {
    metal_set_pipeline(enc, g_pso_mm)
    metal_set_threadgroup_memory_length(enc, metal_q8_mulmm_msl_tgmem, 0)
    metal_set_buffer(enc, bwblob, 0ul, 0)
    metal_set_buffer(enc, bwblob, 0ul, 1)
    metal_set_buffer(enc, bx, 0ul, 2)
    metal_set_buffer(enc, by, yoff, 3)
    metal_set_buffer(enc, bk, 0ul, 4)
    metal_set_buffer(enc, bn, 0ul, 5)
    metal_dispatch_threadgroups(enc, uint3(uint(mp / 32l), uint(d / 64l), 1u), uint3(128u, 1u, 1u))
}

// the M-pad-32 GEMM (B >= 5): raw f32 X, dual-view W scales
def private enc_gemm_b(enc : MetalComputeEncoder?; bwq : MetalBuffer?; wqoff : uint64;
                       bws : MetalBuffer?; wsoff : uint64; bx, by, bk, bn, bs16 : MetalBuffer?; mp32, d : int64) {
    metal_set_pipeline(enc, g_pso_gemm_b)
    metal_set_threadgroup_memory_length(enc, metal_q8_gemm_b_msl_tgmem, 0)
    metal_set_buffer(enc, bwq, wqoff, 0)
    metal_set_buffer(enc, bws, wsoff, 1)
    metal_set_buffer(enc, bws, wsoff, 2)
    metal_set_buffer(enc, bx, 0ul, 3)
    metal_set_buffer(enc, by, 0ul, 4)
    metal_set_buffer(enc, bk, 0ul, 5)
    metal_set_buffer(enc, bn, 0ul, 6)
    metal_set_buffer(enc, bs16, 0ul, 7)
    metal_dispatch_threadgroups(enc, uint3(uint((mp32 / 32l) * (d / 32l)), 1u, 1u), uint3(128u, 1u, 1u))
}

// the split-K pair (per-site policy): the GEMM's grid.y is the k-slice axis, partial planes land
// in bpart, and the reduce sums them into y — the serial encoder orders the two dispatches.
// Planes and grids assume mp == 32 (the policy gate).
def private enc_gemm_sk_b(enc : MetalComputeEncoder?; bwq : MetalBuffer?; wqoff : uint64;
                          bws : MetalBuffer?; wsoff : uint64; bx, bpart, bk, bn, bs16, bsk : MetalBuffer?; d, nsk : int64) {
    metal_set_pipeline(enc, g_pso_gemm_b_sk)
    metal_set_threadgroup_memory_length(enc, metal_q8_gemm_b_sk_msl_tgmem, 0)
    metal_set_buffer(enc, bwq, wqoff, 0)
    metal_set_buffer(enc, bws, wsoff, 1)
    metal_set_buffer(enc, bws, wsoff, 2)
    metal_set_buffer(enc, bx, 0ul, 3)
    metal_set_buffer(enc, bpart, 0ul, 4)
    metal_set_buffer(enc, bk, 0ul, 5)
    metal_set_buffer(enc, bn, 0ul, 6)
    metal_set_buffer(enc, bs16, 0ul, 7)
    metal_set_buffer(enc, bsk, 0ul, 8)
    metal_dispatch_threadgroups(enc, uint3(uint(d / 32l), uint(nsk), 1u), uint3(128u, 1u, 1u))
}

def private enc_sk_reduce_b(enc : MetalComputeEncoder?; bpart, by, btot, bsk : MetalBuffer?; total : int64) {
    metal_set_pipeline(enc, g_pso_skred)
    metal_set_buffer(enc, bpart, 0ul, 0)
    metal_set_buffer(enc, by, 0ul, 1)
    metal_set_buffer(enc, btot, 0ul, 2)
    metal_set_buffer(enc, bsk, 0ul, 3)
    metal_dispatch_threadgroups(enc, uint3(uint((total + 127l) / 128l), 1u, 1u), uint3(128u, 1u, 1u))
}

// the 64-wide N tile (classifier site): the caller gates d % 64 == 0
def private enc_gemm64_b(enc : MetalComputeEncoder?; bwq : MetalBuffer?; wqoff : uint64;
                         bws : MetalBuffer?; wsoff : uint64; bx, by, bk, bn, bs16 : MetalBuffer?; mp32, d : int64) {
    metal_set_pipeline(enc, g_pso_gemm64_b)
    metal_set_threadgroup_memory_length(enc, metal_q8_gemm64_b_msl_tgmem, 0)
    metal_set_buffer(enc, bwq, wqoff, 0)
    metal_set_buffer(enc, bws, wsoff, 1)
    metal_set_buffer(enc, bws, wsoff, 2)
    metal_set_buffer(enc, bx, 0ul, 3)
    metal_set_buffer(enc, by, 0ul, 4)
    metal_set_buffer(enc, bk, 0ul, 5)
    metal_set_buffer(enc, bn, 0ul, 6)
    metal_set_buffer(enc, bs16, 0ul, 7)
    metal_dispatch_threadgroups(enc, uint3(uint((mp32 / 32l) * (d / 64l)), 1u, 1u), uint3(128u, 1u, 1u))
}

// Per-encoder buffer-hazard tracker for the concurrent-encode rail (g_bconcurrent). Each scratch
// buffer owns one bit; a dispatch declares the buffers it reads (rmask) and writes (wmask). A
// barrier is inserted iff the dispatch would read a pending write (RAW), or write a pending write
// (WAW) or a pending read (WAR) — read-after-read runs concurrently. Weights/uniforms are
// read-only and never tracked (a read of them can never hazard). The tracker resets after each
// barrier and at every command-buffer boundary (a fresh encoder + queue order already serialize).
let HZ_BX     = 1u
let HZ_BXB    = 2u
let HZ_BXB2   = 4u
let HZ_BQKV   = 8u
let HZ_BHB    = 16u
let HZ_BHB2   = 32u
let HZ_BPART  = 64u
let HZ_ARENA  = 128u
let HZ_BPSK   = 256u
let HZ_BXF    = 512u
let HZ_BLOG   = 1024u

struct private HazTrack {
    on : bool
    w : uint    // buffers with a pending (unbarriered) write
    r : uint    // buffers with a pending read
}

def private bar_dep(enc : MetalComputeEncoder?; var h : HazTrack; rmask, wmask : uint) {
    if (!h.on) {
        return
    }
    if ((rmask & h.w) != 0u || (wmask & h.w) != 0u || (wmask & h.r) != 0u) {
        metal_memory_barrier(enc)
        h.w = 0u
        h.r = 0u
    }
    h.r |= rmask
    h.w |= wmask
}

def private enc_attn_b(enc : MetalComputeEncoder?; f16 : bool; bq, by, brt, blay, bkvd, bnh, bkvm, bhs, bscale, bqs : MetalBuffer?; nrows, nheads, head_size : int64) {
    metal_set_pipeline(enc, f16 ? g_pso_attnb16 : g_pso_attnb32)
    metal_set_threadgroup_memory_length(enc, f16 ? metal_sq_attn_b16_msl_tgmem : metal_sq_attn_b32_msl_tgmem, 0)
    metal_set_buffer(enc, g_arena, 0ul, 0)
    metal_set_buffer(enc, g_arena, 0ul, 1)
    metal_set_buffer(enc, bq, 0ul, 2)
    metal_set_buffer(enc, by, 0ul, 3)
    metal_set_buffer(enc, brt, 0ul, 4)
    metal_set_buffer(enc, blay, 0ul, 5)
    metal_set_buffer(enc, bkvd, 0ul, 6)
    metal_set_buffer(enc, bnh, 0ul, 7)
    metal_set_buffer(enc, bkvm, 0ul, 8)
    metal_set_buffer(enc, bhs, 0ul, 9)
    metal_set_buffer(enc, bscale, 0ul, 10)
    metal_set_buffer(enc, bqs, 0ul, 11)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads * nrows), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

def private enc_attn_part_b(enc : MetalComputeEncoder?; f16 : bool; bq, bpart, brt, blay, bkvd, bnh, bkvm, bhs, bscale, bqs, bnch : MetalBuffer?; nrows, nchmax, nheads, head_size : int64) {
    metal_set_pipeline(enc, f16 ? g_pso_attnpb16 : g_pso_attnpb32)
    metal_set_threadgroup_memory_length(enc, f16 ? metal_sq_attn_part_b16_msl_tgmem : metal_sq_attn_part_b32_msl_tgmem, 0)
    metal_set_buffer(enc, g_arena, 0ul, 0)
    metal_set_buffer(enc, g_arena, 0ul, 1)
    metal_set_buffer(enc, bq, 0ul, 2)
    metal_set_buffer(enc, bpart, 0ul, 3)
    metal_set_buffer(enc, brt, 0ul, 4)
    metal_set_buffer(enc, blay, 0ul, 5)
    metal_set_buffer(enc, bkvd, 0ul, 6)
    metal_set_buffer(enc, bnh, 0ul, 7)
    metal_set_buffer(enc, bkvm, 0ul, 8)
    metal_set_buffer(enc, bhs, 0ul, 9)
    metal_set_buffer(enc, bscale, 0ul, 10)
    metal_set_buffer(enc, bqs, 0ul, 11)
    metal_set_buffer(enc, bnch, 0ul, 12)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads * nchmax * nrows), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

// q8_0-mirror batch attention twins: the arena binds FOUR times (half-scale + byte views for K
// and V); the row table carries byte offsets and nblk replaces the kv_dim uniform
def private enc_attn_b_q8(enc : MetalComputeEncoder?; bq, by, brt, blay, bnblk, bnh, bkvm, bhs, bscale, bqs : MetalBuffer?; nrows, nheads, head_size : int64) {
    metal_set_pipeline(enc, g_pso_attnb_q8)
    metal_set_threadgroup_memory_length(enc, metal_sq_attn_b_q8_msl_tgmem, 0)
    metal_set_buffer(enc, g_arena, 0ul, 0)
    metal_set_buffer(enc, g_arena, 0ul, 1)
    metal_set_buffer(enc, g_arena, 0ul, 2)
    metal_set_buffer(enc, g_arena, 0ul, 3)
    metal_set_buffer(enc, bq, 0ul, 4)
    metal_set_buffer(enc, by, 0ul, 5)
    metal_set_buffer(enc, brt, 0ul, 6)
    metal_set_buffer(enc, blay, 0ul, 7)
    metal_set_buffer(enc, bnblk, 0ul, 8)
    metal_set_buffer(enc, bnh, 0ul, 9)
    metal_set_buffer(enc, bkvm, 0ul, 10)
    metal_set_buffer(enc, bhs, 0ul, 11)
    metal_set_buffer(enc, bscale, 0ul, 12)
    metal_set_buffer(enc, bqs, 0ul, 13)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads * nrows), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

def private enc_attn_part_b_q8(enc : MetalComputeEncoder?; bq, bpart, brt, blay, bnblk, bnh, bkvm, bhs, bscale, bqs, bnch : MetalBuffer?; nrows, nchmax, nheads, head_size : int64) {
    metal_set_pipeline(enc, g_pso_attnpb_q8)
    metal_set_threadgroup_memory_length(enc, metal_sq_attn_part_b_q8_msl_tgmem, 0)
    metal_set_buffer(enc, g_arena, 0ul, 0)
    metal_set_buffer(enc, g_arena, 0ul, 1)
    metal_set_buffer(enc, g_arena, 0ul, 2)
    metal_set_buffer(enc, g_arena, 0ul, 3)
    metal_set_buffer(enc, bq, 0ul, 4)
    metal_set_buffer(enc, bpart, 0ul, 5)
    metal_set_buffer(enc, brt, 0ul, 6)
    metal_set_buffer(enc, blay, 0ul, 7)
    metal_set_buffer(enc, bnblk, 0ul, 8)
    metal_set_buffer(enc, bnh, 0ul, 9)
    metal_set_buffer(enc, bkvm, 0ul, 10)
    metal_set_buffer(enc, bhs, 0ul, 11)
    metal_set_buffer(enc, bscale, 0ul, 12)
    metal_set_buffer(enc, bqs, 0ul, 13)
    metal_set_buffer(enc, bnch, 0ul, 14)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads * nchmax * nrows), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

def private enc_attn_d_q8(enc : MetalComputeEncoder?; bq, bpart, brt, blay, bnblk, bnh, bkvm, bhs, bscale, bqs, bnsg : MetalBuffer?; nsgs, nrows, nheads : int64) {
    metal_set_pipeline(enc, g_pso_attnd_q8)
    metal_set_threadgroup_memory_length(enc, metal_sq_attn_d_q8_msl_tgmem, 0)
    metal_set_buffer(enc, g_arena, 0ul, 0)
    metal_set_buffer(enc, g_arena, 0ul, 1)
    metal_set_buffer(enc, g_arena, 0ul, 2)
    metal_set_buffer(enc, g_arena, 0ul, 3)
    metal_set_buffer(enc, bq, 0ul, 4)
    metal_set_buffer(enc, bpart, 0ul, 5)
    metal_set_buffer(enc, brt, 0ul, 6)
    metal_set_buffer(enc, blay, 0ul, 7)
    metal_set_buffer(enc, bnblk, 0ul, 8)
    metal_set_buffer(enc, bnh, 0ul, 9)
    metal_set_buffer(enc, bkvm, 0ul, 10)
    metal_set_buffer(enc, bhs, 0ul, 11)
    metal_set_buffer(enc, bscale, 0ul, 12)
    metal_set_buffer(enc, bqs, 0ul, 13)
    metal_set_buffer(enc, bnsg, 0ul, 14)
    // HEAD-FASTEST launch: consecutive tgs share position rows (SLC locality at depth)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads), uint((nsgs + 3l) / 4l), uint(nrows)), uint3(128u, 1u, 1u))
}

def private enc_attn_comb_b(enc : MetalComputeEncoder?; bpart, by, brt, bnch, bnh, bhs : MetalBuffer?; nrows, nheads, head_size : int64) {
    metal_set_pipeline(enc, g_pso_attncombb)
    metal_set_buffer(enc, bpart, 0ul, 0)
    metal_set_buffer(enc, by, 0ul, 1)
    metal_set_buffer(enc, brt, 0ul, 2)
    metal_set_buffer(enc, bnch, 0ul, 3)
    metal_set_buffer(enc, bnh, 0ul, 4)
    metal_set_buffer(enc, bhs, 0ul, 5)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads * nrows), 1u, 1u), uint3(uint(head_size), 1u, 1u))
}

def private enc_attn_d(enc : MetalComputeEncoder?; f16 : bool; bq, bpart, brt, blay, bkvd, bnh, bkvm, bhs, bscale, bqs, bnsg : MetalBuffer?; nsgs, nrows, nheads : int64) {
    metal_set_pipeline(enc, f16 ? g_pso_attnd16 : g_pso_attnd32)
    metal_set_threadgroup_memory_length(enc, f16 ? metal_sq_attn_d16_msl_tgmem : metal_sq_attn_d32_msl_tgmem, 0)
    metal_set_buffer(enc, g_arena, 0ul, 0)
    metal_set_buffer(enc, g_arena, 0ul, 1)
    metal_set_buffer(enc, bq, 0ul, 2)
    metal_set_buffer(enc, bpart, 0ul, 3)
    metal_set_buffer(enc, brt, 0ul, 4)
    metal_set_buffer(enc, blay, 0ul, 5)
    metal_set_buffer(enc, bkvd, 0ul, 6)
    metal_set_buffer(enc, bnh, 0ul, 7)
    metal_set_buffer(enc, bkvm, 0ul, 8)
    metal_set_buffer(enc, bhs, 0ul, 9)
    metal_set_buffer(enc, bscale, 0ul, 10)
    metal_set_buffer(enc, bqs, 0ul, 11)
    metal_set_buffer(enc, bnsg, 0ul, 12)
    // HEAD-FASTEST launch: consecutive tgs share position rows (SLC locality at depth)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads), uint((nsgs + 3l) / 4l), uint(nrows)), uint3(128u, 1u, 1u))
}

def private enc_attn_comb_d(enc : MetalComputeEncoder?; bpart, by, brt, bnsg, bnh, bhs : MetalBuffer?; nrows, nheads, head_size : int64) {
    metal_set_pipeline(enc, g_pso_attncombd)
    metal_set_threadgroup_memory_length(enc, metal_sq_attn_comb_d_msl_tgmem, 0)
    metal_set_buffer(enc, bpart, 0ul, 0)
    metal_set_buffer(enc, by, 0ul, 1)
    metal_set_buffer(enc, brt, 0ul, 2)
    metal_set_buffer(enc, bnsg, 0ul, 3)
    metal_set_buffer(enc, bnh, 0ul, 4)
    metal_set_buffer(enc, bhs, 0ul, 5)
    metal_dispatch_threadgroups(enc, uint3(uint(nheads), uint(nrows), 1u), uint3(uint(head_size), 1u, 1u))
}

def private enc_rope_store_b(enc : MetalComputeEncoder?; f16 : bool; bqkv, bcos, bsin, brt, blay, bkvd, bqd, bhs, bnpq, bnpall, bqs : MetalBuffer?; npairs, nrows : int64) {
    metal_set_pipeline(enc, f16 ? g_pso_rpstb16 : g_pso_rpstb32)
    metal_set_buffer(enc, bqkv, 0ul, 0)
    metal_set_buffer(enc, g_arena, 0ul, 1)
    metal_set_buffer(enc, g_arena, 0ul, 2)
    metal_set_buffer(enc, bcos, 0ul, 3)
    metal_set_buffer(enc, bsin, 0ul, 4)
    metal_set_buffer(enc, brt, 0ul, 5)
    metal_set_buffer(enc, blay, 0ul, 6)
    metal_set_buffer(enc, bkvd, 0ul, 7)
    metal_set_buffer(enc, bqd, 0ul, 8)
    metal_set_buffer(enc, bhs, 0ul, 9)
    metal_set_buffer(enc, bnpq, 0ul, 10)
    metal_set_buffer(enc, bnpall, 0ul, 11)
    metal_set_buffer(enc, bqs, 0ul, 12)
    metal_dispatch_threadgroups(enc, uint3(uint((npairs + 63l) / 64l), uint(nrows), 1u), uint3(64u, 1u, 1u))
}

// q8_0-mirror batched rope+store: arena bound four ways (K/V x scale/byte views); grid covers
// the q pairs + one thread per K/V 32-block, one grid row per batch row
def private enc_rope_store_b_q8(enc : MetalComputeEncoder?; bqkv, bcos, bsin, brt, blay, bnblk, bqd, bhs, bnpq, bqs : MetalBuffer?; nthreads, nrows : int64) {
    metal_set_pipeline(enc, g_pso_rpstb_q8)
    metal_set_buffer(enc, bqkv, 0ul, 0)
    metal_set_buffer(enc, g_arena, 0ul, 1)
    metal_set_buffer(enc, g_arena, 0ul, 2)
    metal_set_buffer(enc, g_arena, 0ul, 3)
    metal_set_buffer(enc, g_arena, 0ul, 4)
    metal_set_buffer(enc, bcos, 0ul, 5)
    metal_set_buffer(enc, bsin, 0ul, 6)
    metal_set_buffer(enc, brt, 0ul, 7)
    metal_set_buffer(enc, blay, 0ul, 8)
    metal_set_buffer(enc, bnblk, 0ul, 9)
    metal_set_buffer(enc, bqd, 0ul, 10)
    metal_set_buffer(enc, bhs, 0ul, 11)
    metal_set_buffer(enc, bnpq, 0ul, 12)
    metal_set_buffer(enc, bqs, 0ul, 13)
    metal_dispatch_threadgroups(enc, uint3(uint((nthreads + 63l) / 64l), uint(nrows), 1u), uint3(64u, 1u, 1u))
}

def private enc_add_rms_b(enc : MetalComputeEncoder?; bx, ba, bw, by, bdim, beps : MetalBuffer?; nrows : int64) {
    metal_set_pipeline(enc, g_pso_addrmsb)
    metal_set_threadgroup_memory_length(enc, metal_add_rms_b_msl_tgmem, 0)
    metal_set_buffer(enc, bx, 0ul, 0)
    metal_set_buffer(enc, ba, 0ul, 1)
    metal_set_buffer(enc, bw, 0ul, 2)
    metal_set_buffer(enc, by, 0ul, 3)
    metal_set_buffer(enc, bdim, 0ul, 4)
    metal_set_buffer(enc, beps, 0ul, 5)
    metal_dispatch_threadgroups(enc, uint3(uint(nrows), 1u, 1u), uint3(1024u, 1u, 1u))
}

// batched rmsnorm (the prologue): the prefill rms kernel is width-agnostic — one tg per row
def private enc_rms_b(enc : MetalComputeEncoder?; bx, bw, by, bdim, beps : MetalBuffer?; nrows : int64) {
    metal_set_pipeline(enc, g_pso_rms)
    metal_set_threadgroup_memory_length(enc, metal_rmsnorm_msl_tgmem, 0)
    metal_set_buffer(enc, bx, 0ul, 0)
    metal_set_buffer(enc, bw, 0ul, 1)
    metal_set_buffer(enc, by, 0ul, 2)
    metal_set_buffer(enc, bdim, 0ul, 3)
    metal_set_buffer(enc, beps, 0ul, 4)
    metal_dispatch_threadgroups(enc, uint3(uint(nrows), 1u, 1u), uint3(1024u, 1u, 1u))
}

// ===== the per-step resource bundle + encode-ahead pipelining =====

// Everything one step's command buffer owns. The encode-ahead rail (g_pre) keeps ONE of these
// pre-encoded for the predicted next position; the next forward() call pokes the token-dependent
// bytes (the embed row + that position's rope row — the only inputs unknown at encode time;
// unified memory reads at execution) and COMMITS. Encode cost thus overlaps the PREVIOUS step's
// GPU run instead of sitting on the per-token wall (the P1b framing lab's overlap mode,
// productionized). The cb is deliberately NOT committed at encode time: a committed cb parked
// across an idle gap (a caller that stops decoding — stream switches, interactive pauses) trips
// the GPU watchdog, and the OS then ignores the process's submissions ("prior/excessive GPU
// errors"); an event-parked variant measured ~neutral anyway (Metal defers scheduling until the
// signal). Spec steps commit at encode by design — they chain the previous cb and finish on
// their own.
struct private StepRes {
    valid : bool
    pos : int64
    uid : uint64
    kdt : KVDtype         // the session/mirror KV codec (kdt == vdt gated)
    s16 : bool
    gpu_logits : bool
    chunked : bool
    spec : bool           // opened with GPU argmax + embed gather — no CPU poke, runs immediately
    committed : bool      // spec pres commit at encode; poked pres commit at poke time (a cb
                          // parked committed across an idle gap trips the GPU watchdog)
    cb : MetalCommandBuffer?
    btok : MetalBuffer?   // the argmax winner (spec steps; read back to verify the prediction)
    koff : uint64         // mirror arena offsets captured at encode time (rebuilds can't
    voff : uint64         // intervene: the slow path discards the pre-step before mirror_prepare)
    cap : int64
    bx : MetalBuffer?
    bxb : MetalBuffer?
    bxb2 : MetalBuffer?
    bqkv : MetalBuffer?
    bh12 : MetalBuffer?
    bcos : MetalBuffer?
    bsin : MetalBuffer?
    bpart : MetalBuffer?
    bxf : MetalBuffer?
    blog : MetalBuffer?
    u_dim : MetalBuffer?
    u_qd : MetalBuffer?
    u_kvd : MetalBuffer?
    u_qkvd : MetalBuffer?
    u_hidden : MetalBuffer?
    u_h2 : MetalBuffer?
    u_hs : MetalBuffer?
    u_kvm : MetalBuffer?
    u_nh : MetalBuffer?
    u_cnt : MetalBuffer?
    u_npq : MetalBuffer?
    u_npall : MetalBuffer?
    u_scale : MetalBuffer?
    u_eps : MetalBuffer?
    u_vocab : MetalBuffer?
    u_nch : MetalBuffer?
    u_nblk : MetalBuffer?  // kv_dim/32 — the block-codec kernels' row-width uniform (q8_0 only)
    bytes_x : uint64
    bytes_q : uint64
    bytes_qkv : uint64
    bytes_h2 : uint64
    bytes_tab : uint64
    bytes_part : uint64
    bytes_log : uint64
}

var private g_pre : StepRes                    // the pre-encoded next step, when valid
// the driver-built rope row for the PREDICTED next position (bit-identical to forward()'s)
var private g_rope_cos_next : array<float>
var private g_rope_sin_next : array<float>

// geometry + pooled buffers + uniforms for one step at `pos` — no input bytes yet (poke_step)
def private acquire_step(t : Model; s : Session; pos : int64; koff, voff : uint64; cap : int64) : StepRes {
    let c = t.config
    let dim = c.dim
    let head_size = c.head_size
    let n_heads = c.n_heads
    let qd = n_heads * head_size
    let kv_dim = layer_kv_dim(c, 0l)
    let hidden = layer_hidden(t, 0l)
    let qkvd = qd + 2l * kv_dim
    let h2 = 2l * hidden
    var r = StepRes(valid = true, pos = pos, uid = s.uid,
        kdt = s.kv_dtype_k, s16 = t.wscale_f16,
        koff = koff, voff = voff, cap = cap,
        bytes_x = uint64(dim * 4l), bytes_q = uint64(qd * 4l), bytes_qkv = uint64(qkvd * 4l),
        bytes_h2 = uint64(h2 * 4l), bytes_tab = uint64((head_size / 2l) * 4l),
        bytes_log = uint64(c.vocab_size * 4l))
    r.bx = pool_acquire(g_pool, g_dev, r.bytes_x)
    r.bxb = pool_acquire(g_pool, g_dev, r.bytes_q)   // rms out AND attention out (qd == dim gated)
    r.bxb2 = pool_acquire(g_pool, g_dev, r.bytes_x)
    r.bqkv = pool_acquire(g_pool, g_dev, r.bytes_qkv)   // q | roped-in-place q, k, raw v
    r.bh12 = pool_acquire(g_pool, g_dev, r.bytes_h2)    // w1 gate | swiglu in place, w3 up
    r.bcos = pool_acquire_untracked(g_upool, g_dev, r.bytes_tab)
    r.bsin = pool_acquire_untracked(g_upool, g_dev, r.bytes_tab)
    r.u_dim = uniform_u32(uint(dim))
    r.u_qd = uniform_u32(uint(qd))
    r.u_kvd = uniform_u32(uint(kv_dim))
    r.u_qkvd = uniform_u32(uint(qkvd))
    r.u_hidden = uniform_u32(uint(hidden))
    r.u_h2 = uniform_u32(uint(h2))
    r.u_hs = uniform_u32(uint(head_size))
    r.u_kvm = uniform_u32(uint(n_heads / layer_n_kv_heads(c, 0l)))
    r.u_nh = uniform_u32(uint(n_heads))
    r.u_cnt = uniform_u32(uint(pos + 1l))
    r.u_npq = uniform_u32(uint(qd / 2l))
    r.u_npall = uniform_u32(uint((qd + kv_dim) / 2l))
    r.u_scale = uniform_f32(1.0 / sqrt(float(head_size)))
    r.u_eps = uniform_f32(c.norm_eps)
    if (r.kdt == KVDtype.q8_0) {
        r.u_nblk = uniform_u32(uint(kv_dim / 32l))
    }
    // logits-on-GPU (default ON, DASLLAMA_METAL_LOGITS=0 or the setter pins the CPU classifier):
    // mirrors the prefill gates — q8 untied classifier, no softcap, no suppress pins
    let cls_fmt = c.shared_weights ? t.emb_fmt : t.wcls_fmt
    let tied_f32 = c.shared_weights && !t.cls_q8 && cls_fmt == KqFmt.q8
    r.gpu_logits = (!g_logits_cpu && get_env_variable("DASLLAMA_METAL_LOGITS") != "0" && empty(g_skip) &&
        cls_fmt == KqFmt.q8 && !tied_f32 && c.final_logit_softcap <= 0.0 &&
        empty(t.suppress) && t.wcls_off >= 0l)
    if (r.gpu_logits) {
        r.bxf = pool_acquire(g_pool, g_dev, r.bytes_x)
        r.blog = pool_acquire(g_pool, g_dev, r.bytes_log)
        r.u_vocab = uniform_u32(uint(c.vocab_size))
    }
    // chunked split-K attention above the single-tg depth (128-row chunks, cross-chunk combine)
    r.chunked = pos + 1l > g_attn_single_max
    let nchunks = (pos + 1l + ATTN_CHUNK_ROWS - 1l) / ATTN_CHUNK_ROWS
    r.bytes_part = uint64(n_heads * nchunks * (head_size + 2l) * 4l)
    if (r.chunked) {
        r.bpart = pool_acquire(g_pool, g_dev, r.bytes_part)
        r.u_nch = uniform_u32(uint(nchunks))
    }
    return r
}

// the token-dependent input bytes — the ONLY step state not known at encode time
def private poke_step(s : Session; r : StepRes) {
    unsafe {
        memcpy(metal_buffer_contents(r.bx), addr < void ? > (s.x[0]), int(r.bytes_x))
        memcpy(metal_buffer_contents(r.bcos), addr < void ? > (s.rope_cos[0]), int(r.bytes_tab))
        memcpy(metal_buffer_contents(r.bsin), addr < void ? > (s.rope_sin[0]), int(r.bytes_tab))
    }
}

// encode the whole step (prologue rms, n_layers fused blocks, optional GPU classifier) into a
// fresh command buffer; commit_now submits it (slow + spec paths), otherwise the caller commits
// after poking the inputs (the encode-ahead rail — see StepRes). A spec step OPENS with argmax
// over the previous step's logits + the winner's embed gather into r.bx — hazard tracking
// chains it behind the previous cb, no CPU poke.
def private encode_step(t : Model; var r : StepRes; commit_now : bool; var prev_blog : MetalBuffer? = null) {
    let c = t.config
    let dim = c.dim
    let head_size = c.head_size
    let n_heads = c.n_heads
    let qd = n_heads * head_size
    let kv_dim = layer_kv_dim(c, 0l)
    let hidden = layer_hidden(t, 0l)
    let qkvd = qd + 2l * kv_dim
    let h2 = 2l * hidden
    let pos = r.pos
    let cap = r.cap
    let rowb = kv_row_bytes(r.kdt, kv_dim)   // kdt == vdt gated
    let f16v = r.kdt == KVDtype.f16
    let scalar_kv = r.kdt == KVDtype.f16 || r.kdt == KVDtype.f32   // block codecs take their own kernels
    let nchunks = (pos + 1l + ATTN_CHUNK_ROWS - 1l) / ATTN_CHUNK_ROWS
    // UNRETAINED: skips per-dispatch retain/release of ~2000 resource references at commit +
    // completion — every referenced object outlives the cb here (resident regions live until
    // shutdown; pooled buffers are held by StepRes until after the wait, and the pool never
    // frees recycled buffers before drain)
    var cb = metal_new_command_buffer_unretained(g_queue)
    var enc = metal_new_compute_encoder(cb)
    if (r.spec) {
        // the speculative head: greedy-pick the previous step's logits, gather the winner's
        // embedding row into r.bx (the cls_q8 linear plane — gated in pre_encode_next)
        unsafe {
            var bembq = upload_region(addr < void ? > (t.qblob[t.emb_q8_off]), uint64(c.vocab_size * dim))
            var bembs = upload_scales(t, t.emb_q8_off, c.vocab_size, dim)
            enc_argmax(enc, prev_blog, r.btok, r.u_vocab)
            enc_embed(enc, r.s16, bembq, bembs, r.btok, r.bx, r.u_dim, dim)
        }
    }
    // layer-0 attention norm — the per-layer trailing add+rms is fused and feeds the NEXT layer
    if (g_skip != "ew") {
        unsafe {
            var bw_att0 = upload_region(addr < void ? > (t.fblob[t.rms_att_off]), uint64(dim * 4l))
            enc_rms(enc, r.bx, bw_att0, r.bxb, r.u_dim, r.u_eps)
        }
    }
    for (l in range64(c.n_layers)) {
        unsafe {
            // attention: fused QKV GEMV (+rope+store on the s16 path — one dispatch, one
            // serialization level) -> single-query attention over [0, pos] -> out-proj ->
            // fused residual add + ffn norm
            if (g_skip != "gemv") {
                var bwqkv = upload_region_cat3(
                    addr < void ? > (t.qblob[t.wq_offs[l]]), uint64(qd * dim),
                    addr < void ? > (t.qblob[t.wk_offs[l]]), uint64(kv_dim * dim),
                    addr < void ? > (t.qblob[t.wv_offs[l]]), uint64(kv_dim * dim))
                var bwqkv_s = upload_scales_cat3(t, t.wq_offs[l], qd, t.wk_offs[l], kv_dim, t.wv_offs[l], kv_dim, dim)
                if (r.s16 && scalar_kv) {
                    enc_gemv_qkv_rs(enc, f16v, bwqkv, bwqkv_s, r.bxb, r.bqkv, g_arena, g_arena,
                        r.koff + uint64((l * cap + pos) * rowb), r.voff + uint64((l * cap + pos) * rowb),
                        r.bcos, r.bsin, r.u_dim, r.u_qkvd, r.u_qd, r.u_kvd, r.u_hs, qkvd)
                } else {
                    enc_gemv(enc, r.s16, bwqkv, bwqkv_s, r.bxb, r.bqkv, r.u_dim, r.u_qkvd, qkvd)
                }
            }
            if (g_skip != "ew" && !(r.s16 && scalar_kv)) {
                if (r.kdt == KVDtype.q8_0) {
                    enc_rope_store_q8(enc, r.bqkv, uint64((qd + kv_dim) * 4l), g_arena, g_arena,
                        r.koff + uint64((l * cap + pos) * rowb), r.voff + uint64((l * cap + pos) * rowb),
                        r.bcos, r.bsin, r.u_qd, r.u_hs, r.u_npq, r.u_nblk, qd / 2l + 2l * (kv_dim / 32l))
                } else {
                    enc_rope_store(enc, f16v, r.bqkv, uint64((qd + kv_dim) * 4l), g_arena, g_arena,
                        r.koff + uint64((l * cap + pos) * rowb), r.voff + uint64((l * cap + pos) * rowb),
                        r.bcos, r.bsin, r.u_qd, r.u_hs, r.u_npq, r.u_npall, (qd + kv_dim) / 2l)
                }
            }
            if (g_skip != "attn") {
                if (r.chunked) {
                    if (r.kdt == KVDtype.q8_0) {
                        enc_attn_part_q8(enc, g_arena, g_arena, r.koff + uint64(l * cap * rowb), r.voff + uint64(l * cap * rowb),
                            r.bqkv, r.bpart, r.u_cnt, r.u_nblk, r.u_nh, r.u_kvm, r.u_hs, r.u_scale, r.u_nch, n_heads * nchunks, head_size)
                    } else {
                        enc_attn_part(enc, f16v, g_arena, g_arena, r.koff + uint64(l * cap * rowb), r.voff + uint64(l * cap * rowb),
                            r.bqkv, r.bpart, r.u_cnt, r.u_kvd, r.u_nh, r.u_kvm, r.u_hs, r.u_scale, r.u_nch, n_heads * nchunks, head_size)
                    }
                    enc_attn_comb(enc, r.bpart, r.bxb, r.u_nch, r.u_nh, r.u_hs, n_heads, head_size)
                } else {
                    if (r.kdt == KVDtype.q8_0) {
                        enc_attn_q8(enc, g_arena, g_arena, r.koff + uint64(l * cap * rowb), r.voff + uint64(l * cap * rowb),
                            r.bqkv, r.bxb, r.u_cnt, r.u_nblk, r.u_nh, r.u_kvm, r.u_hs, r.u_scale, n_heads, head_size)
                    } else {
                        enc_attn(enc, f16v, g_arena, g_arena, r.koff + uint64(l * cap * rowb), r.voff + uint64(l * cap * rowb),
                            r.bqkv, r.bxb, r.u_cnt, r.u_kvd, r.u_nh, r.u_kvm, r.u_hs, r.u_scale, n_heads, head_size)
                    }
                }
            }
            if (g_skip != "gemv") {
                var bwo = upload_region(addr < void ? > (t.qblob[t.wo_offs[l]]), uint64(dim * qd))
                var bwo_s = upload_scales(t, t.wo_offs[l], dim, qd)
                enc_gemv(enc, r.s16, bwo, bwo_s, r.bxb, r.bxb2, r.u_qd, r.u_dim, dim)
            }
            // ffn: fused add+norm -> fused W1|W3 GEMV -> swiglu -> W2 GEMV -> fused add + the
            // NEXT layer's attention norm (plain add on the last layer — the final norm follows)
            if (g_skip != "ew") {
                var bw_ffn = upload_region(addr < void ? > (t.fblob[t.rms_ffn_off + l * dim]), uint64(dim * 4l))
                enc_add_rms_site(enc, r.bx, r.bxb2, bw_ffn, r.bxb, r.u_dim, r.u_eps, dim)
            }
            if (g_skip != "gemv") {
                var bw13 = upload_region_cat3(
                    addr < void ? > (t.qblob[t.w1_offs[l]]), uint64(hidden * dim),
                    addr < void ? > (t.qblob[t.w3_offs[l]]), uint64(hidden * dim), null, 0ul)
                var bw13_s = upload_scales_cat3(t, t.w1_offs[l], hidden, t.w3_offs[l], hidden, 0l, 0l, dim)
                if (r.s16) {   // fused gate+up dot with the swiglu epilogue — one dispatch
                    enc_gemv_w13sw(enc, bw13, bw13_s, r.bxb, r.bh12, r.u_dim, r.u_hidden, hidden)
                } else {
                    enc_gemv(enc, false, bw13, bw13_s, r.bxb, r.bh12, r.u_dim, r.u_h2, h2)
                }
            }
            if (g_skip != "ew" && !r.s16) {
                enc_ew2(enc, g_pso_swiglu, r.bh12, 0ul, r.bh12, uint64(hidden * 4l), r.u_hidden, hidden)
            }
            if (g_skip != "gemv") {
                var bw2 = upload_region(addr < void ? > (t.qblob[t.w2_offs[l]]), uint64(dim * hidden))
                var bw2_s = upload_scales(t, t.w2_offs[l], dim, hidden)
                enc_gemv(enc, r.s16, bw2, bw2_s, r.bh12, r.bxb, r.u_hidden, r.u_dim, dim)
            }
            if (g_skip != "ew") {
                if (l + 1l < c.n_layers) {
                    var bw_attn = upload_region(addr < void ? > (t.fblob[t.rms_att_off + (l + 1l) * dim]), uint64(dim * 4l))
                    enc_add_rms_site(enc, r.bx, r.bxb, bw_attn, r.bxb, r.u_dim, r.u_eps, dim)
                } else {
                    enc_ew2(enc, g_pso_add, r.bx, 0ul, r.bxb, 0ul, r.u_dim, dim)
                }
            }
        }
    }
    if (r.gpu_logits) {
        // final rmsnorm (one row) + classifier GEMV — the caller samples straight from s.logits
        unsafe {
            var bwfin = upload_region(addr < void ? > (t.fblob[t.rms_final_off]), uint64(dim * 4l))
            var bwc = upload_region(addr < void ? > (t.qblob[t.wcls_off]), uint64(c.vocab_size * dim))
            var bwc_s = upload_scales(t, t.wcls_off, c.vocab_size, dim)
            enc_rms(enc, r.bx, bwfin, r.bxf, r.u_dim, r.u_eps)
            enc_gemv(enc, r.s16, bwc, bwc_s, r.bxf, r.blog, r.u_dim, r.u_vocab, c.vocab_size)
        }
    }
    metal_end_encoding(enc)
    metal_release(enc)
    if (commit_now) {
        metal_commit(cb)
        r.committed = true
    }
    r.cb = cb
}

// wait for the step's cb, then hand results back: s.x, the mirrored KV row VERBATIM into the CPU
// cache (the CPU never re-encodes), s.logits when the GPU classifier ran, watermark advance
def private finish_step(t : Model; var s : Session; r : StepRes) : bool {
    let ts_wait = ref_time_ticks()
    metal_wait_until_completed(r.cb)
    let err = metal_command_buffer_error(r.cb)
    g_gpu_ms += (metal_command_buffer_gpu_end_time(r.cb) - metal_command_buffer_gpu_start_time(r.cb)) * 1000.0lf
    g_us_wait += int64(get_time_usec(ts_wait))
    let ok = empty(err)
    if (ok) {
        let ts_rb = ref_time_ticks()
        let c = t.config
        let kv_dim = layer_kv_dim(c, 0l)
        let rowb = kv_row_bytes(s.kv_dtype_k, kv_dim)
        let pos = r.pos
        let cap = r.cap
        unsafe {
            memcpy(addr < void ? > (s.x[0]), metal_buffer_contents(r.bx), int(r.bytes_x))
            var kmp = reinterpret<uint8?>(metal_buffer_contents(g_arena)) + r.koff
            var vmp = reinterpret<uint8?>(metal_buffer_contents(g_arena)) + r.voff
            let prow = kv_phys_row(s, pos)
            for (l in range64(c.n_layers)) {
                memcpy(reinterpret<void?>(kv_layer_kbase(s, l, c.seq_len) + prow * rowb),
                    reinterpret<void?>(kmp + (l * cap + pos) * rowb), int(rowb))
                memcpy(reinterpret<void?>(kv_layer_vbase(s, l, c.seq_len) + prow * rowb),
                    reinterpret<void?>(vmp + (l * cap + pos) * rowb), int(rowb))
            }
            if (r.gpu_logits) {
                memcpy(addr < void ? > (s.logits[0]), metal_buffer_contents(r.blog), int(r.bytes_log))
            }
        }
        if (r.gpu_logits) {
            decode_override_logits_done()
        }
        g_mirrors[s.uid].watermark = r.pos + 1l
        g_us_readback += int64(get_time_usec(ts_rb))
        g_decodes++
    } else {
        to_log(LOG_ERROR, "dasLLAMA metal decode: dispatch failed ({err}) — falling back to the CPU layer loop\n")
        decline(MetalDecodeDecline.gpu_error)
    }
    return ok
}

def private release_step(var r : StepRes) {
    if (r.cb != null) {
        metal_release(r.cb)
        r.cb = null
    }
    if (r.spec) {
        pool_release(g_pool, r.btok, 64ul)
    }
    pool_release(g_pool, r.bx, r.bytes_x)
    pool_release(g_pool, r.bxb, r.bytes_q)
    pool_release(g_pool, r.bxb2, r.bytes_x)
    pool_release(g_pool, r.bqkv, r.bytes_qkv)
    pool_release(g_pool, r.bh12, r.bytes_h2)
    pool_release(g_upool, r.bcos, r.bytes_tab)
    pool_release(g_upool, r.bsin, r.bytes_tab)
    if (r.gpu_logits) {
        pool_release(g_pool, r.bxf, r.bytes_x)
        pool_release(g_pool, r.blog, r.bytes_log)
        pool_release(g_upool, r.u_vocab, 4ul)
    }
    if (r.chunked) {
        pool_release(g_pool, r.bpart, r.bytes_part)
        pool_release(g_upool, r.u_nch, 4ul)
    }
    if (r.u_nblk != null) {
        pool_release(g_upool, r.u_nblk, 4ul)
    }
    pool_release(g_upool, r.u_dim, 4ul)
    pool_release(g_upool, r.u_qd, 4ul)
    pool_release(g_upool, r.u_kvd, 4ul)
    pool_release(g_upool, r.u_qkvd, 4ul)
    pool_release(g_upool, r.u_hidden, 4ul)
    pool_release(g_upool, r.u_h2, 4ul)
    pool_release(g_upool, r.u_hs, 4ul)
    pool_release(g_upool, r.u_kvm, 4ul)
    pool_release(g_upool, r.u_nh, 4ul)
    pool_release(g_upool, r.u_cnt, 4ul)
    pool_release(g_upool, r.u_npq, 4ul)
    pool_release(g_upool, r.u_npall, 4ul)
    pool_release(g_upool, r.u_scale, 4ul)
    pool_release(g_upool, r.u_eps, 4ul)
    r.valid = false
}

// Retire the pre-encoded step. A poked-mode pre was never committed — releasing it simply drops
// the encoded work. A spec pre is already running on stale inputs: wait it out (its mirror-row
// writes land at/above the watermark — unmirrored territory by contract — and its outputs go to
// pooled buffers still held here) and SURFACE any execution error: an eaten fault here would
// poison every later submission ("prior/excessive GPU errors") with no trace of the true site.
def private discard_pre {
    if (!g_pre.valid) {
        return
    }
    var r = g_pre
    g_pre.valid = false
    if (r.committed) {
        metal_wait_until_completed(r.cb)
        let err = metal_command_buffer_error(r.cb)
        if (!empty(err)) {
            to_log(LOG_ERROR, "dasLLAMA metal decode: discarded pre-step failed ({err})\n")
        }
    }
    release_step(r)
}

// Pre-encode the PREDICTED next step (same session, pos + 1) while the current step runs on
// the GPU. When the greedy-speculation gates pass, the new cb commits AND chains itself: it
// opens with argmax(previous logits) + the winner's embed gather and needs NO CPU poke at all
// (the rope row is position-only — built here, bit-identical to forward()'s); otherwise the cb
// stays uncommitted and the next call pokes + commits. Declines quietly when the next step's
// gates fail or the mirror would need a rebuild — the slow path serves those.
def private pre_encode_next(t : Model; s : Session; npos : int64; var prev_blog : MetalBuffer?) {
    // quiet non-decline: skipping the pre-encode just means the slow path serves the next step
    if (decode_decline(t, s, npos) != MetalDecodeDecline.none || !key_exists(g_mirrors, s.uid)) {
        return
    }
    let mr = g_mirrors[s.uid]   // cheap copy, capacity check ONLY — row npos-1 is in flight on the GPU
    if (mr.cap_rows < npos + 1l || mr.kdt != s.kv_dtype_k || mr.vdt != s.kv_dtype_v) {
        return
    }
    let ts_enc = ref_time_ticks()
    var r = acquire_step(t, s, npos, mr.koff, mr.voff, mr.cap_rows)
    if (r.gpu_logits && prev_blog != null && t.cls_q8 && t.config.embed_scale == 1.0 &&
            !has_ple(t.config) && spec_armed()) {
        // speculative chain: the cb runs the moment the previous one drains — the CPU tail,
        // sampling, and the next forward() call all overlap GPU execution
        r.spec = true
        r.btok = pool_acquire(g_pool, g_dev, 64ul)
        build_decode_rope_row(t, npos, g_rope_cos_next, g_rope_sin_next)
        unsafe {
            memcpy(metal_buffer_contents(r.bcos), addr < void ? > (g_rope_cos_next[0]), int(r.bytes_tab))
            memcpy(metal_buffer_contents(r.bsin), addr < void ? > (g_rope_sin_next[0]), int(r.bytes_tab))
        }
        encode_step(t, r, true, prev_blog)
    } else {
        encode_step(t, r, false)
    }
    g_pre = r
    g_us_encode += int64(get_time_usec(ts_enc))
}

// the non-pipelined step: mirror catch-up + encode-now + wait; the NEXT step's pre-encode still
// overlaps this step's GPU run. Serves window starts, prediction misses, and speculation reruns.
def private metal_decode_slow(t : Model; var s : Session; pos : int64) : bool {
    let ts_all = ref_time_ticks()
    let mir = mirror_prepare(t, s, pos)
    if (!mir.ok) {
        decline(MetalDecodeDecline.mirror)
        return false
    }
    var r = acquire_step(t, s, pos, mir.koff, mir.voff, mir.cap)
    poke_step(s, r)
    g_us_setup += int64(get_time_usec(ts_all))
    let ts_enc = ref_time_ticks()
    encode_step(t, r, true)
    g_us_encode += int64(get_time_usec(ts_enc))
    pre_encode_next(t, s, pos + 1l, r.blog)
    let ok = finish_step(t, s, r)
    release_step(r)
    return ok
}

//! The DecodeOverrideFn: runs one token step's whole llama layer stack GPU-resident against the
//! session's KV mirror. true = done (KV row stored CPU-side too, s.x = final residual, s.logits
//! when the GPU classifier gates pass); false = declined, the caller runs the CPU layer loop.
def metal_decode_forward(t : Model; var s : Session; token, pos : int64) : bool {
    // mixed batch/single schedules: a pipelined batch step may still own this session's KV row
    // and logits — land it before reading anything session-side (no-op when nothing pends)
    finish_pending_step()
    g_lp_err = false
    var why = decode_decline(t, s, pos)
    if (why == MetalDecodeDecline.none && !metal_decode_init()) {
        why = MetalDecodeDecline.device
    }
    if (why != MetalDecodeDecline.none) {
        discard_pre()   // the sequence diverged from the prediction
        decline(why)
        return false
    }
    if (!(g_pre.valid && g_pre.uid == s.uid && g_pre.pos == pos &&
            g_pre.kdt == s.kv_dtype_k && g_pre.s16 == t.wscale_f16)) {
        discard_pre()
        return metal_decode_slow(t, s, pos)
    }
    // FAST path: this step's cb was encoded during the previous step's GPU run. A spec cb has
    // been running since the previous step drained (GPU-chained argmax + embed); a poked-mode
    // one needs the embed + rope bytes poked, then commits.
    let ts_all = ref_time_ticks()
    var r = g_pre
    g_pre.valid = false
    if (!r.spec) {
        poke_step(s, r)
        metal_commit(r.cb)
        r.committed = true
    }
    g_tick++
    g_mirrors[s.uid].tick = g_tick   // keep the live session hot for the LRU
    g_us_setup += int64(get_time_usec(ts_all))
    pre_encode_next(t, s, pos + 1l, r.blog)
    let ok = finish_step(t, s, r)
    if (ok && r.spec) {
        // verify the GPU's greedy pick against the caller's actual token — a miss means this
        // whole step ran on the wrong embedding: drop the chained pre-step, rerun with the
        // truth (the garbage KV row sits at/above the rerun's rewound watermark, so the
        // mirror_prepare re-upload repairs it from the CPU cache)
        var tokp = unsafe(reinterpret<uint?>(metal_buffer_contents(r.btok)))
        let picked = int64(unsafe(tokp[0]))
        if (picked != token) {
            g_spec_misses++
            g_spec_cooldown = g_spec_backoff
            g_spec_backoff = min(g_spec_backoff * 2, 256)
            discard_pre()
            release_step(r)
            g_decodes--   // finish_step counted the mispredicted step; the rerun recounts
            return metal_decode_slow(t, s, pos)
        }
        g_spec_hits++
        g_spec_backoff = 1
    }
    release_step(r)
    return ok
}

// ===== the batched-decode driver (P4) =====

// GEMV-vs-GEMM site policy: the fixed-B batched GEMV forms win to B = 4, the M-pad-32 GEMM from
// B ~ 5 on every decode site (P1a lab — all sites are big shapes: the QKV concat, wo, w1/w3, w2).
// DASLLAMA_METAL_BATCH_GEMM_MIN overrides for A/B.
var private g_bgemm_min = -1l
// the step splits into DASLLAMA_METAL_BATCH_NCB command buffers committed as encoded (llama.cpp's
// n_cb shape): the driver's ~3.2ms/step scheduling span (kernelStart..End of a 141-dispatch cb,
// measured flat across B — the GPU idles through it when a single cb commits after the previous
// wait) pipelines behind cb k's execution; only cb 1's slice stays exposed. Queue order serializes
// the pieces, and every encode is token-independent (x is buffer CONTENTS), so nothing races.
// 1 = the single-cb rail (A/B; exact pre-split behavior).
var private g_bncb = -1l
// per-site split-K / 64-wide policy (the GEMM lab: at mp = 32 the batch GEMM is OCCUPANCY-bound
// — d/32 threadgroups starve the starved sites, and grid.y k-slices fix exactly that: kv sk8
// +135%, wo sk4 +21%, qkv sk2 +9%, w2 sk8 +33%; w13's 256 tgs are already fat, and the
// classifier's 4008 prefer the 64-wide tile's per-element ISA edge +7%). DASLLAMA_METAL_BATCH_SK=0
// pins every site back to the planar 32-wide v2 (the A/B rail — default ON so the parity arms
// exercise the policy kernels).
var private g_bsk = true
// DASLLAMA_METAL_BATCH_CONCURRENT: encode the step on a CONCURRENT compute encoder and insert a
// memory barrier ONLY at a real buffer hazard (RAW/WAR/WAW; read-after-read runs free) — mirrors
// ggml-metal's range-tracked concurrency. In our FUSED chain the only exploitable overlap is
// W1 || W3 (both read the ffn-norm output, write disjoint buffers); everything else is a serial
// RAW chain and still barriers. Default ON: measured +1-2% at every B (3B d512 same-window A/B,
// parity-clean) — the concurrent encoder pipelines per-dispatch launch overhead across the buffer
// barriers even on the RAW chain. =0 pins the serial encoder (the A/B rail).
var private g_bconcurrent = true
// DASLLAMA_METAL_BATCH_FUSE: fold the w1 GEMV + w3 GEMV + swiglu triple into one fused dispatch
// (MetalGemvW13Sw*) on the fixed-B path (B <= 4) — matches the single-stream w13sw fusion, cutting
// 2 dispatches + their barriers per layer where inter-dispatch bubbles hurt most. Default ON;
// =0 pins the unfused w1/w3/swiglu triple (the A/B rail).
var private g_bfuse = true

// the mul_mv-shape GEMV (MetalQ8MvB2/B4 — the de-verbatim'd Part-3 winner, in-graph == the lcpp
// reference) at every fixed-B site instead of the x-staged forms (which measured -10.5% in-graph:
// the walkthrough lab win was SLC-inflated). DASLLAMA_METAL_BATCH_MV=0 = the x-staged A/B rail.
var private g_bmv = true

// B = 5..8 rides two 4-column mv tiles (weights stream twice) instead of the GEMM + split-K
// policy — the reference's own shape below B = 9. Measured +12.7% at B=8 (239.4 vs 212.5 t/s).
// DASLLAMA_METAL_BATCH_MV8=0 = the GEMM-policy A/B rail.
var private g_bmv8 = true

// one-deep encode-ahead for the batch driver (the copy-validated pipe): a step commits BEFORE the
// previous one is waited on; sessions then hold the PREVIOUS step's logits until the next call or
// metal_decode_flush. That is exact for fixed-token throughput protocols and WRONG for
// logit-gated sampling (eval_batch's contract) — so this is OPT-IN, for benches only:
// DASLLAMA_METAL_BATCH_PIPE=1. Engages only on the gpu-logits path.
var private g_b_pipe = false
// per-step scratch reused across calls (row table image + per-row mirror slices)
var private g_b_rte : array<uint>
var private g_b_koffs : array<uint64>
var private g_b_voffs : array<uint64>
var private g_b_caps : array<int64>
var private g_b_ulays : array<MetalBuffer?>   // per-layer uniforms, pooled per step

// the model half is decode_shape_decline; per batch: uniform f16/f32 KV (eval_batch snapshots
// the dtypes onto ws.scr), every row within the attention depth cap with a real uid, per-row
// rope rows built, and dim within the add+rms row slabs (the GEMM path caps at 3072, the
// fixed-B GEMV path at ADD_RMS_MAX_DIM — B >= 5 on a wider model declines)
def private batch_decode_decline(t : Model; ws : BatchWorkspace; sessions : array<Session?>; nrows : int64) : MetalDecodeDecline {
    let shape = decode_shape_decline(t)
    if (shape != MetalDecodeDecline.none) {
        return shape
    }
    let c = t.config
    let kdt = ws.scr.kv_dtype_k
    if (ws.scr.kv_dtype_v != kdt ||
            (kdt != KVDtype.f16 && kdt != KVDtype.f32 && kdt != KVDtype.q8_0)) {
        return MetalDecodeDecline.kv_dtype
    }
    if (c.dim > (nrows <= 4l ? ADD_RMS_MAX_DIM : 3072l)) {
        return MetalDecodeDecline.batch_dim_cap
    }
    for (i in range64(nrows)) {
        if (sessions[i].uid == 0ul) {
            return MetalDecodeDecline.no_uid
        }
        if (ws.positions[i] + 1l > MAX_ATTN_ROWS) {
            return MetalDecodeDecline.depth
        }
    }
    if (int64(length(ws.scr.rope_cos)) < nrows * (c.head_size / 2l)) {
        return MetalDecodeDecline.rope_rows
    }
    return MetalDecodeDecline.none
}

//! The BatchDecodeOverrideFn (P4): one eval_batch step's whole layer stack GPU-resident — B rows
//! through ONE weight pass (fixed-B GEMVs to B = 4, the M-pad-32 GEMM above; per-row attention +
//! rope+store reach each row's KV through the mirror-arena row table). STACK-ONLY contract: on
//! true every row's KV row is stored (GPU mirror and CPU cache) and ws.scr.x_b holds the final
//! residuals; the CPU tail (final rms, classifier, scatter, n_past++) stays with the caller.
//! false = declined — the CPU layer loop recomputes from clean state (x_b written only on ok).
def metal_batch_decode_forward(t : Model; var ws : BatchWorkspace; var sessions : array<Session?>; nrows : int64) : bool {
    var why = batch_decode_decline(t, ws, sessions, nrows)
    if (why == MetalDecodeDecline.none && !metal_decode_init()) {
        why = MetalDecodeDecline.device
    }
    if (why != MetalDecodeDecline.none) {
        finish_pending_step()   // the CPU fallback needs the in-flight step's KV/logits landed first
        g_lp_err = false
        batch_decline(why)
        return false
    }
    // a parked single-stream pre-step may reference a batch row's mirror slice — retire it first
    discard_pre()
    // encode-ahead readiness: with a step in flight, mirror_prepare must stay on its fast path
    // (no upload, no eviction, no rebuild — those touch arena slices the GPU is still writing).
    // Any miss → complete the pending step first and take this one synchronously.
    if (g_lp_active) {
        var ready = true
        for (i in range64(nrows)) {
            var sp = sessions[i]
            if (!key_exists(g_mirrors, sp.uid)) {
                ready = false
                break
            }
            let mr = g_mirrors[sp.uid]   // cheap copy — fields tested off it
            if (mr.cap_rows < ws.positions[i] + 1l || mr.kdt != ws.scr.kv_dtype_k ||
                    mr.vdt != ws.scr.kv_dtype_v || mr.watermark < ws.positions[i]) {
                ready = false
                break
            }
        }
        if (!ready) {
            finish_pending_step()
        }
    }
    let ts_all = ref_time_ticks()
    let c = t.config
    let dim = c.dim
    let head_size = c.head_size
    let n_heads = c.n_heads
    let qd = n_heads * head_size
    let kv_dim = layer_kv_dim(c, 0l)
    let hidden = layer_hidden(t, 0l)
    let qkvd = qd + 2l * kv_dim
    let half = head_size / 2l
    let s16 = t.wscale_f16
    let kdt = ws.scr.kv_dtype_k
    let f16 = kdt == KVDtype.f16
    let blockc = kdt == KVDtype.q8_0        // block codec: byte-offset row table + q8 kernel twins
    let rowb = kv_row_bytes(kdt, kv_dim)
    let esz = f16 ? 2l : 4l
    // pin the batch's uids: a later row's mirror alloc must never evict an earlier row's slice
    for (i in range64(nrows)) {
        g_pin |> insert(sessions[i].uid)
    }
    g_b_rte |> resize(int(nrows * 4l))
    g_b_koffs |> resize(int(nrows))
    g_b_voffs |> resize(int(nrows))
    g_b_caps |> resize(int(nrows))
    var maxcnt = 0l
    var rows_ok = true
    for (i in range64(nrows)) {
        var sp = sessions[i]
        let mir = mirror_prepare(t, *sp, ws.positions[i])
        if (!mir.ok) {
            rows_ok = false
            break
        }
        g_b_koffs[i] = mir.koff
        g_b_voffs[i] = mir.voff
        g_b_caps[i] = mir.cap
        let cnt = ws.positions[i] + 1l
        maxcnt = max(maxcnt, cnt)
        // block codecs address in BYTES (there is no scalar element size); scalar mirrors keep
        // their element-offset entries
        g_b_rte[i * 4l] = uint(blockc ? mir.koff : mir.koff / uint64(esz))
        g_b_rte[i * 4l + 1l] = uint(blockc ? mir.voff : mir.voff / uint64(esz))
        g_b_rte[i * 4l + 2l] = uint(mir.cap)
        g_b_rte[i * 4l + 3l] = uint(cnt)
    }
    g_pin |> clear()
    if (!rows_ok) {
        finish_pending_step()   // fall back synchronously — the CPU loop needs completed KV state
        g_lp_err = false
        batch_decline(MetalDecodeDecline.mirror)
        return false
    }
    if (g_bgemm_min < 0l) {
        g_bgemm_min = int64(max(env_int("DASLLAMA_METAL_BATCH_GEMM_MIN", 5), 2))
        g_bsk = env_int("DASLLAMA_METAL_BATCH_SK", 1) != 0
        g_bncb = int64(clamp(env_int("DASLLAMA_METAL_BATCH_NCB", 4), 1, 16))
        g_bconcurrent = env_int("DASLLAMA_METAL_BATCH_CONCURRENT", 1) != 0
        g_bfuse = env_int("DASLLAMA_METAL_BATCH_FUSE", 1) != 0
        g_bmv = env_int("DASLLAMA_METAL_BATCH_MV", 1) != 0
        g_bmv8 = env_int("DASLLAMA_METAL_BATCH_MV8", 1) != 0
        g_b_pipe = env_int("DASLLAMA_METAL_BATCH_PIPE", 0) != 0
    }
    // the fixed-B GEMV forms cover 4 rows — past that the GEMM is mandatory (the knob can only
    // pull the crossover DOWN) unless the MV8 probe column-tiles; nrows > 4 with dim > 3072
    // already declined above
    let mv8 = g_bmv8 && g_bmv && nrows > 4l && nrows <= 8l && dim <= 3072l
    let use_gemm = (nrows >= g_bgemm_min || nrows > 4l) && dim <= 3072l && !mv8
    // the mul_mm rail (opt-in, DASLLAMA_METAL_BATCH_MM=1): the prefill kernel on shared 34B
    // blobs + raw f32 X (quant tails disappear; every GEMM'd width % 64). MEASURED NEGATIVE at
    // M-pad-32 (3B d512 -n 32: B=8 145.2 vs GemmB 153.3, B=16 275.7 vs 285.0): mm's 32x64 tile
    // grids only d/64 threadgroups (48 at d=3072, 16 at kv_dim=1024) — prefill shapes scale the
    // grid on tokens, batch decode can't; GemmB's 32x32 tile = 2x the occupancy and wins even
    // with worse per-element ISA. Kept as the A/B rail for bigger-M futures (B > 16 pads to 64).
    let use_mm = (use_gemm && get_env_variable("DASLLAMA_METAL_BATCH_MM") == "1" &&
        (dim % 64l) == 0l && (qd % 64l) == 0l && (kv_dim % 64l) == 0l && (hidden % 64l) == 0l)
    let mp = use_gemm ? ((nrows + 31l) / 32l) * 32l : (nrows <= 2l ? 2l : nrows <= 4l ? 4l : 8l)
    // the per-site GEMM policy (see g_bsk): each split-K site gates on its OWN K's nkb % S == 0
    // (and the partial planes assume mp == 32); an ineligible site stays on the planar v2
    let sk_ok = use_gemm && !use_mm && g_bsk && mp == 32l
    let qkv_sk = (sk_ok && (dim / 32l) % 2l == 0l) ? 2l : 0l
    let wo_sk = (sk_ok && (qd / 32l) % 4l == 0l) ? 4l : 0l
    let w2_sk = (sk_ok && (hidden / 32l) % 8l == 0l) ? 8l : 0l
    let cls64 = use_gemm && !use_mm && g_bsk && (c.vocab_size % 64l) == 0l
    // the mul_mv rail owns the whole fixed-B path (w13 = two mv dispatches + the ew2 swiglu, so
    // the x-staged w13sw fusion stands down with it)
    let mv = g_bmv && !use_gemm && !use_mm
    // fold w1+w3+swiglu into one fused GEMV on the fixed-B path; a knockout run (g_skip) keeps the
    // stages separable so its per-stage attribution stays valid
    let fuse13 = g_bfuse && !mv && !use_gemm && !use_mm && empty(g_skip)
    let chunked = maxcnt > g_attn_single_max
    let nchmax = (maxcnt + ATTN_CHUNK_ROWS - 1l) / ATTN_CHUNK_ROWS
    // partD (walkthrough kernel-2 winner): fused single-pass slice kernel + parallel comb —
    // lane ownership assumes hs == 128 (32 lanes x half4 = one head); other head sizes keep
    // the proven single/chunked pair. DASLLAMA_METAL_ATTN_D=0 = the A/B rail to the old path.
    let use_attn_d = head_size == 128l && get_env_variable("DASLLAMA_METAL_ATTN_D") != "0"
    let d_nsgs = min((maxcnt + 31l) / 32l, 16l)
    // logits-on-GPU (default ON): the batched final rms + classifier ride the same cb and the
    // CPU tail's classifier (one full weight-plane stream per step — ~6ms of the small-B wall)
    // is skipped via batch_decode_override_logits_done. Gates mirror the single-stream driver's.
    let cls_fmt = c.shared_weights ? t.emb_fmt : t.wcls_fmt
    let tied_f32 = c.shared_weights && !t.cls_q8 && cls_fmt == KqFmt.q8
    let gpu_logits = (!g_logits_cpu && get_env_variable("DASLLAMA_METAL_LOGITS") != "0" && empty(g_skip) &&
        cls_fmt == KqFmt.q8 && !tied_f32 && c.final_logit_softcap <= 0.0 &&
        empty(t.suppress) && t.wcls_off >= 0l && (!use_gemm || (c.vocab_size % (use_mm ? 64l : 32l)) == 0l))
    // pooled per-step buffers (mp rows: the fixed-B forms read their full B, the GEMM its M pad —
    // stale pad rows only ever land in pad outputs, which nothing reads back)
    let bytes_x = uint64(mp * dim * 4l)
    let bytes_xb = uint64(mp * qd * 4l)
    let bytes_qkv = uint64(mp * qkvd * 4l)
    let bytes_h = uint64(mp * hidden * 4l)
    let bytes_tab = uint64(nrows * half * 4l)
    let bytes_rt = uint64(nrows * 16l)
    let bytes_part = uint64(use_attn_d ? nrows * n_heads * d_nsgs * (head_size + 4l) * 4l :
        nrows * n_heads * nchmax * (head_size + 2l) * 4l)
    var bx = pool_acquire(g_pool, g_dev, bytes_x)
    var bxb = pool_acquire(g_pool, g_dev, bytes_xb)
    var bxb2 = pool_acquire(g_pool, g_dev, bytes_x)
    var bqkv = pool_acquire(g_pool, g_dev, bytes_qkv)
    var bhb = pool_acquire(g_pool, g_dev, bytes_h)
    var bhb2 = pool_acquire(g_pool, g_dev, bytes_h)
    var bpart = (use_attn_d || chunked) ? pool_acquire(g_pool, g_dev, bytes_part) : null
    // one pooled partial-plane buffer serves every split-K site (sized to the largest enabled —
    // w2's S8 x 32 x dim is 3MB at 3B); sites run serially, so reuse is free
    var bytes_sk = 0ul
    if (qkv_sk > 0l) {
        bytes_sk = uint64(qkv_sk * 32l * qkvd * 4l)
    }
    if (wo_sk > 0l) {
        bytes_sk = max(bytes_sk, uint64(wo_sk * 32l * dim * 4l))
    }
    if (w2_sk > 0l) {
        bytes_sk = max(bytes_sk, uint64(w2_sk * 32l * dim * 4l))
    }
    var bpsk = bytes_sk > 0ul ? pool_acquire(g_pool, g_dev, bytes_sk) : null
    let bytes_log = uint64(mp * c.vocab_size * 4l)
    var bxf = gpu_logits ? pool_acquire(g_pool, g_dev, bytes_x) : null
    var blog = gpu_logits ? pool_acquire(g_pool, g_dev, bytes_log) : null
    var bcos = pool_acquire_untracked(g_upool, g_dev, bytes_tab)
    var bsin = pool_acquire_untracked(g_upool, g_dev, bytes_tab)
    var brt = pool_acquire_untracked(g_upool, g_dev, bytes_rt)
    var u_dim = uniform_u32(uint(dim))
    var u_qd = uniform_u32(uint(qd))
    var u_kvd = uniform_u32(uint(kv_dim))
    var u_qkvd = uniform_u32(uint(qkvd))
    var u_hidden = uniform_u32(uint(hidden))
    var u_hs = uniform_u32(uint(head_size))
    var u_kvm = uniform_u32(uint(n_heads / layer_n_kv_heads(c, 0l)))
    var u_nh = uniform_u32(uint(n_heads))
    var u_scale = uniform_f32(1.0 / sqrt(float(head_size)))
    var u_eps = uniform_f32(c.norm_eps)
    var u_npq = uniform_u32(uint(qd / 2l))
    var u_npall = uniform_u32(uint((qd + kv_dim) / 2l))
    var u_s16 = uniform_u32(s16 ? 1u : 0u)
    var u_total_dim = uniform_u32(uint(nrows * dim))
    var u_total_h = uniform_u32(uint(nrows * hidden))
    var u_xs4_dim = mv ? uniform_u32(uint(dim / 4l)) : null
    var u_xs4_qd = mv ? uniform_u32(uint(qd / 4l)) : null
    var u_xs4_hid = mv ? uniform_u32(uint(hidden / 4l)) : null
    var u_nrows_mv = (mv && nrows > 2l) ? uniform_u32(uint(min(nrows, 4l))) : null
    var u_nrows_mv2 = (mv && nrows > 4l) ? uniform_u32(uint(nrows - 4l)) : null
    var u_nch = (!use_attn_d && chunked) ? uniform_u32(uint(nchmax)) : null
    var u_nsgs = use_attn_d ? uniform_u32(uint(d_nsgs)) : null
    var u_nblk = blockc ? uniform_u32(uint(kv_dim / 32l)) : null
    var u_vocab = gpu_logits ? uniform_u32(uint(c.vocab_size)) : null
    var u_sk_qkv = qkv_sk > 0l ? uniform_u32(uint(qkv_sk)) : null
    var u_sk_wo = wo_sk > 0l ? uniform_u32(uint(wo_sk)) : null
    var u_sk_w2 = w2_sk > 0l ? uniform_u32(uint(w2_sk)) : null
    var u_skt_qkv = qkv_sk > 0l ? uniform_u32(uint(32l * qkvd)) : null
    var u_skt_dim = (wo_sk > 0l || w2_sk > 0l) ? uniform_u32(uint(32l * dim)) : null
    g_b_ulays |> resize(int(c.n_layers))
    for (l in range64(c.n_layers)) {
        g_b_ulays[l] = uniform_u32(uint(l))
    }
    unsafe {
        memcpy(metal_buffer_contents(bx), addr < void ? > (ws.scr.x_b[0]), int(nrows * dim * 4l))
        memcpy(metal_buffer_contents(bcos), addr < void ? > (ws.scr.rope_cos[0]), int(bytes_tab))
        memcpy(metal_buffer_contents(bsin), addr < void ? > (ws.scr.rope_sin[0]), int(bytes_tab))
        memcpy(metal_buffer_contents(brt), addr < void ? > (g_b_rte[0]), int(bytes_rt))
    }
    g_us_bsetup += int64(get_time_usec(ts_all))
    let ts_enc = ref_time_ticks()
    let ncb = min(g_bncb, c.n_layers)
    let lay_per_cb = (c.n_layers + ncb - 1l) / ncb
    var cbs : array<MetalCommandBuffer?>
    cbs |> reserve(int(ncb))
    // concurrent-encode rail: a concurrent encoder + explicit barriers ONLY at real buffer hazards
    // (the serial encoder implicit-barriers every dispatch). Gated to a full pipeline — a knockout
    // (g_skip) run stays on the serial encoder so a skipped stage can't leave a dependent unbarriered.
    let conc = g_bconcurrent && empty(g_skip)
    var haz = HazTrack(on = conc, w = 0u, r = 0u)
    var cb = metal_new_command_buffer_unretained(g_queue)
    var enc = conc ? metal_new_compute_encoder_concurrent(cb) : metal_new_compute_encoder(cb)
    // layer-0 attention norm (the per-layer trailing add+rms feeds the NEXT layer's)
    if (g_skip != "ew") {
        unsafe {
            var bw_att0 = upload_region(addr < void ? > (t.fblob[t.rms_att_off]), uint64(dim * 4l))
            bar_dep(enc, haz, HZ_BX, HZ_BXB)
            enc_rms_b(enc, bx, bw_att0, bxb, u_dim, u_eps, nrows)
        }
    }
    for (l in range64(c.n_layers)) {
        unsafe {
            var ulay = g_b_ulays[l]
            // attention: QKV (one concat stream) -> per-row rope + KV store -> per-row attention
            // over [0, cnt_b) -> out-proj -> fused residual add + ffn norm
            if (g_skip != "gemv") {
                if (use_mm) {
                    // three per-tensor blob dispatches into the ONE fused QKV buffer — y binds at
                    // the tensor's column byte offset, the qkvd row stride rides the ndim uniform;
                    // disjoint writes, so one barrier (bxb RAW) covers all three — they overlap
                    bar_dep(enc, haz, HZ_BXB, HZ_BQKV)
                    enc_gemm_mm_b(enc, blob_of(t, t.wq_offs[l], qd, dim), bxb, bqkv, 0ul, u_dim, u_qkvd, mp, qd)
                    enc_gemm_mm_b(enc, blob_of(t, t.wk_offs[l], kv_dim, dim), bxb, bqkv, uint64(qd * 4l), u_dim, u_qkvd, mp, kv_dim)
                    enc_gemm_mm_b(enc, blob_of(t, t.wv_offs[l], kv_dim, dim), bxb, bqkv, uint64((qd + kv_dim) * 4l), u_dim, u_qkvd, mp, kv_dim)
                } elif (mv) {
                    bar_dep(enc, haz, HZ_BXB, HZ_BQKV)
                    enc_mv_site(enc, blob_cat3(t, t.wq_offs[l], qd, t.wk_offs[l], kv_dim,
                        t.wv_offs[l], kv_dim, dim), bxb, bqkv, u_dim, u_qkvd, u_xs4_dim, qkvd,
                        nrows, uint64(dim * 4l), u_nrows_mv, u_nrows_mv2)
                } else {
                    var bwqkv = upload_region_cat3(
                        addr < void ? > (t.qblob[t.wq_offs[l]]), uint64(qd * dim),
                        addr < void ? > (t.qblob[t.wk_offs[l]]), uint64(kv_dim * dim),
                        addr < void ? > (t.qblob[t.wv_offs[l]]), uint64(kv_dim * dim))
                    var bwqkv_s = upload_scales_cat3(t, t.wq_offs[l], qd, t.wk_offs[l], kv_dim, t.wv_offs[l], kv_dim, dim)
                    if (qkv_sk > 0l) {
                        bar_dep(enc, haz, HZ_BXB, HZ_BPSK)
                        enc_gemm_sk_b(enc, bwqkv, 0ul, bwqkv_s, 0ul, bxb, bpsk, u_dim, u_qkvd, u_s16, u_sk_qkv, qkvd, qkv_sk)
                        bar_dep(enc, haz, HZ_BPSK, HZ_BQKV)
                        enc_sk_reduce_b(enc, bpsk, bqkv, u_skt_qkv, u_sk_qkv, 32l * qkvd)
                    } elif (use_gemm) {
                        bar_dep(enc, haz, HZ_BXB, HZ_BQKV)
                        enc_gemm_b(enc, bwqkv, 0ul, bwqkv_s, 0ul, bxb, bqkv, u_dim, u_qkvd, u_s16, mp, qkvd)
                    } else {
                        bar_dep(enc, haz, HZ_BXB, HZ_BQKV)
                        enc_gemv_b(enc, nrows > 2l, bwqkv, 0ul, bwqkv_s, 0ul, bxb, bqkv, u_dim, u_qkvd, u_s16, qkvd)
                    }
                }
            }
            if (g_skip != "ew") {
                // rope reads+writes bqkv in place and writes the current KV row into the arena
                bar_dep(enc, haz, HZ_BQKV, HZ_BQKV | HZ_ARENA)
                if (blockc) {
                    enc_rope_store_b_q8(enc, bqkv, bcos, bsin, brt, ulay, u_nblk, u_qd, u_hs, u_npq, u_qkvd,
                        qd / 2l + 2l * (kv_dim / 32l), nrows)
                } else {
                    enc_rope_store_b(enc, f16, bqkv, bcos, bsin, brt, ulay, u_kvd, u_qd, u_hs, u_npq, u_npall, u_qkvd,
                        (qd + kv_dim) / 2l, nrows)
                }
            }
            if (g_skip != "attn") {
                if (use_attn_d) {
                    bar_dep(enc, haz, HZ_BQKV | HZ_ARENA, HZ_BPART)
                    if (blockc) {
                        enc_attn_d_q8(enc, bqkv, bpart, brt, ulay, u_nblk, u_nh, u_kvm, u_hs, u_scale, u_qkvd, u_nsgs,
                            d_nsgs, nrows, n_heads)
                    } else {
                        enc_attn_d(enc, f16, bqkv, bpart, brt, ulay, u_kvd, u_nh, u_kvm, u_hs, u_scale, u_qkvd, u_nsgs,
                            d_nsgs, nrows, n_heads)
                    }
                    bar_dep(enc, haz, HZ_BPART, HZ_BXB)
                    enc_attn_comb_d(enc, bpart, bxb, brt, u_nsgs, u_nh, u_hs, nrows, n_heads, head_size)
                } elif (chunked) {
                    bar_dep(enc, haz, HZ_BQKV | HZ_ARENA, HZ_BPART)
                    if (blockc) {
                        enc_attn_part_b_q8(enc, bqkv, bpart, brt, ulay, u_nblk, u_nh, u_kvm, u_hs, u_scale, u_qkvd, u_nch,
                            nrows, nchmax, n_heads, head_size)
                    } else {
                        enc_attn_part_b(enc, f16, bqkv, bpart, brt, ulay, u_kvd, u_nh, u_kvm, u_hs, u_scale, u_qkvd, u_nch,
                            nrows, nchmax, n_heads, head_size)
                    }
                    bar_dep(enc, haz, HZ_BPART, HZ_BXB)
                    enc_attn_comb_b(enc, bpart, bxb, brt, u_nch, u_nh, u_hs, nrows, n_heads, head_size)
                } else {
                    bar_dep(enc, haz, HZ_BQKV | HZ_ARENA, HZ_BXB)
                    if (blockc) {
                        enc_attn_b_q8(enc, bqkv, bxb, brt, ulay, u_nblk, u_nh, u_kvm, u_hs, u_scale, u_qkvd,
                            nrows, n_heads, head_size)
                    } else {
                        enc_attn_b(enc, f16, bqkv, bxb, brt, ulay, u_kvd, u_nh, u_kvm, u_hs, u_scale, u_qkvd,
                            nrows, n_heads, head_size)
                    }
                }
            }
            if (g_skip != "gemv") {
                if (use_mm) {
                    bar_dep(enc, haz, HZ_BXB, HZ_BXB2)
                    enc_gemm_mm_b(enc, blob_of(t, t.wo_offs[l], dim, qd), bxb, bxb2, 0ul, u_qd, u_dim, mp, dim)
                } elif (mv) {
                    bar_dep(enc, haz, HZ_BXB, HZ_BXB2)
                    enc_mv_site(enc, blob_of(t, t.wo_offs[l], dim, qd), bxb, bxb2,
                        u_qd, u_dim, u_xs4_qd, dim, nrows, uint64(qd * 4l), u_nrows_mv, u_nrows_mv2)
                } else {
                    var bwo = upload_region(addr < void ? > (t.qblob[t.wo_offs[l]]), uint64(dim * qd))
                    var bwo_s = upload_scales(t, t.wo_offs[l], dim, qd)
                    if (wo_sk > 0l) {
                        bar_dep(enc, haz, HZ_BXB, HZ_BPSK)
                        enc_gemm_sk_b(enc, bwo, 0ul, bwo_s, 0ul, bxb, bpsk, u_qd, u_dim, u_s16, u_sk_wo, dim, wo_sk)
                        bar_dep(enc, haz, HZ_BPSK, HZ_BXB2)
                        enc_sk_reduce_b(enc, bpsk, bxb2, u_skt_dim, u_sk_wo, 32l * dim)
                    } elif (use_gemm) {
                        bar_dep(enc, haz, HZ_BXB, HZ_BXB2)
                        enc_gemm_b(enc, bwo, 0ul, bwo_s, 0ul, bxb, bxb2, u_qd, u_dim, u_s16, mp, dim)
                    } else {
                        bar_dep(enc, haz, HZ_BXB, HZ_BXB2)
                        enc_gemv_b(enc, nrows > 2l, bwo, 0ul, bwo_s, 0ul, bxb, bxb2, u_qd, u_dim, u_s16, dim)
                    }
                }
            }
            // ffn: fused add+norm -> W1 / W3 (the cat region at byte offsets) -> swiglu
            // -> W2 -> fused add + the NEXT layer's attention norm (plain add on the last)
            if (g_skip != "ew") {
                var bw_ffn = upload_region(addr < void ? > (t.fblob[t.rms_ffn_off + l * dim]), uint64(dim * 4l))
                // add_rms updates the residual bx in place AND writes the norm-out bxb
                bar_dep(enc, haz, HZ_BX | HZ_BXB2, HZ_BX | HZ_BXB)
                enc_add_rms_b(enc, bx, bxb2, bw_ffn, bxb, u_dim, u_eps, nrows)
            }
            if (g_skip != "gemv") {
                if (use_mm) {
                    bar_dep(enc, haz, HZ_BXB, HZ_BHB | HZ_BHB2)
                    enc_gemm_mm_b(enc, blob_of(t, t.w1_offs[l], hidden, dim), bxb, bhb, 0ul, u_dim, u_hidden, mp, hidden)
                    enc_gemm_mm_b(enc, blob_of(t, t.w3_offs[l], hidden, dim), bxb, bhb2, 0ul, u_dim, u_hidden, mp, hidden)
                } elif (mv) {
                    // two mv dispatches read the same norm and write DISJOINT gate/up — one
                    // barrier covers both; the ew2 swiglu below fuses them (fuse13 stands down)
                    bar_dep(enc, haz, HZ_BXB, HZ_BHB | HZ_BHB2)
                    enc_mv_site(enc, blob_of(t, t.w1_offs[l], hidden, dim), bxb, bhb,
                        u_dim, u_hidden, u_xs4_dim, hidden, nrows, uint64(dim * 4l), u_nrows_mv, u_nrows_mv2)
                    enc_mv_site(enc, blob_of(t, t.w3_offs[l], hidden, dim), bxb, bhb2,
                        u_dim, u_hidden, u_xs4_dim, hidden, nrows, uint64(dim * 4l), u_nrows_mv, u_nrows_mv2)
                } else {
                    var bw13 = upload_region_cat3(
                        addr < void ? > (t.qblob[t.w1_offs[l]]), uint64(hidden * dim),
                        addr < void ? > (t.qblob[t.w3_offs[l]]), uint64(hidden * dim), null, 0ul)
                    var bw13_s = upload_scales_cat3(t, t.w1_offs[l], hidden, t.w3_offs[l], hidden, 0l, 0l, dim)
                    if (fuse13) {
                        // one fused dispatch: gate+up dot + swiglu straight into bhb (w2's input)
                        bar_dep(enc, haz, HZ_BXB, HZ_BHB)
                        enc_gemv_w13sw_b(enc, nrows > 2l, bw13, bw13_s, bxb, bhb, u_dim, u_hidden, u_s16, hidden)
                    } else {
                        // W1 and W3 both read the ffn-norm output and write DISJOINT buffers (bhb / bhb2)
                        // — one barrier (bxb RAW) covers both; they then run concurrently
                        bar_dep(enc, haz, HZ_BXB, HZ_BHB | HZ_BHB2)
                        let w3qoff = uint64(hidden * dim)
                        let w3soff = uint64(hidden * (dim / 32l) * (s16 ? 2l : 4l))
                        if (use_gemm) {
                            enc_gemm_b(enc, bw13, 0ul, bw13_s, 0ul, bxb, bhb, u_dim, u_hidden, u_s16, mp, hidden)
                            enc_gemm_b(enc, bw13, w3qoff, bw13_s, w3soff, bxb, bhb2, u_dim, u_hidden, u_s16, mp, hidden)
                        } else {
                            enc_gemv_b(enc, nrows > 2l, bw13, 0ul, bw13_s, 0ul, bxb, bhb, u_dim, u_hidden, u_s16, hidden)
                            enc_gemv_b(enc, nrows > 2l, bw13, w3qoff, bw13_s, w3soff, bxb, bhb2, u_dim, u_hidden, u_s16, hidden)
                        }
                    }
                }
            }
            if (g_skip != "ew" && !fuse13) {
                // swiglu reads gate (bhb) + up (bhb2), writes silu(gate)*up back into bhb
                bar_dep(enc, haz, HZ_BHB | HZ_BHB2, HZ_BHB)
                enc_ew2(enc, g_pso_swiglu, bhb, 0ul, bhb2, 0ul, u_total_h, nrows * hidden)
            }
            if (g_skip != "gemv") {
                if (use_mm) {
                    bar_dep(enc, haz, HZ_BHB, HZ_BXB)
                    enc_gemm_mm_b(enc, blob_of(t, t.w2_offs[l], dim, hidden), bhb, bxb, 0ul, u_hidden, u_dim, mp, dim)
                } elif (mv) {
                    bar_dep(enc, haz, HZ_BHB, HZ_BXB)
                    enc_mv_site(enc, blob_of(t, t.w2_offs[l], dim, hidden), bhb, bxb,
                        u_hidden, u_dim, u_xs4_hid, dim, nrows, uint64(hidden * 4l), u_nrows_mv, u_nrows_mv2)
                } else {
                    var bw2 = upload_region(addr < void ? > (t.qblob[t.w2_offs[l]]), uint64(dim * hidden))
                    var bw2_s = upload_scales(t, t.w2_offs[l], dim, hidden)
                    if (w2_sk > 0l) {
                        bar_dep(enc, haz, HZ_BHB, HZ_BPSK)
                        enc_gemm_sk_b(enc, bw2, 0ul, bw2_s, 0ul, bhb, bpsk, u_hidden, u_dim, u_s16, u_sk_w2, dim, w2_sk)
                        bar_dep(enc, haz, HZ_BPSK, HZ_BXB)
                        enc_sk_reduce_b(enc, bpsk, bxb, u_skt_dim, u_sk_w2, 32l * dim)
                    } elif (use_gemm) {
                        bar_dep(enc, haz, HZ_BHB, HZ_BXB)
                        enc_gemm_b(enc, bw2, 0ul, bw2_s, 0ul, bhb, bxb, u_hidden, u_dim, u_s16, mp, dim)
                    } else {
                        bar_dep(enc, haz, HZ_BHB, HZ_BXB)
                        enc_gemv_b(enc, nrows > 2l, bw2, 0ul, bw2_s, 0ul, bhb, bxb, u_hidden, u_dim, u_s16, dim)
                    }
                }
            }
            if (g_skip != "ew") {
                if (l + 1l < c.n_layers) {
                    var bw_attn = upload_region(addr < void ? > (t.fblob[t.rms_att_off + (l + 1l) * dim]), uint64(dim * 4l))
                    bar_dep(enc, haz, HZ_BX | HZ_BXB, HZ_BX | HZ_BXB)
                    enc_add_rms_b(enc, bx, bxb, bw_attn, bxb, u_dim, u_eps, nrows)
                } else {
                    // last layer: plain residual add into bx (the final rms reads bx)
                    bar_dep(enc, haz, HZ_BX | HZ_BXB, HZ_BX)
                    enc_ew2(enc, g_pso_add, bx, 0ul, bxb, 0ul, u_total_dim, nrows * dim)
                }
            }
        }
        // group boundary: commit this slice so the driver schedules it under the earlier
        // slices' GPU execution; queue order serializes the pieces
        if ((l + 1l) % lay_per_cb == 0l && l + 1l < c.n_layers) {
            metal_end_encoding(enc)
            metal_release(enc)
            metal_commit(cb)
            cbs |> push(cb)
            cb = metal_new_command_buffer_unretained(g_queue)
            enc = conc ? metal_new_compute_encoder_concurrent(cb) : metal_new_compute_encoder(cb)
            // fresh encoder + in-order queue serialize cross-cb hazards — start the tracker clean
            haz.w = 0u
            haz.r = 0u
        }
    }
    if (gpu_logits) {
        // batched final rmsnorm + classifier — the caller's CPU classifier step is skipped
        unsafe {
            var bwfin = upload_region(addr < void ? > (t.fblob[t.rms_final_off]), uint64(dim * 4l))
            bar_dep(enc, haz, HZ_BX, HZ_BXF)
            enc_rms_b(enc, bx, bwfin, bxf, u_dim, u_eps, nrows)
            bar_dep(enc, haz, HZ_BXF, HZ_BLOG)
            if (use_mm) {
                enc_gemm_mm_b(enc, blob_of(t, t.wcls_off, c.vocab_size, dim), bxf, blog, 0ul, u_dim, u_vocab, mp, c.vocab_size)
            } elif (mv) {
                enc_mv_site(enc, blob_of(t, t.wcls_off, c.vocab_size, dim), bxf, blog,
                    u_dim, u_vocab, u_xs4_dim, c.vocab_size, nrows, uint64(dim * 4l), u_nrows_mv, u_nrows_mv2)
            } else {
                var bwc = upload_region(addr < void ? > (t.qblob[t.wcls_off]), uint64(c.vocab_size * dim))
                var bwc_s = upload_scales(t, t.wcls_off, c.vocab_size, dim)
                if (cls64) {
                    enc_gemm64_b(enc, bwc, 0ul, bwc_s, 0ul, bxf, blog, u_dim, u_vocab, u_s16, mp, c.vocab_size)
                } elif (use_gemm) {
                    enc_gemm_b(enc, bwc, 0ul, bwc_s, 0ul, bxf, blog, u_dim, u_vocab, u_s16, mp, c.vocab_size)
                } else {
                    enc_gemv_b(enc, nrows > 2l, bwc, 0ul, bwc_s, 0ul, bxf, blog, u_dim, u_vocab, u_s16, c.vocab_size)
                }
            }
        }
    }
    metal_end_encoding(enc)
    metal_release(enc)
    metal_commit(cb)
    cbs |> push(cb)
    g_us_bencode += int64(get_time_usec(ts_enc))
    // optimistic watermark at commit: the in-order queue writes row pos before any later-committed
    // work reads the mirror — the pipe's fast-path prepare must not re-upload it from the (not yet
    // written back) CPU cache
    for (i in range64(nrows)) {
        var sp = sessions[i]
        if (key_exists(g_mirrors, sp.uid)) {
            let wm0 = g_mirrors[sp.uid].watermark
            g_mirrors[sp.uid].watermark = max(wm0, ws.positions[i] + 1l)
        }
    }
    // the overlap: the PREVIOUS step completes only now, with this one already committed behind it
    finish_pending_step()
    g_lp_err = false   // a lost pipelined step was logged + counted; this step retries fresh
    // stash this step as the pending one (finish_pending_step just cleared every slot)
    g_lp_active = true
    g_lp_cbs <- cbs
    g_lp_pooled |> push((buf = bx, size = bytes_x))
    g_lp_pooled |> push((buf = bxb, size = bytes_xb))
    g_lp_pooled |> push((buf = bxb2, size = bytes_x))
    g_lp_pooled |> push((buf = bqkv, size = bytes_qkv))
    g_lp_pooled |> push((buf = bhb, size = bytes_h))
    g_lp_pooled |> push((buf = bhb2, size = bytes_h))
    if (use_attn_d || chunked) {
        g_lp_pooled |> push((buf = bpart, size = bytes_part))
    }
    if (bytes_sk > 0ul) {
        g_lp_pooled |> push((buf = bpsk, size = bytes_sk))
    }
    if (use_attn_d) {
        g_lp_upooled |> push((buf = u_nsgs, size = 4ul))
    } elif (chunked) {
        g_lp_upooled |> push((buf = u_nch, size = 4ul))
    }
    if (u_nblk != null) {
        g_lp_upooled |> push((buf = u_nblk, size = 4ul))
    }
    if (gpu_logits) {
        g_lp_pooled |> push((buf = bxf, size = bytes_x))
        g_lp_pooled |> push((buf = blog, size = bytes_log))
        g_lp_upooled |> push((buf = u_vocab, size = 4ul))
    }
    if (qkv_sk > 0l) {
        g_lp_upooled |> push((buf = u_sk_qkv, size = 4ul))
        g_lp_upooled |> push((buf = u_skt_qkv, size = 4ul))
    }
    if (wo_sk > 0l) {
        g_lp_upooled |> push((buf = u_sk_wo, size = 4ul))
    }
    if (w2_sk > 0l) {
        g_lp_upooled |> push((buf = u_sk_w2, size = 4ul))
    }
    if (wo_sk > 0l || w2_sk > 0l) {
        g_lp_upooled |> push((buf = u_skt_dim, size = 4ul))
    }
    if (mv) {
        g_lp_upooled |> push((buf = u_xs4_dim, size = 4ul))
        g_lp_upooled |> push((buf = u_xs4_qd, size = 4ul))
        g_lp_upooled |> push((buf = u_xs4_hid, size = 4ul))
        if (u_nrows_mv != null) {
            g_lp_upooled |> push((buf = u_nrows_mv, size = 4ul))
        }
        if (u_nrows_mv2 != null) {
            g_lp_upooled |> push((buf = u_nrows_mv2, size = 4ul))
        }
    }
    g_lp_upooled |> push((buf = bcos, size = bytes_tab))
    g_lp_upooled |> push((buf = bsin, size = bytes_tab))
    g_lp_upooled |> push((buf = brt, size = bytes_rt))
    g_lp_upooled |> push((buf = u_dim, size = 4ul))
    g_lp_upooled |> push((buf = u_qd, size = 4ul))
    g_lp_upooled |> push((buf = u_kvd, size = 4ul))
    g_lp_upooled |> push((buf = u_qkvd, size = 4ul))
    g_lp_upooled |> push((buf = u_hidden, size = 4ul))
    g_lp_upooled |> push((buf = u_hs, size = 4ul))
    g_lp_upooled |> push((buf = u_kvm, size = 4ul))
    g_lp_upooled |> push((buf = u_nh, size = 4ul))
    g_lp_upooled |> push((buf = u_scale, size = 4ul))
    g_lp_upooled |> push((buf = u_eps, size = 4ul))
    g_lp_upooled |> push((buf = u_npq, size = 4ul))
    g_lp_upooled |> push((buf = u_npall, size = 4ul))
    g_lp_upooled |> push((buf = u_s16, size = 4ul))
    g_lp_upooled |> push((buf = u_total_dim, size = 4ul))
    g_lp_upooled |> push((buf = u_total_h, size = 4ul))
    g_lp_upooled |> reserve((g_lp_upooled |> length()) + length(g_b_ulays))
    for (ul in g_b_ulays) {
        g_lp_upooled |> push((buf = ul, size = 4ul))
    }
    g_b_ulays |> clear()
    g_lp_blog = gpu_logits ? blog : null
    g_lp_xb = null
    if (!gpu_logits) {
        g_lp_xb = bx   // the CPU-classifier tail reads x_b — sync-only writeback
        unsafe {
            g_lp_xb_dst = addr < void ? > (ws.scr.x_b[0])
        }
        g_lp_xb_bytes = nrows * dim * 4l
    }
    g_lp_sessions |> clear()
    g_lp_sessions |> reserve(int(nrows))
    g_lp_positions |> resize(int(nrows))
    g_lp_koffs |> resize(int(nrows))
    g_lp_voffs |> resize(int(nrows))
    g_lp_caps |> resize(int(nrows))
    for (i in range64(nrows)) {
        g_lp_sessions |> push(sessions[i])
        g_lp_positions[i] = ws.positions[i]
        g_lp_koffs[i] = g_b_koffs[i]
        g_lp_voffs[i] = g_b_voffs[i]
        g_lp_caps[i] = g_b_caps[i]
    }
    g_lp_nrows = nrows
    g_lp_layers = c.n_layers
    g_lp_seq = c.seq_len
    g_lp_vocab = c.vocab_size
    g_lp_rowb = rowb
    if (!(g_b_pipe && gpu_logits && empty(g_skip))) {
        finish_pending_step()   // synchronous path: complete + read back immediately
        if (g_lp_err) {
            g_lp_err = false
            return false   // the dispatch failed — the caller reruns this step on the CPU loop
        }
    }
    if (gpu_logits) {
        batch_decode_override_logits_done()
    }
    return true
}

// ===== fused-GEMV cat blobs + the pending-step (encode-ahead) machinery =====

// fused-GEMV weight blobs: 2-3 tensors' q8 rows repacked into ONE contiguous block_q8_0 stream.
// Own cache (NOT the per-tensor blob cache — different content under a colliding base address);
// released in metal_decode_shutdown.
var private g_cat_blobs : table<uint64; MetalBuffer?>

// one q8 region (rows x n) interleaved into `base` at block offset blk0 — the same 34B layout
// as metal_blob_region, targeting a slice of a cat blob
def private repack_blocks(base : int8?; blk0 : int64; t : Model; off, rows, n : int64) {
    let nblocks = rows * (n / 32l)
    unsafe {
        var qp = addr < int8 ? > (t.qblob[off])
        if (t.wscale_f16) {
            var sp16 = addr < uint16 ? > (t.qscales16[off / 32l])
            for (b in range64(nblocks)) {
                let o = (blk0 + b) * 34l
                var pb = reinterpret<uint16?>(base + o)
                pb[0] = sp16[b]
                memcpy(reinterpret<void?>(base + o + 2l), reinterpret<void?>(qp + b * 32l), 32)
            }
        } else {
            var sp = addr < float ? > (t.qscales[off / 32l])
            for (b in range64(nblocks)) {
                let o = (blk0 + b) * 34l
                var ph = reinterpret<float16?>(base + o)
                ph[0] = float16(sp[b])
                memcpy(reinterpret<void?>(base + o + 2l), reinterpret<void?>(qp + b * 32l), 32)
            }
        }
    }
}

def private blob_cat3(t : Model; off0, rows0, off1, rows1, off2, rows2, n : int64) : MetalBuffer? {
    let key = intptr(unsafe(addr(t.qblob[off0])))
    if (key_exists(g_cat_blobs, key)) {
        return g_cat_blobs[key]
    }
    let nb = n / 32l
    var buf = metal_new_buffer_untracked(g_dev, uint64((rows0 + rows1 + rows2) * nb * 34l))
    var base = unsafe(reinterpret<int8?>(metal_buffer_contents(buf)))
    repack_blocks(base, 0l, t, off0, rows0, n)
    repack_blocks(base, rows0 * nb, t, off1, rows1, n)
    if (rows2 > 0l) {
        repack_blocks(base, (rows0 + rows1) * nb, t, off2, rows2, n)
    }
    g_cat_blobs[key] = buf
    return buf
}
// the pending pipelined step (one deep): everything finish_pending_step needs to wait, read
// back, and release AFTER the next step is committed behind it. Sessions are borrowed pointers
// (caller-owned) — always cleared, never deleted.
var private g_lp_active = false
var private g_lp_cbs : array<MetalCommandBuffer?>
var private g_lp_pooled : array<tuple<buf : MetalBuffer?; size : uint64>>
var private g_lp_upooled : array<tuple<buf : MetalBuffer?; size : uint64>>
var private g_lp_blog : MetalBuffer?
var private g_lp_xb : MetalBuffer?      // production sync + CPU-classifier path: x_b writeback
var private g_lp_xb_dst : void?         // ... into the caller's workspace (borrowed, sync-only)
var private g_lp_xb_bytes = 0l
var private g_lp_err = false            // sticky per-step dispatch-error signal for sync callers
var private g_lp_sessions : array<Session?>
var private g_lp_positions : array<int64>
var private g_lp_koffs : array<uint64>
var private g_lp_voffs : array<uint64>
var private g_lp_caps : array<int64>
var private g_lp_nrows = 0l
var private g_lp_layers = 0l
var private g_lp_seq = 0l
var private g_lp_vocab = 0l
var private g_lp_rowb = 0l
var private g_lp_prev_gpu_end = 0.0lf   // last finished step's gpu_end — the handoff stat's anchor

// wait + read back + release the pending step. x_b writes back only on the sync CPU-classifier
// path (g_lp_xb set): with gpu-logits the caller never reads it, and in pipe mode it would
// clobber the next step's freshly written embeds. A dispatch error raises g_lp_err (sync callers
// consume it and rerun on the CPU; a pipelined step is simply lost — logged + counted).
def private finish_pending_step {
    if (!g_lp_active) {
        return
    }
    let ts_wait = ref_time_ticks()
    // in-order queue: the last cb completing implies all of them did
    metal_wait_until_completed(g_lp_cbs[length(g_lp_cbs) - 1])
    var err = ""
    for (c_i in g_lp_cbs) {
        if (empty(err)) {
            err = metal_command_buffer_error(c_i)
        }
        g_sched_b_ms += (metal_command_buffer_kernel_end_time(c_i) - metal_command_buffer_kernel_start_time(c_i)) * 1000.0lf
    }
    // gpu = first-start .. last-end envelope; handoff = the TRUE inter-step GPU idle — this step's
    // start vs whichever bound it last (the previous step's gpu end, or its own sched end)
    let gs0 = metal_command_buffer_gpu_start_time(g_lp_cbs[0])
    let ge_last = metal_command_buffer_gpu_end_time(g_lp_cbs[length(g_lp_cbs) - 1])
    g_gpu_b_ms += (ge_last - gs0) * 1000.0lf
    g_handoff_b_ms += max(0.0lf, (gs0 - max(g_lp_prev_gpu_end, metal_command_buffer_kernel_end_time(g_lp_cbs[0]))) * 1000.0lf)
    if (has_env_variable("DASLLAMA_METAL_PIPE_DEBUG")) {   // per-step pipeline trace (first steps + outliers)
        let gap_ms = (gs0 - g_lp_prev_gpu_end) * 1000.0lf
        let env_ms = (ge_last - gs0) * 1000.0lf
        if (g_batch_steps < 8l || gap_ms > 1.0lf) {
            to_log(LOG_INFO, "metal pipe step {g_batch_steps}: gap={gap_ms}ms env={env_ms}ms\n")
        }
    }
    g_lp_prev_gpu_end = ge_last
    g_us_bwait += int64(get_time_usec(ts_wait))
    for (c_i in g_lp_cbs) {
        metal_release(c_i)
    }
    g_lp_cbs |> clear()   // released above — delete of array<T?> would double-free the pointees
    if (empty(err)) {
        let ts_rb = ref_time_ticks()
        unsafe {
            if (g_lp_xb != null) {   // sync CPU-classifier path only — pipe skips it (stale)
                memcpy(g_lp_xb_dst, metal_buffer_contents(g_lp_xb), int(g_lp_xb_bytes))
            }
            let ab = reinterpret<uint8?>(metal_buffer_contents(g_arena))
            let lgp = g_lp_blog != null ? reinterpret<uint8?>(metal_buffer_contents(g_lp_blog)) : null
            for (i in range64(g_lp_nrows)) {
                var sp = g_lp_sessions[i]
                let pos = g_lp_positions[i]
                let prow = kv_phys_row(*sp, pos)
                let cap = g_lp_caps[i]
                for (l in range64(g_lp_layers)) {
                    memcpy(reinterpret<void?>(kv_layer_kbase(*sp, l, g_lp_seq) + prow * g_lp_rowb),
                        reinterpret<void?>(ab + (g_lp_koffs[i] + uint64((l * cap + pos) * g_lp_rowb))), int(g_lp_rowb))
                    memcpy(reinterpret<void?>(kv_layer_vbase(*sp, l, g_lp_seq) + prow * g_lp_rowb),
                        reinterpret<void?>(ab + (g_lp_voffs[i] + uint64((l * cap + pos) * g_lp_rowb))), int(g_lp_rowb))
                }
                if (lgp != null) {
                    memcpy(addr < void ? > (sp.logits[0]),
                        reinterpret<void?>(lgp + i * g_lp_vocab * 4l), int(g_lp_vocab * 4l))
                }
                if (key_exists(g_mirrors, sp.uid)) {   // never regress the optimistic commit-time mark
                    let wm0 = g_mirrors[sp.uid].watermark
                    g_mirrors[sp.uid].watermark = max(wm0, pos + 1l)
                }
            }
        }
        g_us_breadback += int64(get_time_usec(ts_rb))
        g_batch_steps++
        g_batch_rows += g_lp_nrows
    } else {
        to_log(LOG_ERROR, "dasLLAMA metal batch decode: dispatch failed ({err}) — pipelined step lost\n")
        // the optimistic commit-time watermark now overstates the mirror (its rows are garbage at
        // and above this step's positions) — force a full re-upload from the CPU truth
        for (i in range64(g_lp_nrows)) {
            var sp = g_lp_sessions[i]
            if (key_exists(g_mirrors, sp.uid)) {
                g_mirrors[sp.uid].watermark = 0l
            }
        }
        g_lp_err = true
        batch_decline(MetalDecodeDecline.gpu_error)
    }
    for (pb in g_lp_pooled) {
        var b = pb.buf
        pool_release(g_pool, b, pb.size)
    }
    g_lp_pooled |> clear()
    for (pb in g_lp_upooled) {
        var b = pb.buf
        pool_release(g_upool, b, pb.size)
    }
    g_lp_upooled |> clear()
    g_lp_blog = null
    g_lp_xb = null
    g_lp_xb_dst = null
    g_lp_sessions |> clear()   // borrowed handles — clear, never delete
    g_lp_active = false
}

//! Complete any in-flight pipelined GPU decode step (the encode-ahead rails), landing its logits
//! and KV writeback. Benches call this before closing a timing window; no-op when nothing pends.
def public metal_decode_flush {
    finish_pending_step()
}


[init]
def dasllama_metal_decode_register() {
    // registered but dormant: only select_decode_override("metal") (env rail DASLLAMA_PIN_DECODE
    // in the harnesses) activates it — the batch twin rides the same name — and unsupported
    // shapes decline per call
    register_decode_override("metal", @@metal_decode_forward)
    register_batch_decode_override("metal", @@metal_batch_decode_forward)
    if (has_env_variable("DASLLAMA_METAL_DECODE_SKIP")) {
        g_skip = get_env_variable("DASLLAMA_METAL_DECODE_SKIP")
        to_log(LOG_WARNING, "dasLLAMA metal decode: SKIP={g_skip} — timing-attribution mode, output is GARBAGE\n")
    }
}
