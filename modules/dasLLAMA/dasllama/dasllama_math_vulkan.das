options gen2

module dasllama_math_vulkan shared public

require dasllama/dasllama_math
require dasllama/dasllama_par
require vulkan
require vulkan/vulkan_boost
require spirv/spirv_shader
require spirv/spirv_builtins public
require strings
require math
require daslib/fio

// dasLLAMA Vulkan GPU tier — per-layer MoE expert-stack offload. The loader gathers offloaded
// layers' PREPARED planes into row-major device planes — q8 stacks as int8 quant words + f16
// scales, native K-quant stacks (k4/k5/k6, fmt-tagged) as superblock quant payloads + decoded
// 20B scale rows — and hands them to vk_moe_upload_stack (moe_gpu_upload_resident — every load
// rail, gguf and mapped image alike); the MoE dispatches then route those layers' whole expert
// FFN chain (gate+up GEMMs -> act -> requant -> down) through vk_moe_ffn — decode row counts on
// the GEMV kernels, prefill batches on the per-format tiled GEMMs (q8 and native kq
// alike). Activations arrive in the layer's own form: Q8_0 per-32 scales for q8
// stacks, Q8_K per-256 for kq. Needs 8/16-bit SSBO storage + compute subgroups +
// shaderIntegerDotProduct (probed; any absent = tier off, CPU fallback).
//
// STREAMED layers (DASLLAMA_GPU_MOE_STREAM=N, the N layers below the resident set): the loader
// keeps their device-layout planes in pinned host mirrors; per prefill each layer's stacks DMA
// into one of two device slots whose pseudo-stacks re-base onto the layer, so the batch rail
// runs verbatim. After a layer's fenced chain, the NEXT layer's copy submits UNFENCED and rides
// the GPU-idle window while the CPU runs attention — the following chain's transfer->compute
// barrier orders it (in-order queue). Decode of streamed layers stays on the CPU planes.
//
// Submission model (v2): every stack carries its own descriptor set + host meta/xq/xs/y buffers,
// so descriptors are written ONCE at upload and command buffers are PRE-RECORDED (per stack, plus
// one per gate+up pair — the fused form the engine dispatches). Per token a dispatch is: refresh
// the mapped meta/activation contents, resubmit the recorded buffer, wait on the ONE reusable
// fence, memcpy y back. No per-dispatch allocation, recording, descriptor writes, or queue drain.
// Enabled by DASLLAMA_GPU_MOE_LAYERS=<K> (expert stacks of the LAST K layers).
//
// The registered "vulkan" KernelBackend below stays a named-only stub (priority -1, GEMV slots
// panic) — the MoE tier hooks are the real integration surface.

// ===== the kernel =====
// One SUBGROUP per output row of one region. Weights arrive as TWO row-major planes — packed
// int8 quant words + a float16 scale plane (16-bit storage) — NOT 34-byte blocks: an
// interleaved block layout byte-loads at stride 34 across the warp and measured ~21GB/s on the
// 3060 Ti (~1.2ms per dispatch — the real cost behind stage-1a's "1.35ms/dispatch"). Here 8
// lanes cooperate per 32-weight block (lane = one 4-byte word), so every warp load is 128
// CONSECUTIVE bytes — one transaction. The block's integer dot stays exact — sdot4 (dp4a) per
// word, then a 3-step shuffleXor fold across the 8 lanes; lane 0 applies the f16*f32 scales.
// Params ride in vk_meta (NOT push constants — the generated pusher drops small trailing
// scalars, dasVulkan ROADMAP): [0]=n [1]=d [2]=nregions [3]=pad, then (rel_wblock, xblock)
// uint pairs per region — 32-weight BLOCK offsets (weight blocks relative to the resident
// stack, activation blocks into vk_xq/vk_xs).

var @ssbo @binding = 0 vk_wq : array<uint>
var @ssbo @binding = 1 vk_ws : array<float16>
var @ssbo @binding = 2 vk_meta : array<uint>
var @ssbo @binding = 3 vk_xq : array<uint>
var @ssbo @binding = 4 vk_xs : array<float>
var @ssbo @binding = 5 vk_y : array<float>

[compute_shader(local_size_x=64, local_size_x_id=0, name="q8_moe_groupn_spv")]
def q8_moe_groupn {
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let nreg = vk_meta[2u]
    let rg = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
    if (rg < nreg * d) {
        let r = rg / d
        let row = rg % d
        let nb = n / 32u
        let wb0 = vk_meta[4u + r * 2u] + row * nb
        let xb0 = vk_meta[5u + r * 2u]
        let lane = gl_SubgroupInvocationID
        let my_word = lane % 8u
        let my_boff = lane / 8u
        let bpp = gl_SubgroupSize / 8u   // blocks per cooperative pass
        var acc = 0.0
        var bi0 = 0u
        while (bi0 < nb) {
            let b = bi0 + my_boff
            var idot = 0
            if (b < nb) {
                idot = sdot4(vk_wq[(wb0 + b) * 8u + my_word], vk_xq[(xb0 + b) * 8u + my_word])
            }
            idot += subgroupShuffleXor(idot, 1u)
            idot += subgroupShuffleXor(idot, 2u)
            idot += subgroupShuffleXor(idot, 4u)
            if (my_word == 0u && b < nb) {
                acc += float(idot) * float(vk_ws[wb0 + b]) * vk_xs[xb0 + b]
            }
            bi0 += bpp
        }
        let total = subgroupAdd(acc)
        if (subgroupElect()) {
            vk_y[rg] = total
        }
    }
}

// ===== the prefill kernel =====
// Region-list batched GEMM (the mul_mat_id shape, MoeGpuBatchFn): one workgroup computes a
// 32-row x 32-col output tile of one region, staging both operand tiles through @workgroup
// shared memory so weight rows stream from VRAM once per 32 activation rows (a GEMV-style loop
// would re-stream the whole expert stack once per activation row — ~6GB per dispatch at pp512).
// K advances 8 blocks (256 weights) per barrier pair — one block per pair serialized the whole
// kernel on the load->barrier->compute->barrier cadence (2 loads in flight/thread, 128 barriers
// per 2048-K pass); at 8 deep each thread issues 8+ staging loads back-to-back and barriers drop
// 8x. Same exact int block dots + fp32 block-order accumulation as the decode kernel. vk_meta:
// [0]=n [1]=d [2]=nregions [3]=map_off, per region (rel_wblock, row0, cnt, wg_base) quads,
// then the per-workgroup region map at map_off (workgroup id -> region index — no search).
// Activations read from device-local scratch (bindings 3/4 rebind) — a host-visible image
// re-read ~(d/32) times would cross PCIe every pass.

var @workgroup bt_xw4 : uint4[544]   // x tile: 32 rows x 16 uint4 chunks (8 blocks deep), stride 17
                                     // in v4 units — vectorized so the hot loop issues 1 shared load
                                     // per 4 sdot4 (scalar tiles were LSU-throughput-bound at 1:1)
var @workgroup bt_ww4 : uint4[544]   // w tile: 32 cols x 16 chunks, stride 17. A thread's 4 output
                                     // cols are word+{0,8,16,24} (NOT word*4+c: with 16B elements a
                                     // 4-col-contiguous warp lands 8 lanes on 2 banks; stride-8 cols
                                     // put the 8 lanes on 8 distinct banks every access phase)
var @workgroup bt_xs : float[256]    // x scales per (block, row); kq kernels use [row] only (per-256)
var @workgroup bt_ws : float[256]    // w scales per (block, col)

[compute_shader(local_size_x=256, name="q8_moe_batch_spv")]
def q8_moe_batch {
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let rid = vk_meta[vk_meta[3u] + gl_WorkGroupID.x]
    let rb = 4u + rid * 4u
    let wblk0 = vk_meta[rb]
    let row0 = vk_meta[rb + 1u]
    let cnt = vk_meta[rb + 2u]
    let tix = gl_WorkGroupID.x - vk_meta[rb + 3u]
    let wtiles = (d + 31u) / 32u
    let xt = tix / wtiles
    let wt = tix % wtiles
    let nbb = n / 32u
    let nsteps = (nbb + 7u) / 8u     // q8 K need not be superblock-aligned (nfe 1408 = 44 blocks)
    let tid = gl_LocalInvocationID.x
    let word = tid % 8u
    let grp = tid / 8u               // 0..31: the x row AND w col this thread stages
    let xrow = xt * 32u + grp
    let wcol = wt * 32u + grp
    let slane = tid % 32u            // scale staging: this thread covers (block sblk, lane slane)
    let sblk = tid / 32u
    var f0 = 0.0
    var f1 = 0.0
    var f2 = 0.0
    var f3 = 0.0
    var s = 0u
    while (s < nsteps) {
        let b0 = s * 8u
        for (ki in range(2)) {       // this thread stages v4 chunks word and word+8 of its row/col
            let k = word + uint(ki) * 8u
            let bu = k / 2u
            let h = (k % 2u) * 4u
            var xv = uint4(0u)
            if (xrow < cnt && b0 + bu < nbb) {
                let xb4 = ((row0 + xrow) * nbb + b0 + bu) * 8u + h
                xv = uint4(vk_xq[xb4], vk_xq[xb4 + 1u], vk_xq[xb4 + 2u], vk_xq[xb4 + 3u])
            }
            bt_xw4[grp * 17u + k] = xv
            var wv = uint4(0u)
            if (wcol < d && b0 + bu < nbb) {
                let wb4 = (wblk0 + wcol * nbb + b0 + bu) * 8u + h
                wv = uint4(vk_wq[wb4], vk_wq[wb4 + 1u], vk_wq[wb4 + 2u], vk_wq[wb4 + 3u])
            }
            bt_ww4[grp * 17u + k] = wv
        }
        var xs = 0.0
        if (xt * 32u + slane < cnt && b0 + sblk < nbb) {
            xs = vk_xs[(row0 + xt * 32u + slane) * nbb + b0 + sblk]
        }
        bt_xs[sblk * 32u + slane] = xs
        var ws = 0.0
        if (wt * 32u + slane < d && b0 + sblk < nbb) {
            ws = float(vk_ws[wblk0 + (wt * 32u + slane) * nbb + b0 + sblk])
        }
        bt_ws[sblk * 32u + slane] = ws
        barrier()
        for (bb in range(8)) {       // zero-filled tail blocks contribute 0 (ws = xs = 0)
            let bu = uint(bb)
            var i0 = 0
            var i1 = 0
            var i2 = 0
            var i3 = 0
            for (kk in range(2)) {
                let k = bu * 2u + uint(kk)
                let xw = bt_xw4[grp * 17u + k]
                let wv0 = bt_ww4[word * 17u + k]
                let wv1 = bt_ww4[(word + 8u) * 17u + k]
                let wv2 = bt_ww4[(word + 16u) * 17u + k]
                let wv3 = bt_ww4[(word + 24u) * 17u + k]
                i0 += sdot4(wv0.x, xw.x) + sdot4(wv0.y, xw.y) + sdot4(wv0.z, xw.z) + sdot4(wv0.w, xw.w)
                i1 += sdot4(wv1.x, xw.x) + sdot4(wv1.y, xw.y) + sdot4(wv1.z, xw.z) + sdot4(wv1.w, xw.w)
                i2 += sdot4(wv2.x, xw.x) + sdot4(wv2.y, xw.y) + sdot4(wv2.z, xw.z) + sdot4(wv2.w, xw.w)
                i3 += sdot4(wv3.x, xw.x) + sdot4(wv3.y, xw.y) + sdot4(wv3.z, xw.z) + sdot4(wv3.w, xw.w)
            }
            let xscl = bt_xs[bu * 32u + grp]
            f0 += float(i0) * bt_ws[bu * 32u + word] * xscl
            f1 += float(i1) * bt_ws[bu * 32u + word + 8u] * xscl
            f2 += float(i2) * bt_ws[bu * 32u + word + 16u] * xscl
            f3 += float(i3) * bt_ws[bu * 32u + word + 24u] * xscl
        }
        barrier()
        s++
    }
    if (xrow < cnt) {
        let yb = (row0 + xrow) * d + wt * 32u + word
        if (wt * 32u + word < d) {
            vk_y[yb] = f0
        }
        if (wt * 32u + word + 8u < d) {
            vk_y[yb + 8u] = f1
        }
        if (wt * 32u + word + 16u < d) {
            vk_y[yb + 16u] = f2
        }
        if (wt * 32u + word + 24u < d) {
            vk_y[yb + 24u] = f3
        }
    }
}

// ===== the FFN mid-chain kernels =====
// ONE fused act+requant step between the gate+up and down GEMMs, so the whole MoE FFN is one
// submit with a single mid-chain barrier and the hidden activations never cross PCIe. Only the
// globals a kernel references are emitted into its SPIR-V, so binding 0 doubles as a float view
// here (vk_upf, the up plane) while the GEMMs see it as uint weights; vk_xq/vk_xs (3/4) are the
// requant OUTPUT planes in this set. vk_meta (per FFN dispatch): [0] = element count, [1] =
// gelu flag, [2] = 32-block/superblock count.
//
// The act value is recomputed for the quant pass instead of staged (same expression, same f32
// ops — identical bits both times); vs the old separate act kernel this drops one dispatch, one
// barrier, and the activated plane's write+re-read. Act formulas mirror the CPU swiglu/geglu
// cores; exp/tanh precision differs from libm (the parity bar is token-identical, not
// bit-identical). Requant mirrors quantize_q8_0_into_ptr / quantize_q8_k_into_ptr: per-block
// |x| amax -> d = amax/127 -> q = round(x/d); GPU round() may split rounding-halfway cases
// differently — same parity bar. Q8_K block sums are NOT stored — the kq GEMMs recompute them
// via sdot4 vs 0x01010101 on activation words they load anyway.

var @ssbo @binding = 0 vk_upf : array<float>

// g = act(g) * u — the SwiGLU/GeGLU elementwise core (gelu taken as uint: the emitter's
// scalar params are numeric)
def private act_mul(v, u : float; gelu : uint) : float => (
    gelu != 0u ? (0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)))) * u
        : (v / (1.0 + exp(-v))) * u)

// fused act + q8_0 requant of the hidden rows (gate floats in vk_y, up in vk_upf -> vk_xq
// quant words + vk_xs per-32 scales), one thread per 32-block
[compute_shader(local_size_x=256, name="q8_moe_actrq_spv")]
def q8_moe_actrq {
    let b = gl_GlobalInvocationID.x
    if (b < vk_meta[2u]) {
        let base = b * 32u
        let gelu = vk_meta[1u]
        var amax = 0.0
        for (i in range(32)) {
            amax = max(amax, abs(act_mul(vk_y[base + uint(i)], vk_upf[base + uint(i)], gelu)))
        }
        let d = amax / 127.0
        let id = d != 0.0 ? 1.0 / d : 0.0
        vk_xs[b] = d
        for (w in range(8)) {
            let wb = base + uint(w) * 4u
            var word = 0u
            for (j in range(4)) {
                let q = int(round(act_mul(vk_y[wb + uint(j)], vk_upf[wb + uint(j)], gelu) * id))
                word = word | ((uint(q) & 0xFF) << (uint(j) * 8u))
            }
            vk_xq[b * 8u + uint(w)] = word
        }
    }
}

// the Q8_K twin for the K-quant chains: one thread per 256-weight superblock, per-256 scale
[compute_shader(local_size_x=256, name="q8k_moe_actrq_spv")]
def q8k_moe_actrq {
    let b = gl_GlobalInvocationID.x
    if (b < vk_meta[2u]) {
        let base = b * 256u
        let gelu = vk_meta[1u]
        var amax = 0.0
        for (i in range(256)) {
            amax = max(amax, abs(act_mul(vk_y[base + uint(i)], vk_upf[base + uint(i)], gelu)))
        }
        let d = amax / 127.0
        let id = d != 0.0 ? 1.0 / d : 0.0
        vk_xs[b] = d
        for (w in range(64)) {
            let wb = base + uint(w) * 4u
            var word = 0u
            for (j in range(4)) {
                let q = int(round(act_mul(vk_y[wb + uint(j)], vk_upf[wb + uint(j)], gelu) * id))
                word = word | ((uint(q) & 0xFF) << (uint(j) * 8u))
            }
            vk_xq[b * 64u + uint(w)] = word
        }
    }
}

// ===== the K-quant decode kernels =====
// Native k4/k5/k6 GEMVs over superblock quant payloads + decoded 20B scale rows, with Q8_K
// activations (per-256 f32 scales in vk_xs). Same one-subgroup-per-row shape as q8_moe_groupn,
// but 4 lanes cooperate per 32-block — a lane owns one nibble word pairing weights k / k+16, so
// both halves land in the SAME block and warp loads stay consecutive. The block dot is exact:
// nibble-composed quants are in [0, 63], which sdot4 eats as signed bytes; the K-quant min /
// offset term folds via in-kernel block sums (sdot4 vs 0x01010101 on the activation words the
// dot already loaded). Scale rows read through the binding-1 UINT view (vk_wsu — the q8 kernels
// see the same binding as float16), f16 d / dmin decoded by unpackHalf2x16. vk_meta uses
// SUPERBLOCK units here: [n, d, nrows, 0] then (rel_wsuperblock, xsuperblock) pairs per row.

var @ssbo @binding = 1 vk_wsu : array<uint>

// deposit 4 packed high-bits (one per nibble lane of hb) at bit 4 of each byte lane
def private k5_dep(hb : uint) : uint => (
    ((hb & 1u) << 4u) | ((hb & 2u) << 11u) | ((hb & 4u) << 18u) | ((hb & 8u) << 25u))

// Q4_K: w = d*sc*q - dmin*mn*1, q in [0,15] — per 32-block xs*(d*sc*idot - dmin*mn*bsum)
[compute_shader(local_size_x=64, local_size_x_id=0, name="kq_moe_gemv_k4_spv")]
def kq_moe_gemv_k4 {
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let nreg = vk_meta[2u]
    let rg = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
    if (rg < nreg * d) {
        let r = rg / d
        let row = rg % d
        let nsb = n / 256u
        let wsb0 = vk_meta[4u + r * 2u] + row * nsb
        let xsb0 = vk_meta[5u + r * 2u]
        let lane = gl_SubgroupInvocationID
        let j = lane % 4u
        let brel = lane / 4u
        let bstride = gl_SubgroupSize / 4u
        let nb = nsb * 8u
        var acc = 0.0
        var b0 = 0u
        while (b0 < nb) {
            let bb = b0 + brel
            if (bb < nb) {
                let sb = bb / 8u
                let blk = bb % 8u
                let wsb = wsb0 + sb
                let w = vk_wq[wsb * 32u + blk * 4u + j]
                let xw = (xsb0 + sb) * 64u + blk * 8u + j
                let xlo = vk_xq[xw]
                let xhi = vk_xq[xw + 4u]
                let idot = sdot4(w & 0x0F0F0F0F, xlo) + sdot4((w >> 4u) & 0x0F0F0F0F, xhi)
                let bs = sdot4(xlo, 0x01010101) + sdot4(xhi, 0x01010101)
                let dm = unpackHalf2x16(vk_wsu[wsb * 5u])
                let sc = int((vk_wsu[wsb * 5u + 1u + blk / 4u] >> ((blk % 4u) * 8u)) & 0xFF)
                let mn = int((vk_wsu[wsb * 5u + 3u + blk / 4u] >> ((blk % 4u) * 8u)) & 0xFF)
                acc += vk_xs[xsb0 + sb] * (dm.x * float(sc * idot) - dm.y * float(mn * bs))
            }
            b0 += bstride
        }
        let total = subgroupAdd(acc)
        if (subgroupElect()) {
            vk_y[rg] = total
        }
    }
}

// Q5_K: the k4 fold with the 5th bit OR'd in from the 32B qh region (quants in [0, 31])
[compute_shader(local_size_x=64, local_size_x_id=0, name="kq_moe_gemv_k5_spv")]
def kq_moe_gemv_k5 {
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let nreg = vk_meta[2u]
    let rg = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
    if (rg < nreg * d) {
        let r = rg / d
        let row = rg % d
        let nsb = n / 256u
        let wsb0 = vk_meta[4u + r * 2u] + row * nsb
        let xsb0 = vk_meta[5u + r * 2u]
        let lane = gl_SubgroupInvocationID
        let j = lane % 4u
        let brel = lane / 4u
        let bstride = gl_SubgroupSize / 4u
        let nb = nsb * 8u
        var acc = 0.0
        var b0 = 0u
        while (b0 < nb) {
            let bb = b0 + brel
            if (bb < nb) {
                let sb = bb / 8u
                let blk = bb % 8u
                let wsb = wsb0 + sb
                let w = vk_wq[wsb * 40u + blk * 4u + j]
                let hb = (vk_wq[wsb * 40u + 32u + blk] >> (j * 8u)) & 0xFF
                let xw = (xsb0 + sb) * 64u + blk * 8u + j
                let xlo = vk_xq[xw]
                let xhi = vk_xq[xw + 4u]
                let idot = (sdot4((w & 0x0F0F0F0F) | k5_dep(hb & 0xF), xlo)
                    + sdot4(((w >> 4u) & 0x0F0F0F0F) | k5_dep(hb >> 4u), xhi))
                let bs = sdot4(xlo, 0x01010101) + sdot4(xhi, 0x01010101)
                let dm = unpackHalf2x16(vk_wsu[wsb * 5u])
                let sc = int((vk_wsu[wsb * 5u + 1u + blk / 4u] >> ((blk % 4u) * 8u)) & 0xFF)
                let mn = int((vk_wsu[wsb * 5u + 3u + blk / 4u] >> ((blk % 4u) * 8u)) & 0xFF)
                acc += vk_xs[xsb0 + sb] * (dm.x * float(sc * idot) - dm.y * float(mn * bs))
            }
            b0 += bstride
        }
        let total = subgroupAdd(acc)
        if (subgroupElect()) {
            vk_y[rg] = total
        }
    }
}

// Q6_K: q = (ql | qh<<4) - 32 with per-16 SIGNED int8 sub-scales — the -32 folds as o = 32*s
// against the per-16 block sums (the CPU portable-dot formula); lo/hi nibble halves of a block
// are exactly its two 16-weight sub-scale groups
[compute_shader(local_size_x=64, local_size_x_id=0, name="kq_moe_gemv_k6_spv")]
def kq_moe_gemv_k6 {
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let nreg = vk_meta[2u]
    let rg = gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID
    if (rg < nreg * d) {
        let r = rg / d
        let row = rg % d
        let nsb = n / 256u
        let wsb0 = vk_meta[4u + r * 2u] + row * nsb
        let xsb0 = vk_meta[5u + r * 2u]
        let lane = gl_SubgroupInvocationID
        let j = lane % 4u
        let brel = lane / 4u
        let bstride = gl_SubgroupSize / 4u
        let nb = nsb * 8u
        var acc = 0.0
        var b0 = 0u
        while (b0 < nb) {
            let bb = b0 + brel
            if (bb < nb) {
                let sb = bb / 8u
                let blk = bb % 8u
                let wsb = wsb0 + sb
                let w = vk_wq[wsb * 48u + blk * 4u + j]
                let qhw = wsb * 48u + 32u + (blk / 4u) * 8u + j
                let shift = (blk % 4u) * 2u
                let lo6 = (w & 0x0F0F0F0F) | (((vk_wq[qhw] >> shift) & 0x03030303) << 4u)
                let hi6 = ((w >> 4u) & 0x0F0F0F0F) | (((vk_wq[qhw + 4u] >> shift) & 0x03030303) << 4u)
                let xw = (xsb0 + sb) * 64u + blk * 8u + j
                let xlo = vk_xq[xw]
                let xhi = vk_xq[xw + 4u]
                let ilo = sdot4(lo6, xlo)
                let ihi = sdot4(hi6, xhi)
                let blo = sdot4(xlo, 0x01010101)
                let bhi = sdot4(xhi, 0x01010101)
                let s4 = int4(unpack8(int(vk_wsu[wsb * 5u + blk / 2u])))   // 4 signed sub-scales
                let slo = (blk % 2u) == 0u ? s4.x : s4.z
                let shi = (blk % 2u) == 0u ? s4.y : s4.w
                let dd = unpackHalf2x16(vk_wsu[wsb * 5u + 4u]).x
                acc += vk_xs[xsb0 + sb] * dd * float(slo * (ilo - 32 * blo) + shi * (ihi - 32 * bhi))
            }
            b0 += bstride
        }
        let total = subgroupAdd(acc)
        if (subgroupElect()) {
            vk_y[rg] = total
        }
    }
}

// ===== the K-quant batch kernels =====
// The prefill tile for native k4/k5/k6 stacks — same 32x32 region-list GEMM shape (and meta
// layout) as q8_moe_batch, but region weight offsets are in SUPERBLOCK units and the staged
// w tile holds COMPOSED byte quants: the staging thread reads the nibble words (+ qh bits) and
// writes 8 words per column whose byte lanes are the block's quants in [0, 63] — the compute
// loop is then the exact q8 tile dot (sdot4 over 8 words), with the K-quant min/offset terms
// folded per 16-weight half via inline block sums (sdot4 vs 0x01010101 on x words the dot
// already loads from the tile — 8 threads per row recompute them, cheaper than a staging role
// on this bandwidth-bound kernel). Per-block scales stage as two decoded floats per column
// (k4/k5: d*sc and dmin*mn; k6: d*slo and d*shi per 16-half); x scales are Q8_K per-256.
// K advances one whole superblock (8 blocks) per barrier pair — step index == superblock index,
// so the per-column scale words are step-local and every staging thread decodes one (col, blk).

var @workgroup bt_wsb : float[256]   // second per-(block, col) scale plane (bt_ws carries the first)

// Q4_K tile: per block acc += xs * (d*sc*idot - dmin*mn*bsum)
[compute_shader(local_size_x=256, name="kq_moe_batch_k4_spv")]
def kq_moe_batch_k4 {
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let rid = vk_meta[vk_meta[3u] + gl_WorkGroupID.x]
    let rb = 4u + rid * 4u
    let wsb0 = vk_meta[rb]
    let row0 = vk_meta[rb + 1u]
    let cnt = vk_meta[rb + 2u]
    let tix = gl_WorkGroupID.x - vk_meta[rb + 3u]
    let wtiles = (d + 31u) / 32u
    let xt = tix / wtiles
    let wt = tix % wtiles
    let nsb = n / 256u
    let tid = gl_LocalInvocationID.x
    let word = tid % 8u
    let grp = tid / 8u
    let xrow = xt * 32u + grp
    let wcol = wt * 32u + grp
    let slane = tid % 32u
    let sblk = tid / 32u
    var f0 = 0.0
    var f1 = 0.0
    var f2 = 0.0
    var f3 = 0.0
    var s = 0u
    while (s < nsb) {
        for (ki in range(2)) {
            let k = word + uint(ki) * 8u
            let bu = k / 2u
            let hh = k % 2u          // 0 = lo-nibble words 0..3 of the block, 1 = hi words 4..7
            var xv = uint4(0u)
            if (xrow < cnt) {
                let xb4 = ((row0 + xrow) * nsb + s) * 64u + bu * 8u + hh * 4u
                xv = uint4(vk_xq[xb4], vk_xq[xb4 + 1u], vk_xq[xb4 + 2u], vk_xq[xb4 + 3u])
            }
            bt_xw4[grp * 17u + k] = xv
            var wv = uint4(0u)
            if (wcol < d) {
                let wsb = wsb0 + wcol * nsb + s
                let rb4 = wsb * 32u + bu * 4u
                let r0 = vk_wq[rb4]
                let r1 = vk_wq[rb4 + 1u]
                let r2 = vk_wq[rb4 + 2u]
                let r3 = vk_wq[rb4 + 3u]
                if (hh == 0u) {
                    wv = uint4(r0 & 0x0F0F0F0F, r1 & 0x0F0F0F0F, r2 & 0x0F0F0F0F, r3 & 0x0F0F0F0F)
                } else {
                    wv = uint4((r0 >> 4u) & 0x0F0F0F0F, (r1 >> 4u) & 0x0F0F0F0F,
                        (r2 >> 4u) & 0x0F0F0F0F, (r3 >> 4u) & 0x0F0F0F0F)
                }
            }
            bt_ww4[grp * 17u + k] = wv
        }
        if (tid < 32u) {
            var xs = 0.0
            if (xt * 32u + tid < cnt) {
                xs = vk_xs[(row0 + xt * 32u + tid) * nsb + s]
            }
            bt_xs[tid] = xs
        }
        var sa = 0.0
        var sbv = 0.0
        if (wt * 32u + slane < d) {
            let wsb = wsb0 + (wt * 32u + slane) * nsb + s
            let dm = unpackHalf2x16(vk_wsu[wsb * 5u])
            let sc = int((vk_wsu[wsb * 5u + 1u + sblk / 4u] >> ((sblk % 4u) * 8u)) & 0xFF)
            let mn = int((vk_wsu[wsb * 5u + 3u + sblk / 4u] >> ((sblk % 4u) * 8u)) & 0xFF)
            sa = dm.x * float(sc)
            sbv = dm.y * float(mn)
        }
        bt_ws[sblk * 32u + slane] = sa
        bt_wsb[sblk * 32u + slane] = sbv
        barrier()
        let xscl = bt_xs[grp]
        for (bb in range(8)) {
            let bu = uint(bb)
            var i0 = 0
            var i1 = 0
            var i2 = 0
            var i3 = 0
            var bs = 0
            for (kk in range(2)) {
                let k = bu * 2u + uint(kk)
                let xw = bt_xw4[grp * 17u + k]
                let wv0 = bt_ww4[word * 17u + k]
                let wv1 = bt_ww4[(word + 8u) * 17u + k]
                let wv2 = bt_ww4[(word + 16u) * 17u + k]
                let wv3 = bt_ww4[(word + 24u) * 17u + k]
                bs += sdot4(xw.x, 0x01010101) + sdot4(xw.y, 0x01010101) + sdot4(xw.z, 0x01010101) + sdot4(xw.w, 0x01010101)
                i0 += sdot4(wv0.x, xw.x) + sdot4(wv0.y, xw.y) + sdot4(wv0.z, xw.z) + sdot4(wv0.w, xw.w)
                i1 += sdot4(wv1.x, xw.x) + sdot4(wv1.y, xw.y) + sdot4(wv1.z, xw.z) + sdot4(wv1.w, xw.w)
                i2 += sdot4(wv2.x, xw.x) + sdot4(wv2.y, xw.y) + sdot4(wv2.z, xw.z) + sdot4(wv2.w, xw.w)
                i3 += sdot4(wv3.x, xw.x) + sdot4(wv3.y, xw.y) + sdot4(wv3.z, xw.z) + sdot4(wv3.w, xw.w)
            }
            f0 += xscl * (bt_ws[bu * 32u + word] * float(i0) - bt_wsb[bu * 32u + word] * float(bs))
            f1 += xscl * (bt_ws[bu * 32u + word + 8u] * float(i1) - bt_wsb[bu * 32u + word + 8u] * float(bs))
            f2 += xscl * (bt_ws[bu * 32u + word + 16u] * float(i2) - bt_wsb[bu * 32u + word + 16u] * float(bs))
            f3 += xscl * (bt_ws[bu * 32u + word + 24u] * float(i3) - bt_wsb[bu * 32u + word + 24u] * float(bs))
        }
        barrier()
        s++
    }
    if (xrow < cnt) {
        let yb = (row0 + xrow) * d + wt * 32u + word
        if (wt * 32u + word < d) {
            vk_y[yb] = f0
        }
        if (wt * 32u + word + 8u < d) {
            vk_y[yb + 8u] = f1
        }
        if (wt * 32u + word + 16u < d) {
            vk_y[yb + 16u] = f2
        }
        if (wt * 32u + word + 24u < d) {
            vk_y[yb + 24u] = f3
        }
    }
}

// Q5_K tile: the k4 compose with the 5th bit OR'd in from the qh region
[compute_shader(local_size_x=256, name="kq_moe_batch_k5_spv")]
def kq_moe_batch_k5 {
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let rid = vk_meta[vk_meta[3u] + gl_WorkGroupID.x]
    let rb = 4u + rid * 4u
    let wsb0 = vk_meta[rb]
    let row0 = vk_meta[rb + 1u]
    let cnt = vk_meta[rb + 2u]
    let tix = gl_WorkGroupID.x - vk_meta[rb + 3u]
    let wtiles = (d + 31u) / 32u
    let xt = tix / wtiles
    let wt = tix % wtiles
    let nsb = n / 256u
    let tid = gl_LocalInvocationID.x
    let word = tid % 8u
    let grp = tid / 8u
    let xrow = xt * 32u + grp
    let wcol = wt * 32u + grp
    let slane = tid % 32u
    let sblk = tid / 32u
    var f0 = 0.0
    var f1 = 0.0
    var f2 = 0.0
    var f3 = 0.0
    var s = 0u
    while (s < nsb) {
        for (ki in range(2)) {
            let k = word + uint(ki) * 8u
            let bu = k / 2u
            let hh = k % 2u
            var xv = uint4(0u)
            if (xrow < cnt) {
                let xb4 = ((row0 + xrow) * nsb + s) * 64u + bu * 8u + hh * 4u
                xv = uint4(vk_xq[xb4], vk_xq[xb4 + 1u], vk_xq[xb4 + 2u], vk_xq[xb4 + 3u])
            }
            bt_xw4[grp * 17u + k] = xv
            var wv = uint4(0u)
            if (wcol < d) {
                let wsb = wsb0 + wcol * nsb + s
                let rb4 = wsb * 40u + bu * 4u
                let r0 = vk_wq[rb4]
                let r1 = vk_wq[rb4 + 1u]
                let r2 = vk_wq[rb4 + 2u]
                let r3 = vk_wq[rb4 + 3u]
                let hbw = vk_wq[wsb * 40u + 32u + bu]
                let h0 = hbw & 0xFF
                let h1 = (hbw >> 8u) & 0xFF
                let h2 = (hbw >> 16u) & 0xFF
                let h3 = (hbw >> 24u) & 0xFF
                if (hh == 0u) {
                    wv = uint4((r0 & 0x0F0F0F0F) | k5_dep(h0 & 0xF), (r1 & 0x0F0F0F0F) | k5_dep(h1 & 0xF),
                        (r2 & 0x0F0F0F0F) | k5_dep(h2 & 0xF), (r3 & 0x0F0F0F0F) | k5_dep(h3 & 0xF))
                } else {
                    wv = uint4(((r0 >> 4u) & 0x0F0F0F0F) | k5_dep(h0 >> 4u), ((r1 >> 4u) & 0x0F0F0F0F) | k5_dep(h1 >> 4u),
                        ((r2 >> 4u) & 0x0F0F0F0F) | k5_dep(h2 >> 4u), ((r3 >> 4u) & 0x0F0F0F0F) | k5_dep(h3 >> 4u))
                }
            }
            bt_ww4[grp * 17u + k] = wv
        }
        if (tid < 32u) {
            var xs = 0.0
            if (xt * 32u + tid < cnt) {
                xs = vk_xs[(row0 + xt * 32u + tid) * nsb + s]
            }
            bt_xs[tid] = xs
        }
        var sa = 0.0
        var sbv = 0.0
        if (wt * 32u + slane < d) {
            let wsb = wsb0 + (wt * 32u + slane) * nsb + s
            let dm = unpackHalf2x16(vk_wsu[wsb * 5u])
            let sc = int((vk_wsu[wsb * 5u + 1u + sblk / 4u] >> ((sblk % 4u) * 8u)) & 0xFF)
            let mn = int((vk_wsu[wsb * 5u + 3u + sblk / 4u] >> ((sblk % 4u) * 8u)) & 0xFF)
            sa = dm.x * float(sc)
            sbv = dm.y * float(mn)
        }
        bt_ws[sblk * 32u + slane] = sa
        bt_wsb[sblk * 32u + slane] = sbv
        barrier()
        let xscl = bt_xs[grp]
        for (bb in range(8)) {
            let bu = uint(bb)
            var i0 = 0
            var i1 = 0
            var i2 = 0
            var i3 = 0
            var bs = 0
            for (kk in range(2)) {
                let k = bu * 2u + uint(kk)
                let xw = bt_xw4[grp * 17u + k]
                let wv0 = bt_ww4[word * 17u + k]
                let wv1 = bt_ww4[(word + 8u) * 17u + k]
                let wv2 = bt_ww4[(word + 16u) * 17u + k]
                let wv3 = bt_ww4[(word + 24u) * 17u + k]
                bs += sdot4(xw.x, 0x01010101) + sdot4(xw.y, 0x01010101) + sdot4(xw.z, 0x01010101) + sdot4(xw.w, 0x01010101)
                i0 += sdot4(wv0.x, xw.x) + sdot4(wv0.y, xw.y) + sdot4(wv0.z, xw.z) + sdot4(wv0.w, xw.w)
                i1 += sdot4(wv1.x, xw.x) + sdot4(wv1.y, xw.y) + sdot4(wv1.z, xw.z) + sdot4(wv1.w, xw.w)
                i2 += sdot4(wv2.x, xw.x) + sdot4(wv2.y, xw.y) + sdot4(wv2.z, xw.z) + sdot4(wv2.w, xw.w)
                i3 += sdot4(wv3.x, xw.x) + sdot4(wv3.y, xw.y) + sdot4(wv3.z, xw.z) + sdot4(wv3.w, xw.w)
            }
            f0 += xscl * (bt_ws[bu * 32u + word] * float(i0) - bt_wsb[bu * 32u + word] * float(bs))
            f1 += xscl * (bt_ws[bu * 32u + word + 8u] * float(i1) - bt_wsb[bu * 32u + word + 8u] * float(bs))
            f2 += xscl * (bt_ws[bu * 32u + word + 16u] * float(i2) - bt_wsb[bu * 32u + word + 16u] * float(bs))
            f3 += xscl * (bt_ws[bu * 32u + word + 24u] * float(i3) - bt_wsb[bu * 32u + word + 24u] * float(bs))
        }
        barrier()
        s++
    }
    if (xrow < cnt) {
        let yb = (row0 + xrow) * d + wt * 32u + word
        if (wt * 32u + word < d) {
            vk_y[yb] = f0
        }
        if (wt * 32u + word + 8u < d) {
            vk_y[yb + 8u] = f1
        }
        if (wt * 32u + word + 16u < d) {
            vk_y[yb + 16u] = f2
        }
        if (wt * 32u + word + 24u < d) {
            vk_y[yb + 24u] = f3
        }
    }
}

// Q6_K tile: composed 6-bit quants in [0, 63], the -32 folded per 16-half as slo/shi vs the
// half's block sum — the lo/hi word split (j 0..3 / 4..7) IS the sub-scale split
[compute_shader(local_size_x=256, name="kq_moe_batch_k6_spv")]
def kq_moe_batch_k6 {
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let rid = vk_meta[vk_meta[3u] + gl_WorkGroupID.x]
    let rb = 4u + rid * 4u
    let wsb0 = vk_meta[rb]
    let row0 = vk_meta[rb + 1u]
    let cnt = vk_meta[rb + 2u]
    let tix = gl_WorkGroupID.x - vk_meta[rb + 3u]
    let wtiles = (d + 31u) / 32u
    let xt = tix / wtiles
    let wt = tix % wtiles
    let nsb = n / 256u
    let tid = gl_LocalInvocationID.x
    let word = tid % 8u
    let grp = tid / 8u
    let xrow = xt * 32u + grp
    let wcol = wt * 32u + grp
    let slane = tid % 32u
    let sblk = tid / 32u
    var f0 = 0.0
    var f1 = 0.0
    var f2 = 0.0
    var f3 = 0.0
    var s = 0u
    while (s < nsb) {
        for (ki in range(2)) {
            let k = word + uint(ki) * 8u
            let bu = k / 2u
            let hh = k % 2u
            var xv = uint4(0u)
            if (xrow < cnt) {
                let xb4 = ((row0 + xrow) * nsb + s) * 64u + bu * 8u + hh * 4u
                xv = uint4(vk_xq[xb4], vk_xq[xb4 + 1u], vk_xq[xb4 + 2u], vk_xq[xb4 + 3u])
            }
            bt_xw4[grp * 17u + k] = xv
            var wv = uint4(0u)
            if (wcol < d) {
                let wsb = wsb0 + wcol * nsb + s
                let rb4 = wsb * 48u + bu * 4u
                let r0 = vk_wq[rb4]
                let r1 = vk_wq[rb4 + 1u]
                let r2 = vk_wq[rb4 + 2u]
                let r3 = vk_wq[rb4 + 3u]
                let qb4 = wsb * 48u + 32u + (bu / 4u) * 8u + hh * 4u   // words w and w+4 qh adjoin
                let q0 = vk_wq[qb4]
                let q1 = vk_wq[qb4 + 1u]
                let q2 = vk_wq[qb4 + 2u]
                let q3 = vk_wq[qb4 + 3u]
                let shift = (bu % 4u) * 2u
                if (hh == 0u) {
                    wv = uint4((r0 & 0x0F0F0F0F) | (((q0 >> shift) & 0x03030303) << 4u),
                        (r1 & 0x0F0F0F0F) | (((q1 >> shift) & 0x03030303) << 4u),
                        (r2 & 0x0F0F0F0F) | (((q2 >> shift) & 0x03030303) << 4u),
                        (r3 & 0x0F0F0F0F) | (((q3 >> shift) & 0x03030303) << 4u))
                } else {
                    wv = uint4(((r0 >> 4u) & 0x0F0F0F0F) | (((q0 >> shift) & 0x03030303) << 4u),
                        ((r1 >> 4u) & 0x0F0F0F0F) | (((q1 >> shift) & 0x03030303) << 4u),
                        ((r2 >> 4u) & 0x0F0F0F0F) | (((q2 >> shift) & 0x03030303) << 4u),
                        ((r3 >> 4u) & 0x0F0F0F0F) | (((q3 >> shift) & 0x03030303) << 4u))
                }
            }
            bt_ww4[grp * 17u + k] = wv
        }
        if (tid < 32u) {
            var xs = 0.0
            if (xt * 32u + tid < cnt) {
                xs = vk_xs[(row0 + xt * 32u + tid) * nsb + s]
            }
            bt_xs[tid] = xs
        }
        var sa = 0.0
        var sbv = 0.0
        if (wt * 32u + slane < d) {
            let wsb = wsb0 + (wt * 32u + slane) * nsb + s
            let s4 = int4(unpack8(int(vk_wsu[wsb * 5u + sblk / 2u])))
            let slo = (sblk % 2u) == 0u ? s4.x : s4.z
            let shi = (sblk % 2u) == 0u ? s4.y : s4.w
            let dd = unpackHalf2x16(vk_wsu[wsb * 5u + 4u]).x
            sa = dd * float(slo)
            sbv = dd * float(shi)
        }
        bt_ws[sblk * 32u + slane] = sa
        bt_wsb[sblk * 32u + slane] = sbv
        barrier()
        let xscl = bt_xs[grp]
        for (bb in range(8)) {
            let bu = uint(bb)
            let kL = bu * 2u
            let kH = bu * 2u + 1u
            let xwL = bt_xw4[grp * 17u + kL]
            let xwH = bt_xw4[grp * 17u + kH]
            let blo = sdot4(xwL.x, 0x01010101) + sdot4(xwL.y, 0x01010101) + sdot4(xwL.z, 0x01010101) + sdot4(xwL.w, 0x01010101)
            let bhi = sdot4(xwH.x, 0x01010101) + sdot4(xwH.y, 0x01010101) + sdot4(xwH.z, 0x01010101) + sdot4(xwH.w, 0x01010101)
            let wL0 = bt_ww4[word * 17u + kL]
            let wL1 = bt_ww4[(word + 8u) * 17u + kL]
            let wL2 = bt_ww4[(word + 16u) * 17u + kL]
            let wL3 = bt_ww4[(word + 24u) * 17u + kL]
            let wH0 = bt_ww4[word * 17u + kH]
            let wH1 = bt_ww4[(word + 8u) * 17u + kH]
            let wH2 = bt_ww4[(word + 16u) * 17u + kH]
            let wH3 = bt_ww4[(word + 24u) * 17u + kH]
            let l0 = sdot4(wL0.x, xwL.x) + sdot4(wL0.y, xwL.y) + sdot4(wL0.z, xwL.z) + sdot4(wL0.w, xwL.w)
            let l1 = sdot4(wL1.x, xwL.x) + sdot4(wL1.y, xwL.y) + sdot4(wL1.z, xwL.z) + sdot4(wL1.w, xwL.w)
            let l2 = sdot4(wL2.x, xwL.x) + sdot4(wL2.y, xwL.y) + sdot4(wL2.z, xwL.z) + sdot4(wL2.w, xwL.w)
            let l3 = sdot4(wL3.x, xwL.x) + sdot4(wL3.y, xwL.y) + sdot4(wL3.z, xwL.z) + sdot4(wL3.w, xwL.w)
            let h0 = sdot4(wH0.x, xwH.x) + sdot4(wH0.y, xwH.y) + sdot4(wH0.z, xwH.z) + sdot4(wH0.w, xwH.w)
            let h1 = sdot4(wH1.x, xwH.x) + sdot4(wH1.y, xwH.y) + sdot4(wH1.z, xwH.z) + sdot4(wH1.w, xwH.w)
            let h2 = sdot4(wH2.x, xwH.x) + sdot4(wH2.y, xwH.y) + sdot4(wH2.z, xwH.z) + sdot4(wH2.w, xwH.w)
            let h3 = sdot4(wH3.x, xwH.x) + sdot4(wH3.y, xwH.y) + sdot4(wH3.z, xwH.z) + sdot4(wH3.w, xwH.w)
            f0 += xscl * (bt_ws[bu * 32u + word] * float(l0 - 32 * blo) + bt_wsb[bu * 32u + word] * float(h0 - 32 * bhi))
            f1 += xscl * (bt_ws[bu * 32u + word + 8u] * float(l1 - 32 * blo) + bt_wsb[bu * 32u + word + 8u] * float(h1 - 32 * bhi))
            f2 += xscl * (bt_ws[bu * 32u + word + 16u] * float(l2 - 32 * blo) + bt_wsb[bu * 32u + word + 16u] * float(h2 - 32 * bhi))
            f3 += xscl * (bt_ws[bu * 32u + word + 24u] * float(l3 - 32 * blo) + bt_wsb[bu * 32u + word + 24u] * float(h3 - 32 * bhi))
        }
        barrier()
        s++
    }
    if (xrow < cnt) {
        let yb = (row0 + xrow) * d + wt * 32u + word
        if (wt * 32u + word < d) {
            vk_y[yb] = f0
        }
        if (wt * 32u + word + 8u < d) {
            vk_y[yb + 8u] = f1
        }
        if (wt * 32u + word + 16u < d) {
            vk_y[yb + 16u] = f2
        }
        if (wt * 32u + word + 24u < d) {
            vk_y[yb + 24u] = f3
        }
    }
}

// ===== device state (process-lifetime; never finalized by design) =====

let private WEIGHT_CAP = 4_200_000_000l    // default resident-weight cap. The WDDM cliff is TOTAL
                                           // commit (~6.5GiB of 8GB measured: q8-era K=8 at 4.7GB
                                           // weights + 1.7GB desktop thrashed nonlinearly), and other
                                           // processes are INVISIBLE to the Vulkan budget query (NV
                                           // reports 7.6GB "budget" while nvidia-smi shows 2.6GB in
                                           // use by others) — so the default stays conservative and
                                           // DASLLAMA_GPU_VRAM_MB overrides it for bigger cards
let private VRAM_RESERVE = 805_306_368l    // headroom below the queried heap budget when it clamps
                                           // the cap down: the tier's own scratch (~210MB batch
                                           // planes) + eviction slack
let private STAGING_BYTES = 268_435_456l   // one upload chunk (largest single memcpy)
let private XQ_BYTES = 1_048_576l          // per-stack activation quants capacity
let private XS_BYTES = 262_144l            // ... their scales
let private Y_BYTES = 1_048_576l           // per-stack output capacity
let private META_BYTES = 4_096l            // params + region table
let private MAX_STACKS = 224               // descriptor pool capacity (dense attention planes add
                                           // ~130 small stacks on top of experts+cls; VRAM budget
                                           // still caps the heavy ones far lower)
let private WG_X = 64u                     // kernel workgroup width (spec-const axis 0 can retune it)
let private BATCH_XQ_BYTES = 33_554_432l   // device-local gathered-activation scratch (chunk cap)
let private BATCH_XS_BYTES = 8_388_608l    // ... its scale plane
let private BATCH_Y_BYTES = 67_108_864l    // one output plane (two exist: gate/single + up)
let private BATCH_META_BYTES = 262_144l    // per-stack batch params + regions + workgroup map
let private BATCH_TILE = 32l               // kernel tile edge (rows and cols)
let private BATCH_CHUNK_ROWS = 8_192l      // dispatch row-window cap (bounds y and the wg map)
let private BATCH_HQ_BYTES = 33_554_432l   // device requantized-hidden quants (the down GEMM's activations)
let private BATCH_HS_BYTES = 8_388_608l    // ... its scale plane
let private FFN_META_BYTES = 64l           // act/requant params (element count, gelu flag, block count)
let private CLS_META_BYTES = 64l           // classifier GEMV params (one region, filled once)
let private STREAM_RESERVE = 1_100_000_000l  // VRAM reserved out of the weight budget for the two
                                             // streaming slots when DASLLAMA_GPU_MOE_STREAM is on
                                             // (35B kq measured: 2 x 532.7MB slots = 1065MB;
                                             // ensure_stream_slots asserts actual <= reserve)

struct private HostBuf {
    buf : uint64       // VkBuffer
    mem : uint64       // VkDeviceMemory
    mapped : void?     // persistently mapped (host-coherent)
    bytes : int64
}

struct private VkStack {
    base : int64             // first weight-element offset covered (into the q8 plane)
    elems : int64            // element count (offs resolve against [base, base+elems))
    fmt : int                // int(KqFmt): 0 = q8, 1/2/3 = k4/k5/k6 — picks kernels + meta units
    wqbytes : int64          // quant plane bytes (q8: elems; kq: elems/256 * {128,160,192})
    wqbuf : uint64           // device-local row-major quants (bound as array<uint>)
    wsbuf : uint64           // device-local scale plane (q8: f16 halfwords; kq: 20B decoded rows)
    wsbytes : int64
    // decode/GEMV dispatch state (lazy — first GEMV or cls use creates it; dense-GEMM stacks
    // never do, saving ~2.3MB host + 1MB device per stack across the ~130 dense planes)
    raw_set : VkDescriptorSet   // this stack's 6 bindings, written on first GEMV use
    meta : HostBuf
    xq : HostBuf
    xs : HostBuf
    y : HostBuf              // host-cached readback staging (the same-submit DMA copy lands here)
    y_dev : uint64           // device-local y the kernel writes — see batch_y1_dev on why
    gemv_made : bool
    batch_meta : HostBuf     // prefill batch meta STAGING (lazy — first batch dispatch creates it)
    batch_meta_dev : uint64  // device copy the kernel reads — a host-visible meta costs ~26us of
                             // serialized uncached PCIe reads PER WORKGROUP (measured, 12K wgs/dispatch)
    batch_set : VkDescriptorSet   // batch bindings: stack planes + act-image scratch + a y plane
    batch_y : uint64         // which shared y plane batch_set binds (role-fixed per stack)
    batch_xq : uint64        // which activation-quant plane it binds (gate/up: gathered x; down: hq)
    batch_made : bool
    // classifier dispatch state (lazy — set up on the first cls GEMV against this stack)
    cls_set : VkDescriptorSet
    cls_meta_dev : uint64    // device meta — vocab-row dispatches run ~150K subgroups, each
                             // reading params; host-visible meta would be uncached PCIe reads
    cls_cmd : VkCommandBuffer
    cls_made : bool
}

// one pinned-host-mirror weight stack of a STREAMED layer (device-layout planes kept in
// host-visible memory, DMA'd into a slot per prefill). Stacks arrive gate/up/down per layer
// (loader walk order, descending layers) — group g = stacks [3g, 3g+3), runtime-next = g-1.
struct private StreamStack {
    base : int64
    elems : int64
    fmt : int
    wqbytes : int64
    wsbytes : int64
    hwq : HostBuf            // pinned mirror, quant plane (transfer_src)
    hws : HostBuf            // ... scale plane
}

// one streaming slot: 3 pseudo-stacks (entries in g_gpu.stacks whose plane buffers are the
// slot's device scratch, re-based per layer so find_stack + the whole batch rail work verbatim)
struct private StreamSlot {
    group : int              // stream group currently resident in the slot (-1 = empty)
    s1 : int                 // pseudo-stack indices: gate / up / down
    s3 : int
    s2 : int
}

// pre-recorded decode-form FFN chain per (gate, up, down) stack triple: GEMV pair, fused
// act+requant, down GEMV, y DMA — refreshed per token via the stacks' host-visible metas,
// resubmitted as-is
struct private FfnCmd {
    cmd : VkCommandBuffer
    actrq_set : VkDescriptorSet   // gate y_dev + up y_dev (vk_upf) -> hq/hs device planes
    down_set : VkDescriptorSet    // down stack planes + hq/hs as activations + its y_dev
    nrows : int64                 // row count cmd was recorded with
}

struct private GpuState {
    instance : Instance
    device : Device
    phys : VkPhysicalDevice
    fam : uint
    queue : VkQueue
    pool : CommandPool
    set_layout : DescriptorSetLayout
    desc_pool : DescriptorPool
    pipe_layout : PipelineLayout
    pipeline : Pipeline
    batch_pipeline : Pipeline   // prefill batched-GEMM kernel (same set/pipe layout)
    actrq_pipeline : Pipeline   // fused act(g)*u + q8_0 requant of the hidden rows
    gemv_k4 : Pipeline          // native K-quant GEMVs (per-format decode kernels)
    gemv_k5 : Pipeline
    gemv_k6 : Pipeline
    actrq_k_pipeline : Pipeline // fused act + Q8_K requant twin (per-256 scale) for the kq chains
    batch_k4 : Pipeline         // native K-quant prefill tiles (per-format batch kernels)
    batch_k5 : Pipeline
    batch_k6 : Pipeline
    staging : HostBuf
    fence : VkFence          // the one reusable submit fence (reset after every wait)
    rows_per_wg : int64      // subgroups per workgroup = output rows per workgroup
    weight_budget : int64    // resident-weight cap — queried heap budget minus reserve, or VRAM_BUDGET
    @do_not_delete stacks : array<VkStack>
    ffn_cmds : table<int64; FfnCmd>   // (gate, up, down stack idx triple) -> decode FFN chain
    resident_bytes : int64
    dev_mem : table<uint64; uint64>   // device buffer -> its VkDeviceMemory (rollback destroy needs it)
    // FFN mid-chain state (lazy — first FFN dispatch creates it)
    hq_dev : uint64          // device requantized-hidden quants (the down GEMM's activation image)
    hs_dev : uint64          // ... its scales
    ffn_meta : HostBuf       // act/requant params staging
    ffn_meta_dev : uint64
    ffn_ready : bool
    // prefill batch state (lazy — first batch dispatch creates it; decode-only runs skip it)
    batch_xq_dev : uint64    // device-local gathered-activation quants (chunk window)
    batch_xs_dev : uint64    // ... their scales
    // y is written DEVICE-LOCAL and DMA-copied out in the same submit: shader scatter-stores
    // straight into host-visible memory measured ~100MB/s (~350ms per pp512 down dispatch);
    // the copy engine moves the same bytes in ~2ms
    batch_y1_dev : uint64    // gate / single output plane (kernel writes; also the down output)
    batch_y2_dev : uint64    // up output plane (the fused pair's second write target)
    batch_y1 : HostBuf       // host-cached readback staging for batch_y1_dev
    batch_y2 : HostBuf       // ... and batch_y2_dev (idle since the FFN fold — kept for debug taps)
    batch_actrq_set : VkDescriptorSet   // batch twin of FfnCmd's fused act+requant set (y1/y2 planes)
    batch_cmd : VkCommandBuffer   // one reusable batch cmd buffer, re-recorded per submit
    batch_ready : bool
    // streamed-prefill state (lazy — first streamed dispatch creates the slots)
    @do_not_delete stream_stacks : array<StreamStack>
    stream_slots : array<StreamSlot>
    stream_cmd : VkCommandBuffer  // reusable plane-copy cmd (safe: a fenced chain wait always
                                  // intervenes between two recordings — see copy_stream_group)
    stream_max_gu_wq : int64      // per-plane capacity maxima across streamed stacks
    stream_max_gu_ws : int64
    stream_max_dn_wq : int64
    stream_max_dn_ws : int64
    stream_ready : bool
}

var private g_gpu : GpuState?
var private g_init_failed = false
var private g_vk_prof = false      // DASLLAMA_GPU_PROF=1: per-piece batch-chain timing to log
var private g_vk_prof_probed = false

def private vk_prof : bool {
    if (!g_vk_prof_probed) {
        g_vk_prof_probed = true
        g_vk_prof = get_env_variable("DASLLAMA_GPU_PROF") == "1"
    }
    return g_vk_prof
}

def private dev_raw : VkDevice {
    return boost_value_to_vk(g_gpu.device)
}

// host-visible memory comes in two flavors on discrete cards: write-combined (fast CPU writes,
// ~100x SLOWER CPU reads) and host-cached. Buffers the CPU reads back (the y planes) MUST be
// cached — a 33MB memcpy out of a WC mapping measured ~400ms/layer at pp512
def private find_host_mem(type_bits : uint; cached : bool) : uint {
    if (cached) {
        var mp : VkPhysicalDeviceMemoryProperties
        vkGetPhysicalDeviceMemoryProperties(g_gpu.phys, mp)
        for (i in range(int(mp.memoryTypeCount))) {
            let fl = mp.memoryTypes[i].propertyFlags
            if ((type_bits & (1u << uint(i))) != 0u && fl.host_visible && fl.host_coherent && fl.host_cached) {
                return uint(i)
            }
        }
    }
    var want : VkMemoryPropertyFlags
    want.host_visible = true
    want.host_coherent = true
    return find_memory_type(g_gpu.phys, type_bits, want)
}

def private make_host_buf(bytes : int64; storage : bool; cached : bool = false) : HostBuf {
    var bci : BufferCreateInfo
    bci.size = uint64(bytes)
    bci.usage.storage_buffer = storage
    bci.usage.transfer_src = !storage
    bci.usage.transfer_dst = true   // readback staging targets (y DMA copies land here)
    var b <- create_buffer(g_gpu.device, bci)   // deliberately never finalized (process-lifetime)
    let req = get_buffer_memory_requirements(g_gpu.device, b)
    let mai = MemoryAllocateInfo(allocationSize = req.size,
                                 memoryTypeIndex = find_host_mem(req.memoryTypeBits, cached))
    var m <- allocate_memory(g_gpu.device, mai)
    bind_buffer_memory(g_gpu.device, b, m, 0ul)
    var mapped : void? = null
    let mf : VkMemoryMapFlags
    unsafe {
        vk_check(vkMapMemory(dev_raw(), boost_value_to_vk(m), 0ul, uint64(bytes), mf, addr(mapped)), null)
    }
    return HostBuf(buf = b._vk, mem = m._vk, mapped = mapped, bytes = bytes)
}

def private make_device_buf(bytes : int64) : uint64 {
    var bci : BufferCreateInfo
    bci.size = uint64(bytes)
    bci.usage.storage_buffer = true
    bci.usage.transfer_dst = true
    bci.usage.transfer_src = true   // y planes DMA back out (device -> host staging)
    var b <- create_buffer(g_gpu.device, bci)
    let req = get_buffer_memory_requirements(g_gpu.device, b)
    var want : VkMemoryPropertyFlags
    want.device_local = true
    let mai = MemoryAllocateInfo(allocationSize = req.size,
                                 memoryTypeIndex = find_memory_type(g_gpu.phys, req.memoryTypeBits, want))
    var m <- allocate_memory(g_gpu.device, mai)
    bind_buffer_memory(g_gpu.device, b, m, 0ul)
    g_gpu.dev_mem |> insert(b._vk, m._vk)
    return b._vk
}

def private destroy_device_buf(h : uint64) {
    unsafe {
        vkDestroyBuffer(dev_raw(), reinterpret<VkBuffer>(h), null)
        if (key_exists(g_gpu.dev_mem, h)) {
            vkFreeMemory(dev_raw(), reinterpret<VkDeviceMemory>(g_gpu.dev_mem[h]), null)
            g_gpu.dev_mem |> erase(h)
        }
    }
}

def private destroy_host_buf(hb : HostBuf) {   // vkFreeMemory implicitly unmaps
    unsafe {
        vkDestroyBuffer(dev_raw(), reinterpret<VkBuffer>(hb.buf), null)
        vkFreeMemory(dev_raw(), reinterpret<VkDeviceMemory>(hb.mem), null)
    }
}

def private nonowning_buf(h : uint64) : Buffer {
    var b : Buffer
    b._vk = h
    return b
}

def private vk_moe_init : bool {
    if (g_init_failed) {   // checked FIRST: a partially-built g_gpu must never report ready
        return false
    }
    if (g_gpu != null) {
        return true
    }
    g_init_failed = true   // cleared only on full success
    if (volkInitialize() != 0) {
        to_log(LOG_ERROR, "dasLLAMA vulkan tier: no Vulkan loader\n")
        return false
    }
    g_gpu = new GpuState()
    g_gpu.instance <- create_instance("dasLLAMA vulkan tier", make_api_version(1u, 3u, 0u))
    volkLoadInstance(boost_value_to_vk(g_gpu.instance))
    g_gpu.phys = select_physical_device(g_gpu.instance)
    if (!storage_8_16_supported(g_gpu.phys) || !subgroup_compute_ops_supported(g_gpu.phys)
            || !integer_dot_product_supported(g_gpu.phys)) {
        to_log(LOG_ERROR, "dasLLAMA vulkan tier: device lacks 8/16-bit SSBO storage, compute subgroup ops, or shaderIntegerDotProduct — tier disabled\n")
        return false
    }
    g_gpu.fam = select_graphics_queue_family(g_gpu.phys)
    g_gpu.device <- create_device_storage_8_16_int_dot(g_gpu.phys, g_gpu.fam)
    volkLoadDevice(dev_raw())
    g_gpu.queue = get_device_queue(g_gpu.device, g_gpu.fam, 0u)
    let sg = subgroup_properties(g_gpu.phys).subgroupSize
    g_gpu.rows_per_wg = int64((WG_X + sg - 1u) / sg)   // = gl_NumSubgroups for our 1D workgroup
    // resident-weight cap: the conservative default (see WEIGHT_CAP), an explicit
    // DASLLAMA_GPU_VRAM_MB override winning outright, and the queried heap budget clamping DOWN
    // only (trustworthy as an upper bound — small cards; NOT as free-VRAM, see WEIGHT_CAP).
    // Physical-device-level query — the extension only needs to be advertised, not enabled.
    g_gpu.weight_budget = WEIGHT_CAP
    let vram_override = has_env_variable("DASLLAMA_GPU_VRAM_MB")
    if (vram_override) {
        let mbv = int64(to_int(get_env_variable("DASLLAMA_GPU_VRAM_MB")))
        if (mbv > 0l) {
            g_gpu.weight_budget = mbv * 1_000_000l
        }
    }
    if (device_extension_available(g_gpu.phys, "VK_EXT_memory_budget")) {
        var mb = VkPhysicalDeviceMemoryBudgetPropertiesEXT()   // sType pre-filled by the ctor
        var mp2 = VkPhysicalDeviceMemoryProperties2()
        unsafe {
            mp2.pNext = addr<void?>(mb)
        }
        vkGetPhysicalDeviceMemoryProperties2(g_gpu.phys, mp2)
        var best = -1
        for (i in range(int(mp2.memoryProperties.memoryHeapCount))) {
            if (mp2.memoryProperties.memoryHeaps[i].flags.device_local
                    && (best < 0 || mb.heapBudget[i] > mb.heapBudget[best])) {
                best = i
            }
        }
        if (best >= 0 && mb.heapBudget[best] > 0ul) {
            if (!vram_override) {
                g_gpu.weight_budget = clamp(int64(mb.heapBudget[best]) - VRAM_RESERVE, 0l, g_gpu.weight_budget)
            }
            to_log(LOG_INFO, "dasLLAMA vulkan tier: device heap budget {int64(mb.heapBudget[best]) / 1000000l} MB (used {int64(mb.heapUsage[best]) / 1000000l}), weight cap {g_gpu.weight_budget / 1000000l} MB\n")
        }
    }
    if (to_int(get_env_variable("DASLLAMA_GPU_MOE_STREAM")) > 0) {
        g_gpu.weight_budget = max(g_gpu.weight_budget - STREAM_RESERVE, 0l)   // room for the 2 slots
    }
    var pci : CommandPoolCreateInfo
    pci.queueFamilyIndex = g_gpu.fam
    pci.flags.reset_command_buffer = true   // per-stack buffers re-record if a shape ever changes
    g_gpu.pool <- create_command_pool(g_gpu.device, pci)

    // one layout, MAX_STACKS sets: each resident stack gets its own set at upload time
    var dslci : DescriptorSetLayoutCreateInfo
    dslci.pBindings |> reserve(6)
    for (bind in range(6)) {
        var b0 = DescriptorSetLayoutBinding(binding = uint(bind),
                                            descriptorType = VkDescriptorType.STORAGE_BUFFER,
                                            descriptorCount = 1u)
        b0.stageFlags.compute = true
        dslci.pBindings |> emplace(b0)
    }
    g_gpu.set_layout <- create_descriptor_set_layout(g_gpu.device, dslci)
    var dpci : DescriptorPoolCreateInfo
    dpci.maxSets = uint(4 * MAX_STACKS)   // decode + batch set per stack, 3 per FFN triple, batch act/rq
    var psize = DescriptorPoolSize(type_ = VkDescriptorType.STORAGE_BUFFER, descriptorCount = uint(6 * 4 * MAX_STACKS))
    dpci.pPoolSizes |> push(psize)
    g_gpu.desc_pool <- create_descriptor_pool(g_gpu.device, dpci)

    var shader <- create_shader_module(g_gpu.device, q8_moe_groupn_spv)   // never finalized (kept via pipeline)
    var plci : PipelineLayoutCreateInfo
    plci.pSetLayouts <- [weak_copy(g_gpu.set_layout)]
    g_gpu.pipe_layout <- create_pipeline_layout(g_gpu.device, plci)
    g_gpu.pipeline <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, shader)
    var bshader <- create_shader_module(g_gpu.device, q8_moe_batch_spv)   // same 6-binding layout
    g_gpu.batch_pipeline <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, bshader)
    var ashader <- create_shader_module(g_gpu.device, q8_moe_actrq_spv)
    g_gpu.actrq_pipeline <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, ashader)
    var k4shader <- create_shader_module(g_gpu.device, kq_moe_gemv_k4_spv)
    g_gpu.gemv_k4 <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, k4shader)
    var k5shader <- create_shader_module(g_gpu.device, kq_moe_gemv_k5_spv)
    g_gpu.gemv_k5 <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, k5shader)
    var k6shader <- create_shader_module(g_gpu.device, kq_moe_gemv_k6_spv)
    g_gpu.gemv_k6 <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, k6shader)
    var rkshader <- create_shader_module(g_gpu.device, q8k_moe_actrq_spv)
    g_gpu.actrq_k_pipeline <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, rkshader)
    var bk4shader <- create_shader_module(g_gpu.device, kq_moe_batch_k4_spv)
    g_gpu.batch_k4 <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, bk4shader)
    var bk5shader <- create_shader_module(g_gpu.device, kq_moe_batch_k5_spv)
    g_gpu.batch_k5 <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, bk5shader)
    var bk6shader <- create_shader_module(g_gpu.device, kq_moe_batch_k6_spv)
    g_gpu.batch_k6 <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, bk6shader)

    var fence <- create_fence(g_gpu.device, FenceCreateInfo())   // never finalized (process-lifetime)
    g_gpu.fence = boost_value_to_vk(fence)
    g_gpu.staging = make_host_buf(STAGING_BYTES, false)
    g_init_failed = false
    to_log(LOG_INFO, "dasLLAMA vulkan tier: device ready (subgroup {int(sg)}, {g_gpu.rows_per_wg} rows/wg)\n")
    return true
}

// ===== upload (loader hook — moe_gpu_upload_resident gathers, we just stage) =====

def private staged_upload(dst : uint64; dst_off : int64; bytes : int64) {
    var regions : array<BufferCopy>
    let rc = BufferCopy(srcOffset = 0ul, dstOffset = uint64(dst_off), size = uint64(bytes))
    regions |> push(rc)
    run_cmd_sync(g_gpu.device, g_gpu.pool, g_gpu.queue) $(cmd) {
        cmd_copy_buffer(cmd, nonowning_buf(g_gpu.staging.buf), nonowning_buf(dst), regions)
    }
}

def private upload_region(dst : uint64; src : uint8 const?; total : int64) {
    var off = 0l
    while (off < total) {
        let chunk = min(total - off, STAGING_BYTES)
        unsafe {
            memcpy(g_gpu.staging.mapped, reinterpret<void? -const>(src + off), int(chunk))
        }
        staged_upload(dst, off, chunk)
        off += chunk
    }
}

def private alloc_desc_set : VkDescriptorSet {
    var dsai = VkDescriptorSetAllocateInfo()
    dsai.descriptorPool = boost_value_to_vk(g_gpu.desc_pool)
    dsai.descriptorSetCount = 1u
    var raw_set_layout = boost_value_to_vk(g_gpu.set_layout)
    var raw : VkDescriptorSet
    unsafe {
        dsai.pSetLayouts = addr(raw_set_layout)
        vk_check(vkAllocateDescriptorSets(dev_raw(), dsai, addr(raw)), null)
    }
    return raw
}

def private write_buf_desc(var writes : array<WriteDescriptorSet>; set_ : VkDescriptorSet; binding : uint; buf : uint64; bytes : int64) {
    var w = WriteDescriptorSet(dstSet = vk_value_to_boost(set_), dstBinding = binding,
                               descriptorType = VkDescriptorType.STORAGE_BUFFER, descriptorCount = 1u)
    var binfo = DescriptorBufferInfo(buffer = nonowning_buf(buf), range_ = uint64(bytes))
    w.pBufferInfo |> push(binfo)
    writes |> emplace(w)
}

// weight plane byte sizes for one [rows x n] stack in the tier's device layout
def private stack_plane_bytes(n, rows : int64; fmt : int) : tuple<wq : int64; ws : int64> {
    assert(fmt == 0 || n % 256l == 0l, "dasLLAMA vulkan tier: kq stack row length must be a superblock multiple")
    let nsb = rows * (n / 256l)
    let wqbytes = fmt == 0 ? rows * n : nsb * (fmt == 1 ? 128l : (fmt == 2 ? 160l : 192l))
    let wsbytes = fmt == 0 ? rows * (n / 32l) * 2l : nsb * 20l
    return (wq = wqbytes, ws = wsbytes)
}

// build a VkStack shell (weight plane buffers only — the GEMV/cls scratch is lazy, see
// ensure_gemv_scratch). Resident stacks pass their exact plane sizes; streaming pseudo-stacks
// pass slot capacities and a base of -1 (re-based per layer).
def private make_stack_shell(base, elems : int64; fmt : int; wqcap, wscap : int64) : int {
    var st = VkStack(base = base, elems = elems, fmt = fmt, wqbytes = wqcap,
                     wqbuf = make_device_buf(wqcap), wsbuf = make_device_buf(wscap), wsbytes = wscap)
    g_gpu.stacks |> push(st)
    return length(g_gpu.stacks) - 1
}

// decode/GEMV scratch for one stack (per-stack host bufs + the written 6-binding set). Lazy:
// only stacks that actually serve GEMV or cls dispatches pay for it — dense-GEMM planes never do.
def private ensure_gemv_scratch(si : int) {
    if (g_gpu.stacks[si].gemv_made) {
        return
    }
    g_gpu.stacks[si].raw_set = alloc_desc_set()
    g_gpu.stacks[si].meta = make_host_buf(META_BYTES, true)
    g_gpu.stacks[si].xq = make_host_buf(XQ_BYTES, true)
    g_gpu.stacks[si].xs = make_host_buf(XS_BYTES, true)
    g_gpu.stacks[si].y = make_host_buf(Y_BYTES, false, [cached = true])
    g_gpu.stacks[si].y_dev = make_device_buf(Y_BYTES)
    var writes : array<WriteDescriptorSet>
    write_buf_desc(writes, g_gpu.stacks[si].raw_set, 0u, g_gpu.stacks[si].wqbuf, g_gpu.stacks[si].wqbytes)
    write_buf_desc(writes, g_gpu.stacks[si].raw_set, 1u, g_gpu.stacks[si].wsbuf, g_gpu.stacks[si].wsbytes)
    write_buf_desc(writes, g_gpu.stacks[si].raw_set, 2u, g_gpu.stacks[si].meta.buf, META_BYTES)
    write_buf_desc(writes, g_gpu.stacks[si].raw_set, 3u, g_gpu.stacks[si].xq.buf, XQ_BYTES)
    write_buf_desc(writes, g_gpu.stacks[si].raw_set, 4u, g_gpu.stacks[si].xs.buf, XS_BYTES)
    write_buf_desc(writes, g_gpu.stacks[si].raw_set, 5u, g_gpu.stacks[si].y_dev, Y_BYTES)
    let no_copies : array<CopyDescriptorSet>
    update_descriptor_sets(g_gpu.device, writes, no_copies)
    g_gpu.stacks[si].gemv_made = true
}

// stream arm: keep the gathered device-layout planes in a pinned host mirror (host-visible
// memory the copy engine DMA-streams into a slot per prefill). No VRAM spent here.
def private vk_moe_stream_stack(wq : uint8 const?; ws : uint8 const?; woff : int64; n, rows : int64; fmt : int) : bool {
    let pb = stack_plane_bytes(n, rows, fmt)
    var ss = StreamStack(base = woff, elems = rows * n, fmt = fmt, wqbytes = pb.wq, wsbytes = pb.ws,
                         hwq = make_host_buf(pb.wq, false), hws = make_host_buf(pb.ws, false))
    unsafe {
        memcpy(ss.hwq.mapped, reinterpret<void? -const>(wq), int(pb.wq))
        memcpy(ss.hws.mapped, reinterpret<void? -const>(ws), int(pb.ws))
    }
    if (length(g_gpu.stream_stacks) % 3 == 2) {   // walk order gate/up/down: this one is a down
        g_gpu.stream_max_dn_wq = max(g_gpu.stream_max_dn_wq, pb.wq)
        g_gpu.stream_max_dn_ws = max(g_gpu.stream_max_dn_ws, pb.ws)
    } else {
        g_gpu.stream_max_gu_wq = max(g_gpu.stream_max_gu_wq, pb.wq)
        g_gpu.stream_max_gu_ws = max(g_gpu.stream_max_gu_ws, pb.ws)
    }
    g_gpu.stream_stacks |> emplace(ss)
    return true
}

//! MoeGpuUploadFn: upload one weight stack ([rows x n] at element offset woff) from the
//! loader's gathered row-major planes. q8 (fmt 0): int8 quants + f16 halfword scales; native
//! K-quant (fmt = int(KqFmt) k4/k5/k6): superblock quant payloads + decoded 20B scale rows.
//! stream = pinned host mirror for per-prefill DMA streaming instead of resident VRAM.
def vk_moe_upload_stack(wq : uint8 const?; ws : uint8 const?; woff : int64; n : int64; rows : int64; fmt : int; stream : bool) : bool {
    if (!vk_moe_init()) {
        return false
    }
    if (stream) {
        return vk_moe_stream_stack(wq, ws, woff, n, rows, fmt)
    }
    let pb = stack_plane_bytes(n, rows, fmt)
    let wqbytes = pb.wq
    let wsbytes = pb.ws
    if (g_gpu.resident_bytes + wqbytes + wsbytes > g_gpu.weight_budget) {
        to_log(LOG_INFO, "dasLLAMA vulkan tier: VRAM budget reached at {g_gpu.resident_bytes} bytes\n")
        return false
    }
    if (length(g_gpu.stacks) >= MAX_STACKS) {
        to_log(LOG_INFO, "dasLLAMA vulkan tier: descriptor capacity reached at {length(g_gpu.stacks)} stacks\n")
        return false
    }
    let si = make_stack_shell(woff, rows * n, fmt, wqbytes, wsbytes)
    upload_region(g_gpu.stacks[si].wqbuf, wq, wqbytes)
    upload_region(g_gpu.stacks[si].wsbuf, ws, wsbytes)
    g_gpu.resident_bytes += wqbytes + wsbytes
    return true
}

//! MoeGpuDropFn: roll back the last n uploads (a partial layer triple the budget cut mid-way).
//! Resident: pop the stacks, free their plane buffers, refund the budget; stream: pop the
//! pinned mirrors (slot capacity maxima stay — conservatively oversized slots are harmless).
//! Contract: only freshly-uploaded stacks (no dispatch state yet) may be dropped.
def vk_moe_drop_stacks(n : int; stream : bool) {
    if (stream) {
        for (_i in range(n)) {
            let k = length(g_gpu.stream_stacks) - 1
            destroy_host_buf(g_gpu.stream_stacks[k].hwq)
            destroy_host_buf(g_gpu.stream_stacks[k].hws)
            g_gpu.stream_stacks |> resize(k)
        }
        return
    }
    for (_i in range(n)) {
        let k = length(g_gpu.stacks) - 1
        assert(!g_gpu.stacks[k].gemv_made && !g_gpu.stacks[k].batch_made && !g_gpu.stacks[k].cls_made,
            "dasLLAMA vulkan tier: dropping a stack with live dispatch state")
        g_gpu.resident_bytes -= g_gpu.stacks[k].wqbytes + g_gpu.stacks[k].wsbytes
        destroy_device_buf(g_gpu.stacks[k].wqbuf)
        destroy_device_buf(g_gpu.stacks[k].wsbuf)
        g_gpu.stacks |> resize(k)
    }
}

// ===== dispatch plumbing =====

// stack resolution is by (format, offset) — weight element offsets live in PER-FORMAT plane
// spaces (q8 qblob vs the k4/k5/k6 planes), so a bare offset is ambiguous across formats (a k4
// gate and a k5 down can share the numeric range); within one format's space ranges are disjoint
def private find_stack_idx(w0 : int64; fmt : int) : int {
    for (i in range(length(g_gpu.stacks))) {
        if (g_gpu.stacks[i].fmt == fmt && w0 >= g_gpu.stacks[i].base && w0 < g_gpu.stacks[i].base + g_gpu.stacks[i].elems) {
            return i
        }
    }
    return -1
}

def private find_stack(w0 : int64; fmt : int) : int {
    let i = find_stack_idx(w0, fmt)
    if (i < 0) {
        panic("dasLLAMA vulkan tier: dispatch against a non-resident weight region (fmt {fmt} element offset {w0})")
    }
    return i
}

def private alloc_cmd : VkCommandBuffer {
    var ai = VkCommandBufferAllocateInfo()
    ai.commandPool = boost_value_to_vk(g_gpu.pool)
    ai.level = VkCommandBufferLevel.PRIMARY
    ai.commandBufferCount = 1u
    var raw : VkCommandBuffer
    unsafe {
        vk_check(vkAllocateCommandBuffers(dev_raw(), ai, addr(raw)), null)
    }
    return raw
}

def private cmd_bind_dispatch(raw : VkCommandBuffer; var set_ : VkDescriptorSet; groups : uint) {
    unsafe {
        vkCmdBindDescriptorSets(raw, VkPipelineBindPoint.COMPUTE, boost_value_to_vk(g_gpu.pipe_layout), 0u, 1u, addr(set_), 0u, null)
    }
    vkCmdDispatch(raw, groups, 1u, 1u)
}

// bind the GEMV pipeline for a stack's format (0 = the q8 kernel, 1/2/3 = kq_moe_gemv_k4/5/6)
def private bind_gemv_pipe(raw : VkCommandBuffer; fmt : int) {
    if (fmt == 0) {
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.pipeline, VkPipelineBindPoint.COMPUTE)
    } elif (fmt == 1) {
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.gemv_k4, VkPipelineBindPoint.COMPUTE)
    } elif (fmt == 2) {
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.gemv_k5, VkPipelineBindPoint.COMPUTE)
    } else {
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.gemv_k6, VkPipelineBindPoint.COMPUTE)
    }
}

// ... and the batch-tile twin (0 = q8_moe_batch, 1/2/3 = kq_moe_batch_k4/5/6)
def private bind_batch_pipe(raw : VkCommandBuffer; fmt : int) {
    if (fmt == 0) {
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.batch_pipeline, VkPipelineBindPoint.COMPUTE)
    } elif (fmt == 1) {
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.batch_k4, VkPipelineBindPoint.COMPUTE)
    } elif (fmt == 2) {
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.batch_k5, VkPipelineBindPoint.COMPUTE)
    } else {
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.batch_k6, VkPipelineBindPoint.COMPUTE)
    }
}

def private submit_wait(var raw : VkCommandBuffer) {
    var submit = VkSubmitInfo()
    submit.commandBufferCount = 1u
    var fence = g_gpu.fence
    unsafe {
        submit.pCommandBuffers = addr(raw)
        vk_check(vkQueueSubmit(g_gpu.queue, 1u, addr(submit), fence), null)
        // SPIN the fence, never block: the kernels run tens of µs and a blocking
        // vkWaitForFences pays 1-2ms of OS wake-up latency PER SUBMIT (measured; the
        // llama.cpp spin-tail trick, ggml-vulkan.cpp:2316). One busy core while the GPU
        // runs beats losing the decode budget to sleeps.
        while (vkGetFenceStatus(dev_raw(), fence) != VkResult.SUCCESS) {}
        vk_check(vkResetFences(dev_raw(), 1u, addr(fence)), null)
    }
}

// refresh a stack's mapped GEMV meta from region triples — one entry per ROW (a cnt > 1 region
// expands, entry index = its row, so kernel output lands row-major): [n, d, nrows, 0,
// (rel_wblock, xblock)*]. Block units follow the stack's format — 32-weight blocks for q8,
// 256-weight superblocks for kq; xnb = activation blocks per image row, in the same units
def private fill_stack_meta_rows(si : int; offs : int64 const?; nregions : int64; n, d, nrows, xnb : int64) {
    assert((4l + nrows * 2l) * 4l <= META_BYTES, "dasLLAMA vulkan tier: FFN meta exceeds capacity")
    let unit = g_gpu.stacks[si].fmt != 0 ? 256l : 32l
    unsafe {
        var mp = reinterpret<uint?>(g_gpu.stacks[si].meta.mapped)
        mp[0] = uint(n)
        mp[1] = uint(d)
        mp[2] = uint(nrows)
        mp[3] = 0u
        for (r in range64(nregions)) {
            let rel = offs[r * 3l] - g_gpu.stacks[si].base
            let row0 = offs[r * 3l + 1l]
            let cnt = offs[r * 3l + 2l]
            assert(rel % unit == 0l, "dasLLAMA vulkan tier: unaligned region offset")
            for (i in range64(cnt)) {
                mp[4l + (row0 + i) * 2l] = uint(rel / unit)
                mp[5l + (row0 + i) * 2l] = uint((row0 + i) * xnb)
            }
        }
    }
}

def private fill_stack_acts(si : int; xqp : int8 const?; xsp : float const?; xq_bytes : int64) {
    assert(xq_bytes <= XQ_BYTES, "dasLLAMA vulkan tier: dispatch exceeds scratch capacity")
    let unit = g_gpu.stacks[si].fmt != 0 ? 256l : 32l
    unsafe {
        memcpy(g_gpu.stacks[si].xq.mapped, reinterpret<void? -const>(xqp), int(xq_bytes))
        memcpy(g_gpu.stacks[si].xs.mapped, reinterpret<void? -const>(xsp), int((xq_bytes / unit) * 4l))
    }
}

// ===== the FFN mid-chain state (shared by the decode-GEMV and batch arms) =====

def private ensure_ffn_state {
    if (g_gpu.ffn_ready) {
        return
    }
    g_gpu.hq_dev = make_device_buf(BATCH_HQ_BYTES)
    g_gpu.hs_dev = make_device_buf(BATCH_HS_BYTES)
    g_gpu.ffn_meta = make_host_buf(FFN_META_BYTES, false)   // transfer_src staging
    g_gpu.ffn_meta_dev = make_device_buf(FFN_META_BYTES)
    g_gpu.ffn_ready = true
}

def private fill_ffn_meta(elems, nblocks : int64; is_gelu : bool) {
    unsafe {
        var mp = reinterpret<uint?>(g_gpu.ffn_meta.mapped)
        mp[0] = uint(elems)
        mp[1] = is_gelu ? 1u : 0u
        mp[2] = uint(nblocks)
        mp[3] = 0u
    }
}

// compute -> compute barrier between the chain's kernels (RAW on the y/hidden planes; the
// write-after-read hazard between requant and the down overwrite of y rides the same
// execution dependency)
def private comp_barrier(raw : VkCommandBuffer) {
    var mb : MemoryBarrier
    mb.srcAccessMask.shader_write = true
    mb.dstAccessMask.shader_read = true
    mb.dstAccessMask.shader_write = true
    var mbs : array<MemoryBarrier>
    mbs |> emplace(mb)
    var sstage : VkPipelineStageFlags
    sstage.compute_shader = true
    var dstage : VkPipelineStageFlags
    dstage.compute_shader = true
    let dflags : VkDependencyFlags
    let no_bufb : array<BufferMemoryBarrier>
    let no_imgb : array<ImageMemoryBarrier>
    cmd_pipeline_barrier(vk_value_to_boost(raw), sstage, dstage, dflags, mbs, no_bufb, no_imgb)
}

// ===== prefill batch dispatch (the FFN's batched-GEMM arm) =====

def private ensure_batch_state {
    if (g_gpu.batch_ready) {
        return
    }
    ensure_ffn_state()
    g_gpu.batch_xq_dev = make_device_buf(BATCH_XQ_BYTES)
    g_gpu.batch_xs_dev = make_device_buf(BATCH_XS_BYTES)
    g_gpu.batch_y1_dev = make_device_buf(BATCH_Y_BYTES)
    g_gpu.batch_y2_dev = make_device_buf(BATCH_Y_BYTES)
    g_gpu.batch_y1 = make_host_buf(BATCH_Y_BYTES, true, [cached = true])
    g_gpu.batch_y2 = make_host_buf(BATCH_Y_BYTES, true, [cached = true])
    g_gpu.batch_cmd = alloc_cmd()
    g_gpu.batch_actrq_set = alloc_desc_set()
    var writes : array<WriteDescriptorSet>
    // fused act+requant: gate floats at 5, up floats at 0 (vk_upf) -> hq/hs planes at 3/4;
    // params at 2; 1 unused filler
    write_buf_desc(writes, g_gpu.batch_actrq_set, 0u, g_gpu.batch_y2_dev, BATCH_Y_BYTES)
    write_buf_desc(writes, g_gpu.batch_actrq_set, 1u, g_gpu.batch_xs_dev, BATCH_XS_BYTES)
    write_buf_desc(writes, g_gpu.batch_actrq_set, 2u, g_gpu.ffn_meta_dev, FFN_META_BYTES)
    write_buf_desc(writes, g_gpu.batch_actrq_set, 3u, g_gpu.hq_dev, BATCH_HQ_BYTES)
    write_buf_desc(writes, g_gpu.batch_actrq_set, 4u, g_gpu.hs_dev, BATCH_HS_BYTES)
    write_buf_desc(writes, g_gpu.batch_actrq_set, 5u, g_gpu.batch_y1_dev, BATCH_Y_BYTES)
    let no_copies : array<CopyDescriptorSet>
    update_descriptor_sets(g_gpu.device, writes, no_copies)
    g_gpu.resident_bytes += BATCH_XQ_BYTES + BATCH_XS_BYTES
    g_gpu.batch_ready = true
    to_log(LOG_INFO, "dasLLAMA vulkan tier: prefill batch tier engaged\n")
}

// batch bindings differ from decode's: meta is the stack's BATCH table, activations point at a
// role-fixed image (gate/up: the gathered x scratch; down: the requantized hidden planes), y one
// of the two shared device planes (gate/down write y1_dev, up y2_dev)
def private ensure_batch_set(si : int; ybuf, xqbuf, xsbuf : uint64; xqbytes, xsbytes : int64) {
    if (g_gpu.stacks[si].batch_made && g_gpu.stacks[si].batch_y == ybuf && g_gpu.stacks[si].batch_xq == xqbuf) {
        return
    }
    if (!g_gpu.stacks[si].batch_made) {
        g_gpu.stacks[si].batch_meta = make_host_buf(BATCH_META_BYTES, false)   // transfer_src staging
        g_gpu.stacks[si].batch_meta_dev = make_device_buf(BATCH_META_BYTES)
        g_gpu.stacks[si].batch_set = alloc_desc_set()
        g_gpu.stacks[si].batch_made = true
    }
    var writes : array<WriteDescriptorSet>
    write_buf_desc(writes, g_gpu.stacks[si].batch_set, 0u, g_gpu.stacks[si].wqbuf, g_gpu.stacks[si].wqbytes)
    write_buf_desc(writes, g_gpu.stacks[si].batch_set, 1u, g_gpu.stacks[si].wsbuf, g_gpu.stacks[si].wsbytes)
    write_buf_desc(writes, g_gpu.stacks[si].batch_set, 2u, g_gpu.stacks[si].batch_meta_dev, BATCH_META_BYTES)
    write_buf_desc(writes, g_gpu.stacks[si].batch_set, 3u, xqbuf, xqbytes)
    write_buf_desc(writes, g_gpu.stacks[si].batch_set, 4u, xsbuf, xsbytes)
    write_buf_desc(writes, g_gpu.stacks[si].batch_set, 5u, ybuf, BATCH_Y_BYTES)
    let no_copies : array<CopyDescriptorSet>
    update_descriptor_sets(g_gpu.device, writes, no_copies)
    g_gpu.stacks[si].batch_y = ybuf
    g_gpu.stacks[si].batch_xq = xqbuf
}

// chunk row cap: the gathered x window, both hidden planes, the requant planes, and the down
// output (which reuses y1) must all fit their buffers. xunit = the activation scale block size
// (32 for Q8_0 chains, 256 for Q8_K)
def private ffn_chunk_rows(n, nfe, xunit : int64) : int64 {
    var r = min(BATCH_XQ_BYTES / n, BATCH_XS_BYTES / ((n / xunit) * 4l))
    r = min(r, BATCH_Y_BYTES / (max(nfe, n) * 4l))
    r = min(r, min(BATCH_HQ_BYTES / nfe, BATCH_HS_BYTES / ((nfe / xunit) * 4l)))
    return min(r, BATCH_CHUNK_ROWS)
}

// parallel host memcpy — the batch chain moves 8-34MB per hop and a single lane runs ~14GB/s
// while the box's DRAM feeds 3-4x that; jobque lanes are idle while the GPU owns the layer
let private PAR_COPY_CHUNK = 2_097_152l

def private par_memcpy(dst : void?; src : void?; bytes : int64) {
    if (bytes < PAR_COPY_CHUNK * 2l || !is_job_que_available()) {
        unsafe {
            memcpy(dst, src, int(bytes))
        }
        return
    }
    let nch = int((bytes + PAR_COPY_CHUNK - 1l) / PAR_COPY_CHUNK)
    var d8 = unsafe(reinterpret<uint8?>(dst))
    var s8 = unsafe(reinterpret<uint8?>(src))
    maybe_parallel_for(0, nch, nch) $(cb, ce) {
        unsafe {
            for (c in range(cb, ce)) {
                let off = int64(c) * PAR_COPY_CHUNK
                memcpy(reinterpret<void?>(d8 + off), reinterpret<void?>(s8 + off), int(min(PAR_COPY_CHUNK, bytes - off)))
            }
        }
    }
}

// stage the chunk's activation rows [r0, r0+rows) into the staging window (xq at 0, xs at
// BATCH_XQ_BYTES); the staging->device copies ride the HEAD of the chain submit — a separate
// synchronous copy submit here measured a full extra round trip per chunk
def private upload_batch_acts(xqp : int8 const?; xsp : float const?; r0, rows, n, xunit : int64) {
    let xq_bytes = rows * n
    let xs_bytes = rows * (n / xunit) * 4l
    unsafe {
        var sp = reinterpret<uint8?>(g_gpu.staging.mapped)
        par_memcpy(g_gpu.staging.mapped, reinterpret<void? -const>(xqp + r0 * n), xq_bytes)
        par_memcpy(reinterpret<void?>(sp + BATCH_XQ_BYTES), reinterpret<void? -const>(xsp + r0 * (n / xunit)), xs_bytes)
    }
}

// the chain-head acts copies the fold above defers to record time
def private cmd_copy_batch_acts(raw : VkCommandBuffer; xq_bytes, xs_bytes : int64) {
    var qregions : array<BufferCopy>
    qregions |> push(BufferCopy(srcOffset = 0ul, dstOffset = 0ul, size = uint64(xq_bytes)))
    cmd_copy_buffer(vk_value_to_boost(raw), nonowning_buf(g_gpu.staging.buf), nonowning_buf(g_gpu.batch_xq_dev), qregions)
    var sregions : array<BufferCopy>
    sregions |> push(BufferCopy(srcOffset = uint64(BATCH_XQ_BYTES), dstOffset = 0ul, size = uint64(xs_bytes)))
    cmd_copy_buffer(vk_value_to_boost(raw), nonowning_buf(g_gpu.staging.buf), nonowning_buf(g_gpu.batch_xs_dev), sregions)
}

// build the chunk's meta table: regions clipped to the row window [r0, r1) (row0 re-based to the
// window), then the per-workgroup region map. Weight offsets follow the stack's format units —
// 32-weight blocks for q8, superblocks for kq. Returns the workgroup count.
def private fill_batch_meta(si : int; offs : int64 const?; nregions : int64; n, d : int64; r0, r1 : int64) : uint {
    let wtiles = (d + BATCH_TILE - 1l) / BATCH_TILE
    let unit = g_gpu.stacks[si].fmt != 0 ? 256l : 32l
    unsafe {
        var mp = reinterpret<uint?>(g_gpu.stacks[si].batch_meta.mapped)
        var nrc = 0l
        for (r in range64(nregions)) {
            let b0 = max(offs[r * 3l + 1l], r0)
            let b1 = min(offs[r * 3l + 1l] + offs[r * 3l + 2l], r1)
            if (b1 > b0) {
                nrc++
            }
        }
        let wg_bound = ((r1 - r0 + BATCH_TILE - 1l) / BATCH_TILE + nrc) * wtiles
        assert((4l + nrc * 4l + wg_bound) * 4l <= BATCH_META_BYTES, "dasLLAMA vulkan tier: batch meta exceeds capacity")
        let map_off = 4l + nrc * 4l
        mp[0] = uint(n)
        mp[1] = uint(d)
        mp[2] = uint(nrc)
        mp[3] = uint(map_off)
        var ri = 0l
        var wg = 0l
        for (r in range64(nregions)) {
            let rb0 = max(offs[r * 3l + 1l], r0)
            let rb1 = min(offs[r * 3l + 1l] + offs[r * 3l + 2l], r1)
            if (rb1 <= rb0) {
                continue
            }
            let rel = offs[r * 3l] - g_gpu.stacks[si].base
            assert(rel % unit == 0l, "dasLLAMA vulkan tier: unaligned batch region weight offset")
            let mb = 4l + ri * 4l
            mp[mb] = uint(rel / unit)
            mp[mb + 1l] = uint(rb0 - r0)
            mp[mb + 2l] = uint(rb1 - rb0)
            mp[mb + 3l] = uint(wg)
            let rwg = ((rb1 - rb0 + BATCH_TILE - 1l) / BATCH_TILE) * wtiles
            for (g in range64(rwg)) {
                mp[map_off + wg + g] = uint(ri)
            }
            wg += rwg
            ri++
        }
        return uint(wg)
    }
}

def private cmd_copy_batch_meta(raw : VkCommandBuffer; si : int) {
    var regions : array<BufferCopy>
    regions |> push(BufferCopy(srcOffset = 0ul, dstOffset = 0ul, size = uint64(BATCH_META_BYTES)))
    cmd_copy_buffer(vk_value_to_boost(raw), nonowning_buf(g_gpu.stacks[si].batch_meta.buf),
                    nonowning_buf(g_gpu.stacks[si].batch_meta_dev), regions)
}

def private batch_barrier(raw : VkCommandBuffer; src_transfer : bool) {
    var mb : MemoryBarrier
    mb.srcAccessMask.transfer_write = src_transfer
    mb.srcAccessMask.shader_write = !src_transfer
    mb.dstAccessMask.shader_read = src_transfer
    mb.dstAccessMask.transfer_read = !src_transfer
    var mbs : array<MemoryBarrier>
    mbs |> emplace(mb)
    var sstage : VkPipelineStageFlags
    sstage.transfer = src_transfer
    sstage.compute_shader = !src_transfer
    var dstage : VkPipelineStageFlags
    dstage.compute_shader = src_transfer
    dstage.transfer = !src_transfer
    let dflags : VkDependencyFlags
    let no_bufb : array<BufferMemoryBarrier>
    let no_imgb : array<ImageMemoryBarrier>
    cmd_pipeline_barrier(vk_value_to_boost(raw), sstage, dstage, dflags, mbs, no_bufb, no_imgb)
}

def private cmd_copy_whole(raw : VkCommandBuffer; src, dst : uint64; bytes : int64) {
    var regions : array<BufferCopy>
    regions |> push(BufferCopy(srcOffset = 0ul, dstOffset = 0ul, size = uint64(bytes)))
    cmd_copy_buffer(vk_value_to_boost(raw), nonowning_buf(src), nonowning_buf(dst), regions)
}

// one reusable cmd buffer, re-recorded per submit — prefill submits are few and multi-ms, so
// recording cost is noise (unlike decode's per-token pre-recorded buffers). The chain per chunk:
// meta staging copies, gate+up GEMMs, fused act+requant, down GEMM, gout DMA — one barrier per
// seam, the hidden planes never leave the device.
def private record_ffn_batch_cmd(s1, s3, s2 : int; nwg_gu, nwg_dn : uint; rows, nfe, gout_bytes : int64; axq_bytes, axs_bytes : int64) {
    let raw = g_gpu.batch_cmd
    let rf : VkCommandBufferResetFlags
    vk_check(vkResetCommandBuffer(raw, rf), null)
    let begin = VkCommandBufferBeginInfo()
    vk_check(vkBeginCommandBuffer(raw, begin), null)
    cmd_copy_batch_acts(raw, axq_bytes, axs_bytes)
    cmd_copy_batch_meta(raw, s1)
    cmd_copy_batch_meta(raw, s3)
    cmd_copy_batch_meta(raw, s2)
    cmd_copy_whole(raw, g_gpu.ffn_meta.buf, g_gpu.ffn_meta_dev, FFN_META_BYTES)
    batch_barrier(raw, true)
    bind_batch_pipe(raw, g_gpu.stacks[s1].fmt)
    cmd_bind_dispatch(raw, g_gpu.stacks[s1].batch_set, nwg_gu)
    if (g_gpu.stacks[s3].fmt != g_gpu.stacks[s1].fmt) {
        bind_batch_pipe(raw, g_gpu.stacks[s3].fmt)
    }
    cmd_bind_dispatch(raw, g_gpu.stacks[s3].batch_set, nwg_gu)
    comp_barrier(raw)
    // the requant form follows the DOWN stack — its GEMM is the consumer of hq/hs
    let dkq = g_gpu.stacks[s2].fmt != 0
    cmd_bind_pipeline(vk_value_to_boost(raw), dkq ? g_gpu.actrq_k_pipeline : g_gpu.actrq_pipeline, VkPipelineBindPoint.COMPUTE)
    cmd_bind_dispatch(raw, g_gpu.batch_actrq_set, uint((rows * nfe / (dkq ? 256l : 32l) + 255l) / 256l))
    comp_barrier(raw)
    bind_batch_pipe(raw, g_gpu.stacks[s2].fmt)
    cmd_bind_dispatch(raw, g_gpu.stacks[s2].batch_set, nwg_dn)
    batch_barrier(raw, false)
    cmd_copy_whole(raw, g_gpu.batch_y1_dev, g_gpu.batch_y1.buf, gout_bytes)
    vk_check(vkEndCommandBuffer(raw), null)
}

// the batch arm: chunked over row windows; gate+up share the gathered image and region row
// ranges, the down GEMM reads the requantized hidden planes and reuses y1 for its output
def private vk_moe_ffn_batch(var goutp : float?; offs1 : int64 const?; offs3 : int64 const?; offs2 : int64 const?; nregions : int64; xqp : int8 const?; xsp : float const?; n, nfe, nrows : int64; f1, f3, f2 : int; is_gelu : bool) {
    let s1 = find_stack(unsafe(offs1[0]), f1)
    let s3 = find_stack(unsafe(offs3[0]), f3)
    let s2 = find_stack(unsafe(offs2[0]), f2)
    // the chain's activation form is uniform per layer (Q8_0 or Q8_K) — the upload walk only
    // admits all-q8 or all-kq stack triples, so kq-ness must agree across the three stacks
    let kq = f1 != 0
    assert((f3 != 0) == kq && (f2 != 0) == kq,
        "dasLLAMA vulkan tier: q8/kq mixed stack triple in one FFN chain")
    let xunit = kq ? 256l : 32l
    ensure_batch_state()
    ensure_batch_set(s1, g_gpu.batch_y1_dev, g_gpu.batch_xq_dev, g_gpu.batch_xs_dev, BATCH_XQ_BYTES, BATCH_XS_BYTES)
    ensure_batch_set(s3, g_gpu.batch_y2_dev, g_gpu.batch_xq_dev, g_gpu.batch_xs_dev, BATCH_XQ_BYTES, BATCH_XS_BYTES)
    ensure_batch_set(s2, g_gpu.batch_y1_dev, g_gpu.hq_dev, g_gpu.hs_dev, BATCH_HQ_BYTES, BATCH_HS_BYTES)
    let chunk = ffn_chunk_rows(n, nfe, xunit)
    var r0 = 0l
    while (r0 < nrows) {
        let r1 = min(r0 + chunk, nrows)
        let rows = r1 - r0
        let ts0 = ref_time_ticks()
        let nwg1 = fill_batch_meta(s1, offs1, nregions, n, nfe, r0, r1)
        let nwg3 = fill_batch_meta(s3, offs3, nregions, n, nfe, r0, r1)
        assert(nwg1 == nwg3, "dasLLAMA vulkan tier: fused batch pair region mismatch")
        let nwg2 = fill_batch_meta(s2, offs2, nregions, nfe, n, r0, r1)
        let ts1 = ref_time_ticks()
        upload_batch_acts(xqp, xsp, r0, rows, n, xunit)
        let ts2 = ref_time_ticks()
        fill_ffn_meta(rows * nfe, rows * nfe / xunit, is_gelu)
        record_ffn_batch_cmd(s1, s3, s2, nwg1, nwg2, rows, nfe, rows * n * 4l, rows * n, rows * (n / xunit) * 4l)
        let ts3 = ref_time_ticks()
        submit_wait(g_gpu.batch_cmd)
        let ts4 = ref_time_ticks()
        unsafe {
            par_memcpy(reinterpret<void?>(goutp + r0 * n), g_gpu.batch_y1.mapped, rows * n * 4l)
        }
        if (vk_prof()) {
            to_log(LOG_INFO, "vk_ffn_batch acts {get_time_usec(ts1) - get_time_usec(ts2)} submit {get_time_usec(ts3) - get_time_usec(ts4)} copy {get_time_usec(ts4)}\n")
        }
        r0 = r1
    }
}

// one dense chunk: acts + meta staging copies, one tile GEMM over the single whole-plane
// region, y DMA
def private record_dense_cmd(si : int; nwg : uint; y_bytes : int64; axq_bytes, axs_bytes : int64) {
    let raw = g_gpu.batch_cmd
    let rf : VkCommandBufferResetFlags
    vk_check(vkResetCommandBuffer(raw, rf), null)
    let begin = VkCommandBufferBeginInfo()
    vk_check(vkBeginCommandBuffer(raw, begin), null)
    cmd_copy_batch_acts(raw, axq_bytes, axs_bytes)
    cmd_copy_batch_meta(raw, si)
    batch_barrier(raw, true)
    bind_batch_pipe(raw, g_gpu.stacks[si].fmt)
    cmd_bind_dispatch(raw, g_gpu.stacks[si].batch_set, nwg)
    batch_barrier(raw, false)
    cmd_copy_whole(raw, g_gpu.batch_y1_dev, g_gpu.batch_y1.buf, y_bytes)
    vk_check(vkEndCommandBuffer(raw), null)
}

//! MoeGpuDenseFn: dense batched GEMM — y [nrows x d] = X · Wᵀ over the resident plane
//! ([d x n] at element offset woff), a one-region ride on the batch tile rail. Chunked like
//! the FFN arm; activations arrive pre-quantized in the plane's own form.
def vk_moe_dense(var yp : float?; woff : int64; xqp : int8 const?; xsp : float const?; n, d, nrows : int64; fmt : int) {
    let si = find_stack(woff, fmt)
    let xunit = fmt != 0 ? 256l : 32l
    ensure_batch_state()
    ensure_batch_set(si, g_gpu.batch_y1_dev, g_gpu.batch_xq_dev, g_gpu.batch_xs_dev, BATCH_XQ_BYTES, BATCH_XS_BYTES)
    var chunk = min(BATCH_XQ_BYTES / n, BATCH_XS_BYTES / ((n / xunit) * 4l))
    chunk = min(chunk, BATCH_Y_BYTES / (d * 4l))
    chunk = min(chunk, BATCH_CHUNK_ROWS)
    var region = fixed_array(woff, 0l, nrows)
    var r0 = 0l
    while (r0 < nrows) {
        let r1 = min(r0 + chunk, nrows)
        let rows = r1 - r0
        let ts0 = ref_time_ticks()
        var nwg = 0u
        unsafe {
            nwg = fill_batch_meta(si, addr(region[0]), 1l, n, d, r0, r1)
        }
        let ts1 = ref_time_ticks()
        upload_batch_acts(xqp, xsp, r0, rows, n, xunit)
        let ts2 = ref_time_ticks()
        record_dense_cmd(si, nwg, rows * d * 4l, rows * n, rows * (n / xunit) * 4l)
        let ts3 = ref_time_ticks()
        submit_wait(g_gpu.batch_cmd)
        let ts4 = ref_time_ticks()
        unsafe {
            par_memcpy(reinterpret<void?>(yp + r0 * d), g_gpu.batch_y1.mapped, rows * d * 4l)
        }
        if (vk_prof()) {
            to_log(LOG_INFO, "vk_dense acts {get_time_usec(ts1) - get_time_usec(ts2)} submit {get_time_usec(ts3) - get_time_usec(ts4)} copy {get_time_usec(ts4)}\n")
        }
        r0 = r1
    }
}

// ===== decode dispatch (the FFN's GEMV arm) =====

// pre-record (or refresh on a row-count change) the triple's chain: gate+up GEMV pair, fused
// act+requant, down GEMV, y DMA. Per token only the stacks' host-visible metas, the activation
// image, and the ffn params staging change — the recorded buffer resubmits as-is.
def private ffn_cmd_for(s1, s3, s2 : int; nrows, n, nfe : int64) : VkCommandBuffer {
    let key = (int64(s1) * int64(MAX_STACKS) + int64(s3)) * int64(MAX_STACKS) + int64(s2)
    if (!key_exists(g_gpu.ffn_cmds, key)) {
        var fc = FfnCmd(cmd = alloc_cmd(), actrq_set = alloc_desc_set(),
                        down_set = alloc_desc_set(), nrows = 0l)
        var writes : array<WriteDescriptorSet>
        // fused act+requant: gate floats at 5, up floats at 0 (vk_upf) -> hq/hs planes at 3/4;
        // params at 2; 1 unused filler
        write_buf_desc(writes, fc.actrq_set, 0u, g_gpu.stacks[s3].y_dev, Y_BYTES)
        write_buf_desc(writes, fc.actrq_set, 1u, g_gpu.stacks[s1].wsbuf, g_gpu.stacks[s1].wsbytes)
        write_buf_desc(writes, fc.actrq_set, 2u, g_gpu.ffn_meta_dev, FFN_META_BYTES)
        write_buf_desc(writes, fc.actrq_set, 3u, g_gpu.hq_dev, BATCH_HQ_BYTES)
        write_buf_desc(writes, fc.actrq_set, 4u, g_gpu.hs_dev, BATCH_HS_BYTES)
        write_buf_desc(writes, fc.actrq_set, 5u, g_gpu.stacks[s1].y_dev, Y_BYTES)
        // down: its own planes + host-visible meta, hq/hs as the activation image, its y_dev
        write_buf_desc(writes, fc.down_set, 0u, g_gpu.stacks[s2].wqbuf, g_gpu.stacks[s2].wqbytes)
        write_buf_desc(writes, fc.down_set, 1u, g_gpu.stacks[s2].wsbuf, g_gpu.stacks[s2].wsbytes)
        write_buf_desc(writes, fc.down_set, 2u, g_gpu.stacks[s2].meta.buf, META_BYTES)
        write_buf_desc(writes, fc.down_set, 3u, g_gpu.hq_dev, BATCH_HQ_BYTES)
        write_buf_desc(writes, fc.down_set, 4u, g_gpu.hs_dev, BATCH_HS_BYTES)
        write_buf_desc(writes, fc.down_set, 5u, g_gpu.stacks[s2].y_dev, Y_BYTES)
        let no_copies : array<CopyDescriptorSet>
        update_descriptor_sets(g_gpu.device, writes, no_copies)
        g_gpu.ffn_cmds |> insert(key, fc)
    }
    var fc = g_gpu.ffn_cmds[key]
    if (fc.nrows != nrows) {
        if (fc.nrows != 0l) {
            let rf : VkCommandBufferResetFlags
            vk_check(vkResetCommandBuffer(fc.cmd, rf), null)
        }
        let raw = fc.cmd
        let begin = VkCommandBufferBeginInfo()
        vk_check(vkBeginCommandBuffer(raw, begin), null)
        cmd_copy_whole(raw, g_gpu.ffn_meta.buf, g_gpu.ffn_meta_dev, FFN_META_BYTES)
        batch_barrier(raw, true)
        bind_gemv_pipe(raw, g_gpu.stacks[s1].fmt)
        let ggu = uint((nrows * nfe + g_gpu.rows_per_wg - 1l) / g_gpu.rows_per_wg)
        cmd_bind_dispatch(raw, g_gpu.stacks[s1].raw_set, ggu)
        if (g_gpu.stacks[s3].fmt != g_gpu.stacks[s1].fmt) {
            bind_gemv_pipe(raw, g_gpu.stacks[s3].fmt)
        }
        cmd_bind_dispatch(raw, g_gpu.stacks[s3].raw_set, ggu)
        comp_barrier(raw)
        // the requant form follows the DOWN stack — its GEMV is the consumer of hq/hs
        let dkq = g_gpu.stacks[s2].fmt != 0
        cmd_bind_pipeline(vk_value_to_boost(raw), dkq ? g_gpu.actrq_k_pipeline : g_gpu.actrq_pipeline, VkPipelineBindPoint.COMPUTE)
        cmd_bind_dispatch(raw, fc.actrq_set, uint((nrows * nfe / (dkq ? 256l : 32l) + 255l) / 256l))
        comp_barrier(raw)
        bind_gemv_pipe(raw, g_gpu.stacks[s2].fmt)
        cmd_bind_dispatch(raw, fc.down_set, uint((nrows * n + g_gpu.rows_per_wg - 1l) / g_gpu.rows_per_wg))
        batch_barrier(raw, false)
        cmd_copy_whole(raw, g_gpu.stacks[s2].y_dev, g_gpu.stacks[s2].y.buf, nrows * n * 4l)
        vk_check(vkEndCommandBuffer(raw), null)
        g_gpu.ffn_cmds[key] = FfnCmd(cmd = fc.cmd, actrq_set = fc.actrq_set,
            down_set = fc.down_set, nrows = nrows)
    }
    return fc.cmd
}

def private vk_moe_ffn_gemv(var goutp : float?; offs1 : int64 const?; offs3 : int64 const?; offs2 : int64 const?; nregions : int64; xqp : int8 const?; xsp : float const?; n, nfe, nrows : int64; f1, f3, f2 : int; is_gelu : bool) {
    let s1 = find_stack(unsafe(offs1[0]), f1)
    let s3 = find_stack(unsafe(offs3[0]), f3)
    let s2 = find_stack(unsafe(offs2[0]), f2)
    assert(nrows * max(nfe, n) * 4l <= Y_BYTES, "dasLLAMA vulkan tier: FFN dispatch exceeds output capacity")
    // the chain's activation form is uniform per layer (Q8_0 or Q8_K) — the upload walk only
    // admits all-q8 or all-kq stack triples, so kq-ness must agree across the three stacks
    let kq = f1 != 0
    assert((f3 != 0) == kq && (f2 != 0) == kq,
        "dasLLAMA vulkan tier: q8/kq mixed stack triple in one FFN chain")
    let xunit = kq ? 256l : 32l
    ensure_gemv_scratch(s1)
    ensure_gemv_scratch(s3)
    ensure_gemv_scratch(s2)
    fill_stack_meta_rows(s1, offs1, nregions, n, nfe, nrows, n / xunit)
    fill_stack_meta_rows(s3, offs3, nregions, n, nfe, nrows, n / xunit)
    fill_stack_meta_rows(s2, offs2, nregions, nfe, n, nrows, nfe / xunit)
    fill_stack_acts(s1, xqp, xsp, nrows * n)
    fill_stack_acts(s3, xqp, xsp, nrows * n)
    fill_ffn_meta(nrows * nfe, nrows * nfe / xunit, is_gelu)
    var cmd = ffn_cmd_for(s1, s3, s2, nrows, n, nfe)
    submit_wait(cmd)
    unsafe {
        memcpy(reinterpret<void?>(goutp), g_gpu.stacks[s2].y.mapped, int(nrows * n * 4l))
    }
}

// ===== streamed-prefill dispatch (pinned-mirror layers) =====

def private ensure_stream_slots {
    if (g_gpu.stream_ready) {
        return
    }
    ensure_batch_state()
    g_gpu.stream_cmd = alloc_cmd()
    g_gpu.stream_slots |> reserve(2)
    for (_slot in range(2)) {
        let s1 = make_stack_shell(-1l, 0l, 0, g_gpu.stream_max_gu_wq, g_gpu.stream_max_gu_ws)
        let s3 = make_stack_shell(-1l, 0l, 0, g_gpu.stream_max_gu_wq, g_gpu.stream_max_gu_ws)
        let s2 = make_stack_shell(-1l, 0l, 0, g_gpu.stream_max_dn_wq, g_gpu.stream_max_dn_ws)
        g_gpu.stream_slots |> push(StreamSlot(group = -1, s1 = s1, s3 = s3, s2 = s2))
    }
    g_gpu.stream_ready = true
    let slot_bytes = 2l * (g_gpu.stream_max_gu_wq + g_gpu.stream_max_gu_ws) + g_gpu.stream_max_dn_wq + g_gpu.stream_max_dn_ws
    // the reserve was carved out of the weight budget at init — a lie here silently overcommits
    // past the WDDM paging cliff
    assert(2l * slot_bytes <= STREAM_RESERVE, "dasLLAMA vulkan tier: stream slots exceed STREAM_RESERVE")
    to_log(LOG_INFO, "dasLLAMA vulkan tier: streamed prefill engaged ({length(g_gpu.stream_stacks) / 3} layers, slot {slot_bytes} bytes x 2)\n")
}

// re-base one slot pseudo-stack onto a stream stack (descriptors stay — the slot buffers are
// the same; only the meta-resolution identity and the kernel choice follow the layer)
def private rebase_pseudo(si : int; ss : StreamStack const) {
    g_gpu.stacks[si].base = ss.base
    g_gpu.stacks[si].elems = ss.elems
    g_gpu.stacks[si].fmt = ss.fmt
}

// DMA group g's six planes from the pinned mirrors into slot si. sync submits fenced and
// waits; async submits UNFENCED — the prefetch rides the GPU-idle window while the CPU runs
// the next layers' attention, and the next chain submit's transfer->compute barrier orders it
// (in-order queue). One shared cmd buffer is safe: a fenced chain wait always intervenes
// between two copy recordings, so the prior copy has retired off the queue.
def private copy_stream_group(si, g : int; sync : bool) {
    let slot = g_gpu.stream_slots[si]
    var raw = g_gpu.stream_cmd
    let rf : VkCommandBufferResetFlags
    vk_check(vkResetCommandBuffer(raw, rf), null)
    let begin = VkCommandBufferBeginInfo()
    vk_check(vkBeginCommandBuffer(raw, begin), null)
    for (k in range(3)) {
        let sk = g * 3 + k
        let psi = k == 0 ? slot.s1 : (k == 1 ? slot.s3 : slot.s2)
        cmd_copy_whole(raw, g_gpu.stream_stacks[sk].hwq.buf, g_gpu.stacks[psi].wqbuf, g_gpu.stream_stacks[sk].wqbytes)
        cmd_copy_whole(raw, g_gpu.stream_stacks[sk].hws.buf, g_gpu.stacks[psi].wsbuf, g_gpu.stream_stacks[sk].wsbytes)
    }
    vk_check(vkEndCommandBuffer(raw), null)
    if (sync) {
        submit_wait(raw)
    } else {
        var submit = VkSubmitInfo()
        submit.commandBufferCount = 1u
        let nofence : VkFence
        unsafe {
            submit.pCommandBuffers = addr(raw)
            vk_check(vkQueueSubmit(g_gpu.queue, 1u, addr(submit), nofence), null)
        }
    }
    g_gpu.stream_slots[si].group = g
}

// run one streamed layer's prefill chain: ensure its planes sit in a slot (cold-copy on a
// prefetch miss), re-base the slot's pseudo-stacks so the normal batch rail resolves them,
// dispatch, then fire the next runtime layer's prefetch (group g-1 — the loader walk uploads
// descending, prefill runs ascending)
def private vk_moe_ffn_streamed(g : int; var goutp : float?; offs1 : int64 const?; offs3 : int64 const?; offs2 : int64 const?; nregions : int64; xqp : int8 const?; xsp : float const?; n, nfe, nrows : int64; f1, f3, f2 : int; is_gelu : bool) {
    assert(nrows >= BATCH_TILE, "dasLLAMA vulkan tier: streamed layers serve prefill batches only")
    assert(g_gpu.stream_stacks[g * 3 + 1].fmt == f3 && g_gpu.stream_stacks[g * 3 + 2].fmt == f2,
        "dasLLAMA vulkan tier: streamed group format mismatch")
    ensure_stream_slots()
    var si = -1
    for (i in range(2)) {
        if (g_gpu.stream_slots[i].group == g) {
            si = i
        }
    }
    if (si < 0) {
        si = g % 2
        copy_stream_group(si, g, true)
    }
    rebase_pseudo(g_gpu.stream_slots[si].s1, g_gpu.stream_stacks[g * 3])
    rebase_pseudo(g_gpu.stream_slots[si].s3, g_gpu.stream_stacks[g * 3 + 1])
    rebase_pseudo(g_gpu.stream_slots[si].s2, g_gpu.stream_stacks[g * 3 + 2])
    vk_moe_ffn_batch(goutp, offs1, offs3, offs2, nregions, xqp, xsp, n, nfe, nrows, f1, f3, f2, is_gelu)
    if (g > 0 && g_gpu.stream_slots[(g - 1) % 2].group != g - 1) {
        copy_stream_group((g - 1) % 2, g - 1, false)
    }
}

//! MoeGpuFfnFn: the routed experts' whole FFN chain — gate+up region GEMMs, activation,
//! requant, down GEMM — as device-side submit chains over the resident stacks; the hidden
//! activations never cross PCIe. offs are region triples sharing row ranges that partition
//! [0, nrows); gout returns nrows x n floats. Decode row counts run the GEMV kernels
//! (pre-recorded per stack triple), batches the tiled-GEMM chunk loop — q8 and native kq
//! stacks alike, each on its format's kernels. Streamed layers (pinned mirrors) route through
//! the slot rail; anything else panics (fail-closed — the env gate is the opt-in).
def vk_moe_ffn(var goutp : float?; offs1 : int64 const?; offs3 : int64 const?; offs2 : int64 const?; nregions : int64; xqp : int8 const?; xsp : float const?; n, nfe, nrows : int64; f1, f3, f2 : int; is_gelu : bool) {
    ensure_ffn_state()
    if (find_stack_idx(unsafe(offs1[0]), f1) < 0) {
        let w0 = unsafe(offs1[0])
        var g = -1
        for (i in range(length(g_gpu.stream_stacks) / 3)) {
            if (g_gpu.stream_stacks[i * 3].fmt == f1 && w0 >= g_gpu.stream_stacks[i * 3].base
                    && w0 < g_gpu.stream_stacks[i * 3].base + g_gpu.stream_stacks[i * 3].elems) {
                g = i
            }
        }
        if (g < 0) {
            panic("dasLLAMA vulkan tier: dispatch against a region that is neither resident nor streamed (fmt {f1} element offset {w0})")
        }
        vk_moe_ffn_streamed(g, goutp, offs1, offs3, offs2, nregions, xqp, xsp, n, nfe, nrows, f1, f3, f2, is_gelu)
        return
    }
    if (nrows < BATCH_TILE) {
        vk_moe_ffn_gemv(goutp, offs1, offs3, offs2, nregions, xqp, xsp, n, nfe, nrows, f1, f3, f2, is_gelu)
    } else {
        vk_moe_ffn_batch(goutp, offs1, offs3, offs2, nregions, xqp, xsp, n, nfe, nrows, f1, f3, f2, is_gelu)
    }
}

// ===== classifier dispatch (the head GEMV) =====

// One-time dispatch state for a resident classifier plane: the meta never changes (ONE region
// covering the whole plane, activations at image row 0), so it uploads device-local once and
// the command buffer records once — per token only the activation refresh and the submit remain.
def private ensure_cls(si : int; woff, n, d : int64) {
    if (g_gpu.stacks[si].cls_made) {
        return
    }
    ensure_gemv_scratch(si)
    let unit = g_gpu.stacks[si].fmt != 0 ? 256l : 32l
    let rel = woff - g_gpu.stacks[si].base
    assert(rel % unit == 0l, "dasLLAMA vulkan tier: unaligned classifier plane offset")
    unsafe {
        var mp = reinterpret<uint?>(g_gpu.staging.mapped)
        mp[0] = uint(n)
        mp[1] = uint(d)
        mp[2] = 1u
        mp[3] = 0u
        mp[4] = uint(rel / unit)
        mp[5] = 0u
    }
    g_gpu.stacks[si].cls_meta_dev = make_device_buf(CLS_META_BYTES)
    staged_upload(g_gpu.stacks[si].cls_meta_dev, 0l, CLS_META_BYTES)
    g_gpu.stacks[si].cls_set = alloc_desc_set()
    var writes : array<WriteDescriptorSet>
    write_buf_desc(writes, g_gpu.stacks[si].cls_set, 0u, g_gpu.stacks[si].wqbuf, g_gpu.stacks[si].wqbytes)
    write_buf_desc(writes, g_gpu.stacks[si].cls_set, 1u, g_gpu.stacks[si].wsbuf, g_gpu.stacks[si].wsbytes)
    write_buf_desc(writes, g_gpu.stacks[si].cls_set, 2u, g_gpu.stacks[si].cls_meta_dev, CLS_META_BYTES)
    write_buf_desc(writes, g_gpu.stacks[si].cls_set, 3u, g_gpu.stacks[si].xq.buf, XQ_BYTES)
    write_buf_desc(writes, g_gpu.stacks[si].cls_set, 4u, g_gpu.stacks[si].xs.buf, XS_BYTES)
    write_buf_desc(writes, g_gpu.stacks[si].cls_set, 5u, g_gpu.stacks[si].y_dev, Y_BYTES)
    let no_copies : array<CopyDescriptorSet>
    update_descriptor_sets(g_gpu.device, writes, no_copies)
    var raw = alloc_cmd()
    let begin = VkCommandBufferBeginInfo()
    vk_check(vkBeginCommandBuffer(raw, begin), null)
    bind_gemv_pipe(raw, g_gpu.stacks[si].fmt)
    cmd_bind_dispatch(raw, g_gpu.stacks[si].cls_set, uint((d + g_gpu.rows_per_wg - 1l) / g_gpu.rows_per_wg))
    batch_barrier(raw, false)
    cmd_copy_whole(raw, g_gpu.stacks[si].y_dev, g_gpu.stacks[si].y.buf, d * 4l)
    vk_check(vkEndCommandBuffer(raw), null)
    g_gpu.stacks[si].cls_cmd = raw
    g_gpu.stacks[si].cls_made = true
}

//! MoeGpuClsFn: the classifier GEMV — y[d] = W · x over the resident plane at woff, activation
//! in the plane's own quant form (Q8_0 per-32 / Q8_K per-256), result DMA'd to the host-cached
//! staging in the same submit.
def vk_moe_cls(var yp : float?; woff : int64; xqp : int8 const?; xsp : float const?; n, d : int64; fmt : int) {
    let si = find_stack(woff, fmt)
    assert(d * 4l <= Y_BYTES, "dasLLAMA vulkan tier: classifier vocab exceeds output capacity")
    ensure_cls(si, woff, n, d)
    fill_stack_acts(si, xqp, xsp, n)
    submit_wait(g_gpu.stacks[si].cls_cmd)
    unsafe {
        memcpy(reinterpret<void?>(yp), g_gpu.stacks[si].y.mapped, int(d * 4l))
    }
}

// ===== registration =====

var private g_vk_probed = false
var private g_vk_available = false

def private vulkan_backend_available : bool {
    if (g_vk_probed) {
        return g_vk_available
    }
    g_vk_probed = true
    // panic-free raw probe: the boost creators (create_instance / select_physical_device)
    // panic on failure, so drive the two raw calls directly and treat any error as absent
    if (volkInitialize() != 0) {
        return false
    }
    let ci = VkInstanceCreateInfo()
    var inst : VkInstance
    if (vkCreateInstance(ci, null, inst) != VkResult.SUCCESS) {
        return false
    }
    volkLoadInstance(inst)
    var count = 0u
    vkEnumeratePhysicalDevices(inst, count, null)
    vkDestroyInstance(inst, null)
    g_vk_available = count > 0u
    return g_vk_available
}

[unused_argument(yp, wp, sp, xqp, xsp, n, d)]
def private vulkan_no_gemv(var yp : float?; wp : int8 const?; sp : float const?; xqp : int8 const?; xsp : float const?; n, d : int64) {
    panic("dasLLAMA vulkan backend has no GEMV core — the MoE tier (DASLLAMA_GPU_MOE_LAYERS) is its only dispatch")
}

[unused_argument(yp, wp, sp, xqp, xsp, n, d, ntok)]
def private vulkan_no_batch(var yp : float?; wp : int8 const?; sp : float const?; xqp : int8 const?; xsp : float const?; n, d, ntok : int64) {
    panic("dasLLAMA vulkan backend has no batch core — the MoE tier (DASLLAMA_GPU_MOE_LAYERS) is its only dispatch")
}

[unused_argument(y0p, y1p, y2p, wp, sp, woff0, woff1, woff2, xqp, xsp, n, d0, d1, d2)]
def private vulkan_no_group3(var y0p : float?; var y1p : float?; var y2p : float?; wp : int8 const?; sp : float const?; woff0, woff1, woff2 : int64; xqp : int8 const?; xsp : float const?; n, d0, d1, d2 : int64) {
    panic("dasLLAMA vulkan backend has no group3 core — the MoE tier (DASLLAMA_GPU_MOE_LAYERS) is its only dispatch")
}

[unused_argument(wp, sp, n, d)]
def private vulkan_repack_noop(var wp : int8?; var sp : float?; n, d : int64) {}

[init]
def dasllama_math_vulkan_register() {
    register_kernel_backend(KernelBackend(name = "vulkan", mm = @@vulkan_no_gemv, batch = @@vulkan_no_batch,
        group3 = @@vulkan_no_group3, repack = @@vulkan_repack_noop,
        needs_repack = false, priority = -1, available = @@vulkan_backend_available))
    let layers = int64(to_int(get_env_variable("DASLLAMA_GPU_MOE_LAYERS")))
    let nstream = int64(to_int(get_env_variable("DASLLAMA_GPU_MOE_STREAM")))
    if ((layers > 0l || nstream > 0l) && vulkan_backend_available()) {
        set_moe_gpu_layer_count(layers)
        install_moe_gpu_tier(@@vk_moe_ffn, @@vk_moe_upload_stack, @@vk_moe_cls, @@vk_moe_dense, @@vk_moe_drop_stacks)
    }
}
