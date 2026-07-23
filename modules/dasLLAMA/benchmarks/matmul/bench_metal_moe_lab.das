options gen2
options persistent_heap
options stack = 65536

require ?das_metal metal/das_metal_boost  // nolint:STYLE030 — GPU driver half, Apple builds only
require metal/msl_shader
require dasllama/dasllama_metal_prefill   // the production MoE GEMV MSL (zero copy drift)
require llvm/daslib/f16_cvt
require daslib/fio   // nolint:STYLE030 — used only inside the Apple static_if half
require math   // nolint:STYLE030 — used only inside the Apple static_if half
require strings

// The routed-expert GEMV lab — the 26B tg dig's barrier/dispatch attribution rig. The knockout
// chase convicted the routed k4/q51 expert GEMVs (5.48ms/step at ~163 wGB/s vs the ~300-340 the
// same k4 dot sustains in bench_metal_gemv_kernels), with the dense-shared branch NEAR ceiling in
// the same graph — so the suspect is the DISPATCH STRUCTURE, not the dot: the single-decode
// driver encodes on a SERIAL encoder (implicit barrier after every dispatch, ~330/step) while
// only the batch rail has the concurrent-encoder + explicit-hazard form.
//
// This rig re-creates the routed chain at the EXACT 26B shapes (E=128, nfe=704, dim=2816, k=8,
// 30 layers: w1+w3 k4, geglu ew, w2 q51) on a CONCURRENT encoder and prices each structure:
//   pipe      no barriers, per-layer disjoint buffers — the kernels' pipelined ceiling
//   pipe_c    pipe with CONTIGUOUS sel — expert-scatter locality control (vs pipe's scattered)
//   serial    barrier after EVERY dispatch — today's serial-encoder semantics
//   hazard    barriers only at real hazards (w1 ∥ w3, then ew, then w2) — the port candidate
//   pair13    the fused W1+W3 lab kernel (one dispatch streams both planes per slot) + hazard
//   true_ser  a genuine serial encoder, no explicit barriers — sanity: must match `serial`
// Weight stacks total ~476MB (>> 48MB SLC) and sel rotates per layer, so passes stay DRAM-honest.
// Correctness: exact-arithmetic fills (lab contract: pow2 scales, small ints) check the
// standalone w1 (k4) and w2 (q51) sites against CPU refs — the chain arms are timing-only
// (geglu is transcendental; the production kernels carry their own in-graph parity elsewhere).
//
//   bin/daslang -jit modules/dasLLAMA/benchmarks/matmul/bench_metal_moe_lab.das
//   DASMETAL_LAB_ROUNDS (default 3, best-of), DASMETAL_LAB_PASSES (default 4, per cb),
//   DASMETAL_LAB_ARMS (substring filter)

let E = 128l          // experts per logical stack (w1 planes 0..127, w3 planes 128..255)
let NFE = 704l        // expert FFN width (gemma4.expert_feed_forward_length)
let DIM = 2816l       // model dim
let K = 8l            // experts used per token
let LAYERS = 30l
let KNFE = K * NFE    // one h12 half, floats

let NSB_ROW = DIM / 256l           // k4 superblocks per w1/w3 row
let ESB = NFE * NSB_ROW            // k4 superblocks per expert plane
let NBLK_ROW = NFE / 32l           // q51 blocks per w2 row
let EBLK = DIM * NBLK_ROW          // q51 blocks per expert plane

// ===== the paired W1+W3 lab kernel =====
// MetalMoeGemvK4's dot with a SECOND weight stream: the w3 plane rides +E expert slots in the
// same stack, sharing the x loads (yl/yh) — one dispatch per layer instead of two, and double
// the weight ILP per x fetch. y = [gate: k x nfe][up: k x nfe], the production h12 layout.
class MoeGemvK4Pair {
    @ssbo @binding = 0 ksh : array<float16>
    @ssbo @binding = 1 ks4 : array<uint4>
    @ssbo @binding = 2 kqu : array<uint>
    @ssbo @binding = 3 x : array<float>
    @ssbo @binding = 4 y : array<float>
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 ddim : uint
    @ssbo @binding = 7 sel : array<uint>
    @uniform @binding = 8 esb : uint
    @uniform @binding = 9 eshift : uint     // w3 plane offset, expert slots (= E)
    @uniform @binding = 10 knfe : uint      // up-half y offset, elements

    [metal_kernel(name="lab_moe_k4_pair_msl")]
    def lab_moe_k4_pair {
        let lane = gl_SubgroupInvocationID
        let ix = lane / 8u
        let it = lane % 8u
        let iq = it / 4u
        let ir = it % 4u
        let nb = ndim / 256u
        let slot = gl_WorkGroupID.y
        let eb1 = sel[slot] * esb
        let eb3 = (sel[slot] + eshift) * esb
        let y0 = slot * ddim
        let first_row = (gl_WorkGroupID.x * 2u + gl_SubgroupID) * 2u
        var yl : float[16]
        var yh : float[16]
        var sumf : float[4]
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
            for [unroll_full] (p in range(4)) {   // p/2 = plane (w1/w3), p%2 = row of the pair
                let blk = (p < 2 ? eb1 : eb3) + (first_row + uint(p % 2)) * nb + ib
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
                sumf[p] += dd * ((acc1[0] + 0.00390625 * acc1[1]) * float(sc0 & 255u)
                    + (acc1[2] + 0.00390625 * acc1[3]) * float(sc0 >> 8u) * 0.0625
                    + (acc2[0] + 0.00390625 * acc2[1]) * float(sc2 & 255u)
                    + (acc2[2] + 0.00390625 * acc2[3]) * float(sc2 >> 8u) * 0.0625)
                sumf[p] -= dm * (sumy[0] * float(sc1 & 255u) + sumy[1] * float(sc1 >> 8u)
                    + sumy[2] * float(sc3 & 255u) + sumy[3] * float(sc3 >> 8u))
            }
            ib += 4u
        }
        for [unroll_full] (p in range(4)) {
            let s = simd_sum(sumf[p])
            if (lane == 0u && first_row + uint(p % 2) < ddim) {
                y[(p < 2 ? 0u : knfe) + y0 + first_row + uint(p % 2)] = s
            }
        }
    }
}

// x-vector variant: the checklist's "vector-width ssbo views" lever — the production kernel's 32
// scalar x loads per block become 8 float4 loads (same bytes, quarter the load issues).
class MoeGemvK4X4 {
    @ssbo @binding = 0 ksh : array<float16>
    @ssbo @binding = 1 ks4 : array<uint4>
    @ssbo @binding = 2 kqu : array<uint>
    @ssbo @binding = 3 x4 : array<float4>
    @ssbo @binding = 4 y : array<float>
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 ddim : uint
    @ssbo @binding = 7 sel : array<uint>
    @uniform @binding = 8 esb : uint
    @uniform @binding = 9 xs : uint

    [metal_kernel(name="lab_moe_k4_x4_msl")]
    def lab_moe_k4_x4 {
        let lane = gl_SubgroupInvocationID
        let ix = lane / 8u
        let it = lane % 8u
        let iq = it / 4u
        let ir = it % 4u
        let nb = ndim / 256u
        let slot = gl_WorkGroupID.y
        let eb = sel[slot] * esb
        let xoff = slot * xs
        let y0 = slot * ddim
        let first_row = (gl_WorkGroupID.x * 2u + gl_SubgroupID) * 2u
        var yl : float[16]
        var yh : float[16]
        var sumf : float[2]
        var ib = ix
        while (ib < nb) {
            let yb4 = (xoff + ib * 256u + iq * 64u + ir * 8u) / 4u
            var sumy : float[4]
            for [unroll_full] (h in range(4)) {   // h: yl lo/hi, yh lo/hi — float4 pairs at 0/8/32/40
                let xi = yb4 + (h < 2 ? uint(h) * 8u : 16u + uint(h) * 8u)
                let va = x4[xi]
                let vb = x4[xi + 1u]
                let o = (h % 2) * 8
                if (h < 2) {   // constant under unroll_full — folds per iteration
                    yl[o] = va.x
                    yl[o + 1] = va.y
                    yl[o + 2] = va.z
                    yl[o + 3] = va.w
                    yl[o + 4] = vb.x
                    yl[o + 5] = vb.y
                    yl[o + 6] = vb.z
                    yl[o + 7] = vb.w
                } else {
                    yh[o] = va.x
                    yh[o + 1] = va.y
                    yh[o + 2] = va.z
                    yh[o + 3] = va.w
                    yh[o + 4] = vb.x
                    yh[o + 5] = vb.y
                    yh[o + 6] = vb.z
                    yh[o + 7] = vb.w
                }
                sumy[h] = va.x + va.y + va.z + va.w + vb.x + vb.y + vb.z + vb.w
            }
            for [unroll_full] (r in range(2)) {
                let blk = eb + (first_row + uint(r)) * nb + ib
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
                y[y0 + first_row + uint(r)] = s
            }
        }
    }
}

// 4-rows-per-simdgroup variant: the checklist's "fewer long-running sgs" lever — one x window
// feeds 4 weight rows (half the x issues per weight byte of the production 2-row form).
class MoeGemvK4R4 {
    @ssbo @binding = 0 ksh : array<float16>
    @ssbo @binding = 1 ks4 : array<uint4>
    @ssbo @binding = 2 kqu : array<uint>
    @ssbo @binding = 3 x : array<float>
    @ssbo @binding = 4 y : array<float>
    @uniform @binding = 5 ndim : uint
    @uniform @binding = 6 ddim : uint
    @ssbo @binding = 7 sel : array<uint>
    @uniform @binding = 8 esb : uint
    @uniform @binding = 9 xs : uint

    [metal_kernel(name="lab_moe_k4_r4_msl")]
    def lab_moe_k4_r4 {
        let lane = gl_SubgroupInvocationID
        let ix = lane / 8u
        let it = lane % 8u
        let iq = it / 4u
        let ir = it % 4u
        let nb = ndim / 256u
        let slot = gl_WorkGroupID.y
        let eb = sel[slot] * esb
        let xoff = slot * xs
        let y0 = slot * ddim
        let first_row = (gl_WorkGroupID.x * 2u + gl_SubgroupID) * 4u
        var yl : float[16]
        var yh : float[16]
        var sumf : float[4]
        var ib = ix
        while (ib < nb) {
            let yb = xoff + ib * 256u + iq * 64u + ir * 8u
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
            for [unroll_full] (r in range(4)) {
                let blk = eb + (first_row + uint(r)) * nb + ib
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
        for [unroll_full] (r in range(4)) {
            let s = simd_sum(sumf[r])
            if (lane == 0u && first_row + uint(r) < ddim) {
                y[y0 + first_row + uint(r)] = s
            }
        }
    }
}

// ===== exact-arithmetic fills (bench_metal_gemv_kernels contract, gg stream dropped) =====

def private fill_k4_stack(var kq : array<uint8>; var kc : array<uint8>; rows, k, seed : int64) {
    let nsb = k / 256l
    let total = rows * nsb
    kq |> resize(total * 128l)
    kc |> resize(total * 16l)
    let lut = fixed_array(0.25, 0.5, 1.0, 2.0)
    for (sb in range64(total)) {
        let db = f32_to_f16(lut[int((sb / 5l) % 4l)])
        let mb = f32_to_f16(lut[int((sb / 3l + 1l) % 4l)])
        let co = sb * 16l
        kc[co] = uint8(db & 255u)
        kc[co + 1l] = uint8(db >> 8u)
        kc[co + 2l] = uint8(mb & 255u)
        kc[co + 3l] = uint8(mb >> 8u)
        for (j in range64(4l)) {   // ggml scales[12]: [0,7] values keep the packed high bits zero
            kc[co + 4l + j] = uint8(int((sb + j * 5l + seed) % 8l))
            kc[co + 8l + j] = uint8(int((sb * 3l + j * 7l + 1l) % 8l))
            kc[co + 12l + j] = uint8(int((sb + (j + 4l) * 5l + seed) % 8l) | (int((sb * 3l + (j + 4l) * 7l + 1l) % 8l) << 4))
        }
        let qo = sb * 128l
        for (b in range64(128l)) {
            let lo = int((b * 7l + sb * 13l + seed) % 16l)
            let hi = int((b * 11l + sb * 5l + 3l) % 16l)
            kq[qo + b] = uint8(lo | (hi << 4))
        }
    }
}

// 6-bit sub-scale/min j of a 12-byte packed Q4_K scale block (ggml get_scale_min_k4)
def private k4_sc_mn(kc : array<uint8>; base, j : int64) : int2 {
    if (j < 4l) {
        return int2(int(kc[base + j]) & 63, int(kc[base + j + 4l]) & 63)
    }
    let sc = (int(kc[base + j + 4l]) & 15) | ((int(kc[base + j - 4l]) >> 6) << 4)
    let mn = (int(kc[base + j + 4l]) >> 4) | ((int(kc[base + j]) >> 6) << 4)
    return int2(sc, mn)
}

// CPU k4 ref, one row of one expert plane against xf
def private ref_k4_row(kq : array<uint8>; kc : array<uint8>; xf : array<float>; plane_sb0, row, n : int64) : float {
    var acc = 0.0
    let nsb = n / 256l
    for (sb in range64(nsb)) {
        let g = plane_sb0 + row * nsb + sb
        let co = g * 16l
        let qo = g * 128l
        let dd = f16_to_f32(uint(kc[co]) | (uint(kc[co + 1l]) << 8u))
        let dm = f16_to_f32(uint(kc[co + 2l]) | (uint(kc[co + 3l]) << 8u))
        for (js in range64(8l)) {
            let s = k4_sc_mn(kc, co + 4l, js)
            for (l in range64(32l)) {
                let qv = int(kq[qo + (js / 2l) * 32l + l])
                let q = (js & 1l) == 0l ? (qv & 15) : (qv >> 4)
                acc += (dd * float(s.x) * float(q) - dm * float(s.y)) * xf[sb * 256l + js * 32l + l]
            }
        }
    }
    return acc
}

// q51 production planes: quant 20B blocks ([16B qs][4B qh]) + scale plane (d,m halves per block)
def private fill_q51_stack(var wq : array<uint8>; var ws : array<uint8>; rows, k, seed : int64) {
    let nblk = k / 32l
    let total = rows * nblk
    wq |> resize(total * 20l)
    ws |> resize(total * 4l)
    let lut = fixed_array(0.25, 0.5, 1.0, 2.0)
    for (b in range64(total)) {
        let db = f32_to_f16(lut[int((b / 7l) % 4l)])
        let mb = f32_to_f16(lut[int((b / 3l + 2l) % 4l)])
        let so = b * 4l
        ws[so] = uint8(db & 255u)
        ws[so + 1l] = uint8(db >> 8u)
        ws[so + 2l] = uint8(mb & 255u)
        ws[so + 3l] = uint8(mb >> 8u)
        let qo = b * 20l
        for (j in range64(16l)) {
            let lo = int((j * 7l + b * 13l + seed) % 16l)
            let hi = int((j * 11l + b * 5l + 3l) % 16l)
            wq[qo + j] = uint8(lo | (hi << 4))
        }
        for (j in range64(4l)) {
            wq[qo + 16l + j] = uint8(int((b * 29l + j * 17l + seed) % 256l))
        }
    }
}

// CPU q51 ref, one row of one expert plane: w = d*(q + 16*qh_bit) + m per element
def private ref_q51_row(wq : array<uint8>; ws : array<uint8>; xf : array<float>; plane_b0, row, n : int64) : float {
    var acc = 0.0
    let nblk = n / 32l
    for (blk in range64(nblk)) {
        let g = plane_b0 + row * nblk + blk
        let so = g * 4l
        let dd = f16_to_f32(uint(ws[so]) | (uint(ws[so + 1l]) << 8u))
        let dm = f16_to_f32(uint(ws[so + 2l]) | (uint(ws[so + 3l]) << 8u))
        let qo = g * 20l
        for (l in range64(32l)) {
            let qv = int(wq[qo + (l % 16l)])
            let q = l < 16l ? (qv & 15) : (qv >> 4)
            let hbit = (int(wq[qo + 16l + l / 8l]) >> int(l % 8l)) & 1
            acc += (dd * float(q + 16 * hbit) + dm) * xf[blk * 32l + l]
        }
    }
    return acc
}

// ===== host plumbing =====

def private env_or(name : string; dflt : int) : int {
    return has_env_variable(name) ? to_int(get_env_variable(name)) : dflt
}

def private want(filter_env, name : string) : bool {
    return !has_env_variable(filter_env) || find(get_env_variable(filter_env), name) >= 0
}

// generic (untyped param) — non-Apple builds never compile the Metal-typed body
def private upload_bytes(dev; arr : array<uint8>) {
    var buf = metal_new_buffer_untracked(dev, uint64(length(arr)))
    unsafe {
        memcpy(metal_buffer_contents(buf), addr < void? >(arr[0]), length(arr))
    }
    return buf
}

def private upload_floats(dev; arr : array<float>) {
    var buf = metal_new_buffer_untracked(dev, uint64(length(arr) * 4))
    unsafe {
        memcpy(metal_buffer_contents(buf), addr < void? >(arr[0]), length(arr) * 4)
    }
    return buf
}

def private upload_u32v(dev; v : uint) {
    var buf = metal_new_buffer(dev, 4ul)
    unsafe {
        var p = reinterpret<uint?>(metal_buffer_contents(buf))
        p[0] = v
    }
    return buf
}

// one command buffer: encoder (serial or concurrent), caller encodes, gpu window returned in us
def private time_arm(queue; concurrent : bool; blk) : double {
    var cb = metal_new_command_buffer(queue)
    var enc = concurrent ? metal_new_compute_encoder_concurrent(cb) : metal_new_compute_encoder(cb)
    blk |> invoke(enc)
    metal_end_encoding(enc)
    metal_commit(cb)
    metal_wait_until_completed(cb)
    let err = metal_command_buffer_error(cb)
    if (!empty(err)) {
        panic("command buffer error: {err}")
    }
    let us = (metal_command_buffer_gpu_end_time(cb) - metal_command_buffer_gpu_start_time(cb)) * 1e6lf
    metal_release(enc)
    metal_release(cb)
    return us
}

// bufs slots (array<MetalBuffer?> local to the generic body — Metal types never at module level)
let RB_K4Q = 0     // k4 quant stack, 2E planes (w1 = 0..E-1, w3 = E..2E-1)
let RB_K4S = 1     // k4 16B compact scale stack
let RB_Q51Q = 2    // q51 quant stack, E planes
let RB_Q51S = 3    // q51 d/m scale stack
let RB_X = 4       // shared activation vector [DIM]
let RB_H12 = 5     // per-layer [gate: K x NFE][up: K x NFE]
let RB_DN = 6      // per-layer [K x DIM]
let RB_SEL = 7     // sel rows: [scatter w1 | scatter w3 | contig w1 | contig w3] x LAYERS
let RB_UDIM = 8
let RB_UNFE = 9
let RB_UESB = 10
let RB_UEBLK = 11
let RB_UZERO = 12
let RB_UXSNFE = 13
let RB_UKNFE = 14
let RB_UESHIFT = 15
// psos slots
let RP_K4 = 0      // production expert-indexed k4 GEMV
let RP_Q51 = 1     // production expert-indexed q51 GEMV
let RP_GLU = 2     // production geglu ew
let RP_PAIR = 3    // the lab W1+W3 paired kernel
let RP_X4 = 4      // float4-x variant
let RP_R4 = 5      // 4-rows-per-simdgroup variant

let SEL_ROW_BYTES = 32ul                          // 8 idx uints per row
let SEL_W3_BASE = uint64(LAYERS) * SEL_ROW_BYTES  // block starts in bsel
let SEL_CW1_BASE = SEL_W3_BASE * 2ul
let SEL_CW3_BASE = SEL_W3_BASE * 3ul

def private h12_off(l : int64) : uint64 => uint64(l * 2l * KNFE * 4l)
def private dn_off(l : int64) : uint64 => uint64(l * K * DIM * 4l)

def private enc_lab_w13(bufs; psos; enc; l : int64; sel_base : uint64; up_half : bool) {
    metal_set_pipeline(enc, psos[RP_K4])
    metal_set_buffer(enc, bufs[RB_K4S], 0ul, 0)
    metal_set_buffer(enc, bufs[RB_K4S], 0ul, 1)
    metal_set_buffer(enc, bufs[RB_K4Q], 0ul, 2)
    metal_set_buffer(enc, bufs[RB_X], 0ul, 3)
    metal_set_buffer(enc, bufs[RB_H12], h12_off(l) + (up_half ? uint64(KNFE * 4l) : 0ul), 4)
    metal_set_buffer(enc, bufs[RB_UDIM], 0ul, 5)
    metal_set_buffer(enc, bufs[RB_UNFE], 0ul, 6)
    metal_set_buffer(enc, bufs[RB_SEL], sel_base + uint64(l) * SEL_ROW_BYTES, 7)
    metal_set_buffer(enc, bufs[RB_UESB], 0ul, 8)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 9)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 10)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 11)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 12)
    metal_dispatch_threadgroups(enc, uint3(uint((NFE + 3l) / 4l), uint(K), 1u), uint3(64u, 1u, 1u))
}

// variant dispatch: same superset binding table (extra binds past a pipeline's table are ignored),
// rows_tg = output rows per threadgroup (4 for the 2-row forms, 8 for r4)
def private enc_lab_w13v(bufs; psos; enc; l : int64; sel_base : uint64; up_half : bool; pslot : int; rows_tg : int64) {
    metal_set_pipeline(enc, psos[pslot])
    metal_set_buffer(enc, bufs[RB_K4S], 0ul, 0)
    metal_set_buffer(enc, bufs[RB_K4S], 0ul, 1)
    metal_set_buffer(enc, bufs[RB_K4Q], 0ul, 2)
    metal_set_buffer(enc, bufs[RB_X], 0ul, 3)
    metal_set_buffer(enc, bufs[RB_H12], h12_off(l) + (up_half ? uint64(KNFE * 4l) : 0ul), 4)
    metal_set_buffer(enc, bufs[RB_UDIM], 0ul, 5)
    metal_set_buffer(enc, bufs[RB_UNFE], 0ul, 6)
    metal_set_buffer(enc, bufs[RB_SEL], sel_base + uint64(l) * SEL_ROW_BYTES, 7)
    metal_set_buffer(enc, bufs[RB_UESB], 0ul, 8)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 9)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 10)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 11)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 12)
    metal_dispatch_threadgroups(enc, uint3(uint((NFE + rows_tg - 1l) / rows_tg), uint(K), 1u), uint3(64u, 1u, 1u))
}

def private enc_lab_pair(bufs; psos; enc; l : int64; sel_base : uint64) {
    metal_set_pipeline(enc, psos[RP_PAIR])
    metal_set_buffer(enc, bufs[RB_K4S], 0ul, 0)
    metal_set_buffer(enc, bufs[RB_K4S], 0ul, 1)
    metal_set_buffer(enc, bufs[RB_K4Q], 0ul, 2)
    metal_set_buffer(enc, bufs[RB_X], 0ul, 3)
    metal_set_buffer(enc, bufs[RB_H12], h12_off(l), 4)
    metal_set_buffer(enc, bufs[RB_UDIM], 0ul, 5)
    metal_set_buffer(enc, bufs[RB_UNFE], 0ul, 6)
    metal_set_buffer(enc, bufs[RB_SEL], sel_base + uint64(l) * SEL_ROW_BYTES, 7)
    metal_set_buffer(enc, bufs[RB_UESB], 0ul, 8)
    metal_set_buffer(enc, bufs[RB_UESHIFT], 0ul, 9)
    metal_set_buffer(enc, bufs[RB_UKNFE], 0ul, 10)
    metal_dispatch_threadgroups(enc, uint3(uint((NFE + 3l) / 4l), uint(K), 1u), uint3(64u, 1u, 1u))
}

def private enc_lab_glu(bufs; psos; enc; l : int64) {
    metal_set_pipeline(enc, psos[RP_GLU])
    metal_set_buffer(enc, bufs[RB_H12], h12_off(l), 0)
    metal_set_buffer(enc, bufs[RB_H12], h12_off(l) + uint64(KNFE * 4l), 1)
    metal_set_buffer(enc, bufs[RB_UKNFE], 0ul, 2)
    metal_dispatch_threadgroups(enc, uint3(uint((KNFE + 255l) / 256l), 1u, 1u), uint3(256u, 1u, 1u))
}

def private enc_lab_w2(bufs; psos; enc; l : int64; sel_base : uint64) {
    metal_set_pipeline(enc, psos[RP_Q51])
    metal_set_buffer(enc, bufs[RB_Q51Q], 0ul, 0)
    metal_set_buffer(enc, bufs[RB_Q51S], 0ul, 1)
    metal_set_buffer(enc, bufs[RB_H12], h12_off(l), 2)
    metal_set_buffer(enc, bufs[RB_DN], dn_off(l), 3)
    metal_set_buffer(enc, bufs[RB_UNFE], 0ul, 4)
    metal_set_buffer(enc, bufs[RB_UDIM], 0ul, 5)
    metal_set_buffer(enc, bufs[RB_SEL], sel_base + uint64(l) * SEL_ROW_BYTES, 6)
    metal_set_buffer(enc, bufs[RB_UEBLK], 0ul, 7)
    metal_set_buffer(enc, bufs[RB_UXSNFE], 0ul, 8)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 9)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 10)
    metal_set_buffer(enc, bufs[RB_UZERO], 0ul, 11)
    metal_dispatch_threadgroups(enc, uint3(uint((DIM + 3l) / 4l), uint(K), 1u), uint3(64u, 1u, 1u))
}

// one pass of the routed chain under a barrier pattern: 0 = none, 1 = every dispatch (the
// serial-encoder semantics), 2 = real hazards only (w1 ∥ w3), 3 = paired W1+W3 + real hazards
def private enc_chain_pass(bufs; psos; enc; pattern : int; contig : bool) {
    let w1b = contig ? SEL_CW1_BASE : 0ul
    let w3b = contig ? SEL_CW3_BASE : SEL_W3_BASE
    for (l in range64(LAYERS)) {
        if (pattern == 3) {
            enc_lab_pair(bufs, psos, enc, l, w1b)
        } else {
            enc_lab_w13(bufs, psos, enc, l, w1b, false)
            if (pattern == 1) {
                metal_memory_barrier(enc)
            }
            enc_lab_w13(bufs, psos, enc, l, w3b, true)
        }
        if (pattern != 0) {
            metal_memory_barrier(enc)
        }
        enc_lab_glu(bufs, psos, enc, l)
        if (pattern != 0) {
            metal_memory_barrier(enc)
        }
        enc_lab_w2(bufs, psos, enc, l, w1b)
        if (pattern != 0) {
            metal_memory_barrier(enc)
        }
    }
}

def private compile_or_die(dev; src, entry : string; fastmath : bool) {
    var err : string
    var pso = pipeline_from_source(dev, src, entry, fastmath, err)
    if (pso == null) {
        panic("PSO '{entry}' failed: {err}")
    }
    return pso
}

def private main_apple(dummy) {
    let rounds = env_or("DASMETAL_LAB_ROUNDS", 3)
    let passes = int64(env_or("DASMETAL_LAB_PASSES", 4))
    var dev = metal_create_system_default_device()
    if (dev == null) {
        print("no Metal device\n")
        return
    }
    print("device={metal_device_name(dev)} rounds={rounds} passes={passes}\n")
    var queue = metal_new_command_queue(dev)
    var psos : array<MetalComputePipeline?>
    psos |> push(compile_or_die(dev, metal_moe_gemv_k4_msl, metal_moe_gemv_k4_msl_entry, metal_moe_gemv_k4_msl_fastmath))
    psos |> push(compile_or_die(dev, metal_moe_gemv_q51_msl, metal_moe_gemv_q51_msl_entry, metal_moe_gemv_q51_msl_fastmath))
    psos |> push(compile_or_die(dev, metal_geglu_msl, metal_geglu_msl_entry, metal_geglu_msl_fastmath))
    psos |> push(compile_or_die(dev, lab_moe_k4_pair_msl, lab_moe_k4_pair_msl_entry, lab_moe_k4_pair_msl_fastmath))
    psos |> push(compile_or_die(dev, lab_moe_k4_x4_msl, lab_moe_k4_x4_msl_entry, lab_moe_k4_x4_msl_fastmath))
    psos |> push(compile_or_die(dev, lab_moe_k4_r4_msl, lab_moe_k4_r4_msl_entry, lab_moe_k4_r4_msl_fastmath))

    // host fills (exact-arithmetic contract)
    let t_fill = ref_time_ticks()
    var k4q : array<uint8>
    var k4c : array<uint8>
    fill_k4_stack(k4q, k4c, 2l * E * NFE, DIM, 7l)
    var q51q : array<uint8>
    var q51s : array<uint8>
    fill_q51_stack(q51q, q51s, E * DIM, NFE, 11l)
    var xf : array<float>
    xf |> resize(int(DIM))
    for (i in range64(DIM)) {
        xf[i] = float(int((i * 7l) % 15l) - 7)
    }
    print("host fill: {get_time_usec(t_fill) / 1000} ms (k4 {length(k4q) / 1048576} + {length(k4c) / 1048576} MB, q51 {length(q51q) / 1048576} + {length(q51s) / 1048576} MB)\n")

    // sel rows: scattered = 8 distinct experts spread across the plane, rotated per layer;
    // contig = 8 consecutive experts (the locality control). w3 rows ride the +E plane block.
    var sel : array<uint>
    sel |> resize(int(4l * LAYERS * 8l))
    for (l in range64(LAYERS)) {
        for (s in range64(8l)) {
            let e = uint(s * 16l + (l * 7l + s * 3l) % 16l)
            sel[l * 8l + s] = e
            sel[LAYERS * 8l + l * 8l + s] = e + uint(E)
            sel[2l * LAYERS * 8l + l * 8l + s] = uint((l * 8l + s) % E)
            sel[3l * LAYERS * 8l + l * 8l + s] = uint((l * 8l + s) % E) + uint(E)
        }
    }
    var selb : array<uint8>
    selb |> resize(length(sel) * 4)
    unsafe {
        memcpy(addr < void? >(selb[0]), addr < void? >(sel[0]), length(sel) * 4)
    }

    var bufs : array<MetalBuffer?>
    bufs |> push(upload_bytes(dev, k4q))
    bufs |> push(upload_bytes(dev, k4c))
    bufs |> push(upload_bytes(dev, q51q))
    bufs |> push(upload_bytes(dev, q51s))
    bufs |> push(upload_floats(dev, xf))
    bufs |> push(metal_new_buffer_untracked(dev, uint64(LAYERS * 2l * KNFE * 4l)))
    bufs |> push(metal_new_buffer_untracked(dev, uint64(LAYERS * K * DIM * 4l)))
    bufs |> push(upload_bytes(dev, selb))
    bufs |> push(upload_u32v(dev, uint(DIM)))
    bufs |> push(upload_u32v(dev, uint(NFE)))
    bufs |> push(upload_u32v(dev, uint(ESB)))
    bufs |> push(upload_u32v(dev, uint(EBLK)))
    bufs |> push(upload_u32v(dev, 0u))
    bufs |> push(upload_u32v(dev, uint(NFE)))
    bufs |> push(upload_u32v(dev, uint(KNFE)))
    bufs |> push(upload_u32v(dev, uint(E)))

    // ===== correctness: standalone w1 (k4), w2 (q51), and the pair kernel vs CPU refs =====
    var t = time_arm(queue, false) $(enc : MetalComputeEncoder?) {
        enc_lab_w13(bufs, psos, enc, 0l, 0ul, false)
    }
    var worst = 0.0
    unsafe {
        var y = reinterpret<float const?>(metal_buffer_contents(bufs[RB_H12]))
        for (s in range64(8l)) {
            for (row in [0l, 1l, 350l, NFE - 1l]) {
                let e = int64(sel[s])
                let want_v = ref_k4_row(k4q, k4c, xf, e * ESB, row, DIM)
                worst = max(worst, abs(y[s * NFE + row] - want_v) / max(1.0, abs(want_v)))
            }
        }
    }
    if (worst > 1e-4) {
        panic("w1 k4 correctness: worst rel diff {worst}")
    }
    print("w1 k4 correctness: worst rel {worst}\n")

    t = time_arm(queue, false) $(enc : MetalComputeEncoder?) {
        enc_lab_pair(bufs, psos, enc, 0l, 0ul)
    }
    worst = 0.0
    unsafe {
        var y = reinterpret<float const?>(metal_buffer_contents(bufs[RB_H12]))
        for (s in range64(8l)) {
            for (row in [0l, 1l, 350l, NFE - 1l]) {
                let e1 = int64(sel[s])
                let w1v = ref_k4_row(k4q, k4c, xf, e1 * ESB, row, DIM)
                let w3v = ref_k4_row(k4q, k4c, xf, (e1 + E) * ESB, row, DIM)
                worst = max(worst, abs(y[s * NFE + row] - w1v) / max(1.0, abs(w1v)))
                worst = max(worst, abs(y[KNFE + s * NFE + row] - w3v) / max(1.0, abs(w3v)))
            }
        }
    }
    if (worst > 1e-4) {
        panic("pair k4 correctness: worst rel diff {worst}")
    }
    print("pair k4 correctness: worst rel {worst}\n")

    for (vv in [(RP_X4, 4l, "x4"), (RP_R4, 8l, "r4")]) {
        let pslot = vv._0
        let rtg = vv._1
        t = time_arm(queue, false) $(enc : MetalComputeEncoder?) {
            enc_lab_w13v(bufs, psos, enc, 0l, 0ul, false, pslot, rtg)
        }
        worst = 0.0
        unsafe {
            var y = reinterpret<float const?>(metal_buffer_contents(bufs[RB_H12]))
            for (s in range64(8l)) {
                for (row in [0l, 1l, 350l, NFE - 1l]) {
                    let e = int64(sel[s])
                    let want_v = ref_k4_row(k4q, k4c, xf, e * ESB, row, DIM)
                    worst = max(worst, abs(y[s * NFE + row] - want_v) / max(1.0, abs(want_v)))
                }
            }
        }
        if (worst > 1e-4) {
            panic("{vv._2} k4 correctness: worst rel diff {worst}")
        }
        print("{vv._2} k4 correctness: worst rel {worst}\n")
    }

    // w2 standalone: fill h12 layer 0 gate half with exact ints, then one q51 dispatch
    var h0 : array<float>
    h0 |> resize(int(KNFE))
    for (i in range64(KNFE)) {
        h0[i] = float(int((i * 11l) % 13l) - 6)
    }
    unsafe {
        memcpy(metal_buffer_contents(bufs[RB_H12]), addr < void? >(h0[0]), int(KNFE) * 4)
    }
    t = time_arm(queue, false) $(enc : MetalComputeEncoder?) {
        enc_lab_w2(bufs, psos, enc, 0l, 0ul)
    }
    worst = 0.0
    unsafe {
        var y = reinterpret<float const?>(metal_buffer_contents(bufs[RB_DN]))
        var hslot : array<float>
        hslot |> resize(int(NFE))
        for (s in range64(8l)) {
            for (i in range64(NFE)) {
                hslot[i] = h0[s * NFE + i]
            }
            for (row in [0l, 1l, 1500l, DIM - 1l]) {
                let e = int64(sel[s])
                let want_v = ref_q51_row(q51q, q51s, hslot, e * EBLK, row, NFE)
                worst = max(worst, abs(y[s * DIM + row] - want_v) / max(1.0, abs(want_v)))
            }
        }
        delete hslot
    }
    if (worst > 1e-4) {
        panic("w2 q51 correctness: worst rel diff {worst}")
    }
    print("w2 q51 correctness: worst rel {worst}\n")

    // ===== the timing matrix =====
    // per-pass routed weight bytes: K slots x (2 k4 planes + 1 q51 plane) x LAYERS
    let bytes_pass = double(LAYERS * K * (2l * NFE * NSB_ROW * 144l + DIM * NBLK_ROW * 24l))
    var arms <- [("pipe", 0, false, true), ("pipe_c", 0, true, true), ("serial", 1, false, true),
                 ("hazard", 2, false, true), ("pair13", 3, false, true), ("true_ser", 1, false, false)]
    var best : table<string; double>
    for (a in arms) {
        best[a._0] = 1e18lf
    }
    // true_ser runs pattern-free on a serial encoder — pattern 1 barriers would double-count
    for (_rd in range(rounds)) {
        for (a in arms) {
            if (!want("DASMETAL_LAB_ARMS", a._0)) {
                continue
            }
            let name = a._0
            let pattern = a._3 ? a._1 : 0
            let contig = a._2
            let conc = a._3
            let us = time_arm(queue, conc) $(enc : MetalComputeEncoder?) {
                for (_p in range64(passes)) {
                    enc_chain_pass(bufs, psos, enc, pattern, contig)
                }
            }
            let per = us / double(passes)
            if (per < best[name]) {
                best[name] = per
            }
        }
    }
    print("--- best per arm ({int(bytes_pass) / 1048576} MB routed weights/pass) ---\n")
    for (a in arms) {
        if (!want("DASMETAL_LAB_ARMS", a._0)) {
            continue
        }
        let per = best[a._0]
        let gbs = bytes_pass / per / 1000.0lf
        print("  {a._0}: {per} us/pass  {per / double(LAYERS)} us/layer  {gbs} wGB/s\n")
    }

    // ===== per-site kernel matrix (pipelined, no barriers — kernel throughput only) =====
    // psel >= 0: w1+w3 via that variant; -1: the pair kernel; -2: the q51 w2 site
    let bytes_w13 = double(LAYERS * K * 2l * NFE * NSB_ROW * 144l)
    let bytes_w2 = double(LAYERS * K * DIM * NBLK_ROW * 24l)
    var sarms <- [("w13", RP_K4, 4l), ("w13_x4", RP_X4, 4l), ("w13_r4", RP_R4, 8l),
                  ("w13_pair", -1, 4l), ("w2", -2, 4l)]
    var sbest : table<string; double>
    for (a in sarms) {
        sbest[a._0] = 1e18lf
    }
    for (_rd in range(rounds)) {
        for (a in sarms) {
            if (!want("DASMETAL_LAB_ARMS", a._0)) {
                continue
            }
            let name = a._0
            let pslot = a._1
            let rtg = a._2
            let us = time_arm(queue, true) $(enc : MetalComputeEncoder?) {
                for (_p in range64(passes)) {
                    for (l in range64(LAYERS)) {
                        if (pslot == -1) {
                            enc_lab_pair(bufs, psos, enc, l, 0ul)
                        } elif (pslot == -2) {
                            enc_lab_w2(bufs, psos, enc, l, 0ul)
                        } else {
                            enc_lab_w13v(bufs, psos, enc, l, 0ul, false, pslot, rtg)
                            enc_lab_w13v(bufs, psos, enc, l, SEL_W3_BASE, true, pslot, rtg)
                        }
                    }
                }
            }
            let per = us / double(passes)
            if (per < sbest[name]) {
                sbest[name] = per
            }
        }
    }
    print("--- per-site kernel matrix (w13 {int(bytes_w13) / 1048576} MB, w2 {int(bytes_w2) / 1048576} MB per pass) ---\n")
    for (a in sarms) {
        if (!want("DASMETAL_LAB_ARMS", a._0)) {
            continue
        }
        let per = sbest[a._0]
        let bytes_arm = a._1 == -2 ? bytes_w2 : bytes_w13
        print("  {a._0}: {per} us/pass  {bytes_arm / per / 1000.0lf} wGB/s\n")
    }
    delete sarms
    delete sbest

    delete k4q
    delete k4c
    delete q51q
    delete q51s
    delete xf
    delete sel
    delete selb
    delete h0
    delete arms
    delete best
    for (b in bufs) {
        metal_release(b)
    }
    bufs |> clear()   // handles are metal_release-owned — delete on a ptr array would free pointees
    unsafe {
        delete bufs
    }
    for (p in psos) {
        metal_release(p)
    }
    psos |> clear()
    unsafe {
        delete psos
    }
    metal_release(queue)
    metal_release(dev)
    assert_no_metal_leaks()
}

[export]
def main {
    static_if (typeinfo builtin_module_exists(das_metal)) {
        main_apple(0)
    } else {
        print("das_metal is not built on this platform\n")
    }
}
