options gen2
options stack = 524288

require dastest/testing_boost public
require dasllama/dasllama_math
require ?vulkan dasllama/dasllama_math_vulkan  // nolint:STYLE030 — the MoE GPU tier under test; the test installs its hooks explicitly (no env gate)
require llvm/daslib/f16_cvt
require math

// The Vulkan MoE GPU tier (dasllama_math_vulkan) vs a hand-rolled CPU reference of the same
// fused FFN contract: region-triple gate+up GEMMs -> silu(g)*u -> requant -> down GEMM,
// on synthetic stacks uploaded through the public hook surface (install_moe_gpu_tier /
// moe_gpu_upload_stack / matmul_moe_gpu_ffn). Q8 arms run both dispatch shapes: decode GEMV
// (nrows < 32) and the tiled prefill batch (nrows >= 32, deliberately unpadded). The kq arms
// (native K-quant stacks — k4 gate/up with a k5 and a k6 down, the 35B Q4_K_M shapes) drive
// both shapes too: the per-format GEMV kernels + the Q8_K requant mid-chain at decode row
// counts, the per-format batch tiles at 40 rows. Weight scales are powers of two
// (f16-exact), so the only drift left is the GPU's exp() in silu and round-halfway splits in
// the requant — the compare is relative-tolerance, not bit-exact.
// The gate/up scale planes are SMALL pow2 (2^-7..2^-9) on purpose: they keep |gate| in the
// sane silu regime (~±25). Full-size scales drive |gate| to ~3e5, where CPU exp() and GPU
// exp() saturate DIFFERENTLY (probe-verified 60%+ output divergence) — a regime no real
// model's activations enter. The gelu arm stays untested until a gelu-MoE q8 fixture exists.
// Without dasVulkan (or with no Vulkan device) every check feints cleanly.

let N = 96l         // model dim (x row length), %32
let NFE = 64l       // expert FFN dim, %32
let NE = 3l         // experts in each synthetic stack
let WOFF1 = 1000000l   // gate / up / down stack base element offsets (arbitrary distinct keys —
let WOFF3 = 2000000l   // the tier resolves regions by [base, base+elems) containment)
let WOFF2 = 3000000l
let REL_TOL = 1e-3

def private pow2_scale(i : int64) : float {
    let p = i % 5l
    return p == 0l ? 1.0 : (p == 1l ? 0.5 : (p == 2l ? 2.0 : (p == 3l ? 0.25 : 0.125)))
}

// the small set for gate/up/x scale planes — keeps |gate| in the sane silu regime (see header)
def private pow2_small(i : int64) : float {
    let p = i % 3l
    return p == 0l ? 0.00390625 : (p == 1l ? 0.001953125 : 0.0078125)   // 2^-8 / 2^-9 / 2^-7
}

def private fill_stack(var wq : array<int8>; var ws : array<float>; rows, n : int64; seed : int64; small_scales : bool) {
    wq |> resize(int(rows * n))
    ws |> resize(int(rows * (n / 32l)))
    for (i in range(int(rows * n))) {
        wq[i] = int8((int64(i) * 7l + seed) % 255l - 127l)
    }
    for (i in range(int(rows * (n / 32l)))) {
        ws[i] = small_scales ? pow2_small(int64(i) + seed) : pow2_scale(int64(i) + seed)
    }
}

// the upload wire format: row-major int8 quants as bytes + f16 halfword scales as bytes (LE)
def private pack_upload(wq : array<int8>; ws : array<float>; var wqb : array<uint8>; var wsb : array<uint8>) {
    wqb |> resize(length(wq))
    for (i in range(length(wq))) {
        wqb[i] = uint8(wq[i])
    }
    wsb |> resize(length(ws) * 2)
    for (i in range(length(ws))) {
        let h = f32_to_f16(ws[i])
        wsb[i * 2] = uint8(h & 0xFFu)
        wsb[i * 2 + 1] = uint8((h >> 8u) & 0xFFu)
    }
}

def private fill_acts(var xq : array<int8>; var xs : array<float>; nrows : int64) {
    xq |> resize(int(nrows * N))
    xs |> resize(int(nrows * (N / 32l)))
    for (i in range(int(nrows * N))) {
        xq[i] = int8((int64(i) * 11l + 3l) % 255l - 127l)
    }
    for (i in range(int(nrows * (N / 32l)))) {
        xs[i] = pow2_small(int64(i) + 2l)
    }
}

// one region-triple list over experts[], rows partitioning [0, nrows)
def private make_offs(experts : array<int64>; row0s : array<int64>; cnts : array<int64>; woff : int64; ege : int64) : array<int64> {
    var offs : array<int64>
    offs |> reserve(length(experts) * 3)
    for (r in range(length(experts))) {
        offs |> push(woff + experts[r] * ege)   // nolint:STYLE012 — triple-stride append
        offs |> push(row0s[r])
        offs |> push(cnts[r])
    }
    return <- offs
}

// the CPU reference of the tier's fused-FFN contract (same formulas as the kernels' CPU mirrors)
def private ref_ffn(offs1, offs3, offs2 : array<int64>; nreg : int;
                    wq1 : array<int8>; ws1 : array<float>; wq3 : array<int8>; ws3 : array<float>;
                    wq2 : array<int8>; ws2 : array<float>;
                    xq : array<int8>; xs : array<float>; nrows : int64; var gout : array<float>) {
    gout |> resize(int(nrows * N))
    let nb = int(N / 32l)
    let nbh = int(NFE / 32l)
    var gate : array<float>
    var up : array<float>
    var h : array<float>
    var hq : array<int>
    var hs : array<float>
    gate |> resize(int(NFE))
    up |> resize(int(NFE))
    h |> resize(int(NFE))
    hq |> resize(int(NFE))
    hs |> resize(nbh)
    for (r in range(nreg)) {
        let rb1 = int((offs1[r * 3] - WOFF1) / N)      // first gate-plane row of this region's expert
        let rb3 = int((offs3[r * 3] - WOFF3) / N)
        let rb2 = int((offs2[r * 3] - WOFF2) / NFE)
        let row0 = int(offs1[r * 3 + 1])
        let cnt = int(offs1[r * 3 + 2])
        for (rw in range(row0, row0 + cnt)) {
            for (i in range(int(NFE))) {
                var ag = 0.0
                var au = 0.0
                for (b in range(nb)) {
                    var dg = 0
                    var du = 0
                    for (j in range(32)) {
                        let xv = int(xq[rw * int(N) + b * 32 + j])
                        dg += int(wq1[(rb1 + i) * int(N) + b * 32 + j]) * xv
                        du += int(wq3[(rb3 + i) * int(N) + b * 32 + j]) * xv
                    }
                    let xscl = xs[rw * nb + b]
                    ag += float(dg) * ws1[(rb1 + i) * nb + b] * xscl
                    au += float(du) * ws3[(rb3 + i) * nb + b] * xscl
                }
                gate[i] = ag
                up[i] = au
            }
            for (i in range(int(NFE))) {
                h[i] = (gate[i] / (1.0 + exp(-gate[i]))) * up[i]
            }
            for (b in range(nbh)) {
                var amax = 0.0
                for (j in range(32)) {
                    amax = max(amax, abs(h[b * 32 + j]))
                }
                let d = amax / 127.0
                let id = d != 0.0 ? 1.0 / d : 0.0
                hs[b] = d
                for (j in range(32)) {
                    hq[b * 32 + j] = int(round(h[b * 32 + j] * id))
                }
            }
            for (o in range(int(N))) {
                var y = 0.0
                for (b in range(nbh)) {
                    var dd = 0
                    for (j in range(32)) {
                        dd += int(wq2[(rb2 + o) * int(NFE) + b * 32 + j]) * hq[b * 32 + j]
                    }
                    y += float(dd) * ws2[(rb2 + o) * nbh + b] * hs[b]
                }
                gout[rw * int(N) + o] = y
            }
        }
    }
}

// max |gpu - ref| over the touched rows, and the max |ref| it is judged against
def private compare_gout(gpu, refv : array<float>) : tuple<maxdiff : float; maxref : float> {
    var md = 0.0
    var mr = 0.0
    for (i in range(length(refv))) {
        md = max(md, abs(gpu[i] - refv[i]))
        mr = max(mr, abs(refv[i]))
    }
    return (maxdiff = md, maxref = mr)
}

[test]
def test_vulkan_moe_ffn_vs_ref(t : T?) {
    t |> run("vulkan MoE FFN tier == CPU reference (decode GEMV + prefill batch arms)") @(t : T?) {
        static_if (typeinfo builtin_module_exists(vulkan)) {
            install_moe_gpu_tier(@@vk_moe_ffn, @@vk_moe_upload_stack, @@vk_moe_cls)
            // synthetic stacks: gate/up are (NE*NFE) x N, down is (NE*N) x NFE
            var wq1 : array<int8>
            var ws1 : array<float>
            var wq3 : array<int8>
            var ws3 : array<float>
            var wq2 : array<int8>
            var ws2 : array<float>
            fill_stack(wq1, ws1, NE * NFE, N, 0l, true)
            fill_stack(wq3, ws3, NE * NFE, N, 5l, true)
            fill_stack(wq2, ws2, NE * N, NFE, 9l, false)
            var wqb : array<uint8>
            var wsb : array<uint8>
            pack_upload(wq1, ws1, wqb, wsb)
            var up_ok = moe_gpu_upload_stack(wqb, wsb, WOFF1, N, NE * NFE, 0)
            if (!up_ok) {
                feint("no Vulkan device (or probe/budget refused the upload); skipping\n")
                return
            }
            pack_upload(wq3, ws3, wqb, wsb)
            up_ok = moe_gpu_upload_stack(wqb, wsb, WOFF3, N, NE * NFE, 0)
            t |> success(up_ok, "up stack uploaded")
            pack_upload(wq2, ws2, wqb, wsb)
            up_ok = moe_gpu_upload_stack(wqb, wsb, WOFF2, NFE, NE * N, 0)
            t |> success(up_ok, "down stack uploaded")
            let ege1 = NFE * N     // one expert's element span in the gate/up planes
            let ege2 = N * NFE     // ... and the down plane

            // decode arm: 2 rows, 1 row per region (the engine's k=2 GEMV shape)
            var xq : array<int8>
            var xs : array<float>
            fill_acts(xq, xs, 2l)
            var offs1 <- make_offs([1l, 2l], [0l, 1l], [1l, 1l], WOFF1, ege1)
            var offs3 <- make_offs([1l, 2l], [0l, 1l], [1l, 1l], WOFF3, ege1)
            var offs2 <- make_offs([1l, 2l], [0l, 1l], [1l, 1l], WOFF2, ege2)
            var gout : array<float>
            gout |> resize(int(2l * N))
            matmul_moe_gpu_ffn(gout, offs1, offs3, offs2, 2l, xq, xs, N, NFE, 2l, 0, 0, 0, false)
            var refv : array<float>
            ref_ffn(offs1, offs3, offs2, 2, wq1, ws1, wq3, ws3, wq2, ws2, xq, xs, 2l, refv)
            var c = compare_gout(gout, refv)
            t |> success(c.maxref > 0.0, "decode arm produced non-trivial outputs")
            t |> success(c.maxdiff <= float(REL_TOL) * c.maxref,
                "decode GEMV arm within tolerance (maxdiff {c.maxdiff} vs {float(REL_TOL) * c.maxref} bar, maxref {c.maxref})")

            // batch arm: 40 rows (>= the 32-row tile, deliberately unpadded), 2 regions
            fill_acts(xq, xs, 40l)
            var boffs1 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], WOFF1, ege1)
            var boffs3 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], WOFF3, ege1)
            var boffs2 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], WOFF2, ege2)
            gout |> resize(int(40l * N))
            matmul_moe_gpu_ffn(gout, boffs1, boffs3, boffs2, 2l, xq, xs, N, NFE, 40l, 0, 0, 0, false)
            ref_ffn(boffs1, boffs3, boffs2, 2, wq1, ws1, wq3, ws3, wq2, ws2, xq, xs, 40l, refv)
            c = compare_gout(gout, refv)
            t |> success(c.maxref > 0.0, "batch arm produced non-trivial outputs")
            t |> success(c.maxdiff <= float(REL_TOL) * c.maxref,
                "prefill batch arm within tolerance (maxdiff {c.maxdiff} vs {float(REL_TOL) * c.maxref} bar, maxref {c.maxref})")
        } else {
            feint("dasVulkan not installed; vulkan MoE tier untestable here\n")
        }
    }
}

// ===== streamed-prefill arm (pinned mirrors + slot rotation) =====

// three q8 "layers" uploaded stream=true in loader order (descending layers => group 0 is the
// runtime-LAST layer), dispatched in runtime order 2,1,0 then 2 again — covers the cold copy,
// the unfenced prefetch hit, slot rotation, and an eviction reload, each vs the CPU reference
let SWOFF = 20000000l
let SSTRIDE = 100000l   // per-stack offset stride inside the streamed fixture (%32 == 0)

[test]
def test_vulkan_moe_ffn_streamed(t : T?) {
    t |> run("vulkan MoE streamed prefill (pinned mirrors, 2 slots) == CPU reference") @(t : T?) {
        static_if (typeinfo builtin_module_exists(vulkan)) {
            install_moe_gpu_tier(@@vk_moe_ffn, @@vk_moe_upload_stack, @@vk_moe_cls)
            let ege1 = NFE * N
            let ege2 = N * NFE
            var wq1 : array<array<int8>>
            var ws1 : array<array<float>>
            var wq3 : array<array<int8>>
            var ws3 : array<array<float>>
            var wq2 : array<array<int8>>
            var ws2 : array<array<float>>
            wq1 |> resize(3)
            ws1 |> resize(3)
            wq3 |> resize(3)
            ws3 |> resize(3)
            wq2 |> resize(3)
            ws2 |> resize(3)
            var wqb : array<uint8>
            var wsb : array<uint8>
            var streamed_ok = true
            for (g in range(3)) {
                fill_stack(wq1[g], ws1[g], NE * NFE, N, int64(g) * 31l + 2l, true)
                fill_stack(wq3[g], ws3[g], NE * NFE, N, int64(g) * 31l + 7l, true)
                fill_stack(wq2[g], ws2[g], NE * N, NFE, int64(g) * 31l + 11l, false)
                let base = SWOFF + int64(g) * 3l * SSTRIDE
                pack_upload(wq1[g], ws1[g], wqb, wsb)
                let ok1 = moe_gpu_upload_stack(wqb, wsb, base, N, NE * NFE, 0, [stream = true])
                pack_upload(wq3[g], ws3[g], wqb, wsb)
                let ok3 = moe_gpu_upload_stack(wqb, wsb, base + SSTRIDE, N, NE * NFE, 0, [stream = true])
                pack_upload(wq2[g], ws2[g], wqb, wsb)
                let ok2 = moe_gpu_upload_stack(wqb, wsb, base + 2l * SSTRIDE, NFE, NE * N, 0, [stream = true])
                streamed_ok = streamed_ok && ok1 && ok3 && ok2
            }
            if (!streamed_ok) {
                feint("no Vulkan device (or probe refused the stream uploads); skipping\n")
                return
            }
            var xq : array<int8>
            var xs : array<float>
            var gout : array<float>
            var refv : array<float>
            for (pass_i in [2, 1, 0, 2]) {   // runtime order + an eviction reload of group 2
                let g = pass_i
                let base = SWOFF + int64(g) * 3l * SSTRIDE
                fill_acts(xq, xs, 40l)
                var offs1 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], base, ege1)
                var offs3 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], base + SSTRIDE, ege1)
                var offs2 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], base + 2l * SSTRIDE, ege2)
                gout |> resize(int(40l * N))
                matmul_moe_gpu_ffn(gout, offs1, offs3, offs2, 2l, xq, xs, N, NFE, 40l, 0, 0, 0, false)
                var roffs1 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], WOFF1, ege1)
                var roffs3 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], WOFF3, ege1)
                var roffs2 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], WOFF2, ege2)
                ref_ffn(roffs1, roffs3, roffs2, 2, wq1[g], ws1[g], wq3[g], ws3[g], wq2[g], ws2[g], xq, xs, 40l, refv)
                let c = compare_gout(gout, refv)
                t |> success(c.maxref > 0.0, "streamed group {g} produced non-trivial outputs")
                t |> success(c.maxdiff <= float(REL_TOL) * c.maxref,
                    "streamed group {g} within tolerance (maxdiff {c.maxdiff} vs {float(REL_TOL) * c.maxref} bar, maxref {c.maxref})")
                delete offs1
                delete offs3
                delete offs2
                delete roffs1
                delete roffs3
                delete roffs2
            }
            delete wq1
            delete ws1
            delete wq3
            delete ws3
            delete wq2
            delete ws2
            delete wqb
            delete wsb
            delete xq
            delete xs
            delete gout
            delete refv
        } else {
            feint("dasVulkan not installed; vulkan MoE tier untestable here\n")
        }
    }
}

// ===== native K-quant arms =====

let KN = 512l        // kq model dim (TWO superblocks per row — nsb = 1 hides stride bugs)
let KNFE = 512l      // kq expert FFN dim (two superblocks; the requant/down walk crosses one)
let KWOFF1 = 5000000l
let KWOFF3 = 6000000l
let KWOFF2 = 7000000l

// tiny pow2 set for the kq gate/up d planes — with 6-bit sc (<=63) and 4/5-bit quants the
// per-weight magnitude lands where |gate| stays in the sane silu regime (see header)
def private pow2_tiny(i : int64) : float {
    let p = i % 3l
    return p == 0l ? 0.000244140625 : (p == 1l ? 0.0001220703125 : 0.00048828125)   // 2^-12/-13/-11
}

def private lcg_byte(var st : uint64&) : uint8 {
    st = st * 6364136223846793005ul + 1442695040888963407ul
    return uint8((st >> 33ul) & 255ul)
}

// synthesize one kq stack directly in the tier's DEVICE layout: quant payload (any bytes are
// valid nibble/qh payloads) + 20B scale rows (f16 d/dmin from the pow2 sets, 6-bit sc/mn or
// signed i8 sub-scales)
def private fill_kq_stack(fmt : int; rows, n : int64; seed : int64; tiny_d : bool;
                          var wqb : array<uint8>; var wsb : array<uint8>) {
    let nsb = n / 256l
    let qsb = fmt == 1 ? 128l : (fmt == 2 ? 160l : 192l)
    wqb |> resize(int(rows * nsb * qsb))
    wsb |> resize(int(rows * nsb * 20l))
    var st = uint64(seed) * 2654435761ul + 12345ul
    for (c in wqb) {
        c = lcg_byte(st)
    }
    var sb = 0l
    while (sb < rows * nsb) {
        let so = int(sb * 20l)
        if (fmt == 3) {
            for (i in range(16)) {   // signed i8 sub-scales in a modest range
                wsb[so + i] = uint8(int(lcg_byte(st)) % 63 - 31)
            }
            let h = f32_to_f16(tiny_d ? pow2_tiny(sb + seed) : pow2_small(sb + seed))
            wsb[so + 16] = uint8(h & 0xFFu)
            wsb[so + 17] = uint8((h >> 8u) & 0xFFu)
            wsb[so + 18] = uint8(0)
            wsb[so + 19] = uint8(0)
        } else {
            let hd = f32_to_f16(tiny_d ? pow2_tiny(sb + seed) : pow2_small(sb + seed))
            let hm = f32_to_f16(tiny_d ? pow2_tiny(sb + seed + 1l) : pow2_small(sb + seed + 1l))
            wsb[so] = uint8(hd & 0xFFu)
            wsb[so + 1] = uint8((hd >> 8u) & 0xFFu)
            wsb[so + 2] = uint8(hm & 0xFFu)
            wsb[so + 3] = uint8((hm >> 8u) & 0xFFu)
            for (i in range(16)) {   // 8 sc + 8 mn, 6-bit
                wsb[so + 4 + i] = uint8(int(lcg_byte(st)) & 63)
            }
        }
        sb++
    }
}

// dequantize one device-layout stack to floats — the same extraction the GPU kernels do,
// validated bit-exact against the plane-dequant oracles by the kq gather probe
def private dequant_kq_stack(fmt : int; wqb : array<uint8>; wsb : array<uint8>; rows, n : int64;
                             var w : array<float>) {
    let nsb = n / 256l
    let qsb = fmt == 1 ? 128l : (fmt == 2 ? 160l : 192l)
    w |> resize(int(rows * n))
    for (rr in range64(rows)) {
        for (sbi in range64(nsb)) {
            let qo = (rr * nsb + sbi) * qsb
            let so = (rr * nsb + sbi) * 20l
            let wo = rr * n + sbi * 256l
            if (fmt == 3) {
                let d = f16_to_f32(uint(wsb[int(so) + 16]) | (uint(wsb[int(so) + 17]) << 8u))
                for (blk in range64(8l)) {
                    let shift = int(2l * (blk % 4l))
                    let vlo = int(wsb[int(so + 2l * blk)])
                    let vhi = int(wsb[int(so + 2l * blk + 1l)])
                    let slo = float(vlo >= 128 ? vlo - 256 : vlo)
                    let shi = float(vhi >= 128 ? vhi - 256 : vhi)
                    for (m in range64(16l)) {
                        let b = int(wqb[int(qo + blk * 16l + m)])
                        let hb = int(wqb[int(qo + 128l + (blk / 4l) * 32l + m)])
                        let hb2 = int(wqb[int(qo + 128l + (blk / 4l) * 32l + 16l + m)])
                        let qlo = ((b & 15) | (((hb >> shift) & 3) << 4)) - 32
                        let qhi = ((b >> 4) | (((hb2 >> shift) & 3) << 4)) - 32
                        w[int(wo + blk * 32l + m)] = d * slo * float(qlo)
                        w[int(wo + blk * 32l + 16l + m)] = d * shi * float(qhi)
                    }
                }
            } else {
                let d = f16_to_f32(uint(wsb[int(so)]) | (uint(wsb[int(so) + 1]) << 8u))
                let dmin = f16_to_f32(uint(wsb[int(so) + 2]) | (uint(wsb[int(so) + 3]) << 8u))
                for (blk in range64(8l)) {
                    let d1 = d * float(uint(wsb[int(so + 4l + blk)]))
                    let m1 = dmin * float(uint(wsb[int(so + 12l + blk)]))
                    for (m in range64(16l)) {
                        let b = int(wqb[int(qo + blk * 16l + m)])
                        var qlo = b & 15
                        var qhi = b >> 4
                        if (fmt == 2) {
                            let hb = int(wqb[int(qo + 128l + blk * 4l + m / 4l)])
                            let tt = int(m % 4l)
                            qlo |= ((hb >> tt) & 1) << 4
                            qhi |= ((hb >> (4 + tt)) & 1) << 4
                        }
                        w[int(wo + blk * 32l + m)] = d1 * float(qlo) - m1
                        w[int(wo + blk * 32l + 16l + m)] = d1 * float(qhi) - m1
                    }
                }
            }
        }
    }
}

def private fill_acts_kq(var xq : array<int8>; var xs : array<float>; nrows : int64) {
    xq |> resize(int(nrows * KN))
    xs |> resize(int(nrows * (KN / 256l)))
    for (i in range(int(nrows * KN))) {
        xq[i] = int8((int64(i) * 13l + 5l) % 255l - 127l)
    }
    for (i in range(int(nrows * (KN / 256l)))) {
        xs[i] = pow2_small(int64(i) + 4l)
    }
}

// the CPU reference of the kq FFN contract: dequantized weights against the Q8_K activation
// image (per-256 scales), silu, per-256 Q8_K requant, dequantized-down GEMM — double
// accumulation, judged by the same relative-tolerance bar as the q8 arms
def private ref_ffn_kq(offs1, offs3, offs2 : array<int64>; nreg : int;
                       w1, w3, w2 : array<float>;
                       xq : array<int8>; xs : array<float>; nrows : int64; var gout : array<float>) {
    gout |> resize(int(nrows * KN))
    let nsb = int(KN / 256l)
    let nsbh = int(KNFE / 256l)
    var gate : array<float>
    var up : array<float>
    var h : array<float>
    var hq : array<int>
    var hs : array<float>
    gate |> resize(int(KNFE))
    up |> resize(int(KNFE))
    h |> resize(int(KNFE))
    hq |> resize(int(KNFE))
    hs |> resize(nsbh)
    for (r in range(nreg)) {
        let rb1 = int((offs1[r * 3] - KWOFF1) / KN)
        let rb3 = int((offs3[r * 3] - KWOFF3) / KN)
        let rb2 = int((offs2[r * 3] - KWOFF2) / KNFE)
        let row0 = int(offs1[r * 3 + 1])
        let cnt = int(offs1[r * 3 + 2])
        for (rw in range(row0, row0 + cnt)) {
            for (i in range(int(KNFE))) {
                var ag = 0.0lf
                var au = 0.0lf
                for (sb in range(nsb)) {
                    var dg = 0.0lf
                    var du = 0.0lf
                    for (j in range(256)) {
                        let xv = double(xq[rw * int(KN) + sb * 256 + j])
                        dg += double(w1[(rb1 + i) * int(KN) + sb * 256 + j]) * xv
                        du += double(w3[(rb3 + i) * int(KN) + sb * 256 + j]) * xv
                    }
                    let xscl = double(xs[rw * nsb + sb])
                    ag += dg * xscl
                    au += du * xscl
                }
                gate[i] = float(ag)
                up[i] = float(au)
            }
            for (i in range(int(KNFE))) {
                h[i] = (gate[i] / (1.0 + exp(-gate[i]))) * up[i]
            }
            for (sb in range(nsbh)) {
                var amax = 0.0
                for (j in range(256)) {
                    amax = max(amax, abs(h[sb * 256 + j]))
                }
                let d = amax / 127.0
                let id = d != 0.0 ? 1.0 / d : 0.0
                hs[sb] = d
                for (j in range(256)) {
                    hq[sb * 256 + j] = int(round(h[sb * 256 + j] * id))
                }
            }
            for (o in range(int(KN))) {
                var y = 0.0lf
                for (sb in range(nsbh)) {
                    var dd = 0.0lf
                    for (j in range(256)) {
                        dd += double(w2[(rb2 + o) * int(KNFE) + sb * 256 + j]) * double(hq[sb * 256 + j])
                    }
                    y += dd * double(hs[sb])
                }
                gout[rw * int(KN) + o] = float(y)
            }
        }
    }
}

[test]
def test_vulkan_moe_ffn_kq(t : T?) {
    t |> run("vulkan MoE kq tier (k4 gate/up + k5/k6 down) == CPU reference") @(t : T?) {
        static_if (typeinfo builtin_module_exists(vulkan)) {
            install_moe_gpu_tier(@@vk_moe_ffn, @@vk_moe_upload_stack, @@vk_moe_cls)
            // k4 gate/up stacks + BOTH down formats in one down stack region set is not a real
            // shape — use two arms sharing the gate/up stacks: down = k6, then down = k5 at a
            // fresh offset (the 35B carries k5 downs on most layers, k6 on the last two)
            var wq1 : array<uint8>
            var ws1 : array<uint8>
            var wq3 : array<uint8>
            var ws3 : array<uint8>
            fill_kq_stack(1, NE * KNFE, KN, 21l, true, wq1, ws1)
            fill_kq_stack(1, NE * KNFE, KN, 37l, true, wq3, ws3)
            var up_ok = moe_gpu_upload_stack(wq1, ws1, KWOFF1, KN, NE * KNFE, 1)
            if (!up_ok) {
                feint("no Vulkan device (or probe/budget refused the upload); skipping\n")
                return
            }
            up_ok = moe_gpu_upload_stack(wq3, ws3, KWOFF3, KN, NE * KNFE, 1)
            t |> success(up_ok, "k4 up stack uploaded")
            var w1 : array<float>
            var w3 : array<float>
            dequant_kq_stack(1, wq1, ws1, NE * KNFE, KN, w1)
            dequant_kq_stack(1, wq3, ws3, NE * KNFE, KN, w3)
            let ege1 = KNFE * KN
            let ege2 = KN * KNFE
            var xq : array<int8>
            var xs : array<float>
            var gout : array<float>
            var refv : array<float>
            for (dfmt in [3, 2]) {   // k6 down, then k5 down (fresh stack, same gate/up)
                let dwoff = KWOFF2 + int64(3 - dfmt) * 10000000l
                var wq2 : array<uint8>
                var ws2 : array<uint8>
                fill_kq_stack(dfmt, NE * KN, KNFE, 53l + int64(dfmt), false, wq2, ws2)
                up_ok = moe_gpu_upload_stack(wq2, ws2, dwoff, KNFE, NE * KN, dfmt)
                t |> success(up_ok, "k{dfmt == 3 ? 6 : 5} down stack uploaded")
                var w2 : array<float>
                dequant_kq_stack(dfmt, wq2, ws2, NE * KN, KNFE, w2)

                // decode arm: 2 rows, 1 per region
                fill_acts_kq(xq, xs, 2l)
                var offs1 <- make_offs([1l, 2l], [0l, 1l], [1l, 1l], KWOFF1, ege1)
                var offs3 <- make_offs([1l, 2l], [0l, 1l], [1l, 1l], KWOFF3, ege1)
                var offs2 <- make_offs([1l, 2l], [0l, 1l], [1l, 1l], dwoff, ege2)
                gout |> resize(int(2l * KN))
                matmul_moe_gpu_ffn(gout, offs1, offs3, offs2, 2l, xq, xs, KN, KNFE, 2l, 1, 1, dfmt, false)
                var refoffs2 <- make_offs([1l, 2l], [0l, 1l], [1l, 1l], KWOFF2, ege2)
                ref_ffn_kq(offs1, offs3, refoffs2, 2, w1, w3, w2, xq, xs, 2l, refv)
                var c = compare_gout(gout, refv)
                t |> success(c.maxref > 0.0, "kq decode arm produced non-trivial outputs")
                t |> success(c.maxdiff <= float(REL_TOL) * c.maxref,
                    "kq decode arm (down k{dfmt == 3 ? 6 : 5}) within tolerance (maxdiff {c.maxdiff} vs {float(REL_TOL) * c.maxref} bar, maxref {c.maxref})")

                // batch arm (the kq tile): 40 rows (>= the 32-row tile, unpadded), 2 regions
                fill_acts_kq(xq, xs, 40l)
                var boffs1 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], KWOFF1, ege1)
                var boffs3 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], KWOFF3, ege1)
                var boffs2 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], dwoff, ege2)
                gout |> resize(int(40l * KN))
                matmul_moe_gpu_ffn(gout, boffs1, boffs3, boffs2, 2l, xq, xs, KN, KNFE, 40l, 1, 1, dfmt, false)
                var refboffs2 <- make_offs([0l, 2l], [0l, 24l], [24l, 16l], KWOFF2, ege2)
                ref_ffn_kq(boffs1, boffs3, refboffs2, 2, w1, w3, w2, xq, xs, 40l, refv)
                c = compare_gout(gout, refv)
                t |> success(c.maxref > 0.0, "kq batch-rows arm produced non-trivial outputs")
                t |> success(c.maxdiff <= float(REL_TOL) * c.maxref,
                    "kq batch-rows arm (down k{dfmt == 3 ? 6 : 5}) within tolerance (maxdiff {c.maxdiff} vs {float(REL_TOL) * c.maxref} bar, maxref {c.maxref})")
                delete wq2
                delete ws2
                delete w2
                delete offs1
                delete offs3
                delete offs2
                delete refoffs2
                delete boffs1
                delete boffs3
                delete boffs2
                delete refboffs2
            }

            // classifier arm (the head offload's dispatch): one whole-plane k6 GEMV — ONE
            // region covering every row, device meta + pre-recorded cmd. Two passes with
            // perturbed activations exercise the recorded-resubmit path.
            let CWOFF = 9000000l
            let CVOCAB = 512l
            var wqc : array<uint8>
            var wsc : array<uint8>
            fill_kq_stack(3, CVOCAB, KN, 71l, false, wqc, wsc)
            up_ok = moe_gpu_upload_stack(wqc, wsc, CWOFF, KN, CVOCAB, 3)
            t |> success(up_ok, "k6 classifier plane uploaded")
            var wc : array<float>
            dequant_kq_stack(3, wqc, wsc, CVOCAB, KN, wc)
            var ylog : array<float>
            ylog |> resize(int(CVOCAB))
            let nsbc = int(KN / 256l)
            for (cpass in range(2)) {
                fill_acts_kq(xq, xs, 1l)
                xq[0] = int8(int(xq[0]) + cpass * 3)
                matmul_moe_gpu_cls(ylog, CWOFF, xq, xs, KN, CVOCAB, 3)
                var yref : array<float>
                yref |> resize(int(CVOCAB))
                for (r in range(int(CVOCAB))) {
                    var acc = 0.0lf
                    for (sb in range(nsbc)) {
                        var dd = 0.0lf
                        for (j in range(256)) {
                            dd += double(wc[r * int(KN) + sb * 256 + j]) * double(xq[sb * 256 + j])
                        }
                        acc += dd * double(xs[sb])
                    }
                    yref[r] = float(acc)
                }
                let cc = compare_gout(ylog, yref)
                t |> success(cc.maxref > 0.0, "cls arm pass {cpass} produced non-trivial outputs")
                t |> success(cc.maxdiff <= float(REL_TOL) * cc.maxref,
                    "cls GEMV arm pass {cpass} within tolerance (maxdiff {cc.maxdiff} vs {float(REL_TOL) * cc.maxref} bar, maxref {cc.maxref})")
                delete yref
            }
            delete wqc
            delete wsc
            delete wc
            delete ylog
            delete wq1
            delete ws1
            delete wq3
            delete ws3
            delete w1
            delete w3
            delete xq
            delete xs
            delete gout
            delete refv
        } else {
            feint("dasVulkan not installed; vulkan MoE tier untestable here\n")
        }
    }
}
