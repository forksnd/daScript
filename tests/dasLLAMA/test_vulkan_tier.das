options gen2
options stack = 524288

require dastest/testing_boost public
require dasllama/dasllama_math
require ?vulkan dasllama/dasllama_math_vulkan  // nolint:STYLE030 — the MoE GPU tier under test; the test installs its hooks explicitly (no env gate)
require llvm/daslib/f16_cvt
require math

// The Vulkan MoE GPU tier (dasllama_math_vulkan) vs a hand-rolled CPU reference of the same
// fused FFN contract: region-triple gate+up GEMMs -> silu(g)*u -> q8_0 requant -> down GEMM,
// on synthetic Q8 stacks uploaded through the public hook surface (install_moe_gpu_tier /
// moe_gpu_upload_stack / matmul_q8q8_moe_gpu_ffn). Both dispatch arms run: decode GEMV
// (nrows < 32) and the tiled prefill batch (nrows >= 32, deliberately unpadded). Weight scales
// are powers of two (f16-exact), so the only drift left is the GPU's exp() in silu and
// round-halfway splits in the requant — the compare is relative-tolerance, not bit-exact.
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
            install_moe_gpu_tier(@@vk_moe_ffn, @@vk_moe_upload_stack)
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
            var up_ok = moe_gpu_upload_stack(wqb, wsb, WOFF1, N, NE * NFE)
            if (!up_ok) {
                feint("no Vulkan device (or probe/budget refused the upload); skipping\n")
                return
            }
            pack_upload(wq3, ws3, wqb, wsb)
            up_ok = moe_gpu_upload_stack(wqb, wsb, WOFF3, N, NE * NFE)
            t |> success(up_ok, "up stack uploaded")
            pack_upload(wq2, ws2, wqb, wsb)
            up_ok = moe_gpu_upload_stack(wqb, wsb, WOFF2, NFE, NE * N)
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
            matmul_q8q8_moe_gpu_ffn(gout, offs1, offs3, offs2, 2l, xq, xs, N, NFE, 2l, false)
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
            matmul_q8q8_moe_gpu_ffn(gout, boffs1, boffs3, boffs2, 2l, xq, xs, N, NFE, 40l, false)
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
