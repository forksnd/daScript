options gen2

module dasllama_math_vulkan shared public

require dasllama/dasllama_math
require vulkan
require vulkan/vulkan_boost
require spirv/spirv_shader
require spirv/spirv_builtins public
require strings
require math
require daslib/fio

// dasLLAMA Vulkan GPU tier — per-layer MoE expert-stack offload. The loader gathers offloaded
// layers' PREPARED q8 planes into row-major device planes (int8 quant words + f16 scales) and
// hands them to vk_moe_upload_stack (moe_gpu_upload_resident — every load rail, gguf and mapped
// image alike); the MoE dispatches then route those layers' whole expert FFN chain
// (gate+up GEMMs -> act -> q8 requant -> down) through vk_moe_ffn — decode row counts on the
// GEMV kernel, prefill batches on the tiled GEMM. Needs 8/16-bit SSBO storage + compute
// subgroups (probed; absent = tier off, CPU fallback).
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
// CONSECUTIVE bytes — one transaction. The block's integer dot stays exact via a 3-step
// shuffleXor fold across the 8 lanes; lane 0 of each cluster applies the f16*f32 scales.
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
                let w4 = int4(unpack8(int(vk_wq[(wb0 + b) * 8u + my_word])))
                let x4 = int4(unpack8(int(vk_xq[(xb0 + b) * 8u + my_word])))
                idot = w4.x * x4.x + w4.y * x4.y + w4.z * x4.z + w4.w * x4.w
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
// Same exact int block dots + fp32 block-order accumulation as the decode kernel. vk_meta:
// [0]=n [1]=d [2]=nregions [3]=map_off, per region (rel_wblock, row0, cnt, wg_base) quads,
// then the per-workgroup region map at map_off (workgroup id -> region index — no search).
// Activations read from device-local scratch (bindings 3/4 rebind) — a host-visible image
// re-read ~(d/32) times would cross PCIe every pass.

var @workgroup bt_xw : uint[256]     // x tile: 32 rows x 8 words (one 32-weight block per row)
var @workgroup bt_ww : uint[288]     // w tile: 32 cols x 8 words, stride 9 — stride 8 lands all
                                     // 8 lanes of a warp on 4 banks (8-way conflict on the hot read)
var @workgroup bt_xs : float[32]
var @workgroup bt_ws : float[32]

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
    let tid = gl_LocalInvocationID.x
    let word = tid % 8u
    let grp = tid / 8u               // 0..31: the x row AND w col this thread stages / computes
    let xrow = xt * 32u + grp
    let wcol = wt * 32u + grp
    let wi0 = (tid % 8u) * 4u        // first of the 4 w cols this thread accumulates
    var f0 = 0.0
    var f1 = 0.0
    var f2 = 0.0
    var f3 = 0.0
    var b = 0u
    while (b < nbb) {
        var xv = 0u
        if (xrow < cnt) {
            xv = vk_xq[((row0 + xrow) * nbb + b) * 8u + word]
        }
        bt_xw[tid] = xv
        var wv = 0u
        if (wcol < d) {
            wv = vk_wq[(wblk0 + wcol * nbb + b) * 8u + word]
        }
        bt_ww[grp * 9u + word] = wv
        if (tid < 32u) {
            var xs = 0.0
            if (xt * 32u + tid < cnt) {
                xs = vk_xs[(row0 + xt * 32u + tid) * nbb + b]
            }
            bt_xs[tid] = xs
        } elif (tid < 64u) {
            let i = tid - 32u
            var ws = 0.0
            if (wt * 32u + i < d) {
                ws = float(vk_ws[wblk0 + (wt * 32u + i) * nbb + b])
            }
            bt_ws[i] = ws
        }
        barrier()
        var i0 = 0
        var i1 = 0
        var i2 = 0
        var i3 = 0
        for (j in range(8)) {
            let ju = uint(j)
            let xw = int4(unpack8(int(bt_xw[grp * 8u + ju])))
            let w0 = int4(unpack8(int(bt_ww[(wi0 + 0u) * 9u + ju])))
            let w1 = int4(unpack8(int(bt_ww[(wi0 + 1u) * 9u + ju])))
            let w2 = int4(unpack8(int(bt_ww[(wi0 + 2u) * 9u + ju])))
            let w3 = int4(unpack8(int(bt_ww[(wi0 + 3u) * 9u + ju])))
            i0 += w0.x * xw.x + w0.y * xw.y + w0.z * xw.z + w0.w * xw.w
            i1 += w1.x * xw.x + w1.y * xw.y + w1.z * xw.z + w1.w * xw.w
            i2 += w2.x * xw.x + w2.y * xw.y + w2.z * xw.z + w2.w * xw.w
            i3 += w3.x * xw.x + w3.y * xw.y + w3.z * xw.z + w3.w * xw.w
        }
        let xscl = bt_xs[grp]
        f0 += float(i0) * bt_ws[wi0] * xscl
        f1 += float(i1) * bt_ws[wi0 + 1u] * xscl
        f2 += float(i2) * bt_ws[wi0 + 2u] * xscl
        f3 += float(i3) * bt_ws[wi0 + 3u] * xscl
        barrier()
        b++
    }
    if (xrow < cnt) {
        let yb = (row0 + xrow) * d + wt * 32u + wi0
        if (wt * 32u + wi0 < d) {
            vk_y[yb] = f0
        }
        if (wt * 32u + wi0 + 1u < d) {
            vk_y[yb + 1u] = f1
        }
        if (wt * 32u + wi0 + 2u < d) {
            vk_y[yb + 2u] = f2
        }
        if (wt * 32u + wi0 + 3u < d) {
            vk_y[yb + 3u] = f3
        }
    }
}

// ===== the FFN mid-chain kernels =====
// The act/requant steps between the gate+up and down GEMMs, so the whole MoE FFN is one submit
// and the hidden activations never cross PCIe. Only the globals a kernel references are emitted
// into its SPIR-V, so binding 3 doubles as a float view (vk_up) here while the GEMMs see uints.
// vk_meta (per FFN dispatch): [0] = element count, [1] = gelu flag, [2] = 32-block count.

var @ssbo @binding = 3 vk_up : array<float>

// elementwise g = act(g) * u over the gate/up planes (g in vk_y, u in vk_up). Same formulas as
// the CPU swiglu/geglu cores; exp/tanh precision differs from libm (the parity bar is
// token-identical, not bit-identical).
[compute_shader(local_size_x=256, name="q8_moe_act_spv")]
def q8_moe_act {
    let i = gl_GlobalInvocationID.x
    if (i < vk_meta[0u]) {
        let v = vk_y[i]
        let u = vk_up[i]
        if (vk_meta[1u] != 0u) {
            let kg = 0.7978845608028654   // sqrt(2/pi), the CPU geglu tanh-approx constant
            vk_y[i] = (0.5 * v * (1.0 + tanh(kg * (v + 0.044715 * v * v * v)))) * u
        } else {
            vk_y[i] = (v / (1.0 + exp(-v))) * u
        }
    }
}

// q8_0 requant of the activated hidden rows (vk_y floats -> vk_xq quant words + vk_xs scales),
// one thread per 32-block, mirroring quantize_q8_0_into_ptr: |x| amax -> d = amax/127 ->
// q = round(x/d). GPU round() may split rounding-halfway cases differently — same parity bar.
[compute_shader(local_size_x=256, name="q8_moe_requant_spv")]
def q8_moe_requant {
    let b = gl_GlobalInvocationID.x
    if (b < vk_meta[2u]) {
        let base = b * 32u
        var amax = 0.0
        for (i in range(32)) {
            amax = max(amax, abs(vk_y[base + uint(i)]))
        }
        let d = amax / 127.0
        let id = d != 0.0 ? 1.0 / d : 0.0
        vk_xs[b] = d
        for (w in range(8)) {
            let wb = base + uint(w) * 4u
            var word = 0u
            for (j in range(4)) {
                let q = int(round(vk_y[wb + uint(j)] * id))
                word = word | ((uint(q) & 0xFF) << (uint(j) * 8u))
            }
            vk_xq[b * 8u + uint(w)] = word
        }
    }
}

// ===== device state (process-lifetime; never finalized by design) =====

let private VRAM_BUDGET = 5_500_000_000l   // resident-weight cap: under the measured ~6.5GB WDDM paging cliff
let private STAGING_BYTES = 268_435_456l   // one upload chunk (largest single memcpy)
let private XQ_BYTES = 1_048_576l          // per-stack activation quants capacity
let private XS_BYTES = 262_144l            // ... their scales
let private Y_BYTES = 1_048_576l           // per-stack output capacity
let private META_BYTES = 4_096l            // params + region table
let private MAX_STACKS = 64                // descriptor pool capacity (VRAM budget caps far lower)
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

struct private HostBuf {
    buf : uint64       // VkBuffer
    mem : uint64       // VkDeviceMemory
    mapped : void?     // persistently mapped (host-coherent)
    bytes : int64
}

struct private VkStack {
    base : int64             // first weight-element offset covered (into the q8 plane)
    elems : int64            // element count (offs resolve against [base, base+elems))
    wqbuf : uint64           // device-local row-major quants (bound as array<uint>)
    wsbuf : uint64           // device-local f16 scale plane (bound as array<float16>)
    wsbytes : int64
    raw_set : VkDescriptorSet   // this stack's 6 bindings, written once at upload
    meta : HostBuf
    xq : HostBuf
    xs : HostBuf
    y : HostBuf              // host-cached readback staging (the same-submit DMA copy lands here)
    y_dev : uint64           // device-local y the kernel writes — see batch_y1_dev on why
    batch_meta : HostBuf     // prefill batch meta STAGING (lazy — first batch dispatch creates it)
    batch_meta_dev : uint64  // device copy the kernel reads — a host-visible meta costs ~26us of
                             // serialized uncached PCIe reads PER WORKGROUP (measured, 12K wgs/dispatch)
    batch_set : VkDescriptorSet   // batch bindings: stack planes + act-image scratch + a y plane
    batch_y : uint64         // which shared y plane batch_set binds (role-fixed per stack)
    batch_xq : uint64        // which activation-quant plane it binds (gate/up: gathered x; down: hq)
    batch_made : bool
}

// pre-recorded decode-form FFN chain per (gate, up, down) stack triple: GEMV pair, act, requant,
// down GEMV, y DMA — refreshed per token via the stacks' host-visible metas, resubmitted as-is
struct private FfnCmd {
    cmd : VkCommandBuffer
    act_set : VkDescriptorSet     // gate y_dev (rw) + up y_dev + ffn meta
    rq_set : VkDescriptorSet      // gate y_dev -> hq/hs device planes
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
    act_pipeline : Pipeline     // elementwise act(g)*u over the gate/up planes
    rq_pipeline : Pipeline      // q8_0 requant of the activated hidden rows
    staging : HostBuf
    fence : VkFence          // the one reusable submit fence (reset after every wait)
    rows_per_wg : int64      // subgroups per workgroup = output rows per workgroup
    @do_not_delete stacks : array<VkStack>
    ffn_cmds : table<int64; FfnCmd>   // (gate, up, down stack idx triple) -> decode FFN chain
    resident_bytes : int64
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
    batch_act_set : VkDescriptorSet   // batch twins of FfnCmd's act/requant sets (y1/y2 planes)
    batch_rq_set : VkDescriptorSet
    batch_cmd : VkCommandBuffer   // one reusable batch cmd buffer, re-recorded per submit
    batch_ready : bool
}

var private g_gpu : GpuState?
var private g_init_failed = false

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
    return b._vk
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
    if (!storage_8_16_supported(g_gpu.phys) || !subgroup_compute_ops_supported(g_gpu.phys)) {
        to_log(LOG_ERROR, "dasLLAMA vulkan tier: device lacks 8/16-bit SSBO storage or compute subgroup ops — tier disabled\n")
        return false
    }
    g_gpu.fam = select_graphics_queue_family(g_gpu.phys)
    g_gpu.device <- create_device_storage_8_16(g_gpu.phys, g_gpu.fam)
    volkLoadDevice(dev_raw())
    g_gpu.queue = get_device_queue(g_gpu.device, g_gpu.fam, 0u)
    let sg = subgroup_properties(g_gpu.phys).subgroupSize
    g_gpu.rows_per_wg = int64((WG_X + sg - 1u) / sg)   // = gl_NumSubgroups for our 1D workgroup
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
    var ashader <- create_shader_module(g_gpu.device, q8_moe_act_spv)
    g_gpu.act_pipeline <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, ashader)
    var rshader <- create_shader_module(g_gpu.device, q8_moe_requant_spv)
    g_gpu.rq_pipeline <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, rshader)

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

//! MoeGpuUploadFn: upload one Q8 stack ([rows x n] at element offset woff) from the loader's
//! gathered row-major planes (int8 quants + f16 halfword scales).
def vk_moe_upload_stack(wq : uint8 const?; ws : uint8 const?; woff : int64; n : int64; rows : int64) : bool {
    if (!vk_moe_init()) {
        return false
    }
    let wqbytes = rows * n
    let wsbytes = rows * (n / 32l) * 2l
    if (g_gpu.resident_bytes + wqbytes + wsbytes > VRAM_BUDGET) {
        to_log(LOG_INFO, "dasLLAMA vulkan tier: VRAM budget reached at {g_gpu.resident_bytes} bytes\n")
        return false
    }
    if (length(g_gpu.stacks) >= MAX_STACKS) {
        to_log(LOG_INFO, "dasLLAMA vulkan tier: descriptor capacity reached at {length(g_gpu.stacks)} stacks\n")
        return false
    }
    var st = VkStack(base = woff, elems = rows * n,
                     wqbuf = make_device_buf(wqbytes), wsbuf = make_device_buf(wsbytes), wsbytes = wsbytes,
                     raw_set = alloc_desc_set(),
                     meta = make_host_buf(META_BYTES, true), xq = make_host_buf(XQ_BYTES, true),
                     xs = make_host_buf(XS_BYTES, true), y = make_host_buf(Y_BYTES, false, [cached = true]),
                     y_dev = make_device_buf(Y_BYTES))
    upload_region(st.wqbuf, wq, wqbytes)
    upload_region(st.wsbuf, ws, wsbytes)
    var writes : array<WriteDescriptorSet>
    write_buf_desc(writes, st.raw_set, 0u, st.wqbuf, wqbytes)
    write_buf_desc(writes, st.raw_set, 1u, st.wsbuf, wsbytes)
    write_buf_desc(writes, st.raw_set, 2u, st.meta.buf, META_BYTES)
    write_buf_desc(writes, st.raw_set, 3u, st.xq.buf, XQ_BYTES)
    write_buf_desc(writes, st.raw_set, 4u, st.xs.buf, XS_BYTES)
    write_buf_desc(writes, st.raw_set, 5u, st.y_dev, Y_BYTES)
    let no_copies : array<CopyDescriptorSet>
    update_descriptor_sets(g_gpu.device, writes, no_copies)
    g_gpu.stacks |> push(st)
    g_gpu.resident_bytes += wqbytes + wsbytes
    return true
}

// ===== dispatch plumbing =====

def private find_stack(w0 : int64) : int {
    for (i in range(length(g_gpu.stacks))) {
        if (w0 >= g_gpu.stacks[i].base && w0 < g_gpu.stacks[i].base + g_gpu.stacks[i].elems) {
            return i
        }
    }
    panic("dasLLAMA vulkan tier: dispatch against a non-resident weight region")
    return -1
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
// (rel_wblock, xblock)*] as 32-block offsets; xnb = activation blocks per image row
def private fill_stack_meta_rows(si : int; offs : int64 const?; nregions : int64; n, d, nrows, xnb : int64) {
    assert((4l + nrows * 2l) * 4l <= META_BYTES, "dasLLAMA vulkan tier: FFN meta exceeds capacity")
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
            assert(rel % 32l == 0l, "dasLLAMA vulkan tier: unaligned region offset")
            for (i in range64(cnt)) {
                mp[4l + (row0 + i) * 2l] = uint(rel / 32l)
                mp[5l + (row0 + i) * 2l] = uint((row0 + i) * xnb)
            }
        }
    }
}

def private fill_stack_acts(si : int; xqp : int8 const?; xsp : float const?; xq_bytes : int64) {
    assert(xq_bytes <= XQ_BYTES, "dasLLAMA vulkan tier: dispatch exceeds scratch capacity")
    unsafe {
        memcpy(g_gpu.stacks[si].xq.mapped, reinterpret<void? -const>(xqp), int(xq_bytes))
        memcpy(g_gpu.stacks[si].xs.mapped, reinterpret<void? -const>(xsp), int((xq_bytes / 32l) * 4l))
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
    g_gpu.batch_act_set = alloc_desc_set()
    g_gpu.batch_rq_set = alloc_desc_set()
    var writes : array<WriteDescriptorSet>
    // act: gate plane (rw) at 5, up plane (float view) at 3, params at 2; 0/1/4 unused fillers
    write_buf_desc(writes, g_gpu.batch_act_set, 0u, g_gpu.batch_xq_dev, BATCH_XQ_BYTES)
    write_buf_desc(writes, g_gpu.batch_act_set, 1u, g_gpu.batch_xs_dev, BATCH_XS_BYTES)
    write_buf_desc(writes, g_gpu.batch_act_set, 2u, g_gpu.ffn_meta_dev, FFN_META_BYTES)
    write_buf_desc(writes, g_gpu.batch_act_set, 3u, g_gpu.batch_y2_dev, BATCH_Y_BYTES)
    write_buf_desc(writes, g_gpu.batch_act_set, 4u, g_gpu.batch_xs_dev, BATCH_XS_BYTES)
    write_buf_desc(writes, g_gpu.batch_act_set, 5u, g_gpu.batch_y1_dev, BATCH_Y_BYTES)
    // requant: hidden floats at 5 -> hq/hs planes at 3/4
    write_buf_desc(writes, g_gpu.batch_rq_set, 0u, g_gpu.batch_xq_dev, BATCH_XQ_BYTES)
    write_buf_desc(writes, g_gpu.batch_rq_set, 1u, g_gpu.batch_xs_dev, BATCH_XS_BYTES)
    write_buf_desc(writes, g_gpu.batch_rq_set, 2u, g_gpu.ffn_meta_dev, FFN_META_BYTES)
    write_buf_desc(writes, g_gpu.batch_rq_set, 3u, g_gpu.hq_dev, BATCH_HQ_BYTES)
    write_buf_desc(writes, g_gpu.batch_rq_set, 4u, g_gpu.hs_dev, BATCH_HS_BYTES)
    write_buf_desc(writes, g_gpu.batch_rq_set, 5u, g_gpu.batch_y1_dev, BATCH_Y_BYTES)
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
    write_buf_desc(writes, g_gpu.stacks[si].batch_set, 0u, g_gpu.stacks[si].wqbuf, g_gpu.stacks[si].elems)
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
// output (which reuses y1) must all fit their buffers
def private ffn_chunk_rows(n, nfe : int64) : int64 {
    var r = min(BATCH_XQ_BYTES / n, BATCH_XS_BYTES / ((n / 32l) * 4l))
    r = min(r, BATCH_Y_BYTES / (max(nfe, n) * 4l))
    r = min(r, min(BATCH_HQ_BYTES / nfe, BATCH_HS_BYTES / ((nfe / 32l) * 4l)))
    return min(r, BATCH_CHUNK_ROWS)
}

// stage the chunk's activation rows [r0, r0+rows) into the device-local scratch (the kernel
// re-reads x ~(d/32) times — host-visible would cross PCIe every pass)
def private upload_batch_acts(xqp : int8 const?; xsp : float const?; r0, rows, n : int64) {
    let xq_bytes = rows * n
    let xs_bytes = rows * (n / 32l) * 4l
    unsafe {
        var sp = reinterpret<uint8?>(g_gpu.staging.mapped)
        memcpy(g_gpu.staging.mapped, reinterpret<void? -const>(xqp + r0 * n), int(xq_bytes))
        memcpy(reinterpret<void?>(sp + BATCH_XQ_BYTES), reinterpret<void? -const>(xsp + r0 * (n / 32l)), int(xs_bytes))
    }
    run_cmd_sync(g_gpu.device, g_gpu.pool, g_gpu.queue) $(cmd) {
        var qregions : array<BufferCopy>
        qregions |> push(BufferCopy(srcOffset = 0ul, dstOffset = 0ul, size = uint64(xq_bytes)))
        cmd_copy_buffer(cmd, nonowning_buf(g_gpu.staging.buf), nonowning_buf(g_gpu.batch_xq_dev), qregions)
        var sregions : array<BufferCopy>
        sregions |> push(BufferCopy(srcOffset = uint64(BATCH_XQ_BYTES), dstOffset = 0ul, size = uint64(xs_bytes)))
        cmd_copy_buffer(cmd, nonowning_buf(g_gpu.staging.buf), nonowning_buf(g_gpu.batch_xs_dev), sregions)
    }
}

// build the chunk's meta table: regions clipped to the row window [r0, r1) (row0 re-based to the
// window), then the per-workgroup region map. Returns the workgroup count.
def private fill_batch_meta(si : int; offs : int64 const?; nregions : int64; n, d : int64; r0, r1 : int64) : uint {
    let wtiles = (d + BATCH_TILE - 1l) / BATCH_TILE
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
            assert(rel % 32l == 0l, "dasLLAMA vulkan tier: unaligned batch region weight offset")
            let mb = 4l + ri * 4l
            mp[mb] = uint(rel / 32l)
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
// meta staging copies, gate+up GEMMs, act, requant, down GEMM, gout DMA — one barrier per seam,
// the hidden planes never leave the device.
def private record_ffn_batch_cmd(s1, s3, s2 : int; nwg_gu, nwg_dn : uint; rows, nfe, gout_bytes : int64) {
    let raw = g_gpu.batch_cmd
    let rf : VkCommandBufferResetFlags
    vk_check(vkResetCommandBuffer(raw, rf), null)
    let begin = VkCommandBufferBeginInfo()
    vk_check(vkBeginCommandBuffer(raw, begin), null)
    cmd_copy_batch_meta(raw, s1)
    cmd_copy_batch_meta(raw, s3)
    cmd_copy_batch_meta(raw, s2)
    cmd_copy_whole(raw, g_gpu.ffn_meta.buf, g_gpu.ffn_meta_dev, FFN_META_BYTES)
    batch_barrier(raw, true)
    cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.batch_pipeline, VkPipelineBindPoint.COMPUTE)
    cmd_bind_dispatch(raw, g_gpu.stacks[s1].batch_set, nwg_gu)
    cmd_bind_dispatch(raw, g_gpu.stacks[s3].batch_set, nwg_gu)
    comp_barrier(raw)
    cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.act_pipeline, VkPipelineBindPoint.COMPUTE)
    cmd_bind_dispatch(raw, g_gpu.batch_act_set, uint((rows * nfe + 255l) / 256l))
    comp_barrier(raw)
    cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.rq_pipeline, VkPipelineBindPoint.COMPUTE)
    cmd_bind_dispatch(raw, g_gpu.batch_rq_set, uint((rows * nfe / 32l + 255l) / 256l))
    comp_barrier(raw)
    cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.batch_pipeline, VkPipelineBindPoint.COMPUTE)
    cmd_bind_dispatch(raw, g_gpu.stacks[s2].batch_set, nwg_dn)
    batch_barrier(raw, false)
    cmd_copy_whole(raw, g_gpu.batch_y1_dev, g_gpu.batch_y1.buf, gout_bytes)
    vk_check(vkEndCommandBuffer(raw), null)
}

// the batch arm: chunked over row windows; gate+up share the gathered image and region row
// ranges, the down GEMM reads the requantized hidden planes and reuses y1 for its output
def private vk_moe_ffn_batch(var goutp : float?; offs1 : int64 const?; offs3 : int64 const?; offs2 : int64 const?; nregions : int64; xqp : int8 const?; xsp : float const?; n, nfe, nrows : int64; is_gelu : bool) {
    let s1 = find_stack(unsafe(offs1[0]))
    let s3 = find_stack(unsafe(offs3[0]))
    let s2 = find_stack(unsafe(offs2[0]))
    ensure_batch_state()
    ensure_batch_set(s1, g_gpu.batch_y1_dev, g_gpu.batch_xq_dev, g_gpu.batch_xs_dev, BATCH_XQ_BYTES, BATCH_XS_BYTES)
    ensure_batch_set(s3, g_gpu.batch_y2_dev, g_gpu.batch_xq_dev, g_gpu.batch_xs_dev, BATCH_XQ_BYTES, BATCH_XS_BYTES)
    ensure_batch_set(s2, g_gpu.batch_y1_dev, g_gpu.hq_dev, g_gpu.hs_dev, BATCH_HQ_BYTES, BATCH_HS_BYTES)
    let chunk = ffn_chunk_rows(n, nfe)
    var r0 = 0l
    while (r0 < nrows) {
        let r1 = min(r0 + chunk, nrows)
        let rows = r1 - r0
        let nwg1 = fill_batch_meta(s1, offs1, nregions, n, nfe, r0, r1)
        let nwg3 = fill_batch_meta(s3, offs3, nregions, n, nfe, r0, r1)
        assert(nwg1 == nwg3, "dasLLAMA vulkan tier: fused batch pair region mismatch")
        let nwg2 = fill_batch_meta(s2, offs2, nregions, nfe, n, r0, r1)
        upload_batch_acts(xqp, xsp, r0, rows, n)
        fill_ffn_meta(rows * nfe, rows * nfe / 32l, is_gelu)
        record_ffn_batch_cmd(s1, s3, s2, nwg1, nwg2, rows, nfe, rows * n * 4l)
        submit_wait(g_gpu.batch_cmd)
        unsafe {
            memcpy(reinterpret<void?>(goutp + r0 * n), g_gpu.batch_y1.mapped, int(rows * n * 4l))
        }
        r0 = r1
    }
}

// ===== decode dispatch (the FFN's GEMV arm) =====

// pre-record (or refresh on a row-count change) the triple's chain: gate+up GEMV pair, act,
// requant, down GEMV, y DMA. Per token only the stacks' host-visible metas, the activation
// image, and the ffn params staging change — the recorded buffer resubmits as-is.
def private ffn_cmd_for(s1, s3, s2 : int; nrows, n, nfe : int64) : VkCommandBuffer {
    let key = (int64(s1) * int64(MAX_STACKS) + int64(s3)) * int64(MAX_STACKS) + int64(s2)
    if (!key_exists(g_gpu.ffn_cmds, key)) {
        var fc = FfnCmd(cmd = alloc_cmd(), act_set = alloc_desc_set(), rq_set = alloc_desc_set(),
                        down_set = alloc_desc_set(), nrows = 0l)
        var writes : array<WriteDescriptorSet>
        // act: gate y (rw) at 5, up y (float view) at 3, params at 2; 0/1/4 unused fillers
        write_buf_desc(writes, fc.act_set, 0u, g_gpu.stacks[s1].wqbuf, g_gpu.stacks[s1].elems)
        write_buf_desc(writes, fc.act_set, 1u, g_gpu.stacks[s1].wsbuf, g_gpu.stacks[s1].wsbytes)
        write_buf_desc(writes, fc.act_set, 2u, g_gpu.ffn_meta_dev, FFN_META_BYTES)
        write_buf_desc(writes, fc.act_set, 3u, g_gpu.stacks[s3].y_dev, Y_BYTES)
        write_buf_desc(writes, fc.act_set, 4u, g_gpu.stacks[s1].xs.buf, XS_BYTES)
        write_buf_desc(writes, fc.act_set, 5u, g_gpu.stacks[s1].y_dev, Y_BYTES)
        // requant: hidden floats at 5 -> hq/hs planes at 3/4
        write_buf_desc(writes, fc.rq_set, 0u, g_gpu.stacks[s1].wqbuf, g_gpu.stacks[s1].elems)
        write_buf_desc(writes, fc.rq_set, 1u, g_gpu.stacks[s1].wsbuf, g_gpu.stacks[s1].wsbytes)
        write_buf_desc(writes, fc.rq_set, 2u, g_gpu.ffn_meta_dev, FFN_META_BYTES)
        write_buf_desc(writes, fc.rq_set, 3u, g_gpu.hq_dev, BATCH_HQ_BYTES)
        write_buf_desc(writes, fc.rq_set, 4u, g_gpu.hs_dev, BATCH_HS_BYTES)
        write_buf_desc(writes, fc.rq_set, 5u, g_gpu.stacks[s1].y_dev, Y_BYTES)
        // down: its own planes + host-visible meta, hq/hs as the activation image, its y_dev
        write_buf_desc(writes, fc.down_set, 0u, g_gpu.stacks[s2].wqbuf, g_gpu.stacks[s2].elems)
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
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.pipeline, VkPipelineBindPoint.COMPUTE)
        let ggu = uint((nrows * nfe + g_gpu.rows_per_wg - 1l) / g_gpu.rows_per_wg)
        cmd_bind_dispatch(raw, g_gpu.stacks[s1].raw_set, ggu)
        cmd_bind_dispatch(raw, g_gpu.stacks[s3].raw_set, ggu)
        comp_barrier(raw)
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.act_pipeline, VkPipelineBindPoint.COMPUTE)
        cmd_bind_dispatch(raw, fc.act_set, uint((nrows * nfe + 255l) / 256l))
        comp_barrier(raw)
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.rq_pipeline, VkPipelineBindPoint.COMPUTE)
        cmd_bind_dispatch(raw, fc.rq_set, uint((nrows * nfe / 32l + 255l) / 256l))
        comp_barrier(raw)
        cmd_bind_pipeline(vk_value_to_boost(raw), g_gpu.pipeline, VkPipelineBindPoint.COMPUTE)
        cmd_bind_dispatch(raw, fc.down_set, uint((nrows * n + g_gpu.rows_per_wg - 1l) / g_gpu.rows_per_wg))
        batch_barrier(raw, false)
        cmd_copy_whole(raw, g_gpu.stacks[s2].y_dev, g_gpu.stacks[s2].y.buf, nrows * n * 4l)
        vk_check(vkEndCommandBuffer(raw), null)
        g_gpu.ffn_cmds[key] = FfnCmd(cmd = fc.cmd, act_set = fc.act_set, rq_set = fc.rq_set,
            down_set = fc.down_set, nrows = nrows)
    }
    return fc.cmd
}

def private vk_moe_ffn_gemv(var goutp : float?; offs1 : int64 const?; offs3 : int64 const?; offs2 : int64 const?; nregions : int64; xqp : int8 const?; xsp : float const?; n, nfe, nrows : int64; is_gelu : bool) {
    let s1 = find_stack(unsafe(offs1[0]))
    let s3 = find_stack(unsafe(offs3[0]))
    let s2 = find_stack(unsafe(offs2[0]))
    assert(nrows * max(nfe, n) * 4l <= Y_BYTES, "dasLLAMA vulkan tier: FFN dispatch exceeds output capacity")
    fill_stack_meta_rows(s1, offs1, nregions, n, nfe, nrows, n / 32l)
    fill_stack_meta_rows(s3, offs3, nregions, n, nfe, nrows, n / 32l)
    fill_stack_meta_rows(s2, offs2, nregions, nfe, n, nrows, nfe / 32l)
    fill_stack_acts(s1, xqp, xsp, nrows * n)
    fill_stack_acts(s3, xqp, xsp, nrows * n)
    fill_ffn_meta(nrows * nfe, nrows * nfe / 32l, is_gelu)
    var cmd = ffn_cmd_for(s1, s3, s2, nrows, n, nfe)
    submit_wait(cmd)
    unsafe {
        memcpy(reinterpret<void?>(goutp), g_gpu.stacks[s2].y.mapped, int(nrows * n * 4l))
    }
}

//! MoeGpuFfnFn: the routed experts' whole FFN chain — gate+up region GEMMs, activation, q8
//! requant, down GEMM — as device-side submit chains over the resident stacks; the hidden
//! activations never cross PCIe. offs are region triples sharing row ranges that partition
//! [0, nrows); gout returns nrows x n floats. Decode row counts run the GEMV kernels
//! (pre-recorded per stack triple), batches run the tiled-GEMM chunk loop.
def vk_moe_ffn(var goutp : float?; offs1 : int64 const?; offs3 : int64 const?; offs2 : int64 const?; nregions : int64; xqp : int8 const?; xsp : float const?; n, nfe, nrows : int64; is_gelu : bool) {
    ensure_ffn_state()
    if (nrows < BATCH_TILE) {
        vk_moe_ffn_gemv(goutp, offs1, offs3, offs2, nregions, xqp, xsp, n, nfe, nrows, is_gelu)
    } else {
        vk_moe_ffn_batch(goutp, offs1, offs3, offs2, nregions, xqp, xsp, n, nfe, nrows, is_gelu)
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
    if (has_env_variable("DASLLAMA_GPU_MOE_LAYERS")) {
        let layers = int64(to_int(get_env_variable("DASLLAMA_GPU_MOE_LAYERS")))
        if (layers > 0l && vulkan_backend_available()) {
            set_moe_gpu_layer_count(layers)
            install_moe_gpu_tier(@@vk_moe_ffn, @@vk_moe_upload_stack)
        }
    }
}
