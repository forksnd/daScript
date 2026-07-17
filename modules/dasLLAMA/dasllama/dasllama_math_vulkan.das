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

// dasLLAMA Vulkan GPU tier — stage-1a: per-layer MoE expert-stack offload (decode GEMV).
// The loader uploads offloaded layers' row-major fp32-scale Q8 stacks via vk_moe_upload_stack
// (pre-repack, so the bytes are layout-stable); moe_experts_apply then routes those layers'
// region-list GEMVs through vk_moe_groupn. v1 submission is deliberately naive — one
// run_cmd_sync (queue-drain) per stage — measure first, ring later. Enabled by
// DASLLAMA_GPU_MOE_LAYERS=<K> (expert stacks of the LAST K layers go resident).
//
// The registered "vulkan" KernelBackend below stays a named-only stub (priority -1, GEMV slots
// panic) — the MoE tier hooks are the real integration surface.

// ===== the kernel (dasSpirv — SSBO 32-bit elements only, no subgroup ops) =====
// One thread = one output row of one region: y[r*d+row] = sum over 32-blocks of
// ws[block]*xs[block]*(int8 dot). Weights/activations arrive packed 4-per-uint; sign-extend via
// shift pairs. Params ride in vk_meta (NOT push constants — the generated pusher drops small
// trailing scalars, dasVulkan ROADMAP): [0]=n [1]=d [2]=nregions [3]=pad, then (rel_woff, aoff)
// uint pairs per region, offsets in weight/activation ELEMENTS (bytes for int8).

var @ssbo @binding = 0 vk_w : array<uint>
var @ssbo @binding = 1 vk_ws : array<float>
var @ssbo @binding = 2 vk_meta : array<uint>
var @ssbo @binding = 3 vk_xq : array<uint>
var @ssbo @binding = 4 vk_xs : array<float>
var @ssbo @binding = 5 vk_y : array<float>

[compute_shader(local_size_x=64, name="q8_moe_groupn_spv")]
def q8_moe_groupn {
    let gid = gl_GlobalInvocationID.x
    let n = vk_meta[0u]
    let d = vk_meta[1u]
    let nreg = vk_meta[2u]
    let r = gid / d
    if (r < nreg) {
        let row = gid % d
        let wrow = vk_meta[4u + r * 2u] + row * n
        let abase = vk_meta[5u + r * 2u]
        let nb = n / 32u
        var acc = 0.0
        var b = 0u
        while (b < nb) {
            var idot = 0
            var i = 0u
            while (i < 8u) {
                let wp = vk_w[(wrow >> 2u) + b * 8u + i]
                let xp = vk_xq[(abase >> 2u) + b * 8u + i]
                idot += ((int(wp << 24u)) >> 24) * ((int(xp << 24u)) >> 24)
                idot += ((int(wp << 16u)) >> 24) * ((int(xp << 16u)) >> 24)
                idot += ((int(wp << 8u)) >> 24) * ((int(xp << 8u)) >> 24)
                idot += ((int(wp)) >> 24) * ((int(xp)) >> 24)
                i++
            }
            acc += float(idot) * vk_ws[wrow / 32u + b] * vk_xs[abase / 32u + b]
            b++
        }
        vk_y[r * d + row] = acc
    }
}

// ===== device state (process-lifetime; never finalized by design) =====

let private VRAM_BUDGET = 5_500_000_000l   // resident-weight cap: under the measured ~6.5GB WDDM paging cliff
let private STAGING_BYTES = 268_435_456l   // one upload chunk (largest single memcpy)
let private XQ_BYTES = 1_048_576l          // per-dispatch activation quants capacity
let private XS_BYTES = 262_144l            // ... their scales
let private Y_BYTES = 1_048_576l           // per-dispatch output capacity
let private META_BYTES = 4_096l            // params + region table

struct private HostBuf {
    buf : uint64       // VkBuffer
    mem : uint64       // VkDeviceMemory
    mapped : void?     // persistently mapped (host-coherent)
    bytes : int64
}

struct private VkStack {
    base : int64       // first weight-element offset covered (into the q8 plane)
    elems : int64      // element (= byte) count
    wbuf : uint64      // device-local quants (bound as array<uint>)
    sbuf : uint64      // device-local scales (array<float>)
    sbytes : int64
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
    raw_set : VkDescriptorSet
    pipe_layout : PipelineLayout
    pipeline : Pipeline
    staging : HostBuf
    meta : HostBuf
    xq : HostBuf
    xs : HostBuf
    y : HostBuf
    @do_not_delete stacks : array<VkStack>
    resident_bytes : int64
}

var private g_gpu : GpuState?
var private g_init_failed = false

def private dev_raw : VkDevice {
    return boost_value_to_vk(g_gpu.device)
}

def private make_host_buf(bytes : int64; storage : bool) : HostBuf {
    var bci : BufferCreateInfo
    bci.size = uint64(bytes)
    bci.usage.storage_buffer = storage
    bci.usage.transfer_src = !storage
    var b <- create_buffer(g_gpu.device, bci)   // deliberately never finalized (process-lifetime)
    let req = get_buffer_memory_requirements(g_gpu.device, b)
    var want : VkMemoryPropertyFlags
    want.host_visible = true
    want.host_coherent = true
    let mai = MemoryAllocateInfo(allocationSize = req.size,
                                 memoryTypeIndex = find_memory_type(g_gpu.phys, req.memoryTypeBits, want))
    var m <- allocate_memory(g_gpu.device, mai)
    bind_buffer_memory(g_gpu.device, b, m, 0ul)
    var mapped : void? = null
    let mf : VkMemoryMapFlags
    unsafe {
        vk_check(vkMapMemory(dev_raw(), boost_value_to_vk(m), 0ul, uint64(bytes), mf, addr(mapped)), null)
    }
    var hb : HostBuf
    hb.buf = b._vk
    hb.mem = m._vk
    hb.mapped = mapped
    hb.bytes = bytes
    return hb
}

def private make_device_buf(bytes : int64) : uint64 {
    var bci : BufferCreateInfo
    bci.size = uint64(bytes)
    bci.usage.storage_buffer = true
    bci.usage.transfer_dst = true
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
    if (g_gpu != null) {
        return true
    }
    if (g_init_failed) {
        return false
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
    g_gpu.fam = select_graphics_queue_family(g_gpu.phys)
    g_gpu.device <- create_device(g_gpu.phys, g_gpu.fam)
    volkLoadDevice(dev_raw())
    g_gpu.queue = get_device_queue(g_gpu.device, g_gpu.fam, 0u)
    var pci : CommandPoolCreateInfo
    pci.queueFamilyIndex = g_gpu.fam
    g_gpu.pool <- create_command_pool(g_gpu.device, pci)

    // descriptor layout: 6 storage buffers, one set, rewritten per dispatch (never in flight
    // across a rewrite — run_cmd_sync drains the queue)
    var dslci : DescriptorSetLayoutCreateInfo
    for (bind in range(6)) {
        var b0 = DescriptorSetLayoutBinding(binding = uint(bind),
                                            descriptorType = VkDescriptorType.STORAGE_BUFFER,
                                            descriptorCount = 1u)
        b0.stageFlags.compute = true
        dslci.pBindings |> emplace(b0)
    }
    g_gpu.set_layout <- create_descriptor_set_layout(g_gpu.device, dslci)
    var dpci : DescriptorPoolCreateInfo
    dpci.maxSets = 1u
    var psize = DescriptorPoolSize(type_ = VkDescriptorType.STORAGE_BUFFER, descriptorCount = 6u)
    dpci.pPoolSizes |> push(psize)
    g_gpu.desc_pool <- create_descriptor_pool(g_gpu.device, dpci)
    var dsai = VkDescriptorSetAllocateInfo()
    dsai.descriptorPool = boost_value_to_vk(g_gpu.desc_pool)
    dsai.descriptorSetCount = 1u
    var raw_set_layout = boost_value_to_vk(g_gpu.set_layout)
    unsafe {
        dsai.pSetLayouts = addr(raw_set_layout)
        vk_check(vkAllocateDescriptorSets(dev_raw(), dsai, addr(g_gpu.raw_set)), null)
    }

    var shader <- create_shader_module(g_gpu.device, q8_moe_groupn_spv)   // never finalized (kept via pipeline)
    var plci : PipelineLayoutCreateInfo
    plci.pSetLayouts <- [weak_copy(g_gpu.set_layout)]
    g_gpu.pipe_layout <- create_pipeline_layout(g_gpu.device, plci)
    g_gpu.pipeline <- create_compute_pipeline(g_gpu.device, g_gpu.pipe_layout, shader)

    g_gpu.staging = make_host_buf(STAGING_BYTES, false)
    g_gpu.meta = make_host_buf(META_BYTES, true)
    g_gpu.xq = make_host_buf(XQ_BYTES, true)
    g_gpu.xs = make_host_buf(XS_BYTES, true)
    g_gpu.y = make_host_buf(Y_BYTES, true)
    g_init_failed = false
    to_log(LOG_INFO, "dasLLAMA vulkan tier: device ready\n")
    return true
}

// ===== upload (loader hook, pre-repack) =====

def private copy_to_staging(src : void?; src_off : int64; bytes : int64) {
    unsafe {
        memcpy(g_gpu.staging.mapped, reinterpret<void?>(reinterpret<uint8? -const>(src) + src_off), int(bytes))
    }
}

def private staged_upload(dst : uint64; dst_off : int64; bytes : int64) {
    var regions : array<BufferCopy>
    var rc : BufferCopy
    rc.srcOffset = 0ul
    rc.dstOffset = uint64(dst_off)
    rc.size = uint64(bytes)
    regions |> push(rc)
    run_cmd_sync(g_gpu.device, g_gpu.pool, g_gpu.queue) $(cmd) {
        cmd_copy_buffer(cmd, nonowning_buf(g_gpu.staging.buf), nonowning_buf(dst), regions)
    }
}

def private upload_region(dst : uint64; src : void?; total : int64) {
    var off = 0l
    while (off < total) {
        let chunk = min(total - off, STAGING_BYTES)
        copy_to_staging(src, off, chunk)
        staged_upload(dst, off, chunk)
        off += chunk
    }
}

//! MoeGpuUploadFn: upload one row-major fp32-scale Q8 stack ([rows x n] at element offset woff).
def vk_moe_upload_stack(wp : int8 const?; sp : float const?; woff : int64; n : int64; rows : int64) : bool {
    if (!vk_moe_init()) {
        return false
    }
    let wbytes = rows * n
    let sbytes = (rows * n / 32l) * 4l
    if (g_gpu.resident_bytes + wbytes + sbytes > VRAM_BUDGET) {
        to_log(LOG_INFO, "dasLLAMA vulkan tier: VRAM budget reached at {g_gpu.resident_bytes} bytes\n")
        return false
    }
    var st : VkStack
    st.base = woff
    st.elems = wbytes
    st.wbuf = make_device_buf(wbytes)
    st.sbuf = make_device_buf(sbytes)
    st.sbytes = sbytes
    unsafe {
        upload_region(st.wbuf, reinterpret<void? -const>(wp + woff), wbytes)
        upload_region(st.sbuf, reinterpret<void? -const>(sp + woff / 32l), sbytes)
    }
    g_gpu.stacks |> push(st)
    g_gpu.resident_bytes += wbytes + sbytes
    return true
}

// ===== dispatch (MoeGpuGroupNFn) =====

def private write_buf_desc(var writes : array<WriteDescriptorSet>; binding : uint; buf : uint64; bytes : int64) {
    var w = WriteDescriptorSet(dstSet = vk_value_to_boost(g_gpu.raw_set), dstBinding = binding,
                               descriptorType = VkDescriptorType.STORAGE_BUFFER, descriptorCount = 1u)
    var binfo = DescriptorBufferInfo(buffer = nonowning_buf(buf), range_ = uint64(bytes))
    w.pBufferInfo |> push(binfo)
    writes |> emplace(w)
}

//! MoeGpuGroupNFn: region-list Q8 GEMV over resident stacks. offs as in MatmulQ8Q8GroupNFn —
//! [2r] absolute weight element offset (resolved against the resident stack), [2r+1] activation
//! element offset. y is nregions*d floats.
def vk_moe_groupn(var yp : float?; offs : int64 const?; nregions : int64; xqp : int8 const?; xsp : float const?; n : int64; d : int64) {
    // resolve the stack (all regions of one call live in one stack: same layer+kind)
    let w0 = unsafe(offs[0])
    var si = -1
    for (i in range(length(g_gpu.stacks))) {
        if (w0 >= g_gpu.stacks[i].base && w0 < g_gpu.stacks[i].base + g_gpu.stacks[i].elems) {
            si = i
            break
        }
    }
    if (si < 0) {
        panic("dasLLAMA vulkan tier: dispatch against a non-resident weight region")
    }
    let stack = g_gpu.stacks[si]

    // meta = [n, d, nregions, 0, (rel_woff, aoff)*]
    var meta : array<uint>
    meta |> reserve(int(4l + nregions * 2l))
    meta |> push(uint(n))
    meta |> push(uint(d))
    meta |> push(uint(nregions))
    meta |> push(0u)
    var amax = 0l
    for (r in range64(nregions)) {
        meta |> push(uint(unsafe(offs[r * 2l]) - stack.base))
        let aoff = unsafe(offs[r * 2l + 1l])
        meta |> push(uint(aoff))
        amax = max(amax, aoff)
    }
    let xq_bytes = amax + n
    let xs_bytes = (xq_bytes / 32l) * 4l
    let y_bytes = nregions * d * 4l
    assert(xq_bytes <= XQ_BYTES && y_bytes <= Y_BYTES, "dasLLAMA vulkan tier: dispatch exceeds scratch capacity")
    unsafe {
        memcpy(g_gpu.meta.mapped, addr<void?>(meta[0]), length(meta) * 4)
        memcpy(g_gpu.xq.mapped, reinterpret<void? -const>(xqp), int(xq_bytes))
        memcpy(g_gpu.xs.mapped, reinterpret<void? -const>(xsp), int(xs_bytes))
    }

    var writes : array<WriteDescriptorSet>
    write_buf_desc(writes, 0u, stack.wbuf, stack.elems)
    write_buf_desc(writes, 1u, stack.sbuf, stack.sbytes)
    write_buf_desc(writes, 2u, g_gpu.meta.buf, META_BYTES)
    write_buf_desc(writes, 3u, g_gpu.xq.buf, XQ_BYTES)
    write_buf_desc(writes, 4u, g_gpu.xs.buf, XS_BYTES)
    write_buf_desc(writes, 5u, g_gpu.y.buf, Y_BYTES)
    let no_copies : array<CopyDescriptorSet>
    update_descriptor_sets(g_gpu.device, writes, no_copies)

    let groups = uint((nregions * d + 63l) / 64l)
    run_cmd_sync(g_gpu.device, g_gpu.pool, g_gpu.queue) $(cmd) {
        cmd_bind_pipeline(cmd, g_gpu.pipeline, VkPipelineBindPoint.COMPUTE)
        var no_offsets : array<uint>
        cmd_bind_descriptor_sets(cmd, VkPipelineBindPoint.COMPUTE, g_gpu.pipe_layout, 0u, [vk_value_to_boost(g_gpu.raw_set)], no_offsets)
        cmd_dispatch(cmd, groups, 1u, 1u)
    }
    unsafe {
        memcpy(reinterpret<void?>(yp), g_gpu.y.mapped, int(y_bytes))
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
            install_moe_gpu_tier(@@vk_moe_groupn, @@vk_moe_upload_stack)
        }
    }
}
