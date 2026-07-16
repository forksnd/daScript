options gen2

module dasllama_image shared public

require daslib/fio
require daslib/apply
require strings
require dasllama/dasllama_tokenizer   // Tokenizer — the bulk tokenizer serializer below
require daslib/archive
require daslib/array_boost   // nolint:STYLE030 — temp_array is [template]; STYLE030 misses template-fn refs (same note as dasllama_common)
require dasllama/dasllama_common
require dasllama/dasllama_math   // active_kernel_backend — half the image identity

// The prepared-model image (spec: fmap-model-image plan, locked 2026-07-16): the post-load
// Model dumped once — planes POST-repack, page-aligned — then mapped back with zero O(model)
// allocation or copying. Plane fields become borrowed views over the mapping (the Model
// finalizer forgets them and unmaps); everything else (config, scalars, tokenizer, arch
// blocks) rides a daslib/archive blob and is PARSED at load (the small-structure carve-out).
// Images are BOX + KNOB specific (identity below) and versioned — a mismatch regenerates,
// never migrates. The load path touches only the header + meta blob: bigger-than-RAM images
// stay lazy (full verification belongs to the offline tool, not load).

// bump on ANY layout/serialization change — a mismatched image regenerates from the gguf
let IMAGE_VERSION = 1
let IMAGE_PAGE = 16384l          // Apple-Silicon page: the bytesNoCopy alignment contract
let IMAGE_HEADER_BYTES = 64l     // magic..identity, inside the first page
let IMAGE_MAGIC = 0x314D4C44u    // 'DLM1'

struct private ImgSection {
    name : string
    off : uint64      // absolute byte offset, page-aligned
    bytes : uint64
}

// the box + knob identity an image is valid for; folded into the file name AND the header
def image_identity : string {
    return "v{IMAGE_VERSION}|{active_kernel_backend()}|{get_wscale_f16() ? "s16" : "s32"}"
}

// the image path for (gguf, current identity) — knob combos coexist as separate files
def image_path_for(gguf_path : string) : string {
    return "{gguf_path}.{hash(image_identity()) & 0xFFFFFFFFul}.dlim"
}

def private pad_to_page(cur : uint64) : uint64 {
    let page = uint64(IMAGE_PAGE)
    return (cur + page - 1ul) / page * page
}

def private write_zeros(f : file; n : uint64) {
    if (n == 0ul) {
        return
    }
    var z : array<uint8>
    z |> resize(int(n))
    f |> fwrite(z)
    delete z
}

def private store_u32(var b : array<uint8>; at : int; v : uint) {
    unsafe {
        var p = unsafe(addr<uint?>(b[at]))
        p[0] = v
    }
}

def private store_u64(var b : array<uint8>; at : int; v : uint64) {
    unsafe {
        var p = unsafe(addr<uint64?>(b[at]))
        p[0] = v
    }
}

def private read_u32(p : uint8 const?; at : int64) : uint {
    unsafe {
        var q = reinterpret<uint const?>(p + at)
        return q[0]
    }
}

def private read_u64(p : uint8 const?; at : int64) : uint64 {
    unsafe {
        var q = reinterpret<uint64 const?>(p + at)
        return q[0]
    }
}



// bulk POD-array serialize: ONE stream call for the payload. daslib/archive's per-element
// generic dispatch measured ~340us/element — a 128k-vocab tokenizer took 238s through it.
def private serialize_pod_array(var arch : Archive; var a : array<auto(TT)>) {
    var n = length(a)
    arch |> serialize_raw(n)
    if (arch.reading) {
        a |> resize(n)
    }
    if (n > 0) {
        let nbytes = n * typeinfo sizeof(type<TT>)
        unsafe {
            if (arch.reading) {
                arch.stream->read(addr(a[0]), nbytes)
            } else {
                arch.stream->write(addr(a[0]), nbytes)
            }
        }
    }
}

// string arrays as [lengths][one blob] — two bulk payloads instead of per-string dispatch
def private serialize_strings(var arch : Archive; var a : array<string>) {
    var lens : array<int>
    var blob : array<uint8>
    if (!arch.reading) {
        lens |> reserve(length(a))
        var total = 0
        for (v in a) {
            let n = length(v)
            lens |> push(n)
            total += n
        }
        blob |> resize(total)
        var at = 0
        for (v in a) {
            let n = length(v)
            if (n > 0) {
                peek_data(v) $(bytes : array<uint8> const#) {
                    unsafe {
                        memcpy(addr(blob[at]), addr(bytes[0]), n)
                    }
                }
            }
            at += n
        }
    }
    serialize_pod_array(arch, lens)
    serialize_pod_array(arch, blob)
    if (arch.reading) {
        a |> resize(length(lens))
        var at = 0
        for (i in range(length(lens))) {
            let n = lens[i]
            if (n > 0) {
                unsafe {
                    a[i] = string(temp_array(addr<uint8?>(blob[at]), n, type<uint8>))
                }
            } else {
                a[i] = ""
            }
            at += n
        }
    }
    delete lens
    delete blob
}

// the tokenizer, bulk: vocab/scores/token_type/merges as flat payloads; the lookup tables are
// DERIVED (piece -> index, merge -> rank-order index) and rebuild in milliseconds at load —
// exactly the "parse at load" carve-out the format spec allows
def private serialize_tokenizer(var arch : Archive; var tk : Tokenizer) {
    arch |> serialize_raw(tk.kind)
    serialize_strings(arch, tk.spm.vocab)
    serialize_pod_array(arch, tk.spm.scores)
    arch |> serialize_raw(tk.spm.max_token_length)
    arch |> serialize_raw(tk.spm.add_space_prefix)
    arch |> serialize_raw(tk.spm.bos_id)
    arch |> serialize_raw(tk.spm.eos_id)
    serialize_strings(arch, tk.bpe.vocab)
    serialize_pod_array(arch, tk.bpe.token_type)
    arch |> serialize(tk.bpe.pre)
    arch |> serialize_raw(tk.bpe.bos_id)
    arch |> serialize_raw(tk.bpe.eos_id)
    arch |> serialize_raw(tk.bpe.spm_space)
    arch |> serialize_raw(tk.bpe.add_bos)
    var merges : array<string>
    if (!arch.reading) {
        merges |> resize(length(tk.bpe.merge_rank))
        for (k, v in keys(tk.bpe.merge_rank), values(tk.bpe.merge_rank)) {
            merges[int(v)] := k
        }
    }
    serialize_strings(arch, merges)
    if (arch.reading) {
        for (i, v in count(), tk.spm.vocab) {
            tk.spm.lookup[v] = int64(i)
        }
        for (i, v in count(), tk.bpe.vocab) {
            tk.bpe.lookup[v] = int64(i)
        }
        for (i, v in count(), merges) {
            tk.bpe.merge_rank[v] = int64(i)
        }
    }
    delete merges
}

// every non-array Model field, explicitly, both directions (the archive is bidirectional).
// blocks is NOT serialized — resolve_arch re-binds it from the registry after the read.
// The count tripwire below forces this list to grow with the struct.
def private serialize_model_meta(var arch : Archive; var t : Model) {
    arch |> serialize(t.config)
    arch |> serialize(t.arch)
    arch |> serialize(t.chat_template_str)
    serialize_tokenizer(arch, t.tok)
    arch |> serialize_raw(t.tok_emb_off)
    arch |> serialize_raw(t.rms_att_off)
    arch |> serialize_raw(t.rms_ffn_off)
    arch |> serialize_raw(t.rms_final_off)
    arch |> serialize_raw(t.rms_post_att_off)
    arch |> serialize_raw(t.rms_post_ffn_off)
    arch |> serialize_raw(t.rms_ones_off)
    arch |> serialize_raw(t.out_scale_off)
    arch |> serialize_raw(t.ple_emb_off)
    arch |> serialize_raw(t.ple_model_off)
    arch |> serialize_raw(t.ple_pnorm_off)
    arch |> serialize_raw(t.ple_gate_off)
    arch |> serialize_raw(t.ple_proj_off)
    arch |> serialize_raw(t.ple_post_off)
    arch |> serialize_raw(t.wscale_f16)
    arch |> serialize_raw(t.experts_mx4)
    arch |> serialize_raw(t.ple_model_bf16)
    arch |> serialize_raw(t.kquant_native)
    arch |> serialize_raw(t.kq_repacked)
    arch |> serialize_raw(t.kq_repack_mr4)
    arch |> serialize_raw(t.kq_repack_mr5)
    arch |> serialize_raw(t.kq_repack_mr6)
    arch |> serialize_raw(t.kq_repack_mr40)
    arch |> serialize_raw(t.wcls_fmt)
    arch |> serialize_raw(t.emb_fmt)
    arch |> serialize_raw(t.ple_emb_fmt)
    arch |> serialize_raw(t.router_off)
    arch |> serialize_raw(t.shexp_gate_off)
    arch |> serialize_raw(t.moe_router_scale_off)
    arch |> serialize_raw(t.moe_down_scale_off)
    arch |> serialize_raw(t.rms_pre_ffn2_off)
    arch |> serialize_raw(t.rms_post_ffn1_off)
    arch |> serialize_raw(t.rms_post_ffn2_off)
    arch |> serialize_raw(t.router_b_off)
    arch |> serialize_raw(t.exp_probs_off)
    arch |> serialize_raw(t.web1_off)
    arch |> serialize_raw(t.web2_off)
    arch |> serialize_raw(t.web3_off)
    arch |> serialize_raw(t.sinks_off)
    arch |> serialize_raw(t.wsh1_off)
    arch |> serialize_raw(t.wsh2_off)
    arch |> serialize_raw(t.wsh3_off)
    arch |> serialize_raw(t.wcls_off)
    arch |> serialize_raw(t.cls_q8)
    arch |> serialize_raw(t.emb_q8_off)
    arch |> serialize_raw(t.quant)
    arch |> serialize_raw(t.dn_conv_off)
    arch |> serialize_raw(t.dn_dt_off)
    arch |> serialize_raw(t.dn_a_off)
    arch |> serialize_raw(t.dn_norm_off)
    arch |> serialize_raw(t.dn_ba_f32)
    arch |> serialize_raw(t.dn_betaf_off)
    arch |> serialize_raw(t.dn_alphaf_off)
    arch |> serialize_raw(t.dn_grouped)
    arch |> serialize_raw(t.mtp_ehproj_off)
    arch |> serialize_raw(t.mtp_ehproj_fmt)
    arch |> serialize_raw(t.mtp_enorm_off)
    arch |> serialize_raw(t.mtp_hnorm_off)
    arch |> serialize_raw(t.mtp_headnorm_off)
}

// the tripwire: serialize_model_meta covers 63 fields; blocks + image_map + image_bytes are
// deliberate skips. A new non-array Model field trips this until BOTH lists agree.
let IMAGE_META_FIELDS = 63 + 3

def private count_meta_fields(var t : Model) : int {
    var n = 0
    apply(t) $ [unused_argument(name, field)] (name : string; var field) {
        static_if (!typeinfo is_array(field)) {
            n++
        }
    }
    return n
}

// ===== save =====

//! Serialize a loaded Model as a prepared image: planes page-aligned raw (borrowed back at
//! load), everything else archived. Atomic (tmp + rename). Returns false on any IO failure.
def save_model_image(var t : Model; path : string) : bool {
    let tmp_path = "{path}.tmp"
    var ok = true
    var meta <- new MemSerializer()
    var warch = Archive(reading = false, stream = meta)
    var sections : array<ImgSection>
    fopen(tmp_path, "wb") $(f) {
        if (f == null) {
            ok = false
            return
        }
        // page 0: header placeholder — patched after the walk (fseek back)
        write_zeros(f, uint64(IMAGE_PAGE))
        var cur = uint64(IMAGE_PAGE)
        verify(count_meta_fields(t) == IMAGE_META_FIELDS)   // grew Model? extend serialize_model_meta
        let ts_meta = ref_time_ticks()
        serialize_model_meta(warch, t)
        to_log(LOG_INFO, "dasLLAMA image: meta serialize {get_time_usec(ts_meta) / 1000}ms\n")
        let ts_planes = ref_time_ticks()
        apply(t) $(name : string; var field) {
            static_if (typeinfo is_array(field)) {
                if (!empty(field)) {
                    let nbytes = uint64(length(field)) * uint64(typeinfo sizeof(field[0]))
                    let at = pad_to_page(cur)
                    write_zeros(f, at - cur)
                    f |> fwrite(field)
                    sections |> push(ImgSection(name = name, off = at, bytes = nbytes))
                    cur = at + nbytes
                }
            }
        }
        to_log(LOG_INFO, "dasLLAMA image: planes write {get_time_usec(ts_planes) / 1000}ms\n")
        // meta = [sections][the walk's scalar stream], one blob at the tail
        var smeta <- new MemSerializer()
        var sarch = Archive(reading = false, stream = smeta)
        sarch |> serialize(sections)
        var meta_blob <- smeta->extractData()
        var scalar_blob <- meta->extractData()
        meta_blob |> push_from(scalar_blob)
        let meta_off = cur
        let meta_bytes = uint64(length(meta_blob))
        f |> fwrite(meta_blob)
        cur += meta_bytes
        // pad the file itself to a page multiple (the whole-file bytesNoCopy wrap needs it)
        let fend = pad_to_page(cur)
        write_zeros(f, fend - cur)
        // patch the header
        var hdr : array<uint8>
        hdr |> resize(int(IMAGE_HEADER_BYTES))
        store_u32(hdr, 0, IMAGE_MAGIC)
        store_u32(hdr, 4, uint(IMAGE_VERSION))
        store_u32(hdr, 8, uint(IMAGE_PAGE))
        store_u32(hdr, 12, uint(length(sections)))
        store_u64(hdr, 16, meta_off)
        store_u64(hdr, 24, meta_bytes)
        store_u64(hdr, 32, fend)
        store_u64(hdr, 40, hash(image_identity()))
        fseek(f, 0l, seek_set)
        f |> fwrite(hdr)
        delete hdr
        delete meta_blob
        delete scalar_blob
        unsafe {
            delete smeta
        }
        delete sections
    }
    unsafe {
        delete meta
    }
    if (!ok) {
        return false
    }
    remove(path)   // rename-over is not portable — drop any stale image first
    if (!rename(tmp_path, path)) {
        remove(tmp_path)
        return false
    }
    return true
}

// ===== load =====

//! Map a prepared image and rebuild the Model with zero O(model) work: plane fields borrow
//! the mapping (the Model finalizer forgets + unmaps), the archive blob restores everything
//! else. false = absent / wrong identity / wrong version — the caller regenerates from gguf.
def load_model_image(path : string; var t : Model) : bool {
    if (!stat(path).is_valid) {
        return false
    }
    var msize = 0ul
    var base : void?
    unsafe {
        base = fmap_open(path, addr(msize))
    }
    var bp : uint8 const?
    unsafe {
        bp = reinterpret<uint8 const?>(base)
    }
    if (int64(msize) < IMAGE_HEADER_BYTES ||
            (read_u32(bp, 0l) != IMAGE_MAGIC || read_u32(bp, 4l) != uint(IMAGE_VERSION)) ||
            (read_u32(bp, 8l) != uint(IMAGE_PAGE) || read_u64(bp, 40l) != hash(image_identity()))) {
        unsafe {
            fmap_close(base, msize)
        }
        return false
    }
    let nsections = int(read_u32(bp, 12l))
    let meta_off = read_u64(bp, 16l)
    let meta_bytes = read_u64(bp, 24l)
    let file_bytes = read_u64(bp, 32l)
    if (file_bytes != msize || meta_off + meta_bytes > msize) {
        unsafe {
            fmap_close(base, msize)
        }
        return false
    }
    // the meta blob is the ONE copied+parsed piece (small); planes stay lazy in the mapping
    var meta_blob : array<uint8>
    meta_blob |> resize(int(meta_bytes))
    unsafe {
        memcpy(addr(meta_blob[0]), reinterpret<void?>(bp + int64(meta_off)), int(meta_bytes))
    }
    var meta <- new MemSerializer(meta_blob)
    var rarch = Archive(reading = true, stream = meta)
    var sections : array<ImgSection>
    rarch |> serialize(sections)
    var by_name <- {for (sec in sections); sec.name => sec}
    serialize_model_meta(rarch, t)
    resolve_arch(t, t.arch)   // blocks re-bind + the arch's (idempotent) configure
    var ok = true
    apply(t) $(name : string; var field) {
        static_if (typeinfo is_array(field)) {
            if (key_exists(by_name, name)) {
                let sec = by_name?[name] ?? ImgSection()
                let esz = uint64(typeinfo sizeof(field[0]))
                if (sec.off % uint64(IMAGE_PAGE) != 0ul || sec.off + sec.bytes > msize || sec.bytes % esz != 0ul) {
                    ok = false
                } else {
                    unsafe {
                        _builtin_make_temp_array_i64(field, reinterpret<void?>(bp + int64(sec.off)), int64(sec.bytes / esz))
                    }
                }
            }
        }
    }
    delete meta_blob
    delete sections
    delete by_name
    unsafe {
        delete meta
    }
    if (!ok || nsections < 0) {
        // a malformed section table: forget whatever was borrowed and unmap
        delete t
        return false
    }
    t.image_map = base
    t.image_bytes = msize
    return true
}
