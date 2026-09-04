use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi::paged_attention_turbo4 as ffi_paged_attention_turbo4;
use candle_core::backend::BackendStorage;
use candle_core::{DType, Result, Storage, Tensor};

/// Matches `mistralrs_core::paged_attention::turbo_quant::BLOCK_TURBO4_BYTES` /
/// `GROUP` (2 fixed-point norm bytes + GROUP/2 nibble-packed bytes, GROUP=128).
const BLOCK_TURBO4_BYTES: usize = 66;
const TURBO4_GROUP: usize = 128;

/// Conservative cap on `max_num_blocks_per_seq * block_size` (the cache's allocated per-sequence
/// capacity, not the actual context length -- this stays fixed across a CUDA graph's replays,
/// unlike per-request `context_lens`) for which the fused kernel's `logits` shared-memory buffer
/// (`max_context_len` floats) fits inside every CUDA architecture's default 48KB static/dynamic
/// shared memory budget with headroom to spare, no `cudaFuncAttributeMaxDynamicSharedMemorySize`
/// opt-in required. The kernel itself does request that opt-in unconditionally (defensive
/// headroom, not the load-bearing limit here), but that request calls `exit()` on failure rather
/// than returning a `Result` -- this bound exists so callers never reach it. Sequences needing
/// more context than this fall back to the gather+Sdpa path, which has no such limit.
pub const MAX_FUSED_DECODE_CONTEXT_TOKENS: usize = 8192;

fn io_dtype_code(dtype: DType, op: &str) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("{op} only supports f16, bf16, f32 (got {other:?})"),
    }
}

struct SideInfo<'a> {
    ptr_turbo: Option<&'a Tensor>,
    ptr_plain: Option<&'a Tensor>,
    is_turbo: bool,
    block_size: usize,
    kv_heads: usize,
}

fn classify_side<'a>(
    cache: &'a Tensor,
    head_size: usize,
    groups_per_head: usize,
    io_dtype: DType,
    name: &str,
) -> Result<SideInfo<'a>> {
    if cache.dtype() == DType::U8 {
        let (_num_blocks, block_size, kv_heads, cache_groups, bytes) = cache.dims5()?;
        if bytes != BLOCK_TURBO4_BYTES || cache_groups != groups_per_head {
            candle_core::bail!(
                "paged_attention_turbo4: {name} turbo cache shape {:?} does not match \
                 groups_per_head={groups_per_head}",
                cache.dims()
            );
        }
        Ok(SideInfo {
            ptr_turbo: Some(cache),
            ptr_plain: None,
            is_turbo: true,
            block_size,
            kv_heads,
        })
    } else {
        if cache.dtype() != io_dtype {
            candle_core::bail!(
                "paged_attention_turbo4: {name} plain cache dtype {:?} does not match query \
                 dtype {io_dtype:?}",
                cache.dtype()
            );
        }
        let (_num_blocks, block_size, kv_heads, cache_head_size) = cache.dims4()?;
        if cache_head_size != head_size {
            candle_core::bail!(
                "paged_attention_turbo4: {name} plain cache head_size {cache_head_size} does \
                 not match query head_size {head_size}"
            );
        }
        Ok(SideInfo {
            ptr_turbo: None,
            ptr_plain: Some(cache),
            is_turbo: false,
            block_size,
            kv_heads,
        })
    }
}

/// Fused decode-time paged attention over a Turbo4 (and/or plain-fallback) KV cache: rotates Q
/// forward once, runs the whole KQ/softmax/weighted-V loop directly against the cache with a
/// per-element centroid dequant (no per-token inverse WHT), then inverse-rotates the
/// accumulated output once. See `paged_attention_turbo4_kernel.cu`'s module doc for why this is
/// correct and how it differs from `gather_turbo4_cache` + `Sdpa`.
///
/// `query` must be `(num_seqs, num_heads, head_size)` (decode: exactly one query token per
/// sequence -- this is not a prefill/multi-token-query kernel). `key_cache`/`value_cache` are
/// each either the block-indexed Turbo4-packed layout (u8, `(num_blocks, block_size, kv_heads,
/// groups_per_head, BLOCK_TURBO4_BYTES)` -- the same shape `gather_turbo4_cache` expects) or the
/// block-indexed plain fallback (`(num_blocks, block_size, kv_heads, head_size)`, same dtype as
/// `query`) -- independently, matching the auto-asymmetric/layer-adaptive fallback.
/// `block_table` is `(num_seqs, max_num_blocks_per_seq)`, `context_lens` is `(num_seqs,)`; both
/// are read as GPU tensors with no host sync, so this is safe to call from inside a captured
/// CUDA graph. `context_lens` values must not exceed `max_num_blocks_per_seq * block_size`. See
/// [`MAX_FUSED_DECODE_CONTEXT_TOKENS`] for the resulting capacity bound this function enforces.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_turbo4(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    context_lens: &Tensor,
    scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    let io_dtype = query.dtype();
    let io_dtype_code = io_dtype_code(io_dtype, "paged_attention_turbo4")?;
    let query = query.contiguous()?;
    let (num_seqs, num_heads, head_size) = query.dims3()?;
    if !head_size.is_multiple_of(TURBO4_GROUP) {
        candle_core::bail!(
            "paged_attention_turbo4 requires head_size ({head_size}) to be a multiple of {TURBO4_GROUP}"
        );
    }
    let groups_per_head = head_size / TURBO4_GROUP;
    if groups_per_head > 4 {
        candle_core::bail!(
            "paged_attention_turbo4: head_size {head_size} implies groups_per_head \
             {groups_per_head} > 4, unsupported"
        );
    }

    let key = classify_side(key_cache, head_size, groups_per_head, io_dtype, "key")?;
    let value = classify_side(value_cache, head_size, groups_per_head, io_dtype, "value")?;
    if key.block_size != value.block_size {
        candle_core::bail!(
            "paged_attention_turbo4: key/value block_size mismatch ({} vs {})",
            key.block_size,
            value.block_size
        );
    }
    if key.kv_heads != value.kv_heads {
        candle_core::bail!(
            "paged_attention_turbo4: key/value kv_heads mismatch ({} vs {})",
            key.kv_heads,
            value.kv_heads
        );
    }
    let num_kv_heads = key.kv_heads;
    let block_size = key.block_size;
    if !num_heads.is_multiple_of(num_kv_heads) {
        candle_core::bail!(
            "paged_attention_turbo4: num_heads ({num_heads}) is not a multiple of num_kv_heads \
             ({num_kv_heads})"
        );
    }

    let block_table = block_table.contiguous()?;
    if !matches!(block_table.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "paged_attention_turbo4 expects i32/u32 block_table (got {:?})",
            block_table.dtype()
        );
    }
    let (bt_seqs, max_num_blocks_per_seq) = block_table.dims2()?;
    if bt_seqs != num_seqs {
        candle_core::bail!(
            "paged_attention_turbo4: block_table rows ({bt_seqs}) do not match num_seqs ({num_seqs})"
        );
    }
    if max_num_blocks_per_seq * block_size > MAX_FUSED_DECODE_CONTEXT_TOKENS {
        candle_core::bail!(
            "paged_attention_turbo4: capacity {} exceeds MAX_FUSED_DECODE_CONTEXT_TOKENS ({}); \
             caller should fall back to the gather+Sdpa path",
            max_num_blocks_per_seq * block_size,
            MAX_FUSED_DECODE_CONTEXT_TOKENS
        );
    }

    let context_lens = context_lens.contiguous()?;
    if !matches!(context_lens.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "paged_attention_turbo4 expects i32/u32 context_lens (got {:?})",
            context_lens.dtype()
        );
    }
    if context_lens.dims1()? != num_seqs {
        candle_core::bail!("paged_attention_turbo4: context_lens length does not match num_seqs");
    }

    let out = Tensor::zeros((num_seqs, num_heads, head_size), io_dtype, query.device())?;
    if num_seqs == 0 {
        return Ok(out);
    }

    let (q_s, q_l) = query.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("query must be a cuda tensor"),
    };
    let (o_storage_guard, o_l) = out.storage_and_layout();
    let o_s = match &*o_storage_guard {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("out must be a cuda tensor"),
    };
    let ((q_ptr, _q_guard), (o_ptr, _o_guard)) = match io_dtype {
        DType::F16 => (
            slice_ptr(q_s.as_cuda_slice::<half::f16>()?, q_l.start_offset()),
            slice_ptr(o_s.as_cuda_slice::<half::f16>()?, o_l.start_offset()),
        ),
        DType::BF16 => (
            slice_ptr(q_s.as_cuda_slice::<half::bf16>()?, q_l.start_offset()),
            slice_ptr(o_s.as_cuda_slice::<half::bf16>()?, o_l.start_offset()),
        ),
        DType::F32 => (
            slice_ptr(q_s.as_cuda_slice::<f32>()?, q_l.start_offset()),
            slice_ptr(o_s.as_cuda_slice::<f32>()?, o_l.start_offset()),
        ),
        _ => unreachable!(),
    };

    let (k_cache_s, k_cache_l) = match (key.ptr_turbo, key.ptr_plain) {
        (Some(t), None) => t.storage_and_layout(),
        (None, Some(t)) => t.storage_and_layout(),
        _ => unreachable!(),
    };
    let k_cache_s = match &*k_cache_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("key cache must be a cuda tensor"),
    };
    let (k_ptr, _k_guard) = if key.is_turbo {
        slice_ptr(k_cache_s.as_cuda_slice::<u8>()?, k_cache_l.start_offset())
    } else {
        match io_dtype {
            DType::F16 => slice_ptr(k_cache_s.as_cuda_slice::<half::f16>()?, k_cache_l.start_offset()),
            DType::BF16 => slice_ptr(k_cache_s.as_cuda_slice::<half::bf16>()?, k_cache_l.start_offset()),
            DType::F32 => slice_ptr(k_cache_s.as_cuda_slice::<f32>()?, k_cache_l.start_offset()),
            _ => unreachable!(),
        }
    };

    let (v_cache_s, v_cache_l) = match (value.ptr_turbo, value.ptr_plain) {
        (Some(t), None) => t.storage_and_layout(),
        (None, Some(t)) => t.storage_and_layout(),
        _ => unreachable!(),
    };
    let v_cache_s = match &*v_cache_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("value cache must be a cuda tensor"),
    };
    let (v_ptr, _v_guard) = if value.is_turbo {
        slice_ptr(v_cache_s.as_cuda_slice::<u8>()?, v_cache_l.start_offset())
    } else {
        match io_dtype {
            DType::F16 => slice_ptr(v_cache_s.as_cuda_slice::<half::f16>()?, v_cache_l.start_offset()),
            DType::BF16 => slice_ptr(v_cache_s.as_cuda_slice::<half::bf16>()?, v_cache_l.start_offset()),
            DType::F32 => slice_ptr(v_cache_s.as_cuda_slice::<f32>()?, v_cache_l.start_offset()),
            _ => unreachable!(),
        }
    };

    let (bt_s, bt_l) = block_table.storage_and_layout();
    let bt_s = match &*bt_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("block_table must be a cuda tensor"),
    };
    let (bt_ptr, _bt_guard) = slice_ptr(bt_s.as_cuda_slice::<u32>()?, bt_l.start_offset());

    let (cl_s, cl_l) = context_lens.storage_and_layout();
    let cl_s = match &*cl_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("context_lens must be a cuda tensor"),
    };
    let (cl_ptr, _cl_guard) = slice_ptr(cl_s.as_cuda_slice::<u32>()?, cl_l.start_offset());

    let dev = q_s.device();
    let (k_turbo_ptr, k_plain_ptr) = if key.is_turbo {
        (k_ptr, 0u64)
    } else {
        (0u64, k_ptr)
    };
    let (v_turbo_ptr, v_plain_ptr) = if value.is_turbo {
        (v_ptr, 0u64)
    } else {
        (0u64, v_ptr)
    };

    unsafe {
        ffi_paged_attention_turbo4(
            o_ptr as *const core::ffi::c_void,
            q_ptr as *const core::ffi::c_void,
            k_turbo_ptr as *const core::ffi::c_void,
            k_plain_ptr as *const core::ffi::c_void,
            v_turbo_ptr as *const core::ffi::c_void,
            v_plain_ptr as *const core::ffi::c_void,
            key.is_turbo as i32,
            value.is_turbo as i32,
            bt_ptr as *const i32,
            cl_ptr as *const i32,
            num_seqs as i32,
            num_heads as i32,
            num_kv_heads as i32,
            head_size as i32,
            groups_per_head as i32,
            block_size as i32,
            max_num_blocks_per_seq as i32,
            scale,
            softcapping,
            dev.cuda_stream().cu_stream(),
            io_dtype_code,
        );
    }

    drop(_o_guard);
    drop(o_storage_guard);
    Ok(out)
}
