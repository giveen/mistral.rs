use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi::{
    gather_plain_kv_cache as ffi_gather_plain_kv_cache,
    gather_turbo4_cache as ffi_gather_turbo4_cache,
};
use candle_core::backend::BackendStorage;
use candle_core::{DType, Result, Storage, Tensor};

/// Matches `mistralrs_core::paged_attention::turbo_quant::BLOCK_TURBO4_BYTES`
/// (2 fixed-point norm bytes + GROUP/2 nibble-packed bytes, GROUP=128).
const BLOCK_TURBO4_BYTES: usize = 66;
const TURBO4_GROUP: usize = 128;

fn out_dtype_code(dtype: DType, op: &str) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("{op} only supports f16, bf16, f32 output (got {other:?})"),
    }
}

fn validate_block_table_and_cu_seq_lens(block_table: &Tensor, cu_seq_lens: &Tensor, op: &str) -> Result<()> {
    if !matches!(block_table.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!("{op} expects i32/u32 block_table (got {:?})", block_table.dtype());
    }
    if !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!("{op} expects i32/u32 cu_seq_lens (got {:?})", cu_seq_lens.dtype());
    }
    Ok(())
}

/// Gathers and dequantizes a Turbo4-packed KV cache (`(num_blocks, block_size,
/// kv_heads, groups_per_head, BLOCK_TURBO4_BYTES)`, u8) straight into a flat
/// `(num_tokens, kv_heads, groups_per_head * GROUP)` output tensor -- the
/// WHT-inverse-rotation and Lloyd-Max dequant happen on-GPU inside the
/// kernel, and `block_table`/`cu_seq_lens` are read as GPU tensors, so this
/// call has no host readback (unlike the eager per-row Rust path this
/// replaces, which pulled the block table to the host per batch row).
#[allow(clippy::too_many_arguments)]
pub fn gather_turbo4_cache(
    cache: &Tensor,       // [num_blocks, block_size, kv_heads, groups_per_head, BLOCK_TURBO4_BYTES]
    block_table: &Tensor, // [batch, max_blocks]
    cu_seq_lens: &Tensor, // [batch + 1]
    num_tokens: usize,    // cu_seq_lens[-1]
    out_dtype: DType,
) -> Result<Tensor> {
    if cache.dtype() != DType::U8 {
        candle_core::bail!(
            "gather_turbo4_cache expects a u8 packed cache, got {:?}",
            cache.dtype()
        );
    }
    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;
    validate_block_table_and_cu_seq_lens(&block_table, &cu_seq_lens, "gather_turbo4_cache")?;

    let (_num_blocks, block_size, kv_heads, groups_per_head, block_bytes) = cache.dims5()?;
    if block_bytes != BLOCK_TURBO4_BYTES {
        candle_core::bail!(
            "gather_turbo4_cache expects last dim {BLOCK_TURBO4_BYTES}, got {block_bytes}"
        );
    }
    let head_size = groups_per_head * TURBO4_GROUP;

    let cu_seq_lens_len = cu_seq_lens.dims1()?;
    let num_seqs = cu_seq_lens_len
        .checked_sub(1)
        .ok_or_else(|| candle_core::Error::msg("cu_seq_lens must contain an initial offset"))?;
    let num_tokens_i32 = i32::try_from(num_tokens)
        .map_err(|_| candle_core::Error::msg("num_tokens exceeds the kernel i32 limit"))?;
    let num_seqs_i32 = i32::try_from(num_seqs)
        .map_err(|_| candle_core::Error::msg("num_seqs exceeds the kernel i32 limit"))?;

    let out_dtype_code = out_dtype_code(out_dtype, "gather_turbo4_cache")?;

    if num_tokens == 0 {
        return Tensor::zeros((0, kv_heads, head_size), out_dtype, cache.device());
    }

    let out = Tensor::zeros((num_tokens, kv_heads, head_size), out_dtype, cache.device())?;

    {
        let (c_s, c_l) = cache.storage_and_layout();
        let c_s = match &*c_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("cache must be a cuda tensor"),
        };
        let (c_ptr, _c_guard) = slice_ptr(c_s.as_cuda_slice::<u8>()?, c_l.start_offset());

        let (o_s, o_l) = out.storage_and_layout();
        let o_s = match &*o_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("out must be a cuda tensor"),
        };
        let (o_ptr, _o_guard) = match out_dtype {
            DType::F16 => slice_ptr(o_s.as_cuda_slice::<half::f16>()?, o_l.start_offset()),
            DType::BF16 => slice_ptr(o_s.as_cuda_slice::<half::bf16>()?, o_l.start_offset()),
            DType::F32 => slice_ptr(o_s.as_cuda_slice::<f32>()?, o_l.start_offset()),
            _ => unreachable!(),
        };

        let (bt_s, bt_l) = block_table.storage_and_layout();
        let bt_s = match &*bt_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("block_table must be a cuda tensor"),
        };
        let (bt_ptr, _bt_guard) = slice_ptr(bt_s.as_cuda_slice::<u32>()?, bt_l.start_offset());

        let (cu_s, cu_l) = cu_seq_lens.storage_and_layout();
        let cu_s = match &*cu_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("cu_seq_lens must be a cuda tensor"),
        };
        let (cu_ptr, _cu_guard) = if cu_seq_lens.dtype() == DType::I32 {
            slice_ptr(cu_s.as_cuda_slice::<i32>()?, cu_l.start_offset())
        } else {
            slice_ptr(cu_s.as_cuda_slice::<u32>()?, cu_l.start_offset())
        };

        let (_, block_table_stride) = bt_l.shape().dims2()?;
        let dev = c_s.device();

        unsafe {
            ffi_gather_turbo4_cache(
                c_ptr as *const core::ffi::c_void,
                o_ptr as *const core::ffi::c_void,
                bt_ptr as *const i32,
                cu_ptr as *const i32,
                num_tokens_i32,
                num_seqs_i32,
                block_size as i32,
                block_table_stride as i32,
                kv_heads as i32,
                groups_per_head as i32,
                dev.cuda_stream().cu_stream(),
                out_dtype_code,
            );
        }
    }

    Ok(out)
}

/// Gathers K or V from Turbo4's plain (unquantized elementwise) fallback
/// cache -- shape `(num_blocks, block_size, kv_heads, head_size)`, the layout
/// `write_plain_cache`/`read_plain_cache` use for the auto-asymmetric /
/// layer-adaptive fallback (not the vLLM x-split layout `gather_kv_cache`
/// expects, so that function can't be reused here). `cache` and `out` must
/// share a dtype: the fallback cache is always allocated in the model's own
/// compute dtype, so there's no cast to do.
#[allow(clippy::too_many_arguments)]
pub fn gather_plain_kv_cache(
    cache: &Tensor,       // [num_blocks, block_size, kv_heads, head_size]
    block_table: &Tensor, // [batch, max_blocks]
    cu_seq_lens: &Tensor, // [batch + 1]
    num_tokens: usize,    // cu_seq_lens[-1]
) -> Result<Tensor> {
    let dtype = cache.dtype();
    let dtype_code = out_dtype_code(dtype, "gather_plain_kv_cache")?;

    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;
    validate_block_table_and_cu_seq_lens(&block_table, &cu_seq_lens, "gather_plain_kv_cache")?;

    let (_num_blocks, block_size, kv_heads, head_size) = cache.dims4()?;

    let cu_seq_lens_len = cu_seq_lens.dims1()?;
    let num_seqs = cu_seq_lens_len
        .checked_sub(1)
        .ok_or_else(|| candle_core::Error::msg("cu_seq_lens must contain an initial offset"))?;
    let num_tokens_i32 = i32::try_from(num_tokens)
        .map_err(|_| candle_core::Error::msg("num_tokens exceeds the kernel i32 limit"))?;
    let num_seqs_i32 = i32::try_from(num_seqs)
        .map_err(|_| candle_core::Error::msg("num_seqs exceeds the kernel i32 limit"))?;

    if num_tokens == 0 {
        return Tensor::zeros((0, kv_heads, head_size), dtype, cache.device());
    }

    let out = Tensor::zeros((num_tokens, kv_heads, head_size), dtype, cache.device())?;

    {
        let (c_s, c_l) = cache.storage_and_layout();
        let c_s = match &*c_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("cache must be a cuda tensor"),
        };
        let (o_s, o_l) = out.storage_and_layout();
        let o_s = match &*o_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("out must be a cuda tensor"),
        };

        let ((c_ptr, _c_guard), (o_ptr, _o_guard)) = match dtype {
            DType::F16 => (
                slice_ptr(c_s.as_cuda_slice::<half::f16>()?, c_l.start_offset()),
                slice_ptr(o_s.as_cuda_slice::<half::f16>()?, o_l.start_offset()),
            ),
            DType::BF16 => (
                slice_ptr(c_s.as_cuda_slice::<half::bf16>()?, c_l.start_offset()),
                slice_ptr(o_s.as_cuda_slice::<half::bf16>()?, o_l.start_offset()),
            ),
            DType::F32 => (
                slice_ptr(c_s.as_cuda_slice::<f32>()?, c_l.start_offset()),
                slice_ptr(o_s.as_cuda_slice::<f32>()?, o_l.start_offset()),
            ),
            _ => unreachable!(),
        };

        let (bt_s, bt_l) = block_table.storage_and_layout();
        let bt_s = match &*bt_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("block_table must be a cuda tensor"),
        };
        let (bt_ptr, _bt_guard) = slice_ptr(bt_s.as_cuda_slice::<u32>()?, bt_l.start_offset());

        let (cu_s, cu_l) = cu_seq_lens.storage_and_layout();
        let cu_s = match &*cu_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("cu_seq_lens must be a cuda tensor"),
        };
        let (cu_ptr, _cu_guard) = if cu_seq_lens.dtype() == DType::I32 {
            slice_ptr(cu_s.as_cuda_slice::<i32>()?, cu_l.start_offset())
        } else {
            slice_ptr(cu_s.as_cuda_slice::<u32>()?, cu_l.start_offset())
        };

        let (_, block_table_stride) = bt_l.shape().dims2()?;
        let dev = c_s.device();

        unsafe {
            ffi_gather_plain_kv_cache(
                c_ptr as *const core::ffi::c_void,
                o_ptr as *const core::ffi::c_void,
                bt_ptr as *const i32,
                cu_ptr as *const i32,
                num_tokens_i32,
                num_seqs_i32,
                block_size as i32,
                block_table_stride as i32,
                kv_heads as i32,
                head_size as i32,
                dev.cuda_stream().cu_stream(),
                dtype_code,
            );
        }
    }

    Ok(out)
}
