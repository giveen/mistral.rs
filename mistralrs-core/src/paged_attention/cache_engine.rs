use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::config::{KvCacheLayout, ModelConfigLike};
use super::turbo_quant;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::flashinfer::{register_fa3_prefill_caches, Fa3PrefillWorkspaceRegistration};
use tracing::warn;

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn cuda_supports_fp8(device: &Device) -> bool {
    use candle_core::cuda::cudarc::driver::{result, sys};

    if !mistralrs_paged_attn::USE_FP8 {
        return false;
    }
    let Device::Cuda(cuda) = device else {
        return false;
    };
    let ordinal = cuda.cuda_stream().context().ordinal();
    #[allow(clippy::cast_possible_truncation)]
    let Ok(device) = result::device::get(ordinal as i32) else {
        return false;
    };
    unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .is_ok_and(|major| major >= 8)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
pub enum PagedCacheType {
    #[default]
    Auto,
    F8E4M3,
    /// WHT-rotated 4-bit Lloyd-Max quantized KV cache (~4.1 bits/value). See `turbo_quant`.
    ///
    /// Scaffolding only: `validate` always rejects this today because
    /// `mistralrs-paged-attn`'s `reshape_and_cache`/`gather_kv_cache`/paged-attention compute
    /// kernels don't understand packed sub-byte blocks yet, only f32/f16/bf16/f8e4m3 elements.
    Turbo4,
}

impl PagedCacheType {
    pub fn to_dtype(&self, act_dtype: DType) -> DType {
        match self {
            PagedCacheType::F8E4M3 => DType::F8E4M3,
            // Packed blocks are opaque bytes as far as candle is concerned.
            PagedCacheType::Turbo4 => DType::U8,
            PagedCacheType::Auto => act_dtype,
        }
    }

    pub fn validate(
        &self,
        act_dtype: DType,
        model_config: &dyn ModelConfigLike,
        device: &Device,
        layer_devices: &[Option<Device>],
    ) -> std::result::Result<(), String> {
        if *self == Self::Auto {
            return Ok(());
        }
        if *self == Self::Turbo4 {
            for layer_idx in 0..model_config.num_layers() {
                if !model_config.layer_has_paged_kv_cache(layer_idx) {
                    continue;
                }
                // Turbo4 allocates and dispatches its own packed 5D cache independently of
                // whichever native layout (Standard vs FlashInfer) the model would otherwise
                // prefer -- `CacheEngine::allocate_gpu_cache` and `PagedAttention::forward_turbo4`
                // never consult `kv_cache_layout_for_layer` at all. Only MLA is structurally
                // incompatible (different per-token K/V shape: kv_lora_rank/kpe_head_dim, not a
                // per-head vector Turbo4 can rotate/quantize).
                if let KvCacheLayout::Mla { .. } = model_config.kv_cache_layout_for_layer(layer_idx)
                {
                    return Err(format!(
                        "Turbo4 KV cache does not support the Mla layout (layer {layer_idx})"
                    ));
                }
                let k_head_dim = model_config.k_head_dim_for_layer(layer_idx);
                let v_head_dim = model_config.v_head_dim_for_layer(layer_idx);
                if !k_head_dim.is_multiple_of(turbo_quant::GROUP)
                    || !v_head_dim.is_multiple_of(turbo_quant::GROUP)
                {
                    return Err(format!(
                        "Turbo4 KV cache requires head_dim (got k={k_head_dim}, v={v_head_dim} on layer {layer_idx}) \
                         to be a multiple of the {}-element rotation group",
                        turbo_quant::GROUP
                    ));
                }
            }
            // Writes go through `PagedAttention::forward_turbo4`'s scatter/gather + eager-attention
            // path, not the native reshape_and_cache/paged-attention kernels (which only understand
            // f32/f16/bf16/f8e4m3 element layouts). It doesn't support donor-cache (speculative
            // decoding) attention yet, or MLA (checked above).
            //
            // Fixed-precision 4-bit quantization is lossy in a way plain per-vector fidelity checks
            // don't reveal: softmax attention amplifies small per-key errors non-linearly (most
            // acutely in early layers where attention is more diffuse), and that per-layer error
            // compounds through the residual stream. This was first found by measuring complete
            // output incoherence on Qwen2.5 despite ~99.5% per-vector round-trip cosine similarity --
            // traced to Qwen2.5's high GQA ratio (query-heads-to-kv-heads) amplifying K's
            // quantization error, which `turbo4_layer_plan`'s auto-asymmetric fallback (below) now
            // catches automatically. That fix is ported from and empirically validated by the
            // upstream reference (PPL 2887 -> normal at the same threshold used here), but it's a
            // measured mitigation for one specific mechanism, not a guarantee against every way a
            // fixed 4-bit codebook can misbehave on a model it wasn't validated against.
            let representative_layer =
                (0..model_config.num_layers()).find(|&i| model_config.layer_has_paged_kv_cache(i));
            if let Some(layer_idx) = representative_layer {
                let plan = turbo4_layer_plan(model_config, layer_idx);
                if !plan.k_is_turbo {
                    let num_attn_heads = model_config.num_attn_heads_for_layer(layer_idx);
                    let num_kv_heads = model_config.num_kv_heads_for_layer(layer_idx);
                    warn!(
                        "Turbo4 auto-asymmetric: GQA ratio {}:1 (q_heads={num_attn_heads}, \
                         kv_heads={num_kv_heads}) -- upgrading K from turbo4 to a plain \
                         (unquantized) cache to prevent quality degradation. Disable with \
                         MISTRALRS_TURBO4_AUTO_ASYMMETRIC=0.",
                        if num_kv_heads > 0 {
                            num_attn_heads / num_kv_heads
                        } else {
                            0
                        }
                    );
                }
            }
            warn!(
                "Turbo4 KV cache uses lossy fixed-precision 4-bit quantization. Validate output \
                 quality for your specific model before relying on this in production; if it \
                 degrades, F8E4M3 is a safer lossy option."
            );
            return Ok(());
        }
        if !matches!(act_dtype, DType::F16 | DType::BF16 | DType::F32) {
            return Err(format!(
                "FP8 KV cache requires f16, bf16, or f32 activations, got {act_dtype:?}"
            ));
        }

        for layer_idx in 0..model_config.num_layers() {
            if !model_config.layer_has_paged_kv_cache(layer_idx) {
                continue;
            }
            if matches!(
                model_config.kv_cache_layout_for_layer(layer_idx),
                KvCacheLayout::Mla { .. }
            ) {
                return Err(format!(
                    "FP8 KV cache is not supported for MLA layer {layer_idx}"
                ));
            }
            let layer_device = layer_devices
                .get(layer_idx)
                .and_then(Option::as_ref)
                .unwrap_or(device);
            if layer_device.is_cuda() {
                #[cfg(all(feature = "cuda", target_family = "unix"))]
                if !cuda_supports_fp8(layer_device) {
                    return Err(
                        "FP8 KV cache requires CUDA compute capability 8.0 or newer and a matching CUDA build"
                            .to_string(),
                    );
                }
                #[cfg(not(all(feature = "cuda", target_family = "unix")))]
                return Err("FP8 KV cache requires the CUDA paged-attention backend".to_string());
            } else if layer_device.is_metal() {
                #[cfg(not(feature = "metal"))]
                return Err("FP8 KV cache requires the Metal paged-attention backend".to_string());
            } else {
                return Err(format!(
                    "FP8 KV cache is only supported on CUDA or Metal, got {layer_device:?} for layer {layer_idx}"
                ));
            }
        }
        Ok(())
    }
}

/// Per-layer, per-side effective Turbo4 format: either the packed 4-bit format, or a plain
/// elementwise fallback in the model's own compute dtype (see `turbo_quant::write_plain_cache`).
/// Two independent mitigations, both ported from the upstream reference after
/// it turned out they're what makes Turbo4 safe to use on at least some real models in the
/// first place (see `paged_attention.rs` module docs and the `turbo_quant` module docs):
///
/// - Auto-asymmetric: Turbo4's K quantization error gets amplified by the GQA broadcast factor
///   (every query head in a group re-uses the same slightly-off key), so models with a high
///   query-heads-to-kv-heads ratio see much worse degradation on K than on V. Upstream measured
///   this catastrophically on Qwen2.5 (GQA ratio 7:1, PPL 2887 against a 7.4 baseline) while a
///   lower ratio (4:1) was fine. Default-on above ratio 6, matching upstream's threshold;
///   disable with `MISTRALRS_TURBO4_AUTO_ASYMMETRIC=0`.
/// - Layer-adaptive: boundary layers (first/last) tend to be more sensitive to quantization
///   noise -- attention there is often more diffuse, so softmax has less of a dominant key to
///   fall back on when one candidate's score gets perturbed. Opt-in via
///   `MISTRALRS_TURBO4_LAYER_ADAPTIVE` (0 = off/default, 1 = first+last 4 layers upgraded, 2 =
///   last 8 layers upgraded), matching upstream's modes 1/2 (modes 5-7 are V-only bit-width
///   tradeoffs between turbo2/turbo4 that don't apply to a turbo4-only port).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Turbo4LayerPlan {
    pub k_is_turbo: bool,
    pub v_is_turbo: bool,
}

/// GQA ratio threshold above which auto-asymmetric upgrades K, matching upstream exactly (see
/// the measured Qwen2.5 PPL cliff in the `Turbo4LayerPlan` docs above).
const AUTO_ASYMMETRIC_GQA_THRESHOLD: usize = 6;

/// Pure arithmetic behind auto-asymmetric, split out from [`turbo4_layer_plan`] so it's testable
/// without touching process-wide env vars (Rust tests run concurrently in one process, so two
/// tests racing to set/unset the same env var would be flaky in a way that has nothing to do
/// with whether this logic is correct).
fn auto_asymmetric_upgrades_k(gqa_ratio: usize, disabled: bool) -> bool {
    !disabled && gqa_ratio >= AUTO_ASYMMETRIC_GQA_THRESHOLD
}

/// Pure arithmetic behind layer-adaptive's boundary-layer test, split out for the same reason as
/// [`auto_asymmetric_upgrades_k`]. Modes 5-7 (V-only turbo2/turbo4 bit-width tradeoffs) aren't
/// modeled since this is a turbo4-only port; anything but 1 or 2 is a no-op, matching upstream's
/// "unrecognized/off" fallthrough.
fn layer_is_boundary(adaptive_mode: i32, layer_idx: usize, n_layer: usize) -> bool {
    if n_layer < 8 {
        return false;
    }
    match adaptive_mode {
        1 => layer_idx < 4 || layer_idx >= n_layer - 4,
        2 => layer_idx >= n_layer.saturating_sub(8),
        _ => false,
    }
}

pub(crate) fn turbo4_layer_plan(
    model_config: &dyn ModelConfigLike,
    layer_idx: usize,
) -> Turbo4LayerPlan {
    let num_attn_heads = model_config.num_attn_heads_for_layer(layer_idx);
    let num_kv_heads = model_config.num_kv_heads_for_layer(layer_idx);
    let gqa_ratio = if num_kv_heads > 0 {
        num_attn_heads / num_kv_heads
    } else {
        1
    };
    let auto_asymmetric_disabled =
        std::env::var("MISTRALRS_TURBO4_AUTO_ASYMMETRIC").ok().as_deref() == Some("0");
    let mut k_is_turbo = !auto_asymmetric_upgrades_k(gqa_ratio, auto_asymmetric_disabled);
    let mut v_is_turbo = true;

    let adaptive_mode: i32 = std::env::var("MISTRALRS_TURBO4_LAYER_ADAPTIVE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if layer_is_boundary(adaptive_mode, layer_idx, model_config.num_layers()) {
        k_is_turbo = false;
        v_is_turbo = false;
    }

    Turbo4LayerPlan {
        k_is_turbo,
        v_is_turbo,
    }
}

impl FromStr for PagedCacheType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "f8e4m3" => Ok(Self::F8E4M3),
            "turbo4" => Ok(Self::Turbo4),
            other => Err(format!(
                "Unexpected `PagedCacheType`, got `{other}` but expected `auto`, `f8e4m3`, and `turbo4`."
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
    pub kv_cache_group_ids: Vec<u32>,
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    _fa3_prefill_workspaces: Fa3PrefillWorkspaceRegistration,
    fa3_prefill_num_sm_by_layer: Vec<Option<usize>>,
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        cache_config
            .cache_type
            .validate(dtype, model_config, device, &layer_devices)
            .map_err(candle_core::Error::msg)?;
        let act_dtype = dtype;
        let dtype = cache_config.cache_type.to_dtype(dtype);
        let gpu_cache = Self::allocate_gpu_cache(
            model_config,
            cache_config,
            dtype,
            act_dtype,
            device,
            layer_devices,
        )?;
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        let fa3_prefill_workspaces = register_fa3_prefill_caches(&gpu_cache)?;
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        let fa3_prefill_num_sm_by_layer =
            Self::fa3_prefill_cache_coverage(model_config, cache_config, &gpu_cache)?;
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        let fa3_prefill_num_sm_by_layer = vec![None; model_config.num_layers()];
        Ok(Self {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            _fa3_prefill_workspaces: fa3_prefill_workspaces,
            fa3_prefill_num_sm_by_layer,
            gpu_cache: Arc::new(Mutex::new(gpu_cache)),
        })
    }

    pub fn fa3_prefill_num_sm_by_layer(&self) -> &[Option<usize>] {
        &self.fa3_prefill_num_sm_by_layer
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        // Use blocking lock instead of busy-wait spin loop to avoid CPU waste
        // and potential thread starvation issues
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        act_dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let mut gpu_cache = Vec::new();

        for (layer_idx, device) in layer_devices
            .iter()
            .take(model_config.num_layers())
            .map(|x| x.as_ref().unwrap_or(device))
            .enumerate()
        {
            // Hybrid models keep no paged cache on linear/recurrent layers, but the vec stays indexed
            // by absolute layer index, so those get an empty tensor of the right rank instead.
            let num_gpu_blocks = if model_config.layer_has_paged_kv_cache(layer_idx) {
                cache_config.num_gpu_blocks
            } else {
                0
            };

            if cache_config.cache_type == PagedCacheType::Turbo4 {
                // Shape validated (head_dim % GROUP == 0, Standard layout) by `validate` before
                // `CacheEngine::new` ever calls this. `PagedAttention::forward_turbo4` reshapes
                // this block-indexed tensor to a flat (slots, ...) view to write into it.
                let num_kv_heads = model_config.num_kv_heads_for_layer(layer_idx);
                let k_head_dim = model_config.k_head_dim_for_layer(layer_idx);
                let v_head_dim = model_config.v_head_dim_for_layer(layer_idx);
                let plan = turbo4_layer_plan(model_config, layer_idx);
                let packed_shape = |head_dim: usize| {
                    (
                        num_gpu_blocks,
                        cache_config.block_size,
                        num_kv_heads,
                        head_dim / turbo_quant::GROUP,
                        turbo_quant::BLOCK_TURBO4_BYTES,
                    )
                };
                let plain_shape =
                    |head_dim: usize| (num_gpu_blocks, cache_config.block_size, num_kv_heads, head_dim);
                // The fallback cache is the model's own compute dtype (BF16/F16/F32), not
                // F8E4M3 -- see write_plain_cache's docs for why (candle's CUDA backend has no
                // generic float->F8E4M3 cast kernel to write into it with).
                let key_blocks = if plan.k_is_turbo {
                    Tensor::zeros(packed_shape(k_head_dim), DType::U8, device)?
                } else {
                    Tensor::zeros(plain_shape(k_head_dim), act_dtype, device)?
                };
                let value_blocks = if plan.v_is_turbo {
                    Tensor::zeros(packed_shape(v_head_dim), DType::U8, device)?
                } else {
                    Tensor::zeros(plain_shape(v_head_dim), act_dtype, device)?
                };
                gpu_cache.push((key_blocks, value_blocks));
                continue;
            }

            let requested_kv_cache_layout = model_config.kv_cache_layout_for_layer(layer_idx);
            let kv_cache_layout =
                if matches!(requested_kv_cache_layout, KvCacheLayout::FlashInferHnd)
                    && !device.is_cuda()
                {
                    KvCacheLayout::Standard
                } else {
                    requested_kv_cache_layout
                };
            let (key_blocks, value_blocks) = match kv_cache_layout {
                KvCacheLayout::Standard | KvCacheLayout::StandardNoFlashInfer => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                        layer_idx,
                    );
                    let value_block_shape = Self::calculate_value_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2
                                * key_block_shape.3;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * value_block_shape.0
                                * value_block_shape.1
                                * value_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::FlashInferHnd => {
                    let key_block_shape = Self::calculate_flashinfer_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    let value_blocks = unsafe {
                        Tensor::empty(
                            (
                                num_gpu_blocks,
                                key_block_shape.0,
                                key_block_shape.1,
                                key_block_shape.2,
                            ),
                            dtype,
                            device,
                        )?
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::Mla {
                    kv_lora_rank,
                    kpe_head_dim,
                } => {
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count =
                                num_gpu_blocks * cache_config.block_size * kv_lora_rank;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (num_gpu_blocks, cache_config.block_size, kv_lora_rank),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count =
                                num_gpu_blocks * cache_config.block_size * kpe_head_dim;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (num_gpu_blocks, cache_config.block_size, kpe_head_dim),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    fn fa3_prefill_cache_coverage(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        gpu_cache: &[KVCache],
    ) -> Result<Vec<Option<usize>>> {
        if !mistralrs_paged_attn::USE_FA3_FP8_PAGED || cache_config.num_gpu_blocks == 0 {
            return Ok(vec![None; model_config.num_layers()]);
        }
        (0..model_config.num_layers())
            .map(|layer_idx| -> Result<Option<usize>> {
                if !model_config.layer_has_paged_kv_cache(layer_idx) {
                    return Ok(None);
                }
                let Some((key_cache, value_cache)) = gpu_cache.get(layer_idx) else {
                    return Ok(None);
                };
                let kv_heads = model_config.num_kv_heads_for_layer(layer_idx);
                let q_heads = model_config.num_attn_heads_for_layer(layer_idx);
                let k_head_dim = model_config.k_head_dim_for_layer(layer_idx);
                let v_head_dim = model_config.v_head_dim_for_layer(layer_idx);
                let expected_shape = (
                    cache_config.num_gpu_blocks,
                    kv_heads,
                    cache_config.block_size,
                    k_head_dim,
                );
                if model_config.kv_cache_layout_for_layer(layer_idx) != KvCacheLayout::FlashInferHnd
                    || kv_heads == 0
                    || q_heads == 0
                    || !q_heads.is_multiple_of(kv_heads)
                    || k_head_dim != 256
                    || v_head_dim != k_head_dim
                    || key_cache.dtype() != DType::F8E4M3
                    || value_cache.dtype() != DType::F8E4M3
                    || key_cache.dims4().ok() != Some(expected_shape)
                    || value_cache.dims4().ok() != Some(expected_shape)
                {
                    return Ok(None);
                }
                crate::flashinfer::fa3_prefill_cache_num_sm(
                    key_cache,
                    value_cache,
                    q_heads,
                    kv_heads,
                    k_head_dim,
                    cache_config.block_size,
                )
            })
            .collect()
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.k_head_dim_for_layer(layer_idx) / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.v_head_dim_for_layer(layer_idx),
            block_size,
        )
    }

    fn calculate_flashinfer_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            block_size,
            model_config.k_head_dim_for_layer(layer_idx),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::config::ModelConfigMetadata;

    fn model_config(layout: KvCacheLayout) -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 4096,
            num_layers: 2,
            hidden_size: 1024,
            num_kv_heads: 4,
            num_attn_heads: 16,
            sliding_window: None,
            k_head_dim: 128,
            v_head_dim: 128,
            kv_cache_layout: layout,
        }
    }

    #[test]
    fn fp8_cache_rejects_cpu_before_allocation() {
        let err = PagedCacheType::F8E4M3
            .validate(
                DType::BF16,
                &model_config(KvCacheLayout::Standard),
                &Device::Cpu,
                &[],
            )
            .unwrap_err();
        assert!(err.contains("only supported on CUDA or Metal"));
    }

    #[test]
    fn fp8_cache_rejects_mla_before_allocation() {
        let err = PagedCacheType::F8E4M3
            .validate(
                DType::BF16,
                &model_config(KvCacheLayout::Mla {
                    kv_lora_rank: 512,
                    kpe_head_dim: 64,
                }),
                &Device::Cpu,
                &[],
            )
            .unwrap_err();
        assert!(err.contains("not supported for MLA layer 0"));
    }

    #[test]
    fn turbo4_cache_accepts_valid_standard_layout() {
        PagedCacheType::Turbo4
            .validate(
                DType::BF16,
                &model_config(KvCacheLayout::Standard),
                &Device::Cpu,
                &[],
            )
            .unwrap();
    }

    #[test]
    fn turbo4_cache_rejects_mla_layout() {
        let err = PagedCacheType::Turbo4
            .validate(
                DType::BF16,
                &model_config(KvCacheLayout::Mla {
                    kv_lora_rank: 512,
                    kpe_head_dim: 64,
                }),
                &Device::Cpu,
                &[],
            )
            .unwrap_err();
        assert!(err.contains("does not support the Mla"));
    }

    #[test]
    fn turbo4_cache_rejects_head_dim_not_a_multiple_of_the_group_size() {
        let mut model = model_config(KvCacheLayout::Standard);
        model.k_head_dim = 96;
        let err = PagedCacheType::Turbo4
            .validate(DType::BF16, &model, &Device::Cpu, &[])
            .unwrap_err();
        assert!(err.contains("multiple of the 128-element rotation group"));
    }

    #[test]
    fn turbo4_cache_allocates_packed_u8_blocks() -> Result<()> {
        let model = model_config(KvCacheLayout::Standard);
        let cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 3,
            cache_type: PagedCacheType::Turbo4,
            kv_cache_group_ids: vec![0],
        };
        let engine = CacheEngine::new(&model, &cache, DType::BF16, &Device::Cpu, vec![None; 2])?;
        let kv_cache = engine.get_kv_cache();
        assert_eq!(kv_cache.len(), 2);
        for (key_blocks, value_blocks) in kv_cache.iter() {
            assert_eq!(key_blocks.dtype(), DType::U8);
            assert_eq!(value_blocks.dtype(), DType::U8);
            // (num_gpu_blocks, block_size, kv_heads, groups_per_head, BLOCK_TURBO4_BYTES)
            assert_eq!(key_blocks.dims(), &[3, 32, 4, 1, 66]);
            assert_eq!(value_blocks.dims(), &[3, 32, 4, 1, 66]);
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fa3_prefill_coverage_requires_compatible_registered_caches() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let mut model = model_config(KvCacheLayout::FlashInferHnd);
        model.k_head_dim = 256;
        model.v_head_dim = 256;
        let cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 2,
            cache_type: PagedCacheType::F8E4M3,
            kv_cache_group_ids: vec![0],
        };
        let engine = CacheEngine::new(
            &model,
            &cache,
            DType::BF16,
            &device,
            vec![None; model.num_layers],
        )?;
        let expected_num_sm = if mistralrs_paged_attn::USE_FA3_FP8_PAGED {
            crate::flashinfer::fa3_device_num_sm(&device)
        } else {
            None
        };
        assert_eq!(
            engine.fa3_prefill_num_sm_by_layer(),
            &[expected_num_sm, expected_num_sm]
        );

        model.v_head_dim = 128;
        let incompatible = CacheEngine::new(
            &model,
            &cache,
            DType::BF16,
            &device,
            vec![None; model.num_layers],
        )?;
        assert_eq!(incompatible.fa3_prefill_num_sm_by_layer(), &[None, None]);
        Ok(())
    }

    #[test]
    fn auto_asymmetric_upgrades_k_only_above_the_gqa_threshold() {
        // Qwen2.5's measured-catastrophic config (see Turbo4LayerPlan docs): 4 kv heads / 28 q
        // heads = 7:1, above the threshold, should upgrade K.
        assert!(auto_asymmetric_upgrades_k(7, false));
        // The threshold itself is inclusive (upstream's `gqa_ratio >= 6`).
        assert!(auto_asymmetric_upgrades_k(6, false));
        // Mistral's measured-fine config: 8 kv heads / 32 q heads = 4:1, below threshold.
        assert!(!auto_asymmetric_upgrades_k(4, false));
        assert!(!auto_asymmetric_upgrades_k(1, false));
        // MISTRALRS_TURBO4_AUTO_ASYMMETRIC=0 disables it even above threshold.
        assert!(!auto_asymmetric_upgrades_k(7, true));
    }

    #[test]
    fn layer_is_boundary_matches_upstream_modes() {
        const N: usize = 28;
        // Mode 0 (off/default): never a boundary, regardless of layer count.
        assert!(!layer_is_boundary(0, 0, N));
        assert!(!layer_is_boundary(0, N - 1, N));
        // Mode 1: first 4 + last 4 layers.
        assert!(layer_is_boundary(1, 0, N));
        assert!(layer_is_boundary(1, 3, N));
        assert!(!layer_is_boundary(1, 4, N));
        assert!(!layer_is_boundary(1, N - 5, N));
        assert!(layer_is_boundary(1, N - 4, N));
        assert!(layer_is_boundary(1, N - 1, N));
        // Mode 2: last 8 layers only.
        assert!(!layer_is_boundary(2, 0, N));
        assert!(!layer_is_boundary(2, N - 9, N));
        assert!(layer_is_boundary(2, N - 8, N));
        assert!(layer_is_boundary(2, N - 1, N));
        // Both modes are a no-op below upstream's 8-layer minimum, regardless of position.
        assert!(!layer_is_boundary(1, 0, 4));
        assert!(!layer_is_boundary(2, 3, 4));
    }

    #[test]
    fn turbo4_layer_plan_matches_qwen_family_measurements() {
        // Qwen3-8B-shaped: 32 q heads / 8 kv heads = 4:1, below threshold -- both sides stay
        // packed. This is the config that was verified end to end against a real model.
        let qwen3_shaped = model_config(KvCacheLayout::Standard);
        let plan = turbo4_layer_plan(&qwen3_shaped, 0);
        assert!(plan.k_is_turbo && plan.v_is_turbo);

        // Qwen2.5-shaped: 12 q heads / 2 kv heads = 6:1, at/above threshold -- K upgrades to
        // the plain fallback, V stays packed. This is the exact config that measured PPL 2887
        // upstream and produced complete garbage in our own testing before this fallback existed.
        let mut qwen2_5_shaped = model_config(KvCacheLayout::Standard);
        qwen2_5_shaped.num_attn_heads = 12;
        qwen2_5_shaped.num_kv_heads = 2;
        let plan = turbo4_layer_plan(&qwen2_5_shaped, 0);
        assert!(!plan.k_is_turbo && plan.v_is_turbo);
    }
}
