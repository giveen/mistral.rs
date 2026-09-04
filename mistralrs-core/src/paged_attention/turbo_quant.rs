//! Data format for a TurboQuant-style paged KV cache entry: a fixed random rotation
//! (signed Walsh-Hadamard transform) followed by fixed Lloyd-Max scalar quantization.
//!
//! Ported from the CUDA reference in a llama.cpp fork
//! (github.com/TheTom/llama-cpp-turboquant, branch `feature/turboquant-kv-cache`,
//! `ggml/src/ggml-cuda/turbo-quant.cuh` / `turbo-wht.cu`). Constants are copied verbatim so
//! the encode/decode math matches that reference; nothing here talks to a GPU yet.
//!
//! Rotating a head's `k`/`v` vector spreads its energy evenly across dimensions so the
//! rotated coefficients are approximately Gaussian, which is what the fixed centroid tables
//! below assume. Rescaling by the vector's own L2 norm turns "quantize this head vector"
//! into "quantize a point on the unit sphere" (hence upstream calling this PolarQuant) against
//! centroids tuned for that per-coordinate distribution, and the norm is stored alongside so
//! dequantization can scale back up.
//!
//! `PagedCacheType::Turbo4` is wired into the real PagedAttention dispatch path (see
//! `layers::paged_attention::forward_turbo4`). The write side and CPU/eager read fallback go
//! through the vectorized `Tensor`-op functions below (`quantize_turbo4_tensor`,
//! `write_turbo4_cache`, `read_turbo4_cache`, ...); the batched CUDA decode path instead calls
//! the fused `mistralrs_paged_attn::gather_turbo4_cache` kernel, which reimplements this same
//! rotation/dequant math directly on-GPU (kept in sync via
//! `gather_turbo4_cache_kernel_matches_eager_reference`). `quantize_turbo4`/`dequantize_turbo4`
//! (the scalar, non-vectorized reference) are CPU-test-only.
#![allow(dead_code)]

use std::sync::{Mutex, OnceLock};

use candle_core::{DType, Device, Result, Tensor, D};
use half::f16;

/// Rotation group size: one block covers a full attention head (`head_dim == 128`).
pub(crate) const GROUP: usize = 128;

pub(crate) const BLOCK_TURBO4_BYTES: usize = 2 + GROUP / 2;

/// One quantized head-vector: an fp16 L2 norm plus 128 nibble-packed 4-bit centroid indices.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct BlockTurbo4 {
    pub norm: f16,
    pub qs: [u8; GROUP / 2],
}

const _: () = assert!(std::mem::size_of::<BlockTurbo4>() == BLOCK_TURBO4_BYTES);

// ---- WHT sign arrays (seed=42), ported verbatim from turbo-quant.cuh ----
#[rustfmt::skip]
const WHT_SIGNS1: [f32; GROUP] = [
    -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
    1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
    -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
    1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
];
#[rustfmt::skip]
const WHT_SIGNS2: [f32; GROUP] = [
    1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
];

// ---- 4-bit Lloyd-Max centroids/midpoints for N(0, 1/128), ported verbatim ----
const CENTROIDS_4BIT: [f32; 16] = [
    -0.241529, -0.182877, -0.143016, -0.111036, -0.083292, -0.058050, -0.034299, -0.011349,
    0.011349, 0.034299, 0.058050, 0.083292, 0.111036, 0.143016, 0.182877, 0.241529,
];
const MID_4BIT: [f32; 15] = [
    -0.212203, -0.162947, -0.127026, -0.097164, -0.070671, -0.046174, -0.022824, 0.000000,
    0.022824, 0.046174, 0.070671, 0.097164, 0.127026, 0.162947, 0.212203,
];

const INV_SQRT_GROUP: f32 = 0.088_388_35; // 1 / sqrt(128)

/// In-place radix-2 Walsh-Hadamard butterfly, normalized so the transform is its own inverse.
fn fwht_128(x: &mut [f32; GROUP]) {
    let mut h = 1;
    while h < GROUP {
        let mut i = 0;
        while i < GROUP {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    for v in x.iter_mut() {
        *v *= INV_SQRT_GROUP;
    }
}

/// `signs1 -> WHT -> signs2`, matching `turbo_rotate_forward` in turbo-quant.cuh.
pub(crate) fn rotate_forward(x: &mut [f32; GROUP]) {
    for i in 0..GROUP {
        x[i] *= WHT_SIGNS1[i];
    }
    fwht_128(x);
    for i in 0..GROUP {
        x[i] *= WHT_SIGNS2[i];
    }
}

/// The exact inverse of [`rotate_forward`]. The forward rotation is `S2 . H . S1`, and it's
/// orthogonal: `H` (normalized Hadamard) and `S1`/`S2` (diagonal +-1) are each their own
/// inverse, so the inverse of the product is `S1 . H . S2`.
pub(crate) fn rotate_inverse(x: &mut [f32; GROUP]) {
    for i in 0..GROUP {
        x[i] *= WHT_SIGNS2[i];
    }
    fwht_128(x);
    for i in 0..GROUP {
        x[i] *= WHT_SIGNS1[i];
    }
}

fn l2_norm(x: &[f32; GROUP]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Index of the nearest centroid, given the ascending midpoints between consecutive centroids.
/// `mids.len()` is always 15 (one short of the 16 4-bit centroids), so the count fits in a u8.
fn nearest_centroid(val: f32, mids: &[f32]) -> u8 {
    u8::try_from(mids.iter().filter(|&&m| val >= m).count()).expect("mids.len() < 256")
}

/// Quantize one already-rotated 128-element head vector (i.e. after [`rotate_forward`]).
///
/// Stores a *norm-corrected* scale rather than the raw pre-quantization norm: independently
/// rounding each coordinate to its nearest centroid systematically shrinks the reconstructed
/// vector's magnitude (a well-known scalar-quantization bias), so `norm` alone would leave
/// every dequantized vector slightly undersized. Instead this computes `recon_norm`, the norm
/// of the vector the *chosen* centroids actually reconstruct, and stores `norm / recon_norm` so
/// that `dequantize_turbo4`'s `centroid[idx] * stored_norm` reproduces the original magnitude
/// exactly (`||dequantized|| == ||rotated||`) while the direction is whatever the 4-bit
/// quantization gave. Ported from the upstream reference's `corrected_norm` (see module docs).
pub(crate) fn quantize_turbo4(rotated: &[f32; GROUP]) -> BlockTurbo4 {
    let norm = l2_norm(rotated).max(f32::MIN_POSITIVE);
    let inv_norm = 1.0 / norm;
    let mut qs = [0u8; GROUP / 2];
    let mut recon_sq = 0f32;
    for j in 0..GROUP {
        let idx = nearest_centroid(rotated[j] * inv_norm, &MID_4BIT);
        qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
        let c = CENTROIDS_4BIT[idx as usize];
        recon_sq += c * c;
    }
    let recon_norm = recon_sq.sqrt().max(f32::MIN_POSITIVE);
    let corrected_norm = norm / recon_norm;
    BlockTurbo4 {
        norm: f16::from_f32(corrected_norm),
        qs,
    }
}

/// Dequantize back to a rotated 128-element head vector (still needs [`rotate_inverse`]).
pub(crate) fn dequantize_turbo4(block: &BlockTurbo4) -> [f32; GROUP] {
    let norm = block.norm.to_f32();
    let mut out = [0f32; GROUP];
    for (j, out_val) in out.iter_mut().enumerate() {
        let idx = ((block.qs[j / 2] >> ((j % 2) * 4)) & 0xF) as usize;
        *out_val = CENTROIDS_4BIT[idx] * norm;
    }
    out
}

/// The fixed forward rotation as a `GROUP x GROUP` matrix, `R[:, i] = rotate_forward(e_i)`,
/// built directly from the scalar reference above so the two can never drift apart. `R` is
/// orthogonal (`rotate_forward`/`rotate_inverse` are exact inverses), so the inverse rotation
/// is just `R^T`.
fn rotation_matrix() -> &'static [f32] {
    static MATRIX: OnceLock<Vec<f32>> = OnceLock::new();
    MATRIX.get_or_init(|| {
        let mut r = vec![0f32; GROUP * GROUP];
        for col in 0..GROUP {
            let mut e = [0f32; GROUP];
            e[col] = 1.0;
            rotate_forward(&mut e);
            for (row, &val) in e.iter().enumerate() {
                r[row * GROUP + col] = val;
            }
        }
        r
    })
}

fn rotation_matrix_tensor(device: &Device) -> Result<Tensor> {
    Tensor::from_slice(rotation_matrix(), (GROUP, GROUP), device)
}

fn centroids_4bit_tensor(device: &Device) -> Result<Tensor> {
    Tensor::from_slice(&CENTROIDS_4BIT[..], (CENTROIDS_4BIT.len(),), device)
}

/// Vectorized, backend-agnostic version of [`quantize_turbo4`]/[`rotate_forward`] combined:
/// quantizes every length-`GROUP` vector along `x`'s last dimension in one shot via
/// `candle_core::Tensor` ops (so it runs through whichever backend `x` lives on), rather than
/// the scalar reference's per-element loop.
///
/// `x` may have any leading shape (e.g. `(tokens, kv_heads, GROUP)`) and any float dtype;
/// internally the math runs in f32. Returns `(norm, packed)`: `norm` has `x`'s shape with the
/// last dim dropped, `packed` replaces it with `GROUP / 2` nibble-packed u8 bytes. Unlike the
/// scalar [`quantize_turbo4`], this expects **un-rotated** input and does the forward rotation
/// itself.
pub(crate) fn quantize_turbo4_tensor(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let Some(&last) = x.dims().last() else {
        candle_core::bail!("quantize_turbo4_tensor expects a tensor with at least one dimension");
    };
    if last != GROUP {
        candle_core::bail!("quantize_turbo4_tensor expects last dim {GROUP}, got {last}");
    }
    let device = x.device();
    let x = x.to_dtype(DType::F32)?;

    let r = rotation_matrix_tensor(device)?;
    // rotate_forward(v) = R @ v, so for a batch of row-vectors: rotated = x @ R^T.
    let rotated = x.broadcast_matmul(&r.t()?)?;
    let norm = rotated.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let eps = Tensor::new(f32::MIN_POSITIVE, device)?;
    let norm = norm.broadcast_maximum(&eps)?;
    let normalized = rotated.broadcast_div(&norm)?;

    let centroids = centroids_4bit_tensor(device)?;
    let diff = normalized
        .unsqueeze(D::Minus1)?
        .broadcast_sub(&centroids)?
        .abs()?; // (..., GROUP, 16)
    let idx = diff.argmin(D::Minus1)?.to_dtype(DType::F32)?; // (..., GROUP), values 0..16

    // Norm correction (matches the scalar quantize_turbo4): independently rounding each
    // coordinate to its nearest centroid systematically shrinks the reconstructed vector's
    // magnitude, so storing the raw pre-quantization norm would leave every dequantized vector
    // slightly undersized. Compute recon_norm, the norm of what the *chosen* centroids actually
    // reconstruct, and store norm / recon_norm instead so dequantize_turbo4_tensor's
    // `centroid[idx] * stored_norm` reproduces the original magnitude exactly.
    let flat_idx_u32 = idx.to_dtype(DType::U32)?.flatten_all()?;
    let chosen_vals = centroids
        .index_select(&flat_idx_u32, 0)?
        .reshape(idx.shape())?; // (..., GROUP)
    let recon_norm = chosen_vals.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let recon_norm = recon_norm.broadcast_maximum(&eps)?;
    let corrected_norm = norm.broadcast_div(&recon_norm)?;

    // Nibble-pack adjacent pairs arithmetically (byte = low + 16*high) so this stays pure
    // tensor arithmetic; candle doesn't expose bitwise ops on tensors.
    let mut pair_dims = idx.dims().to_vec();
    let g = pair_dims.pop().expect("checked non-empty above");
    pair_dims.push(g / 2);
    pair_dims.push(2);
    let idx_pairs = idx.reshape(pair_dims)?; // (..., GROUP/2, 2)
    let low = idx_pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let high = idx_pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    let packed = (low + (high * 16.0)?)?.to_dtype(DType::U8)?;

    Ok((corrected_norm.squeeze(D::Minus1)?, packed))
}

/// The inverse of [`quantize_turbo4_tensor`]: dequantizes and applies the inverse rotation, so
/// the result is directly comparable to the original input (no separate `rotate_inverse` call
/// needed, unlike the scalar [`dequantize_turbo4`]). `norm`/`packed` must be exactly what
/// `quantize_turbo4_tensor` returned (same shapes).
pub(crate) fn dequantize_turbo4_tensor(norm: &Tensor, packed: &Tensor) -> Result<Tensor> {
    let device = packed.device();
    let packed_f = packed.to_dtype(DType::F32)?;
    let sixteen = Tensor::new(16f32, device)?;
    let high = packed_f.broadcast_div(&sixteen)?.floor()?;
    let low = packed_f.broadcast_sub(&high.broadcast_mul(&sixteen)?)?;

    let stacked = Tensor::stack(&[&low, &high], D::Minus1)?; // (..., GROUP/2, 2)
    let mut out_dims = stacked.dims().to_vec();
    out_dims.pop();
    out_dims.pop();
    out_dims.push(GROUP);
    let idx = stacked.reshape(out_dims)?.to_dtype(DType::U32)?; // (..., GROUP)

    let centroids = centroids_4bit_tensor(device)?;
    let flat_idx = idx.flatten_all()?;
    let values = centroids.index_select(&flat_idx, 0)?.reshape(idx.shape())?;

    let rotated = values.broadcast_mul(&norm.unsqueeze(D::Minus1)?)?;
    let r = rotation_matrix_tensor(device)?;
    // rotate_inverse(v) = R^T @ v, so for a batch of row-vectors: original = rotated @ R.
    rotated.broadcast_matmul(&r)
}

/// Fixed-point range for the arithmetic 2-byte norm encoding below (candle has no bitwise
/// tensor ops, so this reuses the same base-256 "byte = low + 256*high" trick as nibble
/// packing). Generous headroom over realistic per-head L2 norms; the ~4096/65535 =~ 0.0625
/// absolute quantization step this adds is negligible next to the ~2-5% relative error the
/// 4-bit centroid quantization already carries.
const NORM_SCALE_MAX: f32 = 4096.0;
const NORM_LEVELS: f32 = 65535.0;

/// Pack `(norm, qs)` (the outputs of [`quantize_turbo4_tensor`]) into one U8 tensor whose last
/// dimension is `BLOCK_TURBO4_BYTES` (2 arithmetic-encoded norm bytes, then `qs`'s `GROUP/2`
/// bytes) so a single tensor per K/V side can hold everything a cache slot needs, matching the
/// `(key_cache, value_cache)` shape every paged-attention call site already expects.
pub(crate) fn pack_turbo4_block(norm: &Tensor, qs: &Tensor) -> Result<Tensor> {
    let device = norm.device();
    let level = norm
        .to_dtype(DType::F32)?
        .affine((NORM_LEVELS / NORM_SCALE_MAX).into(), 0.)?
        .broadcast_maximum(&Tensor::new(0f32, device)?)?
        .broadcast_minimum(&Tensor::new(NORM_LEVELS, device)?)?
        .round()?;
    let high = (level.clone() / 256.0)?.floor()?;
    let low = (level - (high.clone() * 256.0)?)?;
    let norm_bytes = Tensor::stack(&[&low, &high], D::Minus1)?
        .unsqueeze(D::Minus2)?
        .to_dtype(DType::U8)?; // (..., 1, 2)
    let mut norm_dims = norm_bytes.dims().to_vec();
    norm_dims.remove(norm_dims.len() - 2);
    *norm_dims.last_mut().unwrap() = 2;
    let norm_bytes = norm_bytes.reshape(norm_dims)?; // (..., 2)
    Tensor::cat(&[&norm_bytes, qs], D::Minus1)
}

/// The inverse of [`pack_turbo4_block`]: splits a packed `BLOCK_TURBO4_BYTES`-wide U8 tensor
/// back into `(norm, qs)`, ready for [`dequantize_turbo4_tensor`].
pub(crate) fn unpack_turbo4_block(packed: &Tensor) -> Result<(Tensor, Tensor)> {
    let last = packed.dims().last().copied().ok_or_else(|| {
        candle_core::Error::msg("unpack_turbo4_block expects a non-scalar tensor")
    })?;
    if last != BLOCK_TURBO4_BYTES {
        candle_core::bail!("unpack_turbo4_block expects last dim {BLOCK_TURBO4_BYTES}, got {last}");
    }
    let norm_bytes = packed.narrow(D::Minus1, 0, 2)?.to_dtype(DType::F32)?;
    let qs = packed.narrow(D::Minus1, 2, GROUP / 2)?;
    let low = norm_bytes.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let high = norm_bytes.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    let level = (low + (high * 256.0)?)?;
    let norm = level.affine((NORM_SCALE_MAX / NORM_LEVELS).into(), 0.)?;
    Ok((norm, qs))
}

// ---- InnerQ: per-channel equalization, calibrated from live K data ----
//
// A single fixed rotation can't fully Gaussianize a channel that's a genuine, consistent
// outlier across every vector it rotates (a "massive activation" channel, well documented in
// real transformers) -- unlike per-vector-random noise, a structural per-channel imbalance
// survives a fixed rotation because it's the same shape on every input. InnerQ equalizes K's
// per-channel RMS before the WHT rotation, calibrated from the first few real K vectors seen,
// so that no single channel's outlier magnitude dominates the 4-bit centroid budget. Query gets
// the exact inverse scale applied before the dot product: `<Q*scale_inv, K*scale> = <Q,K>`, so
// this changes nothing about attention's output beyond quantization fidelity itself. V is left
// alone -- unlike K, correcting V's scale would require compensating the attention *output*
// (equivalent to patching the o_proj weight), which the upstream reference does not appear to
// apply InnerQ to either (its calibration comment is K-specific).
//
// Ported from the upstream reference's InnerQ (see module docs); enabled via
// `MISTRALRS_TURBO4_INNERQ=<N>` (N = number of K vectors to calibrate on, matching the
// reference's `TURBO_INNERQ` env var) and `MISTRALRS_TURBO4_INNERQ_STRENGTH` (default 0.5,
// matching `TURBO_INNERQ_STRENGTH`). Disabled (a pure passthrough) unless the first env var is
// set, matching upstream's opt-in default.

#[derive(Clone, Copy)]
enum InnerQMode {
    Disabled,
    Calibrating { target: usize, strength: f32 },
    Active,
}

struct InnerQState {
    mode: InnerQMode,
    sq_accum: [f32; GROUP],
    count: usize,
    scale: [f32; GROUP],
    scale_inv: [f32; GROUP],
}

impl InnerQState {
    fn from_env() -> Self {
        let target = std::env::var("MISTRALRS_TURBO4_INNERQ")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&t| t > 0);
        let strength = std::env::var("MISTRALRS_TURBO4_INNERQ_STRENGTH")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|&s| s > 0.0 && s <= 1.0)
            .unwrap_or(0.5);
        let mode = match target {
            Some(target) => {
                tracing::info!(
                    "Turbo4 InnerQ calibration started (target={target} vectors, strength={strength:.2})"
                );
                InnerQMode::Calibrating { target, strength }
            }
            None => InnerQMode::Disabled,
        };
        InnerQState {
            mode,
            sq_accum: [0.0; GROUP],
            count: 0,
            scale: [1.0; GROUP],
            scale_inv: [1.0; GROUP],
        }
    }

    /// Compute per-channel scale/scale_inv from the accumulated calibration stats, clamped to
    /// [0.5, 2.0] the same as upstream (a single WHT-rotation-worth of correction shouldn't try
    /// to fix more than a 4x channel imbalance -- that's InnerQ compensating for structural
    /// outliers, not silently absorbing a badly-conditioned model). Auto-disables if channels
    /// are already balanced (max ratio < 1.2), matching upstream's auto-skip.
    fn finalize(&mut self, strength: f32) {
        if self.count == 0 {
            tracing::warn!("Turbo4 InnerQ calibration got 0 vectors, disabling");
            self.mode = InnerQMode::Disabled;
            return;
        }
        let count = self.count as f32;
        let mut rms = [0f32; GROUP];
        let mut mean_rms = 0f32;
        for i in 0..GROUP {
            rms[i] = (self.sq_accum[i] / count).sqrt();
            mean_rms += rms[i];
        }
        mean_rms /= GROUP as f32;

        let mut max_ratio = 0f32;
        let mut min_ratio = f32::MAX;
        for i in 0..GROUP {
            let ratio = if rms[i] > 1e-10 { mean_rms / rms[i] } else { 1.0 };
            let s = ratio.powf(strength).clamp(0.5, 2.0);
            self.scale[i] = s;
            self.scale_inv[i] = 1.0 / s;
            max_ratio = max_ratio.max(ratio);
            min_ratio = min_ratio.min(ratio);
        }

        if max_ratio < 1.2 && min_ratio > 1.0 / 1.2 {
            tracing::info!(
                "Turbo4 InnerQ auto-disabled (channels already balanced, max_ratio={max_ratio:.3})"
            );
            self.mode = InnerQMode::Disabled;
            return;
        }

        tracing::info!(
            "Turbo4 InnerQ finalized ({} vectors, max_ratio={max_ratio:.3}, min_ratio={min_ratio:.3})",
            self.count
        );
        self.mode = InnerQMode::Active;
    }
}

fn innerq_state() -> &'static Mutex<InnerQState> {
    static STATE: OnceLock<Mutex<InnerQState>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(InnerQState::from_env()))
}

/// Calibrates on (while calibrating) or applies the finalized per-channel scale to (once active)
/// `k` before quantization. `k`'s last dimension must be `GROUP`; every other leading dimension
/// is an independent calibration sample. A no-op unless `MISTRALRS_TURBO4_INNERQ` is set.
pub(crate) fn innerq_scale_k(k: &Tensor) -> Result<Tensor> {
    let mut state = innerq_state().lock().expect("InnerQ state mutex poisoned");
    match state.mode {
        InnerQMode::Disabled => Ok(k.clone()),
        InnerQMode::Calibrating { target, strength } => {
            let flat = k.to_dtype(DType::F32)?.reshape(((), GROUP))?;
            let n = flat.dim(0)?;
            let sq_sum: Vec<f32> = flat.sqr()?.sum(0)?.to_vec1()?;
            for (accum, sq) in state.sq_accum.iter_mut().zip(sq_sum) {
                *accum += sq;
            }
            state.count += n;
            if state.count >= target {
                state.finalize(strength);
            }
            Ok(k.clone())
        }
        InnerQMode::Active => {
            let scale = Tensor::from_slice(&state.scale, (GROUP,), k.device())?;
            k.broadcast_mul(&scale)
        }
    }
}

/// The per-channel scale InnerQ wants applied to Q (as `Q * scale_inv`) so that
/// `<Q*scale_inv, K*scale> = <Q,K>` once [`innerq_scale_k`] is actively rescaling K. Returns
/// `None` when InnerQ isn't active yet (the common case, and always true unless
/// `MISTRALRS_TURBO4_INNERQ` is set).
pub(crate) fn innerq_query_scale_inv(device: &Device) -> Result<Option<Tensor>> {
    let state = innerq_state().lock().expect("InnerQ state mutex poisoned");
    match state.mode {
        InnerQMode::Active => Ok(Some(Tensor::from_slice(
            &state.scale_inv,
            (GROUP,),
            device,
        )?)),
        _ => Ok(None),
    }
}

/// Writes `x` (last dim `head_dim`) into `cache` (same last dim, cast to whatever dtype `cache`
/// already is) at the flat slot positions in `slot_mapping` -- a plain elementwise cast, no WHT
/// rotation or Lloyd-Max quantization. This is Turbo4's per-layer/per-side fallback (see
/// `cache_engine::turbo4_layer_plan`): auto-asymmetric upgrades a high-GQA-ratio model's K side
/// to this instead of the packed format, and layer-adaptive upgrades both sides for boundary
/// layers, both because the fixed 4-bit budget is known to degrade badly there (see module
/// docs) while an elementwise cast has none of that structural sensitivity.
///
/// The cache is allocated in the model's own compute dtype (BF16/F16/F32), not F8E4M3: candle's
/// CUDA backend has no generic float->F8E4M3 cast kernel in this build (`Tensor::zeros` in that
/// dtype works, but `to_dtype` into it doesn't -- confirmed empirically, not merely suspected),
/// only the specialized fused kernels the *native* F8E4M3 paged-attention path uses instead of a
/// generic tensor op. Using the compute dtype costs more memory than F8E4M3 would have on this
/// side, but it's guaranteed to work everywhere and is still far cheaper than not falling back
/// at all -- and the fallback's job is correctness, not maximum compression.
///
/// `cache` has shape `(num_slots, num_kv_heads, head_dim)`. `x` has shape `(num_tokens,
/// num_kv_heads, head_dim)`, and `slot_mapping` is a `num_tokens`-length i64 tensor of flat slot
/// indices, matching [`write_turbo4_cache`]'s convention.
pub(crate) fn write_plain_cache(x: &Tensor, cache: &Tensor, slot_mapping: &Tensor) -> Result<()> {
    let (num_tokens, num_kv_heads, head_dim) = x.dims3()?;
    let x_cast = x.to_dtype(cache.dtype())?;
    let index = slot_mapping
        .to_dtype(DType::U32)?
        .reshape((num_tokens, 1, 1))?
        .broadcast_as((num_tokens, num_kv_heads, head_dim))?
        .contiguous()?;
    cache.scatter_set(&index, &x_cast, 0)
}

/// The inverse of [`write_plain_cache`]: gathers elements for `block_table` (a sequence's
/// physical block ids, in order) back to `(seq_len, num_kv_heads, head_dim)`, still in the
/// cache's own dtype (unlike [`read_turbo4_cache`], the caller casts back to the compute dtype,
/// though in practice that's usually already a no-op since the cache *is* the compute dtype).
/// `cache` has the block-indexed shape `(num_gpu_blocks, block_size, num_kv_heads, head_dim)`.
pub(crate) fn read_plain_cache(cache: &Tensor, block_table: &[u32], seq_len: usize) -> Result<Tensor> {
    let (_num_gpu_blocks, block_size, num_kv_heads, head_dim) = cache.dims4()?;
    let device = cache.device();
    let block_ids = Tensor::from_slice(block_table, (block_table.len(),), device)?;
    let gathered = cache.index_select(&block_ids, 0)?; // (blocks_needed, block_size, kv_heads, head_dim)
    let flat = gathered.reshape((block_table.len() * block_size, num_kv_heads, head_dim))?;
    flat.narrow(0, 0, seq_len)
}

/// Quantizes `x` (last dim `GROUP`) and scatter-writes the packed blocks into `cache` (last dim
/// `BLOCK_TURBO4_BYTES`) at the flat slot positions in `slot_mapping`.
///
/// `cache` has shape `(num_slots, num_kv_heads, groups_per_head, BLOCK_TURBO4_BYTES)` (flat
/// slots = `num_gpu_blocks * block_size`, matching `reshape_and_cache`'s `slot_mapping`
/// convention). `x` has shape `(num_tokens, num_kv_heads, groups_per_head, GROUP)`, and
/// `slot_mapping` is a `num_tokens`-length i64 tensor of flat slot indices.
pub(crate) fn write_turbo4_cache(x: &Tensor, cache: &Tensor, slot_mapping: &Tensor) -> Result<()> {
    let (num_tokens, num_kv_heads, groups_per_head, group) = x.dims4()?;
    if group != GROUP {
        candle_core::bail!("write_turbo4_cache expects last dim {GROUP}, got {group}");
    }
    let (norm, qs) = quantize_turbo4_tensor(x)?;
    let packed = pack_turbo4_block(&norm, &qs)?; // (tokens, kv_heads, groups_per_head, BLOCK_TURBO4_BYTES)

    let index = slot_mapping
        .to_dtype(DType::U32)?
        .reshape((num_tokens, 1, 1, 1))?
        .broadcast_as((
            num_tokens,
            num_kv_heads,
            groups_per_head,
            BLOCK_TURBO4_BYTES,
        ))?
        .contiguous()?;
    cache.scatter_set(&index, &packed, 0)
}

/// The inverse of [`write_turbo4_cache`]: gathers packed blocks for `block_table` (a sequence's
/// physical block ids, in order) and dequantizes them back to `(seq_len, num_kv_heads,
/// groups_per_head, GROUP)`, ready to feed into eager attention. `cache` has the block-indexed
/// shape `(num_gpu_blocks, block_size, num_kv_heads, groups_per_head, BLOCK_TURBO4_BYTES)`.
pub(crate) fn read_turbo4_cache(
    cache: &Tensor,
    block_table: &[u32],
    seq_len: usize,
) -> Result<Tensor> {
    let (_num_gpu_blocks, block_size, num_kv_heads, groups_per_head, block_bytes) =
        cache.dims5()?;
    if block_bytes != BLOCK_TURBO4_BYTES {
        candle_core::bail!(
            "read_turbo4_cache expects last dim {BLOCK_TURBO4_BYTES}, got {block_bytes}"
        );
    }
    let device = cache.device();
    let block_ids = Tensor::from_slice(block_table, (block_table.len(),), device)?;
    let gathered = cache.index_select(&block_ids, 0)?; // (blocks_needed, block_size, kv_heads, groups, bytes)
    let flat = gathered.reshape((
        block_table.len() * block_size,
        num_kv_heads,
        groups_per_head,
        block_bytes,
    ))?;
    let packed = flat.narrow(0, 0, seq_len)?;
    let (norm, qs) = unpack_turbo4_block(&packed)?;
    dequantize_turbo4_tensor(&norm, &qs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-Gaussian test vector (xorshift32 + a crude Box-Muller-ish sum),
    /// good enough to sanity-check round-trip fidelity without pulling in `rand`.
    #[allow(clippy::cast_possible_truncation)] // test fixture only, precision doesn't matter
    fn test_vector(seed: u32) -> [f32; GROUP] {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (f64::from(state) / f64::from(u32::MAX)) as f32 - 0.5
        };
        let mut out = [0f32; GROUP];
        for v in out.iter_mut() {
            *v = next() + next() + next(); // sum of uniforms, roughly bell-shaped
        }
        out
    }

    fn cosine_similarity(a: &[f32; GROUP], b: &[f32; GROUP]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = l2_norm(a);
        let nb = l2_norm(b);
        dot / (na * nb)
    }

    #[test]
    fn rotation_round_trips_exactly() {
        let original = test_vector(42);
        let mut x = original;
        rotate_forward(&mut x);
        rotate_inverse(&mut x);
        for (a, b) in original.iter().zip(x.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "rotation did not invert cleanly: {a} vs {b}"
            );
        }
    }

    #[test]
    fn turbo4_round_trip_preserves_direction() {
        for seed in [1, 7, 1234, 999_999] {
            let original = test_vector(seed);
            let mut rotated = original;
            rotate_forward(&mut rotated);

            let block = quantize_turbo4(&rotated);
            let mut recovered = dequantize_turbo4(&block);
            rotate_inverse(&mut recovered);

            let sim = cosine_similarity(&original, &recovered);
            assert!(sim > 0.9, "seed {seed}: cosine similarity too low: {sim}");
        }
    }

    /// The whole point of storing `norm / recon_norm` instead of the raw norm: nearest-centroid
    /// quantization systematically shrinks the reconstructed vector, so an uncorrected norm would
    /// leave every dequantized vector measurably undersized. With the correction, magnitude
    /// should come back essentially exact (direction is the only thing the 4 bits can't capture).
    #[test]
    fn turbo4_norm_correction_preserves_reconstructed_magnitude() {
        for seed in [1, 7, 1234, 999_999] {
            let original = test_vector(seed);
            let mut rotated = original;
            rotate_forward(&mut rotated);

            let block = quantize_turbo4(&rotated);
            let recovered = dequantize_turbo4(&block);

            let original_norm = l2_norm(&rotated);
            let recovered_norm = l2_norm(&recovered);
            let relative_error = (recovered_norm - original_norm).abs() / original_norm;
            // f16::from_f32 on the stored corrected_norm is the only remaining error source
            // (~3-4 significant decimal digits), not the correction math itself.
            assert!(
                relative_error < 1e-3,
                "seed {seed}: norm-corrected reconstruction magnitude off by {relative_error}: \
                 original={original_norm} recovered={recovered_norm}"
            );
        }
    }

    #[test]
    fn rotation_matrix_matches_scalar_rotate_forward() -> Result<()> {
        let device = Device::Cpu;
        let original = test_vector(2024);
        let mut expected = original;
        rotate_forward(&mut expected);

        let x = Tensor::from_slice(&original, (1, GROUP), &device)?;
        let r = rotation_matrix_tensor(&device)?;
        let rotated = x.broadcast_matmul(&r.t()?)?;
        let got: Vec<f32> = rotated.reshape(GROUP)?.to_vec1()?;

        for (e, g) in expected.iter().zip(got.iter()) {
            assert!((e - g).abs() < 1e-4, "rotation matrix mismatch: {e} vs {g}");
        }
        Ok(())
    }

    #[test]
    fn tensor_round_trip_preserves_direction() -> Result<()> {
        let device = Device::Cpu;
        for seed in [1, 7, 1234, 999_999] {
            let original = test_vector(seed);
            let x = Tensor::from_slice(&original, (1, 1, GROUP), &device)?;

            let (norm, packed) = quantize_turbo4_tensor(&x)?;
            let recovered = dequantize_turbo4_tensor(&norm, &packed)?;
            let recovered: Vec<f32> = recovered.reshape(GROUP)?.to_vec1()?;
            let recovered: [f32; GROUP] = recovered.try_into().unwrap();

            let sim = cosine_similarity(&original, &recovered);
            assert!(
                sim > 0.9,
                "seed {seed}: tensor round trip cosine too low: {sim}"
            );
        }
        Ok(())
    }

    #[test]
    fn tensor_quantize_matches_scalar_reference_within_tolerance() -> Result<()> {
        let device = Device::Cpu;
        let original = test_vector(77);
        let mut rotated_scalar = original;
        rotate_forward(&mut rotated_scalar);
        let scalar_block = quantize_turbo4(&rotated_scalar);
        let scalar_norm = scalar_block.norm.to_f32();

        let x = Tensor::from_slice(&original, (1, GROUP), &device)?;
        let (norm, packed) = quantize_turbo4_tensor(&x)?;
        let tensor_norm: f32 = norm.reshape(())?.to_scalar()?;
        let tensor_packed: Vec<u8> = packed.reshape(GROUP / 2)?.to_vec1()?;

        assert!(
            (scalar_norm - tensor_norm).abs() / scalar_norm < 1e-3,
            "norm mismatch: scalar={scalar_norm} tensor={tensor_norm}"
        );
        // Both derive the same rotation via `rotate_forward`, so barring a value landing
        // exactly on a centroid boundary, the packed nibbles should match exactly.
        assert_eq!(scalar_block.qs.to_vec(), tensor_packed);
        Ok(())
    }

    #[test]
    fn pack_unpack_round_trips_norm_and_qs() -> Result<()> {
        let device = Device::Cpu;
        let original = test_vector(55);
        let mut rotated = original;
        rotate_forward(&mut rotated);
        let block = quantize_turbo4(&rotated);

        let norm = Tensor::new(block.norm.to_f32(), &device)?;
        let qs = Tensor::from_slice(&block.qs[..], (GROUP / 2,), &device)?;
        let packed = pack_turbo4_block(&norm, &qs)?;
        assert_eq!(packed.dims(), &[BLOCK_TURBO4_BYTES]);

        let (norm2, qs2) = unpack_turbo4_block(&packed)?;
        let norm2: f32 = norm2.to_scalar()?;
        let qs2: Vec<u8> = qs2.to_vec1()?;
        assert!(
            (norm2 - block.norm.to_f32()).abs() < 1.0, // within one 0.0625-ish quantization step
            "norm round trip: {} vs {}",
            block.norm.to_f32(),
            norm2
        );
        assert_eq!(qs2, block.qs.to_vec());
        Ok(())
    }

    /// Simulates a tiny paged cache: 2 physical blocks, block_size 2, 1 kv head, 1 group
    /// (head_dim == GROUP). Writes 3 tokens via [`write_turbo4_cache`] at hand-picked slots
    /// spanning both blocks, then reads them back for one sequence via [`read_turbo4_cache`]
    /// and checks the recovered vectors are close to the originals.
    #[test]
    fn write_then_read_turbo4_cache_round_trips() -> Result<()> {
        let device = Device::Cpu;
        const NUM_BLOCKS: usize = 2;
        const BLOCK_SIZE: usize = 2;
        const KV_HEADS: usize = 1;
        const GROUPS_PER_HEAD: usize = 1;

        let tokens: Vec<[f32; GROUP]> = [11, 22, 33].iter().map(|&s| test_vector(s)).collect();
        let flat: Vec<f32> = tokens.iter().flatten().copied().collect();
        let x = Tensor::from_slice(&flat, (3, KV_HEADS, GROUPS_PER_HEAD, GROUP), &device)?;

        // Flat cache used for writing: (num_slots, kv_heads, groups_per_head, BLOCK_TURBO4_BYTES).
        // Zeroed rather than uninitialized so the untouched slot 0 reads back as zero, not garbage.
        let flat_cache = Tensor::zeros(
            (
                NUM_BLOCKS * BLOCK_SIZE,
                KV_HEADS,
                GROUPS_PER_HEAD,
                BLOCK_TURBO4_BYTES,
            ),
            DType::U8,
            &device,
        )?;
        // Slots 1, 2, 3 get the 3 tokens; slot 0 (block 0 offset 0) is left untouched.
        let slot_mapping = Tensor::new(&[1i64, 2, 3], &device)?;
        write_turbo4_cache(&x, &flat_cache, &slot_mapping)?;

        // read_turbo4_cache expects the block-indexed shape; same storage, different view.
        let block_cache = flat_cache.reshape((
            NUM_BLOCKS,
            BLOCK_SIZE,
            KV_HEADS,
            GROUPS_PER_HEAD,
            BLOCK_TURBO4_BYTES,
        ))?;
        // block_table [0, 1] covers flat slots [0, 1, 2, 3]; read all 4 and skip the untouched slot 0.
        let recovered = read_turbo4_cache(&block_cache, &[0, 1], NUM_BLOCKS * BLOCK_SIZE)?;
        let recovered_vals: Vec<f32> = recovered
            .reshape(NUM_BLOCKS * BLOCK_SIZE * GROUP)?
            .to_vec1()?;

        for (i, expected) in tokens.iter().enumerate() {
            let slot = i + 1; // matches slot_mapping above
            let got: [f32; GROUP] = recovered_vals[slot * GROUP..(slot + 1) * GROUP]
                .try_into()
                .unwrap();
            let sim = cosine_similarity(expected, &got);
            assert!(sim > 0.9, "token {i} (slot {slot}): cosine too low: {sim}");
        }
        Ok(())
    }

    /// Validates the fused CUDA write kernel (`mistralrs_paged_attn::write_turbo4_cache`)
    /// against the eager Tensor-op reference (`write_turbo4_cache` in this module): writes the
    /// same tokens into two separate caches via the two paths, then reads both back through the
    /// shared `read_turbo4_cache` reference and compares the dequantized vectors. Compares
    /// reconstructed vectors rather than raw packed bytes since the two paths compute the
    /// rotation differently (matmul-against-a-precomputed-matrix vs explicit sign+butterfly), so
    /// individual centroid picks can legitimately differ by one bin right at a boundary --
    /// matches `tensor_quantize_matches_scalar_reference_within_tolerance`'s tolerance-based
    /// philosophy for the same reason. Also exercises one padding token (negative slot_mapping
    /// entry) to check the kernel's `if (flat_slot < 0) return;` guard.
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn write_turbo4_cache_kernel_matches_eager_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        const NUM_BLOCKS: usize = 4;
        const BLOCK_SIZE: usize = 2;
        const KV_HEADS: usize = 1;
        const GROUPS_PER_HEAD: usize = 1;

        let tokens: Vec<[f32; GROUP]> = [5, 17, 31].iter().map(|&s| test_vector(s)).collect();
        let flat: Vec<f32> = tokens.iter().flatten().copied().collect();
        let x = Tensor::from_slice(&flat, (3, KV_HEADS, GROUPS_PER_HEAD, GROUP), &device)?;
        // Slot mapping includes one padding entry (-1): the fused kernel must skip it, and the
        // eager reference must never actually be asked to write it (it isn't padding-aware).
        let slot_mapping_eager = Tensor::new(&[0i64, 1, 2], &device)?;
        let slot_mapping_kernel = Tensor::new(&[0i64, -1, 1, 2], &device)?;
        let x_kernel = Tensor::cat(
            &[&x.narrow(0, 0, 1)?, &Tensor::zeros((1, KV_HEADS, GROUPS_PER_HEAD, GROUP), DType::F32, &device)?, &x.narrow(0, 1, 2)?],
            0,
        )?;

        let cache_shape = (
            NUM_BLOCKS * BLOCK_SIZE,
            KV_HEADS,
            GROUPS_PER_HEAD,
            BLOCK_TURBO4_BYTES,
        );
        let flat_cache_eager = Tensor::zeros(cache_shape, DType::U8, &device)?;
        write_turbo4_cache(&x, &flat_cache_eager, &slot_mapping_eager)?;

        // write_turbo4_cache (both the eager reference and the fused kernel) takes the flat
        // slot-indexed cache shape; block-indexed reshapes are only needed for the read side.
        let flat_cache_kernel = Tensor::zeros(cache_shape, DType::U8, &device)?;
        mistralrs_paged_attn::write_turbo4_cache(&x_kernel, &flat_cache_kernel, &slot_mapping_kernel)?;
        let block_cache_kernel = flat_cache_kernel.reshape((
            NUM_BLOCKS,
            BLOCK_SIZE,
            KV_HEADS,
            GROUPS_PER_HEAD,
            BLOCK_TURBO4_BYTES,
        ))?;

        let block_cache_eager = flat_cache_eager.reshape((
            NUM_BLOCKS,
            BLOCK_SIZE,
            KV_HEADS,
            GROUPS_PER_HEAD,
            BLOCK_TURBO4_BYTES,
        ))?;
        let recovered_eager = read_turbo4_cache(&block_cache_eager, &[0, 1], 3)?;
        let recovered_kernel = read_turbo4_cache(&block_cache_kernel, &[0, 1], 3)?;

        let eager_vals: Vec<f32> = recovered_eager.reshape(3 * GROUP)?.to_vec1()?;
        let kernel_vals: Vec<f32> = recovered_kernel.reshape(3 * GROUP)?.to_vec1()?;
        for (i, expected) in tokens.iter().enumerate() {
            let eager: [f32; GROUP] = eager_vals[i * GROUP..(i + 1) * GROUP].try_into().unwrap();
            let kernel: [f32; GROUP] = kernel_vals[i * GROUP..(i + 1) * GROUP].try_into().unwrap();
            let sim_eager = cosine_similarity(expected, &eager);
            let sim_kernel = cosine_similarity(expected, &kernel);
            assert!(sim_eager > 0.9, "token {i}: eager reference cosine too low: {sim_eager}");
            assert!(sim_kernel > 0.9, "token {i}: fused write kernel cosine too low: {sim_kernel}");
            let sim_cross = cosine_similarity(&eager, &kernel);
            assert!(
                sim_cross > 0.999,
                "token {i}: fused write kernel disagrees with eager reference: cosine {sim_cross}"
            );
        }

        // Slot 3 (block 1, offset 1) is never targeted by any slot_mapping entry, padding or
        // otherwise; if the fused kernel's `if (flat_slot < 0) return;` guard were broken, an
        // out-of-bounds write from the padding entry (-1) would be the likeliest way to disturb
        // memory near it, so this is a cheap sanity check that nothing did.
        let untouched_slot: Vec<u8> = block_cache_kernel
            .narrow(0, 1, 1)?
            .narrow(1, 1, 1)?
            .reshape(BLOCK_TURBO4_BYTES)?
            .to_vec1()?;
        assert!(
            untouched_slot.iter().all(|&b| b == 0),
            "fused write kernel touched a slot no entry (padding or otherwise) targeted"
        );
        Ok(())
    }

    /// Validates the fused CUDA gather kernel (`mistralrs_paged_attn::gather_turbo4_cache`)
    /// against the eager Tensor-op reference (`read_turbo4_cache`) it's meant to replace in
    /// `forward_turbo4`: same packed cache, same block table, dequantized independently by two
    /// different code paths, and the results must agree closely. Also exercises a batch of two
    /// sequences with different lengths through one kernel launch, since that's the whole point
    /// of `cu_seq_lens`-based addressing (a real forward pass gathers the entire batch in one
    /// call instead of the old per-row Rust loop).
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn gather_turbo4_cache_kernel_matches_eager_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        const NUM_BLOCKS: usize = 4;
        const BLOCK_SIZE: usize = 2;
        const KV_HEADS: usize = 1;
        const GROUPS_PER_HEAD: usize = 1;

        // Two sequences sharing one physical cache: seq 0 has 3 tokens (slots 0..3, spanning
        // blocks 0-1), seq 1 has 1 token (slot 4, block 2). Block 3 is never written.
        let tokens: Vec<[f32; GROUP]> = [7, 13, 29, 41].iter().map(|&s| test_vector(s)).collect();
        let flat: Vec<f32> = tokens.iter().flatten().copied().collect();
        let x = Tensor::from_slice(&flat, (4, KV_HEADS, GROUPS_PER_HEAD, GROUP), &device)?;

        let flat_cache = Tensor::zeros(
            (
                NUM_BLOCKS * BLOCK_SIZE,
                KV_HEADS,
                GROUPS_PER_HEAD,
                BLOCK_TURBO4_BYTES,
            ),
            DType::U8,
            &device,
        )?;
        let slot_mapping = Tensor::new(&[0i64, 1, 2, 4], &device)?;
        write_turbo4_cache(&x, &flat_cache, &slot_mapping)?;

        let block_cache = flat_cache.reshape((
            NUM_BLOCKS,
            BLOCK_SIZE,
            KV_HEADS,
            GROUPS_PER_HEAD,
            BLOCK_TURBO4_BYTES,
        ))?;

        // Eager reference: one `read_turbo4_cache` call per sequence.
        let seq0_ref = read_turbo4_cache(&block_cache, &[0, 1], 3)?; // blocks 0,1 -> slots 0..3
        let seq1_ref = read_turbo4_cache(&block_cache, &[2], 1)?; // block 2 -> slot 4

        // Fused kernel: one call across both sequences.
        let block_table = Tensor::from_slice(&[0u32, 1, 2, 0], (2, 2), &device)?; // [batch=2, max_blocks=2]
        let cu_seq_lens = Tensor::from_slice(&[0i32, 3, 4], (3,), &device)?; // seq0: [0,3), seq1: [3,4)
        let gathered =
            mistralrs_paged_attn::gather_turbo4_cache(&block_cache, &block_table, &cu_seq_lens, 4, DType::F32)?;
        assert_eq!(gathered.dims(), &[4, KV_HEADS, GROUP]);

        let seq0_kernel = gathered.narrow(0, 0, 3)?.reshape((3, KV_HEADS, GROUPS_PER_HEAD, GROUP))?;
        let seq1_kernel = gathered.narrow(0, 3, 1)?.reshape((1, KV_HEADS, GROUPS_PER_HEAD, GROUP))?;

        for (name, reference, kernel, len) in
            [("seq0", &seq0_ref, &seq0_kernel, 3), ("seq1", &seq1_ref, &seq1_kernel, 1)]
        {
            let ref_vals: Vec<f32> = reference.reshape(len * GROUP)?.to_vec1()?;
            let kernel_vals: Vec<f32> = kernel.reshape(len * GROUP)?.to_vec1()?;
            for i in 0..len {
                let a: [f32; GROUP] = ref_vals[i * GROUP..(i + 1) * GROUP].try_into().unwrap();
                let b: [f32; GROUP] = kernel_vals[i * GROUP..(i + 1) * GROUP].try_into().unwrap();
                let sim = cosine_similarity(&a, &b);
                assert!(sim > 0.999, "{name} token {i}: kernel vs eager reference cosine too low: {sim}");
                for (av, bv) in a.iter().zip(b.iter()) {
                    assert!(
                        (av - bv).abs() < 1e-3,
                        "{name} token {i}: kernel vs eager reference element mismatch: {av} vs {bv}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Same idea as [`gather_turbo4_cache_kernel_matches_eager_reference`] but for the plain
    /// (unquantized) fallback side: `mistralrs_paged_attn::gather_plain_kv_cache` against
    /// `read_plain_cache`.
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn gather_plain_kv_cache_kernel_matches_eager_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        const NUM_BLOCKS: usize = 4;
        const BLOCK_SIZE: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = GROUP;

        let tokens: Vec<[f32; HEAD_DIM]> = [7, 13, 29, 41].iter().map(|&s| test_vector(s)).collect();
        let flat: Vec<f32> = tokens.iter().flatten().copied().collect();
        let x = Tensor::from_slice(&flat, (4, KV_HEADS, HEAD_DIM), &device)?;

        let flat_cache = Tensor::zeros((NUM_BLOCKS * BLOCK_SIZE, KV_HEADS, HEAD_DIM), DType::BF16, &device)?;
        let slot_mapping = Tensor::new(&[0i64, 1, 2, 4], &device)?;
        write_plain_cache(&x, &flat_cache, &slot_mapping)?;

        let block_cache = flat_cache.reshape((NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM))?;

        let seq0_ref = read_plain_cache(&block_cache, &[0, 1], 3)?.to_dtype(DType::F32)?;
        let seq1_ref = read_plain_cache(&block_cache, &[2], 1)?.to_dtype(DType::F32)?;

        let block_table = Tensor::from_slice(&[0u32, 1, 2, 0], (2, 2), &device)?;
        let cu_seq_lens = Tensor::from_slice(&[0i32, 3, 4], (3,), &device)?;
        let gathered =
            mistralrs_paged_attn::gather_plain_kv_cache(&block_cache, &block_table, &cu_seq_lens, 4)?
                .to_dtype(DType::F32)?;
        assert_eq!(gathered.dims(), &[4, KV_HEADS, HEAD_DIM]);

        let seq0_kernel = gathered.narrow(0, 0, 3)?;
        let seq1_kernel = gathered.narrow(0, 3, 1)?;

        for (name, reference, kernel, len) in
            [("seq0", &seq0_ref, &seq0_kernel, 3), ("seq1", &seq1_ref, &seq1_kernel, 1)]
        {
            let ref_vals: Vec<f32> = reference.reshape(len * HEAD_DIM)?.to_vec1()?;
            let kernel_vals: Vec<f32> = kernel.reshape(len * HEAD_DIM)?.to_vec1()?;
            for (av, bv) in ref_vals.iter().zip(kernel_vals.iter()) {
                assert!(
                    (av - bv).abs() < 1e-3,
                    "{name}: plain kernel vs eager reference element mismatch: {av} vs {bv}"
                );
            }
        }
        Ok(())
    }

    /// The plain elementwise fallback (`turbo4_layer_plan`'s auto-asymmetric/layer-adaptive
    /// escape hatch): no WHT rotation or centroid quantization, just a cast into the cache's own
    /// dtype, so recovered values should match far more tightly than the 4-bit round trip above.
    /// Uses BF16 rather than F8E4M3 for the cache dtype -- matches what
    /// `cache_engine::allocate_gpu_cache` actually allocates (the model's compute dtype, not
    /// F8E4M3; see `write_plain_cache`'s docs for why) and, unlike F8E4M3, is uniformly
    /// supported by candle's CUDA backend for every op this needs.
    #[test]
    fn write_then_read_plain_cache_round_trips_cpu() -> Result<()> {
        write_then_read_plain_cache_round_trips_on(Device::Cpu)
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn write_then_read_plain_cache_round_trips_cuda() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        write_then_read_plain_cache_round_trips_on(device)
    }

    fn write_then_read_plain_cache_round_trips_on(device: Device) -> Result<()> {
        const NUM_BLOCKS: usize = 2;
        const BLOCK_SIZE: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = GROUP;

        let tokens: Vec<[f32; HEAD_DIM]> = [11, 22, 33].iter().map(|&s| test_vector(s)).collect();
        let flat: Vec<f32> = tokens.iter().flatten().copied().collect();
        let x = Tensor::from_slice(&flat, (3, KV_HEADS, HEAD_DIM), &device)?;

        let flat_cache = Tensor::zeros(
            (NUM_BLOCKS * BLOCK_SIZE, KV_HEADS, HEAD_DIM),
            DType::BF16,
            &device,
        )?;
        let slot_mapping = Tensor::new(&[1i64, 2, 3], &device)?;
        write_plain_cache(&x, &flat_cache, &slot_mapping)?;

        let block_cache = flat_cache.reshape((NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM))?;
        let recovered = read_plain_cache(&block_cache, &[0, 1], NUM_BLOCKS * BLOCK_SIZE)?
            .to_dtype(DType::F32)?;
        let recovered_vals: Vec<f32> = recovered
            .reshape(NUM_BLOCKS * BLOCK_SIZE * HEAD_DIM)?
            .to_vec1()?;

        for (i, expected) in tokens.iter().enumerate() {
            let slot = i + 1;
            let got: [f32; HEAD_DIM] = recovered_vals[slot * HEAD_DIM..(slot + 1) * HEAD_DIM]
                .try_into()
                .unwrap();
            let sim = cosine_similarity(expected, &got);
            assert!(
                sim > 0.999,
                "token {i} (slot {slot}): plain cache round trip should be near-exact, cosine: {sim}"
            );
        }
        Ok(())
    }

    /// Exercises `InnerQState::finalize` directly rather than through the env-var-gated global
    /// singleton (`innerq_state()`), since that's process-wide state and Rust tests run
    /// concurrently in the same process -- there's no way to isolate one test's env vars from
    /// another's view of it. Builds a state with one deliberately dominant channel and checks
    /// finalize both detects it (scale < 1 for the loud channel, > 1 for the quiet ones) and
    /// keeps scale/scale_inv exact reciprocals (the property that makes <Q,K> preservation exact).
    #[test]
    fn innerq_finalize_downweights_the_dominant_channel() {
        let mut state = InnerQState {
            mode: InnerQMode::Calibrating {
                target: 100,
                strength: 0.5,
            },
            sq_accum: [1.0; GROUP],
            count: 100,
            scale: [1.0; GROUP],
            scale_inv: [1.0; GROUP],
        };
        // Channel 0 carries 100x the energy of every other channel.
        state.sq_accum[0] = 100.0;
        state.finalize(0.5);

        assert!(
            matches!(state.mode, InnerQMode::Active),
            "expected InnerQ to activate given a 10x RMS imbalance"
        );
        assert!(
            state.scale[0] < 1.0,
            "dominant channel should be scaled down, got {}",
            state.scale[0]
        );
        assert!(
            state.scale[1] > 1.0,
            "quiet channel should be scaled up, got {}",
            state.scale[1]
        );
        for i in 0..GROUP {
            let product = state.scale[i] * state.scale_inv[i];
            assert!(
                (product - 1.0).abs() < 1e-6,
                "channel {i}: scale*scale_inv should be exactly 1, got {product}"
            );
            assert!(
                (0.5..=2.0).contains(&state.scale[i]),
                "channel {i}: scale {} outside the [0.5, 2.0] clamp",
                state.scale[i]
            );
        }
    }

    /// The auto-skip path: if every channel already has the same RMS, InnerQ shouldn't bother
    /// (matches upstream's "channels already balanced" check) -- there's nothing to equalize,
    /// and forcing a no-op scale through the pipeline would just be wasted work.
    #[test]
    fn innerq_finalize_auto_disables_when_channels_are_balanced() {
        let mut state = InnerQState {
            mode: InnerQMode::Calibrating {
                target: 100,
                strength: 0.5,
            },
            sq_accum: [1.0; GROUP],
            count: 100,
            scale: [1.0; GROUP],
            scale_inv: [1.0; GROUP],
        };
        state.finalize(0.5);
        assert!(
            matches!(state.mode, InnerQMode::Disabled),
            "expected InnerQ to auto-disable when channels are already balanced"
        );
    }

    /// The actual point of InnerQ: with K rescaled per-channel and Q rescaled by the exact
    /// inverse, the dot product `<Q,K>` must come out unchanged (that's what makes it safe to
    /// apply before attention without altering anything but quantization fidelity).
    #[test]
    fn innerq_scale_and_inverse_preserve_dot_product() {
        let mut state = InnerQState {
            mode: InnerQMode::Calibrating {
                target: 100,
                strength: 0.5,
            },
            sq_accum: [1.0; GROUP],
            count: 100,
            scale: [1.0; GROUP],
            scale_inv: [1.0; GROUP],
        };
        state.sq_accum[3] = 50.0;
        state.sq_accum[9] = 0.02;
        state.finalize(0.5);
        assert!(matches!(state.mode, InnerQMode::Active));

        let q = test_vector(11);
        let k = test_vector(22);
        let original_dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

        let scaled_dot: f32 = (0..GROUP)
            .map(|i| (q[i] * state.scale_inv[i]) * (k[i] * state.scale[i]))
            .sum();

        let relative_error = (scaled_dot - original_dot).abs() / original_dot.abs();
        assert!(
            relative_error < 1e-4,
            "InnerQ should preserve <Q,K> exactly: original={original_dot} scaled={scaled_dot}"
        );
    }
}
