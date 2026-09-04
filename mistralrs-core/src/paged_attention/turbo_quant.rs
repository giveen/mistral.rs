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
//! This is CPU reference / data-format scaffolding only. `PagedCacheType::validate` in
//! `cache_engine.rs` refuses to let a model actually run with `PagedCacheType::Turbo4`: the
//! native `reshape_and_cache`/`gather_kv_cache`/paged-attention kernels in `mistralrs-paged-attn`
//! only understand f32/f16/bf16/f8e4m3 element layouts, not these packed sub-byte blocks.

// Scaffolding: nothing outside `cache_engine::CacheEngine::calculate_turbo4_block_shape` and
// `#[cfg(test)]` calls into this yet, pending the paged-attention kernel work described above.
#![allow(dead_code)]

use std::sync::OnceLock;

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
pub(crate) fn quantize_turbo4(rotated: &[f32; GROUP]) -> BlockTurbo4 {
    let norm = l2_norm(rotated).max(f32::MIN_POSITIVE);
    let inv_norm = 1.0 / norm;
    let mut qs = [0u8; GROUP / 2];
    for j in 0..GROUP {
        let idx = nearest_centroid(rotated[j] * inv_norm, &MID_4BIT);
        qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
    }
    BlockTurbo4 {
        norm: f16::from_f32(norm),
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

    Ok((norm.squeeze(D::Minus1)?, packed))
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
}
