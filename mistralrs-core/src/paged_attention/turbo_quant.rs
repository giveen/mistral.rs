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
}
