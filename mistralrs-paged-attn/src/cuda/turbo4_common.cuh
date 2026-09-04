#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Shared constants/helpers for the Turbo4 KV cache CUDA kernels
// (gather_turbo4_cache_kernel.cu, write_turbo4_cache_kernel.cu). Mirrors
// mistralrs-core/src/paged_attention/turbo_quant.rs exactly: signed
// Walsh-Hadamard rotation (seed=42) + fixed 4-bit Lloyd-Max centroids tuned
// for N(0, 1/128), plus the arithmetic 2-byte fixed-point norm encoding used
// by quantize_turbo4_tensor/pack_turbo4_block (NOT the f16 encoding the
// scalar BlockTurbo4 reference type uses -- that one is CPU-test-only).
// Kept in one header, included by both kernels, so the two can't drift.

namespace vllm {
namespace turbo4 {

constexpr int GROUP = 128;
constexpr int BLOCK_BYTES = 2 + GROUP / 2; // 66
constexpr float INV_SQRT_GROUP = 0.088388350f; // 1 / sqrt(128)
constexpr float NORM_SCALE_MAX = 4096.0f;
constexpr float NORM_LEVELS = 65535.0f;

__constant__ float kWhtSigns1[GROUP] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
};
__constant__ float kWhtSigns2[GROUP] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
};
__constant__ float kCentroids4Bit[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f, -0.083292f, -0.058050f, -0.034299f, -0.011349f,
    0.011349f, 0.034299f, 0.058050f, 0.083292f, 0.111036f, 0.143016f, 0.182877f, 0.241529f,
};
// Ascending midpoints between consecutive centroids; nearest_centroid(v) = count(mid <= v).
__constant__ float kMid4Bit[15] = {
    -0.212203f, -0.162947f, -0.127026f, -0.097164f, -0.070671f, -0.046174f, -0.022824f, 0.0f,
    0.022824f, 0.046174f, 0.070671f, 0.097164f, 0.127026f, 0.162947f, 0.212203f,
};

/// In-place radix-2 Hadamard butterfly, normalized so the transform is its own inverse (the
/// 1/sqrt(GROUP) scale is applied by the caller, not baked in here, since the write kernel
/// wants it fused with a later per-element multiply and the read kernel doesn't).
/// One thread per element; each stage's (tid, tid+h) pair is owned exclusively by thread tid
/// (the "low" side, `(tid & h) == 0`), so no intra-stage sync is needed between the read and
/// the write, only an end-of-stage syncthreads before the next stage's pairs read them.
__device__ __forceinline__ void fwht_128(float *sh, int tid) {
#pragma unroll
  for (int h = 1; h < GROUP; h <<= 1) {
    if ((tid & h) == 0) {
      const float a = sh[tid];
      const float b = sh[tid + h];
      sh[tid] = a + b;
      sh[tid + h] = a - b;
    }
    __syncthreads();
  }
}

template <typename io_t> __device__ __forceinline__ float load_f32(const io_t *ptr);
template <> __device__ __forceinline__ float load_f32<uint16_t>(const uint16_t *ptr) {
  return __half2float(__ushort_as_half(*ptr));
}
template <>
__device__ __forceinline__ float load_f32<__nv_bfloat16>(const __nv_bfloat16 *ptr) {
  return __bfloat162float(*ptr);
}
template <> __device__ __forceinline__ float load_f32<float>(const float *ptr) {
  return *ptr;
}

template <typename io_t> __device__ __forceinline__ void store_f32(io_t *ptr, float v);
template <> __device__ __forceinline__ void store_f32<uint16_t>(uint16_t *ptr, float v) {
  *ptr = __half_as_ushort(__float2half_rn(v));
}
template <>
__device__ __forceinline__ void store_f32<__nv_bfloat16>(__nv_bfloat16 *ptr, float v) {
  *ptr = __float2bfloat16_rn(v);
}
template <> __device__ __forceinline__ void store_f32<float>(float *ptr, float v) {
  *ptr = v;
}

/// idx = number of ascending midpoints <= v (matches turbo_quant.rs's nearest_centroid);
/// branch-free so all 128 threads take the same path regardless of their value.
__device__ __forceinline__ uint32_t nearest_centroid(float v) {
  uint32_t idx = 0;
#pragma unroll
  for (int i = 0; i < 15; i++) {
    idx += (v >= kMid4Bit[i]) ? 1u : 0u;
  }
  return idx;
}

} // namespace turbo4
} // namespace vllm
