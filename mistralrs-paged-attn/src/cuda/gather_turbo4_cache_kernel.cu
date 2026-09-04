#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"
#include "turbo4_common.cuh"

#include <algorithm>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                       \
      exit(err);                                                              \
    }                                                                         \
  } while (0)

namespace vllm {
namespace turbo4 {

/// One CUDA block per (token, kv_head, group); GROUP=128 threads, one per
/// element. Dequantizes a packed Turbo4 block (2-byte fixed-point norm + 64
/// nibble-packed 4-bit centroid indices) and applies the exact inverse
/// rotation (rotate_inverse = signs2 -> FWHT -> signs1, see turbo_quant.rs),
/// writing the result straight into a flat (num_tokens, kv_heads, head_size)
/// output tensor -- the same layout `gather_kv_cache` produces for
/// f16/bf16/f32/f8e4m3 caches, so downstream code (unpack_gathered_kv, Sdpa)
/// doesn't need to know this cache was Turbo4-packed at all.
///
/// Block table / cu_seq_lens addressing mirrors gather_kv_cache_kernel.cu:
/// GPU-resident, no host readback, so this is safe to call from inside a
/// captured CUDA graph (unlike the per-row `.to_vec1()` block-table
/// materialization the eager Rust path used before this kernel existed).
template <typename out_t>
__global__ void
gather_turbo4_cache_kernel(const uint8_t *__restrict__ cache,
                           out_t *__restrict__ out,
                           const int32_t *__restrict__ block_table,
                           const int32_t *__restrict__ cu_seq_lens,
                           const int32_t num_seqs, const int32_t block_size,
                           const int32_t block_table_stride,
                           const int32_t kv_heads,
                           const int32_t groups_per_head,
                           const int32_t head_size) {
  const int32_t token_id = blockIdx.x;
  const int32_t head_idx = blockIdx.y;
  const int32_t group_idx = blockIdx.z;
  const int32_t tid = threadIdx.x;

  __shared__ int32_t s_flat_slot;
  __shared__ float s_norm;
  __shared__ float sh[GROUP];

  if (tid == 0) {
    // Binary search cu_seq_lens (cumulative token counts) for batch_id, the
    // largest i such that cu_seq_lens[i] <= token_id.
    int32_t lo = 0, hi = num_seqs;
    while (lo < hi) {
      int32_t mid = (lo + hi + 1) / 2;
      if (cu_seq_lens[mid] <= token_id) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    const int32_t batch_id = lo;
    const int32_t batch_offset = token_id - cu_seq_lens[batch_id];
    const int32_t block_table_id = batch_offset / block_size;
    const int32_t slot = batch_offset % block_size;
    const int32_t block_id =
        block_table[batch_id * block_table_stride + block_table_id];
    s_flat_slot = block_id * block_size + slot;
  }
  __syncthreads();

  const int64_t block_stride =
      static_cast<int64_t>(kv_heads) * groups_per_head * BLOCK_BYTES;
  const uint8_t *block_ptr = cache +
                             static_cast<int64_t>(s_flat_slot) * block_stride +
                             static_cast<int64_t>(head_idx) * groups_per_head *
                                 BLOCK_BYTES +
                             static_cast<int64_t>(group_idx) * BLOCK_BYTES;

  if (tid == 0) {
    const uint32_t low = block_ptr[0];
    const uint32_t high = block_ptr[1];
    const float level = static_cast<float>(low + high * 256u);
    s_norm = level * (NORM_SCALE_MAX / NORM_LEVELS);
  }
  __syncthreads();

  const uint8_t byte = block_ptr[2 + tid / 2];
  const uint32_t nibble = (tid % 2 == 0) ? (byte & 0xF) : (byte >> 4);
  sh[tid] = kCentroids4Bit[nibble] * s_norm;
  __syncthreads();

  // rotate_inverse: x *= signs2; fwht_128(x) (1/sqrt(N) scale applied below); x *= signs1.
  sh[tid] *= kWhtSigns2[tid];
  __syncthreads();

  fwht_128(sh, tid);

  const float rotated = sh[tid] * INV_SQRT_GROUP * kWhtSigns1[tid];

  const int64_t out_base = static_cast<int64_t>(token_id) * kv_heads * head_size +
                           static_cast<int64_t>(head_idx) * head_size +
                           static_cast<int64_t>(group_idx) * GROUP + tid;
  store_f32(out + out_base, rotated);
}

} // namespace turbo4
} // namespace vllm

#define CALL_GATHER_TURBO4(OUT_T)                                            \
  vllm::turbo4::gather_turbo4_cache_kernel<OUT_T><<<grid, block, 0, stream>>>( \
      reinterpret_cast<const uint8_t *>(cache), reinterpret_cast<OUT_T *>(out), \
      block_table, cu_seq_lens, num_seqs, block_size, block_table_stride,     \
      kv_heads, groups_per_head, head_size);

extern "C" void gather_turbo4_cache(
    const void *cache, // [num_blocks, block_size, kv_heads, groups_per_head, BLOCK_BYTES]
    void *out,         // [num_tokens, kv_heads, groups_per_head * GROUP]
    const int32_t *block_table, // [batch, max_blocks]
    const int32_t *cu_seq_lens, // [batch + 1]
    int32_t num_tokens, int32_t num_seqs, int32_t block_size,
    int32_t block_table_stride, int32_t kv_heads, int32_t groups_per_head,
    cudaStream_t stream,
    uint32_t out_dtype // 0 => f16; 1 => bf16; 2 => f32
) {
  if (num_tokens <= 0) {
    return;
  }
  const int32_t head_size = groups_per_head * vllm::turbo4::GROUP;
  dim3 grid(num_tokens, kv_heads, groups_per_head);
  dim3 block(vllm::turbo4::GROUP);

  if (out_dtype == 0) {
    CALL_GATHER_TURBO4(uint16_t)
  } else if (out_dtype == 1) {
    CALL_GATHER_TURBO4(__nv_bfloat16)
  } else if (out_dtype == 2) {
    CALL_GATHER_TURBO4(float)
  }
  CUDA_CHECK(cudaGetLastError());
}
