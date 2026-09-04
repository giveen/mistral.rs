#include <cfloat>
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

/// Sums `val` across all GROUP=128 threads in the block via `scratch` and returns the total to
/// every thread. Callable more than once per kernel (e.g. once for the pre-quant norm, once for
/// the post-quant recon norm) since it leaves `scratch` fully consumed (synced) on return.
__device__ __forceinline__ float block_reduce_sum(float val, float *scratch, int tid) {
  scratch[tid] = val;
  __syncthreads();
#pragma unroll
  for (int stride = GROUP / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}

/// One CUDA block per (token, kv_head, group); GROUP=128 threads, one per
/// element. Forward rotation (signs1 -> FWHT -> signs2, see
/// turbo_quant.rs's rotate_forward) + nearest-4-bit-centroid quantization +
/// norm correction, matching `quantize_turbo4_tensor` exactly, then packs
/// and scatter-writes the block straight into the cache at `slot_mapping`'s
/// flat slot for this token -- no separate host-side quantize-then-scatter
/// step. `slot_mapping` (and every other pointer here) is GPU-resident, no
/// host readback, matching `reshape_and_cache_kernel.cu`'s convention;
/// negative entries are padding tokens and are skipped, same as that
/// kernel's `if (slot_idx < 0) return;`.
///
/// Turbo4's packed cache layout is `(num_blocks, block_size, kv_heads,
/// groups_per_head, BLOCK_BYTES)` -- num_blocks and block_size are the two
/// outermost, contiguous dims, so `slot_mapping`'s flat slot index (which
/// already spans num_blocks * block_size) addresses this cache directly with
/// no block/offset decomposition, unlike the vLLM x-split layout
/// `reshape_and_cache_kernel.cu` targets.
template <typename in_t>
__global__ void
write_turbo4_cache_kernel(const in_t *__restrict__ x, uint8_t *__restrict__ cache,
                          const int64_t *__restrict__ slot_mapping,
                          const int32_t kv_heads, const int32_t groups_per_head,
                          const int32_t head_size) {
  const int32_t token_id = blockIdx.x;
  const int32_t head_idx = blockIdx.y;
  const int32_t group_idx = blockIdx.z;
  const int32_t tid = threadIdx.x;

  const int64_t flat_slot = slot_mapping[token_id];
  if (flat_slot < 0) {
    return;
  }

  __shared__ float sh[GROUP];
  __shared__ float scratch[GROUP];
  __shared__ uint8_t s_idx[GROUP];

  const int64_t x_base = static_cast<int64_t>(token_id) * kv_heads * head_size +
                         static_cast<int64_t>(head_idx) * head_size +
                         static_cast<int64_t>(group_idx) * GROUP + tid;
  sh[tid] = load_f32(x + x_base);
  sh[tid] *= kWhtSigns1[tid];
  __syncthreads();

  fwht_128(sh, tid);
  sh[tid] = sh[tid] * INV_SQRT_GROUP * kWhtSigns2[tid]; // sh[tid] is now rotated[tid]

  const float norm_sq = block_reduce_sum(sh[tid] * sh[tid], scratch, tid);
  const float norm = fmaxf(sqrtf(norm_sq), FLT_MIN);
  const float inv_norm = 1.0f / norm;

  const uint32_t idx = nearest_centroid(sh[tid] * inv_norm);
  s_idx[tid] = static_cast<uint8_t>(idx);
  const float c = kCentroids4Bit[idx];

  const float recon_sq = block_reduce_sum(c * c, scratch, tid);
  const float recon_norm = fmaxf(sqrtf(recon_sq), FLT_MIN);
  const float corrected_norm = norm / recon_norm;

  const int64_t block_stride =
      static_cast<int64_t>(kv_heads) * groups_per_head * BLOCK_BYTES;
  uint8_t *block_ptr = cache + flat_slot * block_stride +
                       static_cast<int64_t>(head_idx) * groups_per_head * BLOCK_BYTES +
                       static_cast<int64_t>(group_idx) * BLOCK_BYTES;

  if (tid == 0) {
    const float level = roundf(
        fminf(fmaxf(corrected_norm * (NORM_LEVELS / NORM_SCALE_MAX), 0.0f), NORM_LEVELS));
    const float high_f = floorf(level / 256.0f);
    const float low_f = level - high_f * 256.0f;
    block_ptr[0] = static_cast<uint8_t>(low_f);
    block_ptr[1] = static_cast<uint8_t>(high_f);
  }
  __syncthreads();

  if (tid < GROUP / 2) {
    block_ptr[2 + tid] = (s_idx[2 * tid] & 0xF) | (s_idx[2 * tid + 1] << 4);
  }
}

} // namespace turbo4
} // namespace vllm

#define CALL_WRITE_TURBO4(IN_T)                                              \
  vllm::turbo4::write_turbo4_cache_kernel<IN_T><<<grid, block, 0, stream>>>( \
      reinterpret_cast<const IN_T *>(x), reinterpret_cast<uint8_t *>(cache), \
      slot_mapping, kv_heads, groups_per_head, head_size);

extern "C" void write_turbo4_cache(
    const void *x,     // [num_tokens, kv_heads, groups_per_head, GROUP]
    void *cache,       // [num_blocks, block_size, kv_heads, groups_per_head, BLOCK_BYTES]
    const int64_t *slot_mapping, // [num_tokens]
    int32_t num_tokens, int32_t kv_heads, int32_t groups_per_head,
    cudaStream_t stream,
    uint32_t in_dtype // 0 => f16; 1 => bf16; 2 => f32
) {
  if (num_tokens <= 0) {
    return;
  }
  const int32_t head_size = groups_per_head * vllm::turbo4::GROUP;
  dim3 grid(num_tokens, kv_heads, groups_per_head);
  dim3 block(vllm::turbo4::GROUP);

  if (in_dtype == 0) {
    CALL_WRITE_TURBO4(uint16_t)
  } else if (in_dtype == 1) {
    CALL_WRITE_TURBO4(__nv_bfloat16)
  } else if (in_dtype == 2) {
    CALL_WRITE_TURBO4(float)
  }
  CUDA_CHECK(cudaGetLastError());
}
