#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"
#include "turbo4_common.cuh"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                      \
      exit(err);                                                             \
    }                                                                        \
  } while (0)

// Fused decode-time paged attention for the Turbo4 KV cache.
//
// Ported from TheTom/llama-cpp-turboquant's flash_attn_ext_vec kernel and
// llama-graph.cpp's surrounding rotate/un-rotate ops (see
// ggml_turbo_wht(..., 0, ...) before flash attention, ggml_turbo_wht(..., 1,
// ...) after it). The WHT rotation is orthogonal, so rotating BOTH sides of
// a dot product by the same matrix preserves it, and rotation commutes with
// the softmax-weighted sum over V. That means: rotate Q forward once (one
// vector), run the whole KQ/softmax/V loop directly against the
// already-rotated K/V cache with a plain per-element centroid dequant (no
// per-token inverse WHT at all), then inverse-rotate the accumulated output
// once. This replaces gather_turbo4_cache_kernel.cu's approach of
// inverse-rotating every cached K/V element on every decode step, which is
// the expensive part this design avoids.
//
// InnerQ (see turbo_quant.rs) is applied entirely on the Rust side before
// this kernel ever runs (K is pre-scaled before write_turbo4_cache, Q is
// pre-scaled by the inverse before this call), so this kernel has no InnerQ
// awareness at all.
//
// Grid: (num_heads, num_seqs). Block: PA_NUM_THREADS (== GROUP) threads.
// One block computes one (seq, head) query row's full attention output,
// looping over context_lens[seq_idx] (a device pointer -- never read on the
// host), so this has the same zero-host-visible-shape-variation property as
// vLLM's paged_attention_v1_kernel and is safe under CUDA graph capture.
//
// Unlike paged_attention_v1/v2, this is decode-only (one query token per
// sequence) and V1-style only (no partitioned/v2 long-context split yet --
// the caller is expected to bound max_num_blocks_per_seq * block_size and
// fall back to the gather+Sdpa path for very long context).
namespace vllm {
namespace turbo4 {

constexpr int PA_NUM_THREADS = GROUP; // 128
constexpr int PA_WARP_SIZE = 32;
constexpr int PA_NUM_WARPS = PA_NUM_THREADS / PA_WARP_SIZE;
constexpr int PA_MAX_GROUPS_PER_HEAD = 4; // covers head_size up to 512

__device__ __forceinline__ float pa_warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  }
  return v;
}

__device__ __forceinline__ float pa_warp_reduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
  }
  return v;
}

template <typename scalar_t>
__global__ void paged_attention_turbo4_kernel(
    scalar_t *__restrict__ out,          // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ query,  // [num_seqs, num_heads, head_size]
    const uint8_t *__restrict__ key_cache_turbo,   // flat-slot turbo4 layout, or null
    const scalar_t *__restrict__ key_cache_plain,  // flat-slot plain layout, or null
    const uint8_t *__restrict__ value_cache_turbo, // flat-slot turbo4 layout, or null
    const scalar_t *__restrict__ value_cache_plain,// flat-slot plain layout, or null
    const bool key_is_turbo, const bool value_is_turbo,
    const int32_t *__restrict__ block_table,  // [num_seqs, max_num_blocks_per_seq]
    const int32_t *__restrict__ context_lens, // [num_seqs]
    const int32_t num_kv_heads, const int32_t head_size,
    const int32_t groups_per_head, const int32_t block_size,
    const int32_t max_num_blocks_per_seq, const float scale,
    const float softcapping) {
  const int32_t head_idx = blockIdx.x;
  const int32_t seq_idx = blockIdx.y;
  const int32_t num_heads = gridDim.x;
  const int32_t gqa_ratio = num_heads / num_kv_heads;
  const int32_t kv_head_idx = head_idx / gqa_ratio;
  const int32_t tid = threadIdx.x;
  const int32_t warp_id = tid / PA_WARP_SIZE;
  const int32_t lane = tid % PA_WARP_SIZE;

  const int32_t context_len = context_lens[seq_idx];

  extern __shared__ float smem[];
  float *q_shared = smem;             // head_size floats
  float *logits = smem + head_size;   // context_len floats (capacity: launcher's bound)
  __shared__ float red[PA_NUM_WARPS];
  __shared__ float rot_scratch[GROUP];

  // ---- Load Q, forward-rotating each group iff K went through Turbo4 ----
  const scalar_t *q_ptr =
      query + (static_cast<int64_t>(seq_idx) * num_heads + head_idx) * head_size;
  for (int32_t g = 0; g < groups_per_head; g++) {
    float qv = load_f32(q_ptr + g * GROUP + tid);
    if (key_is_turbo) {
      rot_scratch[tid] = qv * kWhtSigns1[tid];
      __syncthreads();
      fwht_128(rot_scratch, tid);
      qv = rot_scratch[tid] * INV_SQRT_GROUP * kWhtSigns2[tid];
      __syncthreads();
    }
    q_shared[g * GROUP + tid] = qv;
  }
  __syncthreads();

  // ---- KQ scores: one warp per KV token per iteration ----
  const int32_t *bt_row =
      block_table + static_cast<int64_t>(seq_idx) * max_num_blocks_per_seq;
  const int64_t turbo_block_stride =
      static_cast<int64_t>(num_kv_heads) * groups_per_head * BLOCK_BYTES;

  float qk_max = -FLT_MAX;
  for (int32_t t = warp_id; t < context_len; t += PA_NUM_WARPS) {
    const int32_t block_id = bt_row[t / block_size];
    const int64_t flat_slot = static_cast<int64_t>(block_id) * block_size + (t % block_size);

    float partial = 0.f;
    if (key_is_turbo) {
      const uint8_t *block_ptr = key_cache_turbo + flat_slot * turbo_block_stride +
                                 static_cast<int64_t>(kv_head_idx) * groups_per_head * BLOCK_BYTES;
      for (int32_t d = lane; d < head_size; d += PA_WARP_SIZE) {
        const int32_t g = d / GROUP;
        const int32_t e = d % GROUP;
        const uint8_t *gp = block_ptr + g * BLOCK_BYTES;
        const uint32_t low = gp[0];
        const uint32_t high = gp[1];
        const float norm = static_cast<float>(low + high * 256u) * (NORM_SCALE_MAX / NORM_LEVELS);
        const uint8_t byte = gp[2 + e / 2];
        const uint32_t nibble = (e % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        partial += q_shared[d] * (kCentroids4Bit[nibble] * norm);
      }
    } else {
      const int64_t base = flat_slot * static_cast<int64_t>(num_kv_heads) * head_size +
                           static_cast<int64_t>(kv_head_idx) * head_size;
      for (int32_t d = lane; d < head_size; d += PA_WARP_SIZE) {
        partial += q_shared[d] * load_f32(key_cache_plain + base + d);
      }
    }

    float qk = pa_warp_reduce_sum(partial) * scale;
    if (softcapping != 1.0f) {
      qk = tanhf(qk / softcapping) * softcapping;
    }
    if (lane == 0) {
      logits[t] = qk;
    }
    qk_max = fmaxf(qk_max, qk);
  }

  // ---- Block-wide softmax reduction (max, then exp + sum) ----
  qk_max = pa_warp_reduce_max(qk_max);
  if (lane == 0) {
    red[warp_id] = qk_max;
  }
  __syncthreads();
  qk_max = (lane < PA_NUM_WARPS) ? red[lane] : -FLT_MAX;
  qk_max = pa_warp_reduce_max(qk_max);
  qk_max = __shfl_sync(0xFFFFFFFFu, qk_max, 0);

  float exp_sum = 0.f;
  for (int32_t t = tid; t < context_len; t += PA_NUM_THREADS) {
    const float val = __expf(logits[t] - qk_max);
    logits[t] = val;
    exp_sum += val;
  }
  exp_sum = pa_warp_reduce_sum(exp_sum);
  if (lane == 0) {
    red[warp_id] = exp_sum;
  }
  __syncthreads();
  exp_sum = (lane < PA_NUM_WARPS) ? red[lane] : 0.f;
  exp_sum = pa_warp_reduce_sum(exp_sum);
  exp_sum = __shfl_sync(0xFFFFFFFFu, exp_sum, 0);
  const float inv_sum = 1.0f / (exp_sum + 1e-6f);
  __syncthreads();

  // ---- Weighted V sum: thread `tid` owns element `tid` of every group ----
  float acc[PA_MAX_GROUPS_PER_HEAD];
#pragma unroll
  for (int32_t g = 0; g < PA_MAX_GROUPS_PER_HEAD; g++) {
    acc[g] = 0.f;
  }

  for (int32_t t = 0; t < context_len; t++) {
    const float w = logits[t] * inv_sum;
    const int32_t block_id = bt_row[t / block_size];
    const int64_t flat_slot = static_cast<int64_t>(block_id) * block_size + (t % block_size);

    if (value_is_turbo) {
      const uint8_t *block_ptr = value_cache_turbo + flat_slot * turbo_block_stride +
                                 static_cast<int64_t>(kv_head_idx) * groups_per_head * BLOCK_BYTES;
      for (int32_t g = 0; g < groups_per_head; g++) {
        const uint8_t *gp = block_ptr + g * BLOCK_BYTES;
        const uint32_t low = gp[0];
        const uint32_t high = gp[1];
        const float norm = static_cast<float>(low + high * 256u) * (NORM_SCALE_MAX / NORM_LEVELS);
        const uint8_t byte = gp[2 + tid / 2];
        const uint32_t nibble = (tid % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        acc[g] += w * (kCentroids4Bit[nibble] * norm);
      }
    } else {
      const int64_t base = flat_slot * static_cast<int64_t>(num_kv_heads) * head_size +
                           static_cast<int64_t>(kv_head_idx) * head_size;
      for (int32_t g = 0; g < groups_per_head; g++) {
        acc[g] += w * load_f32(value_cache_plain + base + g * GROUP + tid);
      }
    }
  }

  // ---- Inverse-rotate the output, iff V went through Turbo4, then store ----
  scalar_t *out_ptr =
      out + (static_cast<int64_t>(seq_idx) * num_heads + head_idx) * head_size;
  for (int32_t g = 0; g < groups_per_head; g++) {
    float v = acc[g];
    if (value_is_turbo) {
      rot_scratch[tid] = v * kWhtSigns2[tid];
      __syncthreads();
      fwht_128(rot_scratch, tid);
      v = rot_scratch[tid] * INV_SQRT_GROUP * kWhtSigns1[tid];
      __syncthreads();
    }
    store_f32(out_ptr + g * GROUP + tid, v);
  }
}

} // namespace turbo4
} // namespace vllm

#define CALL_PAGED_ATTENTION_TURBO4(SCALAR_T)                                \
  vllm::turbo4::paged_attention_turbo4_kernel<SCALAR_T>                      \
      <<<grid, block, shared_mem_bytes, stream>>>(                           \
          reinterpret_cast<SCALAR_T *>(out),                                 \
          reinterpret_cast<const SCALAR_T *>(query),                         \
          reinterpret_cast<const uint8_t *>(key_cache_turbo),                \
          reinterpret_cast<const SCALAR_T *>(key_cache_plain),               \
          reinterpret_cast<const uint8_t *>(value_cache_turbo),              \
          reinterpret_cast<const SCALAR_T *>(value_cache_plain),             \
          key_is_turbo != 0, value_is_turbo != 0, block_table, context_lens, \
          num_kv_heads, head_size, groups_per_head, block_size,              \
          max_num_blocks_per_seq, scale, softcapping);

extern "C" void paged_attention_turbo4(
    void *out,               // [num_seqs, num_heads, head_size]
    const void *query,       // [num_seqs, num_heads, head_size]
    const void *key_cache_turbo, const void *key_cache_plain,
    const void *value_cache_turbo, const void *value_cache_plain,
    int32_t key_is_turbo, int32_t value_is_turbo,
    const int32_t *block_table,  // [num_seqs, max_num_blocks_per_seq]
    const int32_t *context_lens, // [num_seqs]
    int32_t num_seqs, int32_t num_heads, int32_t num_kv_heads,
    int32_t head_size, int32_t groups_per_head, int32_t block_size,
    int32_t max_num_blocks_per_seq, float scale, float softcapping,
    cudaStream_t stream,
    uint32_t io_dtype // 0 => f16; 1 => bf16; 2 => f32
) {
  if (num_seqs <= 0 || num_heads <= 0) {
    return;
  }

  dim3 grid(num_heads, num_seqs);
  dim3 block(vllm::turbo4::PA_NUM_THREADS);
  const int32_t max_context_len = max_num_blocks_per_seq * block_size;
  const size_t shared_mem_bytes =
      (static_cast<size_t>(head_size) + static_cast<size_t>(max_context_len)) *
      sizeof(float);

  if (io_dtype == 0) {
    CUDA_CHECK(cudaFuncSetAttribute(
        vllm::turbo4::paged_attention_turbo4_kernel<uint16_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_mem_bytes));
    CALL_PAGED_ATTENTION_TURBO4(uint16_t)
  } else if (io_dtype == 1) {
    CUDA_CHECK(cudaFuncSetAttribute(
        vllm::turbo4::paged_attention_turbo4_kernel<__nv_bfloat16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_mem_bytes));
    CALL_PAGED_ATTENTION_TURBO4(__nv_bfloat16)
  } else if (io_dtype == 2) {
    CUDA_CHECK(cudaFuncSetAttribute(
        vllm::turbo4::paged_attention_turbo4_kernel<float>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_mem_bytes));
    CALL_PAGED_ATTENTION_TURBO4(float)
  }
  CUDA_CHECK(cudaGetLastError());
}
