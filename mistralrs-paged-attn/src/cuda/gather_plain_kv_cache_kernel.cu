#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

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

/// Gathers K or V from Turbo4's plain (unquantized elementwise) fallback
/// cache -- shape (num_blocks, block_size, kv_heads, head_size), the layout
/// `write_plain_cache`/`read_plain_cache` in turbo_quant.rs use for the
/// per-layer/per-side auto-asymmetric and layer-adaptive fallback (NOT the
/// vLLM x-split layout `gather_kv_cache_kernel.cu` expects for native
/// f16/bf16/f32/f8e4m3 caches, so that kernel can't be reused here).
///
/// One CUDA block per output token, cooperatively copying kv_heads *
/// head_size elements. GPU-resident block_table/cu_seq_lens addressing
/// mirrors gather_kv_cache_kernel.cu, so this is graph-capture safe.
template <typename scalar_t>
__global__ void gather_plain_kv_cache_kernel(
    const scalar_t *__restrict__ cache, // [num_blocks, block_size, kv_heads, head_size]
    scalar_t *__restrict__ out,         // [num_tokens, kv_heads, head_size]
    const int32_t *__restrict__ block_table, // [batch, max_blocks]
    const int32_t *__restrict__ cu_seq_lens, // [batch + 1]
    const int32_t num_tokens, const int32_t num_seqs,
    const int32_t block_size, const int32_t block_table_stride,
    const int32_t kv_heads, const int32_t head_size) {
  const int32_t token_id = blockIdx.x;
  if (token_id >= num_tokens) {
    return;
  }

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
  if (batch_id >= num_seqs) {
    return;
  }

  const int32_t batch_offset = token_id - cu_seq_lens[batch_id];
  const int32_t block_table_id = batch_offset / block_size;
  const int32_t slot = batch_offset % block_size;
  const int32_t block_id =
      block_table[batch_id * block_table_stride + block_table_id];

  const int64_t src_base =
      (static_cast<int64_t>(block_id) * block_size + slot) *
      static_cast<int64_t>(kv_heads) * head_size;
  const int64_t out_base = static_cast<int64_t>(token_id) * kv_heads * head_size;

  const int32_t n = kv_heads * head_size;
  for (int32_t i = threadIdx.x; i < n; i += blockDim.x) {
    out[out_base + i] = cache[src_base + i];
  }
}

} // namespace vllm

#define CALL_GATHER_PLAIN_KV(SCALAR_T)                                       \
  vllm::gather_plain_kv_cache_kernel<SCALAR_T><<<grid, block, 0, stream>>>(  \
      reinterpret_cast<const SCALAR_T *>(cache),                             \
      reinterpret_cast<SCALAR_T *>(out), block_table, cu_seq_lens,           \
      num_tokens, num_seqs, block_size, block_table_stride, kv_heads,        \
      head_size);

extern "C" void gather_plain_kv_cache(
    const void *cache, // [num_blocks, block_size, kv_heads, head_size]
    void *out,         // [num_tokens, kv_heads, head_size]
    const int32_t *block_table, // [batch, max_blocks]
    const int32_t *cu_seq_lens, // [batch + 1]
    int32_t num_tokens, int32_t num_seqs, int32_t block_size,
    int32_t block_table_stride, int32_t kv_heads, int32_t head_size,
    cudaStream_t stream,
    uint32_t dtype // 0 => f16; 1 => bf16; 2 => f32
) {
  if (num_tokens <= 0) {
    return;
  }
  dim3 grid(num_tokens);
  dim3 block(std::min(kv_heads * head_size, 512));

  if (dtype == 0) {
    CALL_GATHER_PLAIN_KV(uint16_t)
  } else if (dtype == 1) {
    CALL_GATHER_PLAIN_KV(__nv_bfloat16)
  } else if (dtype == 2) {
    CALL_GATHER_PLAIN_KV(float)
  }
  CUDA_CHECK(cudaGetLastError());
}
