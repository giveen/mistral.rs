pub const USE_FP8: bool = cfg!(has_fp8);

mod backend;
mod ffi;

pub use backend::{
    concat_and_cache_mla, context_attention_fwd_mla, copy_blocks, fa3_fp8_decode,
    fa3_prepare_decode_metadata, fa3_prepare_paged_metadata, flash_attn_sinks,
    flash_attn_sinks_varlen, flashinfer_decode, flashinfer_mla_decode, gather_kv_cache,
    gather_kv_cache_flashinfer, gather_mla_cache, gather_plain_kv_cache,
    gather_plain_kv_cache_into, gather_turbo4_cache, gather_turbo4_cache_into,
    is_flashinfer_cache, kv_scale_update, paged_attention, paged_attention_turbo4,
    reshape_and_cache, reshape_and_cache_flashinfer, swap_blocks, write_turbo4_cache,
    Fa3DecodeMetadata, Fa3DecodeParams, Fa3DecodeSchedule, Fa3PagedMetadataLayout,
    FlashInferDecodeScratch, FA3_DECODE_MAX_QUERY_LEN, MAX_FUSED_DECODE_CONTEXT_TOKENS,
    USE_FA3_FP8_PAGED,
};
