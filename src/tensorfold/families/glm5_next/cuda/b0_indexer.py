"""Adapters around the pinned original B0 query/pool quantization kernels.

This module is not yet wired into the engine's index scoring/top-k path. Its
132-byte canonical rows are page-size-one views of the original cache layout.
Real B0 multi-row pages place all codes before all scales; never reshape those
pages into canonical rows without extracting both regions.
"""
import torch

WIDTH = 128
HEADS = 32  # actual target config rev25a44; do not infer from older source comments
ROW_BYTES = 132


def ape_dtype(mode):
    if mode not in ('vendor', 'b0'):
        raise ValueError('indexer mode must be vendor or b0')
    return torch.float32 if mode == 'b0' else torch.bfloat16


def _error(error, device):
    if error.shape != () or error.dtype != torch.bool or error.device != device:
        raise ValueError('scalar boolean error latch on the input device required')


def validate_query(q, weights, error):
    if q.ndim != 3 or q.shape[0] == 0 or q.shape[1:] != (HEADS, WIDTH) or q.dtype != torch.bfloat16:
        raise ValueError('B0 index queries must be BF16 [rows,32,128]')
    if weights.shape != q.shape[:2] or weights.dtype != torch.bfloat16 or weights.device != q.device:
        raise ValueError('original BF16 head weights must match query rows/heads')
    _error(error, q.device)


def query(q, weights, error):
    """Original FWHT/FP8 query with B0 scale folding for target index_n_heads=32."""
    validate_query(q, weights, error)
    if q.device.type != 'cuda':
        raise ValueError('query quantization requires CUDA')
    from .b0_indexer_kernel import fwht128_quant_fp8

    finite_q, finite_w = torch.isfinite(q), torch.isfinite(weights)
    error.logical_or_(~finite_q.all() | ~finite_w.all())
    clean_q = torch.where(finite_q, q, 0).reshape(-1, WIDTH).contiguous()
    clean_w = torch.where(finite_w, weights, 0)
    codes, scales = fwht128_quant_fp8(clean_q)
    codes = codes.view(q.shape)
    scales = scales.view(q.shape[0], HEADS, 1)
    folded = (clean_w.unsqueeze(-1) * scales * (WIDTH ** -0.5 * HEADS ** -0.5)).squeeze(-1)
    return codes, folded


def validate_pools(keys, gates, ape, error):
    if keys.ndim != 3 or keys.shape[0] == 0 or keys.shape[1:] != (4, WIDTH) or keys.dtype != torch.bfloat16:
        raise ValueError('B0 pool keys must be BF16 [pools,4,128]')
    if gates.shape != keys.shape or gates.dtype != torch.bfloat16 or gates.device != keys.device:
        raise ValueError('original BF16 gate scores must match the pool keys')
    if ape.shape != (4, WIDTH) or ape.dtype != torch.float32 or ape.device != keys.device:
        raise ValueError('original FP32 pool position bias required; do not downcast to BF16')
    _error(error, keys.device)


def compress(keys, gates, ape, error):
    """Return original B0 quantized pools as canonical [pools,132] byte rows."""
    validate_pools(keys, gates, ape, error)
    if keys.device.type != 'cuda':
        raise ValueError('pool quantization requires CUDA')
    from .b0_indexer_kernel import kpool_compress_and_write_cache

    fk, fg, fa = torch.isfinite(keys), torch.isfinite(gates), torch.isfinite(ape)
    error.logical_or_(~fk.all() | ~fg.all() | ~fa.all())
    clean_k = torch.where(fk, keys, 0)
    clean_g = torch.where(fg, gates, 0)
    clean_a = torch.where(fa, ape, 0)
    # No untrusted scatter indices: ask the original kernel for compressed rows.
    # The dummy cache is not written (write_cache=False).
    dummy = torch.empty((1, 1, ROW_BYTES), dtype=torch.uint8, device=keys.device)
    loc = torch.zeros(keys.shape[0], dtype=torch.int64, device=keys.device)
    codes, scales = kpool_compress_and_write_cache(dummy, clean_k, clean_g, clean_a, loc,
                                                  pool_size=4, round_scale=True,
                                                  return_compressed=True, write_cache=False)
    return torch.cat((codes.view(torch.uint8), scales[:, None].contiguous().view(torch.uint8)), dim=1)


def page_rows(cache):
    """Extract logical rows from original B0 [pages,page_size,132] byte storage.

    Each physical page is [page_size*128 code bytes, page_size*4 scale bytes],
    NOT interleaved [128 codes,4 scale] rows. Copying is explicit here; a future
    paged scoring backend can consume the original page layout without this copy.
    """
    if cache.ndim != 3 or cache.shape[2] != ROW_BYTES or cache.shape[0] == 0 or cache.shape[1] == 0 or cache.dtype != torch.uint8 or not cache.is_contiguous():
        raise ValueError('contiguous uint8 [pages,page_size,132] cache required')
    pages, size, _ = cache.shape
    flat = cache.view(pages, size * ROW_BYTES)
    codes = flat[:, :size * WIDTH].reshape(pages, size, WIDTH)
    scales = flat[:, size * WIDTH:].reshape(pages, size, 4)
    return torch.cat((codes, scales), dim=-1).reshape(-1, ROW_BYTES)


def validate_selection(q, weights, packed, positions, error, prefill, short_prefill):
    validate_query(q, weights, error)
    if packed.ndim != 2 or packed.shape[1] != ROW_BYTES or packed.dtype != torch.uint8 or not packed.is_contiguous() or packed.device != q.device:
        raise ValueError('resident canonical uint8 [pools,132] required')
    if positions.shape != (q.shape[0],) or positions.dtype not in (torch.int32, torch.int64) or positions.device != q.device:
        raise ValueError('one integer token position per query required')
    if type(prefill) is not bool or type(short_prefill) is not bool or (short_prefill and not prefill):
        raise ValueError('short_prefill is only valid on the explicit prefill path')
    if packed.shape[0] * 4 + 3 >= 2**31:
        raise ValueError('indexer positions exceed int32 kernel range')


def select(q, weights, packed, positions, error, *, prefill, short_prefill=False):
    """Unpaged B0 score/top-k adapter using the original vLLM CUDA primitives.

    No model/engine calls: only DeepGEMM MQA logits and original top-k/expansion
    kernels. Paged-decode performance and full engine admission remain separate.
    ``short_prefill`` is the original metadata predicate (whole chunk <=2048),
    not an implicit per-row substitute for a long prefill.
    """
    validate_selection(q, weights, packed, positions, error, prefill, short_prefill)
    if q.device.type != 'cuda':
        raise ValueError('B0 index selection requires CUDA')
    q_codes, folded = query(q, weights, error)
    p64 = positions.to(torch.int64)
    visible = (p64 + 1) // 4
    bad_pos = (p64 < 0) | (visible > packed.shape[0])
    error.logical_or_(bad_pos.any())
    safe_pos = torch.where(bad_pos, 0, p64).to(torch.int32)
    if short_prefill:
        error.logical_or_((p64 >= 2048).any())
        cols = torch.arange(2048, device=q.device, dtype=torch.int32)
        tokens = torch.where(cols[None, :] <= safe_pos[:, None], cols[None, :], -1)
        return tokens, (tokens >= 0).sum(-1).to(torch.int32)

    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    from vllm import _custom_ops  # registers the original torch.ops._C top-k kernels
    from .b0_indexer_kernel import expand_pools_and_append_tail

    values = packed[:, :WIDTH].contiguous()
    scales = packed[:, WIDTH:].contiguous().view(torch.float32).reshape(-1)
    bad_keys = (((values & 127) == 127).any(-1) | ~torch.isfinite(scales) | (scales <= 0))
    error.logical_or_(bad_keys.any())
    values = torch.where(bad_keys[:, None], 0, values).view(torch.float8_e4m3fn)
    scales = torch.where(bad_keys, 1, scales)
    # Original top-k always returns 512 slots. Explicit masked padding avoids
    # exposing unallocated columns for short decode contexts. It is outside
    # every query's exclusive end, so no padded key can be selected.
    padding = max(0, 512 - values.shape[0])
    if padding:
        values = torch.cat((values, values.new_zeros((padding, WIDTH))), 0)
        scales = torch.cat((scales, scales.new_ones(padding)), 0)
    start = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
    end = visible.clamp(0, packed.shape[0]).to(torch.int32)
    logits = fp8_fp4_mqa_logits((q_codes, None), (values, scales), folded, start, end, clean_logits=True)
    ids = torch.full((q.shape[0], 512), -1, dtype=torch.int32, device=q.device)
    if prefill:
        torch.ops._C.top_k_per_row_prefill(logits, start, end, ids, q.shape[0], logits.stride(0), logits.stride(1), 512)
    else:
        torch.ops._C.top_k_per_row_decode(logits, 1, end, ids, q.shape[0], logits.stride(0), logits.stride(1), 512)
    invalid = (ids < -1) | (ids >= end[:, None])
    error.logical_or_(invalid.any())
    ids = torch.where(invalid, -1, ids)
    # Preserve original B0's select_k-1 history groups plus incomplete tail.
    expanded = expand_pools_and_append_tail(ids[:, :511].to(torch.int64), safe_pos + 1, 4)
    tokens = torch.full((q.shape[0], 2048), -1, dtype=torch.int32, device=q.device)
    tokens[:, :expanded.shape[1]] = expanded
    future = tokens > safe_pos[:, None]
    error.logical_or_(future.any())
    tokens = torch.where(future, -1, tokens)
    return tokens, (tokens >= 0).sum(-1).to(torch.int32)
