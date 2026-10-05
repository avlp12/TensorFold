"""Adapters around the pinned original B0 query/pool quantization kernels.

This module is not yet wired into the engine's index scoring/top-k path. Its
132-byte canonical rows are page-size-one views of the original cache layout.
Real B0 multi-row pages place all codes before all scales; never reshape those
pages into canonical rows without extracting both regions.
"""
import torch

WIDTH = 128
HEADS = 16
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
        raise ValueError('B0 index queries must be BF16 [rows,16,128]')
    if weights.shape != q.shape[:2] or weights.dtype != torch.bfloat16 or weights.device != q.device:
        raise ValueError('original BF16 head weights must match query rows/heads')
    _error(error, q.device)


def query(q, weights, error):
    """Original FWHT/FP8 query with exact B0 scale-folding order and zero heads."""
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
    codes = torch.cat((codes, codes.new_zeros((q.shape[0], 32 - HEADS, WIDTH))), dim=1)
    folded = torch.cat((folded, folded.new_zeros((q.shape[0], 32 - HEADS))), dim=1)
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
