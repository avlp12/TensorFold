"""B0 GLM MLA storage oracle: E4M3FN, 128-value blocks, arbitrary FP32 scales.

This is a codec component, not yet connected to TensorFold's attention readers.
The optional zero RoPE pad matches B0's 656-byte NoPE-GLM packed cache entry.
"""
from __future__ import annotations

import torch

WIDTH = 512
GROUP = 128
SCALE_BYTES = 16
ENTRY_BYTES = 656


def _shape(x: torch.Tensor) -> None:
    if x.ndim != 2 or x.shape[1] != WIDTH or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        raise ValueError('B0 MLA codec requires BF16 [rows,512] with contiguous columns')


def pack_reference(x: torch.Tensor) -> torch.Tensor:
    """CPU definition corresponding to concat_and_cache_ds_mla_kernel."""
    _shape(x)
    if x.device.type != 'cpu':
        raise ValueError('the storage oracle is CPU-only; use pack_cuda for GPU')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('nonfinite KV input')
    grouped = x.float().reshape(-1, WIDTH // GROUP, GROUP)
    scales = (grouped.abs().amax(-1) / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    codes = (grouped / scales[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    out = torch.zeros((x.shape[0], ENTRY_BYTES), dtype=torch.uint8, device='cpu')
    out[:, :WIDTH] = codes.reshape(-1, WIDTH).view(torch.uint8)
    out[:, WIDTH:WIDTH + SCALE_BYTES] = scales.contiguous().view(torch.uint8)
    return out


def unpack_reference(packed: torch.Tensor) -> torch.Tensor:
    if packed.device.type != 'cpu' or packed.ndim != 2 or packed.shape[1] != ENTRY_BYTES or packed.dtype != torch.uint8:
        raise ValueError('expected CPU uint8 [rows,656]')
    codes = packed[:, :WIDTH].contiguous().view(torch.float8_e4m3fn).float().reshape(-1, 4, GROUP)
    scales = packed[:, WIDTH:WIDTH + SCALE_BYTES].contiguous().view(torch.float32)
    return (codes * scales[..., None]).reshape(-1, WIDTH)


def pack_cuda(x: torch.Tensor) -> torch.Tensor:
    _shape(x)
    if x.device.type != 'cuda':
        raise ValueError('pack_cuda requires a CUDA tensor')
    from .b0_fp8_kernel import pack
    return pack(x)
