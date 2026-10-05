"""Eager sparse attention reader for B0's packed FP8 cache.

This component is not yet wired into TensorFold's scheduler or graph capture.
Cache values remain FP8 with per-128-element FP32 scales; products accumulate in
FP32. No persistent BF16 substitute cache is created.
"""
import math
import torch


def validate(q, cache, indices, lengths, scale, *, check_values=True):
    if q.ndim != 3 or q.shape[0] == 0 or q.shape[2] != 512 or q.dtype != torch.bfloat16 or q.stride(-1) != 1:
        raise ValueError('queries must be BF16 [rows,heads,512]')
    if cache.ndim != 2 or cache.shape[0] == 0 or cache.shape[1] != 656 or cache.dtype != torch.uint8 or cache.stride(-1) != 1 or cache.stride(0) % 4:
        raise ValueError('cache must be byte rows of the B0 656-byte layout')
    if indices.ndim != 2 or indices.shape[0] != q.shape[0] or indices.dtype != torch.int64 or not indices.is_contiguous():
        raise ValueError('indices must be contiguous int64 [rows,keys]')
    if lengths.shape != (q.shape[0],) or lengths.dtype != torch.int32 or not lengths.is_contiguous():
        raise ValueError('lengths must be contiguous int32 [rows]')
    if not all(t.device == q.device for t in (cache, indices, lengths)):
        raise ValueError('all tensors must share a device')
    if not math.isfinite(scale) or scale <= 0 or indices.shape[1] == 0 or q.shape[1] == 0:
        raise ValueError('positive finite scale, nonempty heads/key extent required')
    if check_values and bool(((indices < -1) | (indices >= cache.shape[0])).any()):
        raise ValueError('cache index outside -1 padding or resident rows')
    if check_values and bool(((lengths < 0) | (lengths > indices.shape[1])).any()):
        raise ValueError('invalid selected-key length')


def attention_cuda(q, cache, indices, lengths, scale):
    validate(q, cache, indices, lengths, scale)
    if q.device.type != 'cuda':
        raise ValueError('CUDA tensors required')
    from .b0_attention_kernel import attention
    return attention(q, cache, indices, lengths, scale)


class AttentionResult:
    """Call checked() after replay and before consuming outputs.

    Graph capture cannot synchronously inspect device values. Invalid indices
    are masked to prevent out-of-bounds access, but set an error flag so they
    are rejected rather than accepted as a different attention problem.
    """
    def __init__(self, values, error):
        self._values = values
        self._error = error

    def checked(self):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('check attention results outside CUDA graph capture')
        if bool(self._error):
            raise ValueError('invalid selected cache index or length during attention')
        if not bool(torch.isfinite(self._values).all()):
            raise ValueError('nonfinite attention result')
        return self._values


def attention_capture(q, cache, indices, lengths, scale):
    """Capture-compatible component; caller must check the returned result.

    Full engine integration must aggregate/check each layer's error flags before
    publishing tokens. This helper does not implement that scheduler contract.
    """
    validate(q, cache, indices, lengths, scale, check_values=False)
    if q.device.type != 'cuda':
        raise ValueError('CUDA tensors required')
    bad_indices = (indices < -1) | (indices >= cache.shape[0])
    bad_lengths = (lengths < 0) | (lengths > indices.shape[1])
    error = bad_indices.any() | bad_lengths.any()
    safe_indices = torch.where(bad_indices, -1, indices)
    safe_lengths = lengths.clamp(0, indices.shape[1])
    from .b0_attention_kernel import attention
    values = attention(q, cache, safe_indices, safe_lengths, scale)
    return AttentionResult(values, error)
