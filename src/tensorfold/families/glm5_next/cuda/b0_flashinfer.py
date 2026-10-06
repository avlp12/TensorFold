"""Adapter to the original B0 SM120 forward_mqa, over owned linear rows.

The caller has already checked resident rows, generations and request bounds.
An identity page table maps those physical rows without changing selection.
Empty selections name the dedicated sink, never another request's first row.
"""
from types import SimpleNamespace

import torch

PAGE = 64
TOPK = 2048
WORKSPACE_BYTES = 394 * 1024 * 1024


def page_padding(rows):
    # Include the sink at rows, then round the byte allocation to full pages.
    return -(-(rows + 1) // PAGE) * PAGE - rows


def prepare(indices, lengths, rows):
    if indices.ndim != 2 or not 0 < indices.shape[1] <= TOPK:
        raise ValueError('FlashInfer selection must have1..2048 columns')
    if indices.dtype != torch.int64 or lengths.dtype != torch.int32 or lengths.shape != indices.shape[:1]:
        raise ValueError('int64 selected indices and int32 row lengths required')
    active = torch.arange(indices.shape[1], device=indices.device)[None, :] < lengths[:, None]
    selected = torch.where(active, indices, -1)
    empty = ~(selected >= 0).any(-1)
    selected = torch.nn.functional.pad(selected.to(torch.int32), (0, TOPK - indices.shape[1]), value=-1)
    selected[:, 0] = torch.where(empty, rows, selected[:, 0])
    return selected.contiguous(), empty


class Reader:
    def __init__(self, storage, rows):
        import hashlib
        import inspect
        from pathlib import Path
        from . import b0_flashinfer_backend as original
        # This is the original serving image's implementation, not a newer API
        # inferred from its name. Pin the actual source before calling it.
        expected = ORIGINAL_SOURCE_SHA256
        actual = hashlib.sha256(Path(inspect.getfile(original)).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('B0 FlashInfer source changed: ' + actual)
        impl = object.__new__(original.FlashInferMLASparseSM120Impl)
        impl.num_heads = 16
        impl.kv_lora_rank = 512
        impl.qk_nope_head_dim = 256
        impl.kernel_qk_rope_head_dim = 64
        impl.rope_pad = 64
        impl.kv_scale_format = 'arbitrary_fp32'
        impl._workspace_buffer = None
        self.impl, self.storage, self.rows = impl, storage, rows
        self.block_table = torch.arange(storage.shape[0] // PAGE, device=storage.device,
                                        dtype=torch.int32).reshape(1, -1)
        self.calls = 0

    def __call__(self, q, indices, lengths, scale):
        if q.dtype != torch.bfloat16 or q.shape[1:] != (16, 512):
            raise ValueError('B0 FlashInfer requires BF16 [T,16,512] queries')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('B0 FlashInfer ownership adapter currently requires eager execution')
        selected, empty = prepare(indices, lengths, self.rows)
        self.impl.topk_indices_buffer = selected
        self.impl.scale = float(scale)
        metadata = SimpleNamespace(req_id_per_token=torch.zeros(q.shape[0], dtype=torch.int32, device=q.device),
                                   block_table=self.block_table, block_size=PAGE)
        values, _ = self.impl.forward_mqa(q, self.storage.view(-1, PAGE, 656), metadata, None)
        values.masked_fill_(empty[:, None, None], 0)
        if self.calls == 0:
            # Validate the changed path on the actual model's first query, not
            # a separate synthetic load. The prior checked reader is the oracle
            # for adapter geometry/scaling; BF16 outputs need not be bitwise equal.
            from .b0_attention_kernel import attention
            reference = attention(q[:1], self.storage[:self.rows], indices[:1], lengths[:1], scale)
            error = ((values[:1].float() - reference).square().sum(-1)
                     / reference.square().sum(-1).clamp_min(1e-30)).max().item()
            if not torch.isfinite(values[:1]).all() or not error <= 1e-4:
                raise ValueError(f'B0 FlashInfer integrated reader NMSE={error} exceeds1e-4')
            self.validation_nmse = error
            print(f'B0_FLASHINFER_INTEGRATED_READER_CHECK max_head_nmse={error:.12g}', flush=True)
        self.calls += 1
        return values

ORIGINAL_SOURCE_SHA256 = 'f1854c0cce9d749d5ce67dd5132f6541c9d01dec4c2bb414267815ed6fd620c6'
