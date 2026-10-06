"""B0 packed-cache lifecycle component for a shared, request-partitioned pool.

The b0_fp8 profile connects each layer to an owned pool and checks all ranks
before token publication. The optional original FlashInfer reader runs eagerly;
the generic CUDA operations and sticky device error can be captured;
check() must run outside capture, before publishing any token. A failed pool is
poisoned for its lifetime, including subsequent graph replays and slot reuse.
"""
from dataclasses import dataclass

import torch

from .b0_fp8 import ENTRY_BYTES, WIDTH, pack_cuda, pack_reference


@dataclass(frozen=True)
class Extent:
    owner: str
    base: int
    capacity: int
    generation: int


class PackedCachePool:
    def __init__(self, rows: int, *, device="cpu", attention="triton"):
        if type(rows) is not int or rows <= 0:
            raise ValueError("positive integer pool size required")
        if attention not in ('triton', 'flashinfer'):
            raise ValueError('B0 attention must be triton or flashinfer')
        from .b0_flashinfer import page_padding
        self.attention_backend = attention
        self.reader = None
        self.fallback_calls = 0
        # Rejected writes go to rows. FlashInfer needs a whole physical page
        # containing that sink; only logical rows are exposed to the allocator.
        padding = page_padding(rows) if attention == 'flashinfer' else 1
        self.storage = torch.zeros((rows + padding, ENTRY_BYTES), dtype=torch.uint8, device=device)
        self.valid = torch.zeros(rows + 1, dtype=torch.bool, device=device)
        self.error = torch.zeros((), dtype=torch.bool, device=device)
        self.generations = torch.zeros(rows + 1, dtype=torch.int64, device=device)
        self.rows = rows
        self._extents = {}
        self._generation = 0

    def _placement(self, owner, base, capacity, *, replacing=None):
        if self.storage.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot change cache extents during capture")
        if not isinstance(owner, str) or not owner:
            raise ValueError("nonempty owner required")
        if owner in self._extents and self._extents[owner] is not replacing:
            raise ValueError("duplicate owner")
        if type(base) is not int or type(capacity) is not int or base < 0 or capacity <= 0 or base + capacity > self.rows:
            raise ValueError("extent outside pool")
        if any(e is not replacing and base < e.base + e.capacity and e.base < base + capacity for e in self._extents.values()):
            raise ValueError("overlapping cache extents")

    def reserve(self, owner: str, base: int, capacity: int) -> Extent:
        """Host-side slot lifecycle; call outside graphs with no in-flight users."""
        self._placement(owner, base, capacity)
        self._generation += 1
        extent = Extent(owner, base, capacity, self._generation)
        self._extents[owner] = extent
        self.valid[base:base + capacity].zero_()
        self.generations[base:base + capacity].fill_(extent.generation)
        return extent

    def _owned(self, extent):
        if not isinstance(extent, Extent) or self._extents.get(extent.owner) is not extent:
            raise ValueError("stale or foreign cache extent")

    def release(self, extent):
        self._owned(extent)
        if self.storage.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot release cache extents during capture")
        self.valid[extent.base:extent.base + extent.capacity].zero_()
        self.generations[extent.base:extent.base + extent.capacity].zero_()
        del self._extents[extent.owner]

    def validate_rebind(self, extent, base, capacity, used):
        self._owned(extent)
        self._placement(extent.owner, base, capacity, replacing=extent)
        if type(used) is not int or used < 0 or used > min(extent.capacity, capacity):
            raise ValueError("retained rows outside old or new extent")

    def rebind(self, extent, base, capacity, used):
        """Renew ownership AFTER the caller copies payload+validity to base.

        Never copy generation identities. The retained destination prefix keeps
        its validity; all other old rows and the new tail become unreadable.
        The caller must retire old graphs and quiesce all users before mutation.
        """
        self.validate_rebind(extent, base, capacity, used)
        old_end = extent.base + extent.capacity
        if base > extent.base:
            self.valid[extent.base:min(old_end, base)].zero_()
        if base + used < old_end:
            self.valid[max(extent.base, base + used):old_end].zero_()
        self.valid[base + used:base + capacity].zero_()
        self.generations[extent.base:old_end].zero_()
        self._generation += 1
        fresh = Extent(extent.owner, base, capacity, self._generation)
        self.generations[base:base + capacity].fill_(fresh.generation)
        self._extents[extent.owner] = fresh
        return fresh

    def _indices(self, indices, ndim):
        if indices.ndim != ndim or indices.dtype != torch.int64 or not indices.is_contiguous() or indices.device != self.storage.device:
            raise ValueError("contiguous int64 indices on the pool device required")

    def write(self, extent, positions, values):
        """Store original BF16 rows at request-local positions; no dtype conversion.

        A malformed batch is entirely diverted to the sink and poisons the pool.
        Graph replay re-evaluates index, duplicate and finite-value checks.
        """
        self._owned(extent)
        self._indices(positions, 1)
        if values.shape != (positions.numel(), WIDTH) or values.dtype != torch.bfloat16 or values.device != self.storage.device or values.stride(-1) != 1 or positions.numel() == 0:
            raise ValueError("nonempty BF16 rows matching positions required")
        ordered = positions.sort().values
        bad = ((positions < 0) | (positions >= extent.capacity)).any()
        bad = bad | (self.generations[extent.base] != extent.generation)
        bad = bad | (ordered[1:] == ordered[:-1]).any() | ~torch.isfinite(values).all()
        self.error.logical_or_(bad)
        # Sanitize before packing: the CPU oracle rejects nonfinite inputs and
        # CUDA must never propagate them even on an already rejected batch.
        safe_values = torch.where(torch.isfinite(values), values, 0)
        packed = pack_cuda(safe_values) if values.is_cuda else pack_reference(safe_values)
        safe_positions = torch.where(bad, self.rows, positions + extent.base)
        self.storage.index_copy_(0, safe_positions, packed)
        self.valid.index_fill_(0, safe_positions, True)

    def selected(self, extent, indices, lengths):
        """Map request-local selected keys, rejecting cross-request/unwritten rows.

        -1 remains padding; indices past each length are inactive. Invalid lengths
        or active keys poison the pool while safe indices prevent invalid reads.
        """
        self._owned(extent)
        self._indices(indices, 2)
        if indices.shape[0] == 0 or indices.shape[1] == 0 or lengths.shape != (indices.shape[0],) or lengths.dtype != torch.int32 or not lengths.is_contiguous() or lengths.device != self.storage.device:
            raise ValueError("nonempty selected keys with contiguous int32 lengths required")
        safe_lengths = lengths.clamp(0, indices.shape[1])
        active = torch.arange(indices.shape[1], device=indices.device)[None, :] < safe_lengths[:, None]
        outside = (indices < -1) | (indices >= extent.capacity)
        lookup = (indices.clamp(0, extent.capacity - 1) + extent.base).reshape(-1)
        resident = self.valid[lookup].reshape(indices.shape)
        bad_keys = active & (outside | ((indices >= 0) & ~resident))
        stale = self.generations[extent.base] != extent.generation
        self.error.logical_or_(stale | bad_keys.any() | ((lengths < 0) | (lengths > indices.shape[1])).any())
        mapped = torch.where(~stale & active & ~outside & resident & (indices >= 0), indices + extent.base, -1)
        return mapped.contiguous(), safe_lengths.contiguous()

    def attention(self, extent, q, indices, lengths, scale):
        """CUDA reader bridge; caller must check the whole pool before using output."""
        from .b0_attention import attention_capture

        mapped, safe_lengths = self.selected(extent, indices, lengths)
        fast_indices = mapped
        if self.attention_backend == 'flashinfer' and mapped.shape[1] > 2048 and bool((safe_lengths <= 2048).all()):
            # Columns beyond the active prefix contain no selected key. Keep
            # transition rows with >2048 active keys on the exact generic path.
            fast_indices = mapped[:, :2048].contiguous()
        if self.attention_backend == 'flashinfer' and fast_indices.shape[1] <= 2048:
            from .b0_flashinfer import Reader
            from .b0_attention import AttentionResult
            if self.reader is None:
                self.reader = Reader(self.storage, self.rows)
            values = self.reader(q, fast_indices, safe_lengths, scale)
            return AttentionResult(values, self.error, error_latch=self.error)
        if self.attention_backend == 'flashinfer':
            # Vendor dense/sparse transition can select2051 keys. Preserve all
            # keys using the checked reader rather than silently truncating.
            self.fallback_calls += 1
        return attention_capture(q, self.storage[:self.rows], mapped, safe_lengths, scale,
                                 error_latch=self.error)

    def check(self):
        if self.storage.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("check cache errors outside CUDA graph capture")
        if bool(self.error):
            raise ValueError("B0 cache pool poisoned by an invalid read, write or attention result")


class CacheView:
    """One layer/request binding used by forward's latent dispatch.

    Attention values here are intermediate graph buffers. The shared sticky
    error must be checked by Caches.check_b0 before sampling or publishing.
    """
    def __init__(self, pool, extent, dense_limit):
        pool._owned(extent)
        if type(dense_limit) is not int or dense_limit <= 0:
            raise ValueError("positive dense attention limit required")
        self.pool, self.extent = pool, extent
        self.dense_limit = dense_limit
        self.dense_keys = min(dense_limit, extent.capacity)

    def _positions(self, pos, rows):
        if pos.shape != (1,) or pos.dtype not in (torch.int32, torch.int64) or pos.device != self.pool.storage.device:
            raise ValueError("one device integer position required")
        return pos.to(torch.int64) + torch.arange(rows, device=pos.device, dtype=torch.int64)

    def write(self, values, pos):
        self.pool.write(self.extent, self._positions(pos, values.shape[0]), values)

    def dense(self, q, pos, out, scale):
        if q.shape[0] == 0:
            return out
        positions = self._positions(pos, q.shape[0])
        keys = torch.arange(self.dense_keys, device=q.device, dtype=torch.int64)
        indices = keys[None, :].expand(q.shape[0], -1).contiguous()
        # Keep int64 bounds checks before converting the length for the kernel.
        lengths = positions + 1
        self.pool.error.logical_or_(((lengths < 1) | (lengths > self.dense_keys)).any())
        safe = lengths.clamp(0, self.dense_keys).to(torch.int32)
        values = self.pool.attention(self.extent, q, indices, safe, scale)._values
        out.copy_(values)
        return out

    def sparse(self, q, pos, tokens, counts, out, scale):
        if tokens.dtype not in (torch.int32, torch.int64) or counts.dtype != torch.int32:
            raise ValueError("integer token indices and int32 counts required")
        positions = self._positions(pos, q.shape[0])
        active = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < counts[:, None]
        future = active & (tokens >= 0) & (tokens > positions[:, None])
        missing = (positions >= self.dense_limit) & (counts <= 0)
        self.pool.error.logical_or_(future.any() | missing.any())
        selected = torch.where(future, -1, tokens).to(torch.int64).contiguous()
        values = self.pool.attention(self.extent, q, selected, counts, scale)._values
        # Zero counts name rows already computed by the dense path.
        out.copy_(torch.where((counts > 0)[:, None, None], values, out))
        return out


def _segments(views, rows, count):
    """Validate host bindings before any segmented write; graph capture unsupported."""
    if rows.pos.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("B0 host-bound segmented dispatch must remain eager")
    if len(views) != len(rows.segments) or sum(s[2] for s in rows.segments) != count:
        raise ValueError("B0 segment count does not match request bindings")
    for view, segment in zip(views, rows.segments, strict=True):
        base, pos, n = segment[:3]
        view.pool._owned(view.extent)
        if base != view.extent.base or pos < 0 or n <= 0 or pos + n > view.extent.capacity:
            raise ValueError("B0 segment outside its owned request extent")
    at = 0
    for view, segment in zip(views, rows.segments, strict=True):
        base, pos, n = segment[:3]
        expected = torch.arange(pos, pos + n, device=rows.pos.device, dtype=rows.pos.dtype)
        if not torch.equal(rows.pos[at:at+n], expected) or not bool((rows.base[at:at+n] == base).all()):
            for affected in views:
                affected.pool.error.fill_(True)
            raise ValueError("B0 device segment table differs from host ownership")
        at += n


def segmented_write(views, rows, values):
    _segments(views, rows, values.shape[0])
    at = 0
    for view, segment in zip(views, rows.segments, strict=True):
        n = segment[2]
        view.write(values[at:at+n], rows.pos[at:at+1])
        at += n


def segmented_attention(views, rows, q, tokens, counts, out, scale, *, full_selection=False):
    _segments(views, rows, q.shape[0])
    if (tokens is None) != (counts is None):
        raise ValueError("B0 sparse tokens and counts must both be present")
    at = 0
    for view, segment in zip(views, rows.segments, strict=True):
        _, pos, n = segment[:3]
        position = rows.pos[at:at+1]
        dense = 0 if full_selection else (n if tokens is None else min(n, max(0, view.dense_limit - pos)))
        if full_selection:
            if counts is None:
                raise ValueError("full B0 selection requires counts for every row")
            view.pool.error.logical_or_((counts[at:at+n] <= 0).any())
        if dense:
            view.dense(q[at:at+dense], position, out[at:at+dense], scale)
        if tokens is not None:
            view.sparse(q[at:at+n], position, tokens[at:at+n], counts[at:at+n], out[at:at+n], scale)
        at += n
    return out
