"""B0 packed-cache lifecycle component for a shared, request-partitioned pool.

This is NOT enabled by TF_GLM_KV or wired to the engine scheduler yet. Each layer
needs its own pool. CUDA operations and the sticky device error can be captured;
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
    def __init__(self, rows: int, *, device="cpu"):
        if type(rows) is not int or rows <= 0:
            raise ValueError("positive integer pool size required")
        # The last row is a write sink for rejected indices, never readable.
        self.storage = torch.zeros((rows + 1, ENTRY_BYTES), dtype=torch.uint8, device=device)
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
