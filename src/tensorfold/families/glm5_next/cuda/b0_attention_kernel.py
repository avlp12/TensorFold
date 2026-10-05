"""Reference CUDA attention component with B0 block-scale FP8 reads."""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q, C, S, IDX, LEN, OUT,
               QS0: tl.constexpr, QS1: tl.constexpr, CS: tl.constexpr,
               H: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr,
               TILE: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 512)
    q = tl.load(Q + row * QS0 + head * QS1 + d).to(tl.float32)
    length = tl.load(LEN + row)
    m = tl.full((), float('-inf'), tl.float32)
    total = tl.full((), 0., tl.float32)
    acc = tl.full((512,), 0., tl.float32)
    for start in range(tl.cdiv(K, TILE)):
        pos = start * TILE + tl.arange(0, TILE)
        token = tl.load(IDX + row * K + pos, mask=pos < length, other=-1)
        ok = (pos < length) & (token >= 0)
        codes = tl.load(C + token[:, None] * CS + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        scales = tl.load(S + token[:, None] * (CS // 4) + 128 + d[None, :] // 128,
                         mask=ok[:, None], other=1.)
        kv = codes * scales
        scores = tl.sum(kv * q[None, :], 1) * SCALE
        scores = tl.where(ok, scores, float('-inf'))
        new_m = tl.maximum(m, tl.max(scores, 0))
        alpha = tl.where(m == float('-inf'), 0., tl.exp(m - new_m))
        prob = tl.where(ok, tl.exp(scores - new_m), 0.)
        acc = acc * alpha + tl.sum(kv * prob[:, None], 0)
        total = total * alpha + tl.sum(prob, 0)
        m = new_m
    result = tl.where(total > 0, acc / total, 0.)
    tl.store(OUT + (row * H + head) * 512 + d, result)


def attention(q, cache, indices, lengths, scale):
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    _attention[(q.shape[0], q.shape[1])](
        q, cache.view(torch.float8_e4m3fn), cache.view(torch.float32), indices,
        lengths, out, QS0=q.stride(0), QS1=q.stride(1), CS=cache.stride(0),
        H=q.shape[1], K=indices.shape[1], SCALE=float(scale), TILE=16,
        num_warps=8)
    return out
