"""Standalone B0 MLA packing leaf; no attention or model integration implied."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(X, C, S, RAW, XS: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, 512)
    x = tl.load(X + row * XS + k).to(tl.float32)
    groups = tl.reshape(x, (4, 128))
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(groups), 1), 448.0), 1.1754943508222875e-38)
    values = tl.div_rn(groups, scale[:, None])
    values = tl.minimum(tl.maximum(values, -448.0), 448.0)
    tl.store(C + row * 656 + k, tl.reshape(values, (512,)).to(tl.float8e4nv))
    tl.store(S + row * 164 + 128 + tl.arange(0, 4), scale)
    tl.store(RAW + row * 656 + 528 + tl.arange(0, 128), 0)


def pack(x):
    out = torch.empty((x.shape[0], 656), dtype=torch.uint8, device=x.device)
    _pack[(x.shape[0],)](x, out.view(torch.float8_e4m3fn), out.view(torch.float32), out,
                       XS=x.stride(0), num_warps=4)
    return out
