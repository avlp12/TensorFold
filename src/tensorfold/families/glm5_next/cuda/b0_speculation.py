"""Actual B0 selector and standard rejection primitives in TensorFold rounds.

This is an eager integration path. It gathers target vocabulary shards and uses
vLLM's low-level kernels, never a vLLM model or inference endpoint. Admission is
kept closed until this path and the model lifecycle are validated together.
"""
import math
import numpy as np
import torch


class Proposal(list):
    """Chosen draft chain plus its conditional, pre-temperature sparse logits."""
    def __init__(self, tokens, candidates, scores):
        super().__init__(tokens)
        self.candidates, self.scores = candidates, scores


def settings(sampling):
    temp = float(sampling.temperature) if sampling is not None else 0.0
    seed = int(sampling.seed) if sampling is not None else 0
    if not math.isfinite(temp) or temp < 0 or not 0 <= seed < 2**63:
        raise ValueError('B0 sampling requires finite nonnegative temperature and int64 seed')
    return temp, seed


def propose(d, tokens, values, projected, anchor, first, sampling):
    """Use the original BF16 selector expression and original GPU walk kernel."""
    from vllm.model_executor.models.qwen3_dflash2 import _score_edges
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import _selector_walk_kernel
    from vllm.triton_utils import triton

    depth, k = tokens.shape
    if not 1 <= depth <= 3 or k != d.top_k or d.block != 4 or d.edge != 1 or d.noise != 1:
        raise ValueError('B0 DFlash2 requires block4, edge1, noise1 and at most3 drafts')
    temp, seed = settings(sampling)
    # Only the referenced codebook rows cross to the GPU; both codebooks retain
    # their original BF16 values. Remapping changes lookup addresses, not order.
    ids, inverse = np.unique(np.concatenate(([anchor], tokens.reshape(-1))), return_inverse=True)
    dev = d.dev
    pred = torch.as_tensor(d.pred[ids], dtype=torch.bfloat16, device=dev)
    succ = torch.as_tensor(d.succ[ids], dtype=torch.bfloat16, device=dev)
    mapped = torch.as_tensor(inverse[1:].reshape(1, depth, k), dtype=torch.int64, device=dev)
    hidden = torch.as_tensor(projected, dtype=torch.bfloat16, device=dev).reshape(1, depth, -1)
    unary = torch.as_tensor(values, dtype=torch.float32, device=dev).reshape(1, depth, k)
    anchors = torch.tensor([inverse[0]], dtype=torch.int64, device=dev)
    scores = _score_edges(pred, succ, mapped, unary, hidden, anchors, k).contiguous()
    candidates = torch.as_tensor(tokens, dtype=torch.int64, device=dev).contiguous()
    pos = torch.arange(first, first + depth, dtype=torch.int64, device=dev)
    mapping = torch.zeros(depth, dtype=torch.int32, device=dev)
    temperatures = torch.tensor([temp], dtype=torch.float32, device=dev)
    seeds = torch.tensor([seed], dtype=torch.int64, device=dev)
    chosen = torch.empty(depth, dtype=torch.int64, device=dev)
    realized = torch.empty((depth, k), dtype=torch.float32, device=dev)
    _selector_walk_kernel[(1,)](scores, candidates, pos, mapping, temperatures,
        seeds, chosen, realized, num_steps=depth, top_k=k,
        BLOCK_K=triton.next_power_of_2(k), SAMPLE_PROBABILISTIC=True,
        USE_FP64=False, num_warps=1)
    return Proposal(chosen.tolist(), candidates, realized)


def full_logits(w, local):
    """Gather arbitrary contiguous vocabulary partitions in rank order."""
    if w.world == 1:
        if w.vocab_offset != 0 or local.shape[1] != w.cfg.vocab:
            raise ValueError('single-rank vocabulary is incomplete')
        return local.float().contiguous()
    if w.comm is None:
        raise ValueError('B0 rejection requires the actual TP communicator')
    layout = w.meta.get('b0_rejection_vocab_layout')
    if layout is None:
        mine = torch.tensor([w.vocab_offset, local.shape[1]], dtype=torch.int64, device=local.device)
        all_sizes = torch.empty(w.world * 2, dtype=torch.int64, device=local.device)
        w.comm.all_gather(mine, all_sizes)
        layout = all_sizes.view(w.world, 2).tolist()
        end = 0
        for offset, size in layout:
            if offset != end or size <= 0:
                raise ValueError('TP vocabulary partitions must cover every token exactly once')
            end += size
        if end != w.cfg.vocab:
            raise ValueError('TP vocabulary does not match the target checkpoint')
        w.meta['b0_rejection_vocab_layout'] = layout
    if layout[w.rank] != [w.vocab_offset, local.shape[1]]:
        raise ValueError('TP vocabulary layout changed')
    width = max(size for _, size in layout)
    padded = torch.full((local.shape[0], width), -float('inf'), dtype=torch.float32, device=local.device)
    padded[:, :local.shape[1]].copy_(local)
    gathered = torch.empty(w.world * padded.numel(), dtype=torch.float32, device=local.device)
    w.comm.all_gather(padded.reshape(-1), gathered)
    gathered = gathered.view(w.world, local.shape[0], width)
    return torch.cat([gathered[r, :, :size] for r, (_, size) in enumerate(layout)], dim=1)


def verify(w, logits, tokens, proposal, position, sampling, *, stop_eos=False):
    """Return exactly the accepted prefix plus original rejection/bonus sample."""
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    rows = len(tokens)
    if not 1 <= rows <= 4 or logits.shape[0] != rows:
        raise ValueError('B0 verification window must contain pending plus at most3 drafts')
    depth = rows - 1
    if depth and (not isinstance(proposal, Proposal) or list(proposal[:depth]) != list(tokens[1:])):
        raise ValueError('conditional proposal logits do not match verified draft tokens')
    temp, seed = settings(sampling)
    target = full_logits(w, logits)
    dev, vocab = target.device, target.shape[1]
    mapping = torch.zeros(rows, dtype=torch.int32, device=dev)
    temperatures = torch.tensor([temp], dtype=torch.float32, device=dev)
    seeds = torch.tensor([seed], dtype=torch.int64, device=dev)
    if temp > 0:
        target = target.clone()  # preserve engine scratch/logits for later consumers
        if temp != 1:
            apply_temperature(target, mapping, temperatures)
        min_p, top_p = float(sampling.min_p), float(sampling.top_p)
        if not 0 <= min_p <= 1 or not 0 < top_p <= 1:
            raise ValueError('invalid B0 min_p/top_p')
        if min_p:
            apply_min_p(target, mapping, torch.tensor([min_p], dtype=torch.float32, device=dev))
        k = int(sampling.top_k)
        top_k = torch.full((rows,), k, dtype=torch.int32, device=dev) if 0 < k < vocab else None
        top_p_t = torch.full((rows,), top_p, dtype=torch.float32, device=dev) if top_p < 1 else None
        if top_k is not None or top_p_t is not None:
            target = apply_top_k_top_p(target, top_k, top_p_t)
    # Always provide probabilistic draft storage, including greedy target calls.
    # A short round has no draft rows, but the ABI still uses the fixed k=3.
    draft = torch.full((1, 3, vocab), -float('inf'), dtype=torch.float32, device=dev)
    if depth:
        draft[0, :depth].scatter_(1, proposal.candidates[:depth], proposal.scores[:depth])
    out, count = rejection_sample(target, draft,
        torch.tensor(tokens, dtype=torch.int64, device=dev),
        torch.tensor([0, rows], dtype=torch.int32, device=dev),
        torch.arange(position, position + rows, dtype=torch.int64, device=dev),
        torch.zeros(1, dtype=torch.int32, device=dev), mapping,
        torch.arange(rows, dtype=torch.int32, device=dev), temperatures, seeds, 3,
        synthetic_conditional_rates=None, use_fp64=False, use_block_verification=False)
    n = int(count[0].item())
    if not 1 <= n <= rows:
        raise ValueError('invalid original rejection output length')
    chosen = out[0, :n].tolist()
    if stop_eos:
        for i, token in enumerate(chosen):
            if token in w.cfg.eos:
                chosen = chosen[:i + 1]
                break
    return chosen
