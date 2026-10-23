"""Neuron embedding lookups with int32 ids and a sort-based backward (plain PyTorch, no NKI).

`embedding(idx, weight)` is F.embedding(idx, weight) with
(a) int32 ids: int64 ids are slower to index with on Neuron, and a d24 forward does 13 lookups (wte + 12
    value-embedding tables);
(b) in training, a backward without a scatter. The default embedding backward is an index_add scatter that
    updates one row at a time (~13 ms per 32768 x 1536 table at 16384 tokens). Here the ids are sorted
    once per batch on the host (shared by all tables). In sorted order, tile j of 128 tokens touches the vocab tiles
    (128 rows each) vlo_j..vhi_j, and these (token tile, vocab tile) pairs form a staircase of at most
    N/128 + V/128 - 1 pairs. Each pair is a 128x128 one-hot matmul (one bmm over all pairs), and the pairs of one
    vocab tile are summed by a 0/1 matmul. Everything accumulates in fp32 and rounds once to bf16, like the default
    backward (bitwise equal on real batches).
The plan is cached on the ids tensor (identity + version counter), so the 12 value-embedding lookups of a forward
reuse the wte plan, and a new batch (or an in-place overwrite of the ids buffer) rebuilds it.
"""
import torch
import torch.nn.functional as F

_TILE = 128


def _grad_weight(g, perm, jp, vp, local):
    """g (N, D) grad rows. perm (N,): sorted position -> row of g. Pair p: token tile jp[p], vocab tile vp[p]
    (padding pairs have vp = V/128), local[p] (128,) = each token's row inside vocab tile vp[p]."""
    N, D = g.shape
    P = jp.numel()
    VT = P - N // _TILE                                                       # P = N/128 + V/128 (padded)
    G = g.index_select(0, perm).view(N // _TILE, _TILE, D).index_select(0, jp)   # (P, 128 tokens, D)
    rows = torch.arange(_TILE, dtype=local.dtype, device=g.device)
    oh = (local[:, :, None] == rows).to(torch.float32)                        # (P, token, row)
    part = torch.bmm(oh.transpose(1, 2), G.float())                           # (P, row, D): one pair's rows
    tiles = torch.arange(VT, dtype=vp.dtype, device=g.device)
    A = (vp[None, :] == tiles[:, None]).to(torch.float32)                     # (V/128, P): pair -> vocab tile
    return (A @ part.view(P, _TILE * D)).view(VT * _TILE, D).to(g.dtype)


_compiled = None


def _grad_weight_compiled(*args):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(_grad_weight, backend="neuron", dynamic=False)
    return _compiled(*args)


class _EmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, idx, *plan):
        ctx.save_for_backward(*plan)
        return F.embedding(idx, weight)

    @staticmethod
    def backward(ctx, grad_output):
        plan = ctx.saved_tensors
        g = grad_output.reshape(plan[0].numel(), -1).contiguous()
        return (_grad_weight_compiled(g, *plan), None) + (None,) * len(plan)


def _plan(idx32, num_weights):
    """(perm, jp, vp, local) on idx32's device, from a sort of the ids on the host."""
    sid, perm = torch.sort(idx32.reshape(-1).cpu())
    nT, VT = sid.numel() // _TILE, num_weights // _TILE
    key = sid // _TILE
    vlo, vhi = key[0::_TILE], key[_TILE - 1::_TILE]
    span = vhi - vlo + 1
    jp = torch.repeat_interleave(torch.arange(nT), span)
    vp = vlo[jp] + torch.arange(jp.numel()) - (torch.cumsum(span, 0) - span)[jp]
    pad = nT + VT - jp.numel()                                                # >= 1
    jp = F.pad(jp, (0, pad))
    vp = F.pad(vp, (0, pad), value=VT)                                        # padding pairs hit no vocab tile
    local = sid.view(nT, _TILE)[jp] - _TILE * vp[:, None]
    return tuple(t.to(torch.int32).to(idx32.device) for t in (perm, jp, vp, local))


_last = None


def embedding(idx, weight):
    """F.embedding(idx, weight) on Neuron: int32 ids; the sort-based backward when training."""
    global _last
    training = (weight.requires_grad and torch.is_grad_enabled()
                and idx.numel() % _TILE == 0 and weight.shape[0] % _TILE == 0)
    try:
        version = idx._version
    except RuntimeError:  # inference tensors have no version counter: no caching
        version = None
    key = (version, weight.shape[0], training)
    if version is not None and _last is not None and _last[0] is idx and _last[1] == key:
        idx32, plan = _last[2], _last[3]
    else:
        idx32 = idx.to(torch.int32) if idx.dtype == torch.int64 else idx
        plan = _plan(idx32, weight.shape[0]) if training else None
        if version is not None:
            _last = (idx, key, idx32, plan)
    if plan is None:
        return F.embedding(idx32, weight)
    return _EmbeddingFn.apply(weight, idx32, *plan)
