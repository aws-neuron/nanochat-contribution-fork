"""Neuron embedding lookups whose backward runs the NKI scatter-add kernel (nanochat/trn/kernels/embedding.py).

`embedding(idx, weight)` is F.embedding(idx, weight) with (a) int32 indices -- on Neuron, int64 indices take a host
round trip (device->host copy + two stream synchronizes) on every call, and a forward
does 13 lookups (wte + 12 value-embedding tables); the gathered rows are bitwise identical -- and (b) in training the
kernel backward: NKI sorted one-hot-matmul scatter-add (exact fp32 sum, one bf16 rounding, like torch's formula)
instead of the ~11 ms eager HLO scatter. One plan (token sort, ~1.5 ms of torch ops) per forward is shared by wte and
every value-embedding table: the cast indices and the plan are cached on the ids tensor identity + version counter,
so the 12 value-embedding lookups of the same forward reuse them and a new batch (new tensor, or the same buffer
overwritten in place) rebuilds. Plain path (no plan): eval / no grad, LNC1, unsupported shapes
(N % 256 != 0, V % 128 != 0, D % 512 != 0, D > 2048).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from nanochat.trn.kernels import embedding as _k
from nanochat.trn.kernels.embedding import N_TILE, P_MAX, PHASE_B_BATCH



def _plan_rest(sid, perm, num_weights: int, B: int, PB: int):
    """sid, perm (N,) int32 (sorted ids and the permutation). Returns the 7 kernel plan tensors."""
    N = sid.numel()
    n_tiles = N // P_MAX
    T2 = n_tiles // 2
    V_T = num_weights // P_MAX
    NONE_S = 2 * P_MAX                      # scratch has 256 rows: row 256 = out of bounds = no store
    dev = sid.device
    key = sid >> 7
    sid_T = sid.view(n_tiles, P_MAX).t()                                        # (128, n_tiles)
    perm_T = perm.view(n_tiles, P_MAX).t().contiguous()
    vlo = key[0::P_MAX]
    vhi = key[P_MAX - 1::P_MAX]
    span = vhi - vlo + 1
    prev_hi = torch.cat([vhi.new_full((1,), -1), vhi[:-1]])
    nxt_lo = torch.cat([vlo[1:], vlo.new_full((1,), -1)])
    multi = span >= 2
    OOB = torch.full_like(vlo, num_weights)
    keep0 = (vlo == prev_hi).to(torch.int32)
    dest0 = torch.where(multi | (vlo != nxt_lo), vlo * P_MAX, OOB)
    v1 = torch.where(multi, vhi, torch.full_like(vhi, -1))
    keep1 = (~multi).to(torch.int32)
    dest1 = torch.where(multi & (vhi != nxt_lo), vhi * P_MAX, OOB)
    sdest0 = torch.full_like(vlo, NONE_S)
    sdest1 = torch.full_like(vlo, NONE_S)
    # row split: core 1 starts a fresh chain at tile T2; if vocab tile v* = vhi[T2-1] == vlo[T2] straddles the split,
    # both cores park their fp32 partial for v* in the scratch (core 0 at slot 1 of tile T2-1, core 1 at slot 0 of the
    # tile je where its chain for v* ends) and the fixup step stores the sum.
    jidx = torch.arange(n_tiles, dtype=torch.int32, device=dev)
    straddle = vhi[T2 - 1] == vlo[T2]
    vs = vhi[T2 - 1]
    keep0 = torch.where(jidx == T2, torch.zeros_like(keep0), keep0)
    is_last0 = jidx == T2 - 1
    # core-1 chain of v*: tiles j >= T2 with span 1 and vlo[j+1] == v* continue it; je = T2 + length of that leading run
    cont = ((span == 1) & (nxt_lo == vs) & (jidx >= T2)).to(torch.float32)
    tri_incl = torch.triu(torch.ones(n_tiles, n_tiles, dtype=torch.float32, device=dev))           # tri[j', j] = 1 if j' <= j
    broken = ((1.0 - cont) * (jidx >= T2).to(torch.float32)) @ tri_incl                           # number of non-continuing tiles in [T2, j]
    run = ((jidx >= T2) & (broken == 0)).to(torch.int32).sum()
    je = T2 + run
    is_je = jidx == je
    dest1 = torch.where(straddle & is_last0, OOB, dest1)
    dest0 = torch.where(straddle & is_last0 & (~multi), OOB, dest0)
    sdest1 = torch.where(straddle & is_last0, torch.zeros_like(sdest1), sdest1)
    dest0 = torch.where(straddle & is_je, OOB, dest0)
    sdest0 = torch.where(straddle & is_je, torch.full_like(sdest0, P_MAX), sdest0)
    fix_dest = torch.where(straddle, vs * P_MAX, vs.new_full((), num_weights)).reshape(1, 1).to(torch.int32)
    tA = torch.cat([sid_T - P_MAX * vlo[None, :], sid_T - P_MAX * v1[None, :]], dim=1).to(torch.float32)
    tile_info = torch.stack([keep0, dest0, keep1, dest1, sdest0, sdest1], dim=1).to(torch.int32)
    # Phase B slots: interior (j, v) with vlo_j < v < vhi_j in (j, v) order, then un-hit vocab tiles, then padding
    m = (span - 2).clamp_min(0)
    tri = torch.triu(torch.ones(n_tiles, n_tiles, dtype=torch.float32, device=dev), diagonal=1)
    cum = (m.to(torch.float32) @ tri).to(torch.int32)                          # exclusive prefix
    n_int = cum[-1] + m[-1]
    s = torch.arange(PB, dtype=torch.int32, device=dev)
    j_s = ((s[:, None] >= cum[None, :]).to(torch.int32).sum(1) - 1).clamp_(0, n_tiles - 1)
    v_s = vlo[j_s] + 1 + (s - cum[j_s])
    is_int = s < n_int
    # zero stores: vocab tiles that no token hits AND that no interior slot covers (an interior slot of a tile whose
    # span jumps over empty vocab tiles already stores zeros there). Interior slots are disjoint across token tiles,
    # so interior + zero slots <= V/128 = PB: the slot budget can never overflow.
    u = torch.arange(V_T, dtype=key.dtype, device=dev)
    hit = (key[:, None] == u[None, :]).to(torch.float32).sum(0) > 0
    covered = ((u[:, None] > vlo[None, :]) & (u[:, None] < vhi[None, :])).to(torch.float32).sum(1) > 0
    unhit = (~hit & ~covered).to(torch.float32)
    triV = torch.triu(torch.ones(V_T, V_T, dtype=torch.float32, device=dev), diagonal=1)
    rank_u = (unhit @ triV).to(torch.int32)
    n_unhit = unhit.sum().to(torch.int32)
    sel = (unhit > 0)[None, :] & (rank_u[None, :] == (s - n_int)[:, None])
    dest_unhit = (sel.to(torch.float32) @ (torch.arange(V_T, dtype=torch.float32, device=dev) * P_MAX)).to(torch.int32)
    is_unhit = (s >= n_int) & (s < n_int + n_unhit)
    dest_B = torch.where(is_int, v_s * P_MAX, torch.where(is_unhit, dest_unhit, torch.full_like(s, num_weights)))
    j_out = torch.where(is_int, j_s, torch.zeros_like(j_s))
    v_out = torch.where(is_int, v_s, torch.full_like(v_s, -1))
    # core 0 executes slots [0, PB/2), core 1 [PB/2, PB): interleave so both cores get the same share of real slots.
    # Padding / zero slots gather valid rows (tile 0) and never match: out-of-bounds (skipped) gathers measured
    # ~140 us slower per launch than real ones (bench: v5E 827 us vs v5E_nooob 680 us).
    order = torch.cat([torch.arange(0, PB, 2, device=dev), torch.arange(1, PB, 2, device=dev)])
    dest_B = dest_B[order]
    j_out = j_out[order]
    v_out = v_out[order]
    is_int = is_int[order]
    tB = (sid_T[:, j_out.long()] - P_MAX * v_out[None, :]).to(torch.float32)
    perm_B = perm_T[:, j_out.long()]
    # DMA batching reads a (128, B) index block column-major: desired[p, b] (flat r = p*B + b) sits at [r % 128, r // 128]
    perm_snake = perm_B.view(P_MAX, PB // B, B).permute(1, 0, 2).reshape(PB // B, P_MAX * B).view(PB // B, B, P_MAX).permute(2, 0, 1).reshape(P_MAX, PB)
    return perm_T, tA, tile_info, perm_snake.contiguous(), tB, dest_B.view(PB, 1), fix_dest


_compiled_plan_rest = None


def _get_plan_rest():
    global _compiled_plan_rest
    if _compiled_plan_rest is None:
        _compiled_plan_rest = torch.compile(_plan_rest, backend="neuron", dynamic=False)
    return _compiled_plan_rest


class EmbeddingPlan:
    """Sorted-token plan for one batch of token ids, shared by every embedding table looked up with these ids."""

    def __init__(self, tensors, num_weights: int, n: int):
        self.tensors = tensors
        self.num_weights = num_weights
        self.n = n


def supported(idx: torch.Tensor, num_weights: int, embedding_dim: int) -> bool:
    if not _k._NKI_OK or idx.device.type != "neuron":
        return False
    if int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")) != 2:
        return False
    n = idx.numel()
    return (n % (2 * P_MAX) == 0 and n >= 2 * P_MAX and num_weights % P_MAX == 0
            and embedding_dim % N_TILE == 0 and embedding_dim <= 2048)


def build_plan(idx: torch.Tensor, num_weights: int, embedding_dim: int):
    """Build the plan once per step (idx (B, T) int64/int32 on the Neuron device). None if unsupported."""
    if not supported(idx, num_weights, embedding_dim):
        return None
    sid, perm = torch.sort(idx.reshape(-1).to(torch.int32))
    tensors = _get_plan_rest()(sid, perm.to(torch.int32), num_weights, PHASE_B_BATCH, num_weights // P_MAX)
    return EmbeddingPlan(tuple(tensors), num_weights, idx.numel())


class _NeuronEmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, idx, *plan_tensors):
        ctx.save_for_backward(*plan_tensors)
        ctx.num_weights = weight.shape[0]
        ctx.weight_dtype = weight.dtype
        ctx.embedding_dim = weight.shape[1]
        return F.embedding(idx, weight)

    @staticmethod
    def backward(ctx, grad_output):
        plan_tensors = ctx.saved_tensors
        D = ctx.embedding_dim
        g = grad_output.reshape(-1, D)
        if g.dtype != torch.bfloat16:
            g = g.to(torch.bfloat16)   # never happens in nanochat (bf16 activations); keeps the kernel contract explicit
        g = g.contiguous()
        grad_weight = _k._KERNEL_G2(g, *plan_tensors, ctx.num_weights)
        if grad_weight.dtype != ctx.weight_dtype:
            grad_weight = grad_weight.to(ctx.weight_dtype)
        return (grad_weight, None) + (None,) * len(plan_tensors)


def _prepared(idx: torch.Tensor, num_weights: int, embedding_dim: int, training: bool):
    """(int32 ids, plan | None) for this lookup, cached across the lookups of one forward (same ids tensor object at
    the same version counter, same table shape); a fresh batch or an in-place overwrite of the ids buffer misses."""
    global _last
    try:
        version = idx._version
    except RuntimeError:  # inference tensors have no version counter: no caching, the cast is cheap
        version = None
    key = (id(idx), version, num_weights, embedding_dim, training)
    if version is not None and _last is not None and _last[0] is idx and _last[1] == key:
        return _last[2], _last[3]
    idx32 = idx.to(torch.int32) if idx.dtype == torch.int64 else idx
    plan = build_plan(idx32, num_weights, embedding_dim) if training else None
    if version is not None:
        _last = (idx, key, idx32, plan)  # holds the ids alive until the next batch (one (B, T) int64 tensor)
    return idx32, plan


_last = None


def embedding(idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """F.embedding(idx, weight) on Neuron: int32 indices, NKI backward when training (see module docstring)."""
    training = weight.requires_grad and torch.is_grad_enabled()
    idx32, plan = _prepared(idx, weight.shape[0], weight.shape[1], training)
    if plan is None:
        return F.embedding(idx32, weight)
    assert plan.n == idx32.numel() and plan.num_weights == weight.shape[0], "plan does not match this lookup"
    return _NeuronEmbeddingFn.apply(weight, idx32, *plan.tensors)
