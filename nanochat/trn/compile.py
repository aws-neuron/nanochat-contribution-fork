"""Compile helpers of the Neuron backend.

  compile_model          the backend's compile_model hook: per-Block torch.compile, then the NKI prewarm
  prewarm_nki_kernels    compile the NKI attention kernels in parallel across ranks before the first step

Cold compile. Each NKI kernel is compiled by the first rank that traces it while every other rank waits on the same
file lock, and all ranks then move to the same next kernel, so a d24 Block's attention kernels (forward and backward
for each window) serialize on the critical path of step 0. prewarm_nki_kernels() compiles them under FakeTensorMode
(no device work, no parameter or RNG access) with the kernel variants distributed round-robin over the ranks, so the
wall time drops to the longest single kernel. The cached kernel is keyed exactly like the real trace (shapes, dtypes,
scalars, lnc), so the trace hits. Nothing changes in any graph or number: the same kernels compile, only earlier and
concurrently.

The kernel cache is keyed by the kernel file name and function name, not by its source text: use a fresh
TORCH_NEURONX_HLO_CACHE_DIR after editing a kernel.
"""
import math
import time

import torch

from nanochat.common import COMPUTE_DTYPE, print0, get_dist_info


def compile_model(model, batch_size=None, seq_len=None, reference=None):
    """Per-Block torch.compile on Neuron; models off the Neuron device take the reference path (upstream: one
    torch.compile over the model).

    One graph per Block instead of one for the whole model: each Block -- norm + attention + MLP, see Block.forward --
    is a compact subgraph that compiles quickly and with little host memory. The 24 Blocks share one code object, so
    dynamo dedupes them by structure into one compiled graph pair per (window, has value embedding) pattern (3 at d24
    SSSL); the value embedding is a graph input. Everything outside the Blocks (embeddings, residual mix, lm_head + CE)
    runs eagerly or in its own compiled graph (nanochat/trn/fused_ce.py). Given the training shape (batch_size,
    seq_len), the NKI kernels are then prewarmed across ranks.
    """
    p = next(model.parameters(), None)
    if p is None or p.device.type != "neuron":
        return reference(model, batch_size=batch_size, seq_len=seq_len)
    blocks = model.transformer.h
    for i, block in enumerate(blocks):
        blocks[i] = torch.compile(block, backend="neuron", dynamic=False)
    patterns = {(tuple(w), str(i) in model.value_embeds) for i, w in enumerate(model.window_sizes)}
    print0(f"torch.compile per-Block enabled (backend=neuron, {len(blocks)} Blocks, "
           f"{len(patterns)} distinct graph pattern(s))")
    if batch_size is not None and seq_len is not None:
        _, rank, _, world_size = get_dist_info()
        prewarm_nki_kernels(model, batch_size, seq_len, rank, world_size)
    return model


def prewarm_nki_kernels(model, batch_size: int, seq_len: int, rank: int, world_size: int) -> int:
    """Compile the NKI attention forward and backward for each distinct window in parallel across ranks
    (round-robin), under FakeTensorMode. Returns the number of tasks this rank ran. A failure only prints a warning
    (the trace then compiles the kernel as before)."""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch_neuronx.utils import get_logical_neuron_cores
        from nanochat.trn import attention as bthd
    except Exception as e:
        print0(f"NKI prewarm: unavailable ({e!r}); skipped")
        return 0

    cfg = model.config
    B, T = batch_size, seq_len
    H = cfg.n_head
    D = cfg.n_embd // cfg.n_head
    if not (bthd.BTHD_NKI_AVAILABLE and cfg.n_kv_head == cfg.n_head and T % 2048 == 0):
        return 0
    windows = sorted({sw for sw in (bthd.bthd_window(w, T) for w in model.window_sizes) if sw is not None})
    tasks = [(kind, sw) for sw in windows for kind in ("attn_fwd", "attn_bwd")]
    t_start = time.time()
    lnc = int(get_logical_neuron_cores())
    device = next(model.parameters()).device
    mine = tasks[rank::world_size] if world_size > 0 else tasks
    done = 0
    with FakeTensorMode():
        for kind, sw in mine:
            try:
                if kind == "attn_fwd":
                    q, k, v = (torch.empty(B, T, H, D, dtype=COMPUTE_DTYPE, device=device) for _ in range(3))
                    bthd._bthd_wrapped_fwd[(lnc,)](
                        q, k, v, softmax_scale=1.0 / math.sqrt(D), causal=True, sliding_window=sw, return_lse=True)
                else:
                    q, k, v, o, dy = (torch.empty(B, T, H, D, dtype=COMPUTE_DTYPE, device=device) for _ in range(5))
                    lse = torch.empty(B, H, 128, T // 128, dtype=torch.float32, device=device)
                    bthd._bthd_wrapped_bwd[(lnc,)](
                        q, k, v, o, dy, lse, softmax_scale=1.0 / math.sqrt(D), causal=True, sliding_window=sw)
                done += 1
            except Exception as e:
                print(f"[rank {rank}] NKI prewarm: {kind} failed ({e!r}); the trace will compile it")
    print0(f"NKI prewarm: {len(tasks)} kernel tasks over {world_size} ranks, rank 0 ran {len(mine)} in "
           f"{time.time() - t_start:.1f} s")
    return done
