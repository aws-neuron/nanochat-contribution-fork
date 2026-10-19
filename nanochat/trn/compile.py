"""Compile helpers of the Neuron backend.

  compile_model          the backend's compile_model hook: FP8 checkpoint policy, per-Block torch.compile, NKI prewarm
  prewarm_nki_kernels    cold-compile helper: compile every NKI kernel variant in parallel across ranks
  content_hash           cache-key hygiene: hash of the source files that matter for a private compile cache

Cache-key hygiene. torch_neuronx keys a compiled NKI kernel (TorchNeuronNKIKernelV3.dump_config, persistent key in
TORCH_NEURONX_HLO_CACHE_DIR) by the kernel source file's BASENAME + the function's __qualname__ + argument shapes /
dtypes / scalars / lnc + the torch_neuronx version hash -- NOT by the kernel's source text: a warm cache silently serves
the old kernel to an edited file (a renamed file only costs one recompile). Key private caches by content_hash() of
nanochat/trn/**/*.py (and nanochat/gpt.py, flash_attention.py, backend.py). Also, TorchNeuron keys eager NEFFs
by the op's arguments INCLUDING the device kwarg (see nanochat.trn.optim._zeros) and the optimizer's finish graph
takes the rank as a device tensor (nanochat/trn/optim.py) so every rank shares one NEFF.

Cold compile. Two things make a fully-cold step 0 (empty HLO/NEFF/NKI
caches) take minutes at any rank count even though the compiler service compiles asynchronously:

1. torch_neuronx's NeuronOutputCode._ensure_neff_loaded() ends with torch.neuron.synchronize() the FIRST time a
   torch.compile'd graph is executed. That waits for the graph's own asynchronous neuronx-cc compile before Python
   can trace the next graph, so the Block forwards, the CE forward/backward, the Block backwards and the optimizer
   graphs compile strictly one after another. (Not patched here; file it against torch_neuronx.)

2. Each NKI kernel is compiled to BIR (TorchNeuronNKIKernelV3.dump_config) by the first rank that traces it while
   every other rank waits on the same FileLock, and all ranks then move to the same next kernel: the kernels of a
   d24 FP8 Block (attention fwd/bwd x {sw512, sw0}, the FP8 MLP kernels) serialize into ~120 s on
   the critical path. prewarm_nki_kernels() compiles them under FakeTensorMode (no device work, no parameter or RNG
   access) with the kernel variants distributed round-robin over the ranks, so the wall time drops to the longest
   single kernel. The persisted dump_config is keyed exactly like the real trace (shapes/dtypes/scalars/lnc), so the
   trace hits.

Neither changes any graph, NEFF or arithmetic: the same HLO is compiled, only earlier / concurrently.
"""
import hashlib
import math
import time
from pathlib import Path

import torch

from nanochat.common import COMPUTE_DTYPE, print0, get_dist_info


def content_hash(paths=None, root=None) -> str:
    """sha256 (first 16 hex digits) of the given source files (default: every .py under nanochat/trn plus
    nanochat/gpt.py, flash_attention.py, backend.py). Use it to key private compile caches."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    if paths is None:
        paths = sorted((root / "nanochat" / "trn").rglob("*.py")) + [
            root / "nanochat" / n for n in ("gpt.py", "flash_attention.py", "backend.py")]
    h = hashlib.sha256()
    for f in paths:
        f = Path(f)
        h.update(str(f.relative_to(root) if f.is_relative_to(root) else f).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def compile_model(model, fp8=False, batch_size=None, seq_len=None, reference=None):
    """Per-Block torch.compile on Neuron; models off the Neuron device take the reference path (upstream: one
    torch.compile over the model).

    Runs after the FP8 conversion (the callers' order). First the FP8 checkpoint policy, then the compile, then --
    given the training shape (batch_size, seq_len) -- the NKI kernel prewarm across ranks (prewarm_nki_kernels).

    neuronx-cc chokes on the full model graph (compile time / memory), while each Block -- norm + attention + MLP,
    see Block.forward -- is a well-formed subgraph that compiles cleanly. The 24 Blocks share one code object, so
    dynamo dedupes them by structure into one compiled graph pair per (window, has value embedding) pattern (3 at d24
    SSSL); the value embedding is a graph input. Everything outside the Blocks (embeddings, residual mix, lm_head + CE)
    runs eagerly or in its own compiled graph (nanochat/trn/fused_ce.py).

    With --fp8 every Block gets the FP8 checkpoint policy (nanochat.trn.fp8.fp8_recompute_contexts; it only acts
    when the Block is checkpointed): NKI custom-autograd state is opaque to AOT's normal min-cut, and without it each
    d24 Block exports 438.5 MiB for backward instead of the BF16 baseline's 246.5 MiB.
    """
    p = next(model.parameters(), None)
    if p is None or p.device.type != "neuron":
        return reference(model, fp8=fp8, batch_size=batch_size, seq_len=seq_len)
    blocks = model.transformer.h
    if fp8:
        from nanochat.trn.fp8 import fp8_recompute_contexts
        for block in blocks:
            block.checkpoint_context_fn = fp8_recompute_contexts
    for i, block in enumerate(blocks):
        blocks[i] = torch.compile(block, backend="neuron", dynamic=False)
    patterns = {(tuple(w), str(i) in model.value_embeds) for i, w in enumerate(model.window_sizes)}
    print0(f"torch.compile per-Block enabled (backend=neuron, fp8={bool(fp8)}, {len(blocks)} Blocks, "
           f"{len(patterns)} distinct graph pattern(s))")
    if batch_size is not None and seq_len is not None:
        _, rank, _, world_size = get_dist_info()
        prewarm_nki_kernels(model, batch_size, seq_len, rank, world_size)
    return model


def _attention_windows(model, T):
    """Distinct NKI sliding-window values used by the model's layers (mirrors nanochat.trn.attention.bthd_window)."""
    from nanochat.trn.attention import bthd_window
    return sorted({sw for sw in (bthd_window(w, T) for w in model.window_sizes) if sw is not None})


def prewarm_nki_kernels(model, batch_size: int, seq_len: int, rank: int, world_size: int) -> int:
    """Compile every NKI kernel variant of this model in parallel across ranks (round-robin), under FakeTensorMode.

    Tasks: attention forward and backward per distinct window and forward+backward of each distinct FP8 Linear
    shape. Returns the number of tasks this rank ran. Failures only
    print a warning (the trace compiles as before).
    """
    p = next(model.parameters(), None)
    if p is None or p.device.type != "neuron":
        return 0
    t_start = time.time()
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
    lnc = int(get_logical_neuron_cores())
    device = next(model.parameters()).device

    tasks = []
    if bthd.BTHD_NKI_AVAILABLE and cfg.n_kv_head == cfg.n_head and T % 2048 == 0:
        for sw in _attention_windows(model, T):
            tasks.append(("attn_fwd", sw))
            tasks.append(("attn_bwd", sw))
    fp8_by_shape = {}
    for m in model.modules():
        if getattr(m, "_nanochat_fp8", False) and m.training:
            fp8_by_shape.setdefault((m.in_features, m.out_features), m)
    for key in sorted(fp8_by_shape):
        tasks.append(("fp8_linear", fp8_by_shape[key]))
    if not tasks:
        return 0
    mine = tasks[rank::world_size] if world_size > 0 else tasks
    done = 0
    with FakeTensorMode():
        for kind, arg in mine:
            try:
                if kind == "attn_fwd":
                    q, k, v = (torch.empty(B, T, H, D, dtype=COMPUTE_DTYPE, device=device) for _ in range(3))
                    bthd._bthd_wrapped_fwd[(lnc,)](
                        q, k, v, softmax_scale=1.0 / math.sqrt(D), causal=True, sliding_window=arg, return_lse=True)
                elif kind == "attn_bwd":
                    q, k, v, o, dy = (torch.empty(B, T, H, D, dtype=COMPUTE_DTYPE, device=device) for _ in range(5))
                    lse = torch.empty(B, H, 128, T // 128, dtype=torch.float32, device=device)
                    bthd._bthd_wrapped_bwd[(lnc,)](
                        q, k, v, o, dy, lse, softmax_scale=1.0 / math.sqrt(D), causal=True, sliding_window=arg)
                elif kind == "fp8_linear":
                    from nanochat.trn import fp8 as nki_fp8
                    m = arg
                    rows = B * T
                    x = torch.empty(rows, m.in_features, dtype=COMPUTE_DTYPE, device=device, requires_grad=True)
                    w = torch.empty(m.out_features, m.in_features, dtype=m.weight.dtype, device=device,
                                    requires_grad=True)
                    split = nki_fp8.split_save_bits(m.out_features, m.in_features)
                    y = nki_fp8._NeuronFloat8LinearFn.apply(
                        x, w, m._forward_nstep, m._dgrad_nstep, nki_fp8.choose_kblock(rows), split)
                    torch.autograd.grad(y, (x, w), torch.empty_like(y))
                done += 1
            except Exception as e:
                print(f"[rank {rank}] NKI prewarm: {kind} failed ({e!r}); the trace will compile it")
    print0(f"NKI prewarm: {len(tasks)} kernel tasks over {world_size} ranks, rank 0 ran {len(mine)} in "
           f"{time.time() - t_start:.1f} s")
    return done
