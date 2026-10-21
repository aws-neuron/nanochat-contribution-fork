"""Compile helpers of the Neuron backend.

  compile_model          the backend's compile_model hook: per-Block torch.compile
"""
import torch

from nanochat.common import print0


def compile_model(model, batch_size=None, seq_len=None, reference=None):
    """Per-Block torch.compile on Neuron; models off the Neuron device take the reference path (upstream: one
    torch.compile over the model).

    One graph per Block instead of one for the whole model: each Block -- norm + attention + MLP, see Block.forward --
    is a compact subgraph that compiles quickly and with little host memory. The 24 Blocks share one code object, so
    dynamo dedupes them by structure into one compiled graph pair per (window, has value embedding) pattern (3 at d24
    SSSL); the value embedding is a graph input. Everything outside the Blocks (embeddings, residual mix, lm_head + CE)
    runs eagerly or in its own compiled graph (nanochat/trn/fused_ce.py).
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
    return model
