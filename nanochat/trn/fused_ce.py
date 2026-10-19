"""Fused lm_head + softcap + cross-entropy for Trainium training (pretrain and SFT); GPT.forward reaches it through
NeuronBackend.head_ce_loss for Neuron tensors (off-device tensors take upstream's logits -> softcap -> F.cross_entropy).

Equivalent to
    logits = softcap * tanh((x @ W.T)[:, :vocab_size] / softcap)   # fp32
    F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="mean")
but compiled as ONE torch.compile(backend="neuron") graph: AOTAutograd synthesizes the backward, so the eager chunked-CE
backward dispatch storm (the source of the SFT per-step drift, 5.2 -> 9.6 s over 479 steps on trn2) never runs.
Chunking over the flattened B*T dim keeps the peak transient at chunk_size*V instead of the full (B*T, V) fp32 logits.

Two choices, both measured at d24 / MBS 8 / T 2048 and fixed here:
  * the loss is written as nll = logsumexp(z) - z[target] with the cast weight transposed ONCE outside the chunk loop
    (algebraically -log_softmax(z)[target]; log_softmax is z - logsumexp(z) internally). Saved-for-backward bytes
    7216 -> 4240 MiB, CE fwd 42 -> 28 ms, bwd 66 -> 58 ms; dx / dW differ from the naive form by 1 ulp on 0.01 % /
    0.15 % of elements (validated by a loss-trajectory A/B before it became the C2 default).
  * the graph is partitioned with torch's min_cut_rematerialization_partition (`neuron_mincut_backend`: recompute cheap
    pointwise ops in the backward instead of saving their fp32 outputs): saved bytes 4240 -> 2192 MiB. Scoped to this
    compile; the Block compiles keep torch_neuronx's default partitioner.
"""
import torch


def neuron_mincut_backend(gm, example_inputs, **kwargs):
    """The torch_neuronx dynamo backend with min-cut partitioning, for one compile."""
    import torch_neuronx.neuron_dynamo_backend.backend as nb
    from torch._functorch.partitioners import min_cut_rematerialization_partition

    orig = nb.aot_autograd

    def aot_autograd_mincut(**kw):
        kw.setdefault("partition_fn", min_cut_rematerialization_partition)
        return orig(**kw)

    nb.aot_autograd = aot_autograd_mincut
    try:
        return nb.neuron_backend(gm, example_inputs, **kwargs)
    finally:
        nb.aot_autograd = orig


def head_ce_loss_math(x, W, targets, vocab_size, softcap, ignore_index=-1, chunk_size=512):
    """The loss as plain PyTorch (what the compiled `head_ce_loss` implements; usable eagerly on any device).

    x:       (B, T, D) activation dtype (bf16 on Neuron)
    W:       (V_pad, D) lm_head weight (fp32 master weight in nanochat's Linear)
    targets: (B, T) int64; positions == ignore_index contribute 0 loss / 0 grad
    Returns: scalar fp32 loss, mean over non-ignored tokens.
    """
    B, T, D = x.shape
    BT = B * T
    x_flat = x.reshape(BT, D)
    t_flat = targets.reshape(BT)
    mask = t_flat != ignore_index
    safe_t = t_flat.masked_fill(~mask, 0)
    # clamp_min(1) avoids 0/0 when a microbatch is all-ignore (SFT packed conversations can produce mask.sum() == 0).
    n_valid = mask.sum().to(torch.float32).clamp_min(1.0)

    W_t = W.to(x.dtype).t()  # one transposed view for all chunks
    total = x.new_zeros((), dtype=torch.float32)
    for start in range(0, BT, chunk_size):  # static bounds -> unrolled in-graph
        end = min(start + chunk_size, BT)
        logits = (x_flat[start:end] @ W_t)[:, :vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        idx = safe_t[start:end].unsqueeze(-1)
        nll = torch.logsumexp(logits, dim=-1) - logits.gather(1, idx).squeeze(-1)
        total = total + (nll * mask[start:end].to(torch.float32)).sum()
    return total / n_valid


# dynamic=False: pretrain / SFT use fixed (B, T) buffers, so B*T is constant across steps -> one NEFF, no recompile.
head_ce_loss = torch.compile(head_ce_loss_math, dynamic=False, fullgraph=True, backend=neuron_mincut_backend)
