"""Training-loop hooks of the Neuron backend, used by scripts/base_train.py and scripts/chat_sft.py through
nanochat.backend. Nothing here changes a graph or a number; these are the host-side loop adjustments the
Trainium port needs (init on CPU, prefetch after the optimizer launch, ...).
"""
from nanochat import trn
from nanochat.common import print0, get_peak_flops, COMPUTE_DTYPE


def peak_flops(device_type):
    """BF16 peak per rank (nanochat.common's table lists the peak per LNC2 rank, the training configuration)."""
    if device_type != "neuron":
        return float("inf")
    flops = get_peak_flops(trn.target()) * trn.logical_cores() / 2
    print0(f"Neuron: {trn.target()} | lnc={trn.logical_cores()} | Peak FLOPS per rank (BF16): {flops:.2e}")
    return flops


def materialize(model, device):
    """Initialize on CPU (Neuron has no seeded device RNG), then move. The transformer blocks live in COMPUTE_DTYPE:
    the Trainium DistMuonAdamW (nanochat/trn/optim.py) keeps each Muon matrix's fp32 master in the optimizer state
    of the rank that owns it, so the model's copy is the bf16 compute copy (upstream keeps fp32 params)."""
    if device.type != "neuron":
        model.to_empty(device=device)
        model.init_weights()
        return
    model.to_empty(device="cpu")
    model.init_weights()
    model.transformer.h.to(dtype=COMPUTE_DTYPE)
    model.to(device=device)


def prefetch_after_optimizer_step(device_type):
    """On Neuron the host->device copy inside next(train_loader) synchronizes the stream, so a prefetch right after
    backward() makes the host wait for the whole backward before it can even trace/launch the optimizer (at step 0
    the optimizer graphs then compile only after every Block-backward compile has finished). Fetching after the
    optimizer launches keeps the batch order identical and lets the optimizer's host work overlap the backward."""
    return device_type == "neuron"


def empty_cache(device_type):
    if device_type == "neuron":
        trn.empty_cache()
