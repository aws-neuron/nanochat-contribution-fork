"""NeuronBackend: nanochat's backend hooks (nanochat/backend.py) on AWS Trainium.

Each hook is device-aware and falls back to the torch reference for tensors / device types that are not Neuron,
so a CPU run inside a Trainium venv behaves like upstream nanochat.
"""
from nanochat import trn
from nanochat.backend import TorchBackend
from nanochat.common import print0, get_peak_flops, COMPUTE_DTYPE
from nanochat.trn import attention as _attention
from nanochat.trn import embedding as _embedding


class NeuronBackend(TorchBackend):
    name = "neuron"
    flash_attn = _attention.flash_attn

    def embedding(self, idx, weight):
        # F.embedding with int32 indices (the faster index dtype on Neuron) and, in training, a sort-based
        # backward (no scatter) with one host sort per forward shared by wte and the value-embedding tables
        # (nanochat/trn/embedding.py). Non-Neuron tensors: the reference.
        if idx.device.type != "neuron":
            return super().embedding(idx, weight)
        return _embedding.embedding(idx, weight)

    def head_ce_loss(self, x, weight, targets, vocab_size, softcap):
        # One compiled lm_head + softcap + CE graph, chunked over B*T (nanochat/trn/fused_ce.py). Off-device tensors
        # take the upstream path.
        if x.device.type != "neuron":
            return None
        from nanochat.trn.fused_ce import head_ce_loss
        return head_ce_loss(x, weight, targets, vocab_size=vocab_size, softcap=softcap, ignore_index=-1, chunk_size=1024)

    def peak_flops(self, device_type):
        # BF16 peak per rank (nanochat.common's table lists it per LNC2 rank, the training configuration)
        if device_type != "neuron":
            return float("inf")
        flops = get_peak_flops(trn.target()) * trn.logical_cores() / 2
        print0(f"Neuron: {trn.target()} | lnc={trn.logical_cores()} | Peak FLOPS per rank (BF16): {flops:.2e}")
        return flops

    def materialize(self, model, device):
        # Initialize on CPU (Neuron has no seeded device RNG), then move. The transformer blocks live in
        # COMPUTE_DTYPE: the Trainium DistMuonAdamW (nanochat/trn/optim.py) keeps each Muon matrix's fp32 master in
        # the optimizer state of the rank that owns it, so the model's copy is the bf16 compute copy.
        if device.type != "neuron":
            return super().materialize(model, device)
        model.to_empty(device="cpu")
        model.init_weights()
        model.transformer.h.to(dtype=COMPUTE_DTYPE)
        model.to(device=device)

    def compile_model(self, model, batch_size=None, seq_len=None):
        from nanochat.trn import compile as _compile
        return _compile.compile_model(model, batch_size=batch_size, seq_len=seq_len, reference=super().compile_model)

    def prefetch_after_optimizer_step(self, device_type):
        # The host->device copy in next(train_loader) synchronizes the stream, so a prefetch right after backward()
        # makes the host wait for the whole backward before it can launch the optimizer. Fetching after the
        # optimizer launches keeps the batch order and lets the optimizer's host work overlap the backward.
        return device_type == "neuron"

    def empty_cache(self, device_type):
        if device_type == "neuron":
            trn.empty_cache()
