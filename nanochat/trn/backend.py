"""NeuronBackend: the nanochat.backend protocol on AWS Trainium.

Each hook is device-aware and falls back to the torch reference for tensors / device types that are not
Neuron, so a CPU run inside a Trainium venv behaves like upstream nanochat.
"""
from nanochat.backend import TorchBackend
from nanochat.trn import attention as _attention
from nanochat.trn import embedding as _embedding


class NeuronBackend(TorchBackend):
    name = "neuron"
    flash_attn = _attention.flash_attn

    # ---- model hooks (nanochat/gpt.py)
    def embedding(self, idx, weight):
        # F.embedding with int32 indices (TorchNeuron lowers int64 indices through a CPU round-trip on every
        # call) and, in training, the NKI scatter-add backward with one token-sort plan per forward shared by wte
        # and the value-embedding tables (nanochat/trn/embedding.py). Non-Neuron tensors: the reference.
        if idx.device.type != "neuron":
            return super().embedding(idx, weight)
        return _embedding.embedding(idx, weight)

    def head_ce_loss(self, x, weight, targets, vocab_size, softcap):
        # One compiled lm_head + softcap + CE graph (nanochat/trn/fused_ce.py): AOTAutograd synthesizes the backward,
        # so the eager chunked-CE backward dispatch storm never runs. Off-device tensors take the upstream path.
        if x.device.type != "neuron":
            return None
        from nanochat.trn.fused_ce import head_ce_loss
        return head_ce_loss(x, weight, targets, vocab_size=vocab_size, softcap=softcap, ignore_index=-1, chunk_size=512)

    # ---- training-script hooks (nanochat/trn/train_hooks.py, nanochat/trn/compile.py)
    def peak_flops(self, device_type):
        from nanochat.trn import train_hooks
        return train_hooks.peak_flops(device_type)

    def materialize(self, model, device):
        from nanochat.trn import train_hooks
        train_hooks.materialize(model, device)

    def compile_model(self, model, batch_size=None, seq_len=None):
        from nanochat.trn import compile as _compile
        return _compile.compile_model(model, batch_size=batch_size, seq_len=seq_len, reference=super().compile_model)

    def prefetch_after_optimizer_step(self, device_type):
        from nanochat.trn import train_hooks
        return train_hooks.prefetch_after_optimizer_step(device_type)

    def empty_cache(self, device_type):
        from nanochat.trn import train_hooks
        train_hooks.empty_cache(device_type)
