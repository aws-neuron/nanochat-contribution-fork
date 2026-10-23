"""Accelerator backend hook for nanochat.

The model (gpt.py), the optimizer selection (optim.py) and the training scripts call the few hooks below.
`TorchBackend` is the reference implementation: upstream nanochat's behaviour (FA3/SDPA attention, logits ->
F.cross_entropy, one torch.compile over the model). A device package can override the hooks it needs; AWS
Trainium's lives in nanochat/trn (nanochat.trn.backend.NeuronBackend) and is selected here, once, when a Neuron
device is available.

Every hook is device-aware: a backend must behave like the reference for tensors that are not on its device, so a
CPU run inside a Trainium venv still takes the upstream code paths.

Hooks (T = torch.Tensor)
  name: str                         "torch" | "neuron" (nanochat/optim.py selects the distributed optimizer by it)
  flash_attn: namespace             flash_attn_func(q, k, v, causal, window_size) / flash_attn_with_kvcache(...)
  embedding(idx: T, weight: T) -> T                 == F.embedding(idx, weight)
  head_ce_loss(x: T, weight: T, targets: T, vocab_size: int, softcap: float) -> T | None
                                    fused lm_head + softcap + mean CE for training, or None (gpt.py computes it)
  peak_flops(device_type: str) -> float
  materialize(model, device) -> None                (to_empty + init_weights)
  compile_model(model, batch_size=None, seq_len=None) -> model
                                    upstream: torch.compile(model, dynamic=False); a backend may also pre-compile
                                    kernels for the (batch_size, seq_len) training shape
  prefetch_after_optimizer_step(device_type: str) -> bool
  empty_cache(device_type: str) -> None
"""
import torch
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn as _torch_flash_attn


class TorchBackend:
    name = "torch"
    flash_attn = _torch_flash_attn

    def embedding(self, idx, weight):
        return F.embedding(idx, weight)

    def head_ce_loss(self, x, weight, targets, vocab_size, softcap):
        return None  # upstream: logits -> softcap -> F.cross_entropy in GPT.forward

    def peak_flops(self, device_type):
        return float("inf")  # MFU not meaningful for CPU/MPS

    def materialize(self, model, device):
        model.to_empty(device=device)  # all tensors get storage on the target device, uninitialized
        model.init_weights()

    def compile_model(self, model, batch_size=None, seq_len=None):
        return torch.compile(model, dynamic=False)  # the inputs never change shape so dynamic=False is safe

    def prefetch_after_optimizer_step(self, device_type):
        return False

    def empty_cache(self, device_type):
        pass


def _select():
    from nanochat import trn
    if trn.is_available():
        return trn.get_backend()
    return TorchBackend()


backend = _select()
