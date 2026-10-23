"""nanochat.trn -- the AWS Trainium (Neuron) backend package: trn2 (NeuronCore gen3) and trn3 (gen4), LNC2.

Everything Neuron-specific lives here; the upstream-facing modules (nanochat/gpt.py, optim.py,
flash_attention.py, common.py, scripts/base_train.py, scripts/chat_sft.py) only talk to the backend protocol in
nanochat/backend.py, whose reference implementation is plain torch (nanochat/optim.py, scripts/base_eval.py,
scripts/chat_eval.py and scripts/chat_sft.py select trn/optim.py, trn/core_eval.py, trn/engine.py by backend / device).

Layout
  trn/__init__.py           backend selection in ONE place: is_available(), get_backend(), plus the tiny runtime
                            shims common.py needs (synchronize, ...)
  trn/backend.py            NeuronBackend: the nanochat.backend protocol implemented with the modules below
  trn/attention.py          BTHD-native NKI flash attention (fwd+bwd) for training shapes, nkilib sliding-window
                            fallback, and the Neuron flavour of flash_attn_func / flash_attn_with_kvcache
  trn/embedding.py          token-sort plan + autograd.Function for the NKI embedding backward
  trn/compile.py            per-Block torch.compile, NKI prewarm, cache-key notes
  trn/fused_ce.py           fused lm_head + softcap + CE (one compiled graph, min-cut partitioned)
  trn/optim.py              DistMuonAdamW: fp32 master shards, the fused-launch step (6 torch.compile graphs)
  trn/core_eval.py          fixed-shape batched CORE evaluation (scripts/base_eval.py on Neuron)
  trn/engine.py             NeuronEngine: bucketed / pooled KV caches, batched decode (chat_eval, chat_sft, base_eval)
  trn/train_hooks.py        training-loop hooks for scripts/base_train.py and scripts/chat_sft.py
  trn/kernels/              pure NKI kernel code: attention_bthd, embedding

This module must stay import-light (torch only): nanochat.common imports it at import time.
"""
import os

import torch

DEVICE_TYPE = "neuron"
SUPPORTED_TARGETS = ("trn2", "trn3")  # NeuronCore gen3 / gen4 (checked when the backend is instantiated)
_target = None


def target() -> str:
    """The platform target ("trn2" = NeuronCore gen3, "trn3" = gen4), resolved ONCE and cached.

    torch_neuronx honours NEURON_PLATFORM_TARGET_OVERRIDE (the CPU kernel-gate / test path). Never call this
    inside code that dynamo traces: get_platform_target() is a graph break, and a kernel call that graph-breaks
    silently runs eagerly (+860 ms/step when this happened during the trn2 port)."""
    global _target
    if _target is None:
        # honour the override explicitly: torch_neuronx resolves from NRT and ignores it, but the CPU
        # kernel-gate / test path (no NRT) and cross-target gating both key on it.
        _target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")
        if not _target:
            from torch_neuronx.utils import get_platform_target
            _target = str(get_platform_target())
    return _target


def is_available() -> bool:
    """Is the Neuron device backend usable in this process?"""
    try:
        torch.device("neuron")
        return True
    except Exception:
        return False


def logical_cores() -> int:
    """NEURON_LOGICAL_NC_CONFIG (LNC); the runtime default on trn2/trn3 is 2 physical cores per logical core."""
    return int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))


# ---- runtime shims used by nanochat.common's device dispatch (no torch_neuronx import elsewhere) ----
def synchronize():
    import torch_neuronx
    torch_neuronx.synchronize()


def max_memory_allocated() -> int:
    """Peak allocated bytes reported by torch_neuronx's allocator statistics, or 0 when unavailable."""
    try:
        import torch_neuronx
        if hasattr(torch_neuronx, "max_memory_allocated"):
            return torch_neuronx.max_memory_allocated()
    except ImportError:
        pass
    return 0


def has_memory_stats() -> bool:
    try:
        import torch_neuronx
        return hasattr(torch_neuronx, "max_memory_allocated")
    except ImportError:
        return False


def empty_cache():
    import torch_neuronx
    torch_neuronx.empty_cache()


def require_supported_target():
    """The kernels (attention_bthd, embedding) are validated on trn2 (gen3) and trn3 (gen4);
    fail early and clearly on any other platform target instead of tracing kernels there."""
    if target() not in SUPPORTED_TARGETS:
        raise RuntimeError(
            f"nanochat.trn supports the {SUPPORTED_TARGETS} platform targets; torch_neuronx reports "
            f"{target()!r}.")


_backend = None


def get_backend():
    """The NeuronBackend singleton (nanochat.backend selects it when is_available())."""
    global _backend
    if _backend is None:
        require_supported_target()
        from nanochat.trn.backend import NeuronBackend
        _backend = NeuronBackend()
    return _backend
