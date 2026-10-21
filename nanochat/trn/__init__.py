"""nanochat.trn -- the AWS Trainium (Neuron) backend.

Everything Neuron-specific lives here. The upstream modules reach it through the hooks in nanochat/backend.py
(model, optimizer, training scripts) and a few device checks (nanochat/common.py; scripts/base_eval.py,
chat_eval.py and chat_sft.py pick trn/core_eval.py and trn/engine.py on a Neuron device).

  trn/__init__.py     is_available(), get_backend() and the small runtime shims nanochat/common.py needs
  trn/backend.py      NeuronBackend: the nanochat.backend hooks
  trn/attention.py    NKI flash attention for training shapes, and the Neuron flash_attn_func / flash_attn_with_kvcache
  trn/compile.py      per-Block torch.compile and the NKI kernel prewarm
  trn/fused_ce.py     fused lm_head + softcap + CE (one compiled graph)
  trn/optim.py        DistMuonAdamW: fp32 master shards and the fused 6-launch step
  trn/core_eval.py    fixed-shape batched CORE evaluation
  trn/engine.py       NeuronEngine: static-shape, bucketed KV caches and batched decode
  trn/kernels/        NKI kernel code (attention_bthd.py)

This module must stay import-light (torch only): nanochat.common imports it at import time.
"""
import os

import torch

_target = None


def is_available() -> bool:
    """Is the Neuron device backend usable in this process?"""
    try:
        torch.device("neuron")
        return True
    except Exception:
        return False


def target() -> str:
    """The platform target (e.g. "trn2"), resolved once. Do not call it inside traced code (it is a graph break)."""
    global _target
    if _target is None:
        from torch_neuronx.utils import get_platform_target
        _target = str(get_platform_target())
    return _target


def logical_cores() -> int:
    """NEURON_LOGICAL_NC_CONFIG (LNC): physical NeuronCores per rank; the runtime default is 2."""
    return int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))


def synchronize():
    import torch_neuronx
    torch_neuronx.synchronize()


def max_memory_allocated() -> int:
    """Peak allocated bytes from torch_neuronx's allocator statistics, or 0 when it does not report them."""
    import torch_neuronx
    fn = getattr(torch_neuronx, "max_memory_allocated", None)
    return fn() if fn is not None else 0


def empty_cache():
    import torch_neuronx
    torch_neuronx.empty_cache()


_backend = None


def get_backend():
    """The NeuronBackend singleton (nanochat.backend selects it when is_available())."""
    global _backend
    if _backend is None:
        from nanochat.trn.backend import NeuronBackend
        _backend = NeuronBackend()
    return _backend
