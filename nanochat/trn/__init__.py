"""nanochat.trn -- the AWS Trainium (Neuron) backend: everything Neuron-specific lives here.

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
