"""Training attention on Neuron through the BTHD-native NKI flash attention kernels (nanochat/trn/kernels/attention_bthd.py).
Forward and backward read and write nanochat's native (B, T, H, D) layout, so q/k/v/o/dy need no relayout.
`bthd_attention` returns None when the kernel does not cover the call; the caller then takes the SDPA path. Covered:
gradients recorded, bf16/fp16, MHA (H_kv == H), T a multiple of 2048, head_dim <= 128 and a multiple of 16, B a
multiple of the logical core count and H even (the forward runs pairs of heads of whole batch rows per core), window
full causal or 512 (nanochat's S-layers at sequence_len 2048). No-grad
forwards (evaluation) keep SDPA so the reported eval numbers are those of the reference path; decode uses the KV cache.
"""
import math
import torch
import torch._dynamo
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores
from nanochat.trn.kernels.attention_bthd import bthd_flash_fwd, bthd_flash_bwd

_fwd = wrap_nki(bthd_flash_fwd)
_bwd = wrap_nki(bthd_flash_bwd)


@torch._dynamo.allow_in_graph
class _BTHDFlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, sliding_window, lnc):
        # the kernel applies softmax_scale itself, so no scaled copy of q is made (or saved for the backward)
        out, lse = _fwd[(lnc,)](q, k, v, softmax_scale=softmax_scale, causal=True, sliding_window=sliding_window, return_lse=True)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale, ctx.sliding_window, ctx.lnc = softmax_scale, sliding_window, lnc
        return out

    @staticmethod
    def backward(ctx, dy):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _bwd[(ctx.lnc,)](q, k, v, out, dy.contiguous(), lse,  # dy must be contiguous BTHD for the kernel's strides
                                      softmax_scale=ctx.softmax_scale, causal=True, sliding_window=ctx.sliding_window)
        return dq, dk, dv, None, None, None


def _kernel_window(window_size, T):
    """nanochat's (left, right) window -> the kernel's sliding_window: 0 = full causal, 512 = (512, 0), None = unsupported.
    (512, 0) attends keys q - 512 <= k <= q (513 keys), the same as the SDPA mask in nanochat.flash_attention."""
    left = window_size[0]
    if left < 0 or left >= T:
        return 0
    return 512 if left == 512 else None


def bthd_attention(q, k, v, window_size):
    """Causal attention on (B, T, H, D) through the NKI kernel, or None if the kernel does not cover this call."""
    B, T, H, D = q.shape
    window = _kernel_window(window_size, T)
    lnc = int(get_logical_neuron_cores())
    if not (torch.is_grad_enabled() and q.device.type == "neuron" and q.dtype in (torch.bfloat16, torch.float16)
            and q.shape == k.shape == v.shape and q.dtype == k.dtype == v.dtype and T % 2048 == 0
            and D <= 128 and D % 16 == 0 and B % lnc == 0 and H % 2 == 0 and window is not None):
        return None
    return _BTHDFlashAttn.apply(q, k, v, 1.0 / math.sqrt(D), window, lnc)
