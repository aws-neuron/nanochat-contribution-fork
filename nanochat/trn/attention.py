"""Neuron attention for nanochat. Three pieces:

1. BTHD-native flash attention (the training path). BOTH forward and backward consume/produce native (B,T,H,D) --
   no q/k/v/o/dy/grad relayout anywhere. fwd = bthd_flash_fwd(return_lse=True), bwd = bthd_flash_bwd
   (nanochat/trn/kernels/attention_bthd.py). Taken whenever the shapes allow AND gradients are being
   recorded: MHA (H_kv == H), sliding_window in {0, 512} (nanochat's L-layers map to 0 = full causal, S-layers to
   512 = keys i-512 <= j <= i, flash-attention's window_size=(512, 0)), Q packed in 16 tiles of 128 tokens
   (T % 2048 == 0). The no-grad forwards of the eval stages (base_eval's CORE buckets and bpb batches, whose
   (8, 2048) shape equals the training shape) keep the SDPA path so the reported eval numbers are those of the SDPA
   reference (the BTHD forward's rounding would move CORE by ~1e-3); decode uses the kv-cache path, which the kernel
   does not implement.

2. nkilib sliding-window attention for the SDPA fallback's S-layers ((B,H,T,D) layout, attention_cte /
   attention_bwd with their native `sliding_window`): avoids materializing the T x T score matrix that SDPA + an
   explicit attn_mask would produce on Neuron (SDPA with an explicit mask does not take the fused flash-attention path).

3. The Neuron flavour of nanochat.flash_attention's public API: flash_attn_func (BTHD kernel, else SDPA with
   contiguous (B,H,T,D) inputs -- strided views fall off the fused kernel path) and flash_attn_with_kvcache
   (static-shape single-token decode over the whole pre-allocated cache, slice-based prefill). Non-Neuron tensors
   are handed to the torch reference implementation unchanged.
"""
import math
from types import SimpleNamespace

import torch
import torch._dynamo
import torch.nn.functional as F

from nanochat import flash_attention as _ref

# ---------------------------------------------------------------------------------------------------------
# nkilib sliding-window attention ((B,H,T,D) layout; SDPA-fallback S-layers)
# ---------------------------------------------------------------------------------------------------------
SWA_NKI_AVAILABLE = False
try:
    from nkilib.core.attention.attention_bwd import attention_bwd
    from nkilib.core.attention.attention_cte import attention_cte
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    _swa_wrapped_fwd = wrap_nki(attention_cte)
    _swa_wrapped_bwd = wrap_nki(attention_bwd)
    SWA_NKI_AVAILABLE = True
except ImportError:
    pass


@torch._dynamo.allow_in_graph
class _NKIFlashAttnSWA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sliding_window):
        """
        q: (B, H, T, D) bf16 — PRE-SCALED by 1/sqrt(D)
        k: (B, H_kv, T, D) bf16
        v: (B, H_kv, T, D) bf16
        sliding_window: int
        Returns: output (B, H, T, D)
        """
        B, H, T, D = q.shape
        H_kv = k.shape[1]
        lnc = int(get_logical_neuron_cores())
        grid = (lnc,)

        # Forward kernel: attention_cte
        # Layout with tp_q=True: (batch, seqlen, d) where batch = B*H
        q_fwd = q.reshape(B * H, T, D)
        k_fwd = k.reshape(B * H_kv, T, D)
        v_fwd = v.reshape(B * H_kv, T, D)

        # SWA mode requires scale=1.0; caller pre-scales Q
        output, neg_max, sum_recip = _swa_wrapped_fwd[grid](
            q_fwd, k_fwd, v_fwd,
            tp_q=True,
            tp_k=True,
            scale=1.0,
            causal_mask=True,
            sliding_window=sliding_window,
            cache_softmax=True,
        )

        # output: (B*H, T, D), neg_max/sum_recip: (B*H, 128, T//128)
        output = output.reshape(B, H, T, D)

        # Compute LSE for backward: lse = -(neg_max + log(sum_recip))
        # neg_max, sum_recip are (B*H, 128, T//128)
        lse = -(neg_max + torch.log(sum_recip))
        lse = lse.to(torch.float32)
        # Reshape to (B, H, 128, T//128) for backward
        lse = lse.reshape(B, H, 128, T // 128)

        ctx.save_for_backward(q, k, v, output, lse)
        ctx.sliding_window = sliding_window
        ctx.shape_info = (B, H, H_kv, T, D)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        B, H, H_kv, T, D = ctx.shape_info

        lnc = int(get_logical_neuron_cores())
        grid = (lnc,) if (B * H_kv) % lnc == 0 else (1,)

        # Backward kernel: attention_bwd
        # Layout: (bs, nheads, d_head, seq_len) i.e. (B, H, D, T)
        q_bwd = q.transpose(-2, -1).contiguous()       # (B, H, T, D) -> (B, H, D, T)
        k_bwd = k.transpose(-2, -1).contiguous()       # (B, H_kv, T, D) -> (B, H_kv, D, T)
        v_bwd = v.transpose(-2, -1).contiguous()       # (B, H_kv, T, D) -> (B, H_kv, D, T)
        o_bwd = output.transpose(-2, -1).contiguous()  # (B, H, T, D) -> (B, H, D, T)
        dy_bwd = grad_output.transpose(-2, -1).contiguous()  # (B, H, T, D) -> (B, H, D, T)

        grad_q_nki, grad_k_nki, grad_v_nki = _swa_wrapped_bwd[grid](
            q_bwd, k_bwd, v_bwd, o_bwd, dy_bwd, lse,
            use_causal_mask=True,
            mixed_precision=True,
            softmax_scale=1.0,
            sliding_window=ctx.sliding_window,
        )

        # Transpose gradients back: (B, H, D, T) -> (B, H, T, D)
        grad_q = grad_q_nki.transpose(-2, -1)
        grad_k = grad_k_nki.transpose(-2, -1)
        grad_v = grad_v_nki.transpose(-2, -1)

        return grad_q, grad_k, grad_v, None


def nki_flash_attn_swa(q, k, v, window_size):
    """Flash attention with sliding window via NKI kernels.

    Args:
        q, k, v: (B, H, T, D) tensors in bf16
        window_size: int, number of past tokens to attend to

    Returns:
        output: (B, H, T, D) tensor
    """
    if not SWA_NKI_AVAILABLE:
        raise RuntimeError("NKI attention requires Neuron device with nkilib")

    B, H, T, D = q.shape
    scale = 1.0 / math.sqrt(D)

    # Pre-scale Q (attention_cte requires scale=1.0 for SWA mode)
    q_scaled = q * scale

    return _NKIFlashAttnSWA.apply(q_scaled, k, v, window_size)


# ---------------------------------------------------------------------------------------------------------
# BTHD-native flash attention (training fast path)
# ---------------------------------------------------------------------------------------------------------
BTHD_NKI_AVAILABLE = False
try:
    from nanochat.trn.kernels.attention_bthd import bthd_flash_fwd, bthd_flash_bwd
    from torch_neuronx.nki_hop import wrap_nki
    from torch_neuronx.utils import get_logical_neuron_cores

    _bthd_wrapped_fwd = wrap_nki(bthd_flash_fwd)
    _bthd_wrapped_bwd = wrap_nki(bthd_flash_bwd)
    BTHD_NKI_AVAILABLE = True
except ImportError:
    pass


@torch._dynamo.allow_in_graph
class _BTHDFlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale, sliding_window):
        """
        q, k, v: (B, T, H, D) bf16 — NATIVE layout, NOT pre-scaled.
                 MHA only (H_kv == H). softmax_scale passed to the kernel.
        Returns: output (B, T, H, D)
        """
        lnc = int(get_logical_neuron_cores())
        grid = (lnc,)

        # NATIVE BTHD — no reshape, no transpose, NO pre-scale buffer. Passing
        # softmax_scale to the kernel (vs q*scale) avoids a 24MiB/layer copy
        # that would also be saved for backward.
        out, lse = _bthd_wrapped_fwd[grid](
            q, k, v,
            softmax_scale=softmax_scale,
            causal=True,
            sliding_window=sliding_window,
            return_lse=True,
        )
        # Save exactly the 5 tensors backward needs (references, not copies).
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.sliding_window = sliding_window
        ctx.grid = grid
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, out, lse = ctx.saved_tensors
        grid = ctx.grid

        # NATIVE BTHD backward — q,k,v,o,dy,lse all native; dq,dk,dv native out.
        # dy must be contiguous BTHD for the kernel's .ap strides.
        dq, dk, dv = _bthd_wrapped_bwd[grid](
            q, k, v, out, grad_output.contiguous(), lse,
            softmax_scale=ctx.softmax_scale,
            causal=True,
            sliding_window=ctx.sliding_window,
        )
        return dq, dk, dv, None, None


def bthd_flash_attn_swa(q, k, v, window_size):
    """Native-BTHD flash attention with sliding window via NKI kernels.

    Args:
        q, k, v: (B, T, H, D) bf16 — NATIVE layout (no pre-transpose needed).
                 nanochat already applies its q*1.2/k*1.2 upstream; here we pass
                 the 1/sqrt(D) softmax scale to the kernel (no extra buffer).
        window_size: int, past tokens to attend to besides the current one. 0 => full causal;
                     512 => keys i-512 <= j <= i (513 keys). (caller maps -1/full -> 0.)
    Returns:
        output: (B, T, H, D)
    """
    if not BTHD_NKI_AVAILABLE:
        raise RuntimeError("BTHD NKI attention requires Neuron + nanochat.trn.kernels.attention_bthd "
                           "+ torch_neuronx.nki_hop")
    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    return _BTHDFlashAttn.apply(q, k, v, scale, window_size)


def bthd_window(window_size, T):
    """Map nanochat's (left, right) window to the kernel's sliding_window: 0 = full causal, 512 = SWA, None = unsupported.

    The kernel's sliding_window is the left window of flash-attention's window_size: keys i - left <= j <= i."""
    left = window_size[0]
    if left < 0 or left >= T:
        return 0
    if left == 512:
        return 512
    return None


# ---------------------------------------------------------------------------------------------------------
# Public API (same signatures as nanochat.flash_attention)
# ---------------------------------------------------------------------------------------------------------
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """nanochat.flash_attention._sdpa_attention plus the nkilib sliding-window kernel.

    q, k, v are (B, H, T, D). Full-context same-length and single-token queries take the reference SDPA
    calls; a same-length windowed attention on Neuron takes the NKI kernel instead of SDPA + explicit mask.
    """
    Tq, Tk, window = q.size(2), k.size(2), window_size[0]
    full_causal = (window < 0 or window >= Tq) and Tq == Tk
    if (not full_causal and Tq != 1 and SWA_NKI_AVAILABLE
            and Tq == Tk and window >= 0 and q.device.type == "neuron"):
        return nki_flash_attn_swa(q, k, v, window)
    return _ref._sdpa_attention(q, k, v, window_size, enable_gqa)


def bthd_supported(q, k, window_size):
    """Shape rule of the BTHD kernel: MHA, T % 2048 == 0, window in {full, 512} (see module docstring, item 1)."""
    return (BTHD_NKI_AVAILABLE and q.size(2) == k.size(2) and q.size(1) % 2048 == 0
            and bthd_window(window_size, q.size(1)) is not None)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Attention on (B, T, H, D): the BTHD NKI kernel for training (gradients recorded, supported shapes), else SDPA
    with contiguous (B, H, T, D) operands. Non-Neuron tensors go to the torch reference."""
    if _ref.USE_FA3 or q.device.type != "neuron":
        return _ref.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if torch.is_grad_enabled() and bthd_supported(q, k, window_size):
        return bthd_flash_attn_swa(q, k, v, bthd_window(window_size, q.size(1)))
    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D). Neuron needs contiguous inputs (strided views fall
    # off the fused kernel path).
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2).contiguous()  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """Inference attention with an in-place-updated KV cache (see nanochat.flash_attention for the contract)."""
    if _ref.USE_FA3 or q.device.type != "neuron":
        return _ref.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
                                            causal=causal, window_size=window_size)

    # Static-shape KV attention for single-token decode: writes new K/V via scatter_ (shape-invariant) and runs
    # SDPA over the full pre-allocated (B, T_max, H_kv, D) cache with a boolean position+window mask. Every decode
    # step has identical shapes -> one neuronx-cc compile for the whole loop instead of one per step, and rows at
    # different positions (NeuronEngine's bucketed, pooled caches) attend only to their own valid prefix. Prefill
    # (T_new > 1) uses the slice path below because len(prompt) is problem-specific (a one-shot cost per bucket).
    B, T_new, H, D = q.shape
    if T_new == 1 and k is not None and v is not None:
        T_max = k_cache.size(1)
        device = q.device
        # scatter_ update: (B, 1, H_kv, D) written at index cache_seqlens[b] along dim=1
        idx = cache_seqlens.unsqueeze(-1).to(torch.long)  # (B, 1)
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, T_new, k_cache.size(2), D)
        k_cache.scatter_(1, idx_exp, k)
        v_cache.scatter_(1, idx_exp, v)
        # position mask: valid = (t <= cache_seqlens); window mask if set
        t_idx = torch.arange(T_max, device=device)
        valid = t_idx.unsqueeze(0) <= cache_seqlens.unsqueeze(-1)  # (B, T_max)
        window = window_size[0]
        if window >= 0:
            in_window = t_idx.unsqueeze(0) >= (cache_seqlens.unsqueeze(-1) - window)
            valid = valid & in_window
        attn_mask = valid.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T_max)
        q_sdpa = q.transpose(1, 2)                    # (B, H, 1, D)
        k_sdpa = k_cache.transpose(1, 2)              # (B, H_kv, T_max, D)
        v_sdpa = v_cache.transpose(1, 2)
        enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
        y = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, enable_gqa=enable_gqa
        )
        return y.transpose(1, 2)

    # Default SDPA fallback: manually manage KV cache (slice-based), same as the reference but with the
    # Neuron _sdpa_attention (nkilib sliding window for a windowed same-length prefill).
    pos = cache_seqlens[0].item()  # assume uniform position across batch
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
