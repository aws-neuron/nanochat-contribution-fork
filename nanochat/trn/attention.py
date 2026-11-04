"""Neuron attention for nanochat: the Neuron flavour of nanochat.flash_attention's public API.

- flash_attn_func: upstream's SDPA path with contiguous (B, H, T, D) operands (strided views fall off the fused kernel
  path). Full-context layers take SDPA(is_causal=True). Sliding-window layers take SDPA with upstream's explicit
  causal band mask, with flash-attention's window semantics: window_size=(W, 0) keeps keys i - W <= j <= i (W + 1
  keys, 513 for the d24 S-layers). This relies on a TorchNeuron version whose SDPA lowering recognizes the band
  mask and runs it on fused sliding-window attention kernels; without that, SDPA with an explicit mask
  materializes the T x T scores and the S-layer attention is several times slower.
- flash_attn_with_kvcache: static-shape single-token decode over the whole pre-allocated cache, slice-based prefill.

Non-Neuron tensors are handed to the torch reference implementation unchanged.
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from nanochat import flash_attention as _ref


# ---------------------------------------------------------------------------------------------------------
# Public API (same signatures as nanochat.flash_attention)
# ---------------------------------------------------------------------------------------------------------
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Attention on (B, T, H, D): upstream's SDPA path with contiguous (B, H, T, D) operands. Non-Neuron tensors go
    to the torch reference."""
    if _ref.USE_FA3 or q.device.type != "neuron":
        return _ref.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D). Neuron needs contiguous inputs (strided views fall
    # off the fused kernel path).
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    enable_gqa = q.size(1) != k.size(1)
    y = _ref._sdpa_attention(q, k, v, window_size, enable_gqa)
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

    # Default SDPA fallback: manually manage KV cache (slice-based), same as the reference.
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
    y_sdpa = _ref._sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)
    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
