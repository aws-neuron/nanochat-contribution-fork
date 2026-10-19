"""Pure NKI kernel code for the Trainium backend (no torch autograd, no nanochat model knowledge).

  attention_bthd.py  BTHD-native flash attention forward / backward (vendored)
  embedding.py       sorted one-hot-matmul embedding backward (scatter-add)
  fp8_common.py      constants and helpers shared by the FP8 families
  fp8_quantize.py    E4M3 activation save, E5M2 gradient scale state
  fp8_gemm_fwd.py    tensorwise GEMM, fused forward+save (+ the unused streamed GEMM, kept verbatim)
  fp8_dgrad.py       scaled dgrad, fused grad-scale-state + dgrad
  fp8_wgrad.py       saved-activation wgrad (plain, persistent, grouped)

torch_neuronx names compiled NKI kernels by the kernel function's __qualname__ (the HLO custom call) and keys the
persistent dump_config cache by source-file basename + __qualname__ + shapes/dtypes/lnc (not by source text), so
the split from the former nanochat/nki_fp8.py changed no emitted code -- only the cache keys, which is one
recompile. Always use a fresh TORCH_NEURONX_HLO_CACHE_DIR after editing a kernel (see nanochat/trn/compile.py).
"""
