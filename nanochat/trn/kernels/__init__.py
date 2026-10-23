"""Pure NKI kernel code for the Trainium backend (no torch autograd, no nanochat model knowledge).

  attention_bthd.py  BTHD-native flash attention forward / backward (vendored)
  embedding.py       sorted one-hot-matmul embedding backward (scatter-add)

torch_neuronx names compiled NKI kernels by the kernel function's __qualname__ (the HLO custom call) and keys the
persistent dump_config cache by source-file basename + __qualname__ + shapes/dtypes/lnc (not by source text).
Always use a fresh TORCH_NEURONX_HLO_CACHE_DIR after editing a kernel (see nanochat/trn/compile.py).
"""
