# NKI kernels for nanochat on AWS Trainium

This directory adds two NKI kernels to the Neuron training path. Both are ported from the `optimized_a` branch.

| Kernel | File | Replaces |
|---|---|---|
| Flash attention forward and backward (`bthd_flash_fwd`, `bthd_flash_bwd`) | `kernels/attention_bthd.py` | SDPA in `flash_attn_func` during training |
| Embedding backward (`embedding_backward_sorted_v5`) | `kernels/embedding.py` | The scatter-add backward of `F.embedding` |

The PyTorch code around the kernels is in `attention.py` (autograd function, shape checks) and `embedding.py` (token-sort plan, autograd function). `nanochat/flash_attention.py` and `nanochat/gpt.py` call them only when the tensors are on the `neuron` device. CUDA, CPU and MPS are unchanged.

## Attention kernel

- **Layout:** reads q, k, v and writes the output and all gradients in nanochat's native (B, T, H, D) layout. The SDPA path transposes to (B, H, T, D) and back.
- **Algorithm:** flash attention with a 128 × 2048 score section kept in SBUF, so T = 2048 needs one softmax with no online merge. The forward also returns the log-sum-exp, and the backward uses it.
- **Masks:** causal, with an optional 512-token sliding window. The window matches `window_size=(512, 0)` and nanochat's SDPA mask: query q attends keys q − 512 ≤ k ≤ q, which is 513 keys.
- **When it is used:** gradients are being recorded, the dtype is bf16 or fp16, k and v have as many heads as q, T is a multiple of 2048, head_dim ≤ 128 and is a multiple of 16, B is a multiple of the logical core count, H is even, and the window is full or 512. Any other call falls back to SDPA. With nanochat's default `SSSL` pattern at `--max-seq-len=2048`, both the S-layers (window 512) and the L-layers (full) use the kernel.
- **Not used for:** evaluation forwards (no gradients) and KV-cache decoding.

## Embedding backward kernel

- **What it replaces:** the default backward of each embedding lookup, a scatter-add of (B·T, D) gradient rows into a (V, D) table.
- **Plan (PyTorch, once per forward):** sort the token ids. The token-embedding table and every value-embedding table use the same ids, so they share one plan.
- **Kernel:** for each tile of 128 sorted tokens, it builds a 128 × 128 one-hot matrix and multiplies it with the gradient tile on the Tensor engine, accumulating in fp32. Vocabulary tiles that border neighbouring token tiles are summed across tiles. Each output tile is rounded to bf16 once, as torch does. The two physical cores of an LNC2 logical core each take half of the sorted tiles.
- **Forward:** a plain `F.embedding` with int32 ids. int64 ids cost a host round trip on each call on Neuron.
- **When it is used:** training, LNC2, B·T a multiple of 256, V a multiple of 128, and D a multiple of 512 with D ≤ 2048. At d8 (D = 512) it is used. At d12 (D = 768) it falls back to the default backward.

## Changes from optimized_a

The `optimized_a` attention kernel did not work with the public `torch-neuronx:2.14.0-neuronx-py314-sdk2.33.1` image (neuronx-cc 2.28.7631, nki 0.7.0):

1. **Compile error:** neuronx-cc rejected the forward with `NCC_ISCH715` (accumulation group interleaving). The fix turns on non-SSA legalization for the forward, a setting the backward already used.
2. **Wrong sliding-window outputs:** with 3 PE transposes per sliding-window group, query rows from position 256 on were off by up to 0.79. With 1 or 2 transposes the outputs are correct, and the kernel now uses 2 (`_V3_PE_SWA = 2`), the same as the full-causal setting.
3. **Off-by-one window:** `optimized_a`'s window attended 512 keys. The kernel now uses `optimized_b`'s mask, which attends 513 keys, matching nanochat's SDPA path.

The embedding kernel is unchanged from `optimized_a`.

## Correctness

All checks ran on one trn2.3xlarge (LNC2, 4 logical NeuronCores). The reference is PyTorch on CPU in fp32, computed from the same bf16 inputs. "Relative error" below is the max absolute error divided by the max absolute reference value.

| Check | Result |
|---|---|
| Attention, full causal: B 2–4, H 4–12, T 2048, eager and `torch.compile` | Output, dq, dk and dv relative error 2e-3 to 6e-3 |
| Attention, window 512: T 2048 and 4096 | Relative error 2e-3 to 5e-3 |
| Attention backward alone, given the reference output and LSE | Relative error ≤ 4.5e-3 |
| Neuron SDPA vs. the same fp32 reference, for scale | Max absolute error 1.3e-2 |
| Embedding backward, uniform and skewed token ids, D 512 and 1024 | Bitwise equal to the fp32-sum / bf16-round reference |

## Performance

`base_train` on one NeuronCore with `--max-seq-len=2048 --device-batch-size=2` (4,096 tokens per step) and bf16. Two trees ran side by side with fresh compile caches: this branch (kernels) and the `riv` tip (SDPA and the default embedding backward). The median step time excludes the first 10 steps.

| Run | Kernels used | Median step, `riv` | Median step, kernels | Change |
|---|---|---|---|---|
| d8, 60 steps | Attention and embedding | 151.5 ms | 121.4 ms | −20% |
| d8, 400 steps | Attention and embedding | 174.6 ms | 130.8 ms | −25% |
| d12, 200 steps | Attention only | 305.7 ms | 254.6 ms | −17% |

Training quality is unchanged within run-to-run noise:

| Run | Final val bpb, `riv` | Final val bpb, kernels |
|---|---|---|
| d8, 400 steps | 1.8828 | 1.8757 |
| d12, 200 steps | 1.9760 | 1.9769 |

The losses of the two trees match to 4 decimal places for the first 19–24 steps. After that, bf16 rounding differences move them apart by at most 0.27 at d8 and 0.05 at d12, and they converge again. On the d12 checkpoints, `base_eval` with 16 examples per CORE task gives CORE 0.0056 on both trees, with val bpb 1.9769 (kernels) and 1.9760 (`riv`).

Other runs:

- **2-process `torchrun`:** d12 on two NeuronCores completed 40 steps at 246–249 ms/step. There was no SDPA run to compare against.
- **Cold first step (compile included):** 103 s with kernels vs. 135 s on `riv` at d12. At d8 the two trees were close, and which one was faster depended on the run.

## Limits

- **Configuration:** the numbers come from small models (d8, d12), batch 2 and one NeuronCore. I did not measure d24 or a multi-node run.
- **Comparison with `optimized_a`:** fixes 1 and 2 give up some of `optimized_a`'s kernel tuning. I didn't benchmark against that branch, and I did not take per-kernel timings on this image.
- **Shapes that fall back:** batch size 1 per core, odd head counts (for example d10), and sequence lengths that are not a multiple of 2048 use SDPA. Embedding widths that are not a multiple of 512 use the default embedding backward.
- **Compile cache:** torch-neuronx keys its NKI compile cache by file name and function name, not by source text. After editing a kernel, use a fresh `TORCH_NEURONX_HLO_CACHE_DIR` or set `NKI_ENABLE_TRACE_CACHE=0`.
