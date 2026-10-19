"""Minimal FP8 training for nanochat with one public, backend-selected API.

Both CUDA and Trainium implement the same tensorwise dynamic-scaling recipe:
E4M3 input/weight in forward, E5M2 grad-output with E4M3 weight/activation in
backward, saturation, scaled matmul, and reduced-precision output. CUDA uses
``torch._scaled_mm``. The accelerator backend (nanochat.backend) may supply its own
Float8Linear class -- Trainium's is ``nanochat.trn.fp8.NeuronFloat8Linear`` -- selected
before compilation; no NKI dependency is imported on CUDA/CPU hosts.

This keeps a compact public facade over backend-specific implementations. We
only need the "tensorwise" recipe (one scalar scale per tensor), not the full
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor
subclass dispatch tables, etc.)

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Run the backend's native FP8 Tensor Engine matmul
  4. Dequantize with the inverse operand scales

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

CUDA uses finite-only E4M3FN with maximum 448. Trainium uses native E4M3
with maximum normal value 240; the format roles and scaling algorithm match,
but exact cross-backend codepoints are therefore not expected.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().

How this differs from torchao's approach
========================================
torchao uses a "tensor subclass" architecture: Float8TrainingTensor is a subclass
of torch.Tensor that bundles FP8 data + scale + metadata. It implements
__torch_dispatch__ with a dispatch table that intercepts every aten op (mm, t,
reshape, clone, ...) and handles it in FP8-aware fashion. When you call
  output = input @ weight.T
the @ operator dispatches to aten.mm, which gets intercepted and routed to
torch._scaled_mm behind the scenes. This is ~2000 lines of code because you need
a handler for every tensor operation that might touch an FP8 tensor.

We take a simpler approach: a single autograd.Function (_Float8Matmul) that takes
full-precision inputs, quantizes to FP8 internally, calls _scaled_mm, and returns
full-precision outputs. Marked @allow_in_graph so torch.compile treats it as one
opaque node rather than trying to trace inside.

The trade-off is in how torch.compile sees the two approaches:
  - torchao: compile decomposes the tensor subclass (via __tensor_flatten__) and
    sees every individual op (amax, scale, cast, _scaled_mm) as separate graph
    nodes. Inductor can fuse these with surrounding operations (e.g. fuse the
    amax computation with the preceding layer's activation function).
  - ours: compile sees a single opaque call. It can optimize everything around
    the FP8 linear (attention, norms, etc.) but cannot fuse across the boundary.

Both call the exact same cuBLAS _scaled_mm kernel — the GPU matmul is identical.
The difference is only in the "glue" ops (amax, scale, cast) which are tiny
compared to the matmul. In practice this means our version is slightly faster
(less compilation overhead, no tensor subclass dispatch cost) but can produce
subtly different floating-point rounding paths under torch.compile, since Inductor
generates a different graph. Numerics are bitwise identical in eager mode.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn

from nanochat.backend import backend
from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    # Compute the max absolute value across the entire tensor
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast — without it, compile/eager can diverge)
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # convert FP8 values back to the original range during the matmul)
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    return x.t().contiguous().t()


# allow_in_graph tells torch.compile to treat this as an opaque operation —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# for how this differs from torchao's tensor subclass approach.
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer.

    The forward quantizes input and weight to FP8 and saves
    the quantized tensors + scales for backward.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to e4m3 (higher precision format)
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            # use_fast_accum=True accumulates the dot products in lower precision.
            # Slightly less accurate but measurably faster. Standard practice for
            # the forward pass; we use False in backward for more precise gradients.
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === GEMM 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg needs row-major,
        # so we must call .contiguous() to physically rearrange the memory.
        go_T = go_fp8.t().contiguous()  # [N, B] row-major
        in_col = _to_col_major(in_fp8)    # [B, K] column-major
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


class CudaFloat8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute.

    Weights and biases remain in their original precision (e.g. fp32/bf16).
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.
    """

    _nanochat_fp8 = True
    fp8_backend = "cuda"

    @staticmethod
    def is_supported(module):
        return (
            module.in_features % 16 == 0
            and module.out_features % 16 == 0
            and min(module.in_features, module.out_features) >= 128
        )

    def forward(self, input):
        # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # reduced precision input, and we no longer rely on autocast to do this.
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """Create Float8Linear from nn.Linear, sharing the same weight and bias.

        Uses meta device to avoid allocating a temporary weight tensor — we
        create the module shell on meta (shapes/dtypes only, no memory), then
        point .weight and .bias to the original module's parameters.
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.train(mod.training)
        return new_mod


# The backend's Float8Linear (Trainium: NeuronFloat8Linear, imported here -- before torch.compile -- so its
# custom-op registration is outside the traced region) or the _scaled_mm implementation above.
Float8Linear = backend.Float8Linear if backend.Float8Linear is not None else CudaFloat8Linear
# Device types with an FP8 Linear implementation (nanochat.backend).
DEVICE_TYPES = backend.fp8_device_types


class Float8LinearConfig:
    """Minimal config matching torchao's API. Only tensorwise recipe is supported."""

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: Root module to convert.
        config: Float8LinearConfig (accepted for API compat, only tensorwise supported).
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
            are considered. The backend then applies its own scope (nanochat.backend.fp8_module_scope:
            every Linear upstream, the MLP Linears on Trainium) and its hard shape constraints.
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not getattr(child, "_nanochat_fp8", False):
                selected = module_filter_fn is None or module_filter_fn(child, fqn)
                if selected and backend.fp8_module_scope(fqn) and Float8Linear.is_supported(child):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)


def convert_model_to_fp8(model, device_type, recipe="tensorwise"):
    """--fp8 of the training scripts: convert the model's Linears in place (see convert_to_float8_training) and
    report; returns False (and warns) where the device has no FP8 Linear. Shared by base_train and chat_sft so an
    SFT run converts exactly the pretrain's Linears (same Block graphs, compile-cache hits)."""
    from nanochat.common import print0
    if device_type not in DEVICE_TYPES:
        print0(f"Warning: FP8 training requires one of {DEVICE_TYPES}, ignoring --fp8 flag")
        return False

    # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
    def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if min(mod.in_features, mod.out_features) < 128:
            return False
        return True

    fp8_config = Float8LinearConfig.from_recipe_name(recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
    num_fp8 = sum(1 for m in model.modules() if getattr(m, "_nanochat_fp8", False))
    num_skipped = num_linear - num_fp8
    if num_fp8 == 0:
        raise RuntimeError(f"FP8 was requested, but the {device_type} backend supports none of the model's Linears")
    print0(f"✓ FP8 training enabled ({recipe} scaling, {Float8Linear.fp8_backend}) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped}")
    return True


@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    from nanochat.gpt import Linear

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if getattr(module, "_nanochat_fp8", False):
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        linear.train(fp8_module.training)
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)
