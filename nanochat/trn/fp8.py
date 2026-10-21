"""Trainium tensorwise FP8 Linear (wrapper layer).

Owns the ``nki_op`` registrations, the grid-bound kernel aliases, the shape rules, the autograd Function and
``NeuronFloat8Linear`` and the FP8 activation-checkpoint policy (the recipe of nanochat.fp8 on the MLP Linears: E4M3 input/weight forward, E5M2 grad-output
backward, strict-FP8 wgrad from the E4M3 bytes the forward saved). The NKI kernels themselves live in
``nanochat.trn.kernels.fp8_*`` (one module per kernel family); the public FP8 API and conversion policy remain in
``nanochat.fp8``.
"""
import os

import torch
from torch_neuronx import nki_op

from nanochat import trn

from nanochat.common import COMPUTE_DTYPE
from nanochat.trn.kernels.fp8_common import (
    DOUBLE_ROW_K,
    N_TILE,
    P_MAX,
    RESIDENT_B_ELEMENTS,
)
from nanochat.trn.kernels.fp8_dgrad import (
    FUSED_DGRAD_NSPLIT_MIN_N,
    tensorwise_fp8_grad_scale_state_dgrad_v1,
    tensorwise_fp8_linear_dgrad_scaled_v1,
)
from nanochat.trn.kernels.fp8_gemm_fwd import (
    tensorwise_fp8_forward_nosave_v1,
    tensorwise_fp8_forward_save_v1,
    tensorwise_fp8_gemm,
)
from nanochat.trn.kernels.fp8_quantize import (
    tensorwise_e4m3_save,
    tensorwise_fp8_grad_scale_state_v1,
)
from nanochat.trn.kernels.fp8_wgrad import (
    PERSISTENT_WGRAD_SBUF_BUDGET,
    tensorwise_fp8_wgrad_saved,
    tensorwise_fp8_wgrad_saved_grouped_v1,
    tensorwise_fp8_wgrad_saved_persistent_v3,
    tensorwise_fp8_wgrad_saved_persistent_v4,
)

# NeuronCore gen3 (trn2) has no activate2, no engine tensor indirection and a 512-element matmul moving-dim
# limit. Every FP8 kernel now carries in-kernel gen3 branches (bitwise identical to the gen4 instruction
# stream's outputs; see the get_nc_version() gates in kernels/fp8_*.py), so the wrapper is target-uniform
# except for ONE choice: the fused grad-scale+dgrad kernel's K-split schedule (c_fc) is slower than the
# separate pair on gen3, so gen3 fuses only the N-split (c_proj). Resolved ONCE at import: a
# get_platform_target() call inside the traced autograd.Function is a dynamo graph break, and the FP8
# Linears then silently run eagerly (+860 ms/step when this happened during the trn2 port).
GEN4 = trn.target() != "trn2"



def is_contracting_linear(out_features: int, in_features: int) -> bool:
    """True for Linears whose output is (in nanochat) consumed only by the residual add."""
    return in_features > out_features


def split_save_bits(out_features: int, in_features: int) -> bool:
    """Emit the save-only E4M3 kernel next to the fused forward for a CONTRACTING Linear (mlp.c_proj).

    Under "full" recompute activation checkpointing the fused forward kernel returns (out, bytes, inverse_scale)
    from ONE op, so the min-cut partitioner could regenerate the bytes only by re-running the whole GEMM (1.30 ms
    per layer at d24 / MBS 8). The save-only kernel's bytes/scale are bitwise identical to the fused kernel's, so
    with it in the forward the fused kernel's byte outputs are dead in the backward and the partitioner recomputes
    ONLY the save-only op (marked MUST_RECOMPUTE by fp8_recompute_contexts below).
    Expanding Linears (c_fc) are unaffected: their output feeds relu^2, whose backward needs it, so the fused
    kernel is recomputed anyway and its bytes come for free.
    """
    return is_contracting_linear(out_features, in_features)


def fp8_recompute_contexts():
    """torch.utils.checkpoint context_fn for checkpointed Blocks with FP8 Linears: the FP8 saved state follows the
    checkpoint instead of being exported per Block.

    The FP8 Linear's saved state (E4M3 bytes + inverse scale of the input) is produced by NKI custom ops, which
    AOTAutograd's min-cut partitioner cannot see through: without this policy each d24 Block exports 438.5 MiB for
    the backward instead of the BF16 baseline's 246.5 MiB. The save-only op is MUST_RECOMPUTE (one cheap pass over
    the activation); everything else is PREFER_RECOMPUTE, i.e. full recompute. A plain module-level function (no
    closure) so dynamo's checkpoint HOP handling sees a UserFunctionVariable.
    """
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    save_activation_op = torch.ops.nanochat_fp8.save_e4m3_activation_v1_g2.default

    def policy_fn(ctx, op, *args, **kwargs):
        if op is save_activation_op:
            return CheckpointPolicy.MUST_RECOMPUTE
        return CheckpointPolicy.PREFER_RECOMPUTE

    contexts = create_selective_checkpoint_contexts(policy_fn)
    for context in contexts:
        # NKI attention and FP8 kernels enter PyTorch as higher-order operators; the checkpoint dispatch modes
        # keep HOP support disabled unless the caller opts in.
        context.supports_higher_order_operators = True
    return contexts


_FP8_SAVE_E4M3_G2 = tensorwise_e4m3_save[2]
_FP8_GEMM_G2 = tensorwise_fp8_gemm[2]
_FP8_FORWARD_SAVE_G2 = tensorwise_fp8_forward_save_v1[2]
_FP8_FORWARD_NOSAVE_G2 = tensorwise_fp8_forward_nosave_v1[2]
_FP8_GRAD_SCALE_G2 = tensorwise_fp8_grad_scale_state_v1[2]
_FP8_DGRAD_SCALED_G2 = tensorwise_fp8_linear_dgrad_scaled_v1[2]
_FP8_GRAD_SCALE_DGRAD_G2 = tensorwise_fp8_grad_scale_state_dgrad_v1[2]
_FP8_WGRAD_SAVED_G2 = tensorwise_fp8_wgrad_saved[2]
_FP8_WGRAD_SAVED_PERSISTENT_G2 = (
    tensorwise_fp8_wgrad_saved_persistent_v3[2]
)
_FP8_WGRAD_SAVED_PERSISTENT_SCALED_G2 = (
    tensorwise_fp8_wgrad_saved_persistent_v4[2]
)
_FP8_WGRAD_SAVED_GROUPED_G2 = tensorwise_fp8_wgrad_saved_grouped_v1[2]


def _require_lnc2():
    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    if lnc != 2:
        raise RuntimeError(
            "Neuron FP8 requires NEURON_LOGICAL_NC_CONFIG=2; "
            f"the current process is configured for LNC={lnc}"
        )


def choose_nstep(out_features: int) -> int:
    """Choose a PSUM grouping that exactly divides the output tiles."""
    if out_features % N_TILE != 0:
        raise ValueError(f"out_features must be divisible by {N_TILE}")
    n_tiles = out_features // N_TILE
    for candidate in (3, 2, 1):
        if n_tiles % candidate == 0:
            return candidate
    raise AssertionError("unreachable")


def choose_kblock(contraction: int) -> int:
    """Choose the largest validated streamed contraction block."""
    for candidate in (2048, 1024, 512, DOUBLE_ROW_K):
        if contraction % candidate == 0:
            return candidate
    raise ValueError(
        f"streamed FP8 contraction {contraction} must be divisible by {DOUBLE_ROW_K}"
    )


def supports_linear(in_features: int, out_features: int) -> bool:
    """Whether all three Linear GEMMs fit the currently validated kernels."""
    return (
        in_features % N_TILE == 0
        and out_features % N_TILE == 0
        and in_features * out_features <= RESIDENT_B_ELEMENTS
    )


def supports_persistent_wgrad(
    out_features: int,
    in_features: int,
    kblock: int,
) -> bool:
    """Whether one full wgrad accumulator set fits the validated budget."""
    total_m_tiles = out_features // P_MAX
    if total_m_tiles % 2 != 0:
        return False
    m_tiles_per_program = total_m_tiles // 2
    k_tiles_per_block = kblock // P_MAX
    required = (
        P_MAX * m_tiles_per_program * in_features * 4
        + P_MAX * k_tiles_per_block * m_tiles_per_program * P_MAX
        + P_MAX * k_tiles_per_block * P_MAX * 2
        + P_MAX * k_tiles_per_block * N_TILE
        + P_MAX * N_TILE * 2
    )
    return required <= PERSISTENT_WGRAD_SBUF_BUDGET


def grouped_wgrad_mgroup(
    out_features: int,
    in_features: int,
    kblock: int,
) -> int:
    """Return the validated grouped schedule, or zero for the general path."""
    if (out_features, in_features, kblock) == (6144, 1536, 2048):
        # Both targets: on gen3 the kernel issues the wide accumulate matmuls as one 512-wide
        # matmul per PSUM bank (bitwise the gen4 wide form, see _matmul_wgrad_block_wide).
        return 12
    return 0


@nki_op("nanochat_fp8::save_e4m3_activation_v1_g2", mutates_args={})
def fp8_save_activation(
    input_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return native-E4M3 bytes and their replicated inverse scale."""
    return _FP8_SAVE_E4M3_G2(input_2d)


@nki_op("nanochat_fp8::linear_forward_save_v1_g2", mutates_args={})
def fp8_linear_forward_save(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    nstep: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training forward: BF16 output plus native-E4M3 input bytes and replicated inverse scale.

    One fused kernel replaces ``fp8_save_activation`` + ``fp8_linear_forward`` (same math,
    bitwise-identical outputs) and consumes the canonical ``weight[out, in]`` directly.
    """
    return _FP8_FORWARD_SAVE_G2(input_2d, weight, nstep)


@nki_op("nanochat_fp8::linear_forward_nosave_v1_g2", mutates_args={})
def fp8_linear_forward_nosave(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    nstep: int,
) -> torch.Tensor:
    """``fp8_linear_forward_save`` without the E4M3 save: the same BF16 output only."""
    return _FP8_FORWARD_NOSAVE_G2(input_2d, weight, nstep)


@nki_op("nanochat_fp8::linear_forward_g2", mutates_args={})
def fp8_linear_forward(
    input_2d: torch.Tensor,
    weight: torch.Tensor,
    nstep: int,
) -> torch.Tensor:
    """E4M3 input x E4M3 canonical ``weight[out, in]``."""
    return _FP8_GEMM_G2(
        input_2d,
        weight.t(),
        True,
        True,
        nstep,
        1,
    )


@nki_op("nanochat_fp8::linear_dgrad_g2", mutates_args={})
def fp8_linear_dgrad(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    nstep: int,
) -> torch.Tensor:
    """Compute dX with E5M2 grad-output and E4M3 weight."""
    return _FP8_GEMM_G2(
        grad_output,
        weight,
        False,
        True,
        nstep,
        1,
    )


@nki_op("nanochat_fp8::grad_scale_state_v1_g2", mutates_args={})
def fp8_grad_scale_state(
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Compute one shared tensorwise E5M2 scale state for Linear backward."""
    return _FP8_GRAD_SCALE_G2(grad_output)


@nki_op("nanochat_fp8::linear_dgrad_scaled_v1_g2", mutates_args={})
def fp8_linear_dgrad_scaled(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    grad_scale_state: torch.Tensor,
    nstep: int,
) -> torch.Tensor:
    """Compute dX using one gradient scale shared with persistent wgrad."""
    return _FP8_DGRAD_SCALED_G2(
        grad_output,
        weight,
        grad_scale_state,
        nstep,
    )


def fused_dgrad_supported(rows: int, out_features: int, in_features: int, nstep: int) -> bool:
    """Mirror of the shape assertions in ``tensorwise_fp8_grad_scale_state_dgrad_v1`` (double_row path).

    The fused kernel is tuned for the MLP shapes; other Linears (e.g. attention c_proj 1536x1536, whose
    6 weight k tiles per core do not fold into the 4-tile scan chunks) fall back to the separate
    grad_scale_state + dgrad_scaled kernels, which are bitwise identical.
    """
    K, N = out_features, in_features          # dgrad contracts over out_features, produces in_features
    if rows % P_MAX or (rows // P_MAX) % 2 or K % DOUBLE_ROW_K or N % N_TILE or N % (nstep * N_TILE):
        return False
    if K * N > RESIDENT_B_ELEMENTS:
        return False
    k_tiles = K // P_MAX
    if N >= FUSED_DGRAD_NSPLIT_MIN_N:
        if (N // 2) % (nstep * N_TILE):
            return False
        n_own, wk = N // 2, k_tiles
    else:
        if k_tiles % 2:
            return False
        n_own, wk = N, k_tiles // 2
    wu = 1
    if wk % wu or (k_tiles // 2 if N < FUSED_DGRAD_NSPLIT_MIN_N else k_tiles) % wu:
        return False
    w_fold = max(1, 6144 // n_own)
    return wk % w_fold == 0


@nki_op("nanochat_fp8::grad_scale_state_dgrad_v1_g2", mutates_args={})
def fp8_grad_scale_state_dgrad(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    nstep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused: shared E5M2 gradient scale state and dX in one kernel launch."""
    return _FP8_GRAD_SCALE_DGRAD_G2(
        grad_output,
        weight,
        nstep,
    )


@nki_op("nanochat_fp8::linear_wgrad_saved_v1_g2", mutates_args={})
def fp8_linear_wgrad_saved(
    grad_output: torch.Tensor,
    input_bits: torch.Tensor,
    input_inverse_scales: torch.Tensor,
    kblock: int,
) -> torch.Tensor:
    """Compute dW from the E4M3 activation persisted by forward."""
    return _FP8_WGRAD_SAVED_G2(
        grad_output,
        input_bits,
        input_inverse_scales,
        kblock,
    )


@nki_op("nanochat_fp8::linear_wgrad_saved_persistent_v3_g2", mutates_args={})
def fp8_linear_wgrad_saved_persistent(
    grad_output: torch.Tensor,
    input_bits: torch.Tensor,
    input_inverse_scales: torch.Tensor,
    kblock: int,
) -> torch.Tensor:
    """Compute strict FP8 dW with all output columns resident in SBUF."""
    return _FP8_WGRAD_SAVED_PERSISTENT_G2(
        grad_output,
        input_bits,
        input_inverse_scales,
        kblock,
    )


@nki_op(
    "nanochat_fp8::linear_wgrad_saved_persistent_scaled_v1_g2",
    mutates_args={},
)
def fp8_linear_wgrad_saved_persistent_scaled(
    grad_output: torch.Tensor,
    input_bits: torch.Tensor,
    input_inverse_scales: torch.Tensor,
    grad_scale_state: torch.Tensor,
    kblock: int,
) -> torch.Tensor:
    """Compute persistent strict FP8 dW using the shared gradient scale."""
    return _FP8_WGRAD_SAVED_PERSISTENT_SCALED_G2(
        grad_output,
        input_bits,
        input_inverse_scales,
        grad_scale_state,
        kblock,
    )


@nki_op(
    "nanochat_fp8::linear_wgrad_saved_grouped_scaled_v1_g2",
    mutates_args={},
)
def fp8_linear_wgrad_saved_grouped_scaled(
    grad_output: torch.Tensor,
    input_bits: torch.Tensor,
    input_inverse_scales: torch.Tensor,
    grad_scale_state: torch.Tensor,
    kblock: int,
    mgroup: int,
) -> torch.Tensor:
    """Compute grouped-persistent strict FP8 dW with a shared scale."""
    return _FP8_WGRAD_SAVED_GROUPED_G2(
        grad_output,
        input_bits,
        input_inverse_scales,
        grad_scale_state,
        kblock,
        mgroup,
    )


class _NeuronFloat8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_2d,
        master_weight,
        forward_nstep,
        dgrad_nstep,
        wgrad_kblock,
        split_save,
    ):
        weight = master_weight.to(COMPUTE_DTYPE)
        if split_save and input_2d.shape[0] >= 4 * master_weight.shape[0]:
            # The bytes/scale come from fp8_save_activation below, so skip the fused kernel's save
            # (the no-save kernel has only the row-split schedule, M >= 4N).
            output = fp8_linear_forward_nosave(input_2d, weight, forward_nstep)
        else:
            output, input_bits, input_inverse_scales = fp8_linear_forward_save(
                input_2d,
                weight,
                forward_nstep,
            )
        if split_save:
            # Bitwise the same bytes/scale as the fused kernel's, from a save-only kernel: under
            # "full" recompute checkpointing the backward then recomputes this kernel instead of
            # the fused GEMM (see split_save_bits).
            input_bits, input_inverse_scales = fp8_save_activation(input_2d)
        ctx.save_for_backward(
            input_bits,
            input_inverse_scales,
            master_weight,
        )
        ctx.dgrad_nstep = dgrad_nstep
        ctx.wgrad_kblock = wgrad_kblock
        ctx.use_persistent_wgrad = supports_persistent_wgrad(
            master_weight.shape[0],
            master_weight.shape[1],
            wgrad_kblock,
        )
        ctx.grouped_wgrad_mgroup = grouped_wgrad_mgroup(
            master_weight.shape[0],
            master_weight.shape[1],
            wgrad_kblock,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            input_bits,
            input_inverse_scales,
            master_weight,
        ) = ctx.saved_tensors
        weight = master_weight.to(COMPUTE_DTYPE)
        grad_output = grad_output.reshape(-1, master_weight.shape[0])
        grad_input = None
        grad_weight = None
        use_shared_scale_state = (
            (ctx.use_persistent_wgrad or ctx.grouped_wgrad_mgroup != 0)
            and ctx.needs_input_grad[0]
            and ctx.needs_input_grad[1]
        )
        grad_scale_state = None
        if (
            # gen3 (trn2): the fused kernel's K-split schedule (c_fc) is ~5% slower than the separate pair there
            # (the layout fits only with a shared raw region that serializes transpose/quantize overlap); the
            # N-split (c_proj) is ~10% faster. All variants are bitwise identical.
            (GEN4 or master_weight.shape[1] >= FUSED_DGRAD_NSPLIT_MIN_N)
            and use_shared_scale_state
            and ctx.needs_input_grad[0]
            and fused_dgrad_supported(
                grad_output.shape[0], master_weight.shape[0], master_weight.shape[1], ctx.dgrad_nstep
            )
        ):
            # one fused launch: scale state (shared with wgrad) + dX
            grad_scale_state, grad_input = fp8_grad_scale_state_dgrad(
                grad_output,
                weight,
                ctx.dgrad_nstep,
            )
        elif use_shared_scale_state:
            grad_scale_state = fp8_grad_scale_state(grad_output)
        if ctx.needs_input_grad[0] and grad_input is None:
            if grad_scale_state is None:
                grad_input = fp8_linear_dgrad(
                    grad_output,
                    weight,
                    ctx.dgrad_nstep,
                )
            else:
                grad_input = fp8_linear_dgrad_scaled(
                    grad_output,
                    weight,
                    grad_scale_state,
                    ctx.dgrad_nstep,
                )
        if ctx.needs_input_grad[1]:
            if (
                ctx.grouped_wgrad_mgroup != 0
                and grad_scale_state is not None
            ):
                grad_weight = fp8_linear_wgrad_saved_grouped_scaled(
                    grad_output,
                    input_bits,
                    input_inverse_scales,
                    grad_scale_state,
                    ctx.wgrad_kblock,
                    ctx.grouped_wgrad_mgroup,
                )
            elif ctx.use_persistent_wgrad:
                if grad_scale_state is None:
                    grad_weight = fp8_linear_wgrad_saved_persistent(
                        grad_output,
                        input_bits,
                        input_inverse_scales,
                        ctx.wgrad_kblock,
                    )
                else:
                    grad_weight = fp8_linear_wgrad_saved_persistent_scaled(
                        grad_output,
                        input_bits,
                        input_inverse_scales,
                        grad_scale_state,
                        ctx.wgrad_kblock,
                    )
            else:
                grad_weight = fp8_linear_wgrad_saved(
                    grad_output,
                    input_bits,
                    input_inverse_scales,
                    ctx.wgrad_kblock,
                )
            grad_weight = grad_weight.to(master_weight.dtype)
        return grad_input, grad_weight, None, None, None, None


class NeuronFloat8Linear(torch.nn.Linear):
    """Trainium implementation of nanochat's tensorwise FP8 Linear recipe."""

    _nanochat_fp8 = True
    fp8_backend = "neuron"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        _require_lnc2()
        if not supports_linear(in_features, out_features):
            raise ValueError(
                "Neuron FP8 requires in/out features divisible by 512 and "
                f"in_features*out_features <= {RESIDENT_B_ELEMENTS:,}; got "
                f"{in_features} -> {out_features}"
            )
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self._forward_nstep = choose_nstep(out_features)
        self._dgrad_nstep = choose_nstep(in_features)

    @staticmethod
    def is_supported(module: torch.nn.Linear) -> bool:
        return supports_linear(module.in_features, module.out_features)

    @classmethod
    def from_float(cls, module: torch.nn.Linear):
        """Create a backend view sharing the original parameters."""
        with torch.device("meta"):
            converted = cls(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
        converted.weight = module.weight
        converted.bias = module.bias
        converted.train(module.training)
        return converted

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(COMPUTE_DTYPE)
        original_shape = input.shape
        input_2d = input.reshape(-1, self.in_features)
        if input_2d.shape[0] % DOUBLE_ROW_K != 0:
            raise ValueError(
                f"Neuron FP8 row count must be divisible by {DOUBLE_ROW_K}; "
                f"got {input_2d.shape[0]}"
            )

        wgrad_kblock = choose_kblock(input_2d.shape[0])
        if torch.is_grad_enabled():
            output = _NeuronFloat8LinearFn.apply(
                input_2d,
                self.weight,
                self._forward_nstep,
                self._dgrad_nstep,
                wgrad_kblock,
                split_save_bits(self.out_features, self.in_features),
            )
        else:
            output = fp8_linear_forward(
                input_2d,
                self.weight.to(COMPUTE_DTYPE),
                self._forward_nstep,
            )
        output = output.reshape(*original_shape[:-1], self.out_features)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output
