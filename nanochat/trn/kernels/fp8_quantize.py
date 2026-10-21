"""FP8 quantize / save kernels: E4M3 activation save and E5M2 gradient scale state (split from nki_fp8.py)."""

import math
import os

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nki.compiler import CompileConfig
from nki.isa import matmul_perf_mode

from nanochat.trn.kernels.fp8_common import (
    E4M3_MAX,
    E5M2_MAX,
    N_TILE,
    P_MAX,
    _fp8_scale_from_amax,
    _qkv_amax_rows_split2,
    _qkv_combine_amax,
    _qkv_exchange_amax,
    kernel_assert,
)


def _tensorwise_amax_rows_partial(tensor, row_offset: int, row_count: int, width: int):
    """Per-partition running abs-max over a contiguous row range (no partition reduce)."""
    running = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    scan = nl.ndarray((P_MAX, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    per_partition = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    for row_tile in nl.affine_range(row_count // P_MAX):
        row_start = row_offset + row_tile * P_MAX
        nisa.dma_copy(
            dst=scan,
            src=tensor[row_start:row_start + P_MAX, 0:width],
        )
        nisa.tensor_scalar_reduce(
            dst=scan,
            data=scan,
            op0=nl.abs,
            operand0=0.0,
            reduce_op=nl.maximum,
            reduce_res=per_partition,
        )
        nisa.tensor_tensor(
            dst=running,
            data1=running,
            data2=per_partition,
            op=nl.maximum,
        )
    return running


# Opt in to non-SSA legalization (off by default for NKI kernels): 1.49 -> 0.96 ms at d24.
@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def tensorwise_e4m3_save(input_2d):
    """Quantize a BF16 activation once and persist native E4M3 bits in HBM."""
    kernel_assert(len(input_2d.shape) == 2, "input must be rank 2")
    M, K = input_2d.shape
    kernel_assert(M % P_MAX == 0, "input rows must be divisible by 128")
    kernel_assert(K % N_TILE == 0, "input width must be divisible by 512")

    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    row_tiles = M // P_MAX
    kernel_assert(row_tiles % grid == 0, "input row tiles must divide the grid")

    # torch_neuronx cannot expose legacy E4M3 as a graph tensor. Carry the
    # exact bytes as uint8 and reinterpret them only inside NKI kernels.
    saved_bits = nl.ndarray(
        input_2d.shape,
        dtype=nl.uint8,
        buffer=nl.shared_hbm,
    )
    saved_e4m3 = saved_bits.view(nl.float8_e4m3)
    inverse_scales = nl.ndarray(
        (2, P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )

    amax_scratch = nl.ndarray((2, 1, P_MAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    input_running = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    _qkv_amax_rows_split2(input_2d, 0, M, K, input_running)
    input_partial = _qkv_exchange_amax(input_running, amax_scratch, 0)
    nisa.core_barrier(data=amax_scratch, cores=(0, 1))
    input_amax = _qkv_combine_amax(input_partial, amax_scratch, 0)
    input_qscale, input_inverse = _fp8_scale_from_amax(
        input_amax,
        E4M3_MAX,
    )
    nisa.dma_copy(
        dst=inverse_scales[program_id, 0:P_MAX, 0:1],
        src=input_inverse,
    )

    raw = nl.ndarray(
        (P_MAX, K),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    quantized = nl.ndarray(
        (P_MAX, K),
        dtype=nl.float8_e4m3,
        buffer=nl.sbuf,
    )
    rows_per_program = row_tiles // grid
    for local_row_tile in nl.affine_range(rows_per_program):
        row_start = (local_row_tile * grid + program_id) * P_MAX
        nisa.dma_copy(
            dst=raw,
            src=input_2d[row_start:row_start + P_MAX, 0:K],
            dge_mode=nisa.dge_mode.none,
        )
        nisa.activation(
            dst=quantized,
            op=nl.copy,
            data=raw,
            scale=input_qscale,
        )
        nisa.dma_copy(
            dst=saved_e4m3[row_start:row_start + P_MAX, 0:K],
            src=quantized,
            dge_mode=nisa.dge_mode.hwdge,
            engine=nisa.engine.scalar,
        )

    return saved_bits, inverse_scales


# Opt in to non-SSA legalization (off by default for NKI kernels): 0.93 -> 0.47 ms at d24.
@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def tensorwise_fp8_grad_scale_state_v1(grad_output):
    """Compute one E5M2 gradient scale state shared by dgrad and wgrad.

    Each of the two LNC2 cores scans half of the rows, the per-partition
    running maxima are exchanged through a shared-HBM scratch tensor with a
    core barrier, and both cores then derive the identical tensorwise scale.
    (max is order-independent, so the amax is bitwise identical to a full
    single-core scan.)
    """
    kernel_assert(len(grad_output.shape) == 2, "grad_output must be rank 2")
    M, N = grad_output.shape
    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    kernel_assert(grid == 2, "gradient scale sharing requires LNC2")
    kernel_assert((M // P_MAX) % 2 == 0, "row tiles must split across cores")
    half_rows = M // 2

    state = nl.ndarray(
        (2, 2, P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )
    partial = nl.ndarray(
        (2, P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )
    # DMA efficiency: the scan does not care which rows land on which
    # partition, so fold ``rows_per_part`` consecutive rows (>= 12 KB contiguous
    # per partition) into one scan row. Descriptors of ~3 KB (N=1536) run at a
    # much lower DMA-engine rate than 12 KB ones.
    rows_per_part = max(1, (6144 // N))
    kernel_assert(half_rows % (rows_per_part * P_MAX) == 0, "rows must fold evenly")
    grad_view = grad_output.reshape((M // rows_per_part, rows_per_part * N))
    running = _tensorwise_amax_rows_partial(
        grad_view,
        program_id * (half_rows // rows_per_part),
        half_rows // rows_per_part,
        rows_per_part * N,
    )
    nisa.dma_copy(dst=partial[program_id, :, :], src=running)
    nisa.core_barrier(partial, (0, 1))
    other = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=other, src=partial[1 - program_id, :, :])
    merged = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=merged,
        data1=running,
        data2=other,
        op=nl.maximum,
    )
    grad_amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=grad_amax, op=nl.maximum, data=merged)
    grad_scale, grad_inverse = _fp8_scale_from_amax(
        grad_amax,
        E5M2_MAX,
    )
    nisa.dma_copy(dst=state[program_id, 0, :, :], src=grad_scale)
    nisa.dma_copy(dst=state[program_id, 1, :, :], src=grad_inverse)
    return state
