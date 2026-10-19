"""Constants and NKI helpers shared by the FP8 kernel families (split from nki_fp8.py)."""

import math
import os

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nki.isa import matmul_perf_mode

P_MAX = 128
N_TILE = 512
DOUBLE_ROW_K = 256
E4M3_MAX = 240.0
E5M2_MAX = 57344.0
EPS = 1.0e-12
RESIDENT_B_ELEMENTS = 12_000_000


def kernel_assert(condition: bool, error_text: str):
    """Raise an NKI-formatted validation error at specialization time."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def _broadcast_partition_zero(src, dst):
    """Broadcast src[0, 0] to a [128, 1] tile using four shuffle quadrants."""
    shuffle_mask = [0] * 32
    for quadrant in nl.static_range(4):
        start = quadrant * 32
        nisa.nc_stream_shuffle(
            dst=dst[start:start + 32, 0:1],
            src=src[0:1, 0:1],
            shuffle_mask=shuffle_mask,
        )


def _tensorwise_amax_rows(tensor, row_offset: int, row_count: int, width: int):
    """Compute an exact tensorwise abs-max over a contiguous row range."""
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
    amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=amax, op=nl.maximum, data=running)
    return amax


def _fp8_scale_from_amax(amax, fp8_max: float):
    """Return broadcast quantization and inverse scales in FP32."""
    clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=clamped,
        data=amax,
        op0=nl.maximum,
        operand0=EPS,
    )
    reciprocal_zero = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.reciprocal(dst=reciprocal_zero, data=clamped)
    qscale_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=qscale_zero,
        data=reciprocal_zero,
        op0=nl.multiply,
        operand0=fp8_max,
    )
    qscale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _broadcast_partition_zero(qscale_zero, qscale)
    inverse = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inverse, data=qscale)
    return qscale, inverse


# ---------------------------------------------------------------------------
# Split-accumulation abs-max helpers (fp8_quantize: tensorwise_e4m3_save and
# tensorwise_fp8_grad_scale_state_v1 scan their input with these).
# ---------------------------------------------------------------------------


def _qkv_amax_rows_split2(tensor, row_offset: int, row_count: int, width: int, running):
    """Like _qkv_amax_rows_split but alternates the loads between the static sync
    queue and the ScalarE hwdge queue so the scan is not bound by one DMA queue."""
    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    row_tiles = row_count // P_MAX
    kernel_assert(row_tiles % (2 * grid) == 0, "amax row tiles must divide 2 x grid")
    scan_a = nl.ndarray((P_MAX, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    scan_b = nl.ndarray((P_MAX, width), dtype=nl.bfloat16, buffer=nl.sbuf)
    per_a = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    per_b = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for local_pair in nl.affine_range(row_tiles // (2 * grid)):
        row_a = row_offset + ((2 * local_pair) * grid + program_id) * P_MAX
        row_b = row_offset + ((2 * local_pair + 1) * grid + program_id) * P_MAX
        nisa.dma_copy(
            dst=scan_a, src=tensor[row_a:row_a + P_MAX, 0:width], dge_mode=nisa.dge_mode.none
        )
        nisa.dma_copy(
            dst=scan_b, src=tensor[row_b:row_b + P_MAX, 0:width],
            dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar,
        )
        nisa.tensor_scalar_reduce(
            dst=scan_a, data=scan_a, op0=nl.abs, operand0=0.0, reduce_op=nl.maximum, reduce_res=per_a
        )
        nisa.tensor_tensor(dst=running, data1=running, data2=per_a, op=nl.maximum)
        nisa.tensor_scalar_reduce(
            dst=scan_b, data=scan_b, op0=nl.abs, operand0=0.0, reduce_op=nl.maximum, reduce_res=per_b
        )
        nisa.tensor_tensor(dst=running, data1=running, data2=per_b, op=nl.maximum)


def _qkv_exchange_amax(running, scratch, index: int):
    """Combine one operand's per-core partial maxima through shared HBM.

    Write this core's partition-reduced partial into ``scratch[program_id, index]``
    (shared_hbm (2, S, P_MAX, 1) fp32). The caller issues one
    ``nisa.core_barrier(data=scratch, cores=(0, 1))`` after all operands were
    written, then calls ``_qkv_combine_amax`` per operand.
    """
    program_id = nl.program_id(0)
    partial = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=partial, op=nl.maximum, data=running)
    # Tiny exchange DMAs go through the GpSimd (swdge) queue: on the in-order sync
    # queue they would sit behind megabytes of prefetched static loads (~20 us).
    nisa.dma_copy(
        dst=scratch[program_id, index, :, :], src=partial, dge_mode=nisa.dge_mode.swdge
    )
    return partial

def _qkv_combine_amax(partial, scratch, index: int):
    """After the barrier: max(own partial, other core's partial); exact, so the
    result is bitwise identical to a full single-core scan (partition 0 valid)."""
    program_id = nl.program_id(0)
    other = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=other, src=scratch[1 - program_id, index, :, :], dge_mode=nisa.dge_mode.swdge
    )
    amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=amax, data1=partial, data2=other, op=nl.maximum)
    return amax
