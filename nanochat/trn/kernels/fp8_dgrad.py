"""FP8 dgrad kernels: scaled dgrad and the fused grad-scale-state + dgrad kernel (split from nki_fp8.py)."""

import math
import os

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nki.compiler import CompileConfig
from nki.isa import matmul_perf_mode

from nanochat.trn.kernels.fp8_common import (
    DOUBLE_ROW_K,
    E4M3_MAX,
    E5M2_MAX,
    N_TILE,
    P_MAX,
    RESIDENT_B_ELEMENTS,
    _fp8_scale_from_amax,
    _tensorwise_amax_rows,
    kernel_assert,
)
# bytes per SBUF partition left to the compiler (scan / dequant scratch) in the fused dgrad kernel
FUSED_DGRAD_COMPILER_RESERVE = 32 * 1024
FUSED_DGRAD_PINNED_GSCAN = 4
# fused dgrad kernel scheduling knobs (see tensorwise_fp8_grad_scale_state_dgrad_v1)
FUSED_DGRAD_WLOAD_PRIORITY = 0         # DMA QoS priority of the resident weight loads (0 = highest)
FUSED_DGRAD_NSPLIT_MIN_N = 6144      # N-split (each core owns N/2 output columns, no E4M3 exchange) when N >= this
FUSED_DGRAD_NSPLIT_WLOAD_PRIORITY = 0         # N-split: DMA QoS priority of the weight loads (None = default, interleaves with the gradient scan)
FUSED_DGRAD_NSPLIT_WLOAD_PER_GTILE = 0         # N-split: weight-tile loads issued per gradient-tile iteration (0 = all before the scan)
FUSED_DGRAD_WCLAMP_EARLY_TILE = 7         # K-split: weight tile whose clamp is the early scheduling hint (-1: none)
FUSED_DGRAD_WCLAMP_EARLY_GI = 10        # K-split: gradient tile of the early hint
FUSED_DGRAD_WCLAMP_SPREAD_GI0 = 26        # K-split: from this gradient tile on, one weight clamp per gradient tile is forced
FUSED_DGRAD_WX_GI_BACK = 0         # K-split: exchange the E4M3 half after gradient tile (g_tiles - 1 - this)
FUSED_DGRAD_WX_ANCHOR = True      # K-split: anchor the exchange to that gradient tile (throw-away write into the receive slots)
FUSED_DGRAD_WX_CHUNKS = 4         # K-split: number of k-ordered sendrecv chunks for the E4M3 exchange (distinct pipes)
FUSED_DGRAD_B_GATE_SENDRECV = False     # K-split: junk sendrecv into the second gradient buffer on the last exchange pipe (orders B's transposes behind the exchange)
FUSED_DGRAD_NSPLIT_FORCE_TILES = 4         # N-split: weight tiles 0..n-1 are forced during the scan
FUSED_DGRAD_NSPLIT_FORCE_GI0 = 9         # N-split: gradient tile of the first force
FUSED_DGRAD_NSPLIT_FORCE_STEP = 2         # N-split: gradient tiles between forces
FUSED_DGRAD_Q1_AFTER_Q0 = True      # order the E5M2 quantize of row tile 1 after both q(0) halves on ScalarE
FUSED_DGRAD_B_AFTER_WX = False     # K-split: order the second row tile's gradient transposes behind the E4M3 exchange on the DMA rings
FUSED_DGRAD_WSCAN_ACT2_CHUNKS = 0         # N-split: weight-scan chunks reduced on ScalarE (activate2 abs_max registers) instead of VectorE
FUSED_DGRAD_WSCALE_HINT_GI = -1        # N-split: gradient tile after which the weight scale is tied into the running max (priority hint; -1 off)
FUSED_DGRAD_NSPLIT_LATE_CLAMPS = False     # N-split: clamp only the forced weight tiles inside the scan; the rest after the gradient scale
FUSED_DGRAD_Q0_PREFETCH_GI = -1        # K-split: gradient tile after which the first row tile's transposes are issued (-1: after the scan)
FUSED_DGRAD_NSPLIT_GSCAN_ACT2_TILES = 0         # N-split: first n gradient-scan tiles reduced on ScalarE (activate2 abs_max registers)
FUSED_DGRAD_NSPLIT_GSCAN_REGS = False     # N-split: accumulate the gradient abs-max in the VectorE reduction registers (one read-out)
FUSED_DGRAD_GT_PRIORITY = None      # DMA QoS priority (0-3, lower = higher; None = default) of the main-loop gradient transposes
FUSED_DGRAD_B_AFTER_GSR = False     # K-split: hold the second row tile's transposes behind the gradient-partial swap (ScalarE-carried dependency)
FUSED_DGRAD_Q0_SPLIT = True      # quantize the first row tile in two k halves (matmuls start after the first half)
FUSED_DGRAD_SWAP_AB = False     # K-split: first row tile in the B buffers (probe of the compiler's transpose emission order)
FUSED_DGRAD_SPLIT_LAST_STORE = True      # last row tile's last group: dequant + store per 512-column subtile (tail overlap)
FUSED_DGRAD_Q1_SPLIT = False     # quantize the second row tile in two k halves as well (emission-order probe)
FUSED_DGRAD_PE_TRANSPOSE_TILES = 0         # K-split: first n row tiles loaded plain during the scan and transposed on TensorE (nc_transpose) instead of DMA transposes after it
FUSED_DGRAD_PE_GI0 = 40        # K-split: gradient tile after which the first PE-transposed tile is loaded
FUSED_DGRAD_PE_GI_STEP = 4         # K-split: gradient tiles between the two PE-transposed tile loads
FUSED_DGRAD_PE_FORCE_DELAY = 3         # K-split: gradient tiles after a PE-transposed tile's load until its anti-sink hint
FUSED_DGRAD_NSPLIT_GPSIMD_CLAMPS = 0         # N-split: weight tiles 0..n-1 clamped on GpSimd (byte-identical) instead of VectorE
FUSED_DGRAD_KSPLIT_GPSIMD_CLAMPS = 0         # K-split: weight tiles 0..n-1 clamped on GpSimd (byte-identical) instead of VectorE
FUSED_DGRAD_GSCAN_PRIORITY = None      # DMA QoS priority of the gradient-scan loads (None = default)
FUSED_DGRAD_SCAN_PE_A_GI = -1        # K-split: scan tile whose plain data becomes the first dgrad row tile via PE transposes (-1 off)
FUSED_DGRAD_SCAN_PE_B_GI = -1        # K-split: scan tile whose plain data becomes the second dgrad row tile via PE transposes (-1 off)
FUSED_DGRAD_SCAN_PE_FORCE_DELAY = 3         # K-split: gradient tiles after a PE-transposed scan tile until its anti-sink hint
FUSED_DGRAD_PE_FORCE_BOUND = float('inf')  # bound of the (x > bound) predicate of the PE anti-sink hint (false for every value)
FUSED_DGRAD_PE_PARTITION_REDUCE = False     # cross-partition amax via PE transpose + VectorE max instead of the GpSimd partition reduce
FUSED_DGRAD_WX_ANCHOR_SRC = 'gscan'   # K-split: source of the exchange anchor copies: 'gscan' (raw bytes of the anchor tile) or 'reduce' (its per-partition reduce result)
FUSED_DGRAD_SCAN_PE_B_COPY_ENGINE = 'scalar'  # K-split: engine of the PSUM->SBUF copies of the PE-transposed second row tile ('scalar' or 'vector')
FUSED_DGRAD_Q0_PIECES = 2         # number of contiguous k-tile groups the first row tile is quantized in (the first matmuls wait for group 0 only)
FUSED_DGRAD_SHARED_RAW = False     # one shared raw (bf16, transposed) gradient buffer for both pipeline slots (E5M2 buffers separate)
FUSED_DGRAD_B_HOLD_WX = -1        # K-split: hold the second row tile's transposes behind exchange chunk n via a live read of B's raw buffer (-1 off)
FUSED_DGRAD_NSPLIT_SHARED_RAW = True      # N-split: one shared raw gradient buffer for both pipeline slots
FUSED_DGRAD_NSPLIT_Q0_PIECES = 4         # N-split: number of k-tile groups the first row tile is quantized in
FUSED_DGRAD_WLOAD_HWDGE = False     # weight loads with hardware DGE descriptors (no GpSimd descriptor generation)
FUSED_DGRAD_NSPLIT_PE_PARTITION_REDUCE = True      # N-split: cross-partition amax via PE transpose + VectorE max instead of the GpSimd partition reduce
FUSED_DGRAD_LOAD_REDUCE = True      # K-split: scan reduces accumulate into the running max with reduce_cmd=load_reduce (no separate merge instruction)
FUSED_DGRAD_NSPLIT_LOAD_REDUCE = False     # N-split: scan reduces accumulate into the running max with reduce_cmd=load_reduce
FUSED_DGRAD_LAST_TILE_SPLIT = False     # K-split: last gradient scan tile loaded and reduced as two halves
FUSED_DGRAD_NSPLIT_LAST_TILE_SPLIT = True      # N-split: last gradient scan tile loaded and reduced as two halves
FUSED_DGRAD_GSR_BEFORE_WX = False     # K-split: gradient-partial sendrecv + merge issued in the last scan iteration BEFORE the E4M3 exchange (program order)


def _quantize_dgrad_weight_segment(
    weight,
    weight_fp8,
    row_offset: int,
    contraction: int,
    out_features: int,
    qscale,
):
    """Quantize canonical [contraction, out] rows for a dgrad GEMM."""
    raw = nl.ndarray(
        (P_MAX, out_features),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    for k_tile in nl.affine_range(contraction // P_MAX):
        source_row = row_offset + k_tile * P_MAX
        nisa.dma_copy(
            dst=raw,
            src=weight[
                source_row:source_row + P_MAX,
                0:out_features,
            ],
        )
        nisa.activation(
            dst=weight_fp8[:, k_tile, :],
            op=nl.copy,
            data=raw,
            scale=qscale,
        )
        nisa.tensor_scalar(
            dst=weight_fp8[:, k_tile, :],
            data=weight_fp8[:, k_tile, :],
            op0=nl.minimum,
            operand0=E4M3_MAX,
            op1=nl.maximum,
            operand1=-E4M3_MAX,
        )


def _load_quantize_activation_row(
    tensor,
    row_start,
    contraction: int,
    qscale,
    fp8_dtype,
    fp8_max: float,
    raw,
    quantized,
):
    """Load one 128-row activation tile and quantize it directly to FP8."""
    k_tiles = contraction // P_MAX
    for k_tile in nl.affine_range(k_tiles):
        k_start = k_tile * P_MAX
        nisa.dma_transpose(
            dst=raw[:, k_tile, :],
            src=tensor[
                row_start:row_start + P_MAX,
                k_start:k_start + P_MAX,
            ],
        )
    raw_flat = raw.reshape((P_MAX, k_tiles * P_MAX))
    quantized_flat = quantized.reshape((P_MAX, k_tiles * P_MAX))
    nisa.activation(
        dst=quantized_flat,
        op=nl.copy,
        data=raw_flat,
        scale=qscale,
    )
    nisa.tensor_scalar(
        dst=quantized_flat,
        data=quantized_flat,
        op0=nl.minimum,
        operand0=fp8_max,
        op1=nl.maximum,
        operand1=-fp8_max,
    )


def _dgrad_quantize_row_tile(
    grad_output,
    row_start,
    K: int,
    grad_qscale,
    grad_raw,
    grad_fp8,
):
    """Load one 128-row grad tile (transposed) and quantize it to E5M2."""
    _load_quantize_activation_row(
        grad_output,
        row_start,
        K,
        grad_qscale,
        nl.float8_e5m2,
        E5M2_MAX,
        grad_raw,
        grad_fp8,
    )


def _dgrad_quantize_row_tile_split(
    grad_output,
    row_start,
    K: int,
    grad_qscale,
    grad_raw,
    grad_fp8,
):
    """q(0) variant: quantize + clamp in two k-halves so the first matmuls
    (which consume k pairs in order) can start after the first half."""
    k_tiles = K // P_MAX
    for k_tile in nl.static_range(k_tiles):
        k_start = k_tile * P_MAX
        nisa.dma_transpose(
            dst=grad_raw[:, k_tile, :],
            src=grad_output[
                row_start:row_start + P_MAX,
                k_start:k_start + P_MAX,
            ],
        )
    half = k_tiles // 2
    for h in nl.static_range(2):
        lo = h * half
        hi = k_tiles if h == 1 else half
        nisa.activation(
            dst=grad_fp8[:, lo:hi, :],
            op=nl.copy,
            data=grad_raw[:, lo:hi, :],
            scale=grad_qscale,
        )
        nisa.tensor_scalar(
            dst=grad_fp8[:, lo:hi, :],
            data=grad_fp8[:, lo:hi, :],
            op0=nl.minimum,
            operand0=E5M2_MAX,
            op1=nl.maximum,
            operand1=-E5M2_MAX,
        )


def _dgrad_matmul_row_tile(
    weight_fp8,
    out,
    row_start,
    K: int,
    N: int,
    nstep: int,
    output_scale,
    grad_fp8,
):
    """Run the resident dgrad GEMM for one quantized 128-row grad tile."""
    n_tiles = N // N_TILE
    for n_group in nl.affine_range(n_tiles // nstep):
        n_start = n_group * nstep * N_TILE
        accum = nl.ndarray(
            (P_MAX, nstep * N_TILE),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for n_subtile in nl.affine_range(nstep):
            col_start = n_start + n_subtile * N_TILE
            for k_pair in nl.affine_range(K // DOUBLE_ROW_K):
                nisa.nc_matmul(
                    dst=accum[
                        :,
                        n_subtile * N_TILE:(n_subtile + 1) * N_TILE,
                    ],
                    stationary=grad_fp8[
                        :,
                        2 * k_pair:2 * k_pair + 2,
                        :,
                    ],
                    moving=weight_fp8[
                        :,
                        2 * k_pair:2 * k_pair + 2,
                        col_start:col_start + N_TILE,
                    ],
                    perf_mode=matmul_perf_mode.double_row,
                    accumulate=(k_pair > 0),
                )

        result = nl.ndarray(
            (P_MAX, nstep * N_TILE),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        nisa.activation(
            dst=result,
            op=nl.copy,
            data=accum,
            scale=output_scale,
        )
        nisa.dma_copy(
            dst=out[
                row_start:row_start + P_MAX,
                n_start:n_start + nstep * N_TILE,
            ],
            src=result,
        )


def _dgrad_pipelined_rows(
    grad_output,
    weight_fp8,
    out,
    K: int,
    N: int,
    nstep: int,
    grad_qscale,
    output_scale,
    grid,
    program_id,
    rows_per_program: int,
    grad_buffers=None,
):
    """Software-pipelined dgrad main loop over this program's row tiles.

    Engines issue in program order, so with the naive order
    [quantize(i), matmul(i), dequant(i)] the ScalarE stream is
    q(i) d(i) q(i+1) ...: q(i+1) sits behind d(i), which waits for the last
    matmul of tile i -> TensorE idles for the whole quantize+clamp (~8 us per
    128-row tile on c_fc). We instead issue q(i+1) BEFORE the matmuls of
    tile i, using two grad buffer sets (A/B) and two PSUM accumulator sets.
    """
    k_tiles = K // P_MAX
    kernel_assert(rows_per_program >= 4, "need at least 4 row tiles per program")
    kernel_assert(rows_per_program % 2 == 0, "row tiles per program must be even")
    if grad_buffers is None:
        grad_raw_a = nl.ndarray(
            (P_MAX, k_tiles, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        grad_fp8_a = nl.ndarray(
            (P_MAX, k_tiles, P_MAX), dtype=nl.float8_e5m2, buffer=nl.sbuf
        )
        grad_raw_b = nl.ndarray(
            (P_MAX, k_tiles, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf
        )
        grad_fp8_b = nl.ndarray(
            (P_MAX, k_tiles, P_MAX), dtype=nl.float8_e5m2, buffer=nl.sbuf
        )
    else:
        grad_raw_a, grad_fp8_a, grad_raw_b, grad_fp8_b = grad_buffers
    n_pairs = rows_per_program // 2
    _dgrad_quantize_row_tile_split(
        grad_output,
        program_id * P_MAX,
        K,
        grad_qscale,
        grad_raw_a,
        grad_fp8_a,
    )
    for local_pair in nl.affine_range(n_pairs - 1):
        row_a = ((2 * local_pair) * grid + program_id) * P_MAX
        row_b = ((2 * local_pair + 1) * grid + program_id) * P_MAX
        row_next_a = ((2 * local_pair + 2) * grid + program_id) * P_MAX
        _dgrad_quantize_row_tile(
            grad_output, row_b, K, grad_qscale, grad_raw_b, grad_fp8_b
        )
        _dgrad_matmul_row_tile(
            weight_fp8,
            out,
            row_a,
            K,
            N,
            nstep,
            output_scale,
            grad_fp8_a,
        )
        _dgrad_quantize_row_tile(
            grad_output, row_next_a, K, grad_qscale, grad_raw_a, grad_fp8_a
        )
        _dgrad_matmul_row_tile(
            weight_fp8,
            out,
            row_b,
            K,
            N,
            nstep,
            output_scale,
            grad_fp8_b,
        )
    last_a = ((2 * (n_pairs - 1)) * grid + program_id) * P_MAX
    last_b = ((2 * (n_pairs - 1) + 1) * grid + program_id) * P_MAX
    _dgrad_quantize_row_tile(
        grad_output, last_b, K, grad_qscale, grad_raw_b, grad_fp8_b
    )
    _dgrad_matmul_row_tile(
        weight_fp8,
        out,
        last_a,
        K,
        N,
        nstep,
        output_scale,
        grad_fp8_a,
    )
    _dgrad_matmul_row_tile(
        weight_fp8,
        out,
        last_b,
        K,
        N,
        nstep,
        output_scale,
        grad_fp8_b,
    )


# Opt in to non-SSA legalization (off by default for NKI kernels): 1.49 -> 1.35 ms at d24.
@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def tensorwise_fp8_linear_dgrad_scaled_v1(
    grad_output,
    weight,
    grad_scale_state,
    nstep: int,
):
    """Compute dgrad from a shared E5M2 gradient scale."""
    kernel_assert(len(grad_output.shape) == 2, "grad_output must be rank 2")
    kernel_assert(len(weight.shape) == 2, "weight must be rank 2")
    M, K = grad_output.shape
    K_w, N = weight.shape
    kernel_assert(K == K_w, "dgrad contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "M must be divisible by 128")
    kernel_assert(K % DOUBLE_ROW_K == 0, "K must be divisible by 256")
    kernel_assert(N % N_TILE == 0, "N must be divisible by 512")
    kernel_assert(N % (nstep * N_TILE) == 0, "N must divide nstep * 512")
    kernel_assert(
        K * N <= RESIDENT_B_ELEMENTS,
        "weight is too large for the resident dgrad kernel",
    )

    k_tiles = K // P_MAX
    n_tiles = N // N_TILE
    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    kernel_assert(
        (M // P_MAX) % grid == 0,
        "M tiles must divide the SPMD grid",
    )

    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    grad_qscale = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    grad_inverse = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=grad_qscale,
        src=grad_scale_state[program_id, 0, :, :],
    )
    nisa.dma_copy(
        dst=grad_inverse,
        src=grad_scale_state[program_id, 1, :, :],
    )

    # Fold consecutive weight rows per partition for >= 12 KB DMA descriptors
    # (the amax is layout independent).
    w_fold = max(1, (6144 // N))
    kernel_assert(K % (w_fold * P_MAX) == 0, "weight rows must fold evenly")
    weight_view = weight.reshape((K // w_fold, w_fold * N))
    weight_amax = _tensorwise_amax_rows(weight_view, 0, K // w_fold, w_fold * N)
    weight_qscale, weight_inverse = _fp8_scale_from_amax(
        weight_amax,
        E4M3_MAX,
    )
    output_scale = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_tensor(
        dst=output_scale,
        data1=grad_inverse,
        data2=weight_inverse,
        op=nl.multiply,
    )

    weight_fp8 = nl.ndarray(
        (P_MAX, k_tiles, N),
        dtype=nl.float8_e4m3,
        buffer=nl.sbuf,
    )
    _quantize_dgrad_weight_segment(
        weight,
        weight_fp8,
        0,
        K,
        N,
        weight_qscale,
    )

    rows_per_program = (M // P_MAX) // grid
    _dgrad_pipelined_rows(
        grad_output,
        weight_fp8,
        out,
        K,
        N,
        nstep,
        grad_qscale,
        output_scale,
        grid,
        program_id,
        rows_per_program,
    )

    return out


def _fused_force_tile_v3(i: int, half_k: int, early_tile: int, early_gi: int, spread_gi0: int) -> int:
    """Weight tile whose clamped E4M3 bytes are tied into the gradient-scan chain after gradient tile ``i``
    (-1: none). One early hint (``early_tile`` at gradient tile ``early_gi``) pulls the weight scan / scale /
    ScalarE quantize forward; from ``spread_gi0`` on, the remaining tiles follow one per gradient tile (k order)
    so every VectorE clamp runs during the DMA-bound gradient scan instead of after it. The tie is an exact
    no-op on the running max (see the kernel)."""
    if i == early_gi:
        return early_tile
    if i < spread_gi0:
        return -1
    j = i - spread_gi0
    t = 0
    count = 0
    while t < half_k:
        if t != early_tile:
            if count == j:
                return t
            count = count + 1
        t = t + 1
    return -1


def _fused_force_last_gi(half_k: int, early_tile: int, early_gi: int, spread_gi0: int) -> int:
    """Gradient tile after which the last weight tile is forced."""
    n_spread = half_k - 1 if (early_tile >= 0 and early_tile < half_k) else half_k
    last = spread_gi0 + n_spread - 1
    if early_gi > last:
        last = early_gi
    return last


def _fused_nsplit_force_tile(i: int, n_forced: int, gi0: int, step: int) -> int:
    """N-split: weight tiles 0..n_forced-1 are forced at gradient tiles gi0, gi0+step, ... (-1: none)."""
    if i < gi0:
        return -1
    if (i - gi0) % step != 0:
        return -1
    t = (i - gi0) // step
    if t < n_forced:
        return t
    return -1


def _fused_partition_amax(merged, use_pe: bool):
    """Global max of a (128, 1) fp32 per-partition tile, returned in a (128, 1) tile whose partition 0 holds the
    result (like ``tensor_partition_reduce``, whose consumers only read partition 0). With
    FUSED_DGRAD_PE_PARTITION_REDUCE the cross-partition move is a bit-accurate Tensor Engine transpose into PSUM
    followed by a VectorE free-dim max (exact) instead of the GpSimd cross-lane reduce."""
    amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    if use_pe:
        row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=row, data=merged, engine=nisa.engine.tensor)
        nisa.tensor_reduce(dst=amax[0:1, 0:1], op=nl.maximum, data=row, axis=1, keepdims=True)
    else:
        nisa.tensor_partition_reduce(dst=amax, op=nl.maximum, data=merged)
    return amax


def _fused_weight_scale(running_w, program_id, use_pe: bool):
    """Swap the per-partition weight partial maxima with the other core (sendrecv), merge (exact max),
    partition-reduce and derive the E4M3 scale: identical on both cores."""
    other_w = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(
        src=running_w,
        dst=other_w,
        send_to_rank=1 - program_id,
        recv_from_rank=1 - program_id,
        pipe_id=1,
        dma_engine=nisa.dma_engine.gpsimd_dma,
    )
    merged_w = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=merged_w,
        data1=running_w,
        data2=other_w,
        op=nl.maximum,
    )
    weight_amax = _fused_partition_amax(merged_w, use_pe)
    return _fp8_scale_from_amax(weight_amax, E4M3_MAX)


def _dgrad_transpose_row_tile(grad_output, row_start, K: int, grad_raw):
    """Only the (transposed) load of one 128-row grad tile into ``grad_raw`` (the DMA part of
    ``_dgrad_quantize_row_tile_split``). With FUSED_DGRAD_GT_PRIORITY set, the transposes get that DMA
    QoS priority (0-3, lower = higher) so the E4M3 exchange (default priority) overtakes them on the rings."""
    k_tiles = K // P_MAX
    for k_tile in nl.static_range(k_tiles):
        k_start = k_tile * P_MAX
        if FUSED_DGRAD_GT_PRIORITY is None:
            nisa.dma_transpose(
                dst=grad_raw[:, k_tile, :],
                src=grad_output[
                    row_start:row_start + P_MAX,
                    k_start:k_start + P_MAX,
                ],
            )
        else:
            nisa.dma_transpose(
                dst=grad_raw[:, k_tile, :],
                src=grad_output[
                    row_start:row_start + P_MAX,
                    k_start:k_start + P_MAX,
                ],
                priority=FUSED_DGRAD_GT_PRIORITY,
            )


def _dgrad_quantize_row_tile_v3(grad_output, row_start, K: int, grad_qscale, grad_raw, grad_fp8, load: bool = True):
    """``_dgrad_quantize_row_tile`` (transposed load + E5M2 quantize + clamp, same instructions) with the
    transposes issued through ``_dgrad_transpose_row_tile`` (DMA priority knob)."""
    k_tiles = K // P_MAX
    if load:
        _dgrad_transpose_row_tile(grad_output, row_start, K, grad_raw)
    quantized_flat = grad_fp8.reshape((P_MAX, k_tiles * P_MAX))
    if False:
        pass
    else:
        raw_flat = grad_raw.reshape((P_MAX, k_tiles * P_MAX))
        nisa.activation(
            dst=quantized_flat,
            op=nl.copy,
            data=raw_flat,
            scale=grad_qscale,
        )
    nisa.tensor_scalar(
        dst=quantized_flat,
        data=quantized_flat,
        op0=nl.minimum,
        operand0=E5M2_MAX,
        op1=nl.maximum,
        operand1=-E5M2_MAX,
    )


def _dgrad_pe_load_transpose(grad_output, row_start, K: int, grad_raw):
    """Load one 128-row grad tile with a plain (12 KB-row) DMA into ``grad_raw``'s memory viewed as (128 rows, K)
    and transpose it in place block by block on the (idle) Tensor Engine: nc_transpose (bit-accurate on
    NeuronCore-v3+) into PSUM, then an exact bf16 copy back into ``grad_raw[:, k_tile, :]`` - the same bytes the
    DMA transposes would have produced, without 2 KB-packet transposes on the DMA rings after the scan."""
    k_tiles = K // P_MAX
    plain = grad_raw.reshape((P_MAX, K))
    nisa.dma_copy(dst=plain, src=grad_output[row_start:row_start + P_MAX, 0:K])
    for k_tile in nl.static_range(k_tiles):
        blk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(
            dst=blk,
            data=plain[:, k_tile * P_MAX:(k_tile + 1) * P_MAX],
            engine=nisa.engine.tensor,
        )
        nisa.tensor_copy(dst=grad_raw[:, k_tile, :], src=blk, engine=nisa.engine.scalar)


def _dgrad_pe_transpose_loaded(plain, K: int, grad_raw, copy_engine: str = "scalar"):
    """Transpose one already loaded plain (128 rows, K) gradient tile block by block on the Tensor Engine
    (nc_transpose, bit-accurate) into PSUM and copy each (128,128) block exactly (bf16 -> bf16) into
    ``grad_raw[:, k_tile, :]`` with ScalarE: the same bytes the DMA transposes would produce."""
    k_tiles = K // P_MAX
    for k_tile in nl.static_range(k_tiles):
        blk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(
            dst=blk,
            data=plain[:, k_tile * P_MAX:(k_tile + 1) * P_MAX],
            engine=nisa.engine.tensor,
        )
        if copy_engine == "vector":
            nisa.tensor_copy(dst=grad_raw[:, k_tile, :], src=blk, engine=nisa.engine.vector)
        else:
            nisa.tensor_copy(dst=grad_raw[:, k_tile, :], src=blk, engine=nisa.engine.scalar)


def _dgrad_quantize_row_tile_loaded(K: int, grad_qscale, grad_raw, grad_fp8):
    """E5M2 quantize + clamp of an already loaded tile (same instructions as ``_dgrad_quantize_row_tile``)."""
    k_tiles = K // P_MAX
    raw_flat = grad_raw.reshape((P_MAX, k_tiles * P_MAX))
    quantized_flat = grad_fp8.reshape((P_MAX, k_tiles * P_MAX))
    nisa.activation(
        dst=quantized_flat,
        op=nl.copy,
        data=raw_flat,
        scale=grad_qscale,
    )
    nisa.tensor_scalar(
        dst=quantized_flat,
        data=quantized_flat,
        op0=nl.minimum,
        operand0=E5M2_MAX,
        op1=nl.maximum,
        operand1=-E5M2_MAX,
    )


def _dgrad_quantize_row_tile_split_loaded(K: int, grad_qscale, grad_raw, grad_fp8, n_pieces: int = 2):
    """The piecewise quantize + clamp of ``_dgrad_quantize_row_tile_split`` for an already loaded tile: the k tiles
    are quantized in FUSED_DGRAD_Q0_PIECES contiguous groups (same per-element instructions; the first matmuls only
    wait for the first group)."""
    k_tiles = K // P_MAX
    per = k_tiles // n_pieces
    for h in nl.static_range(n_pieces):
        lo = h * per
        hi = k_tiles if h == n_pieces - 1 else (h + 1) * per
        nisa.activation(
            dst=grad_fp8[:, lo:hi, :],
            op=nl.copy,
            data=grad_raw[:, lo:hi, :],
            scale=grad_qscale,
        )
        nisa.tensor_scalar(
            dst=grad_fp8[:, lo:hi, :],
            data=grad_fp8[:, lo:hi, :],
            op0=nl.minimum,
            operand0=E5M2_MAX,
            op1=nl.maximum,
            operand1=-E5M2_MAX,
        )


def _dgrad_matmul_row_tile_v3(
    weight_fp8,
    out,
    row_start,
    K: int,
    N: int,
    col0,
    nstep: int,
    output_scale,
    grad_fp8,
    split_last_store: bool = False,
):
    """Resident dgrad GEMM for one quantized 128-row grad tile over output columns [col0, col0 + N)
    (same instruction sequence as ``_dgrad_matmul_row_tile``; ``weight_fp8`` holds these N columns).
    ``split_last_store``: the last group's dequant + store are issued per 512-column subtile (elementwise,
    identical values) so the stores overlap the remaining dequants at the kernel tail."""
    n_tiles = N // N_TILE
    n_groups = n_tiles // nstep
    for n_group in nl.affine_range(n_groups):
        n_start = n_group * nstep * N_TILE
        accum = nl.ndarray(
            (P_MAX, nstep * N_TILE),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for n_subtile in nl.affine_range(nstep):
            col_start = n_start + n_subtile * N_TILE
            for k_pair in nl.affine_range(K // DOUBLE_ROW_K):
                nisa.nc_matmul(
                    dst=accum[
                        :,
                        n_subtile * N_TILE:(n_subtile + 1) * N_TILE,
                    ],
                    stationary=grad_fp8[
                        :,
                        2 * k_pair:2 * k_pair + 2,
                        :,
                    ],
                    moving=weight_fp8[
                        :,
                        2 * k_pair:2 * k_pair + 2,
                        col_start:col_start + N_TILE,
                    ],
                    perf_mode=matmul_perf_mode.double_row,
                    accumulate=(k_pair > 0),
                )

        result = nl.ndarray(
            (P_MAX, nstep * N_TILE),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        if split_last_store and n_group == n_groups - 1:
            for n_subtile in nl.static_range(nstep):
                c_lo = n_subtile * N_TILE
                nisa.activation(
                    dst=result[:, c_lo:c_lo + N_TILE],
                    op=nl.copy,
                    data=accum[:, c_lo:c_lo + N_TILE],
                    scale=output_scale,
                )
                nisa.dma_copy(
                    dst=out[
                        row_start:row_start + P_MAX,
                        col0 + n_start + c_lo:col0 + n_start + c_lo + N_TILE,
                    ],
                    src=result[:, c_lo:c_lo + N_TILE],
                )
        else:
            nisa.activation(
                dst=result,
                op=nl.copy,
                data=accum,
                scale=output_scale,
            )
            nisa.dma_copy(
                dst=out[
                    row_start:row_start + P_MAX,
                    col0 + n_start:col0 + n_start + nstep * N_TILE,
                ],
                src=result,
            )


def _dgrad_pipelined_rows_v3(
    grad_output,
    weight_fp8,
    out,
    K: int,
    N: int,
    col0,
    nstep: int,
    grad_qscale,
    output_scale,
    grid,
    program_id,
    rows_per_program: int,
    grad_buffers,
    q1_after_q0: bool,
    q0_loaded: bool,
    q1_loaded: bool,
    row_starts=None,
    q0_pieces: int = 2,
):
    """Software-pipelined dgrad main loop (same structure as ``_dgrad_pipelined_rows``) over output columns
    [col0, col0 + N). ``q1_after_q0``: the E5M2 quantize of the second row tile takes its (bit-identical)
    scale from ``grad_qscale + 0.0`` where the 0.0 is reduced from the first tile's clamped bytes, so ScalarE
    cannot hoist q(1) above the two q(0) halves (which gate the first matmuls)."""
    k_tiles = K // P_MAX
    kernel_assert(rows_per_program >= 4, "need at least 4 row tiles per program")
    kernel_assert(rows_per_program % 2 == 0, "row tiles per program must be even")
    grad_raw_a, grad_fp8_a, grad_raw_b, grad_fp8_b = grad_buffers
    n_pairs = rows_per_program // 2
    if not q0_loaded:
        _dgrad_transpose_row_tile(grad_output, program_id * P_MAX, K, grad_raw_a)
    if FUSED_DGRAD_Q0_SPLIT:
        _dgrad_quantize_row_tile_split_loaded(K, grad_qscale, grad_raw_a, grad_fp8_a, q0_pieces)
    else:
        raw_flat0 = grad_raw_a.reshape((P_MAX, k_tiles * P_MAX))
        fp8_flat0 = grad_fp8_a.reshape((P_MAX, k_tiles * P_MAX))
        nisa.activation(dst=fp8_flat0, op=nl.copy, data=raw_flat0, scale=grad_qscale)
        nisa.tensor_scalar(dst=fp8_flat0, data=fp8_flat0, op0=nl.minimum, operand0=E5M2_MAX,
                           op1=nl.maximum, operand1=-E5M2_MAX)
    if q1_after_q0:
        # exact no-op on the scale: (x < -1e6) is false for every E5M2 value (|x| <= 57344) and for NaN,
        # so the reduced 0.0 leaves grad_qscale (> 0, finite) unchanged; NaN + 0.0 stays NaN.
        q1_junk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        q1_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar_reduce(
            dst=q1_junk,
            data=grad_fp8_a[:, k_tiles - 1, :],
            op0=nl.less,
            operand0=-1.0e6,
            reduce_op=nl.maximum,
            reduce_res=q1_zero,
        )
        grad_qscale_b = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=grad_qscale_b,
            data1=grad_qscale,
            data2=q1_zero,
            op=nl.add,
        )
    else:
        grad_qscale_b = grad_qscale
    for local_pair in nl.static_range(n_pairs - 1):
        if row_starts is None:
            row_a = ((2 * local_pair) * grid + program_id) * P_MAX
            row_b = ((2 * local_pair + 1) * grid + program_id) * P_MAX
            row_next_a = ((2 * local_pair + 2) * grid + program_id) * P_MAX
        else:
            row_a = row_starts[2 * local_pair]
            row_b = row_starts[2 * local_pair + 1]
            row_next_a = row_starts[2 * local_pair + 2]
        if q1_loaded and local_pair == 0:
            _dgrad_quantize_row_tile_loaded(K, grad_qscale_b, grad_raw_b, grad_fp8_b)
        elif FUSED_DGRAD_Q1_SPLIT and local_pair == 0:
            # second row tile quantized like the first (two k halves): same instruction shapes for A and B so the
            # compiler does not favour B's transposes over A's (probe of the emission order)
            _dgrad_transpose_row_tile(grad_output, row_b, K, grad_raw_b)
            _dgrad_quantize_row_tile_split_loaded(K, grad_qscale_b, grad_raw_b, grad_fp8_b)
        else:
            _dgrad_quantize_row_tile_v3(
                grad_output, row_b, K, grad_qscale_b, grad_raw_b, grad_fp8_b
            )
        _dgrad_matmul_row_tile_v3(
            weight_fp8,
            out,
            row_a,
            K,
            N,
            col0,
            nstep,
            output_scale,
            grad_fp8_a,
        )
        _dgrad_quantize_row_tile_v3(
            grad_output, row_next_a, K, grad_qscale, grad_raw_a, grad_fp8_a
        )
        _dgrad_matmul_row_tile_v3(
            weight_fp8,
            out,
            row_b,
            K,
            N,
            col0,
            nstep,
            output_scale,
            grad_fp8_b,
        )
    if row_starts is None:
        last_a = ((2 * (n_pairs - 1)) * grid + program_id) * P_MAX
        last_b = ((2 * (n_pairs - 1) + 1) * grid + program_id) * P_MAX
    else:
        last_a = row_starts[2 * (n_pairs - 1)]
        last_b = row_starts[2 * (n_pairs - 1) + 1]
    _dgrad_quantize_row_tile_v3(
        grad_output, last_b, K, grad_qscale_b, grad_raw_b, grad_fp8_b
    )
    _dgrad_matmul_row_tile_v3(
        weight_fp8,
        out,
        last_a,
        K,
        N,
        col0,
        nstep,
        output_scale,
        grad_fp8_a,
    )
    _dgrad_matmul_row_tile_v3(
        weight_fp8,
        out,
        last_b,
        K,
        N,
        col0,
        nstep,
        output_scale,
        grad_fp8_b,
        FUSED_DGRAD_SPLIT_LAST_STORE,
    )


@nki.jit
def tensorwise_fp8_grad_scale_state_dgrad_v1(
    grad_output,
    weight,
    nstep: int,
):
    """Fused ``grad_scale_state`` + ``dgrad_scaled``: returns ``(state, dx)``.

    Same math as running ``tensorwise_fp8_grad_scale_state_v1`` followed by
    ``tensorwise_fp8_linear_dgrad_scaled_v1`` (the returned ``state`` is the
    bitwise-identical (2, 2, 128, 1) scale state consumed by wgrad). Fusing
    lets the ScalarE-bound weight quantization overlap the DMA-bound gradient
    amax scan and saves one kernel launch. Two cross-core work splits:

    * N-split (wide outputs, N >= FUSED_DGRAD_NSPLIT_MIN_N): each LNC2 core
      owns N/2 output columns. It loads, abs-max-scans and E4M3-quantizes only
      weight[:, its columns] (fully resident) and runs the dgrad GEMM of ALL
      row tiles for its columns (same matmul count per core as a row split).
      No weight bytes cross the cores; only the (128, 1) partial maxima are
      swapped (``sendrecv``) so both cores derive the identical weight scale.
    * K-split (narrow outputs): each core loads, scans and quantizes HALF of
      the weight k tiles and sends the clamped E4M3 half to the other core's
      SBUF slots with one ``sendrecv``; both cores then run the row-split GEMM.

    In both splits each core scans half of the gradient rows (running abs-max,
    partial maxima swapped with ``sendrecv``) -> identical gradient scale and
    state on both cores. Every E4M3/E5M2 byte is produced by the same
    instruction sequence (ScalarE activation(copy, scale) + VectorE clamp)
    from the same input and scale regardless of which core computes it, and
    each dX element accumulates its k pairs in the same order in fp32 PSUM, so
    state and dX are bitwise identical to the separate kernels.

    Scheduling hints ("force"): the VectorE clamp of a weight tile is tied into
    the gradient-scan chain as an exact 0.0 (max over (x < -1000) of clamped
    E4M3 values) so the compiler runs the weight path during the scan instead
    of deferring it until after the scan.
    """
    kernel_assert(len(grad_output.shape) == 2, "grad_output must be rank 2")
    kernel_assert(len(weight.shape) == 2, "weight must be rank 2")
    M, K = grad_output.shape
    K_w, N = weight.shape
    kernel_assert(K == K_w, "dgrad contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "M must be divisible by 128")
    kernel_assert(K % DOUBLE_ROW_K == 0, "K must be divisible by 256")
    kernel_assert(N % N_TILE == 0, "N must be divisible by 512")
    kernel_assert(N % (nstep * N_TILE) == 0, "N must divide nstep * 512")
    kernel_assert(
        K * N <= RESIDENT_B_ELEMENTS,
        "weight is too large for the resident dgrad kernel",
    )
    k_tiles = K // P_MAX
    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    kernel_assert(grid == 2, "gradient scale sharing requires LNC2")
    kernel_assert((M // P_MAX) % 2 == 0, "row tiles must split across cores")
    half_rows = M // 2
    nsplit = N >= FUSED_DGRAD_NSPLIT_MIN_N
    pe_preduce = FUSED_DGRAD_NSPLIT_PE_PARTITION_REDUCE if nsplit else FUSED_DGRAD_PE_PARTITION_REDUCE
    if nsplit:
        kernel_assert((N // 2) % (nstep * N_TILE) == 0, "N/2 must divide nstep * 512 for the N-split")
        n_own = N // 2                 # output columns owned by this core
        col0 = program_id * n_own
        wk = k_tiles                   # weight k tiles this core loads / quantizes (its columns)
        my0 = 0
        th0 = 0
        w_row0 = 0
        half_k = k_tiles
    else:
        kernel_assert(k_tiles % 2 == 0, "weight k tiles must split across the two cores")
        half_k = k_tiles // 2
        n_own = N
        col0 = 0
        wk = half_k
        my0 = program_id * half_k
        th0 = (1 - program_id) * half_k
        w_row0 = my0 * P_MAX
    # weight "unit" = the quantize / clamp / force / exchange granularity: 1 k tile (double_row path) or 4 k tiles
    wu = 1
    kernel_assert(wk % wu == 0 and half_k % wu == 0, "weight k tiles must form whole 4-tile units")
    ku = k_tiles // wu
    wk_u = wk // wu
    my0_u = my0 // wu
    th0_u = th0 // wu
    half_u = half_k // wu

    state = nl.ndarray(
        (2, 2, P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.shared_hbm,
    )
    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # ---------------- explicit SBUF layout (bytes per partition) ----------------
    sbuf_top = (nl.tile_size.sbuf_fmax_bytes // 4096) * 4096
    raw_tile_bytes = k_tiles * P_MAX * 2
    fp8_tile_bytes = k_tiles * P_MAX
    addr_fp8_b = sbuf_top - fp8_tile_bytes
    addr_fp8_a = addr_fp8_b - fp8_tile_bytes
    addr_raw_b = addr_fp8_a - raw_tile_bytes
    # shared raw buffer: both row-tile slots transpose into the same 12 KB region (the E5M2 buffers stay separate), so
    # tile i+1's DMA transposes carry a write-after-read dependency on tile i's quantize and cannot be hoisted ahead of
    # it (keeps the rings free for the E4M3 exchange right after the scan); the pipeline is unchanged otherwise.
    shared_raw = FUSED_DGRAD_NSPLIT_SHARED_RAW if nsplit else FUSED_DGRAD_SHARED_RAW
    # gen3 (trn2, NeuronCore-v3) differences, verbatim from the validated trn2 port; the gen4 (trn3) trace is
    # unchanged:
    #   1. K-split SBUF layout: 212984 usable B/partition (sbuf_top 208896) leaves only 24576 B under
    #      FUSED_DGRAD_COMPILER_RESERVE with two raw buffers; sharing them frees 12288 B (same
    #      instruction stream otherwise).  The N-split layout fits as-is (43008 B left).
    #   2. dma_copy(priority=) is NeuronCore-v4 only (nki/isa/_validation.py) -> wload_priority = None
    #      below (FUSED_DGRAD_GT_PRIORITY / FUSED_DGRAD_GSCAN_PRIORITY already default to None).
    #   3. nisa.activate2 is gen4-only but unreachable here: the *_ACT2_* / GSCAN_REGS variant knobs
    #      default to 0/off, so no activate2 instruction is traced on either split.
    if not nsplit and nisa.get_nc_version() < nisa.nc_version.gen4:
        shared_raw = True
    addr_raw_a = addr_raw_b if shared_raw else addr_raw_b - raw_tile_bytes
    addr_wfp8 = addr_raw_a - k_tiles * n_own
    w_fold = max(1, (6144 // n_own))   # weight-scan chunk = w_fold tiles (>= 12 KB per partition)
    kernel_assert(wk % w_fold == 0, "weight tiles must fold evenly")
    rows_per_part = max(1, (6144 // K))
    g_width = rows_per_part * K
    # 12 KB gradient scan tiles (K = 1536): pin four scan buffers so the compiler cannot starve the scan
    # of scratch slots. For 3 KB-row gradients (K = 6144) the compiler's own rotation is better.
    n_gscan_pinned = FUSED_DGRAD_PINNED_GSCAN if rows_per_part > 1 else 0
    addr_gscan = addr_wfp8 - n_gscan_pinned * g_width * 2
    addr_wraw = addr_gscan - wk * n_own * 2
    kernel_assert(addr_wraw >= FUSED_DGRAD_COMPILER_RESERVE, "SBUF layout underflow")
    gscan_bufs = []
    for gb in nl.static_range(n_gscan_pinned):
        gscan_bufs.append(nl.ndarray((P_MAX, g_width), dtype=nl.bfloat16, buffer=nl.sbuf,
                                     address=(0, addr_gscan + gb * g_width * 2)))

    weight_fp8 = nl.ndarray(
        (P_MAX, k_tiles, n_own),
        dtype=nl.float8_e4m3,
        buffer=nl.sbuf,
        address=(0, addr_wfp8),
    )
    grad_raw_a = nl.ndarray(
        (P_MAX, k_tiles, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, addr_raw_a)
    )
    if shared_raw:
        grad_raw_b = grad_raw_a
    else:
        grad_raw_b = nl.ndarray(
            (P_MAX, k_tiles, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, addr_raw_b)
        )
    grad_fp8_a = nl.ndarray(
        (P_MAX, k_tiles, P_MAX), dtype=nl.float8_e5m2, buffer=nl.sbuf, address=(0, addr_fp8_a)
    )
    grad_fp8_b = nl.ndarray(
        (P_MAX, k_tiles, P_MAX), dtype=nl.float8_e5m2, buffer=nl.sbuf, address=(0, addr_fp8_b)
    )
    w_res = nl.ndarray(
        (P_MAX, wk, n_own), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, addr_wraw)
    )

    # ---------------- weight: this core's part -> SBUF, abs-max, cross-core merge ----------------
    wload_priority = FUSED_DGRAD_NSPLIT_WLOAD_PRIORITY if nsplit else FUSED_DGRAD_WLOAD_PRIORITY
    if nisa.get_nc_version() < nisa.nc_version.gen4:
        wload_priority = None   # dma_copy(priority=) is NeuronCore-v4 only (see the gen3 comment block above)
    wl_per = FUSED_DGRAD_NSPLIT_WLOAD_PER_GTILE if nsplit else 0   # weight tile loads issued per gradient-tile iteration
    n_wl_iters = 0 if wl_per == 0 else (wk + wl_per - 1) // wl_per
    for t in nl.static_range(0 if wl_per > 0 else wk):
        if FUSED_DGRAD_WLOAD_HWDGE:
            # hardware DGE (Sync/ScalarE sequencer) descriptors: no GpSimd descriptor-generation instruction per load
            if wload_priority is None:
                nisa.dma_copy(
                    dst=w_res[:, t, :],
                    src=weight[w_row0 + t * P_MAX:w_row0 + (t + 1) * P_MAX, col0:col0 + n_own],
                    dge_mode=nisa.dge_mode.hwdge,
                )
            else:
                nisa.dma_copy(
                    dst=w_res[:, t, :],
                    src=weight[w_row0 + t * P_MAX:w_row0 + (t + 1) * P_MAX, col0:col0 + n_own],
                    priority=wload_priority,
                    dge_mode=nisa.dge_mode.hwdge,
                )
        elif wload_priority is None:
            nisa.dma_copy(
                dst=w_res[:, t, :],
                src=weight[w_row0 + t * P_MAX:w_row0 + (t + 1) * P_MAX, col0:col0 + n_own],
            )
        else:
            nisa.dma_copy(
                dst=w_res[:, t, :],
                src=weight[w_row0 + t * P_MAX:w_row0 + (t + 1) * P_MAX, col0:col0 + n_own],
                priority=wload_priority,
            )
    running_w = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    per_partition_w = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    n_wchunks = wk // w_fold
    n_act2 = FUSED_DGRAD_WSCAN_ACT2_CHUNKS if nsplit else 0
    kernel_assert(n_act2 <= n_wchunks, "too many activate2 weight-scan chunks")
    if wl_per > 0:
        kernel_assert(n_act2 == n_wchunks or n_act2 == 0, "interleaved weight loads: scan all chunks on one engine")
        kernel_assert(wl_per % w_fold == 0, "interleaved weight loads must complete whole scan chunks")
    if n_act2 > 0 and wl_per == 0:
        # ScalarE share of the weight abs-max: in-place exact copy with the abs-max reduction registers
        # (fp32 max of bf16 magnitudes is exact and order independent -> identical amax)
        per_partition_w2 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for c in nl.static_range(n_act2):
            nisa.activate2(
                dst=w_res[:, c * w_fold:(c + 1) * w_fold, :],
                op=nl.copy,
                data=w_res[:, c * w_fold:(c + 1) * w_fold, :],
                imm0=0.0,
                imm1=0.0,
                op0=nl.bypass,
                op1=nl.bypass,
                reduce_op=nl.abs_max,
                reduce_res=(per_partition_w2 if c == n_act2 - 1 else None),
                reduce_cmd=(nisa.reduce_cmd.reset_reduce if c == 0 else nisa.reduce_cmd.reduce),
            )
        nisa.tensor_tensor(
            dst=running_w,
            data1=running_w,
            data2=per_partition_w2,
            op=nl.maximum,
        )
    use_lr = FUSED_DGRAD_NSPLIT_LOAD_REDUCE if nsplit else FUSED_DGRAD_LOAD_REDUCE
    if use_lr:
        # load_reduce ping-pong: running_w' = max(running_w, max|chunk|) in ONE VectorE instruction (no merge op)
        running_w_alt = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        rw_bufs = [running_w, running_w_alt]
        rw_idx = 0
    for c in nl.static_range(n_act2, 0 if wl_per > 0 else n_wchunks):
        abs_scratch = nl.ndarray((P_MAX, w_fold * n_own), dtype=nl.bfloat16, buffer=nl.sbuf)
        if use_lr:
            nisa.tensor_scalar_reduce(
                dst=abs_scratch,
                data=w_res[:, c * w_fold:(c + 1) * w_fold, :],
                op0=nl.abs,
                operand0=0.0,
                reduce_op=nl.maximum,
                reduce_res=rw_bufs[1 - rw_idx],
                reduce_cmd=nisa.reduce_cmd.load_reduce,
                reduce_init=rw_bufs[rw_idx],
            )
            rw_idx = 1 - rw_idx
        else:
            nisa.tensor_scalar_reduce(
                dst=abs_scratch,
                data=w_res[:, c * w_fold:(c + 1) * w_fold, :],
                op0=nl.abs,
                operand0=0.0,
                reduce_op=nl.maximum,
                reduce_res=per_partition_w,
            )
            nisa.tensor_tensor(
                dst=running_w,
                data1=running_w,
                data2=per_partition_w,
                op=nl.maximum,
            )
    if use_lr:
        kernel_assert(wl_per == 0, "load_reduce weight scan needs up-front weight loads")
        running_w = rw_bufs[rw_idx]
    if wl_per == 0:
        weight_qscale, weight_inverse = _fused_weight_scale(running_w, program_id, pe_preduce)

    # ---------------- gradient half scan with the weight quantize (+ exchange) interleaved ----------------
    kernel_assert(half_rows % (rows_per_part * P_MAX) == 0, "rows must fold evenly")
    grad_view = grad_output.reshape((M // rows_per_part, rows_per_part * K))
    g_tiles = (half_rows // rows_per_part) // P_MAX
    g_row0 = program_id * (half_rows // rows_per_part)
    # weight tiles quantized after gradient tile i: [i*wq_per, (i+1)*wq_per). Narrow tiles (1.4 us ScalarE):
    # one per gradient tile; wide tiles (>= 2.6 us): all right away (ScalarE is idle during the scan).
    wq_per = 1 if n_own <= 1536 else wk_u
    n_wq_tiles = (wk_u + wq_per - 1) // wq_per
    wq_start = n_wl_iters   # first gradient-tile iteration that quantizes weight tiles (after the scale exists)
    kernel_assert(wq_start + n_wq_tiles <= g_tiles, "not enough gradient tiles to hide the weight quantize")
    running = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    per_partition = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    n_gact2 = FUSED_DGRAD_NSPLIT_GSCAN_ACT2_TILES if nsplit else 0
    kernel_assert(n_gact2 < g_tiles, "too many activate2 gradient-scan tiles")
    kernel_assert(n_gact2 == 0 or n_act2 == 0, "activate2 reduce registers: weight and gradient shares are exclusive")
    per_partition_g2 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    gscan_regs = FUSED_DGRAD_NSPLIT_GSCAN_REGS if nsplit else False
    if gscan_regs:
        kernel_assert(n_gact2 == 0, "register-accumulated gradient scan excludes the ScalarE share")
        kernel_assert(n_act2 == n_wchunks, "register-accumulated gradient scan needs the weight scan on ScalarE")
    wscale_hint_gi = FUSED_DGRAD_WSCALE_HINT_GI if nsplit else -1
    q0_prefetch_gi = -1 if nsplit else FUSED_DGRAD_Q0_PREFETCH_GI
    n_pe = 0 if nsplit else FUSED_DGRAD_PE_TRANSPOSE_TILES
    if n_pe > 0:
        kernel_assert(q0_prefetch_gi < 0, "PE transposes and DMA prefetch of the first tile are exclusive")
        kernel_assert(FUSED_DGRAD_PE_GI0 + FUSED_DGRAD_PE_GI_STEP + FUSED_DGRAD_PE_FORCE_DELAY < g_tiles, "PE transpose schedule exceeds the scan")
        kernel_assert(FUSED_DGRAD_PE_FORCE_DELAY != FUSED_DGRAD_PE_GI_STEP, "PE transpose forces must not coincide")
    n_early_clamp = wk_u
    if nsplit and FUSED_DGRAD_NSPLIT_LATE_CLAMPS:
        n_early_clamp = FUSED_DGRAD_NSPLIT_FORCE_TILES
    # weight tiles 0..n_gclamp-1 are clamped on GpSimd (byte-identical min/max on every E4M3 pattern incl. NaN/Inf,
    # probed exhaustively) so VectorE keeps its time for the gradient reduces
    n_gclamp = FUSED_DGRAD_NSPLIT_GPSIMD_CLAMPS if nsplit else FUSED_DGRAD_KSPLIT_GPSIMD_CLAMPS
    kernel_assert(n_gclamp <= wk_u, "too many GpSimd clamps")
    # PE-transposed first row tiles taken from the scan's own plain tiles (K-split only)
    pe_a_gi = -1 if nsplit else FUSED_DGRAD_SCAN_PE_A_GI
    pe_b_gi = -1 if nsplit else FUSED_DGRAD_SCAN_PE_B_GI
    if pe_a_gi >= 0 or pe_b_gi >= 0:
        kernel_assert(n_pe == 0 and q0_prefetch_gi < 0, "scan-tile PE transposes exclude the other first-tile paths")
        kernel_assert(rows_per_part == 1, "scan-tile PE transposes need one gradient row per partition")
        kernel_assert(pe_a_gi >= 0, "the first row tile must be the PE-transposed one")
        kernel_assert(pe_a_gi != pe_b_gi, "distinct scan tiles for A and B")
        kernel_assert(2 * g_width * 2 <= wk * n_own * 2, "plain scan tiles must fit in the raw weight region")
        kernel_assert(pe_a_gi >= n_wq_tiles + 4 and (pe_b_gi < 0 or pe_b_gi >= n_wq_tiles + 4),
                      "plain tiles reuse the raw weight region: only after the weight quantize")
        kernel_assert(FUSED_DGRAD_SCAN_PE_FORCE_DELAY < 0 or pe_a_gi + FUSED_DGRAD_SCAN_PE_FORCE_DELAY < g_tiles, "PE force exceeds the scan")
        kernel_assert(FUSED_DGRAD_SCAN_PE_FORCE_DELAY < 0 or pe_b_gi < 0 or pe_b_gi + FUSED_DGRAD_SCAN_PE_FORCE_DELAY < g_tiles, "PE force exceeds the scan")
        plain_a = nl.ndarray((P_MAX, g_width), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, addr_wraw))
        plain_b = nl.ndarray((P_MAX, g_width), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, addr_wraw + g_width * 2))
    if nsplit:
        wx_gi = -1
    else:
        last_force_gi = _fused_force_last_gi(half_k, FUSED_DGRAD_WCLAMP_EARLY_TILE, FUSED_DGRAD_WCLAMP_EARLY_GI,
                                             FUSED_DGRAD_WCLAMP_SPREAD_GI0)
        kernel_assert(last_force_gi < g_tiles, "force schedule exceeds the gradient scan")
        # gradient tile after which the E4M3 half is exchanged
        wx_gi = g_tiles - 1 - FUSED_DGRAD_WX_GI_BACK
        if wx_gi < last_force_gi:
            wx_gi = last_force_gi
        if wx_gi < n_wq_tiles - 1:
            wx_gi = n_wq_tiles - 1
    gsr_in_loop = (not nsplit) and FUSED_DGRAD_WX_ANCHOR and (FUSED_DGRAD_WX_ANCHOR_SRC == "send_other" or FUSED_DGRAD_GSR_BEFORE_WX)
    lt_split = (FUSED_DGRAD_NSPLIT_LAST_TILE_SPLIT if nsplit else FUSED_DGRAD_LAST_TILE_SPLIT)
    if lt_split:
        kernel_assert(n_gact2 == 0 and not gscan_regs, "last-tile split excludes the activate2 / register scan variants")
        kernel_assert((g_width // 2) % P_MAX == 0, "half tiles must be whole k blocks")
    if use_lr:
        kernel_assert(n_gact2 == 0 and not gscan_regs, "load_reduce scan excludes the activate2 / register variants")
        kernel_assert(FUSED_DGRAD_SCAN_PE_A_GI < 0 or FUSED_DGRAD_SCAN_PE_FORCE_DELAY < 0, "load_reduce scan excludes the PE anti-sink hint")
        running_alt = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        rg_bufs = [running, running_alt]
        rg_idx = 0
    if gsr_in_loop:
        kernel_assert(wx_gi == g_tiles - 1, "the send_other anchor needs the exchange on the last gradient tile")
    for i in nl.static_range(g_tiles):
        row_start = g_row0 + i * P_MAX
        if i == pe_a_gi:
            gscan = plain_a
        elif i == pe_b_gi:
            gscan = plain_b
        elif n_gscan_pinned > 0:
            gscan = gscan_bufs[i % n_gscan_pinned]
        else:
            gscan = nl.ndarray((P_MAX, g_width), dtype=nl.bfloat16, buffer=nl.sbuf)
        if lt_split and i == g_tiles - 1:
            for hh in nl.static_range(2):
                nisa.dma_copy(
                    dst=gscan[:, hh * (g_width // 2):(hh + 1) * (g_width // 2)],
                    src=grad_view[row_start:row_start + P_MAX, hh * (g_width // 2):(hh + 1) * (g_width // 2)],
                )
        elif FUSED_DGRAD_GSCAN_PRIORITY is None:
            nisa.dma_copy(
                dst=gscan,
                src=grad_view[row_start:row_start + P_MAX, 0:g_width],
            )
        else:
            nisa.dma_copy(
                dst=gscan,
                src=grad_view[row_start:row_start + P_MAX, 0:g_width],
                priority=FUSED_DGRAD_GSCAN_PRIORITY,
            )
        if i < n_gact2:
            # ScalarE share of the gradient abs-max (ScalarE is idle before the weight quantize): in-place exact
            # copy with the abs-max reduction registers; the per-partition result is merged once (max is exact)
            nisa.activate2(
                dst=gscan,
                op=nl.copy,
                data=gscan,
                imm0=0.0,
                imm1=0.0,
                op0=nl.bypass,
                op1=nl.bypass,
                reduce_op=nl.abs_max,
                reduce_res=(per_partition_g2 if i == n_gact2 - 1 else None),
                reduce_cmd=(nisa.reduce_cmd.reset_reduce if i == 0 else nisa.reduce_cmd.reduce),
            )
            if i == n_gact2 - 1:
                nisa.tensor_tensor(
                    dst=running,
                    data1=running,
                    data2=per_partition_g2,
                    op=nl.maximum,
                )
        elif gscan_regs:
            # VectorE reduction registers accumulate the per-partition abs-max across all gradient tiles (fp32 max
            # of bf16 magnitudes: exact, order independent); read out once after the last tile -> no per-tile
            # reduce_res write / running-max merge on the in-order VectorE queue.
            nisa.tensor_scalar_reduce(
                dst=gscan,
                data=gscan,
                op0=nl.abs,
                operand0=0.0,
                reduce_op=nl.maximum,
                reduce_res=(running if i == g_tiles - 1 else None),
                reduce_cmd=(nisa.reduce_cmd.reset_reduce if i == 0 else nisa.reduce_cmd.reduce),
            )
        elif use_lr:
            if lt_split and i == g_tiles - 1:
                # last tile in two halves: the final reduce ends one half-tile after the last bytes land
                for hh in nl.static_range(2):
                    nisa.tensor_scalar_reduce(
                        dst=gscan[:, hh * (g_width // 2):(hh + 1) * (g_width // 2)],
                        data=gscan[:, hh * (g_width // 2):(hh + 1) * (g_width // 2)],
                        op0=nl.abs,
                        operand0=0.0,
                        reduce_op=nl.maximum,
                        reduce_res=rg_bufs[1 - rg_idx],
                        reduce_cmd=nisa.reduce_cmd.load_reduce,
                        reduce_init=rg_bufs[rg_idx],
                    )
                    rg_idx = 1 - rg_idx
            else:
                nisa.tensor_scalar_reduce(
                    dst=gscan,
                    data=gscan,
                    op0=nl.abs,
                    operand0=0.0,
                    reduce_op=nl.maximum,
                    reduce_res=rg_bufs[1 - rg_idx],
                    reduce_cmd=nisa.reduce_cmd.load_reduce,
                    reduce_init=rg_bufs[rg_idx],
                )
                rg_idx = 1 - rg_idx
        elif lt_split and i == g_tiles - 1:
            # last tile in two halves (plain path): reduce + merge per half; the final merge ends one half-tile after landing
            for hh in nl.static_range(2):
                nisa.tensor_scalar_reduce(
                    dst=gscan[:, hh * (g_width // 2):(hh + 1) * (g_width // 2)],
                    data=gscan[:, hh * (g_width // 2):(hh + 1) * (g_width // 2)],
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
        else:
            nisa.tensor_scalar_reduce(
                dst=gscan,
                data=gscan,
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
        if wl_per > 0 and i < n_wl_iters:
            # interleaved weight-tile loads: this core's tiles [i*wl_per, (i+1)*wl_per) right behind gradient tile i
            for j in nl.static_range(wl_per):
                t = i * wl_per + j
                if t < wk:
                    if wload_priority is None:
                        nisa.dma_copy(
                            dst=w_res[:, t, :],
                            src=weight[w_row0 + t * P_MAX:w_row0 + (t + 1) * P_MAX, col0:col0 + n_own],
                        )
                    else:
                        nisa.dma_copy(
                            dst=w_res[:, t, :],
                            src=weight[w_row0 + t * P_MAX:w_row0 + (t + 1) * P_MAX, col0:col0 + n_own],
                            priority=wload_priority,
                        )
            # weight-scan chunks completed by these loads
            c_lo = (i * wl_per) // w_fold
            c_hi = min(n_wchunks, ((i + 1) * wl_per) // w_fold)
            for c in nl.static_range(c_lo, c_hi):
                if n_act2 > 0:
                    nisa.activate2(
                        dst=w_res[:, c * w_fold:(c + 1) * w_fold, :],
                        op=nl.copy,
                        data=w_res[:, c * w_fold:(c + 1) * w_fold, :],
                        imm0=0.0,
                        imm1=0.0,
                        op0=nl.bypass,
                        op1=nl.bypass,
                        reduce_op=nl.abs_max,
                        reduce_res=(per_partition_w if c == n_wchunks - 1 else None),
                        reduce_cmd=(nisa.reduce_cmd.reset_reduce if c == 0 else nisa.reduce_cmd.reduce),
                    )
                    if c == n_wchunks - 1:
                        nisa.tensor_tensor(
                            dst=running_w,
                            data1=running_w,
                            data2=per_partition_w,
                            op=nl.maximum,
                        )
                else:
                    abs_scratch = nl.ndarray((P_MAX, w_fold * n_own), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_scalar_reduce(
                        dst=abs_scratch,
                        data=w_res[:, c * w_fold:(c + 1) * w_fold, :],
                        op0=nl.abs,
                        operand0=0.0,
                        reduce_op=nl.maximum,
                        reduce_res=per_partition_w,
                    )
                    nisa.tensor_tensor(
                        dst=running_w,
                        data1=running_w,
                        data2=per_partition_w,
                        op=nl.maximum,
                    )
            if i == n_wl_iters - 1:
                weight_qscale, weight_inverse = _fused_weight_scale(running_w, program_id, pe_preduce)
        if i >= wq_start and i < wq_start + n_wq_tiles:
            for j in nl.static_range(wq_per):
                t = (i - wq_start) * wq_per + j
                if t < wk_u:
                    nisa.activation(
                        dst=weight_fp8[:, my0_u + t, :],
                        op=nl.copy,
                        data=w_res[:, t, :],
                        scale=weight_qscale,
                    )
                    if t < n_early_clamp:
                        if t < n_gclamp:
                            nisa.tensor_scalar(
                                dst=weight_fp8[:, my0_u + t, :],
                                data=weight_fp8[:, my0_u + t, :],
                                op0=nl.minimum,
                                operand0=E4M3_MAX,
                                op1=nl.maximum,
                                operand1=-E4M3_MAX,
                                engine=nisa.engine.gpsimd,
                            )
                        else:
                            nisa.tensor_scalar(
                                dst=weight_fp8[:, my0_u + t, :],
                                data=weight_fp8[:, my0_u + t, :],
                                op0=nl.minimum,
                                operand0=E4M3_MAX,
                                op1=nl.maximum,
                                operand1=-E4M3_MAX,
                            )
        if i == wscale_hint_gi:
            # scheduling hint (exact no-op): running = max(running, 0.0) with the 0.0 reduced from
            # (weight_qscale < -1000) -> prioritizes the weight scan / partial swap / scale chain
            hint_junk = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            hint_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            if gscan_regs:
                # accumulate the exact 0.0 into the running-max registers (no-op)
                nisa.tensor_scalar_reduce(
                    dst=hint_junk,
                    data=weight_qscale,
                    op0=nl.less,
                    operand0=-1000.0,
                    reduce_op=nl.maximum,
                    reduce_cmd=nisa.reduce_cmd.reduce,
                )
            elif use_lr:
                nisa.tensor_scalar_reduce(
                    dst=hint_junk,
                    data=weight_qscale,
                    op0=nl.less,
                    operand0=-1000.0,
                    reduce_op=nl.maximum,
                    reduce_res=rg_bufs[1 - rg_idx],
                    reduce_cmd=nisa.reduce_cmd.load_reduce,
                    reduce_init=rg_bufs[rg_idx],
                )
                rg_idx = 1 - rg_idx
            else:
                nisa.tensor_scalar_reduce(
                    dst=hint_junk,
                    data=weight_qscale,
                    op0=nl.less,
                    operand0=-1000.0,
                    reduce_op=nl.maximum,
                    reduce_res=hint_zero,
                )
                nisa.tensor_tensor(
                    dst=running,
                    data1=running,
                    data2=hint_zero,
                    op=nl.maximum,
                )
        if nsplit:
            ft = _fused_nsplit_force_tile(i, FUSED_DGRAD_NSPLIT_FORCE_TILES, FUSED_DGRAD_NSPLIT_FORCE_GI0,
                                          FUSED_DGRAD_NSPLIT_FORCE_STEP)
            if ft >= 0:
                ft = ft + n_gclamp          # forces target VectorE-clamped tiles
                if ft >= wk_u:
                    ft = -1
        else:
            ft = _fused_force_tile_v3(i, half_k, FUSED_DGRAD_WCLAMP_EARLY_TILE, FUSED_DGRAD_WCLAMP_EARLY_GI,
                                      FUSED_DGRAD_WCLAMP_SPREAD_GI0)
        if ft >= 0 and gscan_regs and i == g_tiles - 1:
            ft = -1   # the registers were read out by this tile's reduce; a later accumulate would be lost (harmless but pointless)
        if ft >= 0:
            # scheduling hint (exact no-op): running = max(running, 0.0) where the 0.0 is the max over
            # (x < -1000) of 128 clamped E4M3 values of weight tile ft (0.0 for every value incl. NaN).
            force_junk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            force_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            if gscan_regs:
                nisa.tensor_scalar_reduce(
                    dst=force_junk,
                    data=weight_fp8[:, my0_u + ft, 0:P_MAX],
                    op0=nl.less,
                    operand0=-1000.0,
                    reduce_op=nl.maximum,
                    reduce_cmd=nisa.reduce_cmd.reduce,
                )
            elif use_lr:
                nisa.tensor_scalar_reduce(
                    dst=force_junk,
                    data=weight_fp8[:, my0_u + ft, 0:P_MAX],
                    op0=nl.less,
                    operand0=-1000.0,
                    reduce_op=nl.maximum,
                    reduce_res=rg_bufs[1 - rg_idx],
                    reduce_cmd=nisa.reduce_cmd.load_reduce,
                    reduce_init=rg_bufs[rg_idx],
                )
                rg_idx = 1 - rg_idx
            else:
                nisa.tensor_scalar_reduce(
                    dst=force_junk,
                    data=weight_fp8[:, my0_u + ft, 0:P_MAX],
                    op0=nl.less,
                    operand0=-1000.0,
                    reduce_op=nl.maximum,
                    reduce_res=force_zero,
                )
                nisa.tensor_tensor(
                    dst=running,
                    data1=running,
                    data2=force_zero,
                    op=nl.maximum,
                )
        if i == pe_a_gi:
            _dgrad_pe_transpose_loaded(plain_a, K, grad_raw_a)
            if pe_b_gi >= 0 and pe_b_gi < pe_a_gi:
                # B's plain tile landed earlier; its transposes + copies are emitted right behind A's so the in-order
                # Tensor Engine queue finishes A (which gates the first matmuls) first
                _dgrad_pe_transpose_loaded(plain_b, K, grad_raw_b, FUSED_DGRAD_SCAN_PE_B_COPY_ENGINE)
        if i == pe_b_gi and pe_b_gi > pe_a_gi:
            _dgrad_pe_transpose_loaded(plain_b, K, grad_raw_b, FUSED_DGRAD_SCAN_PE_B_COPY_ENGINE)
        if FUSED_DGRAD_SCAN_PE_FORCE_DELAY >= 0 and ((pe_a_gi >= 0 and i == pe_a_gi + FUSED_DGRAD_SCAN_PE_FORCE_DELAY) or (pe_b_gi >= 0 and i == pe_b_gi + FUSED_DGRAD_SCAN_PE_FORCE_DELAY)):
            # anti-sink hint (exact no-op): running = max(running, 0.0) with the 0.0 reduced from (x > +inf) over the
            # last transposed block (false for EVERY bf16 value: finite, +-inf and NaN), so the compiler finishes the
            # PE transposes + copies during the scan instead of after it.
            pe_raw = grad_raw_a if i == pe_a_gi + FUSED_DGRAD_SCAN_PE_FORCE_DELAY else grad_raw_b
            pe_junk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            pe_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(
                dst=pe_junk,
                data=pe_raw[:, k_tiles - 1, :],
                op0=nl.greater,
                operand0=FUSED_DGRAD_PE_FORCE_BOUND,
                reduce_op=nl.maximum,
                reduce_res=pe_zero,
            )
            nisa.tensor_tensor(
                dst=running,
                data1=running,
                data2=pe_zero,
                op=nl.maximum,
            )
        if (not nsplit) and n_pe >= 1 and i == FUSED_DGRAD_PE_GI0:
            _dgrad_pe_load_transpose(grad_output, program_id * P_MAX, K, grad_raw_a)
        if (not nsplit) and n_pe >= 2 and i == FUSED_DGRAD_PE_GI0 + FUSED_DGRAD_PE_GI_STEP:
            _dgrad_pe_load_transpose(grad_output, (grid + program_id) * P_MAX, K, grad_raw_b)
        if (not nsplit) and n_pe >= 1 and (i == FUSED_DGRAD_PE_GI0 + FUSED_DGRAD_PE_FORCE_DELAY or (n_pe >= 2 and i == FUSED_DGRAD_PE_GI0 + FUSED_DGRAD_PE_GI_STEP + FUSED_DGRAD_PE_FORCE_DELAY)):
            # anti-sink hint (exact no-op): running = max(running, 0.0) with the 0.0 reduced from (x < -1000) over
            # the last transposed block of the tile (false for every bf16 value incl. NaN), so the compiler finishes
            # the PE transposes during the scan instead of after it.
            pe_raw = grad_raw_a if i == FUSED_DGRAD_PE_GI0 + FUSED_DGRAD_PE_FORCE_DELAY else grad_raw_b
            pe_junk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            pe_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(
                dst=pe_junk,
                data=pe_raw[:, k_tiles - 1, :],
                op0=nl.less,
                operand0=-1000.0,
                reduce_op=nl.maximum,
                reduce_res=pe_zero,
            )
            nisa.tensor_tensor(
                dst=running,
                data1=running,
                data2=pe_zero,
                op=nl.maximum,
            )
        if i == q0_prefetch_gi:
            # issue the first row tile's gradient transposes now (DMA rings are HBM-bound during the scan, so
            # this costs ~1.5 MB of scan time) instead of after the scan where they would sit in front of the
            # E4M3 exchange and the first quantize on the rings
            _dgrad_transpose_row_tile(grad_output, program_id * P_MAX, K, grad_raw_a)
        if gsr_in_loop and i == g_tiles - 1:
            if use_lr:
                running = rg_bufs[rg_idx]
            # gradient partial swap + merge issued here (before the exchange in program order) so the E4M3 exchange's
            # send anchors can depend on the received partial: the GpSimd sendrecv handshakes of the exchange then
            # queue behind the gradient swap instead of delaying it
            other = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.sendrecv(
                src=running,
                dst=other,
                send_to_rank=1 - program_id,
                recv_from_rank=1 - program_id,
                pipe_id=0,
                dma_engine=nisa.dma_engine.gpsimd_dma,
            )
            merged = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=merged,
                data1=running,
                data2=other,
                op=nl.maximum,
            )
        if i == wx_gi:
            # E4M3 half weight exchange: this core's tiles -> the other core's slots (SBUF -> SBUF DMA, default dma
            # engine with CoreBarrier sync; gpsimd_dma is limited to 256 B/partition per op), in FUSED_DGRAD_WX_CHUNKS
            # k-ordered chunks on distinct pipes so the other core's first k pairs can start after the first chunk.
            n_wx = FUSED_DGRAD_WX_CHUNKS
            kernel_assert(half_u % n_wx == 0, "exchange chunks must divide the half")
            wx_per = half_u // n_wx
            if FUSED_DGRAD_WX_ANCHOR and FUSED_DGRAD_WX_ANCHOR_SRC in ("send", "send_other"):
                # SEND anchor: a real data dependency of every exchange chunk's SOURCE on this (last) gradient tile: the first
                # 128 columns of the chunk's first k tile are re-clamped with the bound E4M3_MAX + 0.0, where the exact 0.0 is
                # the max over (x > +inf) of 128 values of this tile (false for every finite/inf/NaN value). The re-clamp of
                # already clamped E4M3 bytes is the identity (NaN was mapped to +240 by the first clamp), so the bytes and dX
                # are unchanged, but the sendrecv cannot be issued before this tile has landed.
                sa_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                if FUSED_DGRAD_WX_ANCHOR_SRC == "send_other":
                    # 0.0 from the RECEIVED gradient partial (finite >= 0 by construction; (x > +inf) is false for every value)
                    sa_junk1 = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_scalar_reduce(
                        dst=sa_junk1,
                        data=other,
                        op0=nl.greater,
                        operand0=FUSED_DGRAD_PE_FORCE_BOUND,
                        reduce_op=nl.maximum,
                        reduce_res=sa_zero,
                    )
                else:
                    sa_junk = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_scalar_reduce(
                        dst=sa_junk,
                        data=gscan[:, 0:P_MAX],
                        op0=nl.greater,
                        operand0=FUSED_DGRAD_PE_FORCE_BOUND,
                        reduce_op=nl.maximum,
                        reduce_res=sa_zero,
                    )
                sa_hi = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=sa_hi, data=sa_zero, op0=nl.add, operand0=E4M3_MAX)
                for c in nl.static_range(n_wx):
                    nisa.tensor_scalar(
                        dst=weight_fp8[:, my0_u + c * wx_per, 0:P_MAX],
                        data=weight_fp8[:, my0_u + c * wx_per, 0:P_MAX],
                        op0=nl.minimum,
                        operand0=sa_hi,
                        op1=nl.maximum,
                        operand1=-E4M3_MAX,
                    )
            for c in nl.static_range(n_wx):
                if FUSED_DGRAD_WX_ANCHOR and FUSED_DGRAD_WX_ANCHOR_SRC not in ("send", "send_other"):
                    # anchor: a throw-away write (overwritten by the exchange) into the receive slots that depends
                    # on this gradient tile keeps the compiler from hoisting the exchange into the DMA-bound scan.
                    if FUSED_DGRAD_WX_ANCHOR_SRC == "reduce":
                        # depend on this tile's VectorE reduce result instead of its raw bytes (fp32 -> e4m3 cast copy)
                        nisa.tensor_copy(dst=weight_fp8[:, th0_u + c * wx_per, 0:1], src=per_partition[:, 0:1])
                    else:
                        nisa.tensor_copy(dst=weight_fp8[:, th0_u + c * wx_per, 0:1], src=gscan[:, c:c + 1])
                nisa.sendrecv(
                    src=weight_fp8[:, my0_u + c * wx_per:my0_u + (c + 1) * wx_per, :],
                    dst=weight_fp8[:, th0_u + c * wx_per:th0_u + (c + 1) * wx_per, :],
                    send_to_rank=1 - program_id,
                    recv_from_rank=1 - program_id,
                    pipe_id=2 + c,
                )
                if FUSED_DGRAD_B_GATE_SENDRECV and c == n_wx - 1:
                    # DMA ordering: a tiny junk swap (one element of each k block of the second gradient buffer,
                    # overwritten by that tile's transposes) on the LAST exchange chunk's pipe gives B's 48 transposes
                    # a write-after-write dependency on the exchange, so on the rings the order becomes: first tile's
                    # transposes, E4M3 exchange, second tile's transposes (instead of both tiles ahead of the exchange).
                    nisa.sendrecv(
                        src=grad_raw_b[:, :, 0:1],
                        dst=grad_raw_b[:, :, 0:1],
                        send_to_rank=1 - program_id,
                        recv_from_rank=1 - program_id,
                        pipe_id=2 + c,
                    )

    if use_lr:
        running = rg_bufs[rg_idx]
    # ---------------- exchange partial maxima -> identical gradient scale on both cores ----------------
    # low-latency SBUF<->SBUF swap between the two LNC2 cores (no HBM round trip / core_barrier)
    if not gsr_in_loop:
        other = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.sendrecv(
            src=running,
            dst=other,
            send_to_rank=1 - program_id,
            recv_from_rank=1 - program_id,
            pipe_id=0,
            dma_engine=nisa.dma_engine.gpsimd_dma,
        )
        merged = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=merged,
            data1=running,
            data2=other,
            op=nl.maximum,
        )
    grad_amax = _fused_partition_amax(merged, pe_preduce)
    grad_qscale, grad_inverse = _fp8_scale_from_amax(
        grad_amax,
        E5M2_MAX,
    )
    nisa.dma_copy(dst=state[program_id, 0, :, :], src=grad_qscale)
    nisa.dma_copy(dst=state[program_id, 1, :, :], src=grad_inverse)
    output_scale = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_tensor(
        dst=output_scale,
        data1=grad_inverse,
        data2=weight_inverse,
        op=nl.multiply,
    )
    if n_early_clamp < wk_u:
        # late clamps (tiles only needed from k pair n_early_clamp/2 on): same min/max clamp, with the +E4M3_MAX
        # bound as a (128,1) fp32 tile = 0.0 + E4M3_MAX where the exact 0.0 is reduced from (grad_qscale < -1000),
        # so these run after the gradient scale chain instead of interleaving with it.
        late_junk = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        late_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar_reduce(
            dst=late_junk,
            data=grad_qscale,
            op0=nl.less,
            operand0=-1000.0,
            reduce_op=nl.maximum,
            reduce_res=late_zero,
        )
        late_hi = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=late_hi,
            data=late_zero,
            op0=nl.add,
            operand0=E4M3_MAX,
        )
        for t in nl.static_range(n_early_clamp, wk_u):
            nisa.tensor_scalar(
                dst=weight_fp8[:, my0_u + t, :],
                data=weight_fp8[:, my0_u + t, :],
                op0=nl.minimum,
                operand0=late_hi,
                op1=nl.maximum,
                operand1=-E4M3_MAX,
            )

    if (not nsplit) and FUSED_DGRAD_B_HOLD_WX >= 0:
        # DMA ordering: the second row tile's 48 gradient transposes (buffer B) get a write-after-read dependency on a
        # LIVE read of B's (not yet written) raw buffer that itself depends on the received E4M3 half (chunk
        # FUSED_DGRAD_B_HOLD_WX), so on the rings the exchange runs before B's transposes. Exact no-op on the numerics:
        # max over one element of (garbage * w > +inf) is 0.0 for every value (finite, +-inf, NaN) and output_scale
        # (a product of two positive finite inverses) + 0.0 == output_scale.
        hold_tile = th0 + FUSED_DGRAD_B_HOLD_WX * (half_k // FUSED_DGRAD_WX_CHUNKS)
        # the compiler tracks the 48 transposes per (128,128) block, so the live read must touch every k block of B
        hold_view = grad_raw_b[:, :, 0:1]
        nisa.memset(dst=hold_view, value=0.0)
        hold_w = nl.ndarray((P_MAX, k_tiles), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hold_w, src=weight_fp8[:, hold_tile, 0:k_tiles], engine=nisa.engine.vector)
        hold_prod = nl.ndarray((P_MAX, k_tiles), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=hold_prod,
            data1=hold_view.reshape((P_MAX, k_tiles)),
            data2=hold_w,
            op=nl.multiply,
        )
        hold_junk = nl.ndarray((P_MAX, k_tiles), dtype=nl.float32, buffer=nl.sbuf)
        hold_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar_reduce(
            dst=hold_junk,
            data=hold_prod,
            op0=nl.greater,
            operand0=FUSED_DGRAD_PE_FORCE_BOUND,
            reduce_op=nl.maximum,
            reduce_res=hold_zero,
        )
        nisa.tensor_tensor(
            dst=output_scale,
            data1=output_scale,
            data2=hold_zero,
            op=nl.add,
        )
    if (not nsplit) and FUSED_DGRAD_B_AFTER_GSR:
        # DMA ordering: hold the second row tile's gradient transposes (buffer B) behind the gradient-partial swap so
        # the first tile's transposes (which gate q(0)) get the DMA rings first. The dependency is carried by two
        # throw-away ScalarE ops (overwritten by the transposes): ScalarE has nothing to do until the scale is ready,
        # so the wait cannot stall the VectorE scale chain.
        b_gate = nl.ndarray((P_MAX, k_tiles), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=b_gate,
            data=w_res[:, 0, 0:k_tiles],
            op0=nl.add,
            operand0=other,
            engine=nisa.engine.scalar,
        )
        nisa.tensor_copy(
            dst=grad_raw_b[:, :, 0:1],
            src=b_gate.reshape((P_MAX, k_tiles, 1)),
            engine=nisa.engine.scalar,
        )
    if (not nsplit) and FUSED_DGRAD_B_AFTER_WX:
        # DMA ordering: the gradient transposes of the second row tile (buffer B) get a write-after-write
        # dependency on this throw-away copy (overwritten by the transposes), which itself needs the received
        # E4M3 half -> on the DMA rings the order becomes q(0) transposes, exchange, q(1) transposes instead
        # of both tiles' transposes ahead of the exchange (which gates the other core's first matmul).
        nisa.tensor_copy(dst=grad_raw_b[:, :, 0:1], src=weight_fp8[:, 0:k_tiles, 0:1])
    ksplit_rows = None
    if pe_a_gi >= 0:
        # blocked row split (core c owns rows [c*M/2, (c+1)*M/2) = exactly its scan rows): the PE-transposed scan
        # tiles come first, the remaining row tiles follow in order. Any row order gives bitwise identical dX.
        order = [pe_a_gi]
        if pe_b_gi >= 0:
            order.append(pe_b_gi)
        for r in range(g_tiles):
            if r != pe_a_gi and r != pe_b_gi:
                order.append(r)
        ksplit_rows = []
        for r in order:
            ksplit_rows.append((program_id * g_tiles + r) * P_MAX)
    if nsplit:
        _dgrad_pipelined_rows_v3(
            grad_output,
            weight_fp8,
            out,
            K,
            n_own,
            col0,
            nstep,
            grad_qscale,
            output_scale,
            1,
            0,
            M // P_MAX,
            (grad_raw_a, grad_fp8_a, grad_raw_b, grad_fp8_b),
            FUSED_DGRAD_Q1_AFTER_Q0,
            False,
            False,
            None,
            FUSED_DGRAD_NSPLIT_Q0_PIECES,
        )
    else:
        _dgrad_pipelined_rows_v3(
            grad_output,
            weight_fp8,
            out,
            K,
            N,
            0,
            nstep,
            grad_qscale,
            output_scale,
            grid,
            program_id,
            (M // P_MAX) // grid,
            ((grad_raw_b, grad_fp8_b, grad_raw_a, grad_fp8_a) if FUSED_DGRAD_SWAP_AB
             else (grad_raw_a, grad_fp8_a, grad_raw_b, grad_fp8_b)),
            FUSED_DGRAD_Q1_AFTER_Q0,
            q0_prefetch_gi >= 0 or n_pe >= 1 or pe_a_gi >= 0,
            n_pe >= 2 or pe_b_gi >= 0,
            ksplit_rows,
            FUSED_DGRAD_Q0_PIECES,
        )
    return state, out
