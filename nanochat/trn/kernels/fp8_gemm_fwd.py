"""FP8 forward GEMM kernels: tensorwise GEMM, fused forward+save, streamed GEMM (split from nki_fp8.py)."""

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
    EPS,
    N_TILE,
    P_MAX,
    RESIDENT_B_ELEMENTS,
    _broadcast_partition_zero,
    _fp8_scale_from_amax,
    kernel_assert,
)


# Opt in to non-SSA legalization (off by default for NKI kernels): c_proj 3.81 -> 2.46 ms at d24.
@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def tensorwise_fp8_gemm(
    a,
    b,
    a_e4m3: bool,
    b_e4m3: bool,
    nstep: int,
    mstep: int,
):
    """Compute BF16 ``a @ b`` with fused tensorwise quantization and dequantization.

    Dimensions:
        M: Rows of ``a``.
        K: Contraction dimension.
        N: Columns of ``b``.

    Args:
        a: BF16 [M, K] tensor in HBM.
        b: BF16 [K, N] tensor in HBM.
        a_e4m3: Select E4M3 for A; otherwise E5M2.
        b_e4m3: Select E4M3 for B; otherwise E5M2.
        nstep: Number of adjacent 512-column output tiles grouped per PSUM.
        mstep: Number of adjacent 128-row A tiles quantized together.

    Returns:
        BF16 [M, N] tensor in HBM.

    Notes:
        - Implements one dynamic tensorwise scale per operand.
        - Quantization math is FP32 and casts directly to FP8 before saturation.
        - Uses Trainium2 ``double_row`` FP8 matmul and FP32 accumulation.
        - B must fit resident in SBUF; wgrad uses the streamed kernel below.
    """
    kernel_assert(len(a.shape) == 2, "A must be rank 2")
    kernel_assert(len(b.shape) == 2, "B must be rank 2")
    M, K = a.shape
    K_b, N = b.shape
    kernel_assert(K == K_b, "A and B contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "M must be divisible by 128")
    kernel_assert(K % DOUBLE_ROW_K == 0, "K must be divisible by 256")
    kernel_assert(N % N_TILE == 0, "N must be divisible by 512")
    kernel_assert(N % (nstep * N_TILE) == 0, "N must divide nstep * 512")
    kernel_assert(M % (mstep * P_MAX) == 0, "M must divide mstep * 128")
    kernel_assert(
        K * N <= RESIDENT_B_ELEMENTS,
        "B is too large for the resident kernel",
    )

    a_fp8_dtype = nl.float8_e4m3 if a_e4m3 else nl.float8_e5m2
    b_fp8_dtype = nl.float8_e4m3 if b_e4m3 else nl.float8_e5m2
    a_fp8_max = E4M3_MAX if a_e4m3 else E5M2_MAX
    b_fp8_max = E4M3_MAX if b_e4m3 else E5M2_MAX

    k_tiles = K // P_MAX
    n_tiles = N // N_TILE
    wide_m = mstep * P_MAX
    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    kernel_assert(
        (M // wide_m) % grid == 0,
        "M tiles must divide the SPMD grid",
    )

    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Pass 1A: exact tensorwise abs-max of A.
    a_running = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    a_scan = nl.ndarray((P_MAX, K), dtype=nl.bfloat16, buffer=nl.sbuf)
    a_per_partition = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    for row_tile in nl.affine_range(M // P_MAX):
        nisa.dma_copy(
            dst=a_scan,
            src=a[
                row_tile * P_MAX:(row_tile + 1) * P_MAX,
                0:K,
            ],
        )
        nisa.tensor_scalar_reduce(
            dst=a_scan,
            data=a_scan,
            op0=nl.abs,
            operand0=0.0,
            reduce_op=nl.maximum,
            reduce_res=a_per_partition,
        )
        nisa.tensor_tensor(
            dst=a_running,
            data1=a_running,
            data2=a_per_partition,
            op=nl.maximum,
        )
    a_amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=a_amax, op=nl.maximum, data=a_running)

    # Pass 1B: exact tensorwise abs-max of B.
    b_running = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    b_scan = nl.ndarray((P_MAX, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    b_per_partition = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    for row_tile in nl.affine_range(K // P_MAX):
        nisa.dma_copy(
            dst=b_scan,
            src=b[
                row_tile * P_MAX:(row_tile + 1) * P_MAX,
                0:N,
            ],
        )
        nisa.tensor_scalar_reduce(
            dst=b_scan,
            data=b_scan,
            op0=nl.abs,
            operand0=0.0,
            reduce_op=nl.maximum,
            reduce_res=b_per_partition,
        )
        nisa.tensor_tensor(
            dst=b_running,
            data1=b_running,
            data2=b_per_partition,
            op=nl.maximum,
        )
    b_amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=b_amax, op=nl.maximum, data=b_running)

    # Match fp8.py's formula as closely as current FP32 NKI permits:
    # qscale = reciprocal(max(amax, EPS)) * fp8_max.
    # tensor_scalar does not support divide on this compiler build.
    a_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=a_clamped,
        data=a_amax,
        op0=nl.maximum,
        operand0=EPS,
    )
    a_reciprocal_zero = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.reciprocal(dst=a_reciprocal_zero, data=a_clamped)
    a_qscale_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=a_qscale_zero,
        data=a_reciprocal_zero,
        op0=nl.multiply,
        operand0=a_fp8_max,
    )
    a_qscale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _broadcast_partition_zero(a_qscale_zero, a_qscale)
    a_inverse = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=a_inverse, data=a_qscale)

    b_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=b_clamped,
        data=b_amax,
        op0=nl.maximum,
        operand0=EPS,
    )
    b_reciprocal_zero = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.reciprocal(dst=b_reciprocal_zero, data=b_clamped)
    b_qscale_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=b_qscale_zero,
        data=b_reciprocal_zero,
        op0=nl.multiply,
        operand0=b_fp8_max,
    )
    b_qscale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _broadcast_partition_zero(b_qscale_zero, b_qscale)
    b_inverse = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=b_inverse, data=b_qscale)

    output_scale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=output_scale,
        data1=a_inverse,
        data2=b_inverse,
        op=nl.multiply,
    )

    # Pass 2A: quantize B once and keep it resident for all output row tiles.
    b_fp8 = nl.ndarray(
        (P_MAX, k_tiles, N),
        dtype=b_fp8_dtype,
        buffer=nl.sbuf,
    )
    b_raw = nl.ndarray((P_MAX, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    for k_tile in nl.affine_range(k_tiles):
        nisa.dma_copy(
            dst=b_raw,
            src=b[
                k_tile * P_MAX:(k_tile + 1) * P_MAX,
                0:N,
            ],
        )
        nisa.activation(
            dst=b_fp8[:, k_tile, :],
            op=nl.copy,
            data=b_raw,
            scale=b_qscale,
        )
        nisa.tensor_scalar(
            dst=b_fp8[:, k_tile, :],
            data=b_fp8[:, k_tile, :],
            op0=nl.minimum,
            operand0=b_fp8_max,
            op1=nl.maximum,
            operand1=-b_fp8_max,
        )

    # Pass 2B: stream A row groups, quantize, matmul, and dequantize.
    row_groups_per_program = (M // wide_m) // grid
    for row_group in nl.affine_range(row_groups_per_program):
        row_start = (row_group * grid + program_id) * wide_m
        a_raw = nl.ndarray(
            (P_MAX, k_tiles, wide_m),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        for k_tile in nl.affine_range(k_tiles):
            nisa.dma_transpose(
                dst=a_raw[:, k_tile, :],
                src=a[
                    row_start:row_start + wide_m,
                    k_tile * P_MAX:(k_tile + 1) * P_MAX,
                ],
            )

        a_fp8 = nl.ndarray(
            (P_MAX, k_tiles, wide_m),
            dtype=a_fp8_dtype,
            buffer=nl.sbuf,
        )
        a_raw_flat = a_raw.reshape((P_MAX, k_tiles * wide_m))
        a_fp8_flat = a_fp8.reshape((P_MAX, k_tiles * wide_m))
        nisa.activation(
            dst=a_fp8_flat,
            op=nl.copy,
            data=a_raw_flat,
            scale=a_qscale,
        )
        nisa.tensor_scalar(
            dst=a_fp8_flat,
            data=a_fp8_flat,
            op0=nl.minimum,
            operand0=a_fp8_max,
            op1=nl.maximum,
            operand1=-a_fp8_max,
        )

        for m_tile in nl.affine_range(mstep):
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
                            stationary=a_fp8[
                                :,
                                2 * k_pair:2 * k_pair + 2,
                                m_tile * P_MAX:(m_tile + 1) * P_MAX,
                            ],
                            moving=b_fp8[
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
                output_row = row_start + m_tile * P_MAX
                nisa.dma_copy(
                    dst=out[
                        output_row:output_row + P_MAX,
                        n_start:n_start + nstep * N_TILE,
                    ],
                    src=result,
                )

    return out


@nki.jit
def tensorwise_fp8_gemm_streamed(
    a,
    b,
    a_e4m3: bool,
    b_e4m3: bool,
    kblock: int,
    a_transposed: bool = False,
):
    """Tensorwise FP8 GEMM for contractions whose B operand cannot reside in SBUF.

    This match-first path is intended for nanochat wgrad:

        dW = dY.T @ X

    B uses the conventional GEMM layout ``b[K, N]``. A is either conventional
    ``a[M, K]`` or, when ``a_transposed`` is true, its native transposed source
    layout ``a[K, M]``. The latter lets Linear wgrad consume ``dY[M, N]``
    directly and transpose each tile during its SBUF load.

    Each program owns interleaved 128-row output tiles. For one 512-column
    output tile, all of that program's FP32 partial sums remain in SBUF while
    2,048-wide contraction blocks are quantized and processed.
    """
    kernel_assert(len(a.shape) == 2, "A must be rank 2")
    kernel_assert(len(b.shape) == 2, "B must be rank 2")
    if a_transposed:
        K, M = a.shape
    else:
        M, K = a.shape
    K_b, N = b.shape
    kernel_assert(K == K_b, "A and B contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "M must be divisible by 128")
    kernel_assert(K % kblock == 0, "K must be divisible by kblock")
    kernel_assert(kblock % DOUBLE_ROW_K == 0, "kblock must divide into 256")
    kernel_assert(N % N_TILE == 0, "N must be divisible by 512")

    a_fp8_dtype = nl.float8_e4m3 if a_e4m3 else nl.float8_e5m2
    b_fp8_dtype = nl.float8_e4m3 if b_e4m3 else nl.float8_e5m2
    a_fp8_max = E4M3_MAX if a_e4m3 else E5M2_MAX
    b_fp8_max = E4M3_MAX if b_e4m3 else E5M2_MAX

    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    total_m_tiles = M // P_MAX
    kernel_assert(total_m_tiles % grid == 0, "M tiles must divide the grid")
    m_tiles_per_program = total_m_tiles // grid
    n_tiles = N // N_TILE
    k_tiles_per_block = kblock // P_MAX
    k_blocks = K // kblock

    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Pass 1A: dynamic tensorwise abs-max of A.
    a_running = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    a_scan_width = M if a_transposed else K
    a_scan = nl.ndarray(
        (P_MAX, a_scan_width),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    a_per_partition = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    a_scan_tiles = K // P_MAX if a_transposed else M // P_MAX
    for row_tile in nl.affine_range(a_scan_tiles):
        if a_transposed:
            nisa.dma_copy(
                dst=a_scan,
                src=a[
                    row_tile * P_MAX:(row_tile + 1) * P_MAX,
                    0:M,
                ],
            )
        else:
            nisa.dma_copy(
                dst=a_scan,
                src=a[
                    row_tile * P_MAX:(row_tile + 1) * P_MAX,
                    0:K,
                ],
            )
        nisa.tensor_scalar_reduce(
            dst=a_scan,
            data=a_scan,
            op0=nl.abs,
            operand0=0.0,
            reduce_op=nl.maximum,
            reduce_res=a_per_partition,
        )
        nisa.tensor_tensor(
            dst=a_running,
            data1=a_running,
            data2=a_per_partition,
            op=nl.maximum,
        )
    a_amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=a_amax, op=nl.maximum, data=a_running)

    # Pass 1B: dynamic tensorwise abs-max of B.
    b_running = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
    b_scan = nl.ndarray((P_MAX, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    b_per_partition = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    for row_tile in nl.affine_range(K // P_MAX):
        nisa.dma_copy(
            dst=b_scan,
            src=b[
                row_tile * P_MAX:(row_tile + 1) * P_MAX,
                0:N,
            ],
        )
        nisa.tensor_scalar_reduce(
            dst=b_scan,
            data=b_scan,
            op0=nl.abs,
            operand0=0.0,
            reduce_op=nl.maximum,
            reduce_res=b_per_partition,
        )
        nisa.tensor_tensor(
            dst=b_running,
            data1=b_running,
            data2=b_per_partition,
            op=nl.maximum,
        )
    b_amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_partition_reduce(dst=b_amax, op=nl.maximum, data=b_running)

    # qscale = reciprocal(max(amax, EPS)) * fp8_max.
    a_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=a_clamped,
        data=a_amax,
        op0=nl.maximum,
        operand0=EPS,
    )
    a_reciprocal_zero = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.reciprocal(dst=a_reciprocal_zero, data=a_clamped)
    a_qscale_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=a_qscale_zero,
        data=a_reciprocal_zero,
        op0=nl.multiply,
        operand0=a_fp8_max,
    )
    a_qscale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _broadcast_partition_zero(a_qscale_zero, a_qscale)
    a_inverse = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=a_inverse, data=a_qscale)

    b_clamped = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=b_clamped,
        data=b_amax,
        op0=nl.maximum,
        operand0=EPS,
    )
    b_reciprocal_zero = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.reciprocal(dst=b_reciprocal_zero, data=b_clamped)
    b_qscale_zero = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=b_qscale_zero,
        data=b_reciprocal_zero,
        op0=nl.multiply,
        operand0=b_fp8_max,
    )
    b_qscale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    _broadcast_partition_zero(b_qscale_zero, b_qscale)
    b_inverse = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=b_inverse, data=b_qscale)

    output_scale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=output_scale,
        data1=a_inverse,
        data2=b_inverse,
        op=nl.multiply,
    )

    # One output-column tile at a time. The complete set of output row-tile
    # accumulators owned by this program fits in SBUF for all nanochat wgrads.
    for n_tile in nl.affine_range(n_tiles):
        col_start = n_tile * N_TILE
        accumulators = nl.ndarray(
            (P_MAX, m_tiles_per_program, N_TILE),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.memset(
            dst=accumulators.reshape(
                (P_MAX, m_tiles_per_program * N_TILE)
            ),
            value=0.0,
        )

        b_fp8 = nl.ndarray(
            (P_MAX, k_tiles_per_block, N_TILE),
            dtype=b_fp8_dtype,
            buffer=nl.sbuf,
        )
        b_raw = nl.ndarray(
            (P_MAX, N_TILE),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )

        for k_block_index in nl.affine_range(k_blocks):
            k_start = k_block_index * kblock

            # Quantize one B contraction block and reuse it for every output
            # row tile owned by this program.
            for k_tile in nl.affine_range(k_tiles_per_block):
                k_row = k_start + k_tile * P_MAX
                nisa.dma_copy(
                    dst=b_raw,
                    src=b[
                        k_row:k_row + P_MAX,
                        col_start:col_start + N_TILE,
                    ],
                )
                nisa.activation(
                    dst=b_fp8[:, k_tile, :],
                    op=nl.copy,
                    data=b_raw,
                    scale=b_qscale,
                )
                nisa.tensor_scalar(
                    dst=b_fp8[:, k_tile, :],
                    data=b_fp8[:, k_tile, :],
                    op0=nl.minimum,
                    operand0=b_fp8_max,
                    op1=nl.maximum,
                    operand1=-b_fp8_max,
                )

            for local_m_tile in nl.affine_range(m_tiles_per_program):
                output_row = (
                    local_m_tile * grid + program_id
                ) * P_MAX
                a_raw = nl.ndarray(
                    (P_MAX, k_tiles_per_block, P_MAX),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                for k_tile in nl.affine_range(k_tiles_per_block):
                    k_col = k_start + k_tile * P_MAX
                    if a_transposed:
                        nisa.dma_copy(
                            dst=a_raw[:, k_tile, :],
                            src=a[
                                k_col:k_col + P_MAX,
                                output_row:output_row + P_MAX,
                            ],
                        )
                    else:
                        nisa.dma_transpose(
                            dst=a_raw[:, k_tile, :],
                            src=a[
                                output_row:output_row + P_MAX,
                                k_col:k_col + P_MAX,
                            ],
                        )

                a_fp8 = nl.ndarray(
                    (P_MAX, k_tiles_per_block, P_MAX),
                    dtype=a_fp8_dtype,
                    buffer=nl.sbuf,
                )
                a_raw_flat = a_raw.reshape(
                    (P_MAX, k_tiles_per_block * P_MAX)
                )
                a_fp8_flat = a_fp8.reshape(
                    (P_MAX, k_tiles_per_block * P_MAX)
                )
                nisa.activation(
                    dst=a_fp8_flat,
                    op=nl.copy,
                    data=a_raw_flat,
                    scale=a_qscale,
                )
                nisa.tensor_scalar(
                    dst=a_fp8_flat,
                    data=a_fp8_flat,
                    op0=nl.minimum,
                    operand0=a_fp8_max,
                    op1=nl.maximum,
                    operand1=-a_fp8_max,
                )

                partial = nl.ndarray(
                    (P_MAX, N_TILE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for k_pair in nl.affine_range(kblock // DOUBLE_ROW_K):
                    nisa.nc_matmul(
                        dst=partial,
                        stationary=a_fp8[
                            :,
                            2 * k_pair:2 * k_pair + 2,
                            :,
                        ],
                        moving=b_fp8[
                            :,
                            2 * k_pair:2 * k_pair + 2,
                            :,
                        ],
                        perf_mode=matmul_perf_mode.double_row,
                        accumulate=(k_pair > 0),
                    )
                nisa.tensor_tensor(
                    dst=accumulators[:, local_m_tile, :],
                    data1=accumulators[:, local_m_tile, :],
                    data2=partial,
                    op=nl.add,
                )

        for local_m_tile in nl.affine_range(m_tiles_per_program):
            output_row = (local_m_tile * grid + program_id) * P_MAX
            result = nl.ndarray(
                (P_MAX, N_TILE),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            nisa.activation(
                dst=result,
                op=nl.copy,
                data=accumulators[:, local_m_tile, :],
                scale=output_scale,
            )
            nisa.dma_copy(
                dst=out[
                    output_row:output_row + P_MAX,
                    col_start:col_start + N_TILE,
                ],
                src=result,
            )

    return out


FWD_RING_SLOTS_WIDE = 4
# gen3 (trn2): 4 wide ring slots + the quantized weight = 4 * 39936 + 73728 = 233472 B/partition at the
# c_proj shape, above the gen3 SBUF bound (nl.tile_size.sbuf_fmax_bytes = 212984).  3 slots fit in
# 193536 B and still leave >= 2 prologue scan buffers inside the ring region (2 slots would leave 0).
# Tile -> slot assignment and per-tile instruction content are ring-size independent, so outputs are
# bitwise identical to the 4-slot schedule.
FWD_RING_SLOTS_WIDE_G3 = 3
FWD_RAW_DGE = nisa.dge_mode.hwdge
FWD_RAW_ENGINE = nisa.engine.scalar
FWD_DMA_BYTES = 12 * 1024
FWD_PROLOGUE_BUFFERS = 5
FWD_SBUF_BYTES_CS = 245_752
FWD_PREFETCH_TILES = 2
FWD_KH_SCAN_BEFORE_WEIGHT = 20
FWD_FIRST_TILE_CHUNKS = 4
FWD_KH_FIRST_TILE_HEAD_KTILES = 8
FWD_KH_FIRST_SAVE_DEP = True
FWD_KH_STEPS_PER_QUANTIZE = 6
FWD_RING_SLOTS_CS = 4
FWD_CS_BLOCKS_BEFORE_BSCALE = 2
FWD_CS_HEADSTART_GROUPS = 1
FWD_CS_SCALAR_BLOCKS = (2, 4, 6)
FWD_CS_SCAN_DMA_BYTES = 12 * 1024
FWD_CS_REDUCE_PIECES = 2
FWD_CS_SCAN_BYTES = 24 * 1024
FWD_CS_WEIGHT_GROUP_BYTES = 12 * 1024
FWD_CS_SCAN_ENGINE = nisa.engine.sync
FWD_CS_LATE_SCAN_HWDGE = True
FWD_CS_QDEP_BLOCK = 5


def _fwd_rows_per_partition(width: int, row_count: int) -> int:
    """Rows folded into one partition per scan DMA so descriptors are ~FWD_DMA_BYTES."""
    rpp = max(1, min(8, FWD_DMA_BYTES // (width * 2)))
    while rpp > 1 and row_count % (P_MAX * rpp) != 0:
        rpp //= 2
    return rpp


def _fwd_exchange_amax(partial, pipe_id: int):
    """Exact cross-core max of the per-core partial [128, 1] abs-max: the partials are swapped
    SBUF->SBUF with ``nisa.sendrecv`` on the GpSimd DMA (no HBM round trip, no core_barrier;
    the tiny HBM exchange DMAs used to queue behind bulk traffic for 10-25 us). ``pipe_id`` must
    be unique per exchange within the kernel."""
    program_id = nl.program_id(0)
    other = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.sendrecv(
        src=partial, dst=other, send_to_rank=1 - program_id, recv_from_rank=1 - program_id,
        pipe_id=pipe_id, dma_engine=nisa.dma_engine.gpsimd_dma,
    )
    amax = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=amax, data1=partial, data2=other, op=nl.maximum)
    return amax


def _fwd_transpose_weight_group(weight, raw, k_tile0: int, group: int, out_features: int, in_features: int, row0=0):
    """Grouped hardware-DGE DMA transpose of canonical [out, in] weight columns (k tiles
    k_tile0 .. k_tile0 + group, rows row0 .. row0 + out_features) into [in-part, group, out]."""
    nisa.dma_transpose(
        dst=raw.reshape((P_MAX, group, out_features, 1)),
        src=weight.ap(
            pattern=[[0, 1], [P_MAX, group], [in_features, out_features], [1, P_MAX]],
            offset=row0 * in_features + k_tile0 * P_MAX,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )


def _fwd_scan_load(input_view, scan_bufs, n_pro: int, row_base: int, blk: int, chunks: int = 1, hw_from: int = -1):
    """Load scan block ``blk`` (128 partitions x scan_width) into its buffer; ``chunks`` > 1 splits
    the free dim into separate DMAs (smaller descriptors share the DMA rings more evenly with
    concurrent DMA transposes while the buffer itself stays large). Blocks >= ``hw_from`` (if >= 0)
    are issued as hardware-DGE DMAs from the Sync queue instead of SWDGE: the loads issued after the
    weight-scale chain in program order then cannot be ordered ahead of its GpSimd partition reduce /
    sendrecv in the GpSimd in-order queue (they waited there for Vector Engine reduces)."""
    r0 = row_base + blk * P_MAX
    buf = scan_bufs[blk % n_pro]
    width = buf.shape[1]
    hw = hw_from >= 0 and blk >= hw_from
    if chunks == 1:
        if hw:
            nisa.dma_copy(dst=buf, src=input_view[r0:r0 + P_MAX, :], dge_mode=nisa.dge_mode.hwdge, engine=FWD_CS_SCAN_ENGINE)
        else:
            nisa.dma_copy(dst=buf, src=input_view[r0:r0 + P_MAX, :])
    else:
        w = width // chunks
        for c in nl.static_range(chunks):
            if hw:
                nisa.dma_copy(dst=buf[:, c * w:(c + 1) * w], src=input_view[r0:r0 + P_MAX, c * w:(c + 1) * w],
                              dge_mode=nisa.dge_mode.hwdge, engine=FWD_CS_SCAN_ENGINE)
            else:
                nisa.dma_copy(dst=buf[:, c * w:(c + 1) * w], src=input_view[r0:r0 + P_MAX, c * w:(c + 1) * w])


def _fwd_scan_reduce(scan_bufs, n_pro: int, blk: int, running, per_partition, pieces: int = 1, init=None):
    """Exact abs-max of one scan block folded into ``running``; ``pieces`` > 1 reduces the block in
    equal free-dim pieces (finer granularity lets other small Vector Engine ops interleave sooner).
    ``init`` (a [128, 1] fp32 tile holding +0.0): the first reduce loads it into the accumulator
    (max(0, |x|) == max(|x|): exact) -- a true data dependency used to order the Vector Engine queue."""
    scan = scan_bufs[blk % n_pro]
    if pieces == 1:
        if init is None:
            nisa.tensor_scalar_reduce(
                dst=scan, data=scan, op0=nl.abs, operand0=0.0, reduce_op=nl.maximum, reduce_res=per_partition
            )
        else:
            nisa.tensor_scalar_reduce(
                dst=scan, data=scan, op0=nl.abs, operand0=0.0, reduce_op=nl.maximum, reduce_res=per_partition,
                reduce_cmd=nisa.reduce_cmd.load_reduce, reduce_init=init,
            )
        nisa.tensor_tensor(dst=running, data1=running, data2=per_partition, op=nl.maximum)
    else:
        w = scan.shape[1] // pieces
        for p in nl.static_range(pieces):
            part = scan[:, p * w:(p + 1) * w]
            if p == 0 and init is not None:
                nisa.tensor_scalar_reduce(
                    dst=part, data=part, op0=nl.abs, operand0=0.0, reduce_op=nl.maximum, reduce_res=per_partition,
                    reduce_cmd=nisa.reduce_cmd.load_reduce, reduce_init=init,
                )
            else:
                nisa.tensor_scalar_reduce(
                    dst=part, data=part, op0=nl.abs, operand0=0.0, reduce_op=nl.maximum, reduce_res=per_partition
                )
            nisa.tensor_tensor(dst=running, data1=running, data2=per_partition, op=nl.maximum)


def _fwd_is_scalar_block(blk: int, split_at: int, scan_blocks: int) -> bool:
    """Scan blocks (column-split prologue) whose abs-max is reduced on the Scalar Engine."""
    if nisa.get_nc_version() < nisa.nc_version.gen4:
        # gen3 (trn2): the Scalar Engine scan variant is built on ``nisa.activate2`` (gen4-only).
        # FWD_CS_SCALAR_BLOCKS=() is the documented activate2-free scan: every block takes the
        # Vector Engine ``_fwd_scan_step`` reduce instead, which folds the same per-block maxima
        # in the same block order (exact fp32 max) -> the amax and every output stay bitwise.
        return False
    if blk < split_at or blk >= scan_blocks - 1:
        return False
    for b in FWD_CS_SCALAR_BLOCKS:
        if b == blk:
            return True
    return False


def _fwd_count_scalar_blocks(split_at: int, scan_blocks: int) -> int:
    n = 0
    for blk in range(split_at, scan_blocks):
        if _fwd_is_scalar_block(blk, split_at, scan_blocks):
            n += 1
    return n


def _fwd_scan_reduce_scalar(scan_bufs, n_pro: int, blk: int, first: bool, reduce_res):
    """Exact abs-max of one scan block on the Scalar Engine: activate2(abs) with the engine's reduce
    registers (reset on the first block, accumulated across blocks, read out once via ``reduce_res``).
    The activated |x| is written back in place (bf16, exact) as junk output."""
    scan = scan_bufs[blk % n_pro]
    nisa.activate2(
        dst=scan, op=nl.abs, data=scan, imm0=0.0, imm1=0.0, op0=nl.bypass, op1=nl.bypass,
        reduce_op=nl.maximum, reduce_res=reduce_res,
        reduce_cmd=(nisa.reduce_cmd.reset_reduce if first else nisa.reduce_cmd.reduce),
    )


def _fwd_weight_absmax_state(gen4: bool):
    """Per-prologue state for the weight abs-max chain: ``(b_per_partition, w_pp)``.

    gen4 accumulates in the Scalar Engine reduce registers and reads them out once into
    ``b_per_partition`` (uninitialized until then).  gen3 folds one-shot per-group Vector Engine
    reduces into ``b_per_partition`` with an exact fp32 running max, so it starts at +0.0
    (max(0, |x|) == max(|x|): exact) and ``w_pp`` holds each group's readout."""
    if gen4:
        b_per_partition = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        w_pp = None
    else:
        b_per_partition = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        w_pp = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    return b_per_partition, w_pp


def _fwd_weight_absmax_group(dst, data, reduce_res, reduce_cmd, running=None, per_partition=None):
    """Exact per-partition abs + max of one transposed weight group.  The |x| junk output lands in
    ``dst`` (its pinned alias carries the scheduling dependencies) in both forms.

    gen4: ``nisa.activate2`` on the Scalar Engine, accumulated across groups in the engine's reduce
    registers (reset on the first group; ``reduce_res`` is read out once on the last group).

    gen3 (trn2): activate2 does not exist, and the Scalar Engine's ``nisa.activation`` reduce
    registers only accumulate ADD on this hardware (a reduce_op=maximum chain silently sums --
    probed on device), so the group reduce runs on the Vector Engine instead: one one-shot
    ``tensor_scalar_reduce`` (default reset_reduce + immediate readout, no engine-accumulator state
    left behind to collide with the scan reduces sharing the engine) folded into ``running`` with an
    exact fp32 max -- the same exact form the plain gen3 kernels' scans use."""
    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        nisa.activate2(
            dst=dst, op=nl.abs, data=data,
            imm0=0.0, imm1=0.0, op0=nl.bypass, op1=nl.bypass, reduce_op=nl.maximum,
            reduce_res=reduce_res,
            reduce_cmd=reduce_cmd,
        )
    else:
        nisa.tensor_scalar_reduce(
            dst=dst, data=data, op0=nl.abs, operand0=0.0,
            reduce_op=nl.maximum, reduce_res=per_partition,
        )
        nisa.tensor_tensor(dst=running, data1=running, data2=per_partition, op=nl.maximum)


def _fwd_scan_step_scalar(input_view, scan_bufs, n_scan: int, row_base: int, blk: int, scan_blocks: int, first: bool, reduce_res, chunks: int = 1, hw_from: int = -1):
    _fwd_scan_reduce_scalar(scan_bufs, n_scan, blk, first, reduce_res)
    if blk + n_scan < scan_blocks:
        _fwd_scan_load(input_view, scan_bufs, n_scan, row_base, blk + n_scan, chunks, hw_from)


def _fwd_scan_step(input_view, scan_bufs, n_scan: int, row_base: int, blk: int, scan_blocks: int, running, per_partition, chunks: int = 1, pieces: int = 1, hw_from: int = -1, init=None):
    """Reduce scan block ``blk`` (already loaded), then issue the load of block ``blk + n_scan``
    into the buffer it just freed. Under ``nl.no_reorder`` every DMA trigger also waits for the
    compute instruction that precedes it in program order, so the load must directly follow the
    reduce it truly depends on (never a clamp or a Scalar Engine quantize)."""
    _fwd_scan_reduce(scan_bufs, n_scan, blk, running, per_partition, pieces, init)
    if blk + n_scan < scan_blocks:
        _fwd_scan_load(input_view, scan_bufs, n_scan, row_base, blk + n_scan, chunks, hw_from)


def _fwd_load_kmajor_tile(input_2d, row_start: int, a_raw, K: int, k_tiles: int, tp_group: int):
    """K-major load of one 128-row activation tile: grouped hardware-DGE DMA transposes straight
    from HBM (matmul path)."""
    for group in nl.affine_range(k_tiles // tp_group):
        nisa.dma_transpose(
            dst=a_raw[:, group * tp_group:(group + 1) * tp_group, :].reshape((P_MAX, tp_group, P_MAX, 1)),
            src=input_2d.ap(
                pattern=[[0, 1], [P_MAX, tp_group], [K, P_MAX], [1, P_MAX]],
                offset=row_start * K + group * tp_group * P_MAX,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )


def _fwd_prefetch_tiles(input_2d, raw_ring, a_raw_ring, prefetch_tiles: int, grid, program_id, K: int, k_tiles: int, tp_group: int, first_kmajor: int = 0, load_raw: bool = True):
    """Issue the remaining loads of the first tiles (tile 0's row-major load was issued earlier; the
    K-major transposes of tiles < first_kmajor were issued at the start of the prologue). ``raw_ring`` /
    ``a_raw_ring`` are indexed by tile position."""
    for t in nl.static_range(prefetch_tiles):
        row_start = (t * grid + program_id) * P_MAX
        if t > 0 and load_raw:
            nisa.dma_copy(dst=raw_ring[t], src=input_2d[row_start:row_start + P_MAX, 0:K], dge_mode=FWD_RAW_DGE, engine=FWD_RAW_ENGINE)
        if t < first_kmajor:
            continue
        for group in nl.affine_range(k_tiles // tp_group):
            nisa.dma_transpose(
                dst=a_raw_ring[t][:, group * tp_group:(group + 1) * tp_group, :].reshape((P_MAX, tp_group, P_MAX, 1)),
                src=input_2d.ap(
                    pattern=[[0, 1], [P_MAX, tp_group], [K, P_MAX], [1, P_MAX]],
                    offset=row_start * K + group * tp_group * P_MAX,
                ),
                dge_mode=nisa.dge_mode.hwdge,
            )


def _fwd_tile_prepare(input_2d, a_qscale, row_tile: int, raw, a_raw, a_fp8, K: int, k_tiles: int, tp_group: int, preloaded: bool, save: bool, head_ktiles: int = 0, dep_scale: bool = False):
    """Row-split main loop, stage 1 of a tile: issue its loads (unless prefetched) and its K-major
    quantize (fused Vector Engine multiply + min). Issued one tile ahead of the compute stage so the
    PE-critical quantize precedes the previous tile's save-path quantize in the Vector Engine queue."""
    row_start = row_tile * P_MAX
    if not preloaded:
        if save:
            nisa.dma_copy(dst=raw, src=input_2d[row_start:row_start + P_MAX, 0:K], dge_mode=FWD_RAW_DGE, engine=FWD_RAW_ENGINE)
        _fwd_load_kmajor_tile(input_2d, row_start, a_raw, K, k_tiles, tp_group)
    if head_ktiles > 0 and head_ktiles < k_tiles:
        # First tile only: quantize the leading k tiles first so the first k pairs' matmuls can start
        # while the rest of the tile is still being quantized (same elementwise ops).
        nisa.tensor_scalar(
            dst=a_fp8[:, 0:head_ktiles, :], data=a_raw[:, 0:head_ktiles, :], op0=nl.multiply, operand0=a_qscale,
            op1=nl.minimum, operand1=E4M3_MAX, engine=nisa.engine.vector,
        )
        scale_after = None
        if dep_scale:
            # Copy of the scale that depends on the head chunk (a_qscale + 0 * one quantized element =
            # a_qscale exactly): the first tile's save-path quantize uses it, so the list scheduler cannot
            # order that 2.8 us Vector Engine op ahead of the PE-critical head chunk.
            dep = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=dep, data=a_fp8[:, 0, 0:1], op0=nl.multiply, operand0=0.0, engine=nisa.engine.vector)
            scale_after = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=scale_after, data1=a_qscale, data2=dep, op=nl.add)
        nisa.tensor_scalar(
            dst=a_fp8[:, head_ktiles:k_tiles, :], data=a_raw[:, head_ktiles:k_tiles, :], op0=nl.multiply, operand0=a_qscale,
            op1=nl.minimum, operand1=E4M3_MAX, engine=nisa.engine.vector,
        )
        return scale_after
    else:
        nisa.tensor_scalar(
            dst=a_fp8.reshape((P_MAX, K)), data=a_raw.reshape((P_MAX, K)), op0=nl.multiply, operand0=a_qscale,
            op1=nl.minimum, operand1=E4M3_MAX, engine=nisa.engine.vector,
        )
    return None


def _fwd_matmul_group(accum, a_fp8, b_fp8, n_start: int, K: int, nstep: int):
    """All k pairs of one nstep x 512-column PSUM accumulator group. The first k pair is issued as
    nstep 512-wide matmuls with ``accumulate=False`` (the PE reset flag clears exactly the PSUM bank
    it writes), the remaining k pairs as ONE ``nstep * 512``-wide matmul each with ``accumulate=True``
    (NeuronCore-v4 moving free size up to 4096 fp32 columns). Every output element still accumulates
    the same k pairs in the same order in fp32 PSUM, so the result is bitwise identical to the
    per-512-column chains; the wide instructions save the fixed per-instruction PE overhead."""
    k_pairs = K // DOUBLE_ROW_K
    for n_subtile in nl.affine_range(nstep):
        col_start = n_start + n_subtile * N_TILE
        nisa.nc_matmul(
            dst=accum[:, n_subtile * N_TILE:(n_subtile + 1) * N_TILE],
            stationary=a_fp8[:, 0:2, :],
            moving=b_fp8[:, 0:2, col_start:col_start + N_TILE],
            perf_mode=matmul_perf_mode.double_row,
            accumulate=False,
        )
    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        for k_pair in nl.affine_range(1, k_pairs):
            nisa.nc_matmul(
                dst=accum,
                stationary=a_fp8[:, 2 * k_pair:2 * k_pair + 2, :],
                moving=b_fp8[:, 2 * k_pair:2 * k_pair + 2, n_start:n_start + nstep * N_TILE],
                perf_mode=matmul_perf_mode.double_row,
                accumulate=True,
            )
    else:
        # gen3 (trn2): the moving free size is capped at 512 fp32 columns, so k pairs 1.. are
        # issued as one 512-wide accumulate=True matmul per PSUM bank (same pattern as
        # _matmul_wgrad_block_wide in fp8_wgrad.py).  Every output element still accumulates
        # the same k pairs in the same order in fp32 PSUM -> bitwise the wide form.
        for k_pair in nl.affine_range(1, k_pairs):
            for n_subtile in nl.affine_range(nstep):
                col_start = n_start + n_subtile * N_TILE
                nisa.nc_matmul(
                    dst=accum[:, n_subtile * N_TILE:(n_subtile + 1) * N_TILE],
                    stationary=a_fp8[:, 2 * k_pair:2 * k_pair + 2, :],
                    moving=b_fp8[:, 2 * k_pair:2 * k_pair + 2, col_start:col_start + N_TILE],
                    perf_mode=matmul_perf_mode.double_row,
                    accumulate=True,
                )

def _fwd_tile_compute(
    input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale, row_tile: int,
    raw, quantized, a_fp8, result, K: int, n_groups: int, nstep: int, save: bool = True, save_scale=None,
):
    """Row-split main loop, stage 2 of a tile: E4M3 save path, matmuls, dequant, output store."""
    row_start = row_tile * P_MAX
    if save:
        nisa.tensor_scalar(
            dst=quantized, data=raw, op0=nl.multiply, operand0=(a_qscale if save_scale is None else save_scale),
            op1=nl.minimum, operand1=E4M3_MAX, engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=saved_e4m3[row_start:row_start + P_MAX, 0:K], src=quantized)
    for n_group in nl.affine_range(n_groups):
        n_start = n_group * nstep * N_TILE
        accum = nl.ndarray((P_MAX, nstep * N_TILE), dtype=nl.float32, buffer=nl.psum)
        _fwd_matmul_group(accum, a_fp8, b_fp8, n_start, K, nstep)
        nisa.activation(dst=result[:, n_group, :], op=nl.copy, data=accum, scale=output_scale)
        nisa.dma_copy(
            dst=out[row_start:row_start + P_MAX, n_start:n_start + nstep * N_TILE],
            src=result[:, n_group, :],
        )


def _fwd_pipe_step(
    input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale, grid, program_id,
    t, s: int, has_next: bool, next_preloaded: bool, t_next, s_next: int,
    raw_ring, quant_ring, a_raw_ring, a_fp8_ring, res_ring, K: int, k_tiles: int, tp_group: int, n_groups: int, nstep: int,
    save_scale=None, save: bool = True,
):
    """One software-pipelined step of the row-split main loop: stage 1 of local tile ``t_next`` (slot
    ``s_next``), then stage 2 of local tile ``t`` (slot ``s``)."""
    if has_next:
        # (first step: the next tile's K-major quantize also uses the head-chunk-dependent scale copy)
        _fwd_tile_prepare(input_2d, (a_qscale if save_scale is None else save_scale), t_next * grid + program_id,
                          raw_ring[s_next], a_raw_ring[s_next], a_fp8_ring[s_next], K, k_tiles, tp_group, next_preloaded, save)
    _fwd_tile_compute(input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale, t * grid + program_id,
                      raw_ring[s], quant_ring[s], a_fp8_ring[s], res_ring[s], K, n_groups, nstep, save, save_scale)


def _fwd_row_tile(
    input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale, row_tile: int,
    raw, quantized, a_raw, a_fp8, result,
    K: int, k_tiles: int, n_groups: int, nstep: int, tp_group: int, preloaded: bool,
    col_off=0, save: bool = True, chunks: int = 1,
):
    """One 128-row tile of the fused forward: E4M3 save (row-major, if ``save``) + GEMM (K-major)
    for output columns ``col_off .. col_off + n_groups * nstep * 512``.

    The row-major quantize uses Vector Engine ``tensor_scalar`` (fp32 multiply then round to
    fp8: bitwise identical to the Scalar Engine ``activation`` with ``scale``); the Scalar Engine
    carries the K-major quantize and the PSUM dequant.
    """
    row_start = row_tile * P_MAX
    if not preloaded:
        if save:
            nisa.dma_copy(dst=raw, src=input_2d[row_start:row_start + P_MAX, 0:K], dge_mode=FWD_RAW_DGE, engine=FWD_RAW_ENGINE)
        _fwd_load_kmajor_tile(input_2d, row_start, a_raw, K, k_tiles, tp_group)
    # K-major quantize first (it gates the matmuls); the E4M3 save path follows on the same
    # in-order Vector Engine queue.
    if chunks == 1:
        a_raw_flat = a_raw.reshape((P_MAX, K))
        a_fp8_flat = a_fp8.reshape((P_MAX, K))
        nisa.tensor_scalar(
            dst=a_fp8_flat, data=a_raw_flat, op0=nl.multiply, operand0=a_qscale, op1=nl.minimum, operand1=E4M3_MAX,
            engine=nisa.engine.vector,
        )
    else:
        # First tile only: quantize in k-tile chunks so the matmuls of the first k pairs can
        # start while the rest of the tile is still being quantized (same elementwise ops).
        kc = k_tiles // chunks
        for c in nl.static_range(chunks):
            part = a_fp8[:, c * kc:(c + 1) * kc, :]
            nisa.tensor_scalar(
                dst=part, data=a_raw[:, c * kc:(c + 1) * kc, :], op0=nl.multiply, operand0=a_qscale,
                op1=nl.minimum, operand1=E4M3_MAX, engine=nisa.engine.vector,
            )
    if save:
        nisa.tensor_scalar(
            dst=quantized, data=raw, op0=nl.multiply, operand0=a_qscale, op1=nl.minimum, operand1=E4M3_MAX,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=saved_e4m3[row_start:row_start + P_MAX, 0:K], src=quantized)
    for n_group in nl.affine_range(n_groups):
        n_start = n_group * nstep * N_TILE
        accum = nl.ndarray((P_MAX, nstep * N_TILE), dtype=nl.float32, buffer=nl.psum)
        _fwd_matmul_group(accum, a_fp8, b_fp8, n_start, K, nstep)
        nisa.activation(dst=result[:, n_group, :], op=nl.copy, data=accum, scale=output_scale)
        nisa.dma_copy(
            dst=out[row_start:row_start + P_MAX, col_off + n_start:col_off + n_start + nstep * N_TILE],
            src=result[:, n_group, :],
        )


@nki.jit
def tensorwise_fp8_forward_save_v1(input_2d, weight, nstep: int):
    """Fused training forward: see ``_forward_save_impl``."""
    return _forward_save_impl(input_2d, weight, nstep, True)


@nki.jit
def tensorwise_fp8_forward_nosave_v1(input_2d, weight, nstep: int):
    """``tensorwise_fp8_forward_save_v1`` without the E4M3 save (row split, M >= 4N, only): the same
    BF16 ``out`` alone, without the row-major activation loads, the save quantize and the byte stores.
    For c_proj, whose saved bytes come from ``tensorwise_e4m3_save``."""
    return _forward_save_impl(input_2d, weight, nstep, False)


def _forward_save_impl(input_2d, weight, nstep: int, save: bool):
    """Fused training forward: save native E4M3 input bytes and compute BF16 ``input @ weight.T``.

    Semantics are identical to ``tensorwise_e4m3_save`` followed by ``tensorwise_fp8_gemm``
    (E4M3 x E4M3, ``mstep=1``) on ``weight.t()``: same tensorwise amax/scale formula, same
    quantization (round to E4M3 then clamp to +-240), same per-element matmul accumulation order
    in fp32 PSUM and the same saved byte layout, so ``out``, the saved bytes and the inverse scale
    are bitwise identical. Only data movement, engine placement, instruction shapes and the split
    of work across the two cores differ:

    * abs-max: each core scans half of the activation rows (rows folded into ~24 KB or ~12 KB
      DMA descriptors, exact max on the Vector Engine, some blocks on the Scalar Engine's reduce
      registers) and takes the weight abs-max straight from the transposed weight tiles it owns
      (Scalar Engine ``activate2`` abs + max, exact); the per-core partial (128, 1) maxima are
      swapped SBUF->SBUF with ``nisa.sendrecv`` (exact max, no HBM round trip);
    * quantization: ONE fused Vector Engine ``tensor_scalar(multiply scale, minimum 240)`` -> E4M3
      per tile (bitwise identical to multiply -> E4M3 -> clamp for every bf16 input incl. NaN, and
      to the Scalar Engine ``activation`` path up to the sign of zero, which cannot change ``out``);
    * matmuls: the first k pair of every ``nstep * 512``-column accumulator group is issued as
      ``nstep`` 512-wide matmuls with ``accumulate=False`` (the PE reset clears one PSUM bank), the
      remaining k pairs as one ``nstep * 512``-wide ``accumulate=True`` matmul each (NeuronCore-v4);
    * M < 4N (c_fc): COLUMN split -- every core computes all row tiles for its own N/2 output
      columns, so it only DMA-transposes and quantizes its N-half of the canonical ``[out, in]``
      weight (no quantized-weight exchange); the E4M3 save of a row tile is done by one core
      (row tiles i + pid for even local index i, i - pid for odd i);
    * M >= 4N (c_proj): ROW split, each core transposes and quantizes only its K-half of the
      weight (spread over the scan) and the two E4M3 halves are swapped with one dma-mode
      ``sendrecv``;
    * every per-tile SBUF buffer lives in a ring pinned at explicit addresses and every prologue
      buffer aliases a pinned region that is live later (ring slots / quantized weight): the
      allocator otherwise renames buffers into dead regions and drags in false dependencies;
    * K-major activation tiles come from grouped hardware-DGE DMA transposes; the in-order engine
      queues are ordered with true (value-preserving) data dependencies where the list scheduler
      would otherwise put DMA-waiting reduces ahead of ready work (Vector Engine accumulator
      preload with +0.0, scale copies ``s + 0 * x``).

    Args:
        input_2d: BF16 [M, K] activation in HBM.
        weight: BF16 [N, K] canonical Linear weight in HBM.
        nstep: Number of adjacent 512-column output tiles grouped per PSUM accumulator.

    Returns:
        (BF16 [M, N] output, uint8 [M, K] E4M3 bytes, FP32 [2, 128, 1] replicated inverse scale), or the
        output alone if not ``save``.
    """
    kernel_assert(len(input_2d.shape) == 2, "input must be rank 2")
    kernel_assert(len(weight.shape) == 2, "weight must be rank 2")
    M, K = input_2d.shape
    N, K_w = weight.shape
    kernel_assert(K == K_w, "input and weight contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "input rows must be divisible by 128")
    kernel_assert(K % N_TILE == 0, "input width must be divisible by 512")
    kernel_assert(N % (nstep * N_TILE) == 0, "N must divide nstep * 512")
    kernel_assert(K * N <= RESIDENT_B_ELEMENTS, "weight is too large for the resident kernel")

    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    row_tiles = M // P_MAX
    kernel_assert(row_tiles % grid == 0, "input row tiles must divide the grid")
    kernel_assert(N % (grid * P_MAX) == 0, "weight rows must divide grid * 128")
    k_tiles = K // P_MAX
    n_tiles = N // N_TILE
    n_groups = n_tiles // nstep
    rows_per_program = M // grid
    weight_rows_per_program = N // grid
    tiles_per_core = row_tiles // grid

    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    saved_bits = saved_e4m3 = inverse_scales = None
    if save:
        saved_bits = nl.ndarray((M, K), dtype=nl.uint8, buffer=nl.shared_hbm)
        saved_e4m3 = saved_bits.view(nl.float8_e4m3)
        inverse_scales = nl.ndarray((2, P_MAX, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- SBUF layout: per-tile ring buffers pinned at explicit addresses ---------------
    col_split = M < 4 * N
    kernel_assert(save or not col_split, "the no-save forward supports only the row split (M >= 4N)")
    if col_split:
        # Column split: every core computes ALL row tiles for its own N/2 output columns (no
        # quantized-weight exchange; each core only transposes/quantizes its N-half), and saves
        # the E4M3 bytes of every other row tile (row tiles i + pid for even local index i,
        # i - pid for odd i).
        kernel_assert(N % (grid * nstep * N_TILE) == 0, "N must split across the cores in nstep * 512 groups")
        n_cols = N // grid
        n_groups_c = n_cols // (nstep * N_TILE)
        col_off = program_id * n_cols
        ring = FWD_RING_SLOTS_CS
        raw_bytes = K * 2
        q_bytes = K
        res_bytes = n_groups_c * nstep * N_TILE * 2
        slot_even = 2 * raw_bytes + 2 * q_bytes + res_bytes   # save tile: row-major + K-major + result
        slot_odd = raw_bytes + q_bytes + res_bytes             # GEMM-only tile: K-major + result
        ring_bytes = (ring // 2) * (slot_even + slot_odd)
        w_group = 1
        for cand in (16, 8, 4, 2):
            if w_group == 1 and k_tiles % cand == 0 and cand * n_cols * 2 <= FWD_CS_WEIGHT_GROUP_BYTES:
                w_group = cand
        w_groups = k_tiles // w_group
        w_raw_bytes = w_group * n_cols * 2
        # Scan blocks of ~FWD_CS_SCAN_BYTES per partition: larger than any prologue buffer that dies
        # during the scan (w_bufs groups), so the allocator cannot rename a WAR-dependent scan buffer
        # into a dead region (it did with 12 KB blocks: see NOTES round 2).
        rpp = max(1, min(8, FWD_CS_SCAN_BYTES // (K * 2)))
        while rpp > 1 and rows_per_program % (P_MAX * rpp) != 0:
            rpp //= 2
        scan_width = rpp * K
        scan_bytes = scan_width * 2
        kernel_assert(scan_bytes > w_raw_bytes, "scan blocks must be larger than a weight group")
        b_fp8_base = ring_bytes
        b_fp8_bytes = k_tiles * n_cols
        w_base = b_fp8_base + b_fp8_bytes
        scan_base = w_base + w_groups * w_raw_bytes
        if nisa.get_nc_version() >= nisa.nc_version.gen4:
            sbuf_cap = FWD_SBUF_BYTES_CS
        else:
            # gen3 (trn2): the SBUF bound is nl.tile_size.sbuf_fmax_bytes = 212984 B/partition, so
            # n_scan drops 3 -> 2 at the production c_fc shape (162816 B of pinned ring / quantized
            # weight / raw weight regions + 2 x 24576 B scan blocks = 211968 B).  Fewer scan ring
            # buffers only shorten the DMA lookahead; the block REDUCE order is unchanged, so the
            # amax and every downstream output stay bitwise identical.
            sbuf_cap = nl.tile_size.sbuf_fmax_bytes
        n_scan = min(FWD_PROLOGUE_BUFFERS, (sbuf_cap - scan_base) // scan_bytes)
        kernel_assert(n_scan >= 2, "prologue scan buffers do not fit")
        raw_ring, quant_ring, a_raw_ring, a_fp8_ring, res_ring = [], [], [], [], []
        base = 0
        for s in range(ring):
            if s % 2 == 0:
                raw_ring.append(nl.ndarray((P_MAX, K), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base)))
                quant_ring.append(nl.ndarray((P_MAX, K), dtype=nl.float8_e4m3, buffer=nl.sbuf, address=(0, base + raw_bytes)))
                off = base + raw_bytes + q_bytes
            else:
                raw_ring.append(raw_ring[s - 1])      # unused for GEMM-only tiles
                quant_ring.append(quant_ring[s - 1])
                off = base
            a_raw_ring.append(nl.ndarray((P_MAX, k_tiles, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, off)))
            a_fp8_ring.append(nl.ndarray((P_MAX, k_tiles, P_MAX), dtype=nl.float8_e4m3, buffer=nl.sbuf, address=(0, off + raw_bytes)))
            res_ring.append(nl.ndarray((P_MAX, n_groups_c, nstep * N_TILE), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, off + raw_bytes + q_bytes)))
            base += slot_even if s % 2 == 0 else slot_odd
        b_fp8 = nl.ndarray((P_MAX, k_tiles, n_cols), dtype=nl.float8_e4m3, buffer=nl.sbuf, address=(0, b_fp8_base))
        w_bufs = []
        for g in range(w_groups):
            w_bufs.append(nl.ndarray((P_MAX, w_group, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, w_base + g * w_raw_bytes)))
        scan_bufs = []
        for i in range(n_scan):
            scan_bufs.append(nl.ndarray((P_MAX, scan_width), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, scan_base + i * scan_bytes)))
        # abs-max scratch for the weight tiles (raw values are needed again for the quantize): aliases
        # ring slot 2 (future-live, not prefetched).
        kernel_assert(2 * w_raw_bytes <= b_fp8_bytes, "fp32 abs-max scratch must fit the quantized-weight region")
        kernel_assert(2 * w_raw_bytes <= scan_bytes, "fp32 abs-max scratch must fit a scan buffer")
        scratch = nl.ndarray((P_MAX, w_group * n_cols), dtype=nl.float32, buffer=nl.sbuf, address=(0, b_fp8_base))
        # Junk outputs of the first weight abs-max groups alias the early scan buffers: the early scan
        # loads then (write-after-write) wait for those transposed groups to land, which gives the slow
        # DMA transposes a head start over the scan without any fence.
        head = min(FWD_CS_HEADSTART_GROUPS, w_groups, n_scan)
        head_scratch = []
        for g in range(head):
            head_scratch.append(nl.ndarray((P_MAX, w_group * n_cols), dtype=nl.float32, buffer=nl.sbuf,
                                           address=(0, scan_base + ((g + 1) % head) * scan_bytes)))
        tp_group = 1
        for cand in (8, 4, 2):
            if tp_group == 1 and k_tiles % cand == 0:
                tp_group = cand

        # ---- Prologue -----------------------------------------------------------------------
        # Engine plan (per-engine program order is what the scheduler preserves; only the initial
        # DMA issue is fenced so nothing is hoisted above the weight transposes):
        #   DMA     : weight N-half transposes, first-tile prefetch, scan loads (each after the
        #             reduce that frees its buffer)
        #   ScalarE : weight abs-max per transposed group (activate2 abs + max, exact), then the
        #             six quantizes, then the first tile's K-major quantize
        #   VectorE : scan reduces, weight clamps (once their quantize is done), a-amax chain
        #   GpSimd  : SWDGE descriptors, partition reduces, sendrecv exchanges
        a_view = input_2d.reshape((M // rpp, rpp * K))
        scan_blocks = rows_per_program // (P_MAX * rpp)
        row_base = (program_id * rows_per_program) // rpp
        early_blocks = min(n_scan, scan_blocks)
        prefetch_tiles = min(FWD_PREFETCH_TILES, ring)
        chunks = max(1, scan_bytes // FWD_CS_SCAN_DMA_BYTES)
        split_at = min(scan_blocks, FWD_CS_BLOCKS_BEFORE_BSCALE)
        # scan loads issued after the weight-scale chain (program order) go to the Sync HWDGE queue
        hw_from = (split_at + n_scan) if FWD_CS_LATE_SCAN_HWDGE else -1
        b_per_partition, w_pp = _fwd_weight_absmax_state(
            nisa.get_nc_version() >= nisa.nc_version.gen4
        )
        with nl.no_reorder():
            # DMA issue order (the fence orders dispatch): weight transposes, first-tile prefetch,
            # the head weight abs-max groups (Scalar Engine; their junk outputs alias the early scan
            # buffers), then the early scan loads.
            for g in nl.static_range(w_groups):
                _fwd_transpose_weight_group(weight, w_bufs[g], g * w_group, w_group, n_cols, K, col_off)
            nisa.dma_copy(dst=raw_ring[0], src=input_2d[program_id * P_MAX:program_id * P_MAX + P_MAX, 0:K], dge_mode=FWD_RAW_DGE, engine=FWD_RAW_ENGINE)
            _fwd_load_kmajor_tile(input_2d, program_id * P_MAX, a_raw_ring[0], K, k_tiles, tp_group)
            if prefetch_tiles > 1:
                _fwd_load_kmajor_tile(input_2d, (1 - program_id) * P_MAX, a_raw_ring[1], K, k_tiles, tp_group)
            for g in nl.static_range(head):
                _fwd_weight_absmax_group(
                    dst=head_scratch[g], data=w_bufs[g].reshape((P_MAX, w_group * n_cols)),
                    reduce_res=(b_per_partition if g == w_groups - 1 else None),
                    reduce_cmd=(nisa.reduce_cmd.reset_reduce if g == 0 else nisa.reduce_cmd.reduce),
                    running=b_per_partition, per_partition=w_pp,
                )
            for blk in nl.static_range(early_blocks):
                _fwd_scan_load(a_view, scan_bufs, n_scan, row_base, blk, chunks, hw_from)
        # Remaining weight abs-max groups (Scalar Engine): exact abs + max accumulated in the engine's
        # reduce registers, read out once.
        for g in nl.static_range(head, w_groups):
            _fwd_weight_absmax_group(
                dst=scratch, data=w_bufs[g].reshape((P_MAX, w_group * n_cols)),
                reduce_res=(b_per_partition if g == w_groups - 1 else None),
                reduce_cmd=(nisa.reduce_cmd.reset_reduce if g == 0 else nisa.reduce_cmd.reduce),
                running=b_per_partition, per_partition=w_pp,
            )
        # Scan blocks that are expected to land before the weight scale (accumulated separately, exact).
        a_running0 = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        a_per_partition = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        for blk in nl.static_range(split_at):
            _fwd_scan_step(a_view, scan_bufs, n_scan, row_base, blk, scan_blocks, a_running0, a_per_partition, chunks, FWD_CS_REDUCE_PIECES, hw_from)
        b_partial = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_partition_reduce(dst=b_partial, op=nl.maximum, data=b_per_partition)
        b_amax = _fwd_exchange_amax(b_partial, 1)
        b_qscale, b_inverse = _fp8_scale_from_amax(b_amax, E4M3_MAX)
        # a_running starts at 0.0 * b_qscale (= +0.0, exact): a true dependency that keeps the
        # scheduler from sinking the weight-scale chain behind the remaining scan reduces.
        a_running = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=a_running, data=b_qscale, op0=nl.multiply, operand0=0.0)
        for g in nl.static_range(w_groups):
            dst = b_fp8[:, g * w_group:(g + 1) * w_group, :]
            nisa.tensor_scalar(
                dst=dst, data=w_bufs[g], op0=nl.multiply, operand0=b_qscale, op1=nl.minimum, operand1=E4M3_MAX,
                engine=nisa.engine.vector,
            )
        # Vector Engine queue order: the first Vector Engine scan block after the weight scale loads its
        # accumulator from qdep = +0.0 derived from one element of EVERY quantized weight group
        # (0 * x, then min(., 0) -> +-0.0 for any x incl. NaN; max(+-0, |x|) == max(|x|): exact), so the
        # list scheduler cannot order that reduce -- which waits for its DMA -- ahead of the six quantizes.
        qdep = None
        if FWD_CS_QDEP_BLOCK >= split_at and FWD_CS_QDEP_BLOCK < scan_blocks and not _fwd_is_scalar_block(FWD_CS_QDEP_BLOCK, split_at, scan_blocks):
            qdep_all = nl.ndarray((P_MAX, k_tiles, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=qdep_all, data=b_fp8[:, 0:k_tiles, 0:1], op0=nl.multiply, operand0=0.0,
                               op1=nl.minimum, operand1=0.0, engine=nisa.engine.vector)
            qdep = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=qdep, data=qdep_all, op=nl.maximum, axis=(1, 2))
        # Remaining scan blocks: the ones listed in FWD_CS_SCALAR_BLOCKS are reduced on the (otherwise idle)
        # Scalar Engine, the rest on the Vector Engine, which also carries the six weight quantizes.
        n_scalar_blocks = _fwd_count_scalar_blocks(split_at, scan_blocks)
        a_scalar_partial = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        n_sc = 0
        for blk in nl.static_range(split_at, scan_blocks):
            if _fwd_is_scalar_block(blk, split_at, scan_blocks):
                n_sc += 1
                _fwd_scan_step_scalar(a_view, scan_bufs, n_scan, row_base, blk, scan_blocks, n_sc == 1,
                                      a_scalar_partial if n_sc == n_scalar_blocks else None, chunks, hw_from)
            else:
                _fwd_scan_step(a_view, scan_bufs, n_scan, row_base, blk, scan_blocks, a_running, a_per_partition, chunks, FWD_CS_REDUCE_PIECES, hw_from,
                               qdep if blk == FWD_CS_QDEP_BLOCK else None)
        if n_scalar_blocks > 0:
            nisa.tensor_tensor(dst=a_running, data1=a_running, data2=a_scalar_partial, op=nl.maximum)
        nisa.tensor_tensor(dst=a_running, data1=a_running, data2=a_running0, op=nl.maximum)
        a_partial = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_partition_reduce(dst=a_partial, op=nl.maximum, data=a_running)
        a_amax = _fwd_exchange_amax(a_partial, 0)
        a_qscale, a_inverse = _fp8_scale_from_amax(a_amax, E4M3_MAX)
        nisa.dma_copy(dst=inverse_scales[program_id, 0:P_MAX, 0:1], src=a_inverse)
        output_scale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=output_scale, data1=a_inverse, data2=b_inverse, op=nl.multiply)

        # ---- Main loop: all row tiles, this core's columns; save every other tile ---------------
        kernel_assert(ring % 2 == 0 and row_tiles % ring == 0, "column-split ring must be even and divide the row tiles")
        for slot in nl.static_range(ring):
            _fwd_row_tile(
                input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale,
                slot + program_id if slot % 2 == 0 else slot - program_id,
                raw_ring[slot], quant_ring[slot], a_raw_ring[slot], a_fp8_ring[slot], res_ring[slot],
                K, k_tiles, n_groups_c, nstep, tp_group, slot < prefetch_tiles, col_off, slot % 2 == 0,
                FWD_FIRST_TILE_CHUNKS if slot == 0 else 1,
            )
        for local_group in nl.affine_range(1, row_tiles // ring):
            for slot in nl.static_range(ring):
                _fwd_row_tile(
                    input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale,
                    local_group * ring + slot + program_id if slot % 2 == 0 else local_group * ring + slot - program_id,
                    raw_ring[slot], quant_ring[slot], a_raw_ring[slot], a_fp8_ring[slot], res_ring[slot],
                    K, k_tiles, n_groups_c, nstep, tp_group, False, col_off, slot % 2 == 0,
                )
        return out, saved_bits, inverse_scales

    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        ring = FWD_RING_SLOTS_WIDE
    else:
        ring = FWD_RING_SLOTS_WIDE_G3
    if ring > tiles_per_core:
        ring = tiles_per_core
    raw_bytes = K * 2
    q_bytes = K
    res_bytes = n_groups * nstep * N_TILE * 2
    slot_bytes = 2 * raw_bytes + 2 * q_bytes + res_bytes
    raw_ring = []
    quant_ring = []
    a_raw_ring = []
    a_fp8_ring = []
    res_ring = []
    for s in range(ring):
        base = s * slot_bytes
        raw_ring.append(nl.ndarray((P_MAX, K), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, base)))
        quant_ring.append(
            nl.ndarray((P_MAX, K), dtype=nl.float8_e4m3, buffer=nl.sbuf, address=(0, base + raw_bytes))
        )
        a_raw_ring.append(
            nl.ndarray(
                (P_MAX, k_tiles, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf,
                address=(0, base + raw_bytes + q_bytes),
            )
        )
        a_fp8_ring.append(
            nl.ndarray(
                (P_MAX, k_tiles, P_MAX), dtype=nl.float8_e4m3, buffer=nl.sbuf,
                address=(0, base + 2 * raw_bytes + q_bytes),
            )
        )
        res_ring.append(
            nl.ndarray(
                (P_MAX, n_groups, nstep * N_TILE), dtype=nl.bfloat16, buffer=nl.sbuf,
                address=(0, base + 2 * raw_bytes + 2 * q_bytes),
            )
        )
    if True:
        # ---- K-half weight scheme (M >= 4N): every prologue buffer aliases future-live pinned
        # regions (ring slots / b_fp8) so the allocator cannot rename scan buffers into dead space.
        kernel_assert(k_tiles % grid == 0, "k tiles must split across the cores")
        kt_half = k_tiles // grid
        w_group = max(1, min(kt_half, FWD_DMA_BYTES // (N * 2)))
        while kt_half % w_group != 0:
            w_group -= 1
        w_groups = kt_half // w_group
        w_raw_bytes = w_group * N * 2
        b_fp8_base = ring * slot_bytes
        b_fp8_bytes = k_tiles * N
        b_fp8 = nl.ndarray((P_MAX, k_tiles, N), dtype=nl.float8_e4m3, buffer=nl.sbuf, address=(0, b_fp8_base))
        # this core's K-half of the weight, transposed to matmul layout in BF16 (resident until quantized),
        # at the start of the ring region (the first tiles are prefetched only after the scan)
        w_total = w_groups * w_raw_bytes
        w_bufs = []
        for g in range(w_groups):
            w_bufs.append(nl.ndarray((P_MAX, w_group, N), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, g * w_raw_bytes)))
        rpp = _fwd_rows_per_partition(K, rows_per_program)
        scan_width = rpp * K
        scan_bytes = scan_width * 2
        n_pro = min(FWD_PROLOGUE_BUFFERS, (ring * slot_bytes - w_total) // scan_bytes)
        kernel_assert(n_pro >= 2, "prologue scan buffers do not fit in the ring region")
        scan_bufs = []
        for i in range(n_pro):
            scan_bufs.append(nl.ndarray((P_MAX, scan_width), dtype=nl.bfloat16, buffer=nl.sbuf, address=(0, w_total + i * scan_bytes)))
        # fp32 junk output of the Scalar Engine abs-max: aliases b_fp8 (dead until the quantize)
        kernel_assert(2 * w_raw_bytes <= b_fp8_bytes, "abs-max scratch must fit the quantized-weight region")
        scratch = nl.ndarray((P_MAX, w_group * N), dtype=nl.float32, buffer=nl.sbuf, address=(0, b_fp8_base))
        tp_group = 1
        for cand in (8, 4, 2):
            if tp_group == 1 and k_tiles % cand == 0:
                tp_group = cand
        # Tile t of this core uses ring slot rot[t % ring] = (t + ring - 1) % ring: the first tile takes the
        # LAST slot, whose K-major buffers lie behind every prologue buffer when the check below holds, so
        # its transposes can be issued at the start of the prologue (early_kmajor).
        rot = []
        for t in range(ring):
            rot.append((t + ring - 1) % ring)
        pro_end = w_total + n_pro * scan_bytes
        early_kmajor = (rot[0] * slot_bytes + raw_bytes + q_bytes) >= pro_end
        my_kt0 = program_id * kt_half
        other_kt0 = (1 - program_id) * kt_half
        a_view = input_2d.reshape((M // rpp, rpp * K))
        scan_blocks = rows_per_program // (P_MAX * rpp)
        row_base = (program_id * rows_per_program) // rpp
        early_blocks = min(n_pro, scan_blocks)
        prefetch_tiles = min(FWD_PREFETCH_TILES, ring)
        b_per_partition, w_pp = _fwd_weight_absmax_state(
            nisa.get_nc_version() >= nisa.nc_version.gen4
        )
        a_running0 = nl.full((P_MAX, 1), 0.0, dtype=nl.float32, buffer=nl.sbuf)
        a_per_partition = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        # blocks reduced before the weight chain, quantize/clamp interleave, exchange position
        pre_blocks = min(scan_blocks, max(early_blocks, FWD_KH_SCAN_BEFORE_WEIGHT))
        steps_per_q = FWD_KH_STEPS_PER_QUANTIZE
        with nl.no_reorder():
            # Program order == dispatch order under the fence: all DMA-heavy work first.
            for g in nl.static_range(w_groups):
                _fwd_transpose_weight_group(weight, w_bufs[g], my_kt0 + g * w_group, w_group, N, K)
            if early_kmajor:
                # K-major transposes of this core's first tile: its ring slot (the last one) aliases no
                # prologue buffer, so they can be issued now instead of after the scan.
                _fwd_load_kmajor_tile(input_2d, program_id * P_MAX, a_raw_ring[rot[0]], K, k_tiles, tp_group)
            for blk in nl.static_range(early_blocks):
                _fwd_scan_load(a_view, scan_bufs, n_pro, row_base, blk)
            # Weight abs-max of this core's K-half on the Scalar Engine (exact abs + max, accumulated
            # in the engine's reduce registers, read out once), as the transposed groups land.
            for g in nl.static_range(w_groups):
                _fwd_weight_absmax_group(
                    dst=scratch, data=w_bufs[g].reshape((P_MAX, w_group * N)),
                    reduce_res=(b_per_partition if g == w_groups - 1 else None),
                    reduce_cmd=(nisa.reduce_cmd.reset_reduce if g == 0 else nisa.reduce_cmd.reduce),
                    running=b_per_partition, per_partition=w_pp,
                )
            for blk in nl.static_range(pre_blocks):
                _fwd_scan_step(a_view, scan_bufs, n_pro, row_base, blk, scan_blocks, a_running0, a_per_partition)
            b_partial = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_partition_reduce(dst=b_partial, op=nl.maximum, data=b_per_partition)
            b_amax = _fwd_exchange_amax(b_partial, 1)
            b_qscale, b_inverse = _fp8_scale_from_amax(b_amax, E4M3_MAX)
            a_running = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=a_running, data=b_qscale, op0=nl.multiply, operand0=0.0)
            # Quantize this core's K-half (fused Vector Engine multiply + min(240) -> fp8, exact clamp
            # semantics), one group every FWD_KH_STEPS_PER_QUANTIZE scan steps: spread out, the Vector
            # Engine (2.75 us per scan block vs 3.2 us of DMA per block) never throttles the scan DMA.
            blk = pre_blocks
            for g in nl.static_range(w_groups):
                dst = b_fp8[:, my_kt0 + g * w_group:my_kt0 + (g + 1) * w_group, :]
                nisa.tensor_scalar(
                    dst=dst, data=w_bufs[g], op0=nl.multiply, operand0=b_qscale, op1=nl.minimum, operand1=E4M3_MAX,
                    engine=nisa.engine.vector,
                )
                for _ in nl.static_range(steps_per_q):
                    if blk < scan_blocks:
                        _fwd_scan_step(a_view, scan_bufs, n_pro, row_base, blk, scan_blocks, a_running, a_per_partition)
                        blk += 1
            for _ in nl.static_range(steps_per_q):
                if blk < scan_blocks:
                    _fwd_scan_step(a_view, scan_bufs, n_pro, row_base, blk, scan_blocks, a_running, a_per_partition)
                    blk += 1
            # Swap the E4M3 K-halves SBUF->SBUF (one dma-mode sendrecv; contiguous 36 KB per partition).
            nisa.sendrecv(
                src=b_fp8[:, my_kt0:my_kt0 + kt_half, :],
                dst=b_fp8[:, other_kt0:other_kt0 + kt_half, :],
                send_to_rank=1 - program_id, recv_from_rank=1 - program_id, pipe_id=2,
            )
            for b in nl.static_range(blk, scan_blocks):
                _fwd_scan_step(a_view, scan_bufs, n_pro, row_base, b, scan_blocks, a_running, a_per_partition)
            nisa.tensor_tensor(dst=a_running, data1=a_running, data2=a_running0, op=nl.maximum)
            a_partial = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_partition_reduce(dst=a_partial, op=nl.maximum, data=a_running)
            # First tiles of this core (into ring slots whose prologue aliases are dead by now).
            if save:
                nisa.dma_copy(dst=raw_ring[rot[0]], src=input_2d[program_id * P_MAX:program_id * P_MAX + P_MAX, 0:K], dge_mode=FWD_RAW_DGE, engine=FWD_RAW_ENGINE)
            raw_by_pos = []
            a_raw_by_pos = []
            for t in range(ring):
                raw_by_pos.append(raw_ring[rot[t]])
                a_raw_by_pos.append(a_raw_ring[rot[t]])
            _fwd_prefetch_tiles(input_2d, raw_by_pos, a_raw_by_pos, prefetch_tiles, grid, program_id, K, k_tiles, tp_group,
                                first_kmajor=(1 if early_kmajor else 0), load_raw=save)
            a_amax = _fwd_exchange_amax(a_partial, 0)
            a_qscale, a_inverse = _fp8_scale_from_amax(a_amax, E4M3_MAX)
            if save:
                nisa.dma_copy(dst=inverse_scales[program_id, 0:P_MAX, 0:1], src=a_inverse)
            output_scale = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=output_scale, data1=a_inverse, data2=b_inverse, op=nl.multiply)

    # ---- Main loop (software-pipelined: stage 1 of tile t+1 [loads + K-major quantize] is issued
    # before stage 2 of tile t [save path, matmuls, dequant, store]) ---------------------------------
    n_tiles = tiles_per_core
    n_full = n_tiles // ring
    scale0 = _fwd_tile_prepare(input_2d, a_qscale, program_id, raw_ring[rot[0]], a_raw_ring[rot[0]], a_fp8_ring[rot[0]],
                               K, k_tiles, tp_group, True, save, FWD_KH_FIRST_TILE_HEAD_KTILES, FWD_KH_FIRST_SAVE_DEP and save)
    # first group (static): tiles 0 .. ring-1 (or all tiles if fewer)
    first = min(ring, n_tiles)
    for p in nl.static_range(first):
        _fwd_pipe_step(
            input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale, grid, program_id,
            p, rot[p % ring], p + 1 < n_tiles, (p + 1) < prefetch_tiles, p + 1, rot[(p + 1) % ring],
            raw_ring, quant_ring, a_raw_ring, a_fp8_ring, res_ring, K, k_tiles, tp_group, n_groups, nstep,
            scale0 if p == 0 else None, save,
        )
    if n_full >= 3:
        # middle groups: the next tile always exists
        for local_group in nl.affine_range(1, n_full - 1):
            for p in nl.static_range(ring):
                _fwd_pipe_step(
                    input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale, grid, program_id,
                    local_group * ring + p, rot[p], True, False, local_group * ring + p + 1, rot[(p + 1) % ring],
                    raw_ring, quant_ring, a_raw_ring, a_fp8_ring, res_ring, K, k_tiles, tp_group, n_groups, nstep,
                    None, save,
                )
    # remaining tiles (last full group and the remainder), static
    start = (n_full - 1) * ring if n_full >= 2 else first
    for p in nl.static_range(start, n_tiles):
        _fwd_pipe_step(
            input_2d, out, saved_e4m3, b_fp8, a_qscale, output_scale, grid, program_id,
            p, rot[p % ring], p + 1 < n_tiles, False, p + 1, rot[(p + 1) % ring],
            raw_ring, quant_ring, a_raw_ring, a_fp8_ring, res_ring, K, k_tiles, tp_group, n_groups, nstep,
            None, save,
        )

    return (out, saved_bits, inverse_scales) if save else out
