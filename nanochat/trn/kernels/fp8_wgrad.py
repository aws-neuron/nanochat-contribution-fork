"""FP8 wgrad kernels: saved-activation wgrad, persistent and grouped variants (split from nki_fp8.py)."""

import math
import os

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from nki.isa import matmul_perf_mode

from nanochat.trn.kernels.fp8_common import (
    DOUBLE_ROW_K,
    E5M2_MAX,
    N_TILE,
    P_MAX,
    _fp8_scale_from_amax,
    _tensorwise_amax_rows,
    kernel_assert,
)
PERSISTENT_WGRAD_SBUF_BUDGET = 22 * 1024 * 1024
# Trn3-only (NeuronCore-v4, 32 MiB SBUF) settings for the wide persistent wgrad kernel.
WGRAD_WIDE_N = 2048
WGRAD_WIDE_SBUF_BUDGET = 28 * 1024 * 1024


@nki.jit
def tensorwise_fp8_wgrad_saved(
    grad_output,
    input_bits,
    input_inverse_scales,
    kblock: int,
):
    """Compute wgrad from saved native-E4M3 activation bytes.

    ``grad_output[K, M]`` is consumed as the logical transposed A operand,
    while ``input_bits[K, N]`` is the canonical activation encoded as E4M3.
    The activation was quantized during forward, so this kernel only computes
    the E5M2 grad-output scale and streams the saved E4M3 operand.
    """
    kernel_assert(len(grad_output.shape) == 2, "grad_output must be rank 2")
    kernel_assert(len(input_bits.shape) == 2, "saved input must be rank 2")
    K, M = grad_output.shape
    K_b, N = input_bits.shape
    kernel_assert(K == K_b, "wgrad contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "wgrad rows must be divisible by 128")
    kernel_assert(K % kblock == 0, "contraction must divide kblock")
    kernel_assert(kblock % DOUBLE_ROW_K == 0, "kblock must divide into 256")
    kernel_assert(N % N_TILE == 0, "wgrad columns must be divisible by 512")

    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    total_m_tiles = M // P_MAX
    kernel_assert(total_m_tiles % grid == 0, "M tiles must divide the grid")
    m_tiles_per_program = total_m_tiles // grid
    n_tiles = N // N_TILE
    k_tiles_per_block = kblock // P_MAX
    k_blocks = K // kblock

    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    saved_e4m3 = input_bits.view(nl.float8_e4m3)

    grad_amax = _tensorwise_amax_rows(grad_output, 0, K, M)
    grad_qscale, grad_inverse = _fp8_scale_from_amax(
        grad_amax,
        E5M2_MAX,
    )
    input_inverse = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=input_inverse,
        src=input_inverse_scales[program_id, 0:P_MAX, 0:1],
    )
    output_scale = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_tensor(
        dst=output_scale,
        data1=grad_inverse,
        data2=input_inverse,
        op=nl.multiply,
    )

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

        input_fp8 = nl.ndarray(
            (P_MAX, k_tiles_per_block, N_TILE),
            dtype=nl.float8_e4m3,
            buffer=nl.sbuf,
        )

        for k_block_index in nl.affine_range(k_blocks):
            k_start = k_block_index * kblock

            for k_tile in nl.affine_range(k_tiles_per_block):
                k_row = k_start + k_tile * P_MAX
                nisa.dma_copy(
                    dst=input_fp8[:, k_tile, :],
                    src=saved_e4m3[
                        k_row:k_row + P_MAX,
                        col_start:col_start + N_TILE,
                    ],
                )

            for local_m_tile in nl.affine_range(m_tiles_per_program):
                output_row = (
                    local_m_tile * grid + program_id
                ) * P_MAX
                grad_raw = nl.ndarray(
                    (P_MAX, k_tiles_per_block, P_MAX),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                for k_tile in nl.affine_range(k_tiles_per_block):
                    k_col = k_start + k_tile * P_MAX
                    nisa.dma_copy(
                        dst=grad_raw[:, k_tile, :],
                        src=grad_output[
                            k_col:k_col + P_MAX,
                            output_row:output_row + P_MAX,
                        ],
                    )

                grad_fp8 = nl.ndarray(
                    (P_MAX, k_tiles_per_block, P_MAX),
                    dtype=nl.float8_e5m2,
                    buffer=nl.sbuf,
                )
                grad_raw_flat = grad_raw.reshape(
                    (P_MAX, k_tiles_per_block * P_MAX)
                )
                grad_fp8_flat = grad_fp8.reshape(
                    (P_MAX, k_tiles_per_block * P_MAX)
                )
                nisa.activation(
                    dst=grad_fp8_flat,
                    op=nl.copy,
                    data=grad_raw_flat,
                    scale=grad_qscale,
                )
                nisa.tensor_scalar(
                    dst=grad_fp8_flat,
                    data=grad_fp8_flat,
                    op0=nl.minimum,
                    operand0=E5M2_MAX,
                    op1=nl.maximum,
                    operand1=-E5M2_MAX,
                )

                partial = nl.ndarray(
                    (P_MAX, N_TILE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for k_pair in nl.affine_range(kblock // DOUBLE_ROW_K):
                    nisa.nc_matmul(
                        dst=partial,
                        stationary=grad_fp8[
                            :,
                            2 * k_pair:2 * k_pair + 2,
                            :,
                        ],
                        moving=input_fp8[
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


def _load_quantize_wgrad_group(
    grad_output,
    grad_qscale,
    grad_fp8,
    grad_raw,
    k_start: int,
    k_tiles_per_block: int,
    m_tiles_per_program: int,
    grid: int,
    program_id: int,
    double_raw: bool = False,
):
    """Quantize every output-row tile owned by this program once.

    ``double_raw``: ``grad_raw`` has a leading buffer axis of 2; m tiles alternate
    buffers so the bf16 loads of tile t+1 are not blocked by the quantization of tile t.
    """
    m_range = nl.static_range if double_raw else nl.affine_range
    for local_m_tile in m_range(m_tiles_per_program):
        output_row = (local_m_tile * grid + program_id) * P_MAX
        if double_raw:
            raw = grad_raw[:, local_m_tile % 2]
        else:
            raw = grad_raw
        for k_tile in nl.affine_range(k_tiles_per_block):
            k_col = k_start + k_tile * P_MAX
            nisa.dma_copy(
                dst=raw[:, k_tile, :],
                src=grad_output[
                    k_col:k_col + P_MAX,
                    output_row:output_row + P_MAX,
                ],
                # These affine BF16 loads are numerous enough that software
                # descriptor generation otherwise dominates GPSIMD time.
                dge_mode=nisa.dge_mode.none,
            )
        grad_tile = grad_fp8[
            :,
            :,
            local_m_tile * P_MAX:(local_m_tile + 1) * P_MAX,
        ]
        nisa.activation(
            dst=grad_tile,
            op=nl.copy,
            data=raw,
            scale=grad_qscale,
        )
        nisa.tensor_scalar(
            dst=grad_tile,
            data=grad_tile,
            op0=nl.minimum,
            operand0=E5M2_MAX,
            op1=nl.maximum,
            operand1=-E5M2_MAX,
        )


def _matmul_accumulate_wgrad(
    grad_fp8,
    input_fp8,
    accumulator,
    kblock: int,
):
    partial = nl.ndarray(
        (P_MAX, N_TILE),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    for k_pair in nl.affine_range(kblock // DOUBLE_ROW_K):
        nisa.nc_matmul(
            dst=partial,
            stationary=grad_fp8[
                :,
                2 * k_pair:2 * k_pair + 2,
                :,
            ],
            moving=input_fp8[
                :,
                2 * k_pair:2 * k_pair + 2,
                :,
            ],
            perf_mode=matmul_perf_mode.double_row,
            accumulate=(k_pair > 0),
        )
    nisa.tensor_tensor(
        dst=accumulator,
        data1=accumulator,
        data2=partial,
        op=nl.add,
    )


def _store_wgrad_tile(
    out,
    accumulator,
    output_scale,
    output_row: int,
    col_start: int,
):
    result = nl.ndarray(
        (P_MAX, N_TILE),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.activation(
        dst=result,
        op=nl.copy,
        data=accumulator,
        scale=output_scale,
    )
    nisa.dma_copy(
        dst=out[
            output_row:output_row + P_MAX,
            col_start:col_start + N_TILE,
        ],
        src=result,
    )


@nki.jit
def tensorwise_fp8_wgrad_saved_persistent_v3(
    grad_output,
    input_bits,
    input_inverse_scales,
    kblock: int,
):
    """Compute one strict FP8 wgrad with all column accumulators resident."""
    kernel_assert(len(grad_output.shape) == 2, "grad_output must be rank 2")
    kernel_assert(len(input_bits.shape) == 2, "saved input must be rank 2")
    K, M = grad_output.shape
    K_b, N = input_bits.shape
    kernel_assert(K == K_b, "wgrad contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "wgrad rows must be divisible by 128")
    kernel_assert(K % kblock == 0, "contraction must divide kblock")
    kernel_assert(kblock % DOUBLE_ROW_K == 0, "kblock must divide into 256")
    kernel_assert(N % N_TILE == 0, "wgrad columns must be divisible by 512")

    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    total_m_tiles = M // P_MAX
    kernel_assert(total_m_tiles % grid == 0, "M tiles must divide the grid")
    m_tiles_per_program = total_m_tiles // grid
    n_tiles = N // N_TILE
    k_tiles_per_block = kblock // P_MAX
    k_blocks = K // kblock
    persistent_sbuf_bytes = (
        P_MAX * m_tiles_per_program * N * 4
        + P_MAX * k_tiles_per_block * m_tiles_per_program * P_MAX
        + P_MAX * k_tiles_per_block * P_MAX * 2
        + P_MAX * k_tiles_per_block * N_TILE
        + P_MAX * N_TILE * 2
    )
    kernel_assert(
        persistent_sbuf_bytes <= PERSISTENT_WGRAD_SBUF_BUDGET,
        "persistent wgrad buffers exceed the validated SBUF budget",
    )

    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    saved_e4m3 = input_bits.view(nl.float8_e4m3)

    grad_amax = _tensorwise_amax_rows(grad_output, 0, K, M)
    grad_qscale, grad_inverse = _fp8_scale_from_amax(
        grad_amax,
        E5M2_MAX,
    )
    input_inverse = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=input_inverse,
        src=input_inverse_scales[program_id, 0:P_MAX, 0:1],
    )
    output_scale = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.tensor_tensor(
        dst=output_scale,
        data1=grad_inverse,
        data2=input_inverse,
        op=nl.multiply,
    )

    accumulators = nl.ndarray(
        (P_MAX, m_tiles_per_program, N),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.memset(
        dst=accumulators.reshape((P_MAX, m_tiles_per_program * N)),
        value=0.0,
    )
    grad_fp8 = nl.ndarray(
        (P_MAX, k_tiles_per_block, m_tiles_per_program * P_MAX),
        dtype=nl.float8_e5m2,
        buffer=nl.sbuf,
    )
    grad_raw = nl.ndarray(
        (P_MAX, k_tiles_per_block, P_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    input_fp8 = nl.ndarray(
        (P_MAX, k_tiles_per_block, N_TILE),
        dtype=nl.float8_e4m3,
        buffer=nl.sbuf,
    )

    for k_block_index in nl.affine_range(k_blocks):
        k_start = k_block_index * kblock
        _load_quantize_wgrad_group(
            grad_output,
            grad_qscale,
            grad_fp8,
            grad_raw,
            k_start,
            k_tiles_per_block,
            m_tiles_per_program,
            grid,
            program_id,
        )
        for n_tile in nl.affine_range(n_tiles):
            col_start = n_tile * N_TILE
            for k_tile in nl.affine_range(k_tiles_per_block):
                k_row = k_start + k_tile * P_MAX
                nisa.dma_copy(
                    dst=input_fp8[:, k_tile, :],
                    src=saved_e4m3[
                        k_row:k_row + P_MAX,
                        col_start:col_start + N_TILE,
                    ],
                )
            for local_m_tile in nl.affine_range(m_tiles_per_program):
                grad_start = local_m_tile * P_MAX
                _matmul_accumulate_wgrad(
                    grad_fp8[:, :, grad_start:grad_start + P_MAX],
                    input_fp8,
                    accumulators[
                        :,
                        local_m_tile,
                        col_start:col_start + N_TILE,
                    ],
                    kblock,
                )

    for local_m_tile in nl.affine_range(m_tiles_per_program):
        output_row = (local_m_tile * grid + program_id) * P_MAX
        for n_tile in nl.affine_range(n_tiles):
            col_start = n_tile * N_TILE
            _store_wgrad_tile(
                out,
                accumulators[
                    :,
                    local_m_tile,
                    col_start:col_start + N_TILE,
                ],
                output_scale,
                output_row,
                col_start,
            )

    return out


def _wgrad_last_fold_store(
    out,
    output_scale,
    partial,
    accumulator,
    output_row: int,
    col_start: int,
    n_width: int,
    first: bool,
):
    """Last block, one finished [128, n_width] slab: fold the PSUM tile into the SBUF
    accumulator per bank (``first``: the block is also the first one, so the
    accumulator is initialised by a PSUM copy instead of an add), scale to bf16 and
    store (same ops as the in-loop epilogue)."""
    for bank in nl.static_range(n_width // N_TILE):
        bs = bank * N_TILE
        if first:
            nisa.tensor_copy(
                dst=accumulator[:, bs:bs + N_TILE],
                src=partial[:, bs:bs + N_TILE],
            )
        else:
            nisa.tensor_tensor(
                dst=accumulator[:, bs:bs + N_TILE],
                data1=accumulator[:, bs:bs + N_TILE],
                data2=partial[:, bs:bs + N_TILE],
                op=nl.add,
            )
    result = nl.ndarray(
        (P_MAX, n_width),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.activation(
        dst=result,
        op=nl.copy,
        data=accumulator,
        scale=output_scale,
    )
    nisa.dma_copy(
        dst=out[
            output_row:output_row + P_MAX,
            col_start:col_start + n_width,
        ],
        src=result,
        dge_mode=nisa.dge_mode.hwdge,
    )


def _wgrad_wide_block(
    grad_output,
    saved_e4m3,
    out,
    grad_qscale,
    output_scale,
    grad_fp8,
    grad_raw,
    input_fp8,
    accumulators,
    k_start,
    k_tiles_per_block: int,
    m_tiles_per_program: int,
    n_tiles: int,
    n_width: int,
    kblock: int,
    grid,
    program_id,
    first: bool,
    last: bool,
):
    """One contraction block of the persistent wgrad (Trn3 wide schedule).

    ``first``: the block initialises the SBUF accumulators (PSUM copy instead of
    memset + add).  ``last``: the block also scales and stores every finished
    [128, n_width] slab, so the epilogue overlaps the remaining matmuls; its groups are
    software-pipelined by one (the fold/scale/store of group g is emitted after the
    matmuls of group g+1, so two PSUM tiles are live and the allocator alternates
    them), and the kernel's final group is computed and stored in N_TILE-column pieces.
    For n_width > 512 (multi-bank fp32 PSUM destination) the matmul reset flag only
    clears the PSUM bank the instruction writes, so k pair 0 is issued as one
    N_TILE-wide accumulate=False matmul per bank (exactly the values the Trn2 512-wide
    path produces) and the remaining k pairs accumulate n_width-wide.  PSUM->SBUF folds
    are done per bank (bank 0 first) so the next group's reset matmul on a bank only
    waits for that bank's fold.  Input slices and output slabs use hardware DGE
    (descriptors generated on device: no GpSimd cost and no runtime re-patching when
    HBM addresses change between calls).  The k_pair order per output element and every
    elementwise operation are identical to the reference kernel (bitwise-identical dW).
    """
    wide = n_width > N_TILE
    _load_quantize_wgrad_group(
        grad_output,
        grad_qscale,
        grad_fp8,
        grad_raw,
        k_start,
        k_tiles_per_block,
        m_tiles_per_program,
        grid,
        program_id,
        double_raw=True,
    )
    # Peeled blocks use static loops so the final group of the kernel can get a
    # pipelined (piecewise) epilogue.
    n_range = nl.static_range if last else nl.affine_range
    pending_fold = None
    for n_tile in n_range(n_tiles):
        col_start = n_tile * n_width
        for k_tile in nl.affine_range(k_tiles_per_block):
            k_row = k_start + k_tile * P_MAX
            nisa.dma_copy(
                dst=input_fp8[:, k_tile, :],
                src=saved_e4m3[
                    k_row:k_row + P_MAX,
                    col_start:col_start + n_width,
                ],
                # Static descriptors: software descriptor generation on GpSimd
                # (~0.7 us per DMA) otherwise delays the next chunk at every
                # chunk boundary.
                dge_mode=nisa.dge_mode.hwdge,
            )
        for local_m_tile in n_range(m_tiles_per_program):
            grad_start = local_m_tile * P_MAX
            accumulator = accumulators[
                :,
                local_m_tile,
                col_start:col_start + n_width,
            ]
            partial = nl.ndarray(
                (P_MAX, n_width),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            final_group = (
                last and n_tile == n_tiles - 1 and local_m_tile == m_tiles_per_program - 1
            )
            if final_group:
                # Kernel tail: the final group is computed as N_TILE-wide column pieces
                # (all k pairs of one piece, then the next piece: the Trn2 order per
                # output element) so the fold, scale and store of piece p overlap the
                # matmuls of piece p+1 instead of all following the last matmul.
                if pending_fold is not None:
                    _wgrad_last_fold_store(
                        out, output_scale, pending_fold[0], pending_fold[1],
                        pending_fold[2], pending_fold[3], n_width, first,
                    )
                    pending_fold = None
                output_row = (local_m_tile * grid + program_id) * P_MAX
                result = nl.ndarray(
                    (P_MAX, n_width),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                )
                for piece in nl.static_range(n_width // N_TILE):
                    cs = piece * N_TILE
                    for k_pair in nl.static_range(kblock // DOUBLE_ROW_K):
                        nisa.nc_matmul(
                            dst=partial[:, cs:cs + N_TILE],
                            stationary=grad_fp8[
                                :,
                                2 * k_pair:2 * k_pair + 2,
                                grad_start:grad_start + P_MAX,
                            ],
                            moving=input_fp8[
                                :,
                                2 * k_pair:2 * k_pair + 2,
                                cs:cs + N_TILE,
                            ],
                            perf_mode=matmul_perf_mode.double_row,
                            accumulate=(k_pair > 0),
                        )
                    # The very last piece's fold/scale/store is split in two column
                    # halves so the kernel tail after the last matmul is shorter.
                    if piece == n_width // N_TILE - 1:
                        sub_w = N_TILE // 2
                        n_sub = 2
                    else:
                        sub_w = N_TILE
                        n_sub = 1
                    for sub in nl.static_range(n_sub):
                        ss = cs + sub * sub_w
                        if first:
                            nisa.tensor_copy(
                                dst=accumulator[:, ss:ss + sub_w],
                                src=partial[:, ss:ss + sub_w],
                            )
                        else:
                            nisa.tensor_tensor(
                                dst=accumulator[:, ss:ss + sub_w],
                                data1=accumulator[:, ss:ss + sub_w],
                                data2=partial[:, ss:ss + sub_w],
                                op=nl.add,
                            )
                        nisa.activation(
                            dst=result[:, ss:ss + sub_w],
                            op=nl.copy,
                            data=accumulator[:, ss:ss + sub_w],
                            scale=output_scale,
                        )
                        nisa.dma_copy(
                            dst=out[
                                output_row:output_row + P_MAX,
                                col_start + ss:col_start + ss + sub_w,
                            ],
                            src=result[:, ss:ss + sub_w],
                            dge_mode=nisa.dge_mode.hwdge,
                        )
            elif wide:
                # k pair 0 as N_TILE-wide matmuls with the reset flag: the reset only
                # clears the PSUM bank the instruction writes, so one per bank
                # initialises the whole tile (same values as 0 + x; no VectorE memset,
                # no memset -> matmul dependency at block boundaries).
                for bank in nl.static_range(n_width // N_TILE):
                    bs = bank * N_TILE
                    nisa.nc_matmul(
                        dst=partial[:, bs:bs + N_TILE],
                        stationary=grad_fp8[
                            :,
                            0:2,
                            grad_start:grad_start + P_MAX,
                        ],
                        moving=input_fp8[
                            :,
                            0:2,
                            bs:bs + N_TILE,
                        ],
                        perf_mode=matmul_perf_mode.double_row,
                        accumulate=False,
                    )
                for k_pair in nl.affine_range(kblock // DOUBLE_ROW_K - 1):
                    nisa.nc_matmul(
                        dst=partial,
                        stationary=grad_fp8[
                            :,
                            2 * k_pair + 2:2 * k_pair + 4,
                            grad_start:grad_start + P_MAX,
                        ],
                        moving=input_fp8[
                            :,
                            2 * k_pair + 2:2 * k_pair + 4,
                            :,
                        ],
                        perf_mode=matmul_perf_mode.double_row,
                        accumulate=True,
                    )
            else:
                for k_pair in nl.affine_range(kblock // DOUBLE_ROW_K):
                    nisa.nc_matmul(
                        dst=partial,
                        stationary=grad_fp8[
                            :,
                            2 * k_pair:2 * k_pair + 2,
                            grad_start:grad_start + P_MAX,
                        ],
                        moving=input_fp8[
                            :,
                            2 * k_pair:2 * k_pair + 2,
                            :,
                        ],
                        perf_mode=matmul_perf_mode.double_row,
                        accumulate=(k_pair > 0),
                    )
            if not final_group:
                if last and wide:
                    # Last block (statically unrolled): the fold/scale/store of the
                    # previous group is emitted after this group's matmuls, so two PSUM
                    # tiles are live in program order and the allocator alternates them
                    # (no consecutive-group tile reuse).  The store trigger of a slab is
                    # thereby issued one group later, so it no longer waits inside the
                    # in-order Sync queue ahead of the next chunk's slice loads.
                    if pending_fold is not None:
                        _wgrad_last_fold_store(
                            out, output_scale, pending_fold[0], pending_fold[1],
                            pending_fold[2], pending_fold[3], n_width, first,
                        )
                    pending_fold = (partial, accumulator, (local_m_tile * grid + program_id) * P_MAX, col_start)
                    continue
                if first and wide:
                    for bank in nl.static_range(n_width // N_TILE):
                        bs = bank * N_TILE
                        nisa.tensor_copy(
                            dst=accumulator[:, bs:bs + N_TILE],
                            src=partial[:, bs:bs + N_TILE],
                        )
                elif first:
                    nisa.tensor_copy(dst=accumulator, src=partial)
                elif wide:
                    # Fold per PSUM bank (bank 0 first): the next group's reset matmul
                    # on bank b only has to wait for the fold of bank b, so a PSUM
                    # tile reused by consecutive groups costs ~0.5 us instead of 1.9.
                    for bank in nl.static_range(n_width // N_TILE):
                        bs = bank * N_TILE
                        nisa.tensor_tensor(
                            dst=accumulator[:, bs:bs + N_TILE],
                            data1=accumulator[:, bs:bs + N_TILE],
                            data2=partial[:, bs:bs + N_TILE],
                            op=nl.add,
                        )
                else:
                    nisa.tensor_tensor(
                        dst=accumulator,
                        data1=accumulator,
                        data2=partial,
                        op=nl.add,
                    )
                if last:
                    output_row = (local_m_tile * grid + program_id) * P_MAX
                    result = nl.ndarray(
                        (P_MAX, n_width),
                        dtype=nl.bfloat16,
                        buffer=nl.sbuf,
                    )
                    nisa.activation(
                        dst=result,
                        op=nl.copy,
                        data=accumulator,
                        scale=output_scale,
                    )
                    nisa.dma_copy(
                        dst=out[
                            output_row:output_row + P_MAX,
                            col_start:col_start + n_width,
                        ],
                        src=result,
                        dge_mode=nisa.dge_mode.hwdge,
                    )


@nki.jit
def tensorwise_fp8_wgrad_saved_persistent_v4(
    grad_output,
    input_bits,
    input_inverse_scales,
    grad_scale_state,
    kblock: int,
):
    """Compute persistent strict FP8 wgrad from a shared gradient scale.

    Trn3 (NeuronCore-v4) schedule: 2048-wide moving tiles (4x fewer stationary
    loads), first contraction block initialises the accumulators (no 18 MiB memset),
    last block scales+stores each finished slab while the remaining matmuls run.
    Per-element accumulation order is identical to the Trn2 kernel.
    """
    kernel_assert(len(grad_output.shape) == 2, "grad_output must be rank 2")
    kernel_assert(len(input_bits.shape) == 2, "saved input must be rank 2")
    K, M = grad_output.shape
    K_b, N = input_bits.shape
    kernel_assert(K == K_b, "wgrad contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "wgrad rows must be divisible by 128")
    kernel_assert(K % kblock == 0, "contraction must divide kblock")
    kernel_assert(kblock % DOUBLE_ROW_K == 0, "kblock must divide into 256")
    kernel_assert(N % N_TILE == 0, "wgrad columns must be divisible by 512")

    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    total_m_tiles = M // P_MAX
    kernel_assert(total_m_tiles % grid == 0, "M tiles must divide the grid")
    m_tiles_per_program = total_m_tiles // grid
    k_tiles_per_block = kblock // P_MAX
    k_blocks = K // kblock

    if nisa.get_nc_version() >= nisa.nc_version.gen4 and N % WGRAD_WIDE_N == 0:
        n_width = WGRAD_WIDE_N
        sbuf_budget = WGRAD_WIDE_SBUF_BUDGET
    else:
        n_width = N_TILE
        sbuf_budget = PERSISTENT_WGRAD_SBUF_BUDGET
    n_tiles = N // n_width
    persistent_sbuf_bytes = (
        P_MAX * m_tiles_per_program * N * 4
        + P_MAX * k_tiles_per_block * m_tiles_per_program * P_MAX
        + 2 * P_MAX * k_tiles_per_block * P_MAX * 2
        + P_MAX * k_tiles_per_block * n_width
        + P_MAX * n_width * 2
    )
    kernel_assert(
        persistent_sbuf_bytes <= sbuf_budget,
        "persistent wgrad buffers exceed the validated SBUF budget",
    )

    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    saved_e4m3 = input_bits.view(nl.float8_e4m3)
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
    input_inverse = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=input_inverse,
        src=input_inverse_scales[program_id, 0:P_MAX, 0:1],
    )
    output_scale = nl.ndarray(
        (P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.tensor_tensor(
        dst=output_scale,
        data1=grad_inverse,
        data2=input_inverse,
        op=nl.multiply,
    )

    accumulators = nl.ndarray(
        (P_MAX, m_tiles_per_program, N),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    grad_fp8 = nl.ndarray(
        (P_MAX, k_tiles_per_block, m_tiles_per_program * P_MAX),
        dtype=nl.float8_e5m2,
        buffer=nl.sbuf,
    )
    grad_raw = nl.ndarray(
        (P_MAX, 2, k_tiles_per_block, P_MAX),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    input_fp8 = nl.ndarray(
        (P_MAX, k_tiles_per_block, n_width),
        dtype=nl.float8_e4m3,
        buffer=nl.sbuf,
    )

    # First block initialises the accumulators; last block fuses the scaled store.
    _wgrad_wide_block(
        grad_output, saved_e4m3, out, grad_qscale, output_scale,
        grad_fp8, grad_raw, input_fp8, accumulators,
        0, k_tiles_per_block, m_tiles_per_program, n_tiles, n_width, kblock,
        grid, program_id, first=True, last=(k_blocks == 1),
    )
    if k_blocks > 2:
        for k_block_index in nl.affine_range(k_blocks - 2):
            k_start = (k_block_index + 1) * kblock
            _wgrad_wide_block(
                grad_output, saved_e4m3, out, grad_qscale, output_scale,
                grad_fp8, grad_raw, input_fp8, accumulators,
                k_start, k_tiles_per_block, m_tiles_per_program, n_tiles, n_width, kblock,
                grid, program_id, first=False, last=False,
            )
    if k_blocks > 1:
        _wgrad_wide_block(
            grad_output, saved_e4m3, out, grad_qscale, output_scale,
            grad_fp8, grad_raw, input_fp8, accumulators,
            (k_blocks - 1) * kblock, k_tiles_per_block, m_tiles_per_program, n_tiles, n_width, kblock,
            grid, program_id, first=False, last=True,
        )

    return out


def _matmul_wgrad_block_wide(
    grad_fp8,
    input_fp8,
    accumulator,
    kblock: int,
    width: int,
    init: bool,
    final: bool = False,
    out=None,
    output_scale=None,
    output_row: int = 0,
):
    """Accumulate one contraction block of one m tile over the full N width.

    trn3 allows a 4096-wide fp32 PSUM destination. The PSUM tile spans several
    banks and the hardware zeroing of ``accumulate=False`` only covers one
    bank, so k pair 0 is issued per bank and the remaining k pairs accumulate
    over the full width. The per-element accumulation order is unchanged.
    """
    partial = nl.ndarray(
        (P_MAX, width),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    # k pair 0: one 512-wide matmul per PSUM bank with accumulate=False (the
    # hardware zeroing only covers one bank); k pairs 1.. accumulate full width.
    for n_tile in nl.affine_range(width // N_TILE):
        col_start = n_tile * N_TILE
        nisa.nc_matmul(
            dst=partial[:, col_start:col_start + N_TILE],
            stationary=grad_fp8[:, 0:2, :],
            moving=input_fp8[:, 0:2, col_start:col_start + N_TILE],
            perf_mode=matmul_perf_mode.double_row,
            accumulate=False,
        )
    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        for k_pair in nl.affine_range(kblock // DOUBLE_ROW_K - 1):
            nisa.nc_matmul(
                dst=partial,
                stationary=grad_fp8[
                    :,
                    2 * k_pair + 2:2 * k_pair + 4,
                    :,
                ],
                moving=input_fp8[
                    :,
                    2 * k_pair + 2:2 * k_pair + 4,
                    :,
                ],
                perf_mode=matmul_perf_mode.double_row,
                accumulate=True,
            )
    else:
        # gen3 (trn2): the moving free size is capped at 512 fp32 columns, so k pairs 1.. are
        # issued as one 512-wide accumulate=True matmul per PSUM bank.  Every output element
        # still accumulates the same k pairs in the same order -> bitwise the wide form.
        for k_pair in nl.affine_range(kblock // DOUBLE_ROW_K - 1):
            for n_tile in nl.affine_range(width // N_TILE):
                col_start = n_tile * N_TILE
                nisa.nc_matmul(
                    dst=partial[:, col_start:col_start + N_TILE],
                    stationary=grad_fp8[
                        :,
                        2 * k_pair + 2:2 * k_pair + 4,
                        :,
                    ],
                    moving=input_fp8[
                        :,
                        2 * k_pair + 2:2 * k_pair + 4,
                        col_start:col_start + N_TILE,
                    ],
                    perf_mode=matmul_perf_mode.double_row,
                    accumulate=True,
                )
    # accumulate / store per 512-column tile so the epilogue pipelines
    # (accumulate -> scale -> store of tile n overlaps tile n+1).
    for n_tile in nl.affine_range(width // N_TILE):
        col_start = n_tile * N_TILE
        acc_tile = accumulator[:, col_start:col_start + N_TILE]
        if init:
            nisa.tensor_copy(dst=acc_tile, src=partial[:, col_start:col_start + N_TILE])
        else:
            nisa.tensor_tensor(
                dst=acc_tile,
                data1=acc_tile,
                data2=partial[:, col_start:col_start + N_TILE],
                op=nl.add,
            )
        if final:
            _store_wgrad_tile_static(
                out,
                acc_tile,
                output_scale,
                output_row,
                col_start,
            )


def _wgrad_group_kblock(
    grad_output,
    saved_e4m3,
    grad_qscale,
    grad_fp8,
    grad_raw,
    input_fp8,
    accumulators,
    k_start: int,
    k_tiles_per_block: int,
    kblock: int,
    mgroup: int,
    n_tiles: int,
    m_group: int,
    grid: int,
    program_id: int,
    init: bool,
    final: bool = False,
    out=None,
    output_scale=None,
):
    """Quantize this group's grad tiles for one k block and accumulate all N tiles.

    ``final`` stores every output tile right after its last accumulation so the
    epilogue overlaps with the remaining matmuls instead of trailing them.
    The saved-activation loads for the whole block are issued first so they are
    not queued behind the (many) grad-output load triggers on the sync engine.
    """
    for n_tile in nl.affine_range(n_tiles):
        col_start = n_tile * N_TILE
        for k_tile in nl.affine_range(k_tiles_per_block):
            k_row = k_start + k_tile * P_MAX
            nisa.dma_copy(
                dst=input_fp8[:, k_tile, col_start:col_start + N_TILE],
                src=saved_e4m3[
                    k_row:k_row + P_MAX,
                    col_start:col_start + N_TILE,
                ],
                dge_mode=nisa.dge_mode.hwdge,
            )

    for local_m_tile in nl.affine_range(mgroup):
        owned_m_tile = m_group * mgroup + local_m_tile
        output_row = (owned_m_tile * grid + program_id) * P_MAX
        # One DMA per m tile: [kblock, 128] -> [128 partitions, k_tiles, 128]
        # (16x fewer static-descriptor triggers on the sync engine).
        nisa.dma_copy(
            dst=grad_raw,
            src=grad_output[
                k_start:k_start + kblock,
                output_row:output_row + P_MAX,
            ].rearrange((("t", "p"), "m"), ("p", "t", "m"), {"p": P_MAX}),
            dge_mode=nisa.dge_mode.none,
        )
        grad_tile = grad_fp8[
            :,
            :,
            local_m_tile * P_MAX:(local_m_tile + 1) * P_MAX,
        ]
        nisa.activation(
            dst=grad_tile,
            op=nl.copy,
            data=grad_raw,
            scale=grad_qscale,
        )
        nisa.tensor_scalar(
            dst=grad_tile,
            data=grad_tile,
            op0=nl.minimum,
            operand0=E5M2_MAX,
            op1=nl.maximum,
            operand1=-E5M2_MAX,
        )

    for local_m_tile in nl.affine_range(mgroup):
        grad_start = local_m_tile * P_MAX
        owned_m_tile = m_group * mgroup + local_m_tile
        output_row = (owned_m_tile * grid + program_id) * P_MAX
        _matmul_wgrad_block_wide(
            grad_fp8[:, :, grad_start:grad_start + P_MAX],
            input_fp8,
            accumulators[:, local_m_tile, :],
            kblock,
            n_tiles * N_TILE,
            init,
            final,
            out,
            output_scale,
            output_row,
        )


def _store_wgrad_tile_static(
    out,
    accumulator,
    output_scale,
    output_row: int,
    col_start: int,
):
    """Scale one accumulator tile to BF16 and store it with static DMA descriptors."""
    result = nl.ndarray(
        (P_MAX, N_TILE),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.activation(
        dst=result,
        op=nl.copy,
        data=accumulator,
        scale=output_scale,
    )
    nisa.dma_copy(
        dst=out[
            output_row:output_row + P_MAX,
            col_start:col_start + N_TILE,
        ],
        src=result,
        dge_mode=nisa.dge_mode.hwdge,
    )


@nki.jit
def tensorwise_fp8_wgrad_saved_grouped_v1(
    grad_output,
    input_bits,
    input_inverse_scales,
    grad_scale_state,
    kblock: int,
    mgroup: int,
):
    """Compute strict FP8 wgrad with one persistent output-row group at a time.

    This is the c_fc schedule: retaining every output row would exceed the
    validated SBUF budget, while processing one N tile at a time requantizes
    the much larger grad-output tensor for every tile. A 12-row group keeps all
    N columns resident and reads each grad-output element exactly once.

    trn3 tuning (round 0): hardware DGE for the saved-activation
    loads and the output stores (they were software-DGE on GpSimd); the first
    contraction block initializes the accumulators from PSUM instead of a
    72 KB/partition memset that serialized the group boundary.
    """
    kernel_assert(len(grad_output.shape) == 2, "grad_output must be rank 2")
    kernel_assert(len(input_bits.shape) == 2, "saved input must be rank 2")
    K, M = grad_output.shape
    K_b, N = input_bits.shape
    kernel_assert(K == K_b, "wgrad contraction dimensions must match")
    kernel_assert(M % P_MAX == 0, "wgrad rows must be divisible by 128")
    kernel_assert(K % kblock == 0, "contraction must divide kblock")
    kernel_assert(K // kblock >= 2, "grouped wgrad needs at least two contraction blocks")
    kernel_assert(kblock % DOUBLE_ROW_K == 0, "kblock must divide into 256")
    kernel_assert(N % N_TILE == 0, "wgrad columns must be divisible by 512")
    kernel_assert(N <= 4096, "full-width wgrad matmul needs N <= 4096 (fp32 PSUM)")

    grid = nl.num_programs(0)
    program_id = nl.program_id(0)
    total_m_tiles = M // P_MAX
    kernel_assert(total_m_tiles % grid == 0, "M tiles must divide the grid")
    m_tiles_per_program = total_m_tiles // grid
    kernel_assert(
        m_tiles_per_program % mgroup == 0,
        "owned output rows must divide mgroup",
    )
    m_groups = m_tiles_per_program // mgroup
    n_tiles = N // N_TILE
    k_tiles_per_block = kblock // P_MAX
    k_blocks = K // kblock
    grouped_sbuf_bytes = (
        P_MAX * mgroup * N * 4
        + P_MAX * k_tiles_per_block * mgroup * P_MAX
        + P_MAX * k_tiles_per_block * P_MAX * 2
        + P_MAX * k_tiles_per_block * N
        + P_MAX * N_TILE * 2
    )
    kernel_assert(
        grouped_sbuf_bytes <= PERSISTENT_WGRAD_SBUF_BUDGET,
        "grouped wgrad buffers exceed the validated SBUF budget",
    )

    out = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    saved_e4m3 = input_bits.view(nl.float8_e4m3)
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
        dge_mode=nisa.dge_mode.hwdge,
    )
    nisa.dma_copy(
        dst=grad_inverse,
        src=grad_scale_state[program_id, 1, :, :],
        dge_mode=nisa.dge_mode.hwdge,
    )
    input_inverse = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=input_inverse,
        src=input_inverse_scales[program_id, 0:P_MAX, 0:1],
        dge_mode=nisa.dge_mode.hwdge,
    )
    output_scale = nl.ndarray(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.tensor_tensor(
        dst=output_scale,
        data1=grad_inverse,
        data2=input_inverse,
        op=nl.multiply,
    )

    for m_group in nl.affine_range(m_groups):
        accumulators = nl.ndarray(
            (P_MAX, mgroup, N),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        grad_fp8 = nl.ndarray(
            (P_MAX, k_tiles_per_block, mgroup * P_MAX),
            dtype=nl.float8_e5m2,
            buffer=nl.sbuf,
        )
        grad_raw = nl.ndarray(
            (P_MAX, k_tiles_per_block, P_MAX),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        input_fp8 = nl.ndarray(
            (P_MAX, k_tiles_per_block, N),
            dtype=nl.float8_e4m3,
            buffer=nl.sbuf,
        )

        # k block 0 initializes the accumulators straight from PSUM (no memset).
        _wgrad_group_kblock(
            grad_output, saved_e4m3, grad_qscale, grad_fp8, grad_raw, input_fp8,
            accumulators, 0, k_tiles_per_block, kblock, mgroup, n_tiles,
            m_group, grid, program_id, True,
        )
        for k_block_index in nl.affine_range(k_blocks - 2):
            k_start = (k_block_index + 1) * kblock
            _wgrad_group_kblock(
                grad_output, saved_e4m3, grad_qscale, grad_fp8, grad_raw, input_fp8,
                accumulators, k_start, k_tiles_per_block, kblock, mgroup, n_tiles,
                m_group, grid, program_id, False,
            )
        # last k block: each output tile is scaled and stored right after its
        # final accumulation so the stores overlap with the remaining matmuls.
        _wgrad_group_kblock(
            grad_output, saved_e4m3, grad_qscale, grad_fp8, grad_raw, input_fp8,
            accumulators, (k_blocks - 1) * kblock, k_tiles_per_block, kblock, mgroup,
            n_tiles, m_group, grid, program_id, False, True, out, output_scale,
        )

    return out
