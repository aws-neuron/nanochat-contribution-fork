"""BTHD-native MHA forward/backward kernels for Neuron training.

This is the performance-oriented implementation.  It keeps one 128x2048
score/probability section in SBUF, so the requested T=2048 shape performs a
single softmax rather than an online merge after every 512 keys.  Longer
sequences use the same online max/sum/output combination once per 2048-key
section.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.compiler import CompileConfig
from nki.isa import engine


_Q_TILE = 128
_MM1_TILE = 512
_V_TILE = 128
_MAX_SECTION = 2048
_Q_GROUPS_PER_PACK = 16
_BWD_Q_TILES_PER_GROUP = 4
_BWD_Q_GROUP_TOKENS = _BWD_Q_TILES_PER_GROUP * _Q_TILE
_FLOAT32_MIN = -3.4028235e38
# Pipelined single-section path: QK/EXP work is issued per 512-key-aligned chunk (<= 4 key
# tiles = one fp32 PSUM bank, the backend's per-matmul output limit) restricted to the visible
# key window; masks are pre-added into PSUM as bf16 constants (-3e38 absorbs any finite score
# exactly, and exp() of the scaled value is exactly 0, matching affine_select with _FLOAT32_MIN).
_QK_CHUNK_TILES = 4
_QK_NUM_SLOTS = 4
# Cross-plane software pipeline: QK runs _PIPE_QK_AHEAD groups ahead of PV, EXP _PIPE_EXP_AHEAD
# ahead; per-group state lives in _PIPE_RING-deep rings; K/V and Q packs are prefetched.
_PIPE_QK_AHEAD = 3
_PIPE_EXP_AHEAD = 2
# Per-config compile-time specialization (use_sliding_window is a trace-time constant): the
# 5-tile sliding-window groups run best with 4-deep rings and 2 TensorE P^T transposes per group,
# the up-to-16-tile full-causal groups with 8-deep rings and DMA transposes only.
_PIPE_RING_SWA = 4
_PIPE_RING_FULL = 8
_PE_TRANSPOSE_TILES_SWA = 2
_PE_TRANSPOSE_TILES_FULL = 0
_KV_RING = 3
_Q_PACK_RING = 4
# exp + transpose are issued per "super-chunk" of up to 8 key tiles starting at the window
# start (a 5-tile sliding window is a single exp / a single 5-block DMA transpose).
_EXP_SUPER_TILES = 8
_MASK_VALUE = -3.0e38
# Decoupled cross-plane pipeline: per iteration t the stages are
#   QK matmuls(t+QK_AHEAD) -> TSCR(t+QK_AHEAD-TSCR_LAG) -> EXP(t+EXP_AHEAD) [+ DMA transpose]
#   -> PE transposes + copies(t+TP_AHEAD) -> PV(t) -> normalize/store(t-NORM_LAG),
# so that every in-order engine queue only sees instructions whose inputs were produced in an
# earlier iteration (no cross-engine round trips inside one iteration).
_V3_QK_AHEAD_SWA = 12
_V3_EXP_AHEAD_SWA = 10
_V3_TSCR_LAG_SWA = 1
_V3_NORM_LAG_SWA = 1
_V3_QK_AHEAD_FULL = 6  # with 8 (and a ring of 10) the full-causal pipeline spills on trn2
_V3_EXP_AHEAD_FULL = 5
_V3_TSCR_LAG_FULL = 0
_V3_NORM_LAG_FULL = 1
_V3_TP_AHEAD = 1
_V3_RING_SWA = 16
_V3_RING_FULL = 8
_V3_PE_SWA = 3
_V3_PE_FULL = 2
_V3_DMA_TP_IN_EXP_STAGE = True
_V3_TP_PSUM_RING = 2
_V3_QK_MM_FIRST = True
_V3_COPY_ENGINES = 'ssv'
# per-group-parity engine balance (sw512 / sw0): PE-tile copy engines and normalize engine for even / odd groups
_V8_COPY_ENGINES_SWA = ('ssv', 'ssv')
_V8_NORM_ENGINE_SWA = ('v', 'v')
_V8_COPY_ENGINES_FULL = ('ss', 'ss')
_V8_NORM_ENGINE_FULL = ('v', 'v')
_V3_RECIP_BATCH = 4
_V3_SLOT_COLS_MIN = 0
_V3_REL_INDEX = True
_V3_STORE_BATCH = 1
_V3_SCORES_LIST = False
_V3_STORE_DGE = nisa.dge_mode.swdge
_V3_VLOAD_DGE = nisa.dge_mode.unknown
_V3_FINAL_RING = 4
_V3_PV_RING = 4
# next-set loads issued <load_at> groups into the set, anchored (depends_on) to the PV stage
_V9_LOAD_AT_SWA = 3
_V9_LOAD_LAG_SWA = 0
_V9_LOAD_AT_FULL = 0
_V9_LOAD_LAG_FULL = 0
_V9_LOAD_ANCHOR_SWA = True
_V9_LOAD_ANCHOR_FULL = False
_V9_Q_RING_SWA = 3
_V9_Q_RING_FULL = 2
_V9_ANCHOR_V = False
_V6_HEADS_PER_SET = 2
_V6_KV_RING = 3
_V6_VLOAD_SPLIT = 0
_V6_HEADS_PER_SET_FULL = 2
_V6_KV_RING_FULL = 3
# Probability ring slots hold both heads of a q-tile pair ((128, 2, cols) tensors); the DMA transposes of
# both heads are issued with ONE dma_transpose after the second head's exp (half the DMA instructions).
_V6_TP_PAIR = True
# K^T of both heads of a set in ONE 4D DMA transpose; Q^T packs of both heads in ONE 4D DMA transpose.
_V6_PAIR_LOADS = True
# 5-tile sliding windows: the sliding-masked first tile and the causal-masked last tile are read side by side
# (strided K^T moving operand) into one PSUM slot so that one identity matmul applies both masks; the window
# columns are then ordered [lo, hi, lo+1, lo+2, lo+3] (DMA transposes take columns 0-255, PE the rest).
_V3_MASK_PAIR = True
# LSE of both heads of a set stored with one DMA.
_V6_LSE_PAIR = True
# Scheduling pins (nisa depends_on edges)
_V7_PIN_RECIP = False
_V7_PIN_TSCR = False
_V7_PIN_EXP = False
_V7_PIN_NORM = False
_V7_PIN_COPIES = False
_V7_VCHAIN = True  # as schedule-only edges: same-engine order for the scheduler, no semaphore wait
_V7_SCHAIN = True
_V7_TCHAIN = False
_V7_PVANCH = True
_V7_TPIN = False
# the next head set's K^T and Q^T transposing loads are issued as separate single DMAs at separate triggers
# (qk_r values), each anchored (depends_on) to the latest P@V matmul, so the two 1 MiB blocks never sit back to back on
# the serialized DMA-transpose queue (LOAD_PIECES > 1 would split a load into key/token ranges: measured slower, as
# the dependencies are tracked per destination tensor).  V load (SWDGE, GpSimd queue) unchanged.
_V10_LOAD_PIECES_SWA = 1
_V10_LOAD_PIECES_FULL = 1
_V10_K_TRIG_SWA = (3,)
_V10_Q_TRIG_SWA = (19,)
_V10_K_TRIG_FULL = (6,)
_V10_Q_TRIG_FULL = (20,)
_V10_V_TRIG_SWA = 3
_V10_V_TRIG_FULL = 0
_V10_LOAD_ANCHOR_SWA = True
_V10_LOAD_ANCHOR_FULL = True
# batched reciprocal issued this many iterations ahead of its first consuming normalize
# (sw0, TSCR lag 0): chunk 0 of the next group's QK matmul is issued right after this iteration's P@V so the
# first TSCR chunk of a group never waits on a matmul queued behind the previous P@V in the in-order Tensor queue.
_V10_QK0_EARLY_SWA = False
_V10_QK0_EARLY_FULL = True
_V10_RECIP_LEAD_SWA = 2
_V10_RECIP_LEAD_FULL = 0
# one reciprocal instruction for both heads of a set (needs (b-1)*HS + HS-1 <= exp_ahead + norm_lag - lead)
_V10_RECIP_PAIR_SWA = True
_V10_RECIP_PAIR_FULL = False
# the next head set's K^T and Q^T transposing
# loads are issued as hardware-DGE dma_transposes (dge_mode=hwdge, 16-token source tiles, 4 KB packets on the hardware
# DGE queues) instead of static-descriptor transposes; same data, same destination layout, same anchored triggers.
# Measured: sw512 -1.5%, sw0 -0.6%, outputs bitwise identical.
_V11_HWDGE_KQ_SWA = True
_V11_HWDGE_KQ_FULL = True
# the initial (set 0) K^T / Q^T loads use the same hardware-DGE form.
_V11_HWDGE_KQ_INIT_SWA = True
_V11_HWDGE_KQ_INIT_FULL = True
# trn2 has no nisa.activate2.  'negmin': the TSCR stage produces -rowmax (scale by -softmax_scale,
# min-reduce) so the exp is a plain nisa.activation(exp, scale=-1.0, bias=-rowmax) with no extra instruction;
# 'negate': keep +rowmax and negate it once per group on ScalarE (activation copy, scale=-1.0).  Both are fp32-bitwise
# equivalent to activate2's (in - imm0).
_G3_EXP_FORM = 'negmin'


_PSUM_BANK_BYTES = 2048


def _psum_at(shape, dtype, bank, offset=0):
    """A PSUM tile at a fixed bank and byte offset (per partition). The forward kernel places all its PSUM
    tiles this way, as nkilib attention does, so that they share the 8 PSUM banks (2 KiB each) instead of taking
    one bank per tensor."""
    return nl.ndarray(shape, dtype=dtype, buffer=nl.psum, address=(0, bank * _PSUM_BANK_BYTES + offset))


def _load_kv_plane(
    k,
    v,
    k_transposed,
    v_bthd,
    plane_base,
    token_stride,
    head_dim,
    seqlen,
    q_tiles_per_plane,
):
    """Load one (batch, head) plane of K (transposed, [D, T]) and V ([128, T//128, D])."""

    native_plane_pattern = [
        [token_stride, _Q_TILE],
        [_Q_TILE * token_stride, q_tiles_per_plane],
        [1, head_dim],
    ]
    local_plane_pattern = [
        [q_tiles_per_plane * head_dim, _Q_TILE],
        [head_dim, q_tiles_per_plane],
        [1, head_dim],
    ]
    nisa.dma_transpose(
        dst=k_transposed.ap(
            pattern=[
                [seqlen, head_dim],
                [_Q_TILE, q_tiles_per_plane],
                [1, _Q_TILE],
            ],
            offset=0,
        ),
        src=k.ap(pattern=native_plane_pattern, offset=plane_base),
    )
    nisa.dma_copy(
        dst=v_bthd.ap(pattern=local_plane_pattern, offset=0),
        src=v.ap(pattern=native_plane_pattern, offset=plane_base),
        dge_mode=_V3_VLOAD_DGE,
    )


def _load_q_pack(
    q,
    q_pack,
    q_offset,
    token_stride,
    head_dim,
    pack_capacity,
    num_tokens,
):
    """Load up to eight consecutive native-BTHD Q groups at once."""

    nisa.dma_transpose(
        dst=q_pack.ap(
            [
                [pack_capacity, head_dim],
                [1, 1],
                [1, 1],
                [1, num_tokens],
            ]
        ),
        src=q.ap(
            [
                [token_stride, num_tokens],
                [1, 1],
                [1, 1],
                [1, head_dim],
            ],
            offset=q_offset,
        ),
    )


def _group_key_range(q_tile, causal, use_sliding_window, sliding_window, token_tiles):
    """Return (lo_tile, hi_tile): the inclusive range of 128-key tiles with any visible key
    for Q group ``q_tile`` (causal: k <= q; sliding: q - k < sliding_window)."""

    hi_tile = q_tile if causal else token_tiles - 1
    lo_tile = 0
    if use_sliding_window:
        lo_tile = max(0, (q_tile * _Q_TILE - sliding_window + 1) // _Q_TILE)
    return lo_tile, hi_tile


def _group_chunks(lo_tile, hi_tile):
    """Split [lo_tile, hi_tile] into (first_tile, num_tiles) chunks aligned to 512-key blocks."""

    chunks = []
    for block in range(lo_tile // _QK_CHUNK_TILES, hi_tile // _QK_CHUNK_TILES + 1):
        first = max(lo_tile, block * _QK_CHUNK_TILES)
        last = min(hi_tile, block * _QK_CHUNK_TILES + _QK_CHUNK_TILES - 1)
        chunks.append((first, last - first + 1))
    return chunks


def _group_superchunks(lo_tile, hi_tile):
    """Split [lo_tile, hi_tile] into (first_tile, num_tiles) runs of <= _EXP_SUPER_TILES."""

    chunks = []
    tile = lo_tile
    while tile <= hi_tile:
        num = min(_EXP_SUPER_TILES, hi_tile - tile + 1)
        chunks.append((tile, num))
        tile += num
    return chunks


def _build_mask_constants(identity, zero_tile, mask_causal, mask_sliding):
    """Build the two constant 128x128 additive masks (0 = visible, _MASK_VALUE = masked).

    ``mask_causal`` is used on the diagonal key tile (k_start == q_start): visible iff f <= p.
    ``mask_sliding`` is used on the tile with k_start == q_start - sliding_window: visible iff
    q - k < sliding_window, i.e. f > p.  Both are added into PSUM with an identity matmul.
    """

    nisa.memset(dst=zero_tile, value=0.0)
    nisa.affine_select(
        dst=mask_causal,
        pattern=[[-1, _Q_TILE]],
        offset=0,
        channel_multiplier=1,
        cmp_op=nl.greater_equal,
        on_true_tile=zero_tile,
        on_false_value=_MASK_VALUE,
    )
    nisa.affine_select(
        dst=mask_sliding,
        pattern=[[1, _Q_TILE]],
        offset=-1,
        channel_multiplier=-1,
        cmp_op=nl.greater_equal,
        on_true_tile=zero_tile,
        on_false_value=_MASK_VALUE,
    )


def _qk_slot(slot_base, chunk_index, num_chunks):
    return (slot_base + chunk_index) % _QK_NUM_SLOTS


def _win_col(tile, lo_tile, hi_tile=-1, use_sliding_window=False):
    """Column of key tile ``tile`` inside a ring slot (relative to the window start or absolute).
    Paired 5-tile sliding windows use the column order [lo, hi, lo+1, lo+2, lo+3]."""
    if _window_is_paired(lo_tile, hi_tile, use_sliding_window):
        if tile == lo_tile:
            return 0
        if tile == hi_tile:
            return _Q_TILE
        return (tile - lo_tile + 1) * _Q_TILE
    if _V3_REL_INDEX:
        return (tile - lo_tile) * _Q_TILE
    return tile * _Q_TILE


def _window_is_paired(lo_tile, hi_tile, use_sliding_window):
    return _V3_MASK_PAIR and _V3_REL_INDEX and use_sliding_window and hi_tile - lo_tile == 4


def _qk_matmuls_group(
    q_tile,
    q_transposed,
    k_transposed,
    qk_psum_slots,
    identity,
    mask_causal,
    mask_sliding,
    causal,
    use_sliding_window,
    sliding_window,
    token_tiles,
    slot_base,
    k_pair=None,
    k_h=0,
    mask_pair=None,
    pin_first=False,
    pin_inst=None,
    p0_lo=0,
    p0_hi=99,
    do_masks=True,
):
    """TensorE part of QK for one Q group: one matmul per <=512-key chunk into fp32 PSUM plus the
    identity @ mask accumulations.  With _V3_QK_MM_FIRST all chunk matmuls are issued before the
    (dependent) mask matmuls so that independent matmuls overlap on the PE arrays.  Paired 5-tile
    windows: chunk 0 = tiles (lo, hi) read with a strided K^T access pattern + one mask matmul with
    the [sliding | causal] mask pair, chunk 1 = tiles lo+1..lo+3."""

    q_start = q_tile * _Q_TILE
    q_end = q_start + _Q_TILE
    lo_tile, hi_tile = _group_key_range(q_tile, causal, use_sliding_window, sliding_window, token_tiles)
    if _window_is_paired(lo_tile, hi_tile, use_sliding_window) and k_pair is not None:
        seqlen = k_pair.shape[2]
        heads = k_pair.shape[1]
        slot0 = qk_psum_slots[_qk_slot(slot_base, 0, 2)]
        slot1 = qk_psum_slots[_qk_slot(slot_base, 1, 2)]
        h0 = nisa.nc_matmul(
            dst=slot0[:, nl.ds(0, 2 * _Q_TILE)],
            stationary=q_transposed,
            moving=k_pair.ap(
                [
                    [heads * seqlen, k_pair.shape[0]],
                    [(hi_tile - lo_tile) * _Q_TILE, 2],
                    [1, _Q_TILE],
                ],
                offset=k_h * seqlen + lo_tile * _Q_TILE,
            ),
            accumulate=False,
        )
        if pin_first:
            h0.depends_on(pin_inst)
        h1 = nisa.nc_matmul(
            dst=slot1[:, nl.ds(0, 3 * _Q_TILE)],
            stationary=q_transposed,
            moving=k_transposed[:, nl.ds((lo_tile + 1) * _Q_TILE, 3 * _Q_TILE)],
            accumulate=False,
        )
        if _V7_TCHAIN:
            h1.depends_on(h0)
        h2 = nisa.nc_matmul(
            dst=slot0[:, nl.ds(0, 2 * _Q_TILE)],
            stationary=identity,
            moving=mask_pair,
            accumulate=True,
        )
        if _V7_TCHAIN:
            h2.depends_on(h1)
        return h2
    chunks = _group_chunks(lo_tile, hi_tile)
    num_chunks = len(chunks)
    last = None
    have_last = False

    for phase in range(2 if _V3_QK_MM_FIRST else 1):
        for chunk_index in range(num_chunks):
            first_tile = chunks[chunk_index][0]
            num_tiles = chunks[chunk_index][1]
            cols = num_tiles * _Q_TILE
            k0 = first_tile * _Q_TILE
            qk_psum = qk_psum_slots[_qk_slot(slot_base, chunk_index, num_chunks)][:, nl.ds(0, cols)]
            if phase == 0 and p0_lo <= chunk_index and chunk_index < p0_hi:
                h = nisa.nc_matmul(
                    dst=qk_psum,
                    stationary=q_transposed,
                    moving=k_transposed[:, nl.ds(k0, cols)],
                    accumulate=False,
                )
                if have_last and _V7_TCHAIN:
                    h.depends_on(last)
                if (not have_last) and pin_first:
                    h.depends_on(pin_inst)
                last = h
                have_last = True
            if (phase == 1 or not _V3_QK_MM_FIRST) and do_masks:
                for tile in range(first_tile, first_tile + num_tiles):
                    k_start = tile * _Q_TILE
                    needs_causal_mask = causal and (q_start < k_start + _Q_TILE)
                    needs_sliding_mask = use_sliding_window and ((q_end - 1) - k_start >= sliding_window)
                    assert not (needs_causal_mask and needs_sliding_mask), "window must span >= 2 tiles"
                    if needs_sliding_mask:
                        assert k_start == q_start - sliding_window, "sliding mask tile offset"
                    mask = None
                    if needs_causal_mask:
                        mask = mask_causal
                    elif needs_sliding_mask:
                        mask = mask_sliding
                    if mask is not None:
                        h = nisa.nc_matmul(
                            dst=qk_psum[:, nl.ds((tile - first_tile) * _Q_TILE, _Q_TILE)],
                            stationary=identity,
                            moving=mask,
                            accumulate=True,
                        )
                        if _V7_TCHAIN:
                            h.depends_on(last)
                        last = h
    return last


def _qk_reduce_group(
    q_tile,
    qk_psum_slots,
    scores,
    row_max,
    softmax_scale,
    causal,
    use_sliding_window,
    sliding_window,
    token_tiles,
    slot_base,
    pin_first=False,
    pin_inst=None,
):
    """VectorE part of QK for one Q group: scale + row max (chained through the accumulator)
    from PSUM into the fp32 ``scores`` buffer; the max is written once into ``row_max``."""

    lo_tile, hi_tile = _group_key_range(q_tile, causal, use_sliding_window, sliding_window, token_tiles)
    if _window_is_paired(lo_tile, hi_tile, use_sliding_window):
        chunks = [(lo_tile, 2), (lo_tile + 1, 3)]
    else:
        chunks = _group_chunks(lo_tile, hi_tile)
    num_chunks = len(chunks)
    last = None
    for chunk_index in range(num_chunks):
        first_tile = chunks[chunk_index][0]
        num_tiles = chunks[chunk_index][1]
        cols = num_tiles * _Q_TILE
        qk_psum = qk_psum_slots[_qk_slot(slot_base, chunk_index, num_chunks)][:, nl.ds(0, cols)]
        last_chunk = chunk_index == num_chunks - 1
        h = nisa.tensor_scalar_reduce(
            dst=scores[:, nl.ds(_win_col(first_tile, lo_tile, hi_tile, use_sliding_window), cols)],
            data=qk_psum,
            op0=nl.multiply,
            operand0=(-softmax_scale) if _G3_EXP_FORM == 'negmin' else softmax_scale,
            reduce_op=nl.minimum if _G3_EXP_FORM == 'negmin' else nl.maximum,
            reduce_res=row_max if last_chunk else None,
            reduce_cmd=nisa.reduce_cmd.reset_reduce if chunk_index == 0 else nisa.reduce_cmd.reduce,
        )
        if chunk_index == 0 and pin_first:
            h.depends_on(pin_inst, kind=nisa.dependency.schedule)
        if chunk_index > 0 and _V7_VCHAIN:
            h.depends_on(last, kind=nisa.dependency.schedule)
        last = h
    return last


def _dma_transpose_group(
    q_tile,
    probabilities,
    probabilities_tp,
    pe_transpose_tiles,
    causal,
    use_sliding_window,
    sliding_window,
    slot_cols,
    token_tiles,
):
    """DMA transposes of the probability window (all but the last ``pe_transpose_tiles`` tiles)."""

    lo_tile, hi_tile = _group_key_range(q_tile, causal, use_sliding_window, sliding_window, token_tiles)
    chunks = _group_superchunks(lo_tile, hi_tile)
    for chunk_index in range(len(chunks)):
        first_tile = chunks[chunk_index][0]
        num_tiles = chunks[chunk_index][1]
        c0 = _win_col(first_tile, lo_tile, hi_tile, use_sliding_window)
        if chunk_index == len(chunks) - 1:
            num_tiles -= min(pe_transpose_tiles, num_tiles)
        if num_tiles > 0:
            nisa.dma_transpose(
                dst=probabilities_tp.ap(
                    [
                        [slot_cols, _V_TILE],
                        [1, 1],
                        [_Q_TILE, num_tiles],
                        [1, _Q_TILE],
                    ],
                    offset=c0,
                ),
                src=probabilities.ap(
                    [
                        [slot_cols, _Q_TILE],
                        [1, 1],
                        [_Q_TILE, num_tiles],
                        [1, _Q_TILE],
                    ],
                    offset=c0,
                ),
            )


def _dma_transpose_pair(
    q_tile,
    prob_pair,
    prob_tp_pair,
    pe_transpose_tiles,
    causal,
    use_sliding_window,
    sliding_window,
    slot_cols,
    token_tiles,
):
    """DMA transposes of the probability windows of BOTH heads of a q-tile pair slot (all but the
    last ``pe_transpose_tiles`` tiles) with one dma_transpose per super-chunk."""

    lo_tile, hi_tile = _group_key_range(q_tile, causal, use_sliding_window, sliding_window, token_tiles)
    chunks = _group_superchunks(lo_tile, hi_tile)
    for chunk_index in range(len(chunks)):
        first_tile = chunks[chunk_index][0]
        num_tiles = chunks[chunk_index][1]
        c0 = _win_col(first_tile, lo_tile, hi_tile, use_sliding_window)
        if chunk_index == len(chunks) - 1:
            num_tiles -= min(pe_transpose_tiles, num_tiles)
        if num_tiles > 0:
            nisa.dma_transpose(
                dst=prob_tp_pair.ap(
                    [
                        [2 * slot_cols, _V_TILE],
                        [slot_cols, 2],
                        [_Q_TILE, num_tiles],
                        [1, _Q_TILE],
                    ],
                    offset=c0,
                ),
                src=prob_pair.ap(
                    [
                        [2 * slot_cols, _Q_TILE],
                        [slot_cols, 2],
                        [_Q_TILE, num_tiles],
                        [1, _Q_TILE],
                    ],
                    offset=c0,
                ),
            )


def _exp_only_group(
    q_tile,
    scores,
    row_max,
    probabilities,
    section_sum,
    causal,
    use_sliding_window,
    sliding_window,
    token_tiles,
    pin_first=False,
    pin_inst=None,
):
    """ScalarE part: exp(scores - row_max) over the visible window, row sum chained through the
    accumulator and written once into ``section_sum``."""

    lo_tile, hi_tile = _group_key_range(q_tile, causal, use_sliding_window, sliding_window, token_tiles)
    chunks = _group_superchunks(lo_tile, hi_tile)
    last = None
    for chunk_index in range(len(chunks)):
        first_tile = chunks[chunk_index][0]
        num_tiles = chunks[chunk_index][1]
        cols = num_tiles * _Q_TILE
        c0 = _win_col(first_tile, lo_tile, hi_tile, use_sliding_window)
        last_chunk = chunk_index == len(chunks) - 1
        if _G3_EXP_FORM == 'negmin':
            # row_max holds -max and scores hold -s: exp(-1.0 * (-s) + (-max)) == exp(s - max) bitwise in fp32
            neg_row_max = row_max
            exp_scale = -1.0
        else:
            if chunk_index == 0:
                neg_row_max = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=neg_row_max, op=nl.copy, data=row_max, scale=-1.0)
            exp_scale = 1.0
        h = nisa.activation(
            dst=probabilities[:, nl.ds(c0, cols)],
            op=nl.exp,
            data=scores[:, nl.ds(c0, cols)],
            bias=neg_row_max,
            scale=exp_scale,
            reduce_op=nl.add,
            reduce_res=section_sum if last_chunk else None,
            reduce_cmd=nisa.reduce_cmd.reset_reduce if chunk_index == 0 else nisa.reduce_cmd.reduce,
        )
        if chunk_index == 0 and pin_first:
            h.depends_on(pin_inst)
        if chunk_index > 0 and _V7_SCHAIN:
            h.depends_on(last)
        last = h
    return last


def _pe_transpose_group(
    q_tile,
    probabilities,
    probabilities_tp,
    tp_psums,
    pe_transpose_tiles,
    causal,
    use_sliding_window,
    sliding_window,
    token_tiles,
    pin_t=False,
    pin_t_inst=None,
    pin_s=False,
    pin_s_inst=None,
    pin_v=False,
    pin_v_inst=None,
    copy_engines=_V3_COPY_ENGINES,
):
    """TensorE transposes (bf16 PSUM) + ScalarE copies for the last ``pe_transpose_tiles`` key
    tiles of the window."""

    lo_tile, hi_tile = _group_key_range(q_tile, causal, use_sliding_window, sliding_window, token_tiles)
    num_pe = min(pe_transpose_tiles, hi_tile - lo_tile + 1)
    last_t = None
    last_s = None
    last_v = None
    have_t = False
    have_s = False
    have_v = False
    for j in range(num_pe):
        pe_col = (hi_tile - lo_tile + 1 - num_pe + j) * _Q_TILE
        h = nisa.nc_transpose(
            dst=tp_psums[j],
            data=probabilities[:, nl.ds(pe_col, _Q_TILE)],
            engine=engine.tensor,
        )
        if have_t and _V7_TCHAIN:
            h.depends_on(last_t)
        if (not have_t) and pin_t:
            h.depends_on(pin_t_inst)
        last_t = h
        have_t = True
    for j in range(num_pe):
        pe_col = (hi_tile - lo_tile + 1 - num_pe + j) * _Q_TILE
        copy_engine = copy_engines[j] if j < len(copy_engines) else "s"
        h = nisa.tensor_copy(
            dst=probabilities_tp[:, nl.ds(pe_col, _Q_TILE)],
            src=tp_psums[j],
            engine=engine.vector if copy_engine == "v" else engine.scalar,
        )
        if copy_engine == "v":
            if have_v and _V7_VCHAIN:
                h.depends_on(last_v, kind=nisa.dependency.schedule)
            if (not have_v) and pin_v:
                h.depends_on(pin_v_inst, kind=nisa.dependency.schedule)
            last_v = h
            have_v = True
        else:
            if have_s and _V7_SCHAIN:
                h.depends_on(last_s)
            if (not have_s) and pin_s:
                h.depends_on(pin_s_inst)
            last_s = h
            have_s = True
    return last_t, have_t, last_s, have_s, last_v, have_v


def _load_k_head(k, k_transposed, plane_base, token_stride, head_dim, seqlen, q_tiles_per_plane):
    """Transposing load of one head's K plane into k_transposed [D, T]."""

    nisa.dma_transpose(
        dst=k_transposed.ap(
            pattern=[
                [seqlen, head_dim],
                [_Q_TILE, q_tiles_per_plane],
                [1, _Q_TILE],
            ],
            offset=0,
        ),
        src=k.ap(
            pattern=[
                [token_stride, _Q_TILE],
                [_Q_TILE * token_stride, q_tiles_per_plane],
                [1, head_dim],
            ],
            offset=plane_base,
        ),
    )


def _load_v_set(v, v_set, set_base, token_stride, head_dim, q_tiles_per_plane, heads_per_set):
    """Load V of ``heads_per_set`` adjacent heads of one batch into v_set[128, T//128, heads, D]
    (heads_per_set*D contiguous elements per token in HBM -> larger DMA descriptors)."""

    return nisa.dma_copy(
        dst=v_set.ap(
            pattern=[
                [q_tiles_per_plane * heads_per_set * head_dim, _Q_TILE],
                [heads_per_set * head_dim, q_tiles_per_plane],
                [head_dim, heads_per_set],
                [1, head_dim],
            ],
            offset=0,
        ),
        src=v.ap(
            pattern=[
                [token_stride, _Q_TILE],
                [_Q_TILE * token_stride, q_tiles_per_plane],
                [head_dim, heads_per_set],
                [1, head_dim],
            ],
            offset=set_base,
        ),
        dge_mode=_V3_VLOAD_DGE,
    )


def _load_v_head_of_set(v, v_set, set_base, h, token_stride, head_dim, q_tiles_per_plane, heads_per_set):
    """Load one head's V plane into slot ``h`` of a head-set V buffer [128, T//128, heads, D]."""

    nisa.dma_copy(
        dst=v_set.ap(
            pattern=[
                [q_tiles_per_plane * heads_per_set * head_dim, _Q_TILE],
                [heads_per_set * head_dim, q_tiles_per_plane],
                [1, head_dim],
            ],
            offset=h * head_dim,
        ),
        src=v.ap(
            pattern=[
                [token_stride, _Q_TILE],
                [_Q_TILE * token_stride, q_tiles_per_plane],
                [1, head_dim],
            ],
            offset=set_base + h * head_dim,
        ),
        dge_mode=_V3_VLOAD_DGE,
    )


def _load_k_pair(k, k_pair, set_base, token_stride, head_dim, seqlen, q_tiles_per_plane):
    """Transposing load of the K planes of two adjacent heads into k_pair [D, 2, T] with one DMA."""

    return nisa.dma_transpose(
        dst=k_pair.ap(
            pattern=[
                [2 * seqlen, head_dim],
                [seqlen, 2],
                [_Q_TILE, q_tiles_per_plane],
                [1, _Q_TILE],
            ],
            offset=0,
        ),
        src=k.ap(
            pattern=[
                [token_stride, _Q_TILE],
                [head_dim, 2],
                [_Q_TILE * token_stride, q_tiles_per_plane],
                [1, head_dim],
            ],
            offset=set_base,
        ),
    )


def _load_k_pair_range(k, k_pair, set_base, token_stride, head_dim, seqlen, tile0, num_tiles, hwdge=False):
    """Transposing load of key tiles [tile0, tile0+num_tiles) of the K planes of two adjacent heads into k_pair.

    ``hwdge``: hardware-DGE transpose (16-token source tiles; the DGE transpose requires the source partition
    count in {1, 2, 4, 8, 16}) instead of static descriptors.  Same data movement, same destination layout.
    """

    if hwdge:
        ntok = num_tiles * _Q_TILE
        return nisa.dma_transpose(
            dst=k_pair.ap(
                pattern=[
                    [2 * seqlen, head_dim],
                    [seqlen, 2],
                    [16, ntok // 16],
                    [1, 16],
                ],
                offset=tile0 * _Q_TILE,
            ),
            src=k.ap(
                pattern=[
                    [token_stride, 16],
                    [head_dim, 2],
                    [16 * token_stride, ntok // 16],
                    [1, head_dim],
                ],
                offset=set_base + tile0 * _Q_TILE * token_stride,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
    return nisa.dma_transpose(
        dst=k_pair.ap(
            pattern=[
                [2 * seqlen, head_dim],
                [seqlen, 2],
                [_Q_TILE, num_tiles],
                [1, _Q_TILE],
            ],
            offset=tile0 * _Q_TILE,
        ),
        src=k.ap(
            pattern=[
                [token_stride, _Q_TILE],
                [head_dim, 2],
                [_Q_TILE * token_stride, num_tiles],
                [1, head_dim],
            ],
            offset=set_base + tile0 * _Q_TILE * token_stride,
        ),
    )


def _load_q_pack_pair_range(q, q_pack_pair, q_offset, token_stride, head_dim, pack_capacity, tok0, num_tokens, hwdge=False):
    """Load tokens [tok0, tok0+num_tokens) of the Q^T packs of two adjacent heads (``hwdge``: hardware-DGE form)."""

    if hwdge:
        return nisa.dma_transpose(
            dst=q_pack_pair.ap(
                pattern=[
                    [2 * pack_capacity, head_dim],
                    [pack_capacity, 2],
                    [16, num_tokens // 16],
                    [1, 16],
                ],
                offset=tok0,
            ),
            src=q.ap(
                pattern=[
                    [token_stride, 16],
                    [head_dim, 2],
                    [16 * token_stride, num_tokens // 16],
                    [1, head_dim],
                ],
                offset=q_offset + tok0 * token_stride,
            ),
            dge_mode=nisa.dge_mode.hwdge,
        )
    return nisa.dma_transpose(
        dst=q_pack_pair.ap(
            [
                [2 * pack_capacity, head_dim],
                [pack_capacity, 2],
                [1, 1],
                [1, num_tokens],
            ],
            offset=tok0,
        ),
        src=q.ap(
            [
                [token_stride, num_tokens],
                [head_dim, 2],
                [1, 1],
                [1, head_dim],
            ],
            offset=q_offset + tok0 * token_stride,
        ),
    )


def _load_q_pack_pair(q, q_pack_pair, q_offset, token_stride, head_dim, pack_capacity, num_tokens):
    """Load the Q^T packs of two adjacent heads into q_pack_pair [D, 2, pack_capacity] with one DMA."""

    return nisa.dma_transpose(
        dst=q_pack_pair.ap(
            [
                [2 * pack_capacity, head_dim],
                [pack_capacity, 2],
                [1, 1],
                [1, num_tokens],
            ]
        ),
        src=q.ap(
            [
                [token_stride, num_tokens],
                [head_dim, 2],
                [1, 1],
                [1, head_dim],
            ],
            offset=q_offset,
        ),
    )


def _load_set_kv(k, v, k_heads, v_set, set_base, token_stride, head_dim, seqlen, q_tiles_per_plane, heads_per_set,
                 anchor=False, anchor_inst=None):
    hk = None
    hv = None
    if _V6_PAIR_LOADS and heads_per_set == 2:
        hk = _load_k_pair(k, k_heads[2], set_base, token_stride, head_dim, seqlen, q_tiles_per_plane)
        if anchor:
            hk.depends_on(anchor_inst)
    else:
        for h in range(heads_per_set):
            _load_k_head(k, k_heads[h], set_base + h * head_dim, token_stride, head_dim, seqlen, q_tiles_per_plane)
    if _V6_VLOAD_SPLIT or heads_per_set == 1:
        for h in range(heads_per_set):
            _load_v_head_of_set(v, v_set, set_base, h, token_stride, head_dim, q_tiles_per_plane, heads_per_set)
    else:
        hv = _load_v_set(v, v_set, set_base, token_stride, head_dim, q_tiles_per_plane, heads_per_set)
        if anchor and _V9_ANCHOR_V:
            hv.depends_on(anchor_inst)


def _load_pack_block(q, packs, set_base, block, token_stride, head_dim, q_pack_capacity, heads_per_set,
                     anchor=False, anchor_inst=None):
    if _V6_PAIR_LOADS and heads_per_set == 2:
        hq = _load_q_pack_pair(q, packs[2], set_base + block * q_pack_capacity * token_stride, token_stride, head_dim,
                               q_pack_capacity, q_pack_capacity)
        if anchor:
            hq.depends_on(anchor_inst)
        return
    for h in range(heads_per_set):
        _load_q_pack(
            q,
            packs[h],
            set_base + h * head_dim + block * q_pack_capacity * token_stride,
            token_stride,
            head_dim,
            q_pack_capacity,
            q_pack_capacity,
        )


def _compute_pv_group_set(
    q_tile,
    probabilities_tp,
    v_set,
    h,
    pv_psum,
    causal,
    use_sliding_window,
    sliding_window,
    token_tiles,
    pin_first=False,
    pin_inst=None,
):
    """P @ V over the visible key tiles with V taken from head ``h`` of a head-set V buffer."""

    lo_tile, hi_tile = _group_key_range(q_tile, causal, use_sliding_window, sliding_window, token_tiles)
    last = None
    for tile in range(lo_tile, hi_tile + 1):
        hh = nisa.nc_matmul(
            dst=pv_psum,
            stationary=probabilities_tp[:, nl.ds(_win_col(tile, lo_tile, hi_tile, use_sliding_window), _Q_TILE)],
            moving=v_set[:, tile, h, :],
            accumulate=tile > lo_tile,
        )
        if tile == lo_tile and pin_first:
            hh.depends_on(pin_inst)
        if tile > lo_tile and _V7_TCHAIN:
            hh.depends_on(last)
        last = hh
    return last


def _store_lse_plane(lse, plane_id, q_tiles_per_plane, lse_plane):
    """Store the (128, T//128) LSE block of one (batch, head) plane with one DMA."""

    nisa.dma_copy(
        dst=lse.ap(
            pattern=[[q_tiles_per_plane, _Q_TILE], [1, q_tiles_per_plane]],
            offset=plane_id * _Q_TILE * q_tiles_per_plane,
        ),
        src=lse_plane,
    )


def _store_lse_from_running_state(
    lse,
    lse_offset,
    q_tiles_per_plane,
    running_max,
    running_sum,
    log_sum,
    lse_value,
):
    """Store LSE from the positive-max online softmax state."""

    nisa.activation(dst=log_sum, op=nl.log, data=running_sum)
    nisa.tensor_tensor(
        dst=lse_value,
        data1=running_max,
        data2=log_sum,
        op=nl.add,
    )
    nisa.dma_copy(
        dst=lse.ap(
            pattern=[[q_tiles_per_plane, _Q_TILE], [1, 1]],
            offset=lse_offset,
        ),
        src=lse_value,
    )


@nki.jit
def bthd_flash_fwd(
    q: nl.ndarray,
    k: nl.ndarray,
    v: nl.ndarray,
    softmax_scale: float = None,
    causal: bool = True,
    sliding_window: int = 0,
    return_lse: bool = False,
):
    """Compute MHA directly on equal-shaped ``(B,T,H,D)`` tensors.

    Supported shapes are bf16/fp16 MHA with ``D <= 128``, ``D % 16 == 0``,
    and ``T`` divisible by 512.  For ``T > 2048``, T must also be divisible by
    2048.  Both causal and unmasked attention are supported.  Set
    ``sliding_window=512`` for causal attention over the current token and up
    to 511 preceding tokens.  By default the kernel returns only the BTHD
    result.  With ``return_lse=True`` it returns ``(result, lse)``, where LSE
    is fp32 with shape ``(B,H,128,T//128)``.
    """

    assert q.shape == k.shape, "q and k must have identical BTHD shapes"
    assert q.shape == v.shape, "q and v must have identical BTHD shapes"
    assert q.dtype == k.dtype == v.dtype, "q, k, and v must share a dtype"
    assert q.dtype in (nl.bfloat16, nl.float16), (
        "q, k, and v must be bf16 or fp16; fp32 doubles attention activation "
        "memory and is not supported"
    )

    batch, seqlen, n_heads, head_dim = q.shape
    assert head_dim <= 128, "head_dim must be <= 128"
    assert head_dim % 16 == 0, "head_dim must be a multiple of 16"
    assert seqlen % _MM1_TILE == 0, "seqlen must be a multiple of 512"
    assert sliding_window in (0, 512), "sliding_window must be 0 or 512"
    assert sliding_window == 0 or causal, "sliding-window attention must be causal"

    use_sliding_window = sliding_window > 0

    section_size = min(seqlen, _MAX_SECTION)
    assert seqlen % section_size == 0, "seqlen > 2048 must be divisible by 2048"
    num_sections = seqlen // section_size
    q_tiles_per_plane = seqlen // _Q_TILE
    mm1_tiles_per_section = section_size // _MM1_TILE
    token_tiles_per_section = section_size // _V_TILE

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    result = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    if return_lse:
        lse = nl.ndarray(
            (batch, n_heads, _Q_TILE, q_tiles_per_plane),
            dtype=nl.float32,
            buffer=nl.shared_hbm,
        )

    grid_ndim = nl.program_ndim()
    if grid_ndim == 0:
        num_shards = 1
        shard_id = 0
    else:
        assert grid_ndim == 1, "bthd_flash_fwd expects an LNC1/LNC2 grid"
        num_shards = nl.num_programs(axes=0)
        shard_id = nl.program_id(axis=0)

    num_planes = batch * n_heads
    assert num_planes % num_shards == 0, "B*H must be divisible by the LNC degree"
    planes_per_shard = num_planes // num_shards
    token_stride = n_heads * head_dim

    # Native-BTHD V staging.  Each lane holds tokens lane, lane+128, ...;
    # this makes the HBM stride explicit while giving large per-partition DMAs.
    pipe_ring = _V3_RING_SWA if use_sliding_window else _V3_RING_FULL
    pe_transpose_tiles = _V3_PE_SWA if use_sliding_window else _V3_PE_FULL
    v3_qk_ahead = _V3_QK_AHEAD_SWA if use_sliding_window else _V3_QK_AHEAD_FULL
    v3_exp_ahead = _V3_EXP_AHEAD_SWA if use_sliding_window else _V3_EXP_AHEAD_FULL
    v3_tscr_lag = _V3_TSCR_LAG_SWA if use_sliding_window else _V3_TSCR_LAG_FULL
    v3_norm_lag = _V3_NORM_LAG_SWA if use_sliding_window else _V3_NORM_LAG_FULL
    assert v3_qk_ahead - v3_tscr_lag > v3_exp_ahead >= _V3_TP_AHEAD >= 0
    assert pipe_ring > v3_qk_ahead + v3_norm_lag
    assert q_tiles_per_plane % _V3_STORE_BATCH == 0
    # Ring slots hold one Q group's visible key window (5 tiles for the 512 window, the whole
    # section otherwise); columns inside a slot are relative to the window start.
    if use_sliding_window and _V3_REL_INDEX:
        slot_cols = max((sliding_window // _Q_TILE + 1) * _Q_TILE, _V3_SLOT_COLS_MIN)
    else:
        slot_cols = min(seqlen, _MAX_SECTION)
    # scores: one contiguous (128, ring, cols) tensor unless _V3_SCORES_LIST (a list of separate tensors
    # measured 2.5% slower); the DMA-transposed probability buffers must be distinct tensors.
    if _V3_SCORES_LIST:
        pipeline_scores = []
        for ring_index in range(pipe_ring):
            pipeline_scores.append(nl.ndarray((_Q_TILE, slot_cols), dtype=nl.float32, buffer=nl.sbuf))
    else:
        pipeline_scores_3d = nl.ndarray((_Q_TILE, pipe_ring, slot_cols), dtype=nl.float32, buffer=nl.sbuf)
        pipeline_scores = []
        for ring_index in range(pipe_ring):
            pipeline_scores.append(pipeline_scores_3d[:, ring_index, :])
    pipeline_probabilities = []
    pipeline_probabilities_tp = []
    if _V6_TP_PAIR:
        assert pipe_ring % 2 == 0
        pipeline_prob_pairs = []
        pipeline_prob_tp_pairs = []
        for pair_index in range(pipe_ring // 2):
            pipeline_prob_pairs.append(nl.ndarray((_Q_TILE, 2, slot_cols), dtype=q.dtype, buffer=nl.sbuf))
            pipeline_prob_tp_pairs.append(nl.ndarray((_V_TILE, 2, slot_cols), dtype=q.dtype, buffer=nl.sbuf))
        for ring_index in range(pipe_ring):
            pipeline_probabilities.append(pipeline_prob_pairs[ring_index // 2][:, ring_index % 2, :])
            pipeline_probabilities_tp.append(pipeline_prob_tp_pairs[ring_index // 2][:, ring_index % 2, :])
    else:
        for ring_index in range(pipe_ring):
            pipeline_probabilities.append(nl.ndarray((_Q_TILE, slot_cols), dtype=q.dtype, buffer=nl.sbuf))
        for ring_index in range(pipe_ring):
            pipeline_probabilities_tp.append(nl.ndarray((_V_TILE, slot_cols), dtype=q.dtype, buffer=nl.sbuf))
    v_bthd_ring = [
        nl.ndarray((_Q_TILE, q_tiles_per_plane, head_dim), dtype=q.dtype, buffer=nl.sbuf),
        nl.ndarray((_Q_TILE, q_tiles_per_plane, head_dim), dtype=q.dtype, buffer=nl.sbuf),
        nl.ndarray((_Q_TILE, q_tiles_per_plane, head_dim), dtype=q.dtype, buffer=nl.sbuf),
    ]
    k_transposed_ring = [
        nl.ndarray((head_dim, seqlen), dtype=q.dtype, buffer=nl.sbuf),
        nl.ndarray((head_dim, seqlen), dtype=q.dtype, buffer=nl.sbuf),
        nl.ndarray((head_dim, seqlen), dtype=q.dtype, buffer=nl.sbuf),
    ]
    v_bthd = v_bthd_ring[0]
    k_transposed = k_transposed_ring[0]

    q_transposed = nl.ndarray((head_dim, _Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
    q_pack_capacity = min(seqlen, _Q_GROUPS_PER_PACK * _Q_TILE)

    # Four 512-key fp32 PSUM slots (one bank each) for the chunked QK matmuls of the pipelined
    # path; the general fallback below uses slot 0.
    qk_psum_slots = [_psum_at((_Q_TILE, _QK_CHUNK_TILES * _Q_TILE), nl.float32, bank) for bank in range(4)]
    qk_psum = qk_psum_slots[0]
    qk_copy = nl.ndarray((_Q_TILE, _MM1_TILE), dtype=nl.float32, buffer=nl.sbuf)
    qk_masked = nl.ndarray((_Q_TILE, _MM1_TILE), dtype=nl.float32, buffer=nl.sbuf)
    # Constant additive masks and the identity used to add them into PSUM.
    identity = nl.shared_identity_matrix(_Q_TILE, dtype=q.dtype)
    mask_zero = nl.ndarray((_Q_TILE, _Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
    # [sliding | causal] mask pair: one 256-column identity matmul masks both edge tiles of a paired window.
    mask_pair = nl.ndarray((_Q_TILE, 2 * _Q_TILE), dtype=q.dtype, buffer=nl.sbuf)
    mask_sliding = mask_pair[:, nl.ds(0, _Q_TILE)]
    mask_causal = mask_pair[:, nl.ds(_Q_TILE, _Q_TILE)]
    _build_mask_constants(identity, mask_zero, mask_causal, mask_sliding)
    scores = nl.ndarray((_Q_TILE, section_size), dtype=nl.float32, buffer=nl.sbuf)
    partial_max = nl.ndarray(
        (_Q_TILE, mm1_tiles_per_section),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    section_max = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    new_max = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    negative_max = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    correction = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)

    probabilities = nl.ndarray((_Q_TILE, section_size), dtype=q.dtype, buffer=nl.sbuf)
    partial_sum = nl.ndarray(
        (_Q_TILE, mm1_tiles_per_section),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    section_sum = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    probabilities_tp = nl.ndarray((_V_TILE, _MM1_TILE), dtype=q.dtype, buffer=nl.sbuf)

    running_max = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    running_sum = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    old_sum_scaled = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    running_output = nl.ndarray((_Q_TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    old_output_scaled = nl.ndarray(
        (_Q_TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf
    )
    pv_sbuf = nl.ndarray((_Q_TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    reciprocal_sum = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
    final_output = nl.ndarray((_Q_TILE, head_dim), dtype=q.dtype, buffer=nl.sbuf)
    if return_lse:
        lse_log_sum = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        lse_value = nl.ndarray((_Q_TILE, 1), dtype=nl.float32, buffer=nl.sbuf)
        # Per-plane LSE staging (two-entry ring across planes): one store per plane instead of
        # one 128x4-byte store per Q group.
        lse_planes = [
            nl.ndarray((_Q_TILE, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf),
            nl.ndarray((_Q_TILE, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf),
        ]

    # Per-plane log row sums ([128, T//128]): the LSE (log + subtract) is computed once per plane instead of
    # once per Q group.
    plane_log_sum = nl.ndarray((_Q_TILE, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf)
    final_batches = []
    for ring_index in range(_V3_FINAL_RING):
        final_batches.append(nl.ndarray((_Q_TILE, _V3_STORE_BATCH, head_dim), dtype=q.dtype, buffer=nl.sbuf))
    pv_psum_ring = []
    pv_bytes = head_dim * 4
    assert _V3_PV_RING * pv_bytes <= _PSUM_BANK_BYTES
    for ring_index in range(_V3_PV_RING):
        pv_psum_ring.append(_psum_at((_Q_TILE, head_dim), nl.float32, 4, ring_index * pv_bytes))
    tp_bytes = _Q_TILE * 2
    pipeline_tp_psums = [[_psum_at((_Q_TILE, _Q_TILE), q.dtype, 6, (3 * r + j) * tp_bytes) for j in range(3)] for r in range(2)]

    if num_sections == 1:
        # ---- head-set interleaving: groups are ordered (batch, head-set, q_tile, head-in-set) so that
        # the outputs of HEADS_PER_SET adjacent heads of one 128-token group are stored with one DMA of
        # HEADS_PER_SET*head_dim contiguous elements per token (fewer, larger DMA descriptors), and V of a
        # whole head set is loaded the same way.
        batches_per_shard = planes_per_shard // n_heads
        heads_per_set = _V6_HEADS_PER_SET
        kv_ring = _V6_KV_RING
        if not use_sliding_window and _V6_HEADS_PER_SET_FULL > 0:
            heads_per_set = _V6_HEADS_PER_SET_FULL
        if not use_sliding_window and _V6_KV_RING_FULL > 0:
            kv_ring = _V6_KV_RING_FULL
        if planes_per_shard % n_heads != 0 or n_heads % heads_per_set != 0:
            heads_per_set = 1
        sets_per_batch = n_heads // heads_per_set
        groups_per_set = q_tiles_per_plane * heads_per_set
        total_sets = batches_per_shard * sets_per_batch
        total_groups = total_sets * groups_per_set
        groups_per_pack_block = _Q_GROUPS_PER_PACK * heads_per_set
        pack_blocks_per_set = q_tiles_per_plane // _Q_GROUPS_PER_PACK
        assert q_tiles_per_plane % _Q_GROUPS_PER_PACK == 0
        slots_per_group = 2 if use_sliding_window else _QK_NUM_SLOTS
        # The next head set's K/V load overwrites ring slot (s+1) % KV_RING; in program order it must
        # come after the last PV of the set that used that slot (set s+1-KV_RING): with a 2-deep ring
        # issue it once the QK stage is v3_qk_ahead groups into set s (PV of set s-1 is then complete).
        kv_load_at = 0
        if kv_ring < 3:
            kv_load_at = v3_qk_ahead
        assert kv_load_at < groups_per_set
        _V9_LOAD_AT = _V9_LOAD_AT_SWA if use_sliding_window else _V9_LOAD_AT_FULL
        _V9_LOAD_LAG = _V9_LOAD_LAG_SWA if use_sliding_window else _V9_LOAD_LAG_FULL
        _V9_LOAD_ANCHOR = _V9_LOAD_ANCHOR_SWA if use_sliding_window else _V9_LOAD_ANCHOR_FULL
        _V9_Q_RING = _V9_Q_RING_SWA if use_sliding_window else _V9_Q_RING_FULL
        assert kv_ring >= 3 and pack_blocks_per_set == 1 and _V9_LOAD_AT < groups_per_set
        pv_anchor_ring = []
        for ring_index in range(64):
            pv_anchor_ring.append(None)
        # Reciprocals are batched over consecutive q_tiles of one head; q_tile g+b-1 of head h is exp'd
        # (b-1)*heads_per_set iterations after q_tile g, which must be <= exp_ahead + norm_lag so that its
        # row sum exists (in program order) when the batch's reciprocal is computed.
        recip_lead = _V10_RECIP_LEAD_SWA if use_sliding_window else _V10_RECIP_LEAD_FULL
        recip_batch = 1
        for cand in (4, 2):
            if recip_batch == 1 and cand <= _V3_RECIP_BATCH and q_tiles_per_plane % cand == 0 and (cand - 1) * heads_per_set <= v3_exp_ahead + v3_norm_lag - recip_lead:
                recip_batch = cand
        assert recip_batch == 4

        set_bases = []
        set_plane0 = []
        for set_index in range(total_sets):
            batch_id = (shard_id * planes_per_shard) // n_heads + set_index // sets_per_batch
            head0 = (set_index % sets_per_batch) * heads_per_set
            set_bases.append((batch_id * seqlen * n_heads + head0) * head_dim)
            set_plane0.append(batch_id * n_heads + head0)

        k_sets = []
        v_sets = []
        for ring_index in range(kv_ring):
            k_heads = []
            if _V6_PAIR_LOADS and heads_per_set == 2:
                k_pair = nl.ndarray((head_dim, 2, seqlen), dtype=q.dtype, buffer=nl.sbuf)
                for h in range(heads_per_set):
                    k_heads.append(k_pair[:, h, :])
                k_heads.append(k_pair)
            else:
                for h in range(heads_per_set):
                    k_heads.append(nl.ndarray((head_dim, seqlen), dtype=q.dtype, buffer=nl.sbuf))
            k_sets.append(k_heads)
            v_sets.append(nl.ndarray((_Q_TILE, q_tiles_per_plane, heads_per_set, head_dim), dtype=q.dtype, buffer=nl.sbuf))
        q_pack_sets = []
        for ring_index in range(_V9_Q_RING):
            packs = []
            if _V6_PAIR_LOADS and heads_per_set == 2:
                q_pair = nl.ndarray((head_dim, 2, q_pack_capacity), dtype=q.dtype, buffer=nl.sbuf)
                for h in range(heads_per_set):
                    packs.append(q_pair[:, h, :])
                packs.append(q_pair)
            else:
                for h in range(heads_per_set):
                    packs.append(nl.ndarray((head_dim, q_pack_capacity), dtype=q.dtype, buffer=nl.sbuf))
            q_pack_sets.append(packs)
        # per-(set ring, head) staging of row sums / maxima / reciprocals for the LSE and normalize
        set_section_sum = []
        set_row_max = []
        set_recip = []
        set_section_sum_2d = []
        set_recip_2d = []
        recip_pair = False
        if use_sliding_window and _V10_RECIP_PAIR_SWA and heads_per_set == 2:
            recip_pair = True
        if (not use_sliding_window) and _V10_RECIP_PAIR_FULL and heads_per_set == 2:
            recip_pair = True
        for ring_index in range(2):
            a = []
            b = []
            c = []
            if recip_pair:
                a2 = nl.ndarray((_Q_TILE, heads_per_set, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf)
                c2 = nl.ndarray((_Q_TILE, heads_per_set, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf)
                set_section_sum_2d.append(a2)
                set_recip_2d.append(c2)
                for h in range(heads_per_set):
                    a.append(a2[:, h, :])
                    c.append(c2[:, h, :])
            else:
                set_section_sum_2d.append(None)
                set_recip_2d.append(None)
                for h in range(heads_per_set):
                    a.append(nl.ndarray((_Q_TILE, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf))
                    c.append(nl.ndarray((_Q_TILE, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf))
            for h in range(heads_per_set):
                b.append(nl.ndarray((_Q_TILE, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf))
            set_section_sum.append(a)
            set_row_max.append(b)
            set_recip.append(c)
        if recip_pair:
            assert (recip_batch - 1) * heads_per_set + heads_per_set - 1 <= v3_exp_ahead + v3_norm_lag - recip_lead
        lse_pairs = []
        if return_lse:
            for ring_index in range(2):
                lse_pairs.append(nl.ndarray((_Q_TILE, heads_per_set, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf))
        final_sets = []
        for ring_index in range(_V3_FINAL_RING):
            final_sets.append(nl.ndarray((_Q_TILE, heads_per_set, head_dim), dtype=q.dtype, buffer=nl.sbuf))

        if (_V11_HWDGE_KQ_INIT_SWA if use_sliding_window else _V11_HWDGE_KQ_INIT_FULL) and _V6_PAIR_LOADS and heads_per_set == 2:
            _load_k_pair_range(k, k_sets[0][2], set_bases[0], token_stride, head_dim, seqlen, 0, q_tiles_per_plane, hwdge=True)
            _load_v_set(v, v_sets[0], set_bases[0], token_stride, head_dim, q_tiles_per_plane, heads_per_set)
            _load_q_pack_pair_range(q, q_pack_sets[0][2], set_bases[0], token_stride, head_dim, q_pack_capacity, 0,
                                    q_pack_capacity, hwdge=True)
        else:
            _load_set_kv(k, v, k_sets[0], v_sets[0], set_bases[0], token_stride, head_dim, seqlen,
                         q_tiles_per_plane, heads_per_set)
            _load_pack_block(q, q_pack_sets[0], set_bases[0], 0, token_stride, head_dim, q_pack_capacity,
                             heads_per_set)

        # program-order handles for the scheduling pins
        anchor = None
        have_anchor = False
        last_vec = None
        have_vec = False
        last_sca = None
        have_sca = False
        last_ten = None
        have_ten = False
        for step in range(-v3_qk_ahead, total_groups + v3_norm_lag):
            g_qk = step + v3_qk_ahead
            g_tscr = step + v3_qk_ahead - v3_tscr_lag
            g_exp = step + v3_exp_ahead
            g_tp = step + _V3_TP_AHEAD
            g_pv = step
            g_wb = step - v3_norm_lag

            if use_sliding_window:
                # ---- stage 2: scale + row max from PSUM (VectorE)
                if 0 <= g_tscr < total_groups:
                    t_set = g_tscr // groups_per_set
                    t_r = g_tscr % groups_per_set
                    pin_i = anchor
                    pin_f = have_anchor and _V7_PIN_TSCR
                    if _V7_VCHAIN and have_vec:
                        pin_i = last_vec
                        pin_f = True
                    last_vec = _qk_reduce_group(
                        t_r // heads_per_set,
                        qk_psum_slots,
                        pipeline_scores[g_tscr % pipe_ring],
                        set_row_max[t_set % 2][t_r % heads_per_set][:, nl.ds(t_r // heads_per_set, 1)],
                        softmax_scale,
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        g_tscr * slots_per_group,
                        pin_f,
                        pin_i,
                    )
                    have_vec = True

                # ---- stage 1: QK matmuls (TensorE) + prefetch of the next head set / Q pack block
                if 0 <= g_qk < total_groups:
                    qk_set = g_qk // groups_per_set
                    qk_r = g_qk % groups_per_set
                    qk_group = qk_r // heads_per_set
                    qk_h = qk_r % heads_per_set
                    qk_pb = g_qk // groups_per_pack_block
                    if qk_set + 1 < total_sets:
                        la_inst = None
                        la_have = False
                        if step - 1 >= 0 and pv_anchor_ring[(step - 1) % 64] is not None:
                            la_inst = pv_anchor_ring[(step - 1) % 64]
                            la_have = True
                        ld_anchor = (_V10_LOAD_ANCHOR_SWA if use_sliding_window else _V10_LOAD_ANCHOR_FULL) and la_have
                        ld_pieces = _V10_LOAD_PIECES_SWA if use_sliding_window else _V10_LOAD_PIECES_FULL
                        k_trig = _V10_K_TRIG_SWA if use_sliding_window else _V10_K_TRIG_FULL
                        q_trig = _V10_Q_TRIG_SWA if use_sliding_window else _V10_Q_TRIG_FULL
                        v_trig = _V10_V_TRIG_SWA if use_sliding_window else _V10_V_TRIG_FULL
                        tiles_pp = q_tiles_per_plane // ld_pieces
                        for piece in range(ld_pieces):
                            if qk_r == k_trig[piece]:
                                hk = _load_k_pair_range(k, k_sets[(qk_set + 1) % kv_ring][2], set_bases[qk_set + 1], token_stride,
                                                        head_dim, seqlen, piece * tiles_pp, tiles_pp,
                                                        hwdge=(_V11_HWDGE_KQ_SWA if use_sliding_window else _V11_HWDGE_KQ_FULL))
                                if ld_anchor:
                                    hk.depends_on(la_inst)
                        if qk_r == v_trig:
                            _load_v_set(v, v_sets[(qk_set + 1) % kv_ring], set_bases[qk_set + 1], token_stride, head_dim,
                                        q_tiles_per_plane, heads_per_set)
                        for piece in range(ld_pieces):
                            if qk_r == q_trig[piece]:
                                hq = _load_q_pack_pair_range(q, q_pack_sets[(qk_set + 1) % _V9_Q_RING][2], set_bases[qk_set + 1],
                                                             token_stride, head_dim, q_pack_capacity, piece * tiles_pp * _Q_TILE,
                                                             tiles_pp * _Q_TILE,
                                                             hwdge=(_V11_HWDGE_KQ_SWA if use_sliding_window else _V11_HWDGE_KQ_FULL))
                                if ld_anchor:
                                    hq.depends_on(la_inst)
                    pack_col = (qk_group % _Q_GROUPS_PER_PACK) * _Q_TILE
                    qk0_early = _V10_QK0_EARLY_SWA if use_sliding_window else _V10_QK0_EARLY_FULL
                    qk_p0_lo = 0
                    if qk0_early and g_qk > 0:
                        qk_p0_lo = 1
                    last_ten = _qk_matmuls_group(
                        qk_group,
                        q_pack_sets[qk_pb % _V9_Q_RING][qk_h][:, nl.ds(pack_col, _Q_TILE)],
                        k_sets[qk_set % kv_ring][qk_h],
                        qk_psum_slots,
                        identity,
                        mask_causal,
                        mask_sliding,
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        g_qk * slots_per_group,
                        k_sets[qk_set % kv_ring][heads_per_set] if (_V6_PAIR_LOADS and heads_per_set >= 2) else None,
                        qk_h,
                        mask_pair,
                        _V7_TCHAIN and have_ten,
                        last_ten,
                        qk_p0_lo,
                        99,
                        True,
                    )
                    have_ten = True

                # ---- stage 3: exp + row sum (ScalarE) [+ DMA transposes]
                if 0 <= g_exp < total_groups:
                    exp_slot = g_exp % pipe_ring
                    e_set = g_exp // groups_per_set
                    e_r = g_exp % groups_per_set
                    exp_group = e_r // heads_per_set
                    pin_i = anchor
                    pin_f = have_anchor and _V7_PIN_EXP
                    if _V7_SCHAIN and have_sca:
                        pin_i = last_sca
                        pin_f = True
                    last_sca = _exp_only_group(
                        exp_group,
                        pipeline_scores[exp_slot],
                        set_row_max[e_set % 2][e_r % heads_per_set][:, nl.ds(exp_group, 1)],
                        pipeline_probabilities[exp_slot],
                        set_section_sum[e_set % 2][e_r % heads_per_set][:, nl.ds(exp_group, 1)],
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        pin_f,
                        pin_i,
                    )
                    have_sca = True
                    if _V6_TP_PAIR:
                        assert heads_per_set == 2 and groups_per_set % 2 == 0
                        if exp_slot % 2 == 1:
                            _dma_transpose_pair(
                                exp_group,
                                pipeline_prob_pairs[exp_slot // 2],
                                pipeline_prob_tp_pairs[exp_slot // 2],
                                pe_transpose_tiles,
                                causal,
                                use_sliding_window,
                                sliding_window,
                                slot_cols,
                                token_tiles_per_section,
                            )
                    elif _V3_DMA_TP_IN_EXP_STAGE:
                        _dma_transpose_group(
                            exp_group,
                            pipeline_probabilities[exp_slot],
                            pipeline_probabilities_tp[exp_slot],
                            pe_transpose_tiles,
                            causal,
                            use_sliding_window,
                            sliding_window,
                            slot_cols,
                            token_tiles_per_section,
                        )

                # ---- stage 4: TensorE transposes + ScalarE/VectorE copies [+ DMA transposes]
                if 0 <= g_tp < total_groups:
                    tp_slot = g_tp % pipe_ring
                    tp_group = (g_tp % groups_per_set) // heads_per_set
                    if not _V3_DMA_TP_IN_EXP_STAGE:
                        _dma_transpose_group(
                            tp_group,
                            pipeline_probabilities[tp_slot],
                            pipeline_probabilities_tp[tp_slot],
                            pe_transpose_tiles,
                            causal,
                            use_sliding_window,
                            sliding_window,
                            slot_cols,
                            token_tiles_per_section,
                        )
                    if pe_transpose_tiles > 0:
                        pin_si = anchor
                        pin_sf = have_anchor and _V7_PIN_COPIES
                        pin_vi = anchor
                        pin_vf = have_anchor and _V7_PIN_COPIES
                        if _V7_SCHAIN and have_sca:
                            pin_si = last_sca
                            pin_sf = True
                        if _V7_VCHAIN and have_vec:
                            pin_vi = last_vec
                            pin_vf = True
                        pe_t, pe_ht, pe_s, pe_hs, pe_v, pe_hv = _pe_transpose_group(
                            tp_group,
                            pipeline_probabilities[tp_slot],
                            pipeline_probabilities_tp[tp_slot],
                            pipeline_tp_psums[g_tp % _V3_TP_PSUM_RING],
                            pe_transpose_tiles,
                            causal,
                            use_sliding_window,
                            sliding_window,
                            token_tiles_per_section,
                            (_V7_TCHAIN or _V7_TPIN) and have_ten,
                            last_ten,
                            pin_sf,
                            pin_si,
                            pin_vf,
                            pin_vi,
                            (_V8_COPY_ENGINES_SWA if use_sliding_window else _V8_COPY_ENGINES_FULL)[g_tp % 2],
                        )
                        if pe_ht:
                            last_ten = pe_t
                            have_ten = True
                        if pe_hs:
                            last_sca = pe_s
                            have_sca = True
                        if pe_hv:
                            last_vec = pe_v
                            have_vec = True

                # ---- stage 5: P @ V (TensorE)
                if 0 <= g_pv < total_groups:
                    pv_set = g_pv // groups_per_set
                    pv_r = g_pv % groups_per_set
                    anchor = _compute_pv_group_set(
                        pv_r // heads_per_set,
                        pipeline_probabilities_tp[g_pv % pipe_ring],
                        v_sets[pv_set % kv_ring],
                        pv_r % heads_per_set,
                        pv_psum_ring[g_pv % _V3_PV_RING],
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        (_V7_TCHAIN or _V7_PVANCH) and have_ten,
                        last_ten,
                    )
                    have_anchor = True
                    last_ten = anchor
                    have_ten = True
                    pv_anchor_ring[g_pv % 64] = anchor

                # ---- stage 5b: chunk 0 of the next group's QK (issued after PV so the first TSCR chunk never waits)
                qk0_early = _V10_QK0_EARLY_SWA if use_sliding_window else _V10_QK0_EARLY_FULL
                g_qk1 = g_qk + 1
                if qk0_early and 0 <= g_qk1 < total_groups:
                    e_qk_set = g_qk1 // groups_per_set
                    e_qk_r = g_qk1 % groups_per_set
                    e_qk_group = e_qk_r // heads_per_set
                    e_qk_h = e_qk_r % heads_per_set
                    e_qk_pb = g_qk1 // groups_per_pack_block
                    e_pack_col = (e_qk_group % _Q_GROUPS_PER_PACK) * _Q_TILE
                    last_ten = _qk_matmuls_group(
                        e_qk_group,
                        q_pack_sets[e_qk_pb % _V9_Q_RING][e_qk_h][:, nl.ds(e_pack_col, _Q_TILE)],
                        k_sets[e_qk_set % kv_ring][e_qk_h],
                        qk_psum_slots,
                        identity,
                        mask_causal,
                        mask_sliding,
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        g_qk1 * slots_per_group,
                        k_sets[e_qk_set % kv_ring][heads_per_set] if (_V6_PAIR_LOADS and heads_per_set >= 2) else None,
                        e_qk_h,
                        mask_pair,
                        False,
                        None,
                        0,
                        1,
                        False,
                    )
                    have_ten = True

                # ---- stage 6: normalize (VectorE) + set store (DMA) [+ LSE at plane end]
                if 0 <= g_wb < total_groups:
                    wb_set = g_wb // groups_per_set
                    wb_r = g_wb % groups_per_set
                    wb_group = wb_r // heads_per_set
                    wb_h = wb_r % heads_per_set
                    rec_list = []
                    if g_wb == 0:
                        for gg in range(0, recip_lead + 1):
                            rec_list.append(gg)
                    else:
                        rec_list.append(g_wb + recip_lead)
                    for g_rec in rec_list:
                        rc_set = g_rec // groups_per_set
                        rc_r = g_rec % groups_per_set
                        rc_group = rc_r // heads_per_set
                        rc_h = rc_r % heads_per_set
                        if g_rec < total_groups and rc_group % recip_batch == 0 and (rc_h == 0 or not recip_pair):
                            if recip_pair:
                                h_rec = nisa.reciprocal(
                                    dst=set_recip_2d[rc_set % 2][:, :, nl.ds(rc_group, recip_batch)],
                                    data=set_section_sum_2d[rc_set % 2][:, :, nl.ds(rc_group, recip_batch)],
                                )
                            else:
                                h_rec = nisa.reciprocal(
                                    dst=set_recip[rc_set % 2][rc_h][:, nl.ds(rc_group, recip_batch)],
                                    data=set_section_sum[rc_set % 2][rc_h][:, nl.ds(rc_group, recip_batch)],
                                )
                            if _V7_VCHAIN and have_vec:
                                h_rec.depends_on(last_vec, kind=nisa.dependency.schedule)
                            elif _V7_PIN_RECIP and have_anchor:
                                h_rec.depends_on(anchor)
                            last_vec = h_rec
                            have_vec = True
                    wb_final = final_sets[(g_wb // heads_per_set) % _V3_FINAL_RING]
                    norm_eng = (_V8_NORM_ENGINE_SWA if use_sliding_window else _V8_NORM_ENGINE_FULL)[g_wb % 2]
                    if norm_eng == "s":
                        h_norm = nisa.tensor_scalar(
                            dst=wb_final[:, wb_h, :],
                            data=pv_psum_ring[g_wb % _V3_PV_RING],
                            op0=nl.multiply,
                            operand0=set_recip[wb_set % 2][wb_h][:, nl.ds(wb_group, 1)],
                            engine=engine.scalar,
                        )
                        if _V7_SCHAIN and have_sca:
                            h_norm.depends_on(last_sca)
                        elif _V7_PIN_NORM and have_anchor:
                            h_norm.depends_on(anchor)
                        last_sca = h_norm
                        have_sca = True
                    else:
                        h_norm = nisa.tensor_scalar(
                            dst=wb_final[:, wb_h, :],
                            data=pv_psum_ring[g_wb % _V3_PV_RING],
                            op0=nl.multiply,
                            operand0=set_recip[wb_set % 2][wb_h][:, nl.ds(wb_group, 1)],
                            engine=engine.vector,
                        )
                        if _V7_VCHAIN and have_vec:
                            h_norm.depends_on(last_vec, kind=nisa.dependency.schedule)
                        elif _V7_PIN_NORM and have_anchor:
                            h_norm.depends_on(anchor)
                        last_vec = h_norm
                        have_vec = True
                    if wb_h == heads_per_set - 1:
                        nisa.dma_copy(
                            dst=result.ap(
                                pattern=[
                                    [token_stride, _Q_TILE],
                                    [head_dim, heads_per_set],
                                    [1, head_dim],
                                ],
                                offset=set_bases[wb_set] + wb_group * _Q_TILE * token_stride,
                            ),
                            src=wb_final.ap(
                                pattern=[
                                    [heads_per_set * head_dim, _Q_TILE],
                                    [head_dim, heads_per_set],
                                    [1, head_dim],
                                ],
                                offset=0,
                            ),
                            dge_mode=_V3_STORE_DGE,
                        )
                    if return_lse and wb_group == q_tiles_per_plane - 1 and _V6_LSE_PAIR:
                        nisa.activation(
                            dst=plane_log_sum,
                            op=nl.log,
                            data=set_section_sum[wb_set % 2][wb_h],
                        )
                        nisa.tensor_tensor(
                            dst=lse_pairs[wb_set % 2][:, wb_h, :],
                            data1=plane_log_sum,
                            data2=set_row_max[wb_set % 2][wb_h],
                            op=nl.subtract if _G3_EXP_FORM == 'negmin' else nl.add,
                        )
                        if wb_h == heads_per_set - 1:
                            nisa.dma_copy(
                                dst=lse.ap(
                                    pattern=[
                                        [q_tiles_per_plane, _Q_TILE],
                                        [_Q_TILE * q_tiles_per_plane, heads_per_set],
                                        [1, q_tiles_per_plane],
                                    ],
                                    offset=set_plane0[wb_set] * _Q_TILE * q_tiles_per_plane,
                                ),
                                src=lse_pairs[wb_set % 2].ap(
                                    pattern=[
                                        [heads_per_set * q_tiles_per_plane, _Q_TILE],
                                        [q_tiles_per_plane, heads_per_set],
                                        [1, q_tiles_per_plane],
                                    ],
                                    offset=0,
                                ),
                            )
                    elif return_lse and wb_group == q_tiles_per_plane - 1:
                        nisa.activation(
                            dst=plane_log_sum,
                            op=nl.log,
                            data=set_section_sum[wb_set % 2][wb_h],
                        )
                        nisa.tensor_tensor(
                            dst=lse_planes[g_wb % 2],
                            data1=plane_log_sum,
                            data2=set_row_max[wb_set % 2][wb_h],
                            op=nl.subtract if _G3_EXP_FORM == 'negmin' else nl.add,
                        )
                        _store_lse_plane(
                            lse,
                            set_plane0[wb_set] + wb_h,
                            q_tiles_per_plane,
                            lse_planes[g_wb % 2],
                        )
            else:
                # ---- stage 1: QK matmuls (TensorE) + prefetch of the next head set / Q pack block
                if 0 <= g_qk < total_groups:
                    qk_set = g_qk // groups_per_set
                    qk_r = g_qk % groups_per_set
                    qk_group = qk_r // heads_per_set
                    qk_h = qk_r % heads_per_set
                    qk_pb = g_qk // groups_per_pack_block
                    if qk_set + 1 < total_sets:
                        la_inst = None
                        la_have = False
                        if step - 1 >= 0 and pv_anchor_ring[(step - 1) % 64] is not None:
                            la_inst = pv_anchor_ring[(step - 1) % 64]
                            la_have = True
                        ld_anchor = (_V10_LOAD_ANCHOR_SWA if use_sliding_window else _V10_LOAD_ANCHOR_FULL) and la_have
                        ld_pieces = _V10_LOAD_PIECES_SWA if use_sliding_window else _V10_LOAD_PIECES_FULL
                        k_trig = _V10_K_TRIG_SWA if use_sliding_window else _V10_K_TRIG_FULL
                        q_trig = _V10_Q_TRIG_SWA if use_sliding_window else _V10_Q_TRIG_FULL
                        v_trig = _V10_V_TRIG_SWA if use_sliding_window else _V10_V_TRIG_FULL
                        tiles_pp = q_tiles_per_plane // ld_pieces
                        for piece in range(ld_pieces):
                            if qk_r == k_trig[piece]:
                                hk = _load_k_pair_range(k, k_sets[(qk_set + 1) % kv_ring][2], set_bases[qk_set + 1], token_stride,
                                                        head_dim, seqlen, piece * tiles_pp, tiles_pp,
                                                        hwdge=(_V11_HWDGE_KQ_SWA if use_sliding_window else _V11_HWDGE_KQ_FULL))
                                if ld_anchor:
                                    hk.depends_on(la_inst)
                        if qk_r == v_trig:
                            _load_v_set(v, v_sets[(qk_set + 1) % kv_ring], set_bases[qk_set + 1], token_stride, head_dim,
                                        q_tiles_per_plane, heads_per_set)
                        for piece in range(ld_pieces):
                            if qk_r == q_trig[piece]:
                                hq = _load_q_pack_pair_range(q, q_pack_sets[(qk_set + 1) % _V9_Q_RING][2], set_bases[qk_set + 1],
                                                             token_stride, head_dim, q_pack_capacity, piece * tiles_pp * _Q_TILE,
                                                             tiles_pp * _Q_TILE,
                                                             hwdge=(_V11_HWDGE_KQ_SWA if use_sliding_window else _V11_HWDGE_KQ_FULL))
                                if ld_anchor:
                                    hq.depends_on(la_inst)
                    pack_col = (qk_group % _Q_GROUPS_PER_PACK) * _Q_TILE
                    qk0_early = _V10_QK0_EARLY_SWA if use_sliding_window else _V10_QK0_EARLY_FULL
                    qk_p0_lo = 0
                    if qk0_early and g_qk > 0:
                        qk_p0_lo = 1
                    last_ten = _qk_matmuls_group(
                        qk_group,
                        q_pack_sets[qk_pb % _V9_Q_RING][qk_h][:, nl.ds(pack_col, _Q_TILE)],
                        k_sets[qk_set % kv_ring][qk_h],
                        qk_psum_slots,
                        identity,
                        mask_causal,
                        mask_sliding,
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        g_qk * slots_per_group,
                        k_sets[qk_set % kv_ring][heads_per_set] if (_V6_PAIR_LOADS and heads_per_set >= 2) else None,
                        qk_h,
                        mask_pair,
                        _V7_TCHAIN and have_ten,
                        last_ten,
                        qk_p0_lo,
                        99,
                        True,
                    )
                    have_ten = True

                # ---- stage 2: scale + row max from PSUM (VectorE)
                if 0 <= g_tscr < total_groups:
                    t_set = g_tscr // groups_per_set
                    t_r = g_tscr % groups_per_set
                    pin_i = anchor
                    pin_f = have_anchor and _V7_PIN_TSCR
                    if _V7_VCHAIN and have_vec:
                        pin_i = last_vec
                        pin_f = True
                    last_vec = _qk_reduce_group(
                        t_r // heads_per_set,
                        qk_psum_slots,
                        pipeline_scores[g_tscr % pipe_ring],
                        set_row_max[t_set % 2][t_r % heads_per_set][:, nl.ds(t_r // heads_per_set, 1)],
                        softmax_scale,
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        g_tscr * slots_per_group,
                        pin_f,
                        pin_i,
                    )
                    have_vec = True

                # ---- stage 3: exp + row sum (ScalarE) [+ DMA transposes]
                if 0 <= g_exp < total_groups:
                    exp_slot = g_exp % pipe_ring
                    e_set = g_exp // groups_per_set
                    e_r = g_exp % groups_per_set
                    exp_group = e_r // heads_per_set
                    pin_i = anchor
                    pin_f = have_anchor and _V7_PIN_EXP
                    if _V7_SCHAIN and have_sca:
                        pin_i = last_sca
                        pin_f = True
                    last_sca = _exp_only_group(
                        exp_group,
                        pipeline_scores[exp_slot],
                        set_row_max[e_set % 2][e_r % heads_per_set][:, nl.ds(exp_group, 1)],
                        pipeline_probabilities[exp_slot],
                        set_section_sum[e_set % 2][e_r % heads_per_set][:, nl.ds(exp_group, 1)],
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        pin_f,
                        pin_i,
                    )
                    have_sca = True
                    if _V6_TP_PAIR:
                        assert heads_per_set == 2 and groups_per_set % 2 == 0
                        if exp_slot % 2 == 1:
                            _dma_transpose_pair(
                                exp_group,
                                pipeline_prob_pairs[exp_slot // 2],
                                pipeline_prob_tp_pairs[exp_slot // 2],
                                pe_transpose_tiles,
                                causal,
                                use_sliding_window,
                                sliding_window,
                                slot_cols,
                                token_tiles_per_section,
                            )
                    elif _V3_DMA_TP_IN_EXP_STAGE:
                        _dma_transpose_group(
                            exp_group,
                            pipeline_probabilities[exp_slot],
                            pipeline_probabilities_tp[exp_slot],
                            pe_transpose_tiles,
                            causal,
                            use_sliding_window,
                            sliding_window,
                            slot_cols,
                            token_tiles_per_section,
                        )

                # ---- stage 4: TensorE transposes + ScalarE/VectorE copies [+ DMA transposes]
                if 0 <= g_tp < total_groups:
                    tp_slot = g_tp % pipe_ring
                    tp_group = (g_tp % groups_per_set) // heads_per_set
                    if not _V3_DMA_TP_IN_EXP_STAGE:
                        _dma_transpose_group(
                            tp_group,
                            pipeline_probabilities[tp_slot],
                            pipeline_probabilities_tp[tp_slot],
                            pe_transpose_tiles,
                            causal,
                            use_sliding_window,
                            sliding_window,
                            slot_cols,
                            token_tiles_per_section,
                        )
                    if pe_transpose_tiles > 0:
                        pin_si = anchor
                        pin_sf = have_anchor and _V7_PIN_COPIES
                        pin_vi = anchor
                        pin_vf = have_anchor and _V7_PIN_COPIES
                        if _V7_SCHAIN and have_sca:
                            pin_si = last_sca
                            pin_sf = True
                        if _V7_VCHAIN and have_vec:
                            pin_vi = last_vec
                            pin_vf = True
                        pe_t, pe_ht, pe_s, pe_hs, pe_v, pe_hv = _pe_transpose_group(
                            tp_group,
                            pipeline_probabilities[tp_slot],
                            pipeline_probabilities_tp[tp_slot],
                            pipeline_tp_psums[g_tp % _V3_TP_PSUM_RING],
                            pe_transpose_tiles,
                            causal,
                            use_sliding_window,
                            sliding_window,
                            token_tiles_per_section,
                            (_V7_TCHAIN or _V7_TPIN) and have_ten,
                            last_ten,
                            pin_sf,
                            pin_si,
                            pin_vf,
                            pin_vi,
                            (_V8_COPY_ENGINES_SWA if use_sliding_window else _V8_COPY_ENGINES_FULL)[g_tp % 2],
                        )
                        if pe_ht:
                            last_ten = pe_t
                            have_ten = True
                        if pe_hs:
                            last_sca = pe_s
                            have_sca = True
                        if pe_hv:
                            last_vec = pe_v
                            have_vec = True

                # ---- stage 5: P @ V (TensorE)
                if 0 <= g_pv < total_groups:
                    pv_set = g_pv // groups_per_set
                    pv_r = g_pv % groups_per_set
                    anchor = _compute_pv_group_set(
                        pv_r // heads_per_set,
                        pipeline_probabilities_tp[g_pv % pipe_ring],
                        v_sets[pv_set % kv_ring],
                        pv_r % heads_per_set,
                        pv_psum_ring[g_pv % _V3_PV_RING],
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        (_V7_TCHAIN or _V7_PVANCH) and have_ten,
                        last_ten,
                    )
                    have_anchor = True
                    last_ten = anchor
                    have_ten = True
                    pv_anchor_ring[g_pv % 64] = anchor

                # ---- stage 5b: chunk 0 of the next group's QK (issued after PV so the first TSCR chunk never waits)
                qk0_early = _V10_QK0_EARLY_SWA if use_sliding_window else _V10_QK0_EARLY_FULL
                g_qk1 = g_qk + 1
                if qk0_early and 0 <= g_qk1 < total_groups:
                    e_qk_set = g_qk1 // groups_per_set
                    e_qk_r = g_qk1 % groups_per_set
                    e_qk_group = e_qk_r // heads_per_set
                    e_qk_h = e_qk_r % heads_per_set
                    e_qk_pb = g_qk1 // groups_per_pack_block
                    e_pack_col = (e_qk_group % _Q_GROUPS_PER_PACK) * _Q_TILE
                    last_ten = _qk_matmuls_group(
                        e_qk_group,
                        q_pack_sets[e_qk_pb % _V9_Q_RING][e_qk_h][:, nl.ds(e_pack_col, _Q_TILE)],
                        k_sets[e_qk_set % kv_ring][e_qk_h],
                        qk_psum_slots,
                        identity,
                        mask_causal,
                        mask_sliding,
                        causal,
                        use_sliding_window,
                        sliding_window,
                        token_tiles_per_section,
                        g_qk1 * slots_per_group,
                        k_sets[e_qk_set % kv_ring][heads_per_set] if (_V6_PAIR_LOADS and heads_per_set >= 2) else None,
                        e_qk_h,
                        mask_pair,
                        False,
                        None,
                        0,
                        1,
                        False,
                    )
                    have_ten = True

                # ---- stage 6: normalize (VectorE) + set store (DMA) [+ LSE at plane end]
                if 0 <= g_wb < total_groups:
                    wb_set = g_wb // groups_per_set
                    wb_r = g_wb % groups_per_set
                    wb_group = wb_r // heads_per_set
                    wb_h = wb_r % heads_per_set
                    rec_list = []
                    if g_wb == 0:
                        for gg in range(0, recip_lead + 1):
                            rec_list.append(gg)
                    else:
                        rec_list.append(g_wb + recip_lead)
                    for g_rec in rec_list:
                        rc_set = g_rec // groups_per_set
                        rc_r = g_rec % groups_per_set
                        rc_group = rc_r // heads_per_set
                        rc_h = rc_r % heads_per_set
                        if g_rec < total_groups and rc_group % recip_batch == 0 and (rc_h == 0 or not recip_pair):
                            if recip_pair:
                                h_rec = nisa.reciprocal(
                                    dst=set_recip_2d[rc_set % 2][:, :, nl.ds(rc_group, recip_batch)],
                                    data=set_section_sum_2d[rc_set % 2][:, :, nl.ds(rc_group, recip_batch)],
                                )
                            else:
                                h_rec = nisa.reciprocal(
                                    dst=set_recip[rc_set % 2][rc_h][:, nl.ds(rc_group, recip_batch)],
                                    data=set_section_sum[rc_set % 2][rc_h][:, nl.ds(rc_group, recip_batch)],
                                )
                            if _V7_VCHAIN and have_vec:
                                h_rec.depends_on(last_vec, kind=nisa.dependency.schedule)
                            elif _V7_PIN_RECIP and have_anchor:
                                h_rec.depends_on(anchor)
                            last_vec = h_rec
                            have_vec = True
                    wb_final = final_sets[(g_wb // heads_per_set) % _V3_FINAL_RING]
                    norm_eng = (_V8_NORM_ENGINE_SWA if use_sliding_window else _V8_NORM_ENGINE_FULL)[g_wb % 2]
                    if norm_eng == "s":
                        h_norm = nisa.tensor_scalar(
                            dst=wb_final[:, wb_h, :],
                            data=pv_psum_ring[g_wb % _V3_PV_RING],
                            op0=nl.multiply,
                            operand0=set_recip[wb_set % 2][wb_h][:, nl.ds(wb_group, 1)],
                            engine=engine.scalar,
                        )
                        if _V7_SCHAIN and have_sca:
                            h_norm.depends_on(last_sca)
                        elif _V7_PIN_NORM and have_anchor:
                            h_norm.depends_on(anchor)
                        last_sca = h_norm
                        have_sca = True
                    else:
                        h_norm = nisa.tensor_scalar(
                            dst=wb_final[:, wb_h, :],
                            data=pv_psum_ring[g_wb % _V3_PV_RING],
                            op0=nl.multiply,
                            operand0=set_recip[wb_set % 2][wb_h][:, nl.ds(wb_group, 1)],
                            engine=engine.vector,
                        )
                        if _V7_VCHAIN and have_vec:
                            h_norm.depends_on(last_vec, kind=nisa.dependency.schedule)
                        elif _V7_PIN_NORM and have_anchor:
                            h_norm.depends_on(anchor)
                        last_vec = h_norm
                        have_vec = True
                    if wb_h == heads_per_set - 1:
                        nisa.dma_copy(
                            dst=result.ap(
                                pattern=[
                                    [token_stride, _Q_TILE],
                                    [head_dim, heads_per_set],
                                    [1, head_dim],
                                ],
                                offset=set_bases[wb_set] + wb_group * _Q_TILE * token_stride,
                            ),
                            src=wb_final.ap(
                                pattern=[
                                    [heads_per_set * head_dim, _Q_TILE],
                                    [head_dim, heads_per_set],
                                    [1, head_dim],
                                ],
                                offset=0,
                            ),
                            dge_mode=_V3_STORE_DGE,
                        )
                    if return_lse and wb_group == q_tiles_per_plane - 1 and _V6_LSE_PAIR:
                        nisa.activation(
                            dst=plane_log_sum,
                            op=nl.log,
                            data=set_section_sum[wb_set % 2][wb_h],
                        )
                        nisa.tensor_tensor(
                            dst=lse_pairs[wb_set % 2][:, wb_h, :],
                            data1=plane_log_sum,
                            data2=set_row_max[wb_set % 2][wb_h],
                            op=nl.subtract if _G3_EXP_FORM == 'negmin' else nl.add,
                        )
                        if wb_h == heads_per_set - 1:
                            nisa.dma_copy(
                                dst=lse.ap(
                                    pattern=[
                                        [q_tiles_per_plane, _Q_TILE],
                                        [_Q_TILE * q_tiles_per_plane, heads_per_set],
                                        [1, q_tiles_per_plane],
                                    ],
                                    offset=set_plane0[wb_set] * _Q_TILE * q_tiles_per_plane,
                                ),
                                src=lse_pairs[wb_set % 2].ap(
                                    pattern=[
                                        [heads_per_set * q_tiles_per_plane, _Q_TILE],
                                        [q_tiles_per_plane, heads_per_set],
                                        [1, q_tiles_per_plane],
                                    ],
                                    offset=0,
                                ),
                            )
                    elif return_lse and wb_group == q_tiles_per_plane - 1:
                        nisa.activation(
                            dst=plane_log_sum,
                            op=nl.log,
                            data=set_section_sum[wb_set % 2][wb_h],
                        )
                        nisa.tensor_tensor(
                            dst=lse_planes[g_wb % 2],
                            data1=plane_log_sum,
                            data2=set_row_max[wb_set % 2][wb_h],
                            op=nl.subtract if _G3_EXP_FORM == 'negmin' else nl.add,
                        )
                        _store_lse_plane(
                            lse,
                            set_plane0[wb_set] + wb_h,
                            q_tiles_per_plane,
                            lse_planes[g_wb % 2],
                        )

    for local_plane in range(planes_per_shard if num_sections > 1 else 0):
        plane_id = shard_id * planes_per_shard + local_plane
        batch_id = plane_id // n_heads
        head_id = plane_id % n_heads
        plane_base = (batch_id * seqlen * n_heads + head_id) * head_dim
        _load_kv_plane(k, v, k_transposed, v_bthd, plane_base, token_stride, head_dim, seqlen,
                       q_tiles_per_plane)

        # The online path remains the general fallback for T > 2048.
        for q_tile in nl.sequential_range(q_tiles_per_plane):
            q_start = q_tile * _Q_TILE
            q_end = q_start + _Q_TILE
            q_offset = plane_base + q_start * token_stride

            nisa.dma_transpose(
                dst=q_transposed,
                src=q.ap(
                    pattern=[[token_stride, _Q_TILE], [1, head_dim]],
                    offset=q_offset,
                ),
            )

            if num_sections > 1:
                nisa.memset(dst=running_max, value=_FLOAT32_MIN)
                nisa.memset(dst=running_sum, value=0.0)
                nisa.memset(dst=running_output, value=0.0)

            for section in nl.sequential_range(num_sections):
                section_start = section * section_size
                section_has_compute = ((not causal) or (q_end > section_start)) and (
                    (not use_sliding_window)
                    or (q_start < section_start + section_size - 1 + sliding_window)
                )

                if section_has_compute:
                    nisa.memset(dst=partial_max, value=_FLOAT32_MIN)

                    # MM1: populate all visible 512-key slices and record a
                    # per-slice row maximum.
                    for mm1_tile in nl.sequential_range(mm1_tiles_per_section):
                        k_start = section_start + mm1_tile * _MM1_TILE
                        tile_has_compute = ((not causal) or (q_end > k_start)) and (
                            (not use_sliding_window)
                            or (q_start < k_start + _MM1_TILE - 1 + sliding_window)
                        )
                        if tile_has_compute:
                            nisa.nc_matmul(
                                dst=qk_psum,
                                stationary=q_transposed,
                                moving=k_transposed[:, nl.ds(k_start, _MM1_TILE)],
                                accumulate=False,
                            )

                            needs_causal_mask = causal and (
                                q_start < k_start + _MM1_TILE
                            )
                            if needs_causal_mask or use_sliding_window:
                                nisa.tensor_copy(dst=qk_copy, src=qk_psum)
                                if needs_causal_mask:
                                    nisa.affine_select(
                                        dst=qk_masked,
                                        pattern=[[-1, _MM1_TILE]],
                                        offset=q_start - k_start,
                                        channel_multiplier=1,
                                        cmp_op=nl.greater_equal,
                                        on_true_tile=qk_copy,
                                        on_false_value=_FLOAT32_MIN,
                                    )
                                    mask_source = qk_masked
                                else:
                                    mask_source = qk_copy

                                if use_sliding_window:
                                    nisa.affine_select(
                                        dst=qk_masked,
                                        pattern=[[1, _MM1_TILE]],
                                        offset=(k_start + sliding_window - 1 - q_start),
                                        channel_multiplier=-1,
                                        cmp_op=nl.greater_equal,
                                        on_true_tile=mask_source,
                                        on_false_value=_FLOAT32_MIN,
                                    )
                                score_source = qk_masked
                            else:
                                score_source = qk_psum

                            nisa.tensor_scalar_reduce(
                                dst=scores[:, nl.ds(mm1_tile * _MM1_TILE, _MM1_TILE)],
                                data=score_source,
                                op0=nl.multiply,
                                operand0=softmax_scale,
                                reduce_op=nl.maximum,
                                reduce_res=partial_max[:, mm1_tile],
                            )

                    nisa.tensor_reduce(
                        dst=section_max,
                        op=nl.maximum,
                        data=partial_max,
                        axis=1,
                    )
                    if num_sections == 1:
                        nisa.tensor_scalar(
                            dst=negative_max,
                            data=section_max,
                            op0=nl.multiply,
                            operand0=-1.0,
                        )
                    else:
                        nisa.tensor_tensor(
                            dst=new_max,
                            data1=running_max,
                            data2=section_max,
                            op=nl.maximum,
                        )
                        nisa.activation(
                            dst=correction,
                            op=nl.exp,
                            data=new_max,
                            scale=-1.0,
                            bias=running_max,
                        )
                        nisa.tensor_scalar(
                            dst=negative_max,
                            data=new_max,
                            op0=nl.multiply,
                            operand0=-1.0,
                        )

                    # Exponentiate each visible 512-key slice relative to the
                    # updated online maximum.  DMA transpose converts
                    # [Q,512] into [128,4,Q] without occupying TensorE; the
                    # four resulting 128x128 tiles feed P@V directly.
                    nisa.memset(dst=partial_sum, value=0.0)
                    pv_psum = _psum_at((_Q_TILE, head_dim), nl.float32, 7)
                    nisa.memset(dst=pv_psum, value=0.0)
                    for exp_tile in nl.sequential_range(mm1_tiles_per_section):
                        k_start = section_start + exp_tile * _MM1_TILE
                        tile_has_compute = ((not causal) or (q_end > k_start)) and (
                            (not use_sliding_window)
                            or (q_start < k_start + _MM1_TILE - 1 + sliding_window)
                        )
                        if tile_has_compute:
                            nisa.activation_reduce(
                                dst=probabilities[
                                    :, nl.ds(exp_tile * _MM1_TILE, _MM1_TILE)
                                ],
                                op=nl.exp,
                                data=scores[:, nl.ds(exp_tile * _MM1_TILE, _MM1_TILE)],
                                bias=negative_max,
                                reduce_op=nl.add,
                                reduce_res=partial_sum[:, exp_tile],
                            )
                            nisa.dma_transpose(
                                dst=probabilities_tp.ap(
                                    [
                                        [_MM1_TILE, _V_TILE],
                                        [1, 1],
                                        [_Q_TILE, _MM1_TILE // _V_TILE],
                                        [1, _Q_TILE],
                                    ]
                                ),
                                src=probabilities.ap(
                                    [
                                        [section_size, _Q_TILE],
                                        [1, 1],
                                        [_Q_TILE, _MM1_TILE // _V_TILE],
                                        [1, _Q_TILE],
                                    ],
                                    offset=exp_tile * _MM1_TILE,
                                ),
                            )
                            for v_subtile in nl.sequential_range(_MM1_TILE // _V_TILE):
                                global_v_tile = (
                                    section * token_tiles_per_section
                                    + exp_tile * (_MM1_TILE // _V_TILE)
                                    + v_subtile
                                )
                                v_start = global_v_tile * _V_TILE
                                v_tile_has_compute = (
                                    (not causal) or (q_end > v_start)
                                ) and (
                                    (not use_sliding_window)
                                    or (
                                        q_start < v_start + _V_TILE - 1 + sliding_window
                                    )
                                )
                                if v_tile_has_compute:
                                    nisa.nc_matmul(
                                        dst=pv_psum,
                                        stationary=probabilities_tp[
                                            :,
                                            nl.ds(
                                                v_subtile * _Q_TILE,
                                                _Q_TILE,
                                            ),
                                        ],
                                        moving=v_bthd[:, global_v_tile, :],
                                        accumulate=True,
                                    )

                    nisa.tensor_reduce(
                        dst=section_sum,
                        op=nl.add,
                        data=partial_sum,
                        axis=1,
                    )
                    nisa.tensor_copy(dst=pv_sbuf, src=pv_psum)
                    if num_sections == 1:
                        nisa.tensor_copy(dst=running_sum, src=section_sum)
                        nisa.tensor_copy(dst=running_output, src=pv_sbuf)
                    else:
                        nisa.tensor_tensor(
                            dst=old_sum_scaled,
                            data1=running_sum,
                            data2=correction,
                            op=nl.multiply,
                        )
                        nisa.tensor_tensor(
                            dst=running_sum,
                            data1=old_sum_scaled,
                            data2=section_sum,
                            op=nl.add,
                        )
                        nisa.tensor_scalar(
                            dst=old_output_scaled,
                            data=running_output,
                            op0=nl.multiply,
                            operand0=correction,
                        )
                        nisa.tensor_tensor(
                            dst=running_output,
                            data1=old_output_scaled,
                            data2=pv_sbuf,
                            op=nl.add,
                        )
                        nisa.tensor_copy(dst=running_max, src=new_max)

            nisa.reciprocal(dst=reciprocal_sum, data=running_sum)
            nisa.tensor_scalar(
                dst=final_output,
                data=running_output,
                op0=nl.multiply,
                operand0=reciprocal_sum,
            )
            nisa.dma_copy(
                dst=result.ap(
                    pattern=[[token_stride, _Q_TILE], [1, head_dim]],
                    offset=q_offset,
                ),
                src=final_output.ap(
                    pattern=[[head_dim, _Q_TILE], [1, head_dim]], offset=0
                ),
            )
            if return_lse:
                _store_lse_from_running_state(
                    lse,
                    plane_id * _Q_TILE * q_tiles_per_plane + q_tile,
                    q_tiles_per_plane,
                    running_max,
                    running_sum,
                    lse_log_sum,
                    lse_value,
                )

    if return_lse:
        return result, lse
    return result



# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------
#
# Design:
#   * per plane, Q / dY (both orientations), O rows and per-section K / V / K^T
#     are DMA'd once into SBUF (double-buffered across planes) in the layouts
#     TensorE needs, so no nc_transpose is spent on inputs;
#   * visibility is tracked per 128x128 (q tile, key block); fully masked
#     blocks are skipped everywhere; exp reads the QK PSUM tile directly and
#     the single partially masked diagonal block is fixed in place on bf16 P;
#   * pairs are visited key-tile-outer, so the pairs touching one dK/dV tile
#     are consecutive and accumulate in PSUM; dQ accumulates in PSUM too when
#     its pairs are consecutive (sliding window) and in SBUF fp32 otherwise;
#   * gradients are evicted from PSUM once, straight to bf16, and stored via
#     DMA transpose; dQ uses one N<=512 matmul per key sub-tile;
#   * every SBUF tensor is placed explicitly (hand-rotated rings), the lse of
#     the whole shard is loaded once;
#   * the dQ matmuls of a pair are deferred behind the recompute of the next
#     pair(s) (software pipelining across pairs); on full-causal layers the dS^T of
#     the fully visible pairs comes from a DMA round trip through a private HBM
#     scratch instead of TensorE transposes, and Q^T / dY^T are built by TensorE
#     from the token-major rows to relieve the DMA queues;
#   * the dK/dV (and on sliding-window layers the dQ) PSUM evictions run on VectorE
#     (on ScalarE, its in-order queue would hold the dS^T scratch copies -- and
#     through them the TensorE transposes -- behind the evictions in the
#     tile-ending pairs); K rows are kept per 2048-token section so the deferred
#     dQ of a section's last pairs reads the right rows for seqlen > 2048.

_BWD_KBLK = 128
_BWD_DELTA_REDUCE = "vector"  # "vector": VectorE tensor_reduce | "scalar": ScalarE activation reduce
_BWD_PSUM_LAYOUT = "A"  # "A": qk x2, dp x2, dv x1, dk x1   "B": qk x1, dp x1, dv x2, dk x2
_BWD_MAX_RESIDENT_SEQLEN = 8192
_BWD_PIN_BASE = 0  # SBUF byte offset of the pinned plane-resident load buffers
_BWD_PDS_RING = 8  # manual P / dS ring depth (pairs)
_BWD_TP_RING = 4  # manual dS^T ring depth (pairs)
_BWD_STAGE_RING = 8  # store staging ring depth
# dQ deferral: the dQ matmuls of pair i are issued after the recompute (and dV/dK) of pair i + d.  They fill
# TensorE gaps of the following pairs and give DMA-produced dS^T blocks time to land.
_BWD_DQ_DEFER_SW0 = 3
_BWD_DQ_DEFER_SW512 = 1
# Key blocks of fully visible pairs whose dS^T is produced by DMA (dS rows -> private HBM scratch -> transposed
# HWDGE read) instead of TensorE transposes + ScalarE copies.  Full-causal layers only: the sliding-window
# layers keep the DMA fabric busy with their own loads/stores and lose with any dS^T traffic.
_BWD_DMA_TP_FULL_SW0 = [0, 1]  # trn2: [] 3319 us, [0, 1] 3096 us, [0, 1, 2, 3] 4022 us
_BWD_DMA_TP_FULL_SW512 = []
_BWD_DS_HBM_SLOTS = 4  # HBM scratch slots (pairs) for the dS round trip
# Full-causal layers: build the transposed Q / dY with TensorE transposes + ScalarE copies from the token-major
# rows instead of the two transposed HBM loads (DMA relief; the layer is DMA-bound with the dS round trip).
_BWD_QT_TENSORE_SW0 = 1
# Engine of the PSUM-resident dQ eviction (1 = VectorE tensor_scalar, 0 = ScalarE activation); dK/dV evictions are
# always on VectorE.
_BWD_DQ_EVICT_VECTOR_SW512 = 1
_BWD_DQ_EVICT_VECTOR_SW0 = 0
# Position of the deferred dQ block (matmuls + eviction) of pair i-d inside the step of pair i:
#   1 = after dS (before the dS^T transposes), 2 = after the dS^T transposes (before dV/dK), 3 = after dV/dK.
_BWD_DQ_POS_SW512 = 2  # trn2: 1 -> 2389 us, 2 -> 2274 us
_BWD_DQ_POS_SW0 = 2
# Usable SBUF bytes per partition for the explicit placement; the P/dS ring shrinks when the
# placement would not fit (only seqlen 8192 with many planes per shard; the pretrain shapes keep the default).
_BWD_SBUF_BYTES = 212984  # trn2


def _bwd_load_bthd_transposed(
    src,
    dst,
    src_offset,
    token_stride,
    head_dim,
    dst_capacity,
    num_tokens,
):
    """Gather one BTHD head chunk and transpose it to ``(D, tokens)``."""

    if head_dim == _Q_TILE:
        nisa.dma_transpose(
            dst=dst.ap(
                [
                    [dst_capacity, head_dim],
                    [16, num_tokens // 16],
                    [1, 1],
                    [1, 16],
                ]
            ),
            src=src.ap(
                [
                    [token_stride, 16],
                    [16 * token_stride, num_tokens // 16],
                    [1, 1],
                    [1, head_dim],
                ],
                offset=src_offset,
            ),
            axes=(3, 1, 2, 0),
            dge_mode=nisa.dge_mode.hwdge,
        )
    else:
        nisa.dma_transpose(
            dst=dst.ap(
                [
                    [dst_capacity, head_dim],
                    [1, 1],
                    [1, 1],
                    [1, num_tokens],
                ]
            ),
            src=src.ap(
                [
                    [token_stride, num_tokens],
                    [1, 1],
                    [1, 1],
                    [1, head_dim],
                ],
                offset=src_offset,
            ),
        )


def _bwd_load_bthd_rows(src, dst, src_offset, token_stride, head_dim, num_tokens, dst_free=None, dst_offset=0):
    """Load ``num_tokens`` BTHD rows token-major: dst[(t % 128), dst_offset/D + t // 128, :] = src[t, :].

    ``dst_free`` is the full free size of ``dst`` (defaults to the loaded span, i.e. ``dst`` is filled)."""

    n_tiles = num_tokens // _Q_TILE
    if dst_free is None:
        dst_free = n_tiles * head_dim
    nisa.dma_copy(
        dst=dst.ap(
            pattern=[[dst_free, _Q_TILE], [head_dim, n_tiles], [1, head_dim]],
            offset=dst_offset,
        ),
        src=src.ap(
            pattern=[[token_stride, _Q_TILE], [_Q_TILE * token_stride, n_tiles], [1, head_dim]],
            offset=src_offset,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )


def _bwd_store_bthd(dst, src_bf16, dst_offset, token_stride, head_dim, num_tokens, stage_tp):
    """DMA-transpose a ``(D, tokens)`` bf16 tile to token-major and store it to BTHD."""

    n_tiles = num_tokens // _Q_TILE
    nisa.dma_transpose(
        dst=stage_tp.ap(
            [[n_tiles * head_dim, _Q_TILE], [1, 1], [head_dim, n_tiles], [1, head_dim]]
        ),
        src=src_bf16.ap(
            [[num_tokens, head_dim], [1, 1], [_Q_TILE, n_tiles], [1, _Q_TILE]]
        ),
    )
    nisa.dma_copy(
        dst=dst.ap(
            pattern=[[token_stride, _Q_TILE], [_Q_TILE * token_stride, n_tiles], [1, head_dim]],
            offset=dst_offset,
        ),
        src=stage_tp.ap(
            pattern=[[n_tiles * head_dim, _Q_TILE], [head_dim, n_tiles], [1, head_dim]],
            offset=0,
        ),
        dge_mode=nisa.dge_mode.hwdge,
    )


def _bwd_compute_delta(o_rows, dy_rows, delta, product, q_tiles_per_plane, head_dim):
    """``delta = rowsum(o * dy)`` for every 128-token Q tile of the plane."""

    for q_tile in range(q_tiles_per_plane):
        span = nl.ds(q_tile * head_dim, head_dim)
        nisa.tensor_tensor(
            dst=product,
            data1=dy_rows[:, span],
            data2=o_rows[:, span],
            op=nl.multiply,
        )
        if _BWD_DELTA_REDUCE == "scalar":
            nisa.activation(
                dst=product,
                op=nl.copy,
                data=product,
                reduce_op=nl.add,
                reduce_res=delta[:, q_tile : q_tile + 1],
            )
        else:
            nisa.tensor_reduce(
                dst=delta[:, q_tile : q_tile + 1],
                op=nl.add,
                data=product,
                axis=1,
            )


def _bwd_visible_blocks(q_tile, k_start, causal, use_sliding_window, sliding_window):
    """Visibility of one 128-token Q tile against one 512-key tile.

    Returns ``None`` when no key of the tile is visible, else
    ``(lo, hi, causal_blk, window_blk)`` with ``[lo, hi)`` the 128-aligned local
    key range containing every visible key and ``*_blk`` the local 128-block
    index (or ``None``) whose diagonal needs the causal / window mask.
    """

    q_start = q_tile * _Q_TILE
    q_end = q_start + _Q_TILE
    lo, hi = 0, _MM1_TILE
    causal_blk = None
    window_blk = None
    if causal:
        vis_hi = q_end - k_start
        if vis_hi <= 0:
            return None
        hi = min(_MM1_TILE, -(-vis_hi // _BWD_KBLK) * _BWD_KBLK)
        d = q_start - k_start
        if 0 <= d < _MM1_TILE:
            causal_blk = d // _BWD_KBLK
    if use_sliding_window:
        vis_lo = q_start - sliding_window + 1 - k_start
        if vis_lo >= _MM1_TILE:
            return None
        if vis_lo > 0:
            lo = (vis_lo // _BWD_KBLK) * _BWD_KBLK
            if vis_lo % _BWD_KBLK != 0:
                window_blk = vis_lo // _BWD_KBLK
    if lo >= hi:
        return None
    return lo, hi, causal_blk, window_blk


def _bwd_psum_prepare(psum, a, b, state):
    """Prepare PSUM columns ``[a, b)`` for a matmul: returns the ``accumulate`` flag.

    ``state`` holds one bool per 128-column chunk (True = holds a partial sum).
    Untouched spans are written with accumulate=False; spans mixing touched and
    untouched chunks get their untouched chunks zeroed first (VectorE memset).
    """

    n_init = 0
    n_chunks = 0
    for c in range(a, b, _BWD_KBLK):
        n_chunks += 1
        if state[c // _BWD_KBLK]:
            n_init += 1
    if n_init == 0:
        for c in range(a, b, _BWD_KBLK):
            state[c // _BWD_KBLK] = True
        return False
    if n_init < n_chunks:
        run_start = -1
        for c in range(a, b + _BWD_KBLK, _BWD_KBLK):
            uninit = c < b and not state[c // _BWD_KBLK]
            if uninit and run_start < 0:
                run_start = c
            if (not uninit) and run_start >= 0:
                nisa.memset(dst=psum[:, nl.ds(run_start, c - run_start)], value=0.0)
                run_start = -1
        for c in range(a, b, _BWD_KBLK):
            state[c // _BWD_KBLK] = True
    return True


def _bwd_write_ds_rows(ds_hbm, row0, softmax_dx, j, c0, ncols):
    """dS rows of q tile ``j`` (columns ``[c0, c0+ncols)``) -> HBM scratch rows ``row0 + 128 j + p``."""

    nisa.dma_copy(
        dst=ds_hbm.ap([[_MM1_TILE, _Q_TILE], [1, ncols]], offset=(row0 + j * _Q_TILE) * _MM1_TILE + c0),
        src=softmax_dx[j][:, nl.ds(c0, ncols)],
        dge_mode=nisa.dge_mode.hwdge,
    )


def _bwd_read_ds_tp(ds_hbm, row0, softmax_dx_tp, tp_capacity, i, j0, n_users):
    """Transposed HWDGE read of key block ``i`` for q tiles ``j0 .. j0+n_users-1`` -> ``softmax_dx_tp``."""

    num_tokens = n_users * _Q_TILE
    nisa.dma_transpose(
        dst=softmax_dx_tp.ap(
            [[tp_capacity, _BWD_KBLK], [16, num_tokens // 16], [1, 1], [1, 16]],
            offset=i * _MM1_TILE + j0 * _Q_TILE,
        ),
        src=ds_hbm.ap(
            [[_MM1_TILE, 16], [16 * _MM1_TILE, num_tokens // 16], [1, 1], [1, _BWD_KBLK]],
            offset=(row0 + j0 * _Q_TILE) * _MM1_TILE + i * _BWD_KBLK,
        ),
        axes=(3, 1, 2, 0),
        dge_mode=nisa.dge_mode.hwdge,
    )


def _bwd_transpose_rows(src, dst, n_tiles, scratch):
    """``dst[:, t*128:(t+1)*128] = src[:, t*128:(t+1)*128]^T`` for ``n_tiles`` 128x128 bf16 tiles (TensorE + ScalarE)."""

    per_scratch = _BWD_Q_GROUP_TOKENS // _Q_TILE
    for t0 in range(0, n_tiles, per_scratch):
        n = min(per_scratch, n_tiles - t0)
        for t in range(t0, t0 + n):
            nisa.nc_transpose(
                dst=scratch[:, nl.ds((t - t0) * _Q_TILE, _Q_TILE)],
                data=src[:, nl.ds(t * _Q_TILE, _Q_TILE)],
            )
        nisa.tensor_copy(
            dst=dst[:, nl.ds(t0 * _Q_TILE, n * _Q_TILE)],
            src=scratch[:, nl.ds(0, n * _Q_TILE)],
            engine=engine.scalar,
        )


def _bwd_block_users(vis):
    """Per key block: the (contiguous) list of q tiles whose visible span covers it."""

    n_blocks = _MM1_TILE // _BWD_KBLK
    users = []
    for i in range(n_blocks):
        js = []
        for j in range(_BWD_Q_TILES_PER_GROUP):
            if vis[j] is None:
                continue
            if vis[j][0] <= i * _BWD_KBLK and i * _BWD_KBLK < vis[j][1]:
                js.append(j)
        if len(js) > 0:
            assert len(js) == js[-1] - js[0] + 1, "visible q tiles must be contiguous"
        users.append(js)
    block_order = []
    for _ in range(n_blocks):
        best_i = -1
        best_n = -1
        for i in range(n_blocks):
            already = False
            for ii in block_order:
                if ii == i:
                    already = True
            if already:
                continue
            if len(users[i]) > best_n:
                best_n = len(users[i])
                best_i = i
        block_order.append(best_i)
    return users, block_order


def _bwd_pair_dq(vis, k_rows, softmax_dx_tp, dq_psum, dq_init_state, head_dim):
    """dQ (unscaled) += dS @ K, one matmul per key block; PSUM init tracked per chunk."""

    users, block_order = _bwd_block_users(vis)
    for i in block_order:
        if not users[i]:
            continue
        a = users[i][0] * _Q_TILE
        b = (users[i][-1] + 1) * _Q_TILE
        acc = _bwd_psum_prepare(dq_psum, a, b, dq_init_state)
        nisa.nc_matmul(
            dst=dq_psum[:, nl.ds(a, b - a)],
            stationary=k_rows[:, nl.ds(i * head_dim, head_dim)],
            moving=softmax_dx_tp[:, nl.ds(i * _MM1_TILE + a, b - a)],
            accumulate=acc,
        )


def _bwd_visible_order(vis):
    """Visible q tiles of a pair, largest key span first (stable)."""

    visible = []
    for j in range(_BWD_Q_TILES_PER_GROUP):
        if vis[j] is not None:
            visible.append(j)
    ordered = []
    for _ in range(len(visible)):
        best_j = -1
        best_n = -1
        for j in visible:
            already = False
            for jj in ordered:
                if jj == j:
                    already = True
            if already:
                continue
            n = vis[j][1] - vis[j][0]
            if n > best_n:
                best_n = n
                best_j = j
        ordered.append(best_j)
    return ordered


def _bwd_pair_a1(
    vis, q_tile_start, k_start, q_tiles, k_tile, negative_lse, probabilities, qk_psums, softmax_scale, sliding_window,
):
    """Phase A1 of a pair: P = exp(scale * QK^T - lse), masked diagonal blocks fixed in place."""

    visible = _bwd_visible_order(vis)
    for j in visible:
        lo, hi, causal_blk, window_blk = vis[j]
        n = hi - lo
        qk_psum = qk_psums[j % len(qk_psums)]
        nisa.nc_matmul(
            dst=qk_psum[:, nl.ds(lo, n)],
            stationary=q_tiles[:, nl.ds(j * _Q_TILE, _Q_TILE)],
            moving=k_tile[:, nl.ds(lo, n)],
            accumulate=False,
        )
        nisa.activation(
            dst=probabilities[j][:, nl.ds(lo, n)],
            op=nl.exp,
            data=qk_psum[:, nl.ds(lo, n)],
            bias=negative_lse[:, q_tile_start + j],
            scale=softmax_scale,
        )
        q_start = (q_tile_start + j) * _Q_TILE
        if causal_blk is not None:
            blk = nl.ds(causal_blk * _BWD_KBLK, _BWD_KBLK)
            nisa.affine_select(
                dst=probabilities[j][:, blk],
                pattern=[[-1, _BWD_KBLK]],
                offset=q_start - (k_start + causal_blk * _BWD_KBLK),
                channel_multiplier=1,
                cmp_op=nl.greater_equal,
                on_true_tile=probabilities[j][:, blk],
                on_false_value=0.0,
            )
        if window_blk is not None:
            blk = nl.ds(window_blk * _BWD_KBLK, _BWD_KBLK)
            nisa.affine_select(
                dst=probabilities[j][:, blk],
                pattern=[[1, _BWD_KBLK]],
                offset=(k_start + window_blk * _BWD_KBLK) + sliding_window - 1 - q_start,
                channel_multiplier=-1,
                cmp_op=nl.greater_equal,
                on_true_tile=probabilities[j][:, blk],
                on_false_value=0.0,
            )


def _bwd_pair_dma_flags(vis, dma_blocks):
    n_blocks = _MM1_TILE // _BWD_KBLK
    users, block_order = _bwd_block_users(vis)
    is_dma = []
    for i in range(n_blocks):
        d = False
        for ii in dma_blocks:
            if ii == i:
                d = True
        is_dma.append(d and len(users[i]) > 0)
    any_dma = False
    for i in range(n_blocks):
        if is_dma[i]:
            any_dma = True
    return users, block_order, is_dma, any_dma


def _bwd_pair_a2(
    vis, q_tile_start, dy_tiles, v_tile, delta, probabilities, softmax_dx, dp_psums, dma_blocks, ds_hbm, ds_row0,
):
    """Phase A2 of a pair: dS = P * (dY V^T - delta), then the dS rows of the DMA blocks -> HBM scratch."""

    visible = _bwd_visible_order(vis)
    for j in visible:
        lo, hi, _, _ = vis[j]
        n = hi - lo
        dp_psum = dp_psums[j % len(dp_psums)]
        nisa.nc_matmul(
            dst=dp_psum[:, nl.ds(lo, n)],
            stationary=dy_tiles[:, nl.ds(j * _Q_TILE, _Q_TILE)],
            moving=v_tile[:, nl.ds(lo, n)],
            accumulate=False,
        )
        nisa.scalar_tensor_tensor(
            dst=softmax_dx[j][:, nl.ds(lo, n)],
            data=dp_psum[:, nl.ds(lo, n)],
            op0=nl.subtract,
            operand0=delta[:, q_tile_start + j],
            op1=nl.multiply,
            operand1=probabilities[j][:, nl.ds(lo, n)],
        )
    n_blocks = _MM1_TILE // _BWD_KBLK
    users, block_order, is_dma, any_dma = _bwd_pair_dma_flags(vis, dma_blocks)
    if any_dma:
        for j in visible:
            c0 = -1
            c1 = -1
            for i in range(n_blocks):
                if is_dma[i] and vis[j][0] <= i * _BWD_KBLK and i * _BWD_KBLK < vis[j][1]:
                    if c0 < 0:
                        c0 = i * _BWD_KBLK
                    c1 = (i + 1) * _BWD_KBLK
            if c0 >= 0:
                _bwd_write_ds_rows(ds_hbm, ds_row0, softmax_dx, j, c0, c1 - c0)


def _bwd_pair_t(vis, softmax_dx, softmax_dx_tp, transpose_scratch, dma_blocks, ds_hbm, ds_row0):
    """Phase T of a pair: dS^T block i of q tile j -> softmax_dx_tp[:, i*512 + j*128 : +128]
    (TensorE transposes + ScalarE copies, or transposed HWDGE reads of the HBM scratch for the DMA blocks)."""

    n_blocks = _MM1_TILE // _BWD_KBLK
    users, block_order, is_dma, any_dma = _bwd_pair_dma_flags(vis, dma_blocks)
    for i in block_order:
        js = users[i]
        if not js:
            continue
        if is_dma[i]:
            continue
        for j in js:
            nisa.nc_transpose(
                dst=transpose_scratch[:, nl.ds(j * _Q_TILE, _Q_TILE)],
                data=softmax_dx[j][:, nl.ds(i * _BWD_KBLK, _BWD_KBLK)],
            )
        span = nl.ds(js[0] * _Q_TILE, len(js) * _Q_TILE)
        nisa.tensor_copy(
            dst=softmax_dx_tp[:, nl.ds(i * _MM1_TILE + js[0] * _Q_TILE, len(js) * _Q_TILE)],
            src=transpose_scratch[:, span],
            engine=engine.scalar,
        )
    if any_dma:
        for i in block_order:
            js = users[i]
            if not js or not is_dma[i]:
                continue
            _bwd_read_ds_tp(ds_hbm, ds_row0, softmax_dx_tp, n_blocks * _MM1_TILE, i, js[0], len(js))


def _bwd_pair_k(vis, q_rows, dy_rows, probabilities, softmax_dx, dv_psum, dk_psum, kv_init_state, head_dim):
    """Phase K of a pair: dV += P^T @ dY ;  dK (unscaled) += dS^T @ Q (contract over queries)."""

    visible = _bwd_visible_order(vis)
    for j in visible:
        lo, hi, _, _ = vis[j]
        n = hi - lo
        acc_v = _bwd_psum_prepare(dv_psum, lo, hi, kv_init_state[0])
        acc_k = _bwd_psum_prepare(dk_psum, lo, hi, kv_init_state[1])
        nisa.nc_matmul(
            dst=dv_psum[:, nl.ds(lo, n)],
            stationary=dy_rows[:, nl.ds(j * head_dim, head_dim)],
            moving=probabilities[j][:, nl.ds(lo, n)],
            accumulate=acc_v,
        )
        nisa.nc_matmul(
            dst=dk_psum[:, nl.ds(lo, n)],
            stationary=q_rows[:, nl.ds(j * head_dim, head_dim)],
            moving=softmax_dx[j][:, nl.ds(lo, n)],
            accumulate=acc_k,
        )


def _bwd_step_dq(
    pd, n_pairs, schedule, k_tiles_per_section, n_blocks, head_dim, dq_resident, dq_state, first_pair_of_group,
    last_pair_of_group, k_rows_plane, softmax_dx_tp_ring, dq_psum, dq_evict_vector, softmax_scale, stage_bf16,
    stage_tp, stage_slot, n_stage, dq, plane_base, token_stride, dq_acc,
):
    """Deferred dQ of pair ``pd`` (matmuls + eviction / dq_acc update / store); returns the new stage slot."""

    if not (0 <= pd < n_pairs):
        return stage_slot
    q_group_d = schedule[pd][2]
    vis_d = schedule[pd][3]
    k_rows_span_d = nl.ds(
        (schedule[pd][0] * k_tiles_per_section + schedule[pd][1]) * n_blocks * head_dim, n_blocks * head_dim
    )
    q_span_d = nl.ds(q_group_d * _BWD_Q_GROUP_TOKENS, _BWD_Q_GROUP_TOKENS)
    if dq_resident[q_group_d]:
        if dq_state[q_group_d] is None:
            st = []
            for _ in range(n_blocks):
                st.append(False)
            dq_state[q_group_d] = st
        dq_init_state_d = dq_state[q_group_d]
    else:
        dq_init_state_d = []
        for _ in range(n_blocks):
            dq_init_state_d.append(False)
    _bwd_pair_dq(
        vis_d, k_rows_plane[:, k_rows_span_d], softmax_dx_tp_ring[pd % _BWD_TP_RING], dq_psum, dq_init_state_d, head_dim,
    )
    q_offset = plane_base + q_group_d * _BWD_Q_GROUP_TOKENS * token_stride
    if dq_resident[q_group_d]:
        if pd == last_pair_of_group[q_group_d]:
            if dq_evict_vector:
                nisa.tensor_scalar(
                    dst=stage_bf16[stage_slot], data=dq_psum, op0=nl.multiply, operand0=softmax_scale,
                    engine=engine.vector,
                )
            else:
                nisa.activation(dst=stage_bf16[stage_slot], op=nl.copy, data=dq_psum, scale=softmax_scale)
            _bwd_store_bthd(
                dq, stage_bf16[stage_slot], q_offset, token_stride, head_dim, _BWD_Q_GROUP_TOKENS, stage_tp[stage_slot],
            )
            stage_slot = (stage_slot + 1) % n_stage
    else:
        if pd == first_pair_of_group[q_group_d]:
            nisa.activation(dst=dq_acc[:, q_span_d], op=nl.copy, data=dq_psum, scale=softmax_scale)
        else:
            nisa.scalar_tensor_tensor(
                dst=dq_acc[:, q_span_d],
                data=dq_psum,
                op0=nl.multiply,
                operand0=softmax_scale,
                op1=nl.add,
                operand1=dq_acc[:, q_span_d],
            )
        if pd == last_pair_of_group[q_group_d]:
            nisa.tensor_copy(dst=stage_bf16[stage_slot], src=dq_acc[:, q_span_d], engine=engine.scalar)
            _bwd_store_bthd(
                dq, stage_bf16[stage_slot], q_offset, token_stride, head_dim, _BWD_Q_GROUP_TOKENS, stage_tp[stage_slot],
            )
            stage_slot = (stage_slot + 1) % n_stage
    return stage_slot


# Opt in to non-SSA legalization (off by default for NKI kernels): causal 3.81 -> 3.38 ms at d24.
@nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
def bthd_flash_bwd(
    q: nl.ndarray,
    k: nl.ndarray,
    v: nl.ndarray,
    o: nl.ndarray,
    dy: nl.ndarray,
    lse: nl.ndarray,
    softmax_scale: float = None,
    causal: bool = True,
    sliding_window: int = 0,
):
    """Flash-attention backward with native ``(B,T,H,D)`` tensor I/O.

    ``lse`` must be the fp32 ``(B,H,128,T//128)`` tensor returned by
    :func:`bthd_flash_fwd`.  The implementation recomputes attention weights
    section by section and returns ``(dq, dk, dv)`` directly in BTHD order.
    """

    assert q.shape == k.shape == v.shape, "q, k, and v must share a BTHD shape"
    assert q.shape == o.shape == dy.shape, "o and dy must match the Q BTHD shape"
    assert q.dtype == k.dtype == v.dtype == o.dtype == dy.dtype, (
        "q, k, v, o, and dy must share a dtype"
    )
    assert q.dtype in (nl.bfloat16, nl.float16), (
        "q, k, v, o, and dy must be bf16 or fp16; fp32 doubles attention "
        "activation and gradient memory and is not supported"
    )

    batch, seqlen, n_heads, head_dim = q.shape
    q_tiles_per_plane = seqlen // _Q_TILE
    assert lse.shape == (batch, n_heads, _Q_TILE, q_tiles_per_plane), (
        "lse must have shape (B,H,128,T//128)"
    )
    assert lse.dtype == nl.float32, "lse must be fp32"
    assert head_dim <= _Q_TILE, "head_dim must be <= 128"
    assert head_dim % 16 == 0, "head_dim must be a multiple of 16"
    assert seqlen % _MM1_TILE == 0, "seqlen must be a multiple of 512"
    assert seqlen <= _BWD_MAX_RESIDENT_SEQLEN, "seqlen must be <= 8192 (plane-resident Q/dY)"
    assert sliding_window in (0, 512), "sliding_window must be 0 or 512"
    assert sliding_window == 0 or causal, "sliding-window attention must be causal"

    section_size = min(seqlen, _MAX_SECTION)
    assert seqlen % section_size == 0, "seqlen > 2048 must be divisible by 2048"
    num_sections = seqlen // section_size
    k_tiles_per_section = section_size // _MM1_TILE
    q_group_count = seqlen // _BWD_Q_GROUP_TOKENS
    use_sliding_window = sliding_window > 0
    n_blocks = _MM1_TILE // _BWD_KBLK

    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)

    dq = nl.ndarray(q.shape, dtype=q.dtype, buffer=nl.shared_hbm)
    dk = nl.ndarray(k.shape, dtype=k.dtype, buffer=nl.shared_hbm)
    dv = nl.ndarray(v.shape, dtype=v.dtype, buffer=nl.shared_hbm)

    grid_ndim = nl.program_ndim()
    if grid_ndim == 0:
        num_shards = 1
        shard_id = 0
    else:
        assert grid_ndim == 1, "bthd_flash_bwd expects an LNC1/LNC2 grid"
        num_shards = nl.num_programs(axes=0)
        shard_id = nl.program_id(axis=0)

    num_planes = batch * n_heads
    assert num_planes % num_shards == 0, "B*H must be divisible by the LNC degree"
    planes_per_shard = num_planes // num_shards
    token_stride = n_heads * head_dim

    # ---- pair schedule (compile time): key-tile-outer within a section.
    schedule = []  # (section, local_k_tile, q_group, vis)
    for section in range(num_sections):
        section_start = section * section_size
        for local_k_tile in range(k_tiles_per_section):
            k_start = section_start + local_k_tile * _MM1_TILE
            for q_group in range(q_group_count):
                vis = []
                any_visible = False
                for j in range(_BWD_Q_TILES_PER_GROUP):
                    vv = _bwd_visible_blocks(
                        q_group * _BWD_Q_TILES_PER_GROUP + j, k_start, causal,
                        use_sliding_window, sliding_window,
                    )
                    vis.append(vv)
                    if vv is not None:
                        any_visible = True
                if any_visible:
                    schedule.append((section, local_k_tile, q_group, vis))
    n_pairs = len(schedule)
    first_pair_of_group = []
    last_pair_of_group = []
    pairs_of_group = []
    for _ in range(q_group_count):
        first_pair_of_group.append(-1)
        last_pair_of_group.append(-1)
        pairs_of_group.append(0)
    last_pair_of_tile = []
    for _ in range(num_sections * k_tiles_per_section):
        last_pair_of_tile.append(-1)
    for idx in range(n_pairs):
        g = schedule[idx][2]
        t = schedule[idx][0] * k_tiles_per_section + schedule[idx][1]
        if first_pair_of_group[g] < 0:
            first_pair_of_group[g] = idx
        last_pair_of_group[g] = idx
        pairs_of_group[g] = pairs_of_group[g] + 1
        last_pair_of_tile[t] = idx
    # dQ of a group stays in PSUM iff all of its pairs are consecutive (single section).
    dq_resident = []
    for g in range(q_group_count):
        dq_resident.append(
            num_sections == 1
            and pairs_of_group[g] > 0
            and last_pair_of_group[g] - first_pair_of_group[g] + 1 == pairs_of_group[g]
        )
    any_dq_sbuf = False
    for g in range(q_group_count):
        if pairs_of_group[g] > 0 and not dq_resident[g]:
            any_dq_sbuf = True

    # ---- SBUF: plane-resident inputs, double-buffered across planes when they fit.
    rows_free = q_tiles_per_plane * head_dim
    plane_ring = 3 if seqlen <= 2048 else (2 if seqlen <= 4096 else 1)
    q_res = []
    dy_res = []
    q_rows_res = []
    dy_rows_res = []
    o_rows_res = []
    k_res = []
    v_res = []
    k_rows_res = []
    # The plane-resident load destinations are hand double-buffered (plane_ring), so they are
    # pinned at fixed SBUF addresses: this keeps the backend's load-address rotation
    # (--max-load-color-rotation, enabled by torch_neuronx's --model-type transformer) from
    # re-allocating them every plane, which reschedules the whole kernel.
    elem_bytes = 2
    res_bytes = seqlen * elem_bytes
    rows_bytes = rows_free * elem_bytes
    sec_bytes = section_size * elem_bytes
    # Token-major K rows of the whole plane (one 4 KiB section slice per 2048-token section): the deferred
    # dQ of the last pairs of a section must still read that section's rows after the next section's K^T/V^T
    # have been loaded, so every section keeps its own rows (sections are 2048 tokens; seqlen <= 8192).
    slot_bytes = 2 * res_bytes + 3 * rows_bytes + 2 * sec_bytes + rows_bytes
    for r in range(plane_ring):
        base = _BWD_PIN_BASE + r * slot_bytes
        q_res.append(nl.ndarray((head_dim, seqlen), dtype=q.dtype, buffer=nl.sbuf, address=(0, base)))
        dy_res.append(nl.ndarray((head_dim, seqlen), dtype=q.dtype, buffer=nl.sbuf, address=(0, base + res_bytes)))
        q_rows_res.append(
            nl.ndarray((_Q_TILE, rows_free), dtype=q.dtype, buffer=nl.sbuf, address=(0, base + 2 * res_bytes))
        )
        dy_rows_res.append(
            nl.ndarray((_Q_TILE, rows_free), dtype=q.dtype, buffer=nl.sbuf,
                       address=(0, base + 2 * res_bytes + rows_bytes))
        )
        o_rows_res.append(
            nl.ndarray((_Q_TILE, rows_free), dtype=q.dtype, buffer=nl.sbuf,
                       address=(0, base + 2 * res_bytes + 2 * rows_bytes))
        )
        k_res.append(
            nl.ndarray((head_dim, section_size), dtype=q.dtype, buffer=nl.sbuf,
                       address=(0, base + 2 * res_bytes + 3 * rows_bytes))
        )
        v_res.append(
            nl.ndarray((head_dim, section_size), dtype=q.dtype, buffer=nl.sbuf,
                       address=(0, base + 2 * res_bytes + 3 * rows_bytes + sec_bytes))
        )
        k_rows_res.append(
            nl.ndarray((_Q_TILE, rows_free), dtype=q.dtype, buffer=nl.sbuf,
                       address=(0, base + 2 * res_bytes + 3 * rows_bytes + 2 * sec_bytes))
        )
    # Every SBUF tensor is placed explicitly (the backend allocator then has nothing to rotate,
    # so the schedule no longer depends on its --max-load-color-rotation setting); the pair-level
    # tensors are hand-rotated over _BWD_PDS_RING / _BWD_TP_RING copies instead.
    off = _BWD_PIN_BASE + plane_ring * slot_bytes
    lse_bytes = q_tiles_per_plane * 4
    # lse of every plane of this shard, loaded once (one static DMA, one negate) instead of one
    # tiny DMA + negate per plane on the plane-start critical path.
    negative_lse_all = nl.ndarray(
        (_Q_TILE, planes_per_shard * q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf, address=(0, off)
    )
    off += planes_per_shard * lse_bytes
    delta = nl.ndarray((_Q_TILE, q_tiles_per_plane), dtype=nl.float32, buffer=nl.sbuf, address=(0, off))
    off += lse_bytes
    delta_product = nl.ndarray((_Q_TILE, head_dim), dtype=nl.float32, buffer=nl.sbuf, address=(0, off))
    off += head_dim * 4
    if any_dq_sbuf:
        dq_acc = nl.ndarray((head_dim, seqlen), dtype=nl.float32, buffer=nl.sbuf, address=(0, off))
        off += seqlen * 4
        dq_acc_or_none = dq_acc
    else:
        dq_acc_or_none = None

    pds_bytes = _MM1_TILE * elem_bytes
    # Bytes per partition of everything placed after the P/dS ring (dS^T ring + store staging).
    tail_bytes = _BWD_TP_RING * n_blocks * _MM1_TILE * elem_bytes + _BWD_STAGE_RING * (
        _MM1_TILE * elem_bytes + (_MM1_TILE // _Q_TILE) * head_dim * elem_bytes
    )
    pds_ring = _BWD_PDS_RING
    while pds_ring > 4 and off + pds_ring * 2 * _BWD_Q_TILES_PER_GROUP * pds_bytes + tail_bytes > _BWD_SBUF_BYTES:
        pds_ring -= 1
    assert off + pds_ring * 2 * _BWD_Q_TILES_PER_GROUP * pds_bytes + tail_bytes <= _BWD_SBUF_BYTES, (
        "explicit SBUF placement does not fit: too many planes per shard for this seqlen"
    )
    probabilities_ring = []
    softmax_dx_ring = []
    for r in range(pds_ring):
        probabilities = []
        softmax_dx = []
        for j in range(_BWD_Q_TILES_PER_GROUP):
            probabilities.append(nl.ndarray((_Q_TILE, _MM1_TILE), dtype=q.dtype, buffer=nl.sbuf, address=(0, off)))
            off += pds_bytes
            softmax_dx.append(nl.ndarray((_Q_TILE, _MM1_TILE), dtype=q.dtype, buffer=nl.sbuf, address=(0, off)))
            off += pds_bytes
        probabilities_ring.append(probabilities)
        softmax_dx_ring.append(softmax_dx)
    softmax_dx_tp_ring = []
    for r in range(_BWD_TP_RING):
        softmax_dx_tp_ring.append(
            nl.ndarray((_Q_TILE, n_blocks * _MM1_TILE), dtype=q.dtype, buffer=nl.sbuf, address=(0, off))
        )
        off += n_blocks * _MM1_TILE * elem_bytes

    # Store staging (bf16 (D,512) + token-major transpose), round-robin.
    n_stage = _BWD_STAGE_RING
    stage_bf16 = []
    stage_tp = []
    for _ in range(n_stage):
        stage_bf16.append(nl.ndarray((head_dim, _MM1_TILE), dtype=q.dtype, buffer=nl.sbuf, address=(0, off)))
        off += _MM1_TILE * elem_bytes
        stage_tp.append(
            nl.ndarray((_Q_TILE, (_MM1_TILE // _Q_TILE) * head_dim), dtype=q.dtype, buffer=nl.sbuf, address=(0, off))
        )
        off += (_MM1_TILE // _Q_TILE) * head_dim * elem_bytes
    stage_slot = 0

    # ---- PSUM (16 KiB / partition): 15 KiB used.
    if _BWD_PSUM_LAYOUT == "B":
        n_qk, n_dp, n_kv = 1, 1, 2
    else:
        n_qk, n_dp, n_kv = 2, 2, 1
    qk_psums = []
    for _ in range(n_qk):
        qk_psums.append(nl.ndarray((_Q_TILE, _MM1_TILE), dtype=nl.float32, buffer=nl.psum))
    dp_psums = []
    for _ in range(n_dp):
        dp_psums.append(nl.ndarray((_Q_TILE, _MM1_TILE), dtype=nl.float32, buffer=nl.psum))
    dv_psums = []
    dk_psums = []
    for _ in range(n_kv):
        dv_psums.append(nl.ndarray((head_dim, _MM1_TILE), dtype=nl.float32, buffer=nl.psum))
        dk_psums.append(nl.ndarray((head_dim, _MM1_TILE), dtype=nl.float32, buffer=nl.psum))
    dq_psum = nl.ndarray((head_dim, _MM1_TILE), dtype=nl.float32, buffer=nl.psum)
    transpose_scratch = nl.ndarray((_Q_TILE, _BWD_Q_GROUP_TOKENS), dtype=q.dtype, buffer=nl.psum)

    # dQ deferral and DMA-transposed dS^T blocks (per layer type).
    if use_sliding_window:
        dq_defer = _BWD_DQ_DEFER_SW512
        dma_full = _BWD_DMA_TP_FULL_SW512
    else:
        dq_defer = _BWD_DQ_DEFER_SW0
        dma_full = _BWD_DMA_TP_FULL_SW0
    assert dq_defer < _BWD_TP_RING and dq_defer < pds_ring, "dQ deferral must fit the rings"
    assert dq_defer < _BWD_DS_HBM_SLOTS, "dQ deferral must fit the HBM scratch slots"
    use_ds_hbm = len(dma_full) > 0
    # PSUM-resident dQ eviction engine: VectorE on sliding-window layers (every q group is resident; the eviction
    # otherwise heads ScalarE's queue in the tile-ending pairs), ScalarE on full-causal layers (only q group 0 is
    # resident and its eviction on VectorE lands in front of the next tile's dS STTs: sw0 +1.5%).
    if use_sliding_window:
        dq_evict_vector = _BWD_DQ_EVICT_VECTOR_SW512
        dq_pos = _BWD_DQ_POS_SW512
    else:
        dq_evict_vector = _BWD_DQ_EVICT_VECTOR_SW0
        dq_pos = _BWD_DQ_POS_SW0
    assert dq_defer > 0, "the step loop always defers dQ (dq_defer >= 1)"
    if use_ds_hbm:
        # dS rows of up to _BWD_DS_HBM_SLOTS pairs per shard: rows (slot, q) x 512 keys, bf16.
        ds_hbm = nl.ndarray(
            (num_shards * _BWD_DS_HBM_SLOTS * _BWD_Q_GROUP_TOKENS, _MM1_TILE), dtype=q.dtype, buffer=nl.private_hbm
        )
    else:
        ds_hbm = None
    qt_tensore = _BWD_QT_TENSORE_SW0 and not use_sliding_window and head_dim == _Q_TILE

    nisa.dma_copy(
        dst=negative_lse_all.ap(
            pattern=[[planes_per_shard * q_tiles_per_plane, _Q_TILE], [q_tiles_per_plane, planes_per_shard],
                     [1, q_tiles_per_plane]],
            offset=0,
        ),
        src=lse.ap(
            pattern=[[q_tiles_per_plane, _Q_TILE], [_Q_TILE * q_tiles_per_plane, planes_per_shard],
                     [1, q_tiles_per_plane]],
            offset=shard_id * planes_per_shard * _Q_TILE * q_tiles_per_plane,
        ),
        dge_mode=nisa.dge_mode.none,
    )
    nisa.tensor_scalar(dst=negative_lse_all, data=negative_lse_all, op0=nl.multiply, operand0=-1.0)

    for local_plane in range(planes_per_shard):
        plane_id = shard_id * planes_per_shard + local_plane
        batch_id = plane_id // n_heads
        head_id = plane_id % n_heads
        plane_base = (batch_id * seqlen * n_heads + head_id) * head_dim
        negative_lse = negative_lse_all[:, nl.ds(local_plane * q_tiles_per_plane, q_tiles_per_plane)]
        slot = local_plane % plane_ring
        q_tiles_all = q_res[slot]
        dy_tiles_all = dy_res[slot]
        q_rows = q_rows_res[slot]
        dy_rows = dy_rows_res[slot]
        o_rows = o_rows_res[slot]
        k_section = k_res[slot]
        v_section = v_res[slot]
        k_rows_plane = k_rows_res[slot]

        # Plane-resident Q-side loads.
        if not qt_tensore:
            _bwd_load_bthd_transposed(q, q_tiles_all, plane_base, token_stride, head_dim, seqlen, seqlen)
            _bwd_load_bthd_transposed(dy, dy_tiles_all, plane_base, token_stride, head_dim, seqlen, seqlen)
        _bwd_load_bthd_rows(q, q_rows, plane_base, token_stride, head_dim, seqlen)
        _bwd_load_bthd_rows(dy, dy_rows, plane_base, token_stride, head_dim, seqlen)
        _bwd_load_bthd_rows(o, o_rows, plane_base, token_stride, head_dim, seqlen)
        if qt_tensore:
            _bwd_transpose_rows(q_rows, q_tiles_all, q_tiles_per_plane, transpose_scratch)
            _bwd_transpose_rows(dy_rows, dy_tiles_all, q_tiles_per_plane, transpose_scratch)
        _bwd_compute_delta(o_rows, dy_rows, delta, delta_product, q_tiles_per_plane, head_dim)

        dq_state = []
        for _ in range(q_group_count):
            dq_state.append(None)
        kv_state = []
        for _ in range(k_tiles_per_section):
            kv_state.append(None)
        current_section = -1

        for step in range(n_pairs + dq_defer):
            idx = step
            pd = step - dq_defer
            if idx < n_pairs:
                section = schedule[idx][0]
                local_k_tile = schedule[idx][1]
                q_group = schedule[idx][2]
                vis = schedule[idx][3]
                section_start = section * section_size
                k_start = section_start + local_k_tile * _MM1_TILE
                q_tile_start = q_group * _BWD_Q_TILES_PER_GROUP
                q_span = nl.ds(q_group * _BWD_Q_GROUP_TOKENS, _BWD_Q_GROUP_TOKENS)
                k_span = nl.ds(local_k_tile * _MM1_TILE, _MM1_TILE)

                if section != current_section:
                    section_offset = plane_base + section_start * token_stride
                    _bwd_load_bthd_transposed(
                        k, k_section, section_offset, token_stride, head_dim, section_size, section_size
                    )
                    _bwd_load_bthd_transposed(
                        v, v_section, section_offset, token_stride, head_dim, section_size, section_size
                    )
                    _bwd_load_bthd_rows(
                        k, k_rows_plane, section_offset, token_stride, head_dim, section_size,
                        rows_free, section * (section_size // _Q_TILE) * head_dim,
                    )
                    current_section = section

                full = True
                for j in range(_BWD_Q_TILES_PER_GROUP):
                    if vis[j] is None or vis[j][0] != 0 or vis[j][1] != _MM1_TILE:
                        full = False
                if full:
                    dma_blocks = dma_full
                else:
                    dma_blocks = []
                if use_ds_hbm:
                    ds_row0 = (shard_id * _BWD_DS_HBM_SLOTS + idx % _BWD_DS_HBM_SLOTS) * _BWD_Q_GROUP_TOKENS
                else:
                    ds_row0 = 0

                # phase A1: P = exp(scale QK^T - lse) (+ masks)
                _bwd_pair_a1(
                    vis, q_tile_start, k_start, q_tiles_all[:, q_span], k_section[:, k_span], negative_lse,
                    probabilities_ring[idx % pds_ring], qk_psums, softmax_scale, sliding_window,
                )
                # phase A2: dS = P (dP - delta) (+ dS rows -> HBM scratch on the DMA-transposed pairs)
                _bwd_pair_a2(
                    vis, q_tile_start, dy_tiles_all[:, q_span], v_section[:, k_span], delta,
                    probabilities_ring[idx % pds_ring], softmax_dx_ring[idx % pds_ring], dp_psums,
                    dma_blocks, ds_hbm, ds_row0,
                )
                if dq_pos == 1:
                    stage_slot = _bwd_step_dq(
                        pd, n_pairs, schedule, k_tiles_per_section, n_blocks, head_dim, dq_resident, dq_state,
                        first_pair_of_group, last_pair_of_group, k_rows_plane, softmax_dx_tp_ring, dq_psum,
                        dq_evict_vector, softmax_scale, stage_bf16, stage_tp, stage_slot, n_stage, dq, plane_base,
                        token_stride, dq_acc_or_none,
                    )
                # phase T: dS^T (TensorE transposes + ScalarE copies, or transposed HWDGE reads of the scratch)
                _bwd_pair_t(
                    vis, softmax_dx_ring[idx % pds_ring], softmax_dx_tp_ring[idx % _BWD_TP_RING],
                    transpose_scratch, dma_blocks, ds_hbm, ds_row0,
                )
                if dq_pos == 2:
                    stage_slot = _bwd_step_dq(
                        pd, n_pairs, schedule, k_tiles_per_section, n_blocks, head_dim, dq_resident, dq_state,
                        first_pair_of_group, last_pair_of_group, k_rows_plane, softmax_dx_tp_ring, dq_psum,
                        dq_evict_vector, softmax_scale, stage_bf16, stage_tp, stage_slot, n_stage, dq, plane_base,
                        token_stride, dq_acc_or_none,
                    )
                # phase K: dV += P^T dY, dK += dS^T Q (PSUM-resident across the pairs of the key tile)
                q_rows_span = nl.ds(q_tile_start * head_dim, _BWD_Q_TILES_PER_GROUP * head_dim)
                if kv_state[local_k_tile] is None:
                    st_v = []
                    st_k = []
                    for _ in range(n_blocks):
                        st_v.append(False)
                        st_k.append(False)
                    kv_state[local_k_tile] = [st_v, st_k]
                kv_slot = local_k_tile % n_kv
                _bwd_pair_k(
                    vis, q_rows[:, q_rows_span], dy_rows[:, q_rows_span], probabilities_ring[idx % pds_ring],
                    softmax_dx_ring[idx % pds_ring], dv_psums[kv_slot], dk_psums[kv_slot], kv_state[local_k_tile],
                    head_dim,
                )
                if dq_pos == 3:
                    stage_slot = _bwd_step_dq(
                        pd, n_pairs, schedule, k_tiles_per_section, n_blocks, head_dim, dq_resident, dq_state,
                        first_pair_of_group, last_pair_of_group, k_rows_plane, softmax_dx_tp_ring, dq_psum,
                        dq_evict_vector, softmax_scale, stage_bf16, stage_tp, stage_slot, n_stage, dq, plane_base,
                        token_stride, dq_acc_or_none,
                    )
            else:
                # drain: the deferred dQ of the last pairs of the plane
                stage_slot = _bwd_step_dq(
                    pd, n_pairs, schedule, k_tiles_per_section, n_blocks, head_dim, dq_resident, dq_state,
                    first_pair_of_group, last_pair_of_group, k_rows_plane, softmax_dx_tp_ring, dq_psum,
                    dq_evict_vector, softmax_scale, stage_bf16, stage_tp, stage_slot, n_stage, dq, plane_base,
                    token_stride, dq_acc_or_none,
                )

            # dK / dV eviction + store at the last pair of a key tile
            if idx < n_pairs:
                tile_index = section * k_tiles_per_section + local_k_tile
                if idx == last_pair_of_tile[tile_index]:
                    k_offset = plane_base + k_start * token_stride
                    nisa.tensor_scalar(
                        dst=stage_bf16[stage_slot], data=dk_psums[kv_slot], op0=nl.multiply, operand0=softmax_scale,
                        engine=engine.vector,
                    )
                    _bwd_store_bthd(
                        dk, stage_bf16[stage_slot], k_offset, token_stride, head_dim, _MM1_TILE,
                        stage_tp[stage_slot],
                    )
                    stage_slot = (stage_slot + 1) % n_stage
                    nisa.tensor_copy(dst=stage_bf16[stage_slot], src=dv_psums[kv_slot], engine=engine.vector)
                    _bwd_store_bthd(
                        dv, stage_bf16[stage_slot], k_offset, token_stride, head_dim, _MM1_TILE,
                        stage_tp[stage_slot],
                    )
                    stage_slot = (stage_slot + 1) % n_stage
                    kv_state[local_k_tile] = None

    return dq, dk, dv

