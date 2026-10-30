"""NKI embedding-backward kernel for Neuron (LNC2): sorted one-hot matmul scatter-add.

Pure kernel module; the token-sort plan and the autograd.Function live in nanochat/trn/embedding.py.

Why: the default ``aten::embedding_dense_backward`` (13 launches per step: wte + 12 value-embedding tables, 16384
token rows x 1536 bf16 into a 32768-row table) is an HLO scatter that runs as a serialized per-row
read-modify-write on one physical core: ~11.4 ms per launch, ~148 ms per step. A scatter-add of 50 MB should
take a fraction of a millisecond.

How (exact, race-free, no atomics):
  plan (once per step, shared by all 13 tables; torch ops, ~1.5 ms):
      sort the token ids (torch.sort int32) -> sorted tiles of 128 tokens. The (sorted-tile, vocab-tile) pairs form a
      monotone staircase, so at most N/128 + V/128 - 1 pairs exist. Per sorted tile j with vocab range [vlo_j, vhi_j]:
      slot 0 = vlo_j and slot 1 = vhi_j are "boundary" vocab tiles that may also receive rows from the neighbouring
      tiles (accumulated in fp32 along the chain), the vocab tiles strictly inside (vlo_j, vhi_j) are touched by tile j
      only ("interior": no accumulation). Un-hit vocab tiles get an explicit zero store.
  kernel (nki, grid (2,) = the two physical cores of an LNC2 logical core, each owning half the sorted tiles):
      Phase A: for each of its sorted tiles, gather the 128 grad rows once (indirect DMA), build the 128x128 one-hot
      oh[t, r] = (id_t == 128 v + r) on the Vector engine and compute oh^T @ grad_tile on the Tensor engine (bf16 * {0,1}
      products are exact, PSUM accumulation is fp32); the two boundary slots run acc = acc_prev * keep + psum in fp32
      (two alternating SBUF accumulators), round once to bf16 on the Scalar engine and store the finished tile
      (HWDGE store on the Sync engine, out-of-bounds destination = "no store").
      Phase B: interior vocab tiles and zero stores, batched full-row gathers, cast straight from PSUM.
      The vocab tile that straddles the two cores' tile ranges is finished after a core barrier (both fp32 partials
      parked in a shared scratch, core 0 adds them).
  result: fp32 sum of the same bf16 values, rounded once to bf16 (torch's formula), up to fp32 summation order.
  Measured bitwise equal to torch's eager result on real batches (bench/kvar5.py).

"""
from __future__ import annotations

P_MAX = 128
N_TILE = 512          # fp32 PSUM bank = 512 columns
PHASE_B_BATCH = 16    # Phase-B slots gathered per DMA instruction

_NKI_OK = False
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.compiler import CompileConfig
    _NKI_OK = True
except ImportError:  # pragma: no cover  (non-Neuron host)
    pass


def kernel_assert(condition: bool, error_text: str):
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


if _NKI_OK:

    def _store(dst_t, D, dest, src):
        """Store a (128, D) SBUF tile to dst_t[dest:dest+128, :] (dest = int32 [1,1] SBUF); rows >= len(dst_t) skip."""
        nisa.dma_copy(dst=dst_t.ap(pattern=[[D, P_MAX], [1, D]], scalar_offset=dest, indirect_dim=0), src=src,
                      oob_mode=nisa.oob_mode.skip, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)

    def _onehot_matmul(ps, oh, mv, D):
        for cc in nl.static_range((D + N_TILE - 1) // N_TILE):
            c0 = cc * N_TILE
            cw = min(N_TILE, D - c0)
            nisa.nc_matmul(dst=ps[:, c0:c0 + cw], stationary=oh, moving=mv[:, c0:c0 + cw], accumulate=False)

    # Opt in to non-SSA legalization (off by default for NKI kernels): 0.943 -> 0.890 ms at d24.
    @nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
    def embedding_backward_sorted_v5(grad, perm_T, tA, tile_info, perm_B, tB, dest_B, fix_info, num_weights: int):
        """grad (N, D) bf16 contiguous. perm_T (128, n_tiles) int32: grad row of (partition p, sorted tile j).
        tA (128, 2*n_tiles) fp32: id - 128*v for slot 0 (v = vlo_j) and slot 1 (v = vhi_j, or -1 = pass-through).
        tile_info (n_tiles, 6) int32: keep0, dest0, keep1, dest1, scratch_dest0, scratch_dest1 (row starts; out of
        range = no store). perm_B (128, PB) int32 (per chunk of PHASE_B_BATCH slots in DMA-batching column-major order),
        tB (128, PB) fp32, dest_B (PB, 1) int32. fix_info (1, 1) int32: row start of the straddling vocab tile.
        Returns grad_weight (num_weights, D) bf16."""
        N, D = grad.shape
        n_tiles = perm_T.shape[1]
        PB = perm_B.shape[1]
        B = PHASE_B_BATCH
        V = num_weights
        grid = nl.num_programs(0)
        core = nl.program_id(0)
        kernel_assert(grid == 2, "kernel needs LNC2 (grid (2,))")
        kernel_assert(N == n_tiles * P_MAX and n_tiles % 2 == 0, "token rows must be a multiple of 256")
        kernel_assert(V % P_MAX == 0 and PB % (2 * B) == 0, "vocab / slot shapes")
        T2 = n_tiles // 2
        PB2 = PB // 2
        gw = nl.ndarray((V, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        scratch = nl.ndarray((2 * P_MAX, D), dtype=nl.float32, buffer=nl.shared_hbm)

        iota_row = nl.ndarray((P_MAX, P_MAX), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(iota_row, pattern=[[1, P_MAX]], offset=0, channel_multiplier=0)
        j0 = core * T2
        tA0 = nl.ndarray((P_MAX, T2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tA0, src=tA[:, j0:j0 + T2])
        tA1 = nl.ndarray((P_MAX, T2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tA1, src=tA[:, n_tiles + j0:n_tiles + j0 + T2])
        ti_all = nl.ndarray((P_MAX, T2 * 6), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti_all, src=tile_info.ap(pattern=[[0, P_MAX], [1, T2 * 6]], offset=j0 * 6))
        ti_f = nl.ndarray((P_MAX, T2 * 6), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ti_f, src=ti_all)
        perm_sb = nl.ndarray((P_MAX, T2), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=perm_sb, src=perm_T[:, j0:j0 + T2])
        s0 = core * PB2
        permB_sb = nl.ndarray((P_MAX, PB2), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=permB_sb, src=perm_B[:, s0:s0 + PB2])
        tB_sb = nl.ndarray((P_MAX, PB2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tB_sb, src=tB[:, s0:s0 + PB2])
        dB_all = nl.ndarray((P_MAX, PB2), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=dB_all, src=dest_B.ap(pattern=[[0, P_MAX], [1, PB2]], offset=s0))
        fx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=fx, src=fix_info)

        acc0 = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc0, 0.0)
        acc1 = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(acc1, 0.0)

        # Phase A: this core's sorted tiles, boundary vocab tiles with the fp32 accumulator chain
        for j in nl.static_range(T2):
            gt = nl.ndarray((P_MAX, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=gt, src=grad.ap(pattern=[[D, P_MAX], [1, D]], vector_offset=perm_sb[:, j:j + 1], indirect_dim=0))
            oh0 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=oh0, data=iota_row, op0=nl.equal, operand0=tA0[:, j:j + 1], engine=nisa.engine.vector)
            ps0 = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.psum)
            _onehot_matmul(ps0, oh0, gt, D)
            nisa.scalar_tensor_tensor(dst=acc0, data=acc1, op0=nl.multiply, operand0=ti_f[:, 6 * j:6 * j + 1], op1=nl.add, operand1=ps0)
            accb0 = nl.ndarray((P_MAX, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=accb0, op=nl.copy, data=acc0)
            _store(gw, D, ti_all[0:1, 6 * j + 1:6 * j + 2], accb0)
            _store(scratch, D, ti_all[0:1, 6 * j + 4:6 * j + 5], acc0)
            oh1 = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=oh1, data=iota_row, op0=nl.equal, operand0=tA1[:, j:j + 1], engine=nisa.engine.vector)
            ps1 = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.psum)
            _onehot_matmul(ps1, oh1, gt, D)
            nisa.scalar_tensor_tensor(dst=acc1, data=acc0, op0=nl.multiply, operand0=ti_f[:, 6 * j + 2:6 * j + 3], op1=nl.add, operand1=ps1)
            accb1 = nl.ndarray((P_MAX, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=accb1, op=nl.copy, data=acc1)
            _store(gw, D, ti_all[0:1, 6 * j + 3:6 * j + 4], accb1)
            _store(scratch, D, ti_all[0:1, 6 * j + 5:6 * j + 6], acc1)

        # Phase B: interior vocab tiles and zero stores of un-hit tiles (no accumulation)
        for c in nl.static_range(PB2 // B):
            gbuf = nl.ndarray((P_MAX, B * D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=gbuf, src=grad.ap(pattern=[[D, P_MAX * B], [1, D]], vector_offset=permB_sb[:, c * B:(c + 1) * B], indirect_dim=0),
                          oob_mode=nisa.oob_mode.skip)
            for b in nl.static_range(B):
                s = c * B + b
                oh = nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=oh, data=iota_row, op0=nl.equal, operand0=tB_sb[:, s:s + 1], engine=nisa.engine.vector)
                ps = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.psum)
                _onehot_matmul(ps, oh, gbuf[:, b * D:(b + 1) * D], D)
                accb = nl.ndarray((P_MAX, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                if s % 2 == 1:
                    nisa.tensor_copy(dst=accb, src=ps, engine=nisa.engine.vector)
                else:
                    nisa.activation(dst=accb, op=nl.copy, data=ps)
                _store(gw, D, dB_all[0:1, s:s + 1], accb)

        # the vocab tile straddling the two cores' tile ranges: add the two fp32 partials
        nisa.core_barrier(data=scratch, cores=(0, 1))
        if core == 0:
            pa = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pa, src=scratch[0:P_MAX, :])
            pb = nl.ndarray((P_MAX, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pb, src=scratch[P_MAX:2 * P_MAX, :])
            nisa.tensor_tensor(dst=pa, data1=pa, data2=pb, op=nl.add)
            pab = nl.ndarray((P_MAX, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=pab, op=nl.copy, data=pa)
            _store(gw, D, fx, pab)
        return gw

    _KERNEL_G2 = embedding_backward_sorted_v5[2]
