"""DistMuonAdamW for AWS Trainium: fp32 master shards + the fused 6-launch step.

Same algorithm and hyperparameters as nanochat.optim.DistMuonAdamW (AdamW for embeddings / lm_head / scalars,
Muon for the matrices, ZeRO-2 style sharding); the Trainium version differs in three things that were all
measured on trn3 at d24 / 64 ranks:

1. Master weights: the transformer-block parameters are bf16 (the model's COMPUTE_DTYPE copy) and each rank keeps
   an fp32 master for ONLY the Muon matrices it owns (`fp32_master` in the optimizer state), never re-read from
   p.data. The reduce_scatter slab and the Nesterov momentum are fp32, the all_gather is bf16 (the gathered value is
   written straight into bf16 p.data). Muon arithmetic bit-identical to upstream's replicated fp32 master; it is
   what makes MBS=8 fit. Without a master (all-bf16) val bpb regresses by +0.2 %, so that is not an option here.
2. Merged Muon groups: a matrix and its transpose share a group (mlp.c_fc (4d, d) and mlp.c_proj (d, 4d)), keyed by
   the sorted shape -- see `muon_group_key`, which GPT.setup_optimizer uses to build the groups. As separate groups
   they leave 40 of 64 ranks idle each. Members are viewed in one canonical (rows >= cols) orientation on the way in
   and transposed back on the way out (`_orient`); that is exact because Polar Express is transpose-covariant and
   NorMuon reduces over the longer axis either way.
3. Launch fusion: the optimizer stage on trn3 is bound by the NUMBER of device launches (270 per step: 39 collectives,
   22 compiled kernels, 209 copies), each collective launch paying a runtime enc_barrier and, at 64 ranks, a
   cc_exec_barrier waiting for the slowest host (6-9 ms). torch.compile(backend="neuron") graphs may contain
   _c10d_functional collectives, and those are bit-identical to the eager torch.distributed calls. So one step is:
     a. adamw_reduce   ONE graph: 14 reduce_scatter(avg) into persistent grad slices + 5 all_reduce(avg)
     b. muon_pack_rs   ONE graph: orient (+bf16->fp32), stack and reduce_scatter(avg) EVERY Muon group
     c. muon_step_fused per group (3 kernels): the unchanged Muon update on this rank's owned chunk. It stays in
        its own graph because its bf16 Polar Express is sensitive to the compiler's matmul choices inside a
        bigger graph (fusing everything into 1-2 graphs is 10-27 % slower and drifts the masters).
     d. finish_all     ONE graph: the AdamW updates (this rank's slice of each large param, then all_gather into
        the param; the small replicated params in place) + every Muon group's fp32->bf16 staging + all_gather +
        copy-back with the transposes. The rank enters as a (1,) device tensor so all ranks share one NEFF.
   6 launches, 3 collective NEFFs, 2 kernel<->collective switches: 99.6 -> 61.8 ms at 32 ranks, bitwise
   identical to the eager schedule (701/701 and 2259/2259 state tensors).

State layout (keys, shapes, dtypes, rank ownership) is unchanged from the eager Trainium optimizer, so existing
checkpoints load; `load_state_dict` keeps the fp32 states fp32 (torch would downcast them to the bf16 param dtype).
"""
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as _fc
from torch import Tensor

from nanochat.common import COMPUTE_DTYPE
from nanochat.optim import polar_express_coeffs


def _compile(fn):
    # torch.compile's default "inductor" backend has no Neuron lowering; every optimizer graph compiles to one NEFF.
    return torch.compile(fn, backend="neuron", dynamic=False, fullgraph=True)


def _zeros(*shape, dtype, device):
    """torch.zeros(..., device=) with the fill done by zero_(): TorchNeuron keys eager NEFFs by the op's
    arguments INCLUDING the device kwarg, so torch.zeros(..., device="neuron:k") compiles one NEFF per rank
    (~3 s each, all at step 0) while zero_() keys on shape/dtype only and compiles once per host. Same bytes."""
    return torch.empty(*shape, dtype=dtype, device=device).zero_()


@_compile
def muon_step_fused(
    stacked_grads: Tensor,          # (K, rows, cols) fp32 - this rank's reduced gradient chunk
    stacked_params: Tensor,         # (K, rows, cols) fp32 - this rank's fp32 master chunk (updated in place)
    momentum_buffer: Tensor,        # (K, rows, cols) fp32
    second_momentum_buffer: Tensor, # (K, rows, 1) or (K, 1, cols) fp32 - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """nanochat.optim.muon_step_fused (momentum -> MuonEq -> polar_express -> Muon+ -> variance_reduction ->
    cautious_update) with two Trainium adaptations: the momentum EMA runs in the BUFFER's dtype (fp32 here even if
    the grads arrive bf16; `.to` of the same dtype is a no-op, so with fp32 grads this is exactly upstream's in-place
    form), and a chunk of ONE matrix is squeezed to 2-D for the Polar Express matmuls (batch-1 3-D bmm fails
    neuronx-cc's prologue merge; the bmm/mm results are the same). Everything else is upstream's math."""
    mb_dtype = momentum_buffer.dtype
    momentum = momentum_t.to(mb_dtype)
    g = stacked_grads.to(mb_dtype)
    momentum_buffer.lerp_(g, 1 - momentum)
    g = g.lerp_(momentum_buffer, momentum)

    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    # MuonEq row equilibration: rescale each row to the mean row norm so the spectrum entering orthogonalization is better conditioned
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)

    # Polar Express orthogonalization: replace each update with the nearest orthogonal matrix
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    _squeezed = X.dim() == 3 and X.size(0) == 1
    if _squeezed:
        X = X.squeeze(0)
    if X.size(-2) > X.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    if _squeezed:
        X = X.unsqueeze(0)
    # Cast back to the master dtype (fp32), as upstream casts to the param dtype
    g = X.to(stacked_params.dtype)

    # Muon+ renormalization: snap Frobenius norm to sqrt(min(m, n))
    target_norm = min(g.size(-2), g.size(-1)) ** 0.5
    current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    g = g * (target_norm / current_norm).to(g.dtype)

    # Variance reduction (NorMuon); the factored second moment is fp32 so tiny EMA values do not flush to 0
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update (in the fp32 master)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


def _adamw_math(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    """nanochat.optim.adamw_step_fused's update as a plain function, so the finish graph applies it to every AdamW
    parameter in one compiled graph. Same math as upstream: computed in fp32 (wte / value_embeds are bf16) and
    written back to the param and state dtypes."""
    p32 = p.float()
    exp_avg32 = exp_avg.float()
    exp_avg_sq32 = exp_avg_sq.float()
    grad32 = grad.float()
    p32.mul_(1 - lr_t * wd_t)
    exp_avg32.lerp_(grad32, 1 - beta1_t)
    exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq32 / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    # no tensor alpha: torch_neuronx cannot lower add_(..., alpha=<0-D tensor>)
    p32.sub_(exp_avg32 / denom * step_size)
    p.copy_(p32)
    exp_avg.copy_(exp_avg32)
    exp_avg_sq.copy_(exp_avg_sq32)


# -----------------------------------------------------------------------------
# The three collective graphs of one step

def _make_adamw_reduce_graph(pg):
    """reduce_scatter(avg) of every large AdamW grad into its persistent slice buffer + all_reduce(avg) of every
    small grad in place."""
    def adamw_reduce(slices, grads, small_grads):
        for s, g in zip(slices, grads):
            s.copy_(_fc.reduce_scatter_tensor(g, "avg", 0, pg))
        for g in small_grads:
            g.copy_(_fc.all_reduce(g, "avg", pg))
    return _compile(adamw_reduce)


def _make_muon_pack_rs_graph(pg, specs):
    """Orient every grad into its group's canonical shape, cast bf16->fp32, stack, and reduce_scatter(avg) into
    each group's grad_chunk, for EVERY Muon group. `specs` = per group (orientation flags, pad rows) in call order;
    the grads arrive as one flat list. Padding rows are never consumed (the reduce is row-isolated and their
    owners own no param), so they reuse the first grads instead of a zero fill."""
    def muon_pack_rs(grad_chunks, grads):
        off = 0
        for gc, (flags, pad) in zip(grad_chunks, specs):
            gs = grads[off:off + len(flags)]
            off += len(flags)
            rows = [(g if f else g.mT).to(torch.float32) for g, f in zip(gs, flags)]
            if pad:
                rows = rows + [rows[i % len(rows)] for i in range(pad)]
            gc.copy_(_fc.reduce_scatter_tensor(torch.stack(rows), "avg", 0, pg))
    return _compile(muon_pack_rs)


def _make_finish_graph(pg, world_size, group_flags):
    """The AdamW update of this rank's slice of every large param + all_gather of the updated slices into the
    params, the small replicated params' updates, then every Muon group's fp32->bf16 staging + all_gather +
    copy-back. `group_flags` = per group the orientation flags; the Muon params arrive as one flat list.

    This rank's slice of each large AdamW param is selected with a (1,) int64 device tensor holding the rank, so
    the HLO (and the NEFF) is the same on every rank; with a Python-int slice baked in, every rank compiled its own
    1.8 MB NEFF (36 s each, 64x at 64 ranks). The update math and the all_gather input bytes are unchanged (the
    slice is a copy instead of a view of p; p is fully overwritten by the all_gather either way)."""
    def finish_all(params, slices, exp_avgs, exp_avg_sqs, hps,
                   small_params, small_grads, small_eas, small_esqs, small_hps, stageds, mparams, rank_t):
        for p, s, ea, esq, h in zip(params, slices, exp_avgs, exp_avg_sqs, hps):
            rs = p.shape[0] // world_size
            ps = torch.index_select(p.view(world_size, rs, *p.shape[1:]), 0, rank_t).squeeze(0)
            _adamw_math(ps, s, ea, esq, *h)
            p.copy_(_fc.all_gather_tensor(ps, 0, pg))
        for p, g, ea, esq, h in zip(small_params, small_grads, small_eas, small_esqs, small_hps):
            _adamw_math(p, g, ea, esq, *h)
        off = 0
        for staged, flags in zip(stageds, group_flags):
            ps = mparams[off:off + len(flags)]
            off += len(flags)
            gathered = _fc.all_gather_tensor(staged.to(torch.bfloat16), 0, pg)
            for i, (p, f) in enumerate(zip(ps, flags)):
                p.copy_(gathered[i] if f else gathered[i].mT)
    return _compile(finish_all)


# -----------------------------------------------------------------------------

class DistMuonAdamW(torch.optim.Optimizer):
    """Distributed Muon + AdamW on Trainium (see the module docstring). Drop-in for nanochat.optim.DistMuonAdamW:
    same param_groups contract ('kind' adamw|muon with the same hyperparameter keys), same state_dict layout.

    Muon groups must be built with `muon_group_key` (GPT.setup_optimizer does), so a matrix and its transpose share a
    group; every member of a group must have the canonical shape or its transpose.
    """

    # How GPT.setup_optimizer groups the matrix params for Muon: by SORTED shape, so (4d, d) and (d, 4d) matrices
    # share a group (upstream's exact-shape key would be `tuple(p.shape)`). The order of the members is the model's
    # parameter order, which is what checkpoints and rank ownership depend on -- hence a key the model applies rather
    # than a re-grouping here.
    muon_group_key = staticmethod(lambda p: tuple(sorted(p.shape)))

    # Optimizer state that must keep its SAVED dtype even though the params are bf16 (see load_state_dict).
    _FP32_STATE_KEYS = ("fp32_master", "second_momentum_buffer", "momentum_buffer")

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        for group in self.param_groups:
            if group['kind'] not in ('adamw', 'muon'):
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")
        # Persistent 0-D CPU hyperparameter tensors: the Muon kernel takes one set (groups run one after another);
        # the AdamW updates all sit in ONE graph, so each param gets its own six (see _adamw_hparams).
        for name in ("_muon_momentum_t", "_muon_lr_t", "_muon_wd_t", "_muon_beta2_t"):
            setattr(self, name, torch.tensor(0.0, dtype=torch.float32, device="cpu"))
        self._adamw_hp = {}
        # Persistent device buffers, kept on self (NOT in param_groups / state) so state_dict() does not serialize
        # them: AdamW grad slices per param, the fp32 grad_chunk per Muon group (every group's chunk is live from the
        # pack graph to the finish graph: ~100 MB in total at 32 ranks, no stacked slab anywhere), the compiled
        # graphs, and the rank tensor.
        self._adamw_bufs = {}
        self._muon_bufs = {}
        self._graphs = {}
        self._rank_t = None

    # ---- helpers

    @staticmethod
    def _orient(t: Tensor, shape) -> Tensor:
        """View `t` in the group's canonical `shape` (rows >= cols), transposing it if it is stored the other way.
        A view (`.mT`), never a copy."""
        return t if tuple(t.shape) == tuple(shape) else t.mT

    @staticmethod
    def _canonical_shape(group: dict):
        return tuple(sorted(group['params'][0].shape, reverse=True))

    def _fill_muon_hparams(self, group: dict, shape) -> None:
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

    def _adamw_hparams(self, p, group: dict, step: int):
        """Per-PARAM persistent 0-D CPU tensors. Per param rather than per group on purpose: with one tuple shared
        by the 12 value embeddings dynamo deduplicates the repeated inputs and neuronx-cc then compiles the shared
        `1 - lr*wd` / `lr / bias1` scalars differently from the single-param kernel -- those params came out 1 bf16
        ulp off in 22 % of their elements. Distinct 0-D inputs per param reproduce the eager kernel's graph structure
        per tensor and are bitwise."""
        hp = self._adamw_hp.get(id(p))
        if hp is None:
            hp = tuple(torch.tensor(0.0, dtype=torch.float32, device="cpu") for _ in range(6))
            self._adamw_hp[id(p)] = hp
        step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t = hp
        step_t.fill_(step); lr_t.fill_(group['lr']); beta1_t.fill_(group['betas'][0])
        beta2_t.fill_(group['betas'][1]); eps_t.fill_(group['eps']); wd_t.fill_(group['weight_decay'])
        return hp

    def _grad_chunk(self, group, chunk_size, shape, device):
        """This group's persistent fp32 reduce_scatter output (chunk_size x shape), grown on demand. fp32 because it
        carries the world-size-way gradient AVERAGE: averaging in bf16 rounds at every tree level (measured 4.4e-3
        relative error, enough for an eval regression)."""
        bufs = self._muon_bufs.setdefault(id(group), {})
        need = chunk_size * shape[0] * shape[1]
        buf = bufs.get("grad_chunk")
        if buf is None or buf.numel() < need:
            buf = _zeros(need, dtype=torch.float32, device=device)
            bufs["grad_chunk"] = buf
        return buf.flatten()[:need].view(chunk_size, *shape)

    def _owned_fp32_master(self, state, owned_params, chunk_size, num_owned, shape, device):
        """The fp32 master for the Muon params THIS rank owns. Seeded from the current (bf16) params ONCE; from
        then on it is authoritative and never re-read from p.data. Every param has exactly one owner, so exactly one
        fp32 master, never rounded. (Linear.forward casts weights to the activation dtype for every matmul, so
        fwd/bwd never read the master at full precision anyway.)"""
        fp32_master = state.get("fp32_master")
        if fp32_master is None or fp32_master.shape[0] < chunk_size:
            fp32_master = torch.empty(chunk_size, *shape, dtype=torch.float32, device=device)
            for i in range(num_owned):
                fp32_master[i].copy_(self._orient(owned_params[i].detach(), shape))
            if num_owned < chunk_size:
                fp32_master[num_owned:].zero_()
            state["fp32_master"] = fp32_master
        # muon_step_fused updates its `stacked_params` argument IN PLACE: the update lands in fp32 and persists.
        return fp32_master[:num_owned]

    @staticmethod
    def _pg():
        return dist.distributed_c10d._get_default_group()

    # ---- the step

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        adamw_groups = [g for g in self.param_groups if g['kind'] == 'adamw']
        muon_groups = [g for g in self.param_groups if g['kind'] == 'muon']
        # SFT batches with all-ignore targets can leave p.grad None; zero-fill so every collective runs on every
        # rank (bitwise the explicit-zero-grad step).
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

        # a. AdamW reduce graph (33 inputs, cheap host prep, launched first)
        ainfo = self._reduce_adamw(adamw_groups, world_size)
        # b. Muon pack + reduce_scatter graph for every group (its ~160-input host prep hides under the reduce NEFF)
        minfos = [(g, self._muon_pack_args(g, world_size)) for g in muon_groups]
        if minfos:
            key = ("muon_pack_rs",) + tuple(id(g) for g, _ in minfos)
            fn = self._graphs.get(key)
            if fn is None:
                fn = self._graphs[key] = _make_muon_pack_rs_graph(self._pg(), tuple((i['flags'], i['pad']) for _, i in minfos))
            fn([i['grad_chunk'] for _, i in minfos], [g for _, i in minfos for g in i['grads']])
            for g, i in minfos:
                i['grads'] = None
                for q in g['params']:
                    q.grad = None
        # c. the Muon kernels, one per group, on this rank's owned chunk
        for g, info in minfos:
            self._compute_muon(g, info, rank)
        # d. finish graph: AdamW updates + all_gathers, then every Muon group's staging + all_gather + copy-back
        aargs = self._compute_adamw(adamw_groups, ainfo, rank, world_size)
        key = ("finish",) + tuple(id(g) for g, _ in minfos)
        fn = self._graphs.get(key)
        if fn is None:
            flags = tuple(i['flags'] for _, i in minfos)
            fn = self._graphs[key] = _make_finish_graph(self._pg(), world_size, flags)
        if self._rank_t is None:
            dev = next(q.device for g in self.param_groups for q in g['params'])
            self._rank_t = torch.tensor([rank], dtype=torch.int64, device=dev)
        fn(*aargs, [i['staged'] for _, i in minfos], [q for g, _ in minfos for q in g['params']], self._rank_t)

    def _reduce_adamw(self, groups, world_size: int) -> dict:
        """All AdamW groups at once: small params (< 1024 elements) are all_reduced and updated on every rank;
        large params are reduce_scattered along dim 0 into a persistent per-param slice (shape[0] must divide by
        world_size), updated as a slice, and all_gathered in the finish graph."""
        param_infos = {}
        slices, grads, small_grads = [], [], []
        for group in groups:
            per_param_bufs = self._adamw_bufs.setdefault(id(group), {})
            for p in group['params']:
                grad = p.grad
                if p.numel() < 1024:
                    small_grads.append(grad)
                    param_infos[p] = dict(grad_slice=grad, is_small=True)
                else:
                    assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                    rank_size = grad.shape[0] // world_size
                    grad_slice = per_param_bufs.get(id(p))
                    if grad_slice is None or grad_slice.shape != grad[:rank_size].shape:
                        grad_slice = torch.empty_like(grad[:rank_size])
                        per_param_bufs[id(p)] = grad_slice
                    slices.append(grad_slice); grads.append(grad)
                    param_infos[p] = dict(grad_slice=grad_slice, is_small=False)
        fn = self._graphs.get("adamw_reduce")
        if fn is None:
            fn = self._graphs["adamw_reduce"] = _make_adamw_reduce_graph(self._pg())
        fn(slices, grads, small_grads)
        return param_infos

    def _compute_adamw(self, groups, param_infos: dict, rank: int, world_size: int):
        """Per-param state bookkeeping; returns the AdamW arguments of the finish graph."""
        params, slices, eas, esqs, hps = [], [], [], [], []
        sparams, sgrads, seas, sesqs, shps = [], [], [], [], []
        for group in groups:
            for p in group['params']:
                pinfo = param_infos[p]
                grad_slice = pinfo['grad_slice']
                state = self.state[p]
                if pinfo['is_small']:
                    p_slice = p
                else:
                    rank_size = p.shape[0] // world_size
                    p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                state['step'] += 1
                h = self._adamw_hparams(p, group, state['step'])
                if pinfo['is_small']:
                    sparams.append(p); sgrads.append(grad_slice); seas.append(state['exp_avg']); sesqs.append(state['exp_avg_sq']); shps.append(h)
                else:
                    params.append(p); slices.append(grad_slice); eas.append(state['exp_avg']); esqs.append(state['exp_avg_sq']); hps.append(h)
        return (params, slices, eas, esqs, hps, sparams, sgrads, seas, sesqs, shps)

    def _muon_pack_args(self, group: dict, world_size: int) -> dict:
        """Everything the pack+reduce_scatter of a Muon group needs (buffers, orientation flags, padding), no
        launch. K params are stacked and divided across the ranks: each rank owns ceil(K/N) consecutive members;
        K is zero-padded to ceil(K/N) * N for the collective and the padding is never read back."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        shape = self._canonical_shape(group)
        grad_chunk = self._grad_chunk(group, chunk_size, shape, params[0].device)
        flags = tuple(tuple(q.shape) == shape for q in params)
        return dict(grad_chunk=grad_chunk, chunk_size=chunk_size, flags=flags,
                    pad=padded_num_params - len(params), grads=[q.grad for q in params])

    def _compute_muon(self, group: dict, info: dict, rank: int) -> None:
        """muon_step_fused on this rank's owned chunk (fp32 master updated in place); the staging + all_gather +
        copy-back happen in the finish graph, which reads info['staged']."""
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        device = p.device
        shape = self._canonical_shape(group)
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        # Group-level state lives in the first param's state (as upstream).
        state = self.state[p]
        if "momentum_buffer" not in state:
            # fp32 like the reduce slab: the Nesterov EMA is the one Muon buffer that accumulates across the run.
            state["momentum_buffer"] = _zeros(chunk_size, *shape, dtype=torch.float32, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = _zeros(state_shape, dtype=torch.float32, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = self._owned_fp32_master(state, owned_params, chunk_size, num_owned, shape, device)
            self._fill_muon_hparams(group, shape)
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            staged = state["fp32_master"][:chunk_size]
        else:
            # This rank owns only padding rows: gather its grad_chunk so the collective stays alive on every rank
            # (the rows land in padding positions nobody reads).
            staged = grad_chunk
        info['staged'] = staged

    # ---- checkpointing

    def load_state_dict(self, state_dict) -> None:
        """Restore optimizer state, keeping the fp32-only states fp32.

        torch's Optimizer.load_state_dict casts every floating-point state tensor to `param.dtype`; with bf16 params
        that would silently downcast fp32_master (discarding exactly the precision it exists to hold), the momentum
        EMA and the factored second moment. Stash those, let the base class load the rest, then reinstate them at
        their saved dtype.
        """
        sd_state = state_dict.get("state", {})
        stash = {}
        for pid, pstate in sd_state.items():
            if not isinstance(pstate, dict):
                continue
            keep = {k: v for k, v in pstate.items()
                    if k in self._FP32_STATE_KEYS and isinstance(v, torch.Tensor)}
            if keep:
                stash[pid] = keep
        if not stash:
            return super().load_state_dict(state_dict)
        # Shallow-copy down to the per-param dicts so the caller's state_dict is untouched.
        state_dict = dict(state_dict)
        state_dict["state"] = {
            pid: ({k: v for k, v in pstate.items() if k not in stash.get(pid, ())}
                  if isinstance(pstate, dict) else pstate)
            for pid, pstate in sd_state.items()
        }
        # Same saved-id -> live-param mapping torch builds internally.
        id_map = dict(zip(
            [pid for g in state_dict["param_groups"] for pid in g["params"]],
            [p for g in self.param_groups for p in g["params"]],
        ))
        super().load_state_dict(state_dict)
        for pid, keep in stash.items():
            p = id_map.get(pid)
            if p is None:
                continue
            for k, v in keep.items():
                self.state[p][k] = v.to(device=p.device, dtype=v.dtype)
