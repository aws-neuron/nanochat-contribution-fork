"""CORE evaluation on Trainium: fixed-shape batched scoring (scripts/base_eval.py uses this on Neuron).

Upstream nanochat.core_eval.evaluate_task forwards one example at a time (B = its number of choices) and pads every
row to the longest row of that example. On Neuron the forward runs op-by-op in eager mode at a flat ~28 ktok/s per
rank, and every distinct batch shape compiles its own kernels (~4 min per forward shape with a cold cache), so that
path took 44 min at 16 ranks for the 22 tasks. Here every rank
renders and tokenizes its examples up front, sorts the rows into two sequence-length BUCKETS (1024 and the model's
sequence_len 2048), and forwards fixed batches of TOKENS_PER_BATCH = 16384 tokens per bucket ((16, 1024) and (8, 2048));
dummy BOS rows fill the tail of a bucket. Causal attention makes real rows independent of what follows them, so the
per-row math is the upstream math: MC / schema rows are scored from the model's per-position cross-entropy
(loss_reduction='none'), LM rows from the fp32 logits at the continuation positions with the argmax on the CPU (same
tie-breaking). Per-row losses and logits were verified BITWISE identical to the per-example path for these two
buckets (a 512 bucket is NOT: at T=512 the sliding-window layers take a different attention kernel), and the full run's
CSV is byte-identical. 18.4 min instead of 44.2 at 16 ranks.

The whole benchmark then needs 2 heads x 2 buckets compiled forward shapes, which `warmup_core_shapes` compiles up
front with the ranks working on DIFFERENT shapes (eager Neuron compiles are deduplicated across the ranks of a host
through file locks, so ranks meeting a new shape at the same moment serialize). Per-task results are reduced as a
fixed-shape (1,) count instead of upstream's (len(data),) vector + mean, which compiled a kernel per task.

Rows longer than sequence_len are truncated to their last sequence_len tokens (upstream truncates only for models
that declare `max_seq_len`; a fixed shape needs a bound).
"""
import random
import time

import torch
import torch.distributed as dist

from nanochat.core_eval import (render_prompts_mc, render_prompts_schema, render_prompts_lm,
                                batch_sequences_mc, batch_sequences_schema, batch_sequences_lm)

TOKENS_PER_BATCH = 16384      # rows per batch = TOKENS_PER_BATCH // bucket: (16, 1024) and (8, 2048)
BUCKETS = (1024, 2048)        # the model's sequence_len is always the last bucket
LM_GATHER = 64                # continuation positions gathered per LM row (>= the longest CORE continuation, 49)


def _max_tokens(model):
    return getattr(model, 'max_seq_len', None) or model.config.sequence_len


def _buckets(max_tokens):
    return sorted({b for b in BUCKETS if b < max_tokens} | {max_tokens})


def _rows_for_bucket(bucket):
    return max(1, TOKENS_PER_BATCH // bucket)


def _bucket_for(length, buckets):
    for b in buckets:
        if length <= b:
            return b
    return buckets[-1]


def _lm_gather_len(max_cont):
    # A longer continuation than LM_GATHER widens the gather for that task (rounded to 8).
    return max(LM_GATHER, ((max_cont + 7) // 8) * 8)


def prepare_example(idx, tokenizer, data, task_meta, max_tokens):
    """Render + tokenize one example exactly as nanochat.core_eval.evaluate_example does (same few-shot sampling,
    same batching helpers), truncated to max_tokens. Returns (tokens, start_idxs, end_idxs, gold)."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    new_tokens, new_start_idxs, new_end_idxs = [], [], []
    for t, s, e in zip(tokens, start_idxs, end_idxs):
        if len(t) > max_tokens:
            num_to_crop = len(t) - max_tokens
            new_tokens.append(t[-max_tokens:])  # take the last max_tokens tokens
            new_start_idxs.append(s - num_to_crop)  # shift the indices down
            new_end_idxs.append(e - num_to_crop)
            assert s - num_to_crop >= 0, "this should never happen right?"
            assert e - num_to_crop >= 0, "this should never happen right?"
        else:
            new_tokens.append(t)
            new_start_idxs.append(s)
            new_end_idxs.append(e)
    return new_tokens, new_start_idxs, new_end_idxs, item.get('gold', None)


def _shifted_targets(input_ids):
    """targets[:, t] = input_ids[:, t+1]; the last column is a dummy (0) whose loss is never read."""
    targets = torch.zeros_like(input_ids)
    targets[:, :-1] = input_ids[:, 1:]
    return targets


@torch.no_grad()
def _batch_losses(model, input_ids, targets):
    """Per-position CE losses (R, T) fp32 on CPU: losses[:, t] scores input_ids[:, t+1] from position t (t < T-1)."""
    return model(input_ids, targets, loss_reduction='none').view(input_ids.shape).cpu()


@torch.no_grad()
def _batch_logits_at(model, input_ids, positions):
    """fp32 logits (R, L, V) on CPU at the given positions (R, L) of each row (padded with 0): one fixed-shape
    index_select per (R, T, L) instead of a variable-length slice per example."""
    logits = model(input_ids)  # (R, T, V) fp32 capped logits on device
    R, T, V = logits.shape
    L = positions.size(1)
    flat = (positions + torch.arange(R, dtype=torch.long).unsqueeze(1) * T).reshape(-1).to(logits.device)
    sel = logits.reshape(R * T, V).index_select(0, flat)
    return sel.cpu().view(R, L, V)


def _run_batch(model, task_type, input_ids, chunk, device, lm_pad):
    """Forward one fixed-shape batch; per-row scores for `chunk` (rows are (ex_i, j, tokens, si, ei)): mean
    continuation loss for MC / schema, correctness for LM."""
    ids_dev = input_ids.to(device)
    out = []
    if task_type in ('multiple_choice', 'schema'):
        losses = _batch_losses(model, ids_dev, _shifted_targets(input_ids).to(device))
        for i, (_, _, _, si, ei) in enumerate(chunk):
            out.append(losses[i, si-1:ei-1].mean().item())
    elif task_type == 'language_modeling':
        R = input_ids.size(0)
        L = max(lm_pad, max(ei - si for (_, _, _, si, ei) in chunk))
        positions = torch.zeros((R, L), dtype=torch.long)
        for i, (_, _, _, si, ei) in enumerate(chunk):
            positions[i, :ei-si] = torch.arange(si-1, ei-1)
        sel = _batch_logits_at(model, ids_dev, positions)  # (R, L, V) fp32 on CPU
        for i, (_, _, _, si, ei) in enumerate(chunk):
            predictions = sel[i, :ei-si].argmax(dim=-1)
            actual = input_ids[i, si:ei]
            out.append(bool((predictions == actual).all().item()))
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    return out


@torch.no_grad()
def evaluate_task(model, tokenizer, data, device, task_meta):
    """Drop-in for nanochat.core_eval.evaluate_task: this rank's examples in fixed-shape batches, then the
    all-rank accuracy."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    task_type = task_meta['task_type']
    max_tokens = _max_tokens(model)
    buckets = _buckets(max_tokens)
    pad_token_id = tokenizer.get_bos_token_id()

    # 1. Host: render + tokenize every example of this rank up front.
    prepared = [prepare_example(idx, tokenizer, data, task_meta, max_tokens) for idx in range(rank, len(data), world_size)]
    rows = []  # (ex_i, j, tokens, si, ei)
    for ex_i, (tokens, start_idxs, end_idxs, _) in enumerate(prepared):
        for j, (t, si, ei) in enumerate(zip(tokens, start_idxs, end_idxs)):
            rows.append((ex_i, j, t, si, ei))
    # LM tasks: one continuation-gather length for the whole task, agreed across ranks (normally the warm-up's).
    lm_pad = _lm_gather_len(max((ei - si for (_, _, _, si, ei) in rows), default=1))
    if task_type == 'language_modeling' and dist.is_initialized():
        lm_pad_t = torch.tensor([lm_pad], dtype=torch.int32, device=device)
        dist.all_reduce(lm_pad_t, op=dist.ReduceOp.MAX)
        lm_pad = int(lm_pad_t.item())

    # 2. Group rows by length bucket; fixed rows per batch per bucket, dummy rows fill the tail.
    by_bucket = {b: [] for b in buckets}
    for r in rows:
        by_bucket[_bucket_for(len(r[2]), buckets)].append(r)
    row_scores = {}
    for bucket in buckets:
        brows = by_bucket[bucket]
        R = _rows_for_bucket(bucket)
        for start in range(0, len(brows), R):
            chunk = brows[start:start + R]
            input_ids = torch.full((R, bucket), pad_token_id, dtype=torch.long)
            for i, (_, _, t, _, _) in enumerate(chunk):
                input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            scores = _run_batch(model, task_type, input_ids, chunk, device, lm_pad)
            for (ex_i, j, _, _, _), score in zip(chunk, scores):
                row_scores[(ex_i, j)] = score

    # 3. Score the examples exactly as upstream: LM = all continuation tokens predicted; MC / schema = lowest mean loss.
    n_correct = 0
    for ex_i, (tokens, _, _, gold) in enumerate(prepared):
        if task_type == 'language_modeling':
            n_correct += int(row_scores[(ex_i, 0)])
        else:
            mean_losses = [row_scores[(ex_i, j)] for j in range(len(tokens))]
            n_correct += int(mean_losses.index(min(mean_losses)) == gold)

    # 4. Fixed-shape reduction: the count is an exact integer in fp32, so sum / len(data) in fp32 is the same
    # correctly-rounded division upstream's device mean performs.
    n_correct_t = torch.tensor([float(n_correct)], dtype=torch.float32, device=device)
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(n_correct_t, op=dist.ReduceOp.SUM)
    return (n_correct_t.cpu() / torch.tensor([float(len(data))], dtype=torch.float32)).item()


@torch.no_grad()
def warmup_core_shapes(model, tokenizer, device, bpb_batch=None):
    """Compile the batched-eval forward shapes with the ranks working on DIFFERENT shapes: rank r first runs the
    (bucket, head) combination r % n, so the n combinations compile concurrently and the other ranks find them in
    the cache. Costs one dummy forward per rank.

    bpb_batch: optional (device_batch_size, token_bytes) -- adds the bpb step (its (B, sequence_len) loss forward
    plus the ~10 small accumulation ops of loss_eval.evaluate_bpb's loop) as one more combination, so it compiles here instead
    of inside the first timed bpb step."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    max_tokens = _max_tokens(model)
    buckets = _buckets(max_tokens)
    combos = [(b, head) for b in reversed(buckets) for head in ("loss", "logits")]
    if bpb_batch is not None:
        combos.insert(0, (bpb_batch[0], "bpb"))
    combos = combos[rank % len(combos):] + combos[:rank % len(combos)]
    pad_token_id = tokenizer.get_bos_token_id()
    t0 = time.time()
    for bucket, head in combos:
        if head == "bpb":
            # the ops of nanochat.loss_eval.evaluate_bpb's loop body (fast path: no ignored targets) on a dummy batch
            ids = torch.full((bucket, max_tokens), pad_token_id, dtype=torch.long).to(device)
            loss2d = model(ids, ids, loss_reduction='none').view(-1)
            y = ids.view(-1)
            (y.int() < 0).any()
            num_bytes2d = bpb_batch[1][y]
            total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
            total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
            continue
        R = _rows_for_bucket(bucket)
        ids = torch.full((R, bucket), pad_token_id, dtype=torch.long)
        if head == "loss":
            _batch_losses(model, ids.to(device), _shifted_targets(ids).to(device))
        else:
            _batch_logits_at(model, ids.to(device), torch.zeros((R, _lm_gather_len(1)), dtype=torch.long))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"CORE warm-up: {len(combos)} forward shapes compiled in {time.time() - t0:.1f}s "
              f"(buckets {buckets}, {TOKENS_PER_BATCH} tokens/batch)")
