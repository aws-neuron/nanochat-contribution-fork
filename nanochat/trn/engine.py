"""Inference engine for Trainium: nanochat.engine.Engine with static decode shapes and batched problem decode.

TorchNeuron eager mode compiles every distinct (op, shape) through neuronx-cc (100-120 s per new forward shape) and runs a
decode step as several hundred single-op launches, so the upstream Engine -- a fresh KVCache per call, a prefill of
exactly len(prompt) tokens, a decode cache of exactly len(prompt) + max_tokens -- compiles a new graph for nearly
every problem of chat_eval. NeuronEngine keeps the generation math and instead pins the shapes:

  * prefill length and decode-cache length are rounded up to PREFILL_BUCKETS (BOS padding on the right; the padded
    positions are never attended because Trainium's flash_attn_with_kvcache decodes over the whole pre-allocated
    (B, T_max, H, D) cache with a position mask -- nanochat/trn/attention.py). One prefill NEFF per bucket, one decode
    NEFF per (rows, bucket) for the whole run instead of one per prompt length.
  * KV caches are pooled per (role, B, T) and reset between problems: on Neuron every allocation is a dispatched
    device op, and the bucketed key set is small (upstream frees per call; on CUDA/CPU every distinct prompt length
    would pin a cache forever, so pooling is Neuron-only).
  * `generate_problems_batched` decodes DECODE_BATCH independent greedy problems as ONE (B, 1) decode loop: each
    prompt is prefilled on its own into row b of a shared decode cache at its own position (the cache is `ragged`:
    GPT.forward gathers per-row rotary from cache_seqlens). Decode is memory-bound, so B rows cost about what one row
    costs. Per-row calculator tool use is preserved (each row runs its own state machine). scripts/chat_eval.py
    uses it for the generative tasks (GSM8K / HumanEval: temperature 0, one sample).

Measured on full speedruns: the 128-rank chat_eval stage in 16 min with these shapes; identical scores to the
per-problem path.
"""
import torch
import torch.distributed as dist

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import Engine, KVCache, RowState, sample_next_token, use_calculator

# Prefill / decode-cache lengths are rounded up to one of these (the last is the model's sequence_len).
PREFILL_BUCKETS = (64, 128, 256, 512, 1024, 2048)
# Problems decoded concurrently per rank by generate_problems_batched (chat_eval).
DECODE_BATCH = 16
# The bucketed caches below rely on the static-shape decode of nanochat/trn/attention.py (the whole pre-allocated
# cache with a position mask, the only Neuron decode; a slice-based decode would attend to the BOS padding).


def _bucket(length, max_len):
    for b in PREFILL_BUCKETS:
        if length <= b <= max_len:
            return b
    return length  # no bucket big enough: exact length


class NeuronEngine(Engine):
    decode_batch = DECODE_BATCH

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        self._kv_cache_pool = {}  # (role, B, T) -> KVCache
        # Tool-use special ids, resolved once (the state machine runs per token per row).
        get_special = tokenizer.encode_special
        self._python_start = get_special("<|python_start|>")
        self._python_end = get_special("<|python_end|>")
        self._output_start = get_special("<|output_start|>")
        self._output_end = get_special("<|output_end|>")
        self._assistant_end = get_special("<|assistant_end|>")   # if sampled, ends a row
        self._bos = tokenizer.get_bos_token_id()                 # if sampled, ends a row

    def _kv_kwargs(self):
        m = self.model.config
        return dict(num_heads=m.n_kv_head, head_dim=m.n_embd // m.n_head, num_layers=m.n_layer,
                    device=self.model.get_device(), dtype=COMPUTE_DTYPE)

    def _get_kv(self, role, batch_size, seq_len):
        """A pooled KVCache with cache_seqlens reset. `role` separates prefill / decode caches that may share
        (B, T): one generate() call uses both and must not alias them."""
        key = (role, batch_size, seq_len)
        cache = self._kv_cache_pool.get(key)
        if cache is None:
            cache = self._kv_cache_pool[key] = KVCache(batch_size=batch_size, seq_len=seq_len, **self._kv_kwargs())
        else:
            cache.reset()
        return cache

    def _prefill(self, prompt, max_len):
        """Prefill one prompt padded to its bucket; returns (kv cache, logits at the real last prompt position,
        prompt length). The cache's cache_seqlens still reads the padded length: rows [prompt_len, prefill_len) hold
        the BOS padding's K/V, which the callers never copy (batched decode) or rewind past (generate)."""
        prompt_len = len(prompt)
        prefill_len = _bucket(prompt_len, max_len)
        kv = self._get_kv("prefill", 1, prefill_len)
        padded = prompt + [self._bos] * (prefill_len - prompt_len)
        ids = torch.tensor([padded], dtype=torch.long, device=kv.cache_seqlens.device)
        logits_full = self.model.forward(ids, kv_cache=kv)
        return kv, logits_full[:, prompt_len - 1, :], prompt_len

    def _advance_tool_state(self, state, next_token):
        """One row's calculator state machine (same logic as Engine.generate): <|python_start|> .. <|python_end|>
        captures an expression; its result is queued as <|output_start|> result <|output_end|> forced tokens."""
        if next_token == self._python_start:
            state.in_python_block = True
            state.python_expr_tokens = []
        elif next_token == self._python_end and state.in_python_block:
            state.in_python_block = False
            if state.python_expr_tokens:
                expr = self.tokenizer.decode(state.python_expr_tokens)
                result = use_calculator(expr)
                if result is not None:
                    state.forced_tokens.append(self._output_start)
                    state.forced_tokens.extend(self.tokenizer.encode(str(result)))
                    state.forced_tokens.append(self._output_end)
            state.python_expr_tokens = []
        elif state.in_python_block:
            state.python_expr_tokens.append(next_token)

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Engine.generate with bucketed prefill / decode caches (see the module docstring)."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        max_len = self.model.config.sequence_len
        assistant_end, bos = self._assistant_end, self._bos

        # 1) Batch-1 prefill of the prompt, padded to its bucket
        kv_cache_prefill, logits, prompt_len = self._prefill(tokens, max_len)
        logits = logits.expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row, into a bucketed decode cache (one decode shape per bucket).
        # Rewind the prefill cache to the real prompt length first so only real-token K/V is copied (an eager fill_ with
        # a Python scalar compiles once per distinct prompt length; the batched path below avoids it altogether).
        kv_cache_prefill.cache_seqlens.fill_(prompt_len)
        kv_length_hint = (prompt_len + max_tokens) if max_tokens is not None else max_len
        decode_len = max(_bucket(kv_length_hint, max_len), kv_cache_prefill.max_seq_len)
        kv_cache_decode = self._get_kv("decode", num_samples, decode_len)
        kv_cache_decode.prefill(kv_cache_prefill)

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()
            token_column, token_masks = [], []
            any_forced = False
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                any_forced = any_forced or is_forced
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                self._advance_tool_state(state, next_token)
            yield token_column, token_masks
            num_generated += 1
            # When no row was forced, the sampled ids on device are exactly the next input (no host round trip).
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1) if any_forced else next_ids
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    @torch.inference_mode()
    def evaluate_problems_batched(self, task_object, problem_indices, max_tokens):
        """scripts/chat_eval.py's generative loop (greedy, one sample) for this rank's problems, decode_batch problems
        per batched decode; per-row calculator tool use is preserved. Returns (num_passed, total)."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        indices = list(problem_indices)
        num_passed, total = 0, 0
        for bstart in range(0, len(indices), self.decode_batch):
            conversations = [task_object[i] for i in indices[bstart:bstart + self.decode_batch]]
            prompts = [self.tokenizer.render_for_completion(c) for c in conversations]
            outs = self.generate_problems_batched(prompts, max_tokens=max_tokens, temperature=0.0)
            for conversation, out in zip(conversations, outs):
                total += 1
                num_passed += int(bool(task_object.evaluate(conversation, self.tokenizer.decode(out))))
                print(f"\r\033[KRank {rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)
        return num_passed, total

    def generate_problems_batched(self, prompts, max_tokens, temperature=0.0):
        """Decode many independent problems concurrently in one batched greedy decode loop.

        Each prompt (list[int]) is prefilled on its own (bucketed), its KV copied into row b of a shared (B, T_max)
        decode cache at its own position, then ONE batched decode loop runs over all B rows. Returns list[list[int]]
        of generated token ids (prompt stripped), one per prompt, in input order.
        """
        assert temperature == 0.0, "batched decode supports greedy (temperature=0) only"
        B = len(prompts)
        max_len = self.model.config.sequence_len
        assistant_end, bos = self._assistant_end, self._bos
        device = self.model.get_device()

        # _bucket is monotone in length, so bucketing (max_prompt + max_tokens) already covers max_prompt alone.
        decode_len = _bucket(max(len(p) for p in prompts) + max_tokens, max_len)
        kv_decode = self._get_kv("batchdecode", B, decode_len)
        kv_decode.ragged = True
        prev_embed_rows = [None] * B
        logits0 = None

        # Prefill each prompt separately into row b of the batched decode cache.
        for b, prompt in enumerate(prompts):
            kv_prefill, logits, prompt_len = self._prefill(prompt, max_len)
            kv_decode.k_cache[:, b:b+1, :prompt_len, :, :] = kv_prefill.k_cache[:, :, :prompt_len, :, :]
            kv_decode.v_cache[:, b:b+1, :prompt_len, :, :] = kv_prefill.v_cache[:, :, :prompt_len, :, :]
            kv_decode.cache_seqlens[b] = prompt_len
            if kv_prefill.prev_embedding is not None:
                prev_embed_rows[b] = kv_prefill.prev_embedding[:, -1:, :]  # (1, 1, C) smear state at the real last position
            if logits0 is None:
                logits0 = torch.empty(B, logits.size(-1), device=device, dtype=logits.dtype)
            logits0[b] = logits[0]
        kv_decode.prev_embedding = torch.cat(prev_embed_rows, dim=0) if all(e is not None for e in prev_embed_rows) else None

        # Batched decode loop; each row keeps its own tool-use state.
        row_states = [RowState() for _ in range(B)]
        cur_logits = logits0  # (B, V)
        num_generated = 0
        while num_generated < max_tokens and not all(s.completed for s in row_states):
            sampled = sample_next_token(cur_logits, None, temperature)  # (B, 1); greedy never touches the generator
            sampled_toks = sampled[:, 0].tolist()
            token_column = []
            any_forced = False
            for b, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                any_forced = any_forced or is_forced
                next_token = state.forced_tokens.popleft() if is_forced else sampled_toks[b]
                token_column.append(next_token)
                if not state.completed:
                    if next_token == assistant_end or next_token == bos:
                        state.completed = True
                    else:
                        state.current_tokens.append(next_token)
                self._advance_tool_state(state, next_token)
            num_generated += 1
            if all(s.completed for s in row_states):
                break
            next_ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1) if any_forced else sampled
            cur_logits = self.model.forward(next_ids, kv_cache=kv_decode)[:, -1, :]
        return [s.current_tokens for s in row_states]
