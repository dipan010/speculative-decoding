# Speculative Decoding

Speculative decoding as a clean, composable inference-engine wrapper.  Inspired
by the `InferenceEngine` protocol in [apple/coreai-models](https://github.com/apple/coreai-models),
built in Python with both mock engines (for deterministic testing) and real
HuggingFace model engines (for measurable speedup).

## Quick Start

```bash
# Install
cd speculative-decoding
uv venv --python 3.12 && uv pip install -e . pytest

# Run unit tests (fast, no model downloads)
uv run pytest -m "not slow"

# Run ALL tests including real-model correctness (downloads ~4GB)
uv run pytest

# Run benchmark with K-sweep plot
uv run benchmark
```

## The Algorithm

Speculative decoding accelerates autoregressive generation by using a small,
fast **draft** model to propose K tokens, then having the large **verifier**
model check all K in a single forward pass.

### Why It's Faster

The key insight: a transformer's forward pass cost is roughly the same whether
you process 1 token or K tokens (for small K), because the computation is
dominated by the weight matrices, not the sequence length.  So instead of K
separate verifier forward passes (one per token), we do:

1. K fast draft forward passes (cheap — small model)
2. 1 verifier forward pass over all K tokens (same cost as 1 token)

If the draft model is 5x faster than the verifier and we accept most draft
tokens, we replace K verifier calls with K draft calls + 1 verifier call.

### Greedy Speculative Decoding — Step by Step

**Setup**: Draft model D, verifier model V, draft length K=5.

**Each round**:

1. **Draft**: D generates K=5 tokens autoregressively:
   ```
   Context: "The capital of France is"
   Draft proposes: [Paris, ., It, is, a]
   ```

2. **Verify**: V does ONE forward pass over the full sequence
   `[The, capital, of, France, is, Paris, ., It, is, a]`,
   producing logits at every position.

3. **Accept/Reject**: Compare draft tokens against V's greedy choices:
   ```
   Position 0: V says "Paris"  → draft="Paris"  ✓ ACCEPT
   Position 1: V says "."      → draft="."      ✓ ACCEPT
   Position 2: V says "It"     → draft="It"     ✓ ACCEPT
   Position 3: V says "is"     → draft="is"     ✓ ACCEPT
   Position 4: V says "known"  → draft="a"      ✗ REJECT
   ```

4. **Output**: Accept the matching prefix `[Paris, ., It, is]` (4 tokens),
   then take V's token at the rejection point: `known`.
   Total: **5 tokens from 1 verifier forward pass** instead of 5.

5. If all K tokens are accepted, take the **bonus token** — V's prediction
   at position K — for K+1 tokens from one forward pass.

### Correctness Proof Sketch

**Claim**: Greedy speculative decoding produces token-for-token identical output
to plain greedy decoding with the verifier alone.

**Proof**: By induction on the output position.

- **Base case**: The first output token is argmax(V(prompt)), regardless of
  what the draft proposed (if draft matched, we accept; if not, we take V's
  correction).

- **Inductive step**: Assume positions 0..n-1 are identical to V-only greedy.
  At position n, speculative decoding either:
  - **Accepts** the draft token, which means `draft_token == argmax(V)` at
    that position — identical to V-only.
  - **Rejects** and takes `argmax(V)` as the correction — identical to V-only.
  - **Bonus**: if all K match, the bonus is `argmax(V)` at position K —
    identical to V-only.

  In every case, the token at position n equals what V-only greedy would
  produce. QED.

### Latency Model

Let `t_d` = time for one draft forward pass, `t_v` = time for one verifier
forward pass, `α` = acceptance rate (fraction of draft tokens accepted).

**Verifier-only**: K tokens costs `K × t_v`.

**Speculative**: One round produces `αK + 1` tokens (accepted prefix + 
correction/bonus) and costs `K × t_d + t_v`.

**Speedup** = `(αK + 1) × t_v / (K × t_d + t_v)`

When `t_d << t_v` and `α` is high, this approaches `(αK + 1)x`.

## Architecture

```
speculative_decoding/
├── protocol.py           # InferenceEngine protocol (generate + forward)
├── mock_engine.py        # Deterministic bigram engine for unit tests
├── hf_engine.py          # HuggingFace transformers engine (Qwen2)
├── speculative_engine.py # The speculative decoding wrapper
└── benchmark.py          # Benchmark with K-sweep plot
```

### InferenceEngine Protocol

```python
class InferenceEngine(Protocol):
    vocab_size: int
    eos_token_id: int
    max_context_length: int

    def forward(self, token_ids: list[int]) -> NDArray[np.float32]:
        """(seq_len, vocab_size) logits — one forward pass."""

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> GenerationResult:
        """Greedy autoregressive generation."""
```

Any engine satisfying this protocol can be plugged in as draft or verifier.

### SpeculativeEngine

```python
spec = SpeculativeEngine(draft_engine, verifier_engine, draft_length=5)
result = spec.generate(prompt_tokens, max_tokens=100)
# result.tokens — identical to verifier.generate(prompt_tokens, max_tokens=100).tokens
# result.acceptance_rate — fraction of draft tokens accepted
# result.num_draft_rounds — number of draft/verify cycles
```

## Tests

**Correctness tests** (the centerpiece):
- Fuzz with 20 random logit tables × 4 K values = 80 configurations
- Parametric sweep: 4 vocab sizes × 4 K values with agreeing/disagreeing engines
- All assert `speculative.tokens == verifier.generate().tokens`

**Edge cases**:
- EOS mid-draft (draft/verifier hit EOS partway through K proposals)
- All-reject (draft is always wrong — every round produces only the correction)
- Context-length limit (generation stops at max_context_length)
- K=1 (degenerate: each round proposes one token)
- max_tokens=1 (single token generation)

**Real-model tests** (marked `slow`):
- Qwen2-0.5B/1.5B, multiple prompts, multiple K values
- Same identical-output assertion on real model outputs

## Benchmark

```bash
uv run benchmark
```

Produces `benchmark_results.png` with:
- **Left panel**: Wall-clock speedup vs draft length K
- **Right panel**: Acceptance rate vs draft length K

The sweet spot is typically K=3–7, depending on how well the draft model
approximates the verifier.

### Note on KV Cache

This reference implementation does **not** reuse KV cache across forward passes
(each call recomputes from scratch).  This is intentional — it keeps the code
simple and focuses on demonstrating the algorithm.  In production, KV-cache
reuse would make both draft and verifier forward passes cheaper, amplifying the
speedup.  The benchmark numbers here are conservative lower bounds; a KV-cached
implementation would show larger gains.
