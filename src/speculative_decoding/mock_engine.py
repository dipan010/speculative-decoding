"""Deterministic mock engine for unit testing.

MockEngine is parameterised by an explicit logit table so tests can control
exactly which tokens are produced and verify the speculative-decoding algorithm
without any model weights.

Design choices:
  - The logit table is a (vocab_size, vocab_size) matrix where row `t` gives
    the logits the engine returns when the *most recent* input token is `t`.
    This makes the engine a simple bigram model — deterministic and easy to
    reason about in tests.
  - Greedy decoding simply follows argmax of the bigram table.
  - `forward` applies the bigram rule at every position independently,
    returning shape (seq_len, vocab_size).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .protocol import GenerationResult


class MockEngine:
    """Deterministic bigram engine for testing.

    Args:
        logit_table: shape (vocab_size, vocab_size). Row i is the logit
            distribution returned when the current token is i.
        eos_token_id: which token id counts as end-of-sequence.
        max_context_length: hard limit on total sequence length.
    """

    def __init__(
        self,
        logit_table: NDArray[np.float32],
        eos_token_id: int = 0,
        max_context_length: int = 2048,
    ) -> None:
        assert logit_table.ndim == 2
        assert logit_table.shape[0] == logit_table.shape[1]
        self._logit_table = logit_table.astype(np.float32)
        self._eos_token_id = eos_token_id
        self._max_context_length = max_context_length

    @property
    def vocab_size(self) -> int:
        return self._logit_table.shape[0]

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    def forward(self, token_ids: list[int]) -> NDArray[np.float32]:
        """Return logits for each position based on the bigram table.

        logits[i] = logit_table[token_ids[i]]  (prediction for next token
        given current token at position i).
        """
        indices = np.array(token_ids, dtype=np.int64)
        return self._logit_table[indices]  # (seq_len, vocab_size)

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> GenerationResult:
        """Greedy autoregressive generation using the bigram table."""
        context = list(prompt_tokens)
        generated: list[int] = []

        for _ in range(max_tokens):
            if len(context) >= self._max_context_length:
                break
            # Logits for next token depend only on the last token (bigram)
            logits = self._logit_table[context[-1]]
            next_token = int(np.argmax(logits))
            if next_token == self._eos_token_id:
                break
            generated.append(next_token)
            context.append(next_token)

        return GenerationResult(tokens=generated)


def make_agreeing_engines(
    vocab_size: int = 8,
    eos_token_id: int = 0,
    max_context_length: int = 2048,
    seed: int = 42,
) -> tuple[MockEngine, MockEngine]:
    """Create a draft/verifier pair that AGREE on greedy decoding.

    Both engines use the same logit table, so argmax is identical.
    The draft just has lower-magnitude logits (simulating a weaker model
    that happens to get the same greedy answer).
    """
    rng = np.random.default_rng(seed)
    # Verifier logit table — random but with clear winners per row
    table = rng.standard_normal((vocab_size, vocab_size)).astype(np.float32)
    # Make one token per row a clear argmax winner
    for i in range(vocab_size):
        winner = int(np.argmax(table[i]))
        table[i, winner] += 5.0  # widen the gap

    # Draft: same argmax structure, but noisier / lower magnitude
    draft_table = table * 0.5 + rng.standard_normal(table.shape).astype(np.float32) * 0.1

    # Ensure argmax alignment: force draft to match verifier argmax
    for i in range(vocab_size):
        v_winner = int(np.argmax(table[i]))
        d_winner = int(np.argmax(draft_table[i]))
        if d_winner != v_winner:
            draft_table[i, v_winner] = draft_table[i].max() + 1.0

    verifier = MockEngine(table, eos_token_id=eos_token_id, max_context_length=max_context_length)
    draft = MockEngine(
        draft_table, eos_token_id=eos_token_id, max_context_length=max_context_length
    )
    return draft, verifier


def make_disagreeing_engines(
    vocab_size: int = 8,
    eos_token_id: int = 0,
    max_context_length: int = 2048,
    seed: int = 123,
) -> tuple[MockEngine, MockEngine]:
    """Create a draft/verifier pair where the draft sometimes disagrees.

    The verifier's bigram table is authoritative.  The draft has a
    *different* argmax for some rows, so speculative decoding will
    encounter rejections — exercising the correction path.
    """
    rng = np.random.default_rng(seed)
    verifier_table = rng.standard_normal((vocab_size, vocab_size)).astype(np.float32)
    for i in range(vocab_size):
        winner = int(np.argmax(verifier_table[i]))
        verifier_table[i, winner] += 5.0

    # Draft: deliberately flip some argmax winners
    draft_table = verifier_table.copy()
    flip_rows = rng.choice(vocab_size, size=max(1, vocab_size // 3), replace=False)
    for i in flip_rows:
        if i == eos_token_id:
            continue  # don't mess with EOS row
        current_winner = int(np.argmax(draft_table[i]))
        # Pick a different token as the draft's argmax
        candidates = [t for t in range(vocab_size) if t != current_winner and t != eos_token_id]
        if candidates:
            new_winner = rng.choice(candidates)
            draft_table[i, new_winner] = draft_table[i].max() + 2.0

    verifier = MockEngine(
        verifier_table, eos_token_id=eos_token_id, max_context_length=max_context_length
    )
    draft = MockEngine(
        draft_table, eos_token_id=eos_token_id, max_context_length=max_context_length
    )
    return draft, verifier
