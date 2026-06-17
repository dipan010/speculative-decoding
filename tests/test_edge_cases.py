"""Edge case tests for speculative decoding.

Covers: EOS mid-draft, verifier rejecting all K tokens, context-length limit,
draft_length=1, and single-token generation.
"""

import numpy as np
import pytest

from speculative_decoding import MockEngine, SpeculativeEngine


def _make_engine_with_eos_after(
    n_tokens: int, vocab_size: int = 8, eos: int = 0
) -> MockEngine:
    """Create an engine that generates exactly `n_tokens` tokens before EOS.

    From prompt [1], the chain is: 1 -> 2 -> 3 -> ... -> (n_tokens+1) -> EOS
    so generate() returns n_tokens tokens.
    """
    table = np.full((vocab_size, vocab_size), -10.0, dtype=np.float32)
    # Build chain: 1 -> 2 -> 3 -> ... -> last -> EOS
    prev = 1
    for step in range(n_tokens):
        next_tok = (step + 1) % (vocab_size - 1) + 1  # tokens 2, 3, 4, ...
        table[prev, next_tok] = 10.0
        prev = next_tok
    table[prev, eos] = 10.0  # last token -> EOS
    table[eos, eos] = 10.0
    return MockEngine(table, eos_token_id=eos)


class TestEOSMidDraft:
    """EOS encountered during draft proposal."""

    @pytest.mark.parametrize("k", [3, 5, 8])
    def test_eos_after_2_tokens(self, k: int) -> None:
        """Verifier produces EOS after 2 tokens. Speculative must stop there."""
        verifier = _make_engine_with_eos_after(2, vocab_size=8)
        # Draft: same chain so it also hits EOS at the same point
        draft = _make_engine_with_eos_after(2, vocab_size=8)

        spec = SpeculativeEngine(draft, verifier, draft_length=k)
        baseline = verifier.generate([1], max_tokens=20)
        result = spec.generate([1], max_tokens=20)

        assert result.tokens == baseline.tokens
        assert len(result.tokens) == 2  # exactly 2 before EOS

    def test_eos_on_first_token(self) -> None:
        """Verifier immediately produces EOS."""
        vocab_size = 8
        table = np.full((vocab_size, vocab_size), -10.0, dtype=np.float32)
        # Token 1 -> EOS
        table[1, 0] = 10.0
        table[0, 0] = 10.0

        engine = MockEngine(table, eos_token_id=0)
        spec = SpeculativeEngine(engine, engine, draft_length=5)

        baseline = engine.generate([1], max_tokens=20)
        result = spec.generate([1], max_tokens=20)

        assert result.tokens == baseline.tokens
        assert len(result.tokens) == 0


class TestAllRejected:
    """Verifier rejects all K draft tokens every round."""

    def test_all_rejected_still_correct(self) -> None:
        """Even with 0% acceptance, output must match verifier."""
        vocab_size = 6
        eos = 0
        rng = np.random.default_rng(999)

        # Construct tables where draft and verifier disagree on every row
        verifier_table = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        draft_table = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        for i in range(vocab_size):
            if i == eos:
                verifier_table[i, eos] = 10.0
                draft_table[i, eos] = 10.0
                continue
            # Verifier picks token (i % (vocab_size-1)) + 1
            v_choice = (i % (vocab_size - 1)) + 1
            verifier_table[i, v_choice] = 10.0
            # Draft picks a DIFFERENT token
            d_choices = [t for t in range(1, vocab_size) if t != v_choice]
            d_choice = d_choices[i % len(d_choices)]
            draft_table[i, d_choice] = 10.0

        verifier = MockEngine(verifier_table, eos_token_id=eos)
        draft = MockEngine(draft_table, eos_token_id=eos)

        for k in [1, 3, 5]:
            spec = SpeculativeEngine(draft, verifier, draft_length=k)
            baseline = verifier.generate([1], max_tokens=15)
            result = spec.generate([1], max_tokens=15)

            assert result.tokens == baseline.tokens, (
                f"All-reject case failed with K={k}"
            )


class TestContextLimit:
    """Generation must respect max_context_length."""

    def test_stops_at_context_limit(self) -> None:
        max_ctx = 10
        vocab_size = 6
        # Simple chain: 1->2->3->4->5->1->2->... (never hits EOS)
        table = np.full((vocab_size, vocab_size), -10.0, dtype=np.float32)
        for i in range(1, vocab_size):
            next_t = (i % (vocab_size - 1)) + 1
            table[i, next_t] = 10.0
        table[0, 1] = 10.0  # EOS -> 1 (shouldn't matter)

        verifier = MockEngine(table, eos_token_id=0, max_context_length=max_ctx)
        draft = MockEngine(table, eos_token_id=0, max_context_length=max_ctx)
        spec = SpeculativeEngine(draft, verifier, draft_length=5)

        prompt = [1, 2, 3]  # 3 tokens
        baseline = verifier.generate(prompt, max_tokens=100)
        result = spec.generate(prompt, max_tokens=100)

        # Should produce at most max_ctx - len(prompt) = 7 tokens
        assert len(result.tokens) <= max_ctx - len(prompt)
        assert result.tokens == baseline.tokens


class TestDraftLengthOne:
    """K=1 is a degenerate case — each round proposes one token."""

    def test_k1_still_correct(self) -> None:
        rng = np.random.default_rng(77)
        vocab_size = 8
        verifier_table = rng.standard_normal((vocab_size, vocab_size)).astype(np.float32)
        draft_table = rng.standard_normal((vocab_size, vocab_size)).astype(np.float32)

        verifier = MockEngine(verifier_table, eos_token_id=0)
        draft = MockEngine(draft_table, eos_token_id=0)
        spec = SpeculativeEngine(draft, verifier, draft_length=1)

        for prompt in [[1], [3, 5], [2, 4, 6]]:
            baseline = verifier.generate(prompt, max_tokens=20)
            result = spec.generate(prompt, max_tokens=20)
            assert result.tokens == baseline.tokens


class TestMaxTokensExact:
    """Respect max_tokens exactly."""

    def test_max_tokens_1(self) -> None:
        """Single token generation."""
        vocab_size = 6
        table = np.full((vocab_size, vocab_size), -10.0, dtype=np.float32)
        for i in range(1, vocab_size):
            table[i, (i % (vocab_size - 1)) + 1] = 10.0
        table[0, 1] = 10.0

        verifier = MockEngine(table, eos_token_id=0)
        draft = MockEngine(table, eos_token_id=0)
        spec = SpeculativeEngine(draft, verifier, draft_length=5)

        baseline = verifier.generate([1], max_tokens=1)
        result = spec.generate([1], max_tokens=1)

        assert result.tokens == baseline.tokens
        assert len(result.tokens) <= 1

    def test_max_tokens_respected(self) -> None:
        """Never exceed max_tokens."""
        vocab_size = 6
        table = np.full((vocab_size, vocab_size), -10.0, dtype=np.float32)
        for i in range(1, vocab_size):
            table[i, (i % (vocab_size - 1)) + 1] = 10.0
        table[0, 1] = 10.0

        verifier = MockEngine(table, eos_token_id=0)
        draft = MockEngine(table, eos_token_id=0)

        for max_tok in [1, 2, 3, 5, 7, 10]:
            spec = SpeculativeEngine(draft, verifier, draft_length=5)
            result = spec.generate([1], max_tokens=max_tok)
            assert len(result.tokens) <= max_tok, (
                f"Exceeded max_tokens={max_tok}, got {len(result.tokens)}"
            )
