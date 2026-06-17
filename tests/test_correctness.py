"""CRITICAL CORRECTNESS TEST: speculative decoding must produce token-for-token
identical output to plain greedy decoding with the verifier alone.

This is the centerpiece test.  If speculative decoding changes even one token,
the implementation is broken.
"""

import numpy as np
import pytest

from speculative_decoding import (
    MockEngine,
    SpeculativeEngine,
    make_agreeing_engines,
    make_disagreeing_engines,
)


class TestIdenticalOutput:
    """Greedy speculative decoding MUST match greedy verifier-only decoding."""

    @pytest.fixture(params=[3, 5, 8, 12])
    def vocab_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(params=[1, 3, 5, 7])
    def draft_length(self, request: pytest.FixtureRequest) -> int:
        return request.param

    def test_identical_when_draft_agrees(
        self, vocab_size: int, draft_length: int
    ) -> None:
        """When draft and verifier agree, speculative output == verifier output."""
        draft, verifier = make_agreeing_engines(vocab_size=vocab_size, seed=42)
        spec = SpeculativeEngine(draft, verifier, draft_length=draft_length)

        for prompt in [[1], [2, 3], [1, 4, 2]]:
            # Filter prompt to valid tokens
            prompt = [t % vocab_size for t in prompt]
            if any(t == 0 for t in prompt):  # avoid EOS in prompt
                prompt = [max(1, t) for t in prompt]

            baseline = verifier.generate(prompt, max_tokens=20)
            speculative = spec.generate(prompt, max_tokens=20)

            assert speculative.tokens == baseline.tokens, (
                f"CORRECTNESS VIOLATION: speculative decoding produced different "
                f"tokens than verifier-only greedy!\n"
                f"  prompt={prompt}\n"
                f"  verifier={baseline.tokens}\n"
                f"  speculative={speculative.tokens}"
            )

    def test_identical_when_draft_disagrees(
        self, vocab_size: int, draft_length: int
    ) -> None:
        """Even when draft is wrong, correction must yield verifier's output."""
        if vocab_size < 5:
            pytest.skip("Need vocab >= 5 for disagreeing engines")

        draft, verifier = make_disagreeing_engines(vocab_size=vocab_size, seed=123)
        spec = SpeculativeEngine(draft, verifier, draft_length=draft_length)

        for prompt in [[1], [2, 3], [1, 4, 2]]:
            prompt = [t % vocab_size for t in prompt]
            if any(t == 0 for t in prompt):
                prompt = [max(1, t) for t in prompt]

            baseline = verifier.generate(prompt, max_tokens=30)
            speculative = spec.generate(prompt, max_tokens=30)

            assert speculative.tokens == baseline.tokens, (
                f"CORRECTNESS VIOLATION with disagreeing draft!\n"
                f"  prompt={prompt}\n"
                f"  verifier={baseline.tokens}\n"
                f"  speculative={speculative.tokens}"
            )

    def test_identical_with_random_tables(self) -> None:
        """Fuzz: random logit tables, speculative must still match verifier."""
        rng = np.random.default_rng(2024)
        vocab_size = 10
        eos = 0

        for trial in range(20):
            verifier_table = rng.standard_normal((vocab_size, vocab_size)).astype(np.float32)
            draft_table = rng.standard_normal((vocab_size, vocab_size)).astype(np.float32)

            verifier = MockEngine(verifier_table, eos_token_id=eos)
            draft = MockEngine(draft_table, eos_token_id=eos)

            for k in [1, 3, 5, 10]:
                spec = SpeculativeEngine(draft, verifier, draft_length=k)
                prompt = [int(rng.integers(1, vocab_size))]
                baseline = verifier.generate(prompt, max_tokens=25)
                speculative = spec.generate(prompt, max_tokens=25)

                assert speculative.tokens == baseline.tokens, (
                    f"FUZZ FAILURE trial={trial}, K={k}\n"
                    f"  prompt={prompt}\n"
                    f"  verifier={baseline.tokens}\n"
                    f"  speculative={speculative.tokens}"
                )


class TestAcceptanceRate:
    """Verify acceptance rate reporting is reasonable."""

    def test_perfect_acceptance_when_agreeing(self) -> None:
        draft, verifier = make_agreeing_engines(vocab_size=8, seed=42)
        spec = SpeculativeEngine(draft, verifier, draft_length=5)
        result = spec.generate([1], max_tokens=20)
        # When draft agrees perfectly, acceptance rate should be 1.0
        assert result.acceptance_rate is not None
        assert result.acceptance_rate == pytest.approx(1.0), (
            f"Expected perfect acceptance but got {result.acceptance_rate}"
        )

    def test_partial_acceptance_when_disagreeing(self) -> None:
        draft, verifier = make_disagreeing_engines(vocab_size=8, seed=123)
        spec = SpeculativeEngine(draft, verifier, draft_length=5)
        result = spec.generate([1], max_tokens=20)
        assert result.acceptance_rate is not None
        # Should be between 0 and 1 (not necessarily perfect)
        assert 0.0 <= result.acceptance_rate <= 1.0
