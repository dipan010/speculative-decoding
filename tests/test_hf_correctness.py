"""Correctness test with real HuggingFace models.

Verifies that speculative decoding with Qwen2 draft + verifier produces
token-for-token identical output to verifier-only greedy decoding.

Marked as slow — these tests download models on first run.
"""

import pytest

pytestmark = pytest.mark.slow

from speculative_decoding import HFEngine, SpeculativeEngine

DRAFT_MODEL = "Qwen/Qwen2.5-0.5B"
VERIFIER_MODEL = "Qwen/Qwen2.5-1.5B"

PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def draft_engine() -> HFEngine:
    return HFEngine(DRAFT_MODEL)


@pytest.fixture(scope="module")
def verifier_engine() -> HFEngine:
    return HFEngine(VERIFIER_MODEL)


class TestHFIdenticalOutput:
    """Real-model correctness: speculative must match verifier greedy."""

    @pytest.mark.parametrize("k", [1, 3, 5, 8])
    def test_identical_output(
        self, draft_engine: HFEngine, verifier_engine: HFEngine, k: int
    ) -> None:
        for prompt_text in PROMPTS:
            prompt_tokens = verifier_engine.tokenizer.encode(
                prompt_text, add_special_tokens=False
            )

            baseline = verifier_engine.generate(prompt_tokens, max_tokens=32)
            spec = SpeculativeEngine(draft_engine, verifier_engine, draft_length=k)
            speculative = spec.generate(prompt_tokens, max_tokens=32)

            baseline_text = verifier_engine.tokenizer.decode(baseline.tokens)
            spec_text = verifier_engine.tokenizer.decode(speculative.tokens)

            assert speculative.tokens == baseline.tokens, (
                f"CORRECTNESS VIOLATION with real models!\n"
                f"  prompt: {prompt_text!r}\n"
                f"  K={k}\n"
                f"  verifier:    {baseline_text!r}\n"
                f"  speculative: {spec_text!r}"
            )
