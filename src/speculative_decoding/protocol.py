"""InferenceEngine protocol — the composable contract all engines satisfy.

Inspired by the InferenceEngine protocol in apple/coreai-models (Swift), ported
to Python with two key methods:

    generate(tokens, max_tokens)  — autoregressive greedy decoding, yields tokens
    forward(tokens) -> logits     — single forward pass, returns full logit tensor

The `forward` method is the verification primitive: given a prefix of N tokens,
return logits at every position so the caller can check whether proposed tokens
match argmax(logits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GenerationResult:
    """Accumulates metadata from a generation run."""

    tokens: list[int]
    acceptance_rate: float | None = None  # only set for speculative engines
    num_draft_rounds: int | None = None


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol that all engines must satisfy.

    Engines are stateless with respect to KV cache in this reference
    implementation — each call is self-contained.  Real engines would
    maintain KV state across calls; this keeps the code simple and
    testable while preserving the algorithmic contract.
    """

    @property
    def vocab_size(self) -> int:
        """Size of the token vocabulary."""
        ...

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token id."""
        ...

    @property
    def max_context_length(self) -> int:
        """Maximum number of tokens the engine can process."""
        ...

    def forward(self, token_ids: list[int]) -> NDArray[np.float32]:
        """Run a single forward pass over `token_ids`.

        Returns:
            logits array of shape (seq_len, vocab_size) — one row of logits
            per input position.  The logits at position i correspond to the
            *prediction for the next token* given token_ids[:i+1].
        """
        ...

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> GenerationResult:
        """Greedy autoregressive generation.

        Decodes up to `max_tokens` new tokens, stopping early on EOS.
        Returns a GenerationResult with the generated tokens (NOT including
        the prompt).
        """
        ...
