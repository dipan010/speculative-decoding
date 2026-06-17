"""HuggingFace Transformers-backed inference engine.

Wraps any causal LM from the transformers library to satisfy the
InferenceEngine protocol.  Intended for use with small models like
Qwen2-0.5B (draft) and Qwen2-1.5B (verifier) so benchmarks are
runnable on CPU or MPS without a GPU cluster.

Design notes:
  - Each `forward()` call recomputes from scratch (no KV cache reuse).
    This is intentionally simple — the point is to demonstrate the
    speculative decoding algorithm, not to build a production serving
    engine.  KV-cache reuse would make the verifier forward even cheaper
    relative to autoregressive decoding, amplifying the speedup.
  - `generate()` is plain greedy: argmax at each step, one token at a time.
  - Device selection: MPS if available, else CPU.
"""

from __future__ import annotations

import torch
import numpy as np
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer

from .protocol import GenerationResult


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HFEngine:
    """Inference engine backed by a HuggingFace causal language model.

    Args:
        model_name: HuggingFace model identifier (e.g. "Qwen/Qwen2-0.5B").
        max_context_length: override for the model's max position embeddings.
            Defaults to the model's config value.
        device: torch device. Defaults to MPS > CUDA > CPU.
    """

    def __init__(
        self,
        model_name: str,
        max_context_length: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._device = device or _select_device()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,  # float32 for deterministic argmax on CPU/MPS
        ).to(self._device)
        self._model.eval()

        config_max = getattr(self._model.config, "max_position_embeddings", 2048)
        self._max_context_length = max_context_length or config_max

        # Resolve EOS token id
        if self._tokenizer.eos_token_id is not None:
            self._eos_token_id = self._tokenizer.eos_token_id
        elif hasattr(self._model.config, "eos_token_id") and self._model.config.eos_token_id is not None:
            eos = self._model.config.eos_token_id
            self._eos_token_id = eos[0] if isinstance(eos, list) else eos
        else:
            self._eos_token_id = 0

    @property
    def vocab_size(self) -> int:
        return self._model.config.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        return self._device

    def forward(self, token_ids: list[int]) -> NDArray[np.float32]:
        """Single forward pass — returns logits (seq_len, vocab_size).

        Each row i contains logits predicting the token at position i+1,
        conditioned on token_ids[:i+1].
        """
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            outputs = self._model(input_ids)
        # outputs.logits shape: (1, seq_len, vocab_size)
        return outputs.logits[0].float().cpu().numpy()

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> GenerationResult:
        """Greedy autoregressive generation, one token at a time."""
        context = list(prompt_tokens)
        generated: list[int] = []

        for _ in range(max_tokens):
            if len(context) >= self._max_context_length:
                break
            input_ids = torch.tensor([context], dtype=torch.long, device=self._device)
            with torch.no_grad():
                outputs = self._model(input_ids)
            # Take logits at the last position
            logits = outputs.logits[0, -1]  # (vocab_size,)
            next_token = int(torch.argmax(logits).item())
            if next_token == self._eos_token_id:
                break
            generated.append(next_token)
            context.append(next_token)

        return GenerationResult(tokens=generated)
