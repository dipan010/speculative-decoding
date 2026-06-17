"""Speculative decoding engine — wraps a draft + verifier engine pair.

Algorithm (greedy):
    1. Draft engine proposes K tokens autoregressively.
    2. Verifier does ONE forward pass over the full sequence
       (prompt + accepted + K draft tokens).
    3. Compare: for each draft token at position i, check if
       argmax(verifier_logits[i-1]) == draft_token[i].
       Accept the longest matching prefix.
    4. At the first mismatch (or after all K match), take the
       verifier's own argmax as the next token.
    5. Repeat until max_tokens or EOS.

CORRECTNESS INVARIANT (greedy):
    The output is token-for-token identical to running the verifier
    alone with greedy decoding.  This is because:
    - Every accepted token matches argmax(verifier) at that position.
    - The correction token at a rejection *is* argmax(verifier).
    - If all K draft tokens are accepted, the bonus token is
      argmax(verifier) at position K — exactly what plain greedy
      would have produced next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import GenerationResult, InferenceEngine


@dataclass
class _DraftRoundStats:
    proposed: int
    accepted: int


class SpeculativeEngine:
    """Speculative decoding engine composing a draft and verifier engine.

    Args:
        draft_engine: fast, less accurate engine for proposing tokens.
        verifier_engine: slower, authoritative engine.
        draft_length: number of tokens (K) the draft proposes per round.
    """

    def __init__(
        self,
        draft_engine: InferenceEngine,
        verifier_engine: InferenceEngine,
        draft_length: int = 5,
    ) -> None:
        self._draft = draft_engine
        self._verifier = verifier_engine
        self._draft_length = draft_length

    @property
    def vocab_size(self) -> int:
        return self._verifier.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self._verifier.eos_token_id

    @property
    def max_context_length(self) -> int:
        return self._verifier.max_context_length

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """Delegate to verifier for direct forward passes."""
        return self._verifier.forward(token_ids)

    def generate(self, prompt_tokens: list[int], max_tokens: int) -> GenerationResult:
        """Greedy speculative decoding.

        Produces token-for-token identical output to verifier-only greedy
        decoding, but (with a good draft model) in fewer wall-clock seconds
        because the verifier processes K tokens in one forward pass instead
        of one at a time.
        """
        context = list(prompt_tokens)
        generated: list[int] = []
        round_stats: list[_DraftRoundStats] = []

        while len(generated) < max_tokens:
            remaining = max_tokens - len(generated)
            context_budget = self.max_context_length - len(context)
            if context_budget <= 0:
                break

            # --- Step 1: Draft proposes K tokens ---
            k = min(self._draft_length, remaining, context_budget)
            draft_tokens = self._draft_greedy(context, k)

            if not draft_tokens:
                # Draft produced nothing (e.g. immediate EOS) — fall back to
                # single verifier step
                token = self._verifier_single_step(context)
                if token == self.eos_token_id:
                    break
                generated.append(token)
                context.append(token)
                round_stats.append(_DraftRoundStats(proposed=0, accepted=0))
                continue

            # --- Step 2: Verifier forward pass over context + draft tokens ---
            verify_input = context + draft_tokens
            logits = self._verifier.forward(verify_input)
            # logits shape: (len(verify_input), vocab_size)
            # logits[i] predicts the token at position i+1
            # We care about positions len(context)-1 through len(context)-1+len(draft_tokens)
            # which predict tokens at positions len(context) through len(context)+len(draft_tokens)

            # --- Step 3: Accept longest matching prefix ---
            n_accepted = 0
            base = len(context) - 1  # logits index for "predicts first draft position"

            for i, draft_tok in enumerate(draft_tokens):
                verifier_choice = int(np.argmax(logits[base + i]))
                if verifier_choice == draft_tok:
                    n_accepted += 1
                    # Check if accepted token is EOS
                    if draft_tok == self.eos_token_id:
                        # Append the accepted prefix BEFORE EOS, then stop
                        accepted_prefix = draft_tokens[:n_accepted - 1]  # exclude EOS
                        generated.extend(accepted_prefix)
                        round_stats.append(
                            _DraftRoundStats(proposed=len(draft_tokens), accepted=n_accepted)
                        )
                        return self._make_result(generated, round_stats)
                else:
                    break

            # --- Step 4: Append accepted tokens + correction/bonus ---
            accepted = draft_tokens[:n_accepted]
            generated.extend(accepted)
            context.extend(accepted)

            if n_accepted < len(draft_tokens):
                # Rejection: take verifier's token at the rejection point
                correction = int(np.argmax(logits[base + n_accepted]))
                if correction == self.eos_token_id:
                    round_stats.append(
                        _DraftRoundStats(proposed=len(draft_tokens), accepted=n_accepted)
                    )
                    break
                generated.append(correction)
                context.append(correction)
            else:
                # All K accepted — take bonus token from verifier at position K
                bonus_logits_idx = base + len(draft_tokens)
                bonus = int(np.argmax(logits[bonus_logits_idx]))
                if bonus == self.eos_token_id:
                    round_stats.append(
                        _DraftRoundStats(proposed=len(draft_tokens), accepted=n_accepted)
                    )
                    break
                generated.append(bonus)
                context.append(bonus)

            round_stats.append(
                _DraftRoundStats(proposed=len(draft_tokens), accepted=n_accepted)
            )

            # Clamp to max_tokens and context limit — a full accept + bonus
            # can overshoot either bound within a single round
            max_gen = min(max_tokens, self.max_context_length - len(prompt_tokens))
            if len(generated) > max_gen:
                generated = generated[:max_gen]
                context = list(prompt_tokens) + generated
                break

        return self._make_result(generated, round_stats)

    def _draft_greedy(self, context: list[int], k: int) -> list[int]:
        """Have the draft engine propose up to k tokens greedily.

        Stops early if draft produces EOS (but still returns EOS so the
        verifier can confirm it).
        """
        draft_context = list(context)
        draft_tokens: list[int] = []

        for _ in range(k):
            if len(draft_context) >= self._draft.max_context_length:
                break
            logits = self._draft.forward(draft_context)
            next_token = int(np.argmax(logits[-1]))
            draft_tokens.append(next_token)
            if next_token == self._draft.eos_token_id:
                break  # include EOS in proposal for verifier to check
            draft_context.append(next_token)

        return draft_tokens

    def _verifier_single_step(self, context: list[int]) -> int:
        """Single greedy step from the verifier."""
        logits = self._verifier.forward(context)
        return int(np.argmax(logits[-1]))

    @staticmethod
    def _make_result(
        generated: list[int], round_stats: list[_DraftRoundStats]
    ) -> GenerationResult:
        total_proposed = sum(r.proposed for r in round_stats)
        total_accepted = sum(r.accepted for r in round_stats)
        acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0.0
        return GenerationResult(
            tokens=generated,
            acceptance_rate=acceptance_rate,
            num_draft_rounds=len(round_stats),
        )
