"""Benchmark: measure speculative decoding speedup with real HF models.

Uses Qwen2-0.5B as draft and Qwen2-1.5B as verifier.  Measures:
  - Acceptance rate (avg tokens accepted per draft round)
  - Wall-clock speedup vs verifier-only greedy decoding
  - K-sweep: how speedup varies with draft length K

Produces a plot saved as benchmark_results.png.

Usage:
    uv run benchmark
    uv run python -m speculative_decoding.benchmark
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

from .hf_engine import HFEngine
from .speculative_engine import SpeculativeEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DRAFT_MODEL = "Qwen/Qwen2.5-0.5B"
VERIFIER_MODEL = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 64

PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is an optimization algorithm that",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "The three laws of thermodynamics state that",
]

K_VALUES = [1, 2, 3, 4, 5, 7, 10, 15]


@dataclass
class BenchResult:
    prompt: str
    method: str
    k: int | None
    tokens_generated: int
    wall_seconds: float
    acceptance_rate: float | None
    num_rounds: int | None
    tokens: list[int]


def _tokenize(engine: HFEngine, text: str) -> list[int]:
    return engine.tokenizer.encode(text, add_special_tokens=False)


def _decode(engine: HFEngine, token_ids: list[int]) -> str:
    return engine.tokenizer.decode(token_ids, skip_special_tokens=True)


def benchmark_one(
    prompt_tokens: list[int],
    prompt_text: str,
    draft: HFEngine,
    verifier: HFEngine,
    k: int,
    max_tokens: int,
) -> tuple[BenchResult, BenchResult]:
    """Run verifier-only and speculative, return both results."""
    # --- Verifier-only baseline ---
    t0 = time.perf_counter()
    baseline = verifier.generate(prompt_tokens, max_tokens=max_tokens)
    t_baseline = time.perf_counter() - t0

    baseline_result = BenchResult(
        prompt=prompt_text,
        method="verifier-only",
        k=None,
        tokens_generated=len(baseline.tokens),
        wall_seconds=t_baseline,
        acceptance_rate=None,
        num_rounds=None,
        tokens=baseline.tokens,
    )

    # --- Speculative ---
    spec = SpeculativeEngine(draft, verifier, draft_length=k)
    t0 = time.perf_counter()
    speculative = spec.generate(prompt_tokens, max_tokens=max_tokens)
    t_spec = time.perf_counter() - t0

    # Correctness check — the whole point
    if speculative.tokens != baseline.tokens:
        print(f"  *** CORRECTNESS VIOLATION at K={k} ***")
        print(f"      baseline:    {baseline.tokens[:20]}...")
        print(f"      speculative: {speculative.tokens[:20]}...")
    else:
        print(f"  ✓ K={k}: identical output ({len(speculative.tokens)} tokens)")

    spec_result = BenchResult(
        prompt=prompt_text,
        method=f"speculative-K{k}",
        k=k,
        tokens_generated=len(speculative.tokens),
        wall_seconds=t_spec,
        acceptance_rate=speculative.acceptance_rate,
        num_rounds=speculative.num_draft_rounds,
        tokens=speculative.tokens,
    )

    return baseline_result, spec_result


def plot_results(
    k_values: list[int],
    avg_speedups: list[float],
    avg_acceptance: list[float],
    output_path: str = "benchmark_results.png",
) -> None:
    """Generate the K-sweep plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Speedup vs K
    ax1.plot(k_values, avg_speedups, "bo-", linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color="gray", linestyle="--", label="baseline (1.0x)")
    ax1.set_xlabel("Draft Length K", fontsize=12)
    ax1.set_ylabel("Speedup vs Verifier-Only", fontsize=12)
    ax1.set_title("Wall-Clock Speedup vs Draft Length", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    best_k = k_values[int(np.argmax(avg_speedups))]
    best_speedup = max(avg_speedups)
    ax1.annotate(
        f"Best: K={best_k} ({best_speedup:.2f}x)",
        xy=(best_k, best_speedup),
        xytext=(best_k + 1, best_speedup + 0.1),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    # Acceptance rate vs K
    ax2.plot(k_values, avg_acceptance, "rs-", linewidth=2, markersize=8)
    ax2.set_xlabel("Draft Length K", fontsize=12)
    ax2.set_ylabel("Avg Acceptance Rate", fontsize=12)
    ax2.set_title("Draft Acceptance Rate vs Draft Length", fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")


def main() -> None:
    print("=" * 60)
    print("Speculative Decoding Benchmark")
    print("=" * 60)
    print(f"Draft model:    {DRAFT_MODEL}")
    print(f"Verifier model: {VERIFIER_MODEL}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"K values:       {K_VALUES}")
    print()

    print("Loading models...")
    t0 = time.perf_counter()
    draft = HFEngine(DRAFT_MODEL)
    verifier = HFEngine(VERIFIER_MODEL)
    print(f"Models loaded in {time.perf_counter() - t0:.1f}s on {verifier.device}")
    print()

    # Warmup: one forward pass each to trigger compilation/lazy init
    dummy = _tokenize(verifier, "warmup")
    draft.forward(dummy)
    verifier.forward(dummy)

    # Collect results per K
    k_to_speedups: dict[int, list[float]] = {k: [] for k in K_VALUES}
    k_to_acceptance: dict[int, list[float]] = {k: [] for k in K_VALUES}

    for prompt_text in PROMPTS:
        prompt_tokens = _tokenize(verifier, prompt_text)
        print(f"\nPrompt: {prompt_text!r} ({len(prompt_tokens)} tokens)")
        print("-" * 50)

        # Get baseline timing (run once, reuse)
        t0 = time.perf_counter()
        baseline = verifier.generate(prompt_tokens, max_tokens=MAX_NEW_TOKENS)
        baseline_time = time.perf_counter() - t0
        baseline_text = _decode(verifier, baseline.tokens)
        print(f"  Verifier-only: {len(baseline.tokens)} tokens in {baseline_time:.3f}s")
        print(f"  Output: {baseline_text[:80]}...")

        for k in K_VALUES:
            spec = SpeculativeEngine(draft, verifier, draft_length=k)
            t0 = time.perf_counter()
            result = spec.generate(prompt_tokens, max_tokens=MAX_NEW_TOKENS)
            spec_time = time.perf_counter() - t0

            # Correctness
            if result.tokens != baseline.tokens:
                print(f"  *** K={k}: CORRECTNESS VIOLATION ***")
            else:
                speedup = baseline_time / spec_time if spec_time > 0 else float("inf")
                accept = result.acceptance_rate or 0.0
                k_to_speedups[k].append(speedup)
                k_to_acceptance[k].append(accept)
                print(
                    f"  K={k:2d}: {spec_time:.3f}s "
                    f"(speedup={speedup:.2f}x, "
                    f"accept={accept:.1%}, "
                    f"rounds={result.num_draft_rounds})"
                )

    # Aggregate
    print("\n" + "=" * 60)
    print("Summary (averaged over prompts)")
    print("=" * 60)
    avg_speedups = []
    avg_acceptance = []
    for k in K_VALUES:
        sp = np.mean(k_to_speedups[k]) if k_to_speedups[k] else 1.0
        ac = np.mean(k_to_acceptance[k]) if k_to_acceptance[k] else 0.0
        avg_speedups.append(float(sp))
        avg_acceptance.append(float(ac))
        print(f"  K={k:2d}: avg speedup={sp:.2f}x, avg acceptance={ac:.1%}")

    best_idx = int(np.argmax(avg_speedups))
    print(f"\n  Sweet spot: K={K_VALUES[best_idx]} with {avg_speedups[best_idx]:.2f}x speedup")

    # Plot
    plot_results(K_VALUES, avg_speedups, avg_acceptance)


if __name__ == "__main__":
    main()
