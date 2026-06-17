"""Speculative decoding — a composable inference-engine wrapper.

This package provides:
  - InferenceEngine protocol (generate + forward)
  - MockEngine for deterministic testing
  - HFEngine backed by HuggingFace transformers (lazy-loaded)
  - SpeculativeEngine that wraps draft + verifier for speculative decoding
"""

from .mock_engine import MockEngine, make_agreeing_engines, make_disagreeing_engines
from .protocol import GenerationResult, InferenceEngine
from .speculative_engine import SpeculativeEngine

__all__ = [
    "GenerationResult",
    "HFEngine",
    "InferenceEngine",
    "MockEngine",
    "SpeculativeEngine",
    "make_agreeing_engines",
    "make_disagreeing_engines",
]


def __getattr__(name: str):
    if name == "HFEngine":
        from .hf_engine import HFEngine
        return HFEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
