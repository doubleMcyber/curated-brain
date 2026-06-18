"""Frozen-model protocols (PRD §1.3, §5.3).

The whole layer depends only on (a) an embedding endpoint and (b) a chat/completion
endpoint, each behind a narrow protocol so no concrete vendor leaks into core logic and
so the eval can run against deterministic local implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Maps text to a unit-norm vector. ``model_id`` is recorded into provenance so
    vectors can be re-embedded when the model is upgraded (PRD §12)."""

    model_id: str
    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class LLM(Protocol):
    """A frozen chat/completion endpoint. Used only for *optional* secondary signals
    (predictive surprise, free-form summarization); the core path never requires it."""

    model_id: str

    def complete(self, prompt: str) -> str: ...
