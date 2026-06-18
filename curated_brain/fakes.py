"""Deterministic, local implementations of the frozen-model protocols.

These stand in for real embedding/chat APIs during evaluation. They are pure functions
of their input (no network, no wall-clock, no salted hashing), which is what makes the
whole layer byte-reproducible (AC-1). Swap in a real provider and nothing else changes.
"""

from __future__ import annotations

import numpy as np

from curated_brain.util import stable_hash, tokenize


class DeterministicEmbedder:
    """Hashing-trick bag-of-tokens embedder.

    Each token is hashed to a signed bucket; the accumulated vector is L2-normalized.
    Texts that share vocabulary land closer in cosine space, so semantic recall is
    meaningful while remaining a deterministic function of the text alone.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.model_id = f"det-hash-{dim}-v1"

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in tokenize(text):
            h = stable_hash(tok)
            idx = h % self.dim
            v[idx] += 1.0 if (h >> 32) & 1 else -1.0
        norm = float(np.linalg.norm(v))
        if norm > 0.0:
            v /= norm
        return v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.embed(t) for t in texts])


class RuleBasedLLM:
    """A deterministic, frozen "chat" endpoint.

    The core curation path never depends on free-form generation, so this returns a
    deterministic extractive response (the most salient input line). It exists to satisfy
    the protocol and to be swapped for a real model when richer summarization is wanted.
    """

    model_id = "rule-llm-v1"

    def complete(self, prompt: str) -> str:
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        if not lines:
            return ""
        # Deterministic "salience": the longest line, tie-broken by first occurrence.
        return max(lines, key=lambda ln: (len(ln), -lines.index(ln)))
