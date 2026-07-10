"""Record/replay cache for frozen-model calls (the CI-reproducibility layer, roadmap A).

A real model is slow and not byte-deterministic across machines; CI must still exercise
the *real code path* deterministically. A :class:`Cassette` records every embedding /
completion produced by a real provider, keyed by a stable hash of the input. Wrapping a
provider in :class:`CachedEmbedder` / :class:`CachedLLM`:

* **record mode** (``inner`` given) — calls the real model on a miss and stores the result;
* **replay mode** (``inner=None``) — serves stored results and raises on a miss.

Float vectors serialize as ``tobytes().hex()`` exactly like :mod:`curated_brain.vector`,
so a cassette is strict, byte-stable JSON.
"""

from __future__ import annotations

import json

import numpy as np

from curated_brain.util import stable_hash


class Cassette:
    """An on-disk record of frozen-model outputs, keyed by stable input hash."""

    def __init__(self, embed_model_id: str = "", dim: int = 0) -> None:
        self.embed_model_id = embed_model_id
        self.dim = dim
        self.embed: dict[str, str] = {}     # hash -> hex(float64 vector)
        self.complete: dict[str, str] = {}  # hash -> completion text
        # Logprob completions record under a DISTINCT namespace (keyed by the same prompt
        # hash) so a plain complete() and a complete_with_logprobs() of the same prompt never
        # collide: value is [text, [logprob, ...]], with float64-hex logprobs for byte-stable
        # replay exactly like the embed vectors.
        self.complete_lp: dict[str, list] = {}  # hash -> [text, hex(float64 logprobs)]

    @staticmethod
    def _key(text: str) -> str:
        return format(stable_hash(text), "x")

    def to_dict(self) -> dict:
        return {"embed_model_id": self.embed_model_id, "dim": self.dim,
                "embed": self.embed, "complete": self.complete,
                "complete_lp": self.complete_lp}

    @classmethod
    def from_dict(cls, d: dict) -> Cassette:
        c = cls(d.get("embed_model_id", ""), int(d.get("dim", 0)))
        c.embed = dict(d.get("embed", {}))
        c.complete = dict(d.get("complete", {}))
        c.complete_lp = {k: list(v) for k, v in d.get("complete_lp", {}).items()}
        return c

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)

    @classmethod
    def load(cls, path: str) -> Cassette:
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


class CachedEmbedder:
    """Drop-in ``Embedder`` that records/replays through a :class:`Cassette`."""

    def __init__(self, cassette: Cassette, inner=None) -> None:
        self.cassette = cassette
        self.inner = inner
        self.model_id = inner.model_id if inner is not None else cassette.embed_model_id
        self.dim = inner.dim if inner is not None else cassette.dim
        if inner is not None and not cassette.embed_model_id:
            cassette.embed_model_id = inner.model_id
            cassette.dim = inner.dim

    def embed(self, text: str) -> np.ndarray:
        key = self.cassette._key(text)
        hit = self.cassette.embed.get(key)
        if hit is not None:
            return np.frombuffer(bytes.fromhex(hit), dtype=np.float64)
        if self.inner is None:
            raise KeyError(f"cassette replay miss for embed({text!r})")
        v = np.asarray(self.inner.embed(text), dtype=np.float64)
        self.cassette.embed[key] = v.tobytes().hex()
        if not self.cassette.dim:
            self.cassette.dim = self.dim = int(v.shape[0])
        return v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.embed(t) for t in texts])


class CachedLLM:
    """Drop-in ``LLM`` that records/replays completions through a :class:`Cassette`."""

    def __init__(self, cassette: Cassette, inner=None) -> None:
        self.cassette = cassette
        self.inner = inner
        self.model_id = inner.model_id if inner is not None else "cassette"

    def complete(self, prompt: str) -> str:
        key = Cassette._key(prompt)
        hit = self.cassette.complete.get(key)
        if hit is not None:
            return hit
        if self.inner is None:
            raise KeyError(f"cassette replay miss for complete({prompt[:40]!r}…)")
        out = self.inner.complete(prompt)
        self.cassette.complete[key] = out
        return out

    def complete_with_logprobs(self, prompt: str) -> tuple[str, list[float]]:
        """Record/replay ``(text, per-token logprobs)`` for the predictive-surprise seam
        (:class:`~curated_brain.surprise.PredictiveSurprise`). Keyed by the prompt hash in the
        ``complete_lp`` namespace, distinct from ``complete``; logprobs serialize as float64
        hex so replay is byte-identical. Replay miss RAISES (same strict contract as embed)."""
        key = Cassette._key(prompt)
        hit = self.cassette.complete_lp.get(key)
        if hit is not None:
            text, hexlp = hit
            lps = np.frombuffer(bytes.fromhex(hexlp), dtype=np.float64).tolist()
            return text, lps
        if self.inner is None:
            raise KeyError(
                f"cassette replay miss for complete_with_logprobs({prompt[:40]!r}…)")
        text, lps = self.inner.complete_with_logprobs(prompt)
        hexlp = np.asarray(lps, dtype=np.float64).tobytes().hex()
        self.cassette.complete_lp[key] = [text, hexlp]
        return text, list(lps)
