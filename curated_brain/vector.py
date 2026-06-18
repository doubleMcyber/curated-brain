"""Vector tier — embeddings over episodic + semantic records with metadata filters.

The similarity engine sits behind a ``VectorIndex`` protocol (PRD §5.3) so the in-process
exact index used for deterministic eval can be swapped for ``hnswlib`` / ``faiss`` /
``sqlite-vec`` without touching the tier. Embeddings are unit-norm, so cosine similarity is
a dot product. Vectors serialize as exact hex bytes, keeping snapshots byte-deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from curated_brain.util import normalize


@runtime_checkable
class VectorIndex(Protocol):
    """Minimal ANN seam: add a keyed vector, rank all keys by similarity, remove a key."""

    def add(self, key: int, vector: np.ndarray) -> None: ...

    def rank(self, vector: np.ndarray) -> list[tuple[int, float]]: ...

    def remove(self, key: int) -> None: ...


class BruteForceIndex:
    """Exact cosine search (dot product on unit vectors). Deterministic tie-break by key."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vecs: dict[int, np.ndarray] = {}

    def add(self, key: int, vector: np.ndarray) -> None:
        self._vecs[key] = np.asarray(vector, dtype=np.float64)

    def remove(self, key: int) -> None:
        self._vecs.pop(key, None)

    def rank(self, vector: np.ndarray) -> list[tuple[int, float]]:
        if not self._vecs:
            return []
        keys = list(self._vecs.keys())  # insertion order — deterministic
        mat = np.stack([self._vecs[k] for k in keys])
        scores = mat @ np.asarray(vector, dtype=np.float64)
        order = sorted(range(len(keys)), key=lambda i: (-float(scores[i]), keys[i]))
        return [(keys[i], float(scores[i])) for i in order]

    def to_dict(self) -> dict:
        return {"dim": self.dim,
                "items": [[k, v.tobytes().hex()] for k, v in self._vecs.items()]}

    @classmethod
    def from_dict(cls, d: dict) -> BruteForceIndex:
        idx = cls(d["dim"])
        for k, hexv in d["items"]:
            idx._vecs[int(k)] = np.frombuffer(bytes.fromhex(hexv), dtype=np.float64)
        return idx


@dataclass
class VectorRecord:
    rid: str
    text: str
    wall_ts: float
    session_id: str
    tier: str = "episodic"  # "episodic" | "semantic"
    entities: list[str] = field(default_factory=list)

    @property
    def entities_norm(self) -> list[str]:
        return [normalize(e) for e in self.entities]


class VectorTier:
    """Embedded records + metadata-filtered ANN search (PRD §7 step 2)."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.index = BruteForceIndex(embedder.dim)
        self.meta: dict[int, VectorRecord] = {}
        self._next = 0

    def add(self, *, rid: str, text: str, wall_ts: float, session_id: str,
            tier: str = "episodic", entities: list[str] | None = None,
            embedding: np.ndarray | None = None) -> int:
        key = self._next
        self._next += 1
        emb = embedding if embedding is not None else self.embedder.embed(text)
        self.index.add(key, emb)
        self.meta[key] = VectorRecord(rid=rid, text=text, wall_ts=wall_ts,
                                      session_id=session_id, tier=tier,
                                      entities=list(entities or []))
        return key

    def remove(self, key: int) -> None:
        self.index.remove(key)
        self.meta.pop(key, None)

    def search(self, query, k: int = 8, *, t: float | None = None,
               entity: str | None = None, tier: str | None = None,
               window: tuple[float, float] | None = None) -> list[tuple[VectorRecord, float]]:
        """Top-k records by cosine, after applying causal + metadata filters.

        ``t`` enforces causality (only records with ``wall_ts <= t``); ``entity``/``tier``/
        ``window`` are the metadata filters the retrieval planner uses (PRD §7).
        """
        qv = self.embedder.embed(query) if isinstance(query, str) else np.asarray(query)
        ent = normalize(entity) if entity is not None else None
        out: list[tuple[VectorRecord, float]] = []
        for key, score in self.index.rank(qv):
            r = self.meta[key]
            if t is not None and r.wall_ts > t:
                continue
            if tier is not None and r.tier != tier:
                continue
            if ent is not None and ent not in r.entities_norm:
                continue
            if window is not None and not (window[0] <= r.wall_ts <= window[1]):
                continue
            out.append((r, score))
            if len(out) >= k:
                break
        return out

    def __len__(self) -> int:
        return len(self.meta)

    # ------------------------------------------------------------------ persistence --
    def to_dict(self) -> dict:
        return {
            "next": self._next,
            "index": self.index.to_dict(),
            "meta": [[k, vars(r)] for k, r in self.meta.items()],
        }

    def load(self, d: dict) -> None:
        self._next = d["next"]
        self.index = BruteForceIndex.from_dict(d["index"])
        self.meta = {int(k): VectorRecord(**fields) for k, fields in d["meta"]}
