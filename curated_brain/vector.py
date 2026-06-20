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

from curated_brain.util import jaccard, normalize

# Hybrid-retrieval weights: blend embedding similarity (semantic) with lexical token-overlap
# at the search ranking, so neither modality's blind spot dominates. General + deterministic.
_W_SEM = 0.5
_W_LEX = 0.5
# Over-fetch factor for the optional ANN backend: pull k*_OVERFETCH approximate candidates so
# the metadata filters + hybrid re-rank have headroom (very selective filters could still
# under-recall — filter-pushdown is the documented follow-up). No effect on the exact default.
_OVERFETCH = 8


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


class HnswIndex:
    """Approximate cosine ANN over unit vectors via ``hnswlib`` — an optional PRODUCTION
    backend for scale (sublinear top-k query vs brute force's O(n)). Conforms to
    ``VectorIndex`` so it drops into :class:`VectorTier`.

    Scope: it is *approximate* and *not* byte-deterministic, so the deterministic
    :class:`BruteForceIndex` remains the DEFAULT (AC-1 snapshot determinism). Use this only
    where corpus size makes brute force too slow; ``topk`` is the fast path, while ``rank``
    returns the full ranking for protocol compatibility. Filter-pushdown for selective
    metadata filters (over-fetch correctness) is a documented follow-up.
    """

    def __init__(self, dim: int, *, max_elements: int = 1024, ef_construction: int = 200,
                 m: int = 16, ef: int = 200, seed: int = 100) -> None:
        try:
            import hnswlib
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "HnswIndex requires hnswlib: pip install 'curated-brain[scale]'") from e
        self.dim = dim
        self._cap = max_elements
        self._idx = hnswlib.Index(space="ip", dim=dim)  # unit vectors -> inner product == cosine
        self._idx.init_index(max_elements=max_elements, ef_construction=ef_construction,
                             M=m, random_seed=seed)
        self._idx.set_ef(ef)
        self._idx.set_num_threads(1)
        self._live: set[int] = set()
        self._added = 0

    def add(self, key: int, vector: np.ndarray) -> None:
        if self._added >= self._cap:
            self._cap = max(self._cap * 2, key + 1)
            self._idx.resize_index(self._cap)
        self._idx.add_items(np.asarray(vector, dtype=np.float32).reshape(1, -1),
                            np.asarray([key]))
        self._live.add(key)
        self._added += 1

    def remove(self, key: int) -> None:
        if key in self._live:
            self._idx.mark_deleted(key)
            self._live.discard(key)

    def topk(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Fast approximate top-k (the ANN benefit). Skips deleted keys; ``ip`` distance is
        ``1 - cosine`` for unit vectors, so the score is ``1 - distance``."""
        if not self._live:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        labels, dists = self._idx.knn_query(q, k=min(k, self._added))
        return [(int(lbl), 1.0 - float(d)) for lbl, d in zip(labels[0], dists[0], strict=True)
                if int(lbl) in self._live]

    def rank(self, vector: np.ndarray) -> list[tuple[int, float]]:
        """Full ranking (protocol compatibility, so the tier's filter-then-take-k works)."""
        return self.topk(vector, len(self._live))


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

    def __init__(self, embedder, *, index=None) -> None:
        self.embedder = embedder
        # Default is the exact, byte-deterministic BruteForceIndex (AC-1). Pass a real ANN
        # backend (HnswIndex) for scale — it exposes ``topk``, which `search`/`nearest` use as
        # a sublinear fast path; snapshot/re-embed remain BruteForce features.
        self.index = index if index is not None else BruteForceIndex(embedder.dim)
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

    def remove_by_rid(self, rid: str) -> None:
        for key in [k for k, r in self.meta.items() if r.rid == rid]:
            self.remove(key)

    def search(self, query, k: int = 8, *, t: float | None = None,
               entity: str | None = None, tier: str | None = None,
               window: tuple[float, float] | None = None) -> list[tuple[VectorRecord, float]]:
        """Top-k records by cosine, after applying causal + metadata filters.

        ``t`` enforces causality (only records with ``wall_ts <= t``); ``entity``/``tier``/
        ``window`` are the metadata filters the retrieval planner uses (PRD §7).
        """
        qtext = query if isinstance(query, str) else None
        qv = self.embedder.embed(query) if qtext is not None else np.asarray(query)
        ent = normalize(entity) if entity is not None else None
        # A real ANN backend (HnswIndex) exposes `topk` — over-fetch k*_OVERFETCH candidates
        # (the sublinear fast path); the exact BruteForce default ranks ALL records.
        if hasattr(self.index, "topk"):
            ranked = self.index.topk(qv, k * _OVERFETCH)
        else:
            ranked = self.index.rank(qv)  # [(key, embedding-similarity)]
        if qtext is not None:  # hybrid: re-rank by semantic + lexical BEFORE truncating to k
            ranked = sorted(
                ((key, _W_SEM * cos + _W_LEX * jaccard(qtext, self.meta[key].text))
                 for key, cos in ranked),
                key=lambda kc: (-kc[1], kc[0]))
        out: list[tuple[VectorRecord, float]] = []
        for key, score in ranked:
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

    def nearest(self, embedding: np.ndarray) -> tuple[VectorRecord, float] | None:
        """The single most-similar stored record (for surprise/novelty scoring), or None."""
        emb = np.asarray(embedding, dtype=np.float64)
        ranked = self.index.topk(emb, 1) if hasattr(self.index, "topk") else self.index.rank(emb)
        if not ranked:
            return None
        key, score = ranked[0]
        return self.meta[key], score

    def reembed(self, new_embedder) -> int:
        """Re-embed every stored record's text under ``new_embedder`` (model upgrade,
        PRD §12). The index is rebuilt at the new dimensionality; record metadata and keys
        are preserved (non-lossy — only the vectors change). Returns the count migrated.

        Deterministic: records are visited in insertion order, so a re-embed reproduces a
        byte-identical index for the same inputs and model. Note: the rebuilt index is always
        an exact BruteForceIndex, so re-embedding an opt-in HnswIndex tier demotes it to brute
        force (re-wrap in an HnswIndex afterwards if scale still demands it).
        """
        self.embedder = new_embedder
        new_index = BruteForceIndex(new_embedder.dim)
        for key, rec in self.meta.items():
            new_index.add(key, new_embedder.embed(rec.text))
        self.index = new_index
        return len(self.meta)

    def __len__(self) -> int:
        return len(self.meta)

    # ------------------------------------------------------------------ persistence --
    def to_dict(self) -> dict:
        if not hasattr(self.index, "to_dict"):  # e.g. an opt-in HnswIndex
            raise TypeError(
                f"{type(self.index).__name__} is not serializable; byte-deterministic "
                "snapshots require the default BruteForceIndex (AC-1).")
        return {
            "next": self._next,
            "index": self.index.to_dict(),
            "meta": [[k, vars(r)] for k, r in self.meta.items()],
        }

    def load(self, d: dict) -> None:
        self._next = d["next"]
        self.index = BruteForceIndex.from_dict(d["index"])
        self.meta = {int(k): VectorRecord(**fields) for k, fields in d["meta"]}
