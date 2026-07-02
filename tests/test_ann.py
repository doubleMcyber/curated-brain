"""Optional real ANN backend (hnswlib) — Track C scale.

Skipped when hnswlib isn't installed (the deterministic BruteForceIndex is the default and
needs no extra). When present, this proves the approximate index conforms to the VectorIndex
protocol, agrees with exact brute force at high recall, and handles deletion — so it can be
swapped in where corpus size makes O(n) brute force too slow.
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest

from curated_brain.vector import BruteForceIndex, VectorIndex

HAVE_HNSW = importlib.util.find_spec("hnswlib") is not None
pytestmark = pytest.mark.skipif(not HAVE_HNSW, reason="hnswlib not installed ([scale] extra)")


def _unit_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_hnsw_conforms_and_matches_brute_force_at_scale():
    from curated_brain.vector import HnswIndex

    n, dim, q = 5000, 64, 40
    vecs = _unit_vectors(n, dim, seed=1)
    queries = _unit_vectors(q, dim, seed=2)

    brute = BruteForceIndex(dim)
    hnsw = HnswIndex(dim, max_elements=n)
    assert isinstance(hnsw, VectorIndex)  # satisfies the index seam
    for key in range(n):
        brute.add(key, vecs[key])
        hnsw.add(key, vecs[key])

    # recall@10: fraction of brute-force's true top-10 that the ANN also returns
    hits = total = 0
    t_brute = t_hnsw = 0.0
    for qi in range(q):
        t0 = time.perf_counter()
        exact = {k for k, _ in brute.rank(queries[qi])[:10]}
        t_brute += time.perf_counter() - t0
        t0 = time.perf_counter()
        approx = {k for k, _ in hnsw.topk(queries[qi], 10)}
        t_hnsw += time.perf_counter() - t0
        hits += len(exact & approx)
        total += len(exact)
    recall = hits / total
    assert recall >= 0.85, f"ANN recall@10 {recall:.3f} < 0.85"
    # the ANN top-k is the whole point: sublinear vs brute force's full scan at this size
    assert t_hnsw < t_brute, f"hnsw {t_hnsw:.3f}s not faster than brute {t_brute:.3f}s"


def test_hnsw_topk_and_deletion():
    from curated_brain.vector import HnswIndex

    dim = 16
    vecs = _unit_vectors(50, dim, seed=3)
    idx = HnswIndex(dim, max_elements=50)
    for k in range(50):
        idx.add(k, vecs[k])
    assert len(idx.topk(vecs[0], 5)) == 5
    assert idx.topk(vecs[0], 1)[0][0] == 0  # a vector is its own nearest neighbor
    idx.remove(0)
    assert all(key != 0 for key, _ in idx.topk(vecs[0], 5))  # deleted key never returned


class _PreEmbedded:
    """Embedder stub for the tier test: records are added with explicit ``embedding=`` so the
    tier never calls ``embed`` for storage. ``dim`` is all the tier needs at construction."""

    def __init__(self, dim: int) -> None:
        self.dim = dim


def test_vector_tier_ann_backend_matches_brute_force_at_scale():
    # The optional HnswIndex backend drops into VectorTier: its `search` over-fetches via the
    # index `topk` fast path and agrees with the exact brute-force tier at high recall, while
    # being faster at scale. Query by vector to isolate the semantic path (no lexical hybrid).
    from curated_brain.vector import HnswIndex, VectorTier

    n, dim, q = 5000, 64, 40
    vecs = _unit_vectors(n, dim, seed=1)
    queries = _unit_vectors(q, dim, seed=2)
    emb = _PreEmbedded(dim)
    bf = VectorTier(emb)
    ann = VectorTier(emb, index=HnswIndex(dim, max_elements=n))
    assert type(bf.index).__name__ == "BruteForceIndex"  # default is the exact, deterministic index
    for i in range(n):
        bf.add(rid=f"r{i}", text=f"r{i}", wall_ts=0.0, session_id="s", embedding=vecs[i])
        ann.add(rid=f"r{i}", text=f"r{i}", wall_ts=0.0, session_id="s", embedding=vecs[i])

    hits = total = 0
    t_bf = t_ann = 0.0
    for qi in range(q):
        t0 = time.perf_counter()
        exact = {r.rid for r, _ in bf.search(queries[qi], k=10)}
        t_bf += time.perf_counter() - t0
        t0 = time.perf_counter()
        approx = {r.rid for r, _ in ann.search(queries[qi], k=10)}
        t_ann += time.perf_counter() - t0
        hits += len(exact & approx)
        total += len(exact)
    assert hits / total >= 0.85, f"ANN-tier recall@10 {hits / total:.3f} < 0.85"
    assert t_ann < t_bf, f"ann {t_ann:.3f}s not faster than brute {t_bf:.3f}s"
    # the metadata filters still apply on top of the ANN fast path
    assert ann.search(queries[0], k=10, tier="semantic") == []  # all records are episodic
    # snapshotting an opt-in HnswIndex tier WORKS (records-only payload; previously it
    # raised, so stats()/save() crashed the moment the production index was plugged in)
    blob = ann.to_dict()
    assert "index" not in blob  # ANN vectors are not serialized — texts re-embed on load


def test_hnsw_tier_snapshot_roundtrip_demotes_to_exact():
    # Full round-trip with a real embedder: an Hnsw-backed tier serializes records-only and
    # load() rebuilds vectors from text on an exact BruteForceIndex (documented demotion) —
    # the restored tier answers identically to an exact tier built from the same texts.
    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import HnswIndex, VectorTier

    emb = DeterministicEmbedder(64)
    texts = [f"note {i} about topic-{i}" for i in range(50)]
    ann = VectorTier(emb, index=HnswIndex(64, max_elements=64))
    exact = VectorTier(emb)
    for i, t in enumerate(texts):
        ann.add(rid=f"r{i}", text=t, wall_ts=float(i), session_id="s")
        exact.add(rid=f"r{i}", text=t, wall_ts=float(i), session_id="s")

    restored = VectorTier(emb)
    restored.load(ann.to_dict())
    assert type(restored.index).__name__ == "BruteForceIndex"
    assert len(restored) == len(ann)
    assert [(r.rid, round(s, 12)) for r, s in restored.search("topic-7", k=5)] == \
        [(r.rid, round(s, 12)) for r, s in exact.search("topic-7", k=5)]
