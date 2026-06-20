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
