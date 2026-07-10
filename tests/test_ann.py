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


# --------------------------------------------------------------------------------------
# Rebuild-on-load: an Hnsw-backed tier remembers its index type and rebuilds AS ITSELF
# (an HnswIndex), plus the opt-in on-disk sidecar fast path. Regression cover for the old
# silent demotion to BruteForce on restore/reembed.
# --------------------------------------------------------------------------------------

def _hnsw_tier(emb, n_cap=64):
    from curated_brain.vector import HnswIndex, VectorTier

    return VectorTier(emb, index=HnswIndex(emb.dim, max_elements=n_cap))


def test_restore_preserves_hnsw_index_type():
    # A tier constructed with an HnswIndex serializes records-only and rebuilds a FRESH
    # HnswIndex on load (bare `index=` infers the rebuild factory) — no demotion.
    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import HnswIndex, VectorTier

    emb = DeterministicEmbedder(64)
    src = _hnsw_tier(emb)
    for i in range(60):
        src.add(rid=f"r{i}", text=f"note {i} topic-{i}", wall_ts=float(i), session_id="s")

    blob = src.to_dict()
    assert "index" not in blob  # ANN vectors are derived state, not serialized

    restored = VectorTier(emb, index_factory=lambda dim: HnswIndex(dim))
    restored.load(blob)
    assert type(restored.index) is HnswIndex
    assert len(restored) == len(src)


def test_hnsw_rebuild_topk_matches_brute_force_ground_truth():
    # Post-restore ANN recall stays high: rebuilt-HnswIndex top-10 vs exact brute-force
    # ground truth over ~500 deterministic vectors, recall@10 >= 0.9.
    from curated_brain.vector import HnswIndex, VectorTier

    n, dim = 500, 64
    vecs = _unit_vectors(n, dim, seed=11)
    queries = _unit_vectors(30, dim, seed=12)
    emb = _PreEmbedded(dim)

    src = VectorTier(emb, index=HnswIndex(dim, max_elements=n))
    truth = VectorTier(emb)
    for i in range(n):
        src.add(rid=f"r{i}", text=f"t{i}", wall_ts=0.0, session_id="s", embedding=vecs[i])
        truth.add(rid=f"r{i}", text=f"t{i}", wall_ts=0.0, session_id="s", embedding=vecs[i])

    hits = total = 0
    for qi in range(queries.shape[0]):
        exact = {r.rid for r, _ in truth.search(queries[qi], k=10)}
        approx = {r.rid for r, _ in src.search(queries[qi], k=10)}
        hits += len(exact & approx)
        total += len(exact)
    assert hits / total >= 0.9, f"rebuilt-ANN recall@10 {hits / total:.3f} < 0.9"


def test_hnsw_rebuild_is_deterministic():
    # Same records -> same query results across two independent rebuilds (fixed seed, single
    # thread, insertion order = record order).
    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import HnswIndex, VectorTier

    emb = DeterministicEmbedder(64)
    src = _hnsw_tier(emb)
    for i in range(200):
        src.add(rid=f"r{i}", text=f"note {i} about topic-{i}", wall_ts=float(i), session_id="s")
    blob = src.to_dict()

    r1 = VectorTier(emb, index_factory=lambda dim: HnswIndex(dim))
    r2 = VectorTier(emb, index_factory=lambda dim: HnswIndex(dim))
    r1.load(blob)
    r2.load(blob)
    for i in range(30):
        assert r1.search(f"topic-{i}", k=5) == r2.search(f"topic-{i}", k=5)


def test_hnsw_sidecar_save_load_roundtrip(tmp_path):
    # The on-disk sidecar (native hnswlib save_index/load_index) roundtrips: a sidecar-loaded
    # tier answers identically to one rebuilt from text — proving the fast path is equivalent.
    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import HnswIndex, VectorTier

    emb = DeterministicEmbedder(64)
    side = str(tmp_path / "ann.bin")
    src = VectorTier(emb, index=HnswIndex(64, max_elements=64), ann_path=side)
    for i in range(80):
        src.add(rid=f"r{i}", text=f"note {i} topic-{i}", wall_ts=float(i), session_id="s")
    src.save_sidecar()
    assert (tmp_path / "ann.bin").exists()

    blob = src.to_dict()
    from_side = VectorTier(emb, index_factory=lambda dim: HnswIndex(dim), ann_path=side)
    from_side.load(blob)
    assert type(from_side.index) is HnswIndex
    assert from_side.index.count == len(src)

    rebuilt = VectorTier(emb, index_factory=lambda dim: HnswIndex(dim))
    rebuilt.load(blob)
    for i in range(30):
        assert from_side.search(f"topic-{i}", k=5) == rebuilt.search(f"topic-{i}", k=5)


def test_hnsw_sidecar_corruption_warns_and_rebuilds(tmp_path, caplog):
    # A corrupt/foreign sidecar must NOT crash and must NOT be trusted: WARN (loud) and fall
    # back to the deterministic rebuild.
    import logging

    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import HnswIndex, VectorTier

    emb = DeterministicEmbedder(64)
    side = str(tmp_path / "ann.bin")
    src = VectorTier(emb, index=HnswIndex(64, max_elements=64), ann_path=side)
    for i in range(50):
        src.add(rid=f"r{i}", text=f"note {i} topic-{i}", wall_ts=float(i), session_id="s")
    blob = src.to_dict()
    (tmp_path / "ann.bin").write_bytes(b"not a real hnsw index")

    loaded = VectorTier(emb, index_factory=lambda dim: HnswIndex(dim), ann_path=side)
    with caplog.at_level(logging.WARNING, logger="curated_brain"):
        loaded.load(blob)  # must not raise
    assert any("sidecar" in r.message and "rebuild" in r.message.lower()
               for r in caplog.records), "corrupt sidecar must WARN"
    assert type(loaded.index) is HnswIndex  # rebuilt as itself, not demoted
    assert len(loaded) == len(src)
    assert loaded.search("topic-3", k=3)  # queries still work


def test_curated_brain_restore_preserves_hnsw(tmp_path):
    # End-to-end: a CuratedBrain wired with an HnswIndex factory keeps HNSW across a full
    # snapshot -> restore, and save()/load() persists+reloads the sidecar.
    from curated_brain.backend import CuratedBrain
    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import HnswIndex

    emb = DeterministicEmbedder(64)
    side = str(tmp_path / "brain.ann")
    cb = CuratedBrain(embedder=emb, seed=0,
                      index_factory=lambda dim: HnswIndex(dim, max_elements=64),
                      ann_path=side)
    for i in range(40):
        cb.write(f"note {i} topic-{i}", session_id="s1", timestamp=float(i))

    blob = cb.snapshot()
    cb2 = CuratedBrain(embedder=emb, seed=0,
                       index_factory=lambda dim: HnswIndex(dim, max_elements=64),
                       ann_path=side)
    cb2.restore(blob)
    assert type(cb2.vector.index) is HnswIndex

    # save() persists the sidecar; a fresh brain load()s it as the fast path
    path = str(tmp_path / "brain.snap")
    cb.save(path)
    assert (tmp_path / "brain.ann").exists()
    cb3 = CuratedBrain(embedder=emb, seed=0,
                       index_factory=lambda dim: HnswIndex(dim, max_elements=64),
                       ann_path=side)
    cb3.load(path)
    assert type(cb3.vector.index) is HnswIndex
    assert cb3.vector.index.count == len(cb3.vector)


def test_hnsw_reembed_stays_hnsw():
    # reembed() rebuilds via the tier factory, so an Hnsw-backed tier stays HNSW instead of
    # demoting to BruteForce (the old behavior).
    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import HnswIndex

    emb = DeterministicEmbedder(64)
    src = _hnsw_tier(emb)
    for i in range(40):
        src.add(rid=f"r{i}", text=f"note {i} topic-{i}", wall_ts=float(i), session_id="s")
    src.reembed(DeterministicEmbedder(64))
    assert type(src.index) is HnswIndex


def test_default_bruteforce_snapshot_bytes_unchanged():
    # The records-only representation must ONLY kick in for non-serializable indexes: the
    # default BruteForceIndex tier's snapshot bytes stay byte-identical (it still serializes
    # the exact vectors under "index"). Pin computed from the pre-change tree.
    import hashlib
    import json

    from curated_brain.fakes import DeterministicEmbedder
    from curated_brain.vector import VectorTier

    emb = DeterministicEmbedder(64)
    t = VectorTier(emb)  # default: exact BruteForceIndex
    for i in range(20):
        t.add(rid=f"r{i}", text=f"note {i}", wall_ts=float(i), session_id="s")
    d = t.to_dict()
    assert "index" in d  # default path still serializes exact vectors
    b = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(b).hexdigest() == \
        "378b85ac34f42601ac58a2fde78330b273b2bbdcd32715075b911c98a4122030"
