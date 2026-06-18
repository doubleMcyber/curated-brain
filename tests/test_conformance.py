"""AC-1 — adapter conformance + determinism (PRD §10, Stage 1).

All ``MemoryBackend`` methods implemented; ``snapshot -> restore`` is byte-deterministic;
identical input + seed produces identical outputs.
"""

from __future__ import annotations

import abc

import pytest

from curated_brain.backend import CuratedBrain, MemoryBackend
from curated_brain.dataset import generate
from curated_brain.models import Retrieval, WriteReceipt


def _feed(backend: MemoryBackend, n: int = 80):
    ds = generate(seed=0)
    for obs in ds.observations[:n]:
        backend.write(obs.content, session_id=obs.session_id, timestamp=obs.wall_ts,
                      metadata={"fact": obs.fact} if obs.fact else None)
    return ds


def test_backend_is_abstract():
    assert issubclass(MemoryBackend, abc.ABC)
    with pytest.raises(TypeError):
        MemoryBackend()  # cannot instantiate the ABC


def test_all_methods_present():
    cb = CuratedBrain(seed=0)
    for name in ("write", "query", "consolidate", "stats", "reset", "snapshot", "restore"):
        assert callable(getattr(cb, name))


def test_write_receipt_shape():
    cb = CuratedBrain(seed=0)
    r = cb.write("Alice lives in Berlin.", session_id="s000", timestamp=1.0)
    assert isinstance(r, WriteReceipt)
    assert r.reason in {"stored", "reinforced", "discarded"}
    assert isinstance(r.stored, bool)


def test_query_shape():
    cb = CuratedBrain(seed=0)
    _feed(cb, 40)
    r = cb.query("Where does Alice live?", session_id="s100", timestamp=10_000.0, k=5)
    assert isinstance(r, Retrieval)
    assert r.tokens_in >= 0
    assert isinstance(r.citations, list)


def test_snapshot_restore_byte_deterministic():
    cb = CuratedBrain(seed=0)
    _feed(cb, 60)
    snap1 = cb.snapshot()
    cb.restore(snap1)
    snap2 = cb.snapshot()
    assert snap1 == snap2  # restore reproduces identical state


def test_identical_input_and_seed_identical_output():
    a, b = CuratedBrain(seed=0), CuratedBrain(seed=0)
    _feed(a, 60)
    _feed(b, 60)
    assert a.snapshot() == b.snapshot()
    qa = a.query("What is Alice's email address?", session_id="s200", timestamp=9e6, k=5)
    qb = b.query("What is Alice's email address?", session_id="s200", timestamp=9e6, k=5)
    assert qa == qb


def test_reset_clears_state():
    cb = CuratedBrain(seed=0)
    _feed(cb, 30)
    assert cb.stats().episodic_count > 0
    cb.reset()
    assert cb.stats().episodic_count == 0


def test_restore_into_fresh_backend_matches():
    src = CuratedBrain(seed=0)
    _feed(src, 50)
    blob = src.snapshot()
    dst = CuratedBrain(seed=0)
    dst.restore(blob)
    assert dst.snapshot() == blob
