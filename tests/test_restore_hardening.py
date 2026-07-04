"""Track-H security: restore() takes untrusted bytes, so a malformed/hostile snapshot must
fail with a clear ValueError (not an opaque crash) and cannot force a large allocation."""

from __future__ import annotations

import json

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.vector import BruteForceIndex


def _good_blob() -> bytes:
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0,
             metadata={"fact": {"subject": "Erin", "predicate": "city", "object": "Vienna"}})
    return cb.snapshot()


def test_valid_snapshot_still_restores_identically():
    blob = _good_blob()
    cb = CuratedBrain(seed=0)
    cb.restore(blob)
    assert cb.answer_structured("Erin", "city") == "Vienna"
    assert cb.snapshot() == blob  # happy path byte-identical (hardening is no-op when valid)


def test_non_json_blob_raises_clear_error():
    cb = CuratedBrain(seed=0)
    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        cb.restore(b"\xff\xfe not json at all")
    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        cb.restore(b"{ this is : not json ]")


def test_non_object_json_raises():
    cb = CuratedBrain(seed=0)
    with pytest.raises(ValueError, match="must be a JSON object"):
        cb.restore(b"[1, 2, 3]")


def test_missing_counter_raises():
    cb = CuratedBrain(seed=0)
    with pytest.raises(ValueError, match="counter"):
        cb.restore(json.dumps({"episodic": []}).encode())


def test_episodic_with_unknown_field_raises():
    cb = CuratedBrain(seed=0)
    bad = {"counter": 0, "episodic": [{"id": "x", "evil": "arbitrary injected field"}]}
    with pytest.raises(ValueError, match="unknown fields"):
        cb.restore(json.dumps(bad).encode())


def test_episodic_missing_required_field_raises():
    cb = CuratedBrain(seed=0)
    bad = {"counter": 0, "episodic": [{"id": "x"}]}  # missing content/wall_ts/etc.
    with pytest.raises(ValueError, match="missing required fields"):
        cb.restore(json.dumps(bad).encode())


def test_oversized_vector_hex_is_rejected_not_allocated():
    # A hostile blob with a giant hex string must be rejected by the length check BEFORE
    # np.frombuffer allocates it — bounding memory. A valid vector is exactly dim*16 hex chars.
    # dim 4 -> a valid vector is exactly 64 hex chars
    huge = "a" * 10_000_000  # would be ~5 MB allocated if not rejected
    with pytest.raises(ValueError, match="expected 64"):
        BruteForceIndex.from_dict({"dim": 4, "items": [[0, huge]]})
    # a correctly-sized-but-odd payload still validates cleanly
    with pytest.raises(ValueError, match="expected 64"):
        BruteForceIndex.from_dict({"dim": 4, "items": [[0, "abcd"]]})


def test_malformed_index_shape_raises():
    with pytest.raises(ValueError, match="malformed vector index"):
        BruteForceIndex.from_dict({"items": []})  # no dim
    with pytest.raises(ValueError, match="malformed vector index"):
        BruteForceIndex.from_dict({"dim": 4, "items": "not a list"})


def test_structured_facts_with_injected_or_missing_field_raise():
    # StructuredTier.load has the same Fact(**d) splat surface as episodic records.
    cb = CuratedBrain(seed=0)
    inj = {"counter": 0, "episodic": [],
           "structured": [{"id": "f", "evil": "injected"}]}
    with pytest.raises(ValueError, match="unknown fields"):
        cb.restore(json.dumps(inj).encode())
    incomplete = {"counter": 0, "episodic": [], "structured": [{"id": "f"}]}
    with pytest.raises(ValueError, match="missing required fields"):
        cb.restore(json.dumps(incomplete).encode())
