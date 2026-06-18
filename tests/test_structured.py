"""Structured-tier unit tests (Stage 2): bi-temporal model, supersede, multi-hop."""

from __future__ import annotations

from curated_brain.models import INF
from curated_brain.structured import StructuredTier


def _tier_with_city_history() -> StructuredTier:
    t = StructuredTier()
    t.assert_fact(fact_id="f1", subject="Alice", predicate="city", object="Berlin",
                  valid_from=100.0, created_at=100.0, provenance={"episode_id": "e1"})
    t.assert_fact(fact_id="f2", subject="Alice", predicate="city", object="Munich",
                  valid_from=200.0, created_at=200.0, provenance={"episode_id": "e2"})
    return t


def test_current_returns_latest_open_fact():
    t = _tier_with_city_history()
    cur = t.current("Alice", "city")
    assert cur is not None and cur.object == "Munich"
    assert cur.is_open


def test_supersede_is_non_lossy():
    t = _tier_with_city_history()
    hist = t.history("Alice", "city")
    assert [f.object for f in hist] == ["Berlin", "Munich"]
    old = hist[0]
    # old fact is closed in BOTH valid and transaction time and linked to its replacement
    assert old.valid_to == 200.0
    assert old.expired_at == 200.0
    assert old.superseded_by == "f2"
    # provenance is retained for the superseded record (AC-7 foundation)
    assert old.provenance == {"episode_id": "e1"}
    assert not old.is_open


def test_as_of_is_bitemporal():
    t = _tier_with_city_history()
    assert t.as_of("Alice", "city", 150.0).object == "Berlin"   # before the move
    assert t.as_of("Alice", "city", 250.0).object == "Munich"   # after the move
    assert t.as_of("Alice", "city", 50.0) is None               # before anything known


def test_reassert_same_value_creates_no_row():
    t = _tier_with_city_history()
    before = len(t.facts)
    rec, superseded = t.assert_fact(fact_id="f3", subject="Alice", predicate="city",
                                    object="Munich", valid_from=300.0, created_at=300.0)
    assert superseded is None
    assert rec.id == "f2"  # the existing open fact is returned unchanged
    assert len(t.facts) == before


def test_open_fact_has_infinite_intervals():
    t = _tier_with_city_history()
    cur = t.current("Alice", "city")
    assert cur.valid_to == INF and cur.expired_at == INF


def test_multi_hop_resolution():
    t = StructuredTier()
    t.assert_fact(fact_id="m1", subject="Alice", predicate="manager", object="Bob",
                  valid_from=1.0, created_at=1.0)
    t.assert_fact(fact_id="c1", subject="Bob", predicate="city", object="Lisbon",
                  valid_from=1.0, created_at=1.0)
    hop = t.resolve_path("Alice", ["manager", "city"])
    assert hop is not None and hop.object == "Lisbon"
    # broken chain returns None rather than raising
    assert t.resolve_path("Alice", ["manager", "email"]) is None


def test_entity_matching_is_normalized():
    t = StructuredTier()
    t.assert_fact(fact_id="x", subject="Alice", predicate="city", object="Berlin",
                  valid_from=1.0, created_at=1.0)
    assert t.current("  alice ", "CITY").object == "Berlin"
