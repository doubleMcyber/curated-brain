"""Seeded synthetic longitudinal dataset generator tests (Stage 1)."""

from __future__ import annotations

from curated_brain.dataset import generate


def test_generation_is_deterministic():
    a = generate(seed=0)
    b = generate(seed=0)
    assert a.observations == b.observations
    assert a.probes == b.probes


def test_different_seed_differs():
    a = generate(seed=0)
    b = generate(seed=1)
    assert a.observations != b.observations


def test_all_categories_present():
    ds = generate(seed=0)
    cats = {p.category for p in ds.probes}
    assert {"C1", "C2", "C3", "C5", "C6"} <= cats


def test_stream_is_redundancy_heavy():
    # Selectivity (AC-5) and bounded growth (AC-6) need a stream where the large majority
    # of lines are non-salient (discardable/reinforceable).
    ds = generate(seed=0)
    salient = sum(1 for o in ds.observations if o.salient)
    frac_salient = salient / len(ds.observations)
    assert frac_salient < 0.20  # >= 80% of the stream is noise


def test_long_range_gap():
    # C1 facts must be injected >= 50 sessions before the final session.
    ds = generate(seed=0)
    intro_session = {}  # subject -> earliest session index that asserts an email fact
    for o in ds.observations:
        if o.fact and o.fact["predicate"] == "email":
            s = int(o.session_id[1:])
            intro_session.setdefault(o.fact["subject"], s)
    for p in ds.by_category("C1"):
        assert (ds.n_sessions - 1) - intro_session[p.subject] >= 50


def test_belief_update_has_distinct_stale_value():
    ds = generate(seed=0)
    for p in ds.by_category("C2"):
        if p.stale is not None:
            assert p.stale != p.gold


def test_temporal_probe_targets_past_interval():
    # The as-of timestamp must fall inside the *pre-update* interval, and the gold value
    # must be the old city (not the current one).
    ds = generate(seed=0)
    people = {pp["name"]: pp for pp in ds.people}
    for p in ds.by_category("C6"):
        person = people[p.subject]
        update_ts = ds.base_ts + person["city_update_session"] * ds.day
        assert p.as_of < update_ts
        assert p.gold == person["init_city"]


def test_multi_hop_probe_present():
    ds = generate(seed=0)
    hops = [p for p in ds.probes if p.hops]
    assert hops
    assert all(p.hops == ["manager", "city"] for p in hops)
