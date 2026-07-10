"""Frozen tuning config (``CBConfig``). Gate: default config is a behavioral no-op
(byte-identical snapshots + identical answers); non-default knobs demonstrably change
behavior; the config object is immutable."""

from __future__ import annotations

import dataclasses

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.config import CBConfig
from curated_brain.dataset import generate


def _feed(cb, observations):
    for o in observations:
        cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                 metadata={"fact": o.fact} if o.fact else None)


# (a) A default CBConfig() reproduces the no-config store byte-for-byte over a real workload.
def test_default_config_is_a_behavioral_noop():
    ds = generate(seed=0)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day

    plain = CuratedBrain(seed=0)
    configured = CuratedBrain(seed=0, config=CBConfig())
    _feed(plain, ds.observations)
    _feed(configured, ds.observations)

    assert configured.snapshot() == plain.snapshot()
    probes = (ds.by_category("C1")[:5] + ds.by_category("C2")[:5] + ds.by_category("C6")[:5])
    for p in probes:
        assert (configured.query(p.question, session_id="q", timestamp=last).context
                == plain.query(p.question, session_id="q", timestamp=last).context)

    # A default config also matches the store built with no config argument at all.
    configured.consolidate()
    plain.consolidate()
    assert configured.snapshot() == plain.snapshot()


# (b1) A smaller max_context_items shrinks the served context.
def test_max_context_items_shrinks_context():
    ds = generate(seed=0)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day
    probe = ds.by_category("C1")[0]

    wide = CuratedBrain(seed=0, config=CBConfig(max_context_items=4))
    narrow = CuratedBrain(seed=0, config=CBConfig(max_context_items=2))
    _feed(wide, ds.observations)
    _feed(narrow, ds.observations)

    wide_lines = wide.query(probe.question, session_id="q", timestamp=last).context.splitlines()
    narrow_lines = narrow.query(probe.question, session_id="q",
                                timestamp=last).context.splitlines()
    assert len(narrow_lines) <= 2
    assert len(narrow_lines) < len(wide_lines)


# (b2) A different free-text dedup threshold changes what consolidation merges.
def test_free_dedup_threshold_changes_consolidation():
    ds = generate(seed=0)

    default = CuratedBrain(seed=0)  # free_dedup_threshold = 0.85
    loose = CuratedBrain(seed=0, config=CBConfig(free_dedup_threshold=0.1))
    _feed(default, ds.observations)
    _feed(loose, ds.observations)

    default_report = default.consolidate()
    loose_report = loose.consolidate()
    # A near-zero threshold collapses far more free-text episodes into merged claims.
    assert loose_report.dupes_merged > default_report.dupes_merged


# (c) CBConfig is frozen: assigning to a field raises.
def test_cbconfig_is_frozen():
    cfg = CBConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_context_items = 99  # type: ignore[misc]


# The gate ctor arg still takes precedence over the config's gate knobs.
def test_explicit_gate_wins_over_config():
    from curated_brain.surprise import SurpriseGate

    cfg = CBConfig(budget=0.9)
    cb = CuratedBrain(seed=0, gate=SurpriseGate(budget=0.11), config=cfg)
    assert cb.gate.budget == 0.11
