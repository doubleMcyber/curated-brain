"""Observability metrics (Track G).

The layer exposes its write-decision breakdown and store size so an operator can see
selectivity (Pillar B) and bounded growth at a glance — without reaching into internals.
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.dataset import redundant_stream


def _run(seed: int = 0):
    obs, facts = redundant_stream(seed=seed)
    cb = CuratedBrain(seed=seed)
    for o in obs:
        cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                 metadata={"fact": o.fact} if o.fact else None)
    return cb, obs, facts


def test_metrics_track_decisions_size_and_selectivity():
    cb, obs, facts = _run()
    m = cb.metrics()
    # complete, consistent decision breakdown
    assert m["writes_total"] == len(obs)
    assert m["stored"] + m["reinforced"] + m["discarded"] == len(obs)
    assert m["discard_rate"] == m["discarded"] / len(obs)
    # genuinely selective on a redundant stream (Pillar B) — not logging everything
    assert m["discard_rate"] > 0.3
    # store size reflects exactly the stored episodes, yet every distinct fact survives
    assert m["store_size"] == m["stored"]
    assert m["structured_facts"] == len(facts)


def test_cost_metrics_count_model_calls_and_tokens():
    cb, obs, _ = _run()
    c = cb.metrics()["cost"]
    # every write embeds exactly once; no extractor configured -> no extract calls
    assert c["embed_calls"] == len(obs)
    assert c["extract_calls"] == 0
    assert c["embed_tokens"] > 0
    # no queries issued yet
    assert c["queries"] == 0 and c["context_tokens_served"] == 0
    assert c["avg_context_tokens"] == 0.0

    # query-side cost accrues: one question-embed per query, context tokens summed, avg consistent
    before = cb.metrics()["cost"]["embed_calls"]
    served = 0
    for q in ("Alice city", "Bob role", "Carol email"):
        served += cb.query(q, session_id="q", timestamp=2e9).tokens_in
    c = cb.metrics()["cost"]
    assert c["queries"] == 3
    assert c["embed_calls"] == before + 3
    assert c["context_tokens_served"] == served
    assert c["avg_context_tokens"] == served / 3


def test_metrics_empty_store_is_safe():
    m = CuratedBrain(seed=0).metrics()
    assert m["writes_total"] == 0
    assert m["discard_rate"] == 0.0
    assert m["store_size"] == 0


def test_metrics_are_deterministic():
    assert _run()[0].metrics() == _run()[0].metrics()


def test_metrics_reset_on_restore_but_store_survives():
    cb, _, _ = _run()
    assert cb.metrics()["writes_total"] > 0
    restored = CuratedBrain(seed=0)
    restored.restore(cb.snapshot())
    m = restored.metrics()
    # operational counters describe this instance's activity (none) — but the store is intact
    assert m["writes_total"] == 0
    assert m["cost"]["embed_calls"] == 0 and m["cost"]["queries"] == 0
    assert m["store_size"] == cb.metrics()["store_size"]
    assert m["structured_facts"] == cb.metrics()["structured_facts"]
