"""Track-G cost: token→$ pricing turns CB's metered token counts into a dollar figure so the
'≤ its cost' comparison can be stated in dollars, not just tokens/wall-time."""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.pricing import Pricing


def _fact(s, p, o):
    return {"fact": {"subject": s, "predicate": p, "object": o}}


def test_pricing_estimate_is_deterministic_and_correct():
    p = Pricing(embed_usd_per_1k=0.02, context_usd_per_1k=0.5, extract_usd_per_call=0.001)
    cost = {"embed_tokens": 2000, "context_tokens_served": 100, "extract_calls": 3}
    # 2000/1000*0.02 + 100/1000*0.5 + 3*0.001 = 0.04 + 0.05 + 0.003
    assert abs(p.estimate(cost) - 0.093) < 1e-9


def test_default_pricing_is_zero():
    assert Pricing().estimate({"embed_tokens": 10_000, "context_tokens_served": 5000}) == 0.0


def test_metrics_reports_usd_only_when_pricing_configured():
    # no pricing -> cost block is token-only (unchanged shape)
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0, metadata=_fact("Erin", "city", "Vienna"))
    cb.query("Where does Erin live?", session_id="q", timestamp=1.0)
    assert "estimated_usd" not in cb.metrics()["cost"]

    # with pricing -> estimated_usd + usd_per_query appear and are consistent
    cb2 = CuratedBrain(seed=0, pricing=Pricing(embed_usd_per_1k=0.01, context_usd_per_1k=0.1))
    cb2.write("x", session_id="s", timestamp=0.0, metadata=_fact("Erin", "city", "Vienna"))
    cb2.query("Where does Erin live?", session_id="q", timestamp=1.0)
    m = cb2.metrics()["cost"]
    assert m["estimated_usd"] >= 0.0
    assert abs(m["usd_per_query"] - m["estimated_usd"] / m["queries"]) < 1e-12


def test_empty_state_costs_nothing():
    p = Pricing(embed_usd_per_1k=1.0, context_usd_per_1k=1.0)
    cb = CuratedBrain(seed=0, pricing=p)  # no writes, no queries
    m = cb.metrics()["cost"]
    assert m["estimated_usd"] == 0.0
    assert m["usd_per_query"] == 0.0  # guarded against div-by-zero


def test_a_gate_rejected_write_still_costs_the_embedding():
    # Honest semantics: cost reflects work PERFORMED, not the storage outcome. CB must embed
    # an observation to score its novelty, so that embedding is charged even when the gate
    # then discards/reinforces (no new row). The pricing layer faithfully mirrors that meter.
    p = Pricing(embed_usd_per_1k=1.0)
    cb = CuratedBrain(seed=0, pricing=p)
    cb.write("Erin lives in Vienna.", session_id="s", timestamp=0.0)
    first = cb.metrics()["cost"]["estimated_usd"]
    r = cb.write("Erin lives in Vienna.", session_id="s", timestamp=1.0)  # verbatim -> not stored
    assert r.stored is False
    assert cb.metrics()["cost"]["estimated_usd"] > first  # the embed was still charged
