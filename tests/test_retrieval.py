"""Stage 4 — hybrid retrieval gate: AC-2 (long-range recall), AC-3 (belief updating),
AC-4 (retrieval cost). Plus planner + supersede-filter unit tests."""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.baselines import LongContext, NaiveRAG, NoMemory
from curated_brain.dataset import generate
from curated_brain.eval import candidates_for, correct, extract_value
from curated_brain.retrieval import fuse
from curated_brain.vector import VectorRecord


def _feed_all(ds):
    cb, nv, lc, nm = CuratedBrain(seed=0), NaiveRAG(), LongContext(), NoMemory()
    for o in ds.observations:
        for be, facts in ((cb, True), (nv, False), (lc, False), (nm, False)):
            be.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                     metadata={"fact": o.fact} if (facts and o.fact) else None)
    return cb, nv, lc, nm


def _answer(be, ds, probe, ts):
    r = be.query(probe.question, session_id="q", timestamp=ts, k=8)
    return extract_value(r.context, candidates_for(ds, probe), probe.question), r


# --------------------------------------------------------------------------- AC-2 ----
def test_ac2_long_range_recall():
    ds = generate(seed=0)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day
    cb, nv, lc, nm = _feed_all(ds)
    probes = ds.by_category("C1")

    def recall(be):
        return sum(p.gold.lower() in be.query(p.question, session_id="q", timestamp=last,
                                              k=8).context.lower() for p in probes) / len(probes)

    cb_r = recall(be=cb)
    assert cb_r >= 0.90, f"curated long-range recall {cb_r:.3f} < 0.90"
    # strictly beats every baseline (pre-validates AC-9's C1 column)
    assert cb_r > recall(nv)
    assert cb_r > recall(lc)
    assert cb_r > recall(nm)

    # the probed facts really were injected >= 50 sessions before the final session
    intro = {}
    for o in ds.observations:
        if o.fact and o.fact["predicate"] == "email":
            intro.setdefault(o.fact["subject"], int(o.session_id[1:]))
    assert all((ds.n_sessions - 1) - intro[p.subject] >= 50 for p in probes)


# --------------------------------------------------------------------------- AC-3 ----
def test_ac3_belief_updating():
    ds = generate(seed=0)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day
    cb, nv, lc, nm = _feed_all(ds)
    probes = ds.by_category("C2")

    def metrics(be):
        current = stale = 0
        for p in probes:
            a, r = _answer(be, ds, p, last)
            current += correct(a, p.gold)
            if p.stale and p.stale.lower() in r.context.lower():
                stale += 1
        return current / len(probes), stale

    cb_cur, cb_stale = metrics(cb)
    nv_cur, _ = metrics(nv)
    lc_cur, lc_stale = metrics(lc)

    # AC-3 literal thresholds
    assert cb_stale == 0, f"curated surfaced {cb_stale} stale values (must be 0)"
    assert cb_cur >= 0.95, f"curated current-value accuracy {cb_cur:.3f} < 0.95"
    # strictly better than the baselines that don't reconcile
    assert lc_stale > 0  # long-context stuffing genuinely surfaces stale contradictions
    assert cb_cur > nv_cur
    assert cb_cur > lc_cur


# --------------------------------------------------------------------------- AC-4 ----
def test_ac4_retrieval_cost():
    ds = generate(seed=0)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day
    cb, nv, lc, nm = _feed_all(ds)
    probes = ds.by_category("C3")

    def cost_and_acc(be):
        toks = acc = 0
        for p in probes:
            a, r = _answer(be, ds, p, last)
            toks += r.tokens_in
            acc += correct(a, p.gold)
        return toks / len(probes), acc / len(probes)

    cb_tok, cb_acc = cost_and_acc(cb)
    lc_tok, lc_acc = cost_and_acc(lc)

    assert cb_tok <= 0.25 * lc_tok, f"curated tokens {cb_tok:.1f} vs long-context {lc_tok:.1f}"
    assert cb_acc >= lc_acc, f"curated accuracy {cb_acc:.3f} < long-context {lc_acc:.3f}"


# ------------------------------------------------------------------- planner units ---
def test_planner_routes_exact_relational_temporal():
    ds = generate(seed=0)
    cb, *_ = _feed_all(ds)
    plan = cb.planner.plan("What is Alice's email address?",
                           entities=cb._entities, session_ts=cb._session_ts)
    assert plan.entity == "alice" and plan.predicate == "email" and not plan.open_ended

    hop = cb.planner.plan("What city does Alice's manager live in?",
                          entities=cb._entities, session_ts=cb._session_ts)
    assert hop.hops == ["manager", "city"] and hop.entity == "alice"

    temporal = cb.planner.plan("Where did Alice live as of session 29?",
                               entities=cb._entities, session_ts=cb._session_ts)
    assert temporal.predicate == "city" and temporal.as_of is not None

    nonsense = cb.planner.plan("Tell me something interesting.",
                               entities=cb._entities, session_ts=cb._session_ts)
    assert nonsense.open_ended


# ----------------------------------------------------- open-domain structured backstop ---
def _brain_with_facts():
    cb = CuratedBrain(seed=0)
    for pred, obj in (("city", "Vienna"), ("role", "engineer"), ("email", "erin@x.com")):
        cb.write(f"Erin {pred} {obj}.", session_id="s0", timestamp=0.0,
                 metadata={"fact": {"subject": "Erin", "predicate": pred, "object": obj}})
    return cb


def test_open_domain_query_still_consults_structured_tier():
    # A question that names a known entity but matches NO predicate keyword is open_ended;
    # the planner would otherwise bypass the structured tier and degrade to vector-only.
    cb = _brain_with_facts()
    plan = cb.planner.plan("Tell me about Erin.", entities=cb._entities,
                           session_ts=cb._session_ts)
    assert plan.open_ended and plan.entity == "erin"  # precondition: the bypass case

    ctx = cb.query("Tell me about Erin.", session_id="q", timestamp=1.0).context.lower()
    # the backstop surfaces Erin's high-precision facts instead of returning nothing useful
    assert "vienna" in ctx and "engineer" in ctx


def test_backstop_reserves_room_for_vector_recall():
    # With more structured facts than the budget, the backstop must not crowd out the
    # vector slot entirely (it caps at MAX_CONTEXT_ITEMS - 1 structured lines).
    cb = _brain_with_facts()
    r = cb.query("What about Erin?", session_id="q", timestamp=1.0)
    structured_lines = sum(1 for ln in r.context.splitlines() if "Erin's" in ln)
    assert structured_lines <= 3  # MAX_CONTEXT_ITEMS (4) - 1 reserved for vector


def test_backstop_inert_without_a_known_entity():
    # No recognized entity -> the structured tier is correctly not consulted.
    cb = _brain_with_facts()
    plan = cb.planner.plan("Tell me something interesting.", entities=cb._entities,
                           session_ts=cb._session_ts)
    assert plan.open_ended and plan.entity is None
    ctx = cb.query("Tell me something interesting.", session_id="q", timestamp=1.0).context
    assert "Erin's" not in ctx  # no entity -> no structured backstop lines


def test_fuse_drops_superseded_values():
    hits = [
        (VectorRecord(rid="1", text="Alice lives in Berlin.", wall_ts=1.0, session_id="s"), 0.9),
        (VectorRecord(rid="2", text="Alice lives in Munich.", wall_ts=2.0, session_id="s"), 0.8),
    ]
    out = fuse(hits, now=10.0, stale_objs={"berlin"})
    assert [it.rid for it in out] == ["2"]  # the superseded "Berlin" record is filtered out
