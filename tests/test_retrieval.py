"""Stage 4 — hybrid retrieval gate: AC-2 (long-range recall), AC-3 (belief updating),
AC-4 (retrieval cost). Plus planner + supersede-filter unit tests."""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.baselines import LongContext, NaiveRAG, NoMemory
from curated_brain.config import CBConfig
from curated_brain.dataset import generate
from curated_brain.eval import candidates_for, correct, extract_value
from curated_brain.retrieval import Planner, fuse
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


def test_schema_driven_planner_routes_an_unhardcoded_predicate():
    # A predicate the planner never hardcoded (not in PRED_KEYWORDS), recognized purely
    # because it was STORED and its name appears in the question -> precise routing to the
    # structured tier, not the dump-everything backstop.
    cb = CuratedBrain(seed=0)
    cb.write("Erin's hobby is painting.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Erin", "predicate": "hobby", "object": "painting"}})

    # Without the stored-predicate vocab the planner cannot route "hobby" -> open_ended.
    bare = cb.planner.plan("What is Erin's hobby?", entities=cb._entities,
                           session_ts=cb._session_ts)
    assert bare.open_ended

    # With it, the planner routes precisely.
    plan = cb.planner.plan("What is Erin's hobby?", entities=cb._entities,
                           predicates=frozenset({"hobby"}), session_ts=cb._session_ts)
    assert plan.entity == "erin" and plan.predicate == "hobby" and not plan.open_ended

    # End-to-end: query() derives the vocab from the store and surfaces the exact value.
    ctx = cb.query("What is Erin's hobby?", session_id="q", timestamp=1.0).context.lower()
    assert "painting" in ctx


def test_backstop_surfaces_facts_for_every_named_entity():
    # A question naming TWO known entities (no predicate keyword) must surface facts for
    # BOTH, not just the first — the lever for multi-entity / chained-relation questions.
    cb = CuratedBrain(seed=0)
    cb.write("Quinn works at Umbrella.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Quinn", "predicate": "employer", "object": "Umbrella"}})
    cb.write("Umbrella is headquartered in Cairo.", session_id="s1", timestamp=1.0,
             metadata={"fact": {"subject": "Umbrella", "predicate": "headquarters",
                                "object": "Cairo"}})
    # "works" mis-keywords the plan to role (which Quinn lacks); the no-resolve fallback must
    # still surface BOTH named entities' facts rather than dropping to vector-only.
    ctx = cb.query("Where is Umbrella, the company Quinn works for, headquartered?",
                   session_id="q", timestamp=2.0).context
    assert "Cairo" in ctx                          # Umbrella's HQ fact surfaced
    assert "Quinn" in ctx and "Umbrella" in ctx    # both entities surfaced


def test_backstop_inert_without_a_known_entity():
    # No recognized entity -> the structured tier is correctly not consulted.
    cb = _brain_with_facts()
    plan = cb.planner.plan("Tell me something interesting.", entities=cb._entities,
                           session_ts=cb._session_ts)
    assert plan.open_ended and plan.entity is None
    ctx = cb.query("Tell me something interesting.", session_id="q", timestamp=1.0).context
    assert "Erin's" not in ctx  # no entity -> no structured backstop lines


def test_multihop_query_cites_every_hop_fact():
    # Multi-hop retrieval must surface the provenance of EVERY fact in the chain (not just
    # the final answer), so the whole support set is attributable. Regression: the prior
    # code cited only the last hop -> downstream recall over support sets was incomplete.
    cb = CuratedBrain(seed=0)
    cb.write("Alice's manager is Bob.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Alice", "predicate": "manager", "object": "Bob"}})
    cb.write("Bob lives in Berlin.", session_id="s1", timestamp=1.0,
             metadata={"fact": {"subject": "Bob", "predicate": "city", "object": "Berlin"}})
    r = cb.query("What city does Alice's manager live in?", session_id="q", timestamp=2.0)
    assert "Berlin" in r.context  # final answer still surfaced
    # both hops are cited (structured citations carry wall_ts in provenance)
    structured_cites = [c for c in r.citations if "wall_ts" in c.provenance]
    assert len(structured_cites) >= 2


def test_relation_auto_detection_enables_arbitrary_multihop():
    # A relation other than the hardwired "manager" (here "employer", whose object Acme is
    # itself a known entity) must form a traversable hop chain — general multi-hop.
    cb = CuratedBrain(seed=0)
    cb.write("Alice's employer is Acme.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Alice", "predicate": "employer", "object": "Acme"}})
    cb.write("Acme's city is Berlin.", session_id="s1", timestamp=1.0,
             metadata={"fact": {"subject": "Acme", "predicate": "city", "object": "Berlin"}})
    plan = cb.planner.plan("What city is Alice's employer in?", entities=cb._entities,
                           predicates=frozenset({"employer", "city"}),
                           relation_preds=frozenset({"employer"}), session_ts=cb._session_ts)
    assert plan.hops == ["employer", "city"]  # employer auto-detected as a relation
    ctx = cb.query("What city is Alice's employer in?", session_id="q", timestamp=2.0).context
    assert "Berlin" in ctx  # end-to-end: chain resolves Alice -> Acme -> Berlin


def _mgr_chain_brain():
    # Alice -> Bob -> Carol -> Dave manager chain; cities on the tail two so 3- and 4-hop
    # queries land on distinct answers. Each object is itself a known entity, so "manager"
    # auto-detects as a relation and the chain is traversable at any depth.
    cb = CuratedBrain(seed=0)
    for s, p, o, ts in (("Alice", "manager", "Bob", 0.0), ("Bob", "manager", "Carol", 1.0),
                        ("Carol", "manager", "Dave", 2.0), ("Carol", "city", "Paris", 3.0),
                        ("Dave", "city", "Berlin", 4.0)):
        cb.write(f"{s}'s {p} is {o}.", session_id=f"s{int(ts)}", timestamp=ts,
                 metadata={"fact": {"subject": s, "predicate": p, "object": o}})
    return cb


def test_three_hop_chain_resolves_and_cites_every_hop():
    # "X's manager's manager's city" -> a depth-3 chain (two relations + fronted attribute).
    # The planner recovers the ordered run; the answer is the manager-of-manager's city and
    # every fact traversed is cited so the whole support set is attributable.
    cb = _mgr_chain_brain()
    q = "What city does Alice's manager's manager live in?"
    plan = cb.planner.plan(q, entities=cb._entities,
                           predicates=frozenset({"manager", "city"}),
                           relation_preds=frozenset({"manager"}), session_ts=cb._session_ts)
    assert plan.hops == ["manager", "manager", "city"]
    assert cb.answer_path("Alice", ["manager", "manager", "city"]) == "Paris"
    r = cb.query(q, session_id="q", timestamp=10.0)
    assert "Paris" in r.context  # Alice -> Bob -> Carol -> Paris
    assert sum(1 for c in r.citations if "wall_ts" in c.provenance) >= 3  # all three hops cited


def test_four_hop_chain_resolves_end_to_end():
    # Depth-4: three manager relations then the fronted "city" attribute.
    cb = _mgr_chain_brain()
    q = "What is Alice's manager's manager's manager's city?"
    plan = cb.planner.plan(q, entities=cb._entities,
                           predicates=frozenset({"manager", "city"}),
                           relation_preds=frozenset({"manager"}), session_ts=cb._session_ts)
    assert plan.hops == ["manager", "manager", "manager", "city"]
    assert cb.answer_path("Alice", ["manager", "manager", "manager", "city"]) == "Berlin"
    r = cb.query(q, session_id="q", timestamp=10.0)
    assert "Berlin" in r.context  # Alice -> Bob -> Carol -> Dave -> Berlin
    assert sum(1 for c in r.citations if "wall_ts" in c.provenance) >= 4


def test_chain_with_trailing_possessive_attribute():
    # Mixed relations with the attribute inside the possessive run ("...mentor's email"),
    # not fronted — the run itself carries the trailing attribute.
    cb = CuratedBrain(seed=0)
    for s, p, o, ts in (("Alice", "manager", "Bob", 0.0), ("Bob", "mentor", "Carol", 1.0),
                        ("Carol", "email", "carol@x.com", 2.0)):
        cb.write(f"{s}'s {p} is {o}.", session_id=f"s{int(ts)}", timestamp=ts,
                 metadata={"fact": {"subject": s, "predicate": p, "object": o}})
    q = "What is Alice's manager's mentor's email?"
    plan = cb.planner.plan(q, entities=cb._entities, session_ts=cb._session_ts,
                           predicates=frozenset({"manager", "mentor", "email"}),
                           relation_preds=frozenset({"manager", "mentor"}))
    assert plan.hops == ["manager", "mentor", "email"]
    assert "carol@x.com" in cb.query(q, session_id="q", timestamp=10.0).context


def test_broken_chain_degrades_gracefully():
    # A missing middle hop (Bob has no manager) must not crash: the chain resolves to None,
    # the query falls back to the structured backstop and still returns a sensible payload.
    cb = CuratedBrain(seed=0)
    cb.write("Alice's manager is Bob.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Alice", "predicate": "manager", "object": "Bob"}})
    cb.write("Bob lives in Berlin.", session_id="s1", timestamp=1.0,
             metadata={"fact": {"subject": "Bob", "predicate": "city", "object": "Berlin"}})
    q = "What city does Alice's manager's manager live in?"
    plan = cb.planner.plan(q, entities=cb._entities,
                           predicates=frozenset({"manager", "city"}),
                           relation_preds=frozenset({"manager"}), session_ts=cb._session_ts)
    assert plan.hops == ["manager", "manager", "city"]  # chain parsed
    r = cb.query(q, session_id="q", timestamp=10.0)  # must not raise
    assert r.context  # backstop still surfaces Alice's known facts
    assert "Bob" in r.context


# Plan-identity pins: for questions that produce a 2-hop / single-hop / open plan today, the
# generalized planner must produce the IDENTICAL QueryPlan (the no-op condition). Expected
# fields are hard-coded from the CURRENT code, run before the planner was generalized.
_PLAN_PINS = [
    ("What is Alice's email address?", "alice", "email", None, False, False),
    ("What city does Alice's manager live in?", "alice", "city", ["manager", "city"],
     False, False),
    ("Where did Alice live as of session 29?", "alice", "city", None, True, False),
    ("Tell me something interesting.", None, None, None, False, True),
    ("Who is Alice's manager?", "alice", "manager", None, False, False),
    ("Where does Alice live now?", "alice", "city", None, False, False),
    ("What is Alice's current role?", "alice", "role", None, False, False),
]


def test_plan_identity_unchanged_on_existing_questions():
    ds = generate(seed=0)
    cb, *_ = _feed_all(ds)
    pv, rp, _ = cb._derived_state()
    for q, ent, pred, hops, has_asof, open_ended in _PLAN_PINS:
        p = cb.planner.plan(q, entities=cb._entities, predicates=pv, relation_preds=rp,
                            session_ts=cb._session_ts)
        assert (p.entity, p.predicate, p.hops, p.as_of is not None, p.open_ended) == (
            ent, pred, hops, has_asof, open_ended), q


def test_fuse_drops_superseded_values():
    hits = [
        (VectorRecord(rid="1", text="Alice lives in Berlin.", wall_ts=1.0, session_id="s"), 0.9),
        (VectorRecord(rid="2", text="Alice lives in Munich.", wall_ts=2.0, session_id="s"), 0.8),
    ]
    out = fuse(hits, now=10.0, stale_pairs=[(frozenset({"alice"}), frozenset({"berlin"}))])
    assert [it.rid for it in out] == ["2"]  # the superseded "Berlin" record is filtered out


def test_fuse_drops_multiword_stale_values():
    # Entity-scoped token matching catches a MULTI-WORD stale value, without dropping a
    # record sharing only a common word (and never one about a DIFFERENT entity).
    hits = [
        (VectorRecord(rid="1", text="Priya's address is 14 Rua das Flores, Lisbon.",
                      wall_ts=1.0, session_id="s"), 0.9),
        (VectorRecord(rid="2", text="Priya's address is 88 Calle Mayor, Madrid.",
                      wall_ts=2.0, session_id="s"), 0.8),
    ]
    stale = [(frozenset({"priya"}), frozenset({"14", "rua", "das", "flores", "lisbon"}))]
    out = fuse(hits, now=10.0, stale_pairs=stale)
    assert [it.rid for it in out] == ["2"]  # the multi-word stale "…Lisbon" record is dropped


def test_fuse_drops_by_provenance_rid():
    # A record whose ASSERTED fact is superseded is dropped by exact identity, regardless of
    # phrasing (e.g. pronoun coreference — no subject token in the text at all).
    hits = [
        (VectorRecord(rid="1", text="Her current city is Berlin.", wall_ts=1.0,
                      session_id="s"), 0.9),
        (VectorRecord(rid="2", text="Alice lives in Munich.", wall_ts=2.0,
                      session_id="s"), 0.8),
    ]
    out = fuse(hits, now=10.0, stale_rids={"1"})
    assert [it.rid for it in out] == ["2"]


def test_fuse_keeps_records_merely_mentioning_a_stale_word():
    # The old filter matched value tokens alone, store-wide: once any role "manager" was
    # superseded, EVERY record containing "manager" was filtered ("Erin's manager is Bob").
    # Entity-scoping keeps records about other entities / other relations.
    hits = [
        (VectorRecord(rid="1", text="Erin's manager is Bob.", wall_ts=1.0,
                      session_id="s"), 0.9),
        (VectorRecord(rid="2", text="Who manages the Falcon project?", wall_ts=2.0,
                      session_id="s"), 0.8),
    ]
    # Dana's role "manager" was superseded (by "director") — stale pair is (dana, manager).
    stale = [(frozenset({"dana"}), frozenset({"manager"}))]
    out = fuse(hits, now=10.0, stale_pairs=stale)
    assert [it.rid for it in out] == ["1", "2"]  # neither mentions Dana -> neither dropped


def test_full_supersede_filter_in_core_no_stale_in_context():
    # End-to-end: a multi-word value superseded in the structured tier must NOT surface in
    # query context — proving supersede-filtering works in CORE (not just an adapter).
    cb = CuratedBrain(seed=0)
    for ts, obj in ((0.0, "14 Rua das Flores, Lisbon"), (1.0, "88 Calle Mayor, Madrid")):
        cb.write(f"Priya's mailing address is {obj}.", session_id="s", timestamp=ts,
                 metadata={"fact": {"subject": "Priya", "predicate": "mailing address",
                                    "object": obj}})
    ctx = cb.query("What is Priya's mailing address?", session_id="q", timestamp=2.0).context
    assert "Madrid" in ctx and "Lisbon" not in ctx  # current surfaced, stale filtered in core


# ------------------------------------------------- query-side fuzzy entity fallback ---
def _alice_email_brain():
    cb = CuratedBrain(seed=0)
    cb.write("Alice's email is alice@x.com.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Alice", "predicate": "email",
                                "object": "alice@x.com"}})
    return cb


def test_fuzzy_off_by_default_typo_yields_no_entity():
    # (a) With no cutoff configured, a typo'd name matches no entity — exactly as today.
    cb = _alice_email_brain()
    plan = cb.planner.plan("What is Alise's email?", entities=cb._entities,
                           session_ts=cb._session_ts)
    assert plan.entity is None and plan.open_ended


def test_fuzzy_on_resolves_typo_end_to_end():
    # (b) With a cutoff set, a typo'd name ('Alise') resolves to the stored 'alice' and the
    # store surfaces alice's fact — end-to-end through CuratedBrain(config=...).query.
    cb = CuratedBrain(seed=0, config=CBConfig(fuzzy_entity_cutoff=0.8))
    cb.write("Alice's email is alice@x.com.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Alice", "predicate": "email",
                                "object": "alice@x.com"}})
    ctx = cb.query("What is Alise's email?", session_id="q", timestamp=1.0).context
    assert "alice@x.com" in ctx


def test_fuzzy_fails_closed_on_ambiguity():
    # (c) A token equidistant from TWO entities ('jon' ~ john, joan) must resolve to neither.
    entities = {"john", "joan"}
    assert Planner()._fuzzy_entity({"who", "is", "jon"}, entities, 0.8) is None
    # And end-to-end: no resolution -> no structured backstop line for either.
    cb = CuratedBrain(seed=0, config=CBConfig(fuzzy_entity_cutoff=0.8))
    for name in ("John", "Joan"):
        cb.write(f"{name}'s email is {name.lower()}@x.com.", session_id="s0", timestamp=0.0,
                 metadata={"fact": {"subject": name, "predicate": "email",
                                    "object": f"{name.lower()}@x.com"}})
    plan = cb.planner.plan("What is Jon's email?", entities=cb._entities,
                           session_ts=cb._session_ts, fuzzy_cutoff=0.8)
    assert plan.entity is None


def test_fuzzy_on_does_not_override_exact_match():
    # (d) When an EXACT entity is present the fuzzy path never runs; the exact match stands.
    cb = CuratedBrain(seed=0, config=CBConfig(fuzzy_entity_cutoff=0.8))
    cb.write("Alice's email is alice@x.com.", session_id="s0", timestamp=0.0,
             metadata={"fact": {"subject": "Alice", "predicate": "email",
                                "object": "alice@x.com"}})
    cb.write("Alicia's email is alicia@x.com.", session_id="s1", timestamp=1.0,
             metadata={"fact": {"subject": "Alicia", "predicate": "email",
                                "object": "alicia@x.com"}})
    plan = cb.planner.plan("What is Alice's email?", entities=cb._entities,
                           session_ts=cb._session_ts, fuzzy_cutoff=0.8)
    assert plan.entity == "alice"  # exact, not the fuzzy neighbour alicia


def test_fuzzy_write_path_unaffected_snapshots_identical():
    # (e) Fuzzy is QUERY-side only: a fuzzy-off and a fuzzy-on brain given IDENTICAL writes
    # produce byte-identical snapshots (resolve.py / the write path is untouched).
    ds = generate(seed=0)
    off = CuratedBrain(seed=0)
    on = CuratedBrain(seed=0, config=CBConfig(fuzzy_entity_cutoff=0.8))
    for o in ds.observations:
        for be in (off, on):
            be.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                     metadata={"fact": o.fact} if o.fact else None)
    assert on.snapshot() == off.snapshot()
