"""Phase-2 correctness fixes (red-team v2, PROGRESS.md 2026-07-02): pipe-safe fact keys,
retroactive/out-of-order bi-temporality, unicode tokenization, entity-scoped supersede
filtering, resolver-state persistence, dim-checked restore, configurable context budget,
and MCP wall-clock/locking/atomic-persist behavior."""

from __future__ import annotations

import threading

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.mcp_server import MemoryService
from curated_brain.structured import StructuredTier
from curated_brain.util import tokenize


def _fact(subject, predicate, object):
    return {"fact": {"subject": subject, "predicate": predicate, "object": object}}


# ------------------------------------------------------------------ pipe-safe fact keys --
def test_pipe_in_fact_value_does_not_crash_consolidate():
    cb = CuratedBrain(seed=0)
    cb.write("Team motto recorded.", session_id="s", timestamp=0.0,
             metadata=_fact("Team", "motto", "ship | iterate | learn"))
    cb.write("Motto updated.", session_id="s", timestamp=1.0,
             metadata=_fact("Team", "motto", "measure | cut"))
    rep = cb.consolidate()  # used to raise ValueError: too many values to unpack
    assert rep.episodes_in >= 1
    assert cb.answer_structured("Team", "motto") == "measure | cut"


# ------------------------------------------------------- retroactive + out-of-order time --
def test_retroactive_valid_from_is_recordable():
    # "She moved last month": true-in-the-world (valid time) precedes recording time.
    cb = CuratedBrain(seed=0)
    cb.write("Erin moved to Vienna a month ago.", session_id="s", timestamp=1000.0,
             metadata={"fact": {"subject": "Erin", "predicate": "city", "object": "Vienna",
                                "valid_from": 400.0}})
    f = cb.structured.current("Erin", "city")
    assert f is not None and f.valid_from == 400.0 and f.created_at == 1000.0


def test_out_of_order_write_does_not_invert_intervals():
    # A late-arriving OLDER observation must become history, not corrupt the open fact.
    t = StructuredTier()
    t.assert_fact(fact_id="f1", subject="Alice", predicate="city", object="Munich",
                  valid_from=200.0, created_at=200.0)
    cur, hist = t.assert_fact(fact_id="f2", subject="Alice", predicate="city",
                              object="Berlin", valid_from=100.0, created_at=300.0)
    assert cur.object == "Munich"  # the open fact is untouched and still current
    assert t.current("Alice", "city").object == "Munich"
    assert hist is not None and hist.object == "Berlin"
    assert hist.valid_from == 100.0 and hist.valid_to == 200.0  # closed, NOT inverted
    for f in t.facts:
        assert f.valid_from <= f.valid_to  # no inverted interval anywhere


def test_bad_valid_from_metadata_rejected():
    cb = CuratedBrain(seed=0)
    for bad in (float("nan"), float("inf"), "yesterday", True):
        with pytest.raises(ValueError):
            cb.write("x", session_id="s", timestamp=0.0,
                     metadata={"fact": {"subject": "A", "predicate": "p", "object": "o",
                                        "valid_from": bad}})


# ------------------------------------------------------------------ unicode tokenization --
def test_non_ascii_text_is_storable_and_retrievable():
    assert tokenize("Эрин живёт в Вене") != []  # old tokenizer: [] -> zero vector
    cb = CuratedBrain(seed=0)
    r = cb.write("Эрин живёт в Вене.", session_id="s", timestamp=0.0)
    assert r.stored
    ctx = cb.query("Где живёт Эрин?", session_id="q", timestamp=1.0).context
    assert "Вене" in ctx  # retrievable via its own (non-zero) embedding


def test_accented_names_normalize_consistently():
    cb = CuratedBrain(seed=0)
    cb.write("José's city is Lisbon.", session_id="s", timestamp=0.0,
             metadata=_fact("José", "city", "Lisbon"))
    assert cb.answer_structured("José", "city") == "Lisbon"
    assert "jos" not in {f.subject for f in cb.structured.facts}  # no 'é'-stripped key


# ---------------------------------------------------- entity-scoped supersede filtering --
def test_superseding_a_common_word_value_does_not_filter_other_records():
    # Dana's role "manager" is superseded by "director". Records about OTHER entities that
    # merely contain the word "manager" must still surface (the old store-wide token filter
    # dropped them all, forever).
    cb = CuratedBrain(seed=0)
    cb.write("Erin's manager is Bob.", session_id="s", timestamp=0.0,
             metadata=_fact("Erin", "manager", "Bob"))
    cb.write("Dana's role is manager.", session_id="s", timestamp=1.0,
             metadata=_fact("Dana", "role", "manager"))
    cb.write("Dana's role is director.", session_id="s", timestamp=2.0,
             metadata=_fact("Dana", "role", "director"))
    ctx = cb.query("Tell me about Erin", session_id="q", timestamp=3.0).context
    assert "Erin's manager is Bob." in ctx  # not a Dana record -> never filtered
    ctx2 = cb.query("What is Dana's role?", session_id="q", timestamp=3.0).context
    assert "director" in ctx2 and "Dana's role is manager." not in ctx2  # true stale dropped


def test_stale_value_for_one_entity_current_for_another_is_scoped():
    # "Paris" is stale for Alice but current for Bob: Alice's old record must be filtered
    # while Bob's surfaces (the old global open-values subtraction protected BOTH).
    cb = CuratedBrain(seed=0)
    cb.write("Alice's city is Paris.", session_id="s", timestamp=0.0,
             metadata=_fact("Alice", "city", "Paris"))
    cb.write("Alice's city is Rome.", session_id="s", timestamp=1.0,
             metadata=_fact("Alice", "city", "Rome"))
    cb.write("Bob's city is Paris.", session_id="s", timestamp=2.0,
             metadata=_fact("Bob", "city", "Paris"))
    ctx = cb.query("Where does Alice live?", session_id="q", timestamp=3.0).context
    assert "Alice's city is Paris." not in ctx and "Rome" in ctx
    ctx2 = cb.query("Where does Bob live?", session_id="q", timestamp=3.0).context
    assert "Paris" in ctx2


# ------------------------------------------------------------- resolver-state persistence --
def test_resolver_state_survives_snapshot_restore_exactly():
    cb = CuratedBrain(seed=0)
    # Build real ambiguity history: two full names poison the shared surname "smith".
    cb.write("f", session_id="s", timestamp=0.0, metadata=_fact("Erin Smith", "city", "Vienna"))
    cb.write("f", session_id="s", timestamp=1.0, metadata=_fact("Alan Smith", "city", "Berlin"))
    cb.write("f", session_id="s", timestamp=2.0, metadata=_fact("Kim", "city", "Oslo"))
    before = cb._resolver.to_dict()
    blob = cb.snapshot()
    cb2 = CuratedBrain(seed=0)
    cb2.restore(blob)
    assert cb2._resolver.to_dict() == before  # byte-faithful, incl. poisons + singletons
    assert cb2._resolver.canonical("Smith") == "smith"  # still refused (ambiguous)
    assert cb2._resolver.canonical("Erin") == "erin smith"  # still promoted


def test_restore_rejects_mismatched_embedding_dim():
    cb = CuratedBrain(seed=0, dim=256)
    blob = cb.snapshot()
    other = CuratedBrain(seed=0, dim=64)
    with pytest.raises(ValueError, match="dim"):
        other.restore(blob)  # was: a shape error later, mid-query


# ------------------------------------------------------------------ context budget honors k --
def test_max_context_items_is_configurable():
    texts = [  # lexically diverse so the surprise gate stores every one
        "The astronomy conference gathers in Helsinki.",
        "A pottery conference workshop opened near Kyoto.",
        "Quantum researchers hold a conference aboard the ferry.",
        "The mushroom foragers conference meets at dawn.",
        "Jazz archivists announced a conference in Montevideo.",
        "A glacier science conference relocated to Tromso.",
        "The typography conference celebrates ligatures.",
        "Beekeepers run their conference beside the orchard.",
    ]

    def fill(cb):
        for i, txt in enumerate(texts):
            cb.write(txt, session_id="s", timestamp=float(i))
    small, big = CuratedBrain(seed=0), CuratedBrain(seed=0, max_context_items=8)
    fill(small), fill(big)
    n_small = len(small.query("conference", session_id="q", timestamp=9.0, k=8).citations)
    n_big = len(big.query("conference", session_id="q", timestamp=9.0, k=8).citations)
    assert n_small <= 4  # default budget unchanged (AC-4 / benchmark behavior preserved)
    assert n_big > n_small  # a caller asking for more context actually gets more


# ------------------------------------------------------------- consolidation entity links --
def test_merged_claim_remains_visible_to_entity_filtered_search():
    cb = CuratedBrain(seed=0)
    # Two paraphrases of the same current fact merge into one semantic claim on consolidate.
    cb.write("Erin's city is Vienna.", session_id="s", timestamp=0.0,
             metadata=_fact("Erin", "city", "Vienna"))
    cb.write("Erin is living in Vienna these days.", session_id="s", timestamp=1.0,
             metadata=_fact("Erin", "city", "Vienna"))
    cb.consolidate()
    claims = [r for r in cb._episodes if r.tier == "semantic"]
    if claims:  # merge happened -> the claim must carry its subject for entity filters
        hits = cb.vector.search("Vienna", k=4, t=2.0, entity="erin")
        assert any(r.rid == claims[0].id for r, _ in hits)


# ------------------------------------------------------------------------------ MCP layer --
def test_mcp_defaults_to_wall_clock_not_epoch_zero():
    svc = MemoryService()
    svc.write("Erin's city is Vienna.")
    ts = [r.wall_ts for r in svc.cb._episodes]
    assert ts and all(t > 1e9 for t in ts)  # real time, not the old default 0.0


def test_mcp_persist_is_atomic_and_batched(tmp_path):
    p = str(tmp_path / "store.json")
    svc = MemoryService(path=p, persist_every=2)
    svc.write("Erin's city is Vienna.")
    svc.write("Bob's city is Berlin.")  # second write crosses the batch threshold
    assert (tmp_path / "store.json").exists()
    assert not (tmp_path / "store.json.tmp").exists()  # tmp renamed away (atomic)
    cb2 = CuratedBrain(seed=0)
    cb2.load(p)
    assert cb2.answer_structured("Erin", "city") == "Vienna"


def test_mcp_operations_are_serialized_under_concurrency():
    svc = MemoryService()
    errors: list[Exception] = []

    def hammer(i):
        try:
            for j in range(20):
                svc.write(f"Person{i} note {j} about topic-{i}-{j}.")
                svc.query(f"topic-{i}")
        except Exception as e:  # pragma: no cover - only on regression
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(svc.cb._episodes) <= 80  # store is consistent (no interleaved corruption)
