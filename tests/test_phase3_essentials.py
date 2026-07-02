"""Phase 3a production essentials: hard erasure (forget), inverse/set queries,
snapshot/stats with the production ANN index, and derived-state caching correctness."""

from __future__ import annotations

import importlib.util

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor


def _fact(subject, predicate, object):
    return {"fact": {"subject": subject, "predicate": predicate, "object": object}}


# ------------------------------------------------------------------ inverse queries -----
def test_answer_who_inverse_query():
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0, metadata=_fact("Alice", "city", "Berlin"))
    cb.write("x", session_id="s", timestamp=1.0, metadata=_fact("Bob", "city", "Berlin"))
    cb.write("x", session_id="s", timestamp=2.0, metadata=_fact("Cara", "city", "Oslo"))
    assert cb.answer_who("city", "Berlin") == ["Alice", "Bob"]
    assert cb.answer_who("city", "berlin") == ["Alice", "Bob"]  # normalized match
    # supersede removes a subject from the current answer set but keeps as-of history
    cb.write("x", session_id="s", timestamp=3.0, metadata=_fact("Alice", "city", "Oslo"))
    assert cb.answer_who("city", "Berlin") == ["Bob"]
    assert cb.answer_who("city", "Berlin", at=2.5) == ["Alice", "Bob"]  # believed then


def test_who_reports_to():
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0, metadata=_fact("Erin", "manager", "Bob"))
    cb.write("x", session_id="s", timestamp=1.0, metadata=_fact("Dana", "manager", "Bob"))
    assert cb.answer_who("manager", "Bob") == ["Dana", "Erin"]


# ------------------------------------------------------------------------ erasure -------
def test_forget_subject_erases_facts_episodes_and_history():
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb.write("Erin's city is Vienna.", session_id="s", timestamp=0.0)
    cb.write("Erin has moved to Oslo.", session_id="s", timestamp=1.0)  # history too
    cb.write("Bob's city is Berlin.", session_id="s", timestamp=2.0)
    report = cb.forget("Erin")
    assert report["facts"] == 2 and report["episodes"] >= 1
    assert cb.answer_structured("Erin", "city") == ""
    assert cb.structured.history("Erin", "city") == []  # superseded history erased too
    assert "erin" not in cb._entities
    assert "Erin" not in cb.query("Where does Erin live?", session_id="q",
                                  timestamp=3.0).context
    assert cb.answer_structured("Bob", "city") == "Berlin"  # others untouched


def test_forget_removes_object_side_mentions_on_full_forget():
    # Facts on OTHER subjects that embed the forgotten entity's identity go too.
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0, metadata=_fact("Erin", "manager", "Bob"))
    cb.write("x", session_id="s", timestamp=1.0, metadata=_fact("Bob", "city", "Berlin"))
    cb.forget("Bob")
    assert cb.answer_structured("Bob", "city") == ""
    assert cb.answer_structured("Erin", "manager") == ""  # object-side fact erased


def test_forget_single_predicate_is_scoped():
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0, metadata=_fact("Erin", "city", "Vienna"))
    cb.write("x", session_id="s", timestamp=1.0, metadata=_fact("Erin", "role", "designer"))
    cb.forget("Erin", predicate="city")
    assert cb.answer_structured("Erin", "city") == ""
    assert cb.answer_structured("Erin", "role") == "designer"  # other predicates survive


def test_forget_clears_echo_guard_so_reassertion_works():
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb.write("Erin's city is Vienna.", session_id="s", timestamp=0.0)
    cb.forget("Erin")
    cb.write("Erin's city is Vienna.", session_id="s", timestamp=1.0)  # same bytes, post-erasure
    assert cb.answer_structured("Erin", "city") == "Vienna"  # not suppressed as an echo


def test_forget_survives_snapshot_roundtrip():
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0, metadata=_fact("Erin", "city", "Vienna"))
    cb.forget("Erin")
    cb2 = CuratedBrain(seed=0)
    cb2.restore(cb.snapshot())
    assert cb2.answer_structured("Erin", "city") == ""
    assert cb2.snapshot() == cb.snapshot()


# ------------------------------------------------------------- derived-state caching ----
def test_derived_cache_stays_correct_across_mutations():
    cb = CuratedBrain(seed=0)
    cb.write("x", session_id="s", timestamp=0.0, metadata=_fact("Erin", "city", "Vienna"))
    q = "Where does Erin live?"
    before = cb.query(q, session_id="q", timestamp=1.0).context
    assert "Vienna" in before
    assert cb.query(q, session_id="q", timestamp=1.0).context == before  # cached path, same
    cb.write("x", session_id="s", timestamp=2.0, metadata=_fact("Erin", "city", "Oslo"))
    after = cb.query(q, session_id="q", timestamp=3.0).context  # invalidated on write
    assert "Oslo" in after and "Vienna" not in after
    cb.consolidate()
    assert "Oslo" in cb.query(q, session_id="q", timestamp=4.0).context


# ------------------------------------------------------ production ANN, whole backend ---
@pytest.mark.skipif(importlib.util.find_spec("hnswlib") is None,
                    reason="hnswlib not installed ([scale] extra)")
def test_stats_snapshot_save_work_with_hnsw_backend(tmp_path):
    from curated_brain.vector import HnswIndex, VectorTier

    cb = CuratedBrain(seed=0)
    cb.vector = VectorTier(cb.embedder, index=HnswIndex(cb.embedder.dim, max_elements=64))
    cb.write("Erin's city is Vienna.", session_id="s", timestamp=0.0,
             metadata=_fact("Erin", "city", "Vienna"))
    st = cb.stats()  # previously CRASHED (stats -> snapshot -> vector.to_dict TypeError)
    assert st.bytes > 0 and st.structured_count == 1
    p = str(tmp_path / "store.json")
    cb.save(p)  # previously crashed too
    cb2 = CuratedBrain(seed=0)
    cb2.load(p)
    assert cb2.answer_structured("Erin", "city") == "Vienna"
    assert "Vienna" in cb2.query("Where does Erin live?", session_id="q",
                                 timestamp=1.0).context
