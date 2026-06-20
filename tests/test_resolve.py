"""Entity resolution (P1) — conservative subject canonicalization.

The contract: resolve real variants of the SAME entity to one canonical key, but NEVER merge
two distinct entities (false merge > miss). No fuzzy matching; every merge gated on
store-provable uniqueness and refused on ambiguity; a provable no-op on single-token,
honorific-free subjects (so AC-1/AC-9 are byte-identical).
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.resolve import EntityResolver, strip_honorific
from curated_brain.util import normalize


def test_strip_honorific():
    assert strip_honorific("ms. smith") == "smith"
    assert strip_honorific("dr erin smith") == "erin smith"
    assert strip_honorific("dr") == "dr"          # >= 1 token must remain
    assert strip_honorific("mister") == "mister"  # not in the closed honorific set


def test_noop_on_single_token_names():
    # AC-9 safety: a single-token, honorific-free subject canonicalizes to normalize() exactly.
    r = EntityResolver()
    for name in ["Alice", "Bob", "Carol", "Dan", "Erin", "Frank"]:
        assert r.resolve_and_register(name) == normalize(name)
    assert r.entities == {"alice", "bob", "carol", "dan", "erin", "frank"}


def test_full_name_subsumes_partials():
    r = EntityResolver()
    assert r.resolve_and_register("Erin Smith") == "erin smith"
    assert r.resolve_and_register("Ms. Smith") == "erin smith"  # surname promotion
    assert r.resolve_and_register("Erin") == "erin smith"       # first-name promotion
    assert {"erin", "smith"} <= r.entities                       # planner can find the partials


def test_false_merge_two_homonym_surnames_refuse():
    r = EntityResolver()
    r.resolve_and_register("Erin Smith")
    r.resolve_and_register("John Smith")  # second distinct "Smith" -> poison the surname
    assert r.resolve_and_register("Smith") == "smith"   # ambiguous -> refuse, stays its own
    assert r.canonical("Ms. Smith") == "smith"


def test_false_merge_two_homonym_given_refuse():
    r = EntityResolver()
    r.resolve_and_register("Alex Kim")
    r.resolve_and_register("Alex Ross")  # second distinct "Alex" -> poison the given name
    assert r.resolve_and_register("Alex") == "alex"     # the classic "two Alex" -> refuse


def test_standalone_first_is_not_retro_merged():
    r = EntityResolver()
    assert r.resolve_and_register("Erin") == "erin"     # standalone single-token entity
    r.resolve_and_register("Erin Smith")                # a full name learned later
    assert r.canonical("Erin") == "erin"                # standalone stays separate (under-merge)


def test_no_fuzzy_or_substring_merge():
    r = EntityResolver()
    r.resolve_and_register("Erin Smith")
    for s in ["Smyth", "Smithson", "Erina", "Eri"]:
        assert r.canonical(s) == normalize(s)  # no edit-distance / substring / nickname merge


def test_order_independent_refusal():
    a, b = EntityResolver(), EntityResolver()
    for name in ["Alex Kim", "Alex Ross"]:
        a.resolve_and_register(name)
    for name in ["Alex Ross", "Alex Kim"]:  # reverse order
        b.resolve_and_register(name)
    assert a.canonical("Alex") == b.canonical("Alex") == "alex"  # ambiguous regardless of order


def test_false_merge_cross_index_given_vs_surname_refuse():
    # A token that is one person's GIVEN name AND a different person's SURNAME is ambiguous
    # ACROSS the two indices and must refuse (the same-index guards miss this boundary).
    r = EntityResolver()
    r.resolve_and_register("Kim Lee")     # 'kim' is a given name
    r.resolve_and_register("Robert Kim")  # 'kim' is a DIFFERENT person's surname
    assert r.canonical("Kim") == "kim"    # ambiguous across indices -> refuse, no false merge
    assert "kim" not in r.entities        # not offered to the planner either
    cb = CuratedBrain(seed=0)
    cb.write("Kim Lee lives in Tokyo.", session_id="s", timestamp=0.0,
             metadata={"fact": {"subject": "Kim Lee", "predicate": "city", "object": "Tokyo"}})
    cb.write("Robert Kim lives in Berlin.", session_id="s", timestamp=1.0,
             metadata={"fact": {"subject": "Robert Kim", "predicate": "city", "object": "Berlin"}})
    assert cb.answer_structured("Kim", "city") == ""  # returns NEITHER entity's facts


def test_false_merge_poisoned_one_index_unique_other_refuse():
    # A token that is AMBIGUOUS in one role (given name of two people) must refuse even if it
    # is unique in the OTHER role (a third person's surname) — else it merges into that third.
    r = EntityResolver()
    r.resolve_and_register("Sam Brown")    # 'sam' given (A)
    r.resolve_and_register("Sam Green")    # 'sam' given again -> poisons _given['sam']
    r.resolve_and_register("Robert Sam")   # 'sam' surname of a DISTINCT third person
    assert r.canonical("Sam") == "sam"     # poisoned in one role -> refuse outright
    assert "sam" not in r.entities
    # symmetric: poisoned surname + unique given
    r2 = EntityResolver()
    for n in ["Ann Park", "Ben Park", "Park Cho"]:
        r2.resolve_and_register(n)
    assert r2.canonical("Park") == "park"


def test_subject_case_preserved_when_no_resolution():
    cb = CuratedBrain(seed=0)
    cb.write("Alice lives in Berlin.", session_id="s", timestamp=0.0,
             metadata={"fact": {"subject": "Alice", "predicate": "city", "object": "Berlin"}})
    assert cb.structured.facts[0].subject == "Alice"  # original case kept (byte no-op)


def test_end_to_end_resolution_via_curated_brain():
    cb = CuratedBrain(seed=0)
    meta = {"fact": {"subject": "Ms. Smith", "predicate": "role", "object": "analyst"}}
    cb.write("Erin Smith lives in Vienna.", session_id="s", timestamp=0.0,
             metadata={"fact": {"subject": "Erin Smith", "predicate": "city", "object": "Vienna"}})
    cb.write("Ms. Smith's role is analyst.", session_id="s", timestamp=1.0, metadata=meta)
    assert meta["fact"]["subject"] == "Ms. Smith"  # caller's dict was NOT mutated
    # both facts live under one canonical subject, reachable by every surface form
    assert cb.answer_structured("Erin", "city") == "Vienna"
    assert cb.answer_structured("Erin Smith", "role") == "analyst"
    assert cb.answer_structured("Ms. Smith", "city") == "Vienna"
    assert "Vienna" in cb.query("Where does Erin live?", session_id="q", timestamp=2.0).context


def test_resolution_survives_snapshot_restore():
    cb = CuratedBrain(seed=0)
    cb.write("Erin Smith lives in Vienna.", session_id="s", timestamp=0.0,
             metadata={"fact": {"subject": "Erin Smith", "predicate": "city", "object": "Vienna"}})
    blob = cb.snapshot()
    restored = CuratedBrain(seed=0)
    restored.restore(blob)
    assert restored.snapshot() == blob  # byte-identical (resolver not serialized)
    assert restored.answer_structured("Ms. Smith", "city") == "Vienna"  # resolution rebuilt
