"""Robustness / property tests (Track H — hardening).

A production memory layer must (a) survive adversarial *text* (unicode, control chars,
oversized, empty) without an opaque crash, (b) reject malformed *structured* input
(non-finite timestamps, ill-formed facts, non-str payloads) with a clear, typed error
instead of crashing deep or silently corrupting state, (c) stay byte-deterministic for a
fixed seed + input, and (d) round-trip exactly through snapshot/restore. Inputs are
generated from a fixed seed so any failure reproduces exactly.
"""

from __future__ import annotations

import math
import random

import pytest

from curated_brain.backend import CuratedBrain

# Nasty-but-plausible observation texts the layer must survive without raising.
ADVERSARIAL = [
    "", "   ", "\n\t\n", "a" * 5000, "café ☕ naïve 日本語 😀 𝕦𝕟𝕚𝕔𝕠𝕕𝕖",
    "\x00\x01\x02 control chars", "Erin | city | Vienna", "DROP TABLE facts;--",
    "{}", "[]", "the the the the the", "ZZZZZZ", "123 456 789",
    "   Mixed   CASE   Text   ", "🙂" * 50,
]

_PEOPLE = ["Erin", "Bob", "Cara", "Dana"]
_CITIES = ["Vienna", "Paris", "Oslo", "Berlin"]


def _fuzz_stream(seed: int, n: int = 60):
    """Deterministic mixed stream of (text, session, ts, fact|None): ~40% adversarial junk,
    the rest well-formed fact statements (which also exercise supersede over time)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if rng.random() < 0.4:
            out.append((rng.choice(ADVERSARIAL), f"s{rng.randint(0, 9)}", float(i), None))
        else:
            who, city = rng.choice(_PEOPLE), rng.choice(_CITIES)
            fact = {"subject": who, "predicate": "city", "object": city}
            out.append((f"{who} lives in {city}.", f"s{rng.randint(0, 9)}", float(i), fact))
    return out


def _feed(cb: CuratedBrain, stream):
    for text, sid, ts, fact in stream:
        cb.write(text, session_id=sid, timestamp=ts,
                 metadata={"fact": fact} if fact else None)


def test_write_query_consolidate_never_crash_on_adversarial_input():
    cb = CuratedBrain(seed=0)
    for text, sid, ts, fact in _fuzz_stream(1):
        receipt = cb.write(text, session_id=sid, timestamp=ts,
                           metadata={"fact": fact} if fact else None)
        assert receipt is not None and receipt.reason in {"stored", "reinforced", "discarded"}
    for q in ADVERSARIAL:
        out = cb.query(q, session_id="q", timestamp=10_000.0, k=8)
        assert isinstance(out.context, str) and out.tokens_in >= 0
    cb.consolidate()          # the worker must survive a junk-laden store
    assert cb.snapshot()      # and the result still serializes cleanly


def test_determinism_under_fuzz():
    a, b = CuratedBrain(seed=0), CuratedBrain(seed=0)
    _feed(a, _fuzz_stream(2))
    _feed(b, _fuzz_stream(2))
    assert a.snapshot() == b.snapshot()
    a.consolidate()
    b.consolidate()
    assert a.snapshot() == b.snapshot()


def test_snapshot_restore_roundtrips_under_fuzz():
    cb = CuratedBrain(seed=0)
    _feed(cb, _fuzz_stream(3))
    blob = cb.snapshot()
    restored = CuratedBrain(seed=0)
    restored.restore(blob)
    assert restored.snapshot() == blob
    for q in ["Erin", "Where does Bob live?", "", "🙂"]:
        assert (cb.query(q, session_id="q", timestamp=10_000.0).context
                == restored.query(q, session_id="q", timestamp=10_000.0).context)


def test_invariants_hold_after_fuzz_and_consolidate():
    cb = CuratedBrain(seed=0)
    _feed(cb, _fuzz_stream(4))
    cb.consolidate()
    # provenance is retained for every fact (PRD §8 invariant) — non-vacuous: facts exist.
    assert cb.structured.facts
    assert all(f.provenance.get("session_id") for f in cb.structured.facts)
    # at most one open (current) fact per (subject, predicate) — supersede stayed consistent.
    open_keys = [(f.subject, f.predicate) for f in cb.structured.facts if f.is_open]
    assert len(open_keys) == len(set(open_keys))


# --- boundary validation: reject malformed input loudly instead of crashing/corrupting ---
def test_non_finite_timestamp_is_rejected_not_silently_ghosted():
    # Regression: a NaN/inf wall_ts used to create an "open" fact invisible to as-of queries
    # (silent bi-temporal corruption). It must now be rejected at the boundary.
    cb = CuratedBrain(seed=0)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            cb.write("Erin lives in Vienna.", session_id="s", timestamp=bad,
                     metadata={"fact": {"subject": "Erin", "predicate": "city",
                                        "object": "Vienna"}})
        with pytest.raises(ValueError):
            cb.query("anything", session_id="q", timestamp=bad)
    # nothing leaked into the store
    assert not cb.structured.facts and cb.stats().episodic_count == 0
    assert math.isnan(float("nan"))  # sanity: the guard, not the test, does the work


def test_malformed_fact_and_non_str_inputs_raise_clearly():
    cb = CuratedBrain(seed=0)
    for bad_fact in ({"subject": "A", "predicate": "city"},   # missing object
                     {"subject": "A", "predicate": "city", "object": ""},  # empty
                     {"subject": "A", "predicate": "city", "object": 42},  # non-str
                     "not-a-dict", ["nope"]):
        with pytest.raises(ValueError):
            cb.write("text", session_id="s", timestamp=0.0, metadata={"fact": bad_fact})
    for bad_obs in (None, 123, ["list"]):
        with pytest.raises(TypeError):
            cb.write(bad_obs, session_id="s", timestamp=0.0)
    for bad_q in (None, 123):
        with pytest.raises(TypeError):
            cb.query(bad_q, session_id="q", timestamp=0.0)
