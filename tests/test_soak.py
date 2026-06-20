"""Soak / scale test (Track H + C): the curation thesis at volume.

The core claim is that the store grows with the number of DISTINCT facts, not the number of
observations — it gets more useful the longer it runs, not more bloated. This drives a
redundancy-heavy stream of thousands of observations over a few hundred distinct facts
through the surprise gate and asserts the store stays bounded while every distinct fact stays
recallable. Deterministic (seeded), so a regression reproduces exactly.
"""

from __future__ import annotations

import random
import time

from curated_brain.backend import CuratedBrain

DISTINCT = 500      # distinct (subject, city) facts
REPEATS = 10        # times each is restated (verbatim) -> 5000 observations total


def _redundant_stream(seed: int):
    rng = random.Random(seed)
    facts = [(f"Person{i}", f"Cityplace{i}") for i in range(DISTINCT)]
    stream = [(s, c) for (s, c) in facts for _ in range(REPEATS)]
    rng.shuffle(stream)
    return facts, stream


def test_store_grows_sublinearly_under_redundancy():
    cb = CuratedBrain(seed=0)
    facts, stream = _redundant_stream(0)
    t0 = time.perf_counter()
    for j, (subj, city) in enumerate(stream):
        cb.write(f"{subj} lives in {city}.", session_id=f"s{j % 50}", timestamp=float(j),
                 metadata={"fact": {"subject": subj, "predicate": "city", "object": city}})
    ingest_s = time.perf_counter() - t0

    m = cb.metrics()
    n_obs = DISTINCT * REPEATS
    # the gate kept the episodic store ~ the distinct facts, NOT the total observations
    assert m["store_size"] <= DISTINCT * 1.2, \
        f"store {m['store_size']} not bounded by ~{DISTINCT} distinct facts ({n_obs} obs)"
    # genuinely selective on a 90%-redundant stream (Pillar B)
    assert m["discard_rate"] > 0.5, f"discard_rate {m['discard_rate']:.2f} too low"
    # every distinct fact is captured exactly once in the structured tier (idempotent assert)
    assert m["structured_facts"] == DISTINCT
    assert ingest_s < 60.0, f"ingest of {n_obs} obs took {ingest_s:.1f}s"  # stays tractable


def test_full_recall_after_soak():
    cb = CuratedBrain(seed=0)
    facts, stream = _redundant_stream(0)
    for j, (subj, city) in enumerate(stream):
        cb.write(f"{subj} lives in {city}.", session_id="s", timestamp=float(j),
                 metadata={"fact": {"subject": subj, "predicate": "city", "object": city}})
    cb.consolidate()  # "sleep" — must not lose any distinct fact
    hits = sum(cb.answer_structured(s, "city") == c for s, c in facts)
    assert hits == DISTINCT, f"recall {hits}/{DISTINCT} after soak + consolidation"


def test_snapshot_restore_roundtrips_at_scale():
    cb = CuratedBrain(seed=0)
    _, stream = _redundant_stream(1)
    for j, (subj, city) in enumerate(stream):
        cb.write(f"{subj} lives in {city}.", session_id="s", timestamp=float(j),
                 metadata={"fact": {"subject": subj, "predicate": "city", "object": city}})
    blob = cb.snapshot()
    restored = CuratedBrain(seed=0)
    restored.restore(blob)
    assert restored.snapshot() == blob  # byte-identical round-trip at thousands of records
