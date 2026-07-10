"""Definite-NP + ellipsis coreference — held-out fixture gate (anti-tuning).

``tests/fixtures/coref_holdout.json`` was authored BEFORE the implementation and frozen
(its sha256 is pinned in ``coref_holdout.sha256``). This test replays each case through a
``CuratedBrain`` wired with the opt-in ``HeuristicExtractor(resolve_definite_np=True)`` and
scores the resolved-rate and false-positive count against the bars embedded in the fixture.
The fixture is never adjusted to make the implementation look good; if the bar is not met the
number is reported honestly (the test is marked xfail only if the bar genuinely can't be met).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor
from curated_brain.fakes import DeterministicEmbedder

FIXTURE = Path(__file__).parent / "fixtures" / "coref_holdout.json"
SHA_FILE = Path(__file__).parent / "fixtures" / "coref_holdout.sha256"


def _load() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_is_frozen():
    # The fixture's sha256 must match the pin recorded at authoring time — proof it was not
    # edited to flatter the implementation. Any legitimate post-freeze edit updates both.
    pinned = SHA_FILE.read_text().split()[0]
    actual = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert actual == pinned, "coref_holdout.json changed — re-pin only for a disclosed edit"


def _run_case(case: dict) -> list[dict]:
    """Feed a case's setup then its `text` through a fresh brain; return the facts the
    structured tier gained from the `text` observation (as {subject,predicate,object})."""
    cb = CuratedBrain(embedder=DeterministicEmbedder(64), dim=64, seed=0,
                      extractor=HeuristicExtractor(resolve_definite_np=True))
    t = 0.0
    for obs in case["setup"]:
        cb.write(obs, session_id="s0", timestamp=t)
        t += 1.0
    before = {(f.subject, f.predicate, f.object) for f in cb.structured.facts}
    cb.write(case["text"], session_id="s1", timestamp=t)
    after = {(f.subject, f.predicate, f.object) for f in cb.structured.facts}
    return [{"subject": s, "predicate": p, "object": o} for (s, p, o) in after - before]


def test_holdout_meets_prereg_bar():
    data = _load()
    cases = data["cases"]
    resolved_hits = 0
    resolved_total = 0
    false_positives = 0
    misses: list[str] = []
    fps: list[str] = []
    for case in cases:
        new_facts = _run_case(case)
        exp = case["expect"]
        if exp == "none":
            # Any fact whose subject is a real name (i.e. coreference fired and resolved to an
            # entity) is a false positive. A "none" case must produce no such resolved fact.
            if new_facts:
                false_positives += 1
                fps.append(f"{case['id']}: {new_facts}")
        else:
            resolved_total += 1
            want = {(exp["subject"], exp["predicate"], exp["object"])}
            got = {(f["subject"], f["predicate"], f["object"]) for f in new_facts}
            if want <= got:
                resolved_hits += 1
            else:
                misses.append(f"{case['id']}: want {exp}, got {new_facts}")

    resolved_rate = resolved_hits / resolved_total if resolved_total else 0.0
    assert false_positives == data["target_false_positives"], (
        f"false positives {false_positives} != target "
        f"{data['target_false_positives']}: {fps}")
    assert resolved_rate >= data["target_resolved"], (
        f"resolved-rate {resolved_rate:.3f} < target {data['target_resolved']}: {misses}")


# ---------------------------------------------------------- mechanism unit tests --------
def test_definite_np_resolves_unique_role_holder():
    ext = HeuristicExtractor(resolve_definite_np=True)
    roles = {"manager": "Erin"}
    assert ext.extract("The manager moved to Vienna.",
                       resolve_role=roles.get) == [
        {"subject": "Erin", "predicate": "city", "object": "Vienna"}]


def test_definite_np_fails_closed_on_ambiguity_and_unknown():
    ext = HeuristicExtractor(resolve_definite_np=True)
    # ambiguous role (backend returns None) -> no coreference fires
    assert ext.extract("The manager moved to Vienna.", resolve_role=lambda n: None) == []
    # feature OFF (default) -> byte-identical: no resolution even with a hook available
    off = HeuristicExtractor()
    assert off.extract("The manager moved to Vienna.") == []


def test_ellipsis_reuses_last_named_subject():
    ext = HeuristicExtractor(resolve_definite_np=True)
    assert ext.extract("Erin moved to Berlin. Relocated to Vienna.") == [
        {"subject": "Erin", "predicate": "city", "object": "Berlin"},
        {"subject": "Erin", "predicate": "city", "object": "Vienna"}]
    # no prior subject -> subjectless clause resolves to nothing (fail-closed)
    ext.reset()
    assert ext.extract("Moved to Vienna last spring.") == []
