"""Track B eval — does extraction-ON preserve the structured-tier capabilities end-to-end?

A scoped, reproducible A/B: the SAME world is loaded two ways — (1) the legacy path with the
dataset's spoon-fed ``metadata.fact``, and (2) raw text only, facts derived by the wired-in
``LLMExtractor``. Both must answer the same C1/C2/C5/C6 probes. The extraction side replays a
cassette of GENUINE completions recorded from a cached ``Qwen3.5-0.8B`` (CPU, greedy), so this
is a real signal — not a mock — and runs deterministically offline.

This is the in-repo, fast capability eval; the full LongMemEval head-to-head vs Mem0/Letta/Zep
(Track D) is separate and needs external systems + real compute (see PROGRESS.md). The one
extraction miss (the weak model drops Frank's project) is asserted explicitly, not hidden.
"""

from __future__ import annotations

from pathlib import Path

from curated_brain.backend import CuratedBrain
from curated_brain.cassette import CachedLLM, Cassette
from curated_brain.extraction import LLMExtractor
from curated_brain.fakes import DeterministicEmbedder

FIXTURE = Path(__file__).parent / "fixtures" / "beval_cassette.json"

# (raw observation, wall_ts, the gold triple a perfect extractor / the spoon-feed would yield)
SCENARIO = [
    ("Erin lives in Vienna.", 0.0,
     {"subject": "Erin", "predicate": "city", "object": "Vienna"}),
    ("Bob was promoted to engineering manager.", 0.0,
     {"subject": "Bob", "predicate": "role", "object": "engineering manager"}),
    ("Cara's email address is cara@example.com.", 0.0,
     {"subject": "Cara", "predicate": "email", "object": "cara@example.com"}),
    ("Dana's manager is Erin.", 0.0,
     {"subject": "Dana", "predicate": "manager", "object": "Erin"}),
    ("Erin moved to Berlin.", 100.0,
     {"subject": "Erin", "predicate": "city", "object": "Berlin"}),
    ("Frank now leads the Apollo project.", 0.0,
     {"subject": "Frank", "predicate": "project", "object": "Apollo"}),
]


def _spoonfed() -> CuratedBrain:
    cb = CuratedBrain(embedder=DeterministicEmbedder(64), dim=64, seed=0)
    for text, ts, fact in SCENARIO:
        cb.write(text, session_id="s", timestamp=ts, metadata={"fact": fact})
    return cb


def _extraction_on() -> CuratedBrain:
    ext = LLMExtractor(CachedLLM(Cassette.load(str(FIXTURE)), inner=None))
    cb = CuratedBrain(embedder=DeterministicEmbedder(64), dim=64, seed=0, extractor=ext)
    for text, ts, _ in SCENARIO:
        cb.write(text, session_id="s", timestamp=ts)  # raw text only — no metadata.fact
    return cb


def test_extraction_on_matches_spoonfed_on_C1_C2_C5_C6():
    spoonfed, extracted = _spoonfed(), _extraction_on()
    for cb in (spoonfed, extracted):
        # C1 — long-range exact recall
        assert cb.answer_structured("Cara", "email") == "cara@example.com"
        assert cb.answer_structured("Bob", "role") == "engineering manager"
        # C2 — belief update: the current value is the new one (supersede worked)
        assert cb.answer_structured("Erin", "city") == "Berlin"
        # C6 — as-of-time: before the move, the old value
        assert cb.answer_structured("Erin", "city", at=50.0) == "Vienna"
        # C5 — relational multi-hop (Dana -> manager -> Erin -> city)
        assert cb.answer_path("Dana", ["manager", "city"]) == "Berlin"
        # C5 x C6 — multi-hop *and* as-of-time
        assert cb.answer_path("Dana", ["manager", "city"], at=50.0) == "Vienna"


def test_extraction_miss_is_surfaced_not_hidden():
    # Honest accounting: the small model failed to extract Frank's project, so extraction-ON
    # loses a fact the spoon-feed keeps. This is the documented motivation to use a stronger
    # model for the Track D benchmark — not something to paper over.
    assert _spoonfed().answer_structured("Frank", "project") == "Apollo"
    assert _extraction_on().answer_structured("Frank", "project") == ""
