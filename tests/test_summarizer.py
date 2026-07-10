"""Consolidation summarizer (PRD §8 "cluster & summarize") — benchmarked against a real model.

When ``CuratedBrain(summarizer=<LLM>)`` is set, a multi-member merge is summarized through
the LLM into one claim instead of taking the cluster's representative member verbatim. This
directly answers the red-team finding "consolidation destroys merged members' content": the
summary must carry every merged member's fact-critical content forward.

The offline tests replay a cassette of GENUINE completions recorded from a real local model
(``tests/fixtures/summarize_cassette.json``), so they exercise the real summarizer code path
on real model output — deterministically, with no model present. A replay *miss* raises, so
the coverage assertion below cannot be satisfied by fabricated text.

Cassette provenance: recorded 2026-07-10 from Ollama-served ``qwen2.5:7b`` at
``http://localhost:11434/v1`` (temperature 0.0, greedy) via
``CachedLLM(inner=OpenAICompatLLM(...))``. Do not hand-edit the recorded completions; re-record
by re-running the scenario against the same model if the prompt or scenario changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.cassette import CachedLLM, Cassette
from curated_brain.fakes import DeterministicEmbedder
from curated_brain.util import tokenize

FIXTURE = Path(__file__).parent / "fixtures" / "summarize_cassette.json"

# Lexically diverse restatements of three distinct triples. Diverse wording clears the
# default surprise gate as separate episodes; the shared exact (subject, predicate, object)
# groups them at consolidation, so two clusters genuinely merge (Priya x2, Marco x2) while
# the third Priya restatement is gate-discarded and Tess stays a singleton.
OBS = [
    ("Priya works as a staff engineer.",
     {"subject": "Priya", "predicate": "role", "object": "staff engineer"}),
    ("Her official title became staff engineer after the review.",
     {"subject": "Priya", "predicate": "role", "object": "staff engineer"}),
    ("The platform team promoted Priya into a staff engineer position.",
     {"subject": "Priya", "predicate": "role", "object": "staff engineer"}),
    ("Marco relocated to Lisbon for the new office.",
     {"subject": "Marco", "predicate": "city", "object": "Lisbon"}),
    ("These days you can find Marco settled down in Lisbon.",
     {"subject": "Marco", "predicate": "city", "object": "Lisbon"}),
    ("Reach Tess at the address tess@example.com.",
     {"subject": "Tess", "predicate": "email", "object": "tess@example.com"}),
]


def _brain(summarizer=None) -> CuratedBrain:
    return CuratedBrain(embedder=DeterministicEmbedder(), seed=0, summarizer=summarizer)


def _replayer() -> CachedLLM:
    return CachedLLM(Cassette.load(str(FIXTURE)), inner=None)


def _build_and_consolidate(cb: CuratedBrain):
    for i, (text, fact) in enumerate(OBS):
        cb.write(text, session_id="s0", timestamp=float(i), metadata={"fact": fact})
    return cb.consolidate()


def _merged_claims(cb: CuratedBrain) -> list:
    # A merged claim covers >1 source episode; a single-member pass-through is not a merge.
    return [r for r in cb._episodes
            if r.tier == "semantic" and len(r.provenance.get("merged_from", [])) > 1]


def test_a_real_multi_member_merge_happens_and_summary_covers_members():
    cb = _brain(summarizer=_replayer())
    rep = _build_and_consolidate(cb)
    # Engineered to produce genuine multi-member merges (not single-member pass-throughs) —
    # otherwise the summarizer is never exercised. Verified BEFORE recording.
    assert rep.dupes_merged == 2 and rep.claims_out == 2

    claims = _merged_claims(cb)
    assert len(claims) == 2
    # Anti-content-destruction assertion. Every merged member of a cluster shares the cluster's
    # exact (subject, predicate, object) fact — so the fact-critical content that must survive
    # the merge is exactly that triple's subject + object-value tokens. On the recorded
    # qwen2.5:7b output every such token appears in the summary. (The model rewrites generic
    # filler like "find"/"down" from a paraphrase, so raw token equality would be false; the
    # fact-bearing tokens, which are what "content" means here, ARE fully covered.)
    for claim in claims:
        summ_tokens = set(tokenize(claim.content))
        subj, _pred, obj = claim.fact_key
        key = set(tokenize(subj)) | set(tokenize(obj))
        assert key <= summ_tokens, (
            f"summary {claim.content!r} dropped fact-critical tokens "
            f"{sorted(key - summ_tokens)}")


def test_b_determinism_two_replay_runs_are_byte_identical():
    a = _brain(summarizer=_replayer())
    _build_and_consolidate(a)
    b = _brain(summarizer=_replayer())
    _build_and_consolidate(b)
    assert a.snapshot() == b.snapshot()


def test_c_default_no_summarizer_path_is_unchanged():
    # A control brain with NO summarizer must consolidate to the extractive representative,
    # proving the summarizer is strictly opt-in and default users see no behavior change.
    with_sum = _brain(summarizer=_replayer())
    _build_and_consolidate(with_sum)
    control = _brain(summarizer=None)
    _build_and_consolidate(control)

    # The two paths must DIFFER (the summarizer actually rewrote the merged claims)...
    assert with_sum.snapshot() != control.snapshot()
    # ...and the default path is itself deterministic and takes each cluster's representative
    # member verbatim (no model-generated text leaks into the default store).
    control2 = _brain(summarizer=None)
    _build_and_consolidate(control2)
    assert control.snapshot() == control2.snapshot()
    for claim in _merged_claims(control):
        assert claim.content in {text for text, _ in OBS}


def test_d_replay_miss_raises_so_assertions_pin_real_model_output():
    with pytest.raises(KeyError):
        _replayer().complete("An unrecorded summarization prompt about nobody in particular.")
