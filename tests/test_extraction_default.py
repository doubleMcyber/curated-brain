"""Phase 1 — the honest configuration: extraction as the benchmarked path.

Three claims locked here:
1. First-person text (most real agent-memory input) extracts to facts about the declared
   speaker — previously it extracted to NOTHING (patterns required a capitalized name).
2. One predicate vocabulary: verb-extracted facts ("moved to") supersede and resolve
   against possessive/spoon-fed "city" facts (previously two schemas coexisted).
3. **AC-9 holds with the spoon-feeding withheld**: CuratedBrain ingesting the SAME raw
   text as every baseline (extractor-derived facts only, no ``metadata.fact``) still
   strictly beats naive-RAG, long-context, and no-memory on every category C1–C6.
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.eval import run_harness
from curated_brain.extraction import HeuristicExtractor, resolve_first_person


# ------------------------------------------------------------------ first person --------
def test_first_person_resolves_to_speaker():
    ext = HeuristicExtractor()
    assert ext.extract("My email address is erin@example.com.", speaker="Erin") == [
        {"subject": "Erin", "predicate": "email address", "object": "erin@example.com"}]
    assert ext.extract("I moved to Berlin.", speaker="erin") == [
        {"subject": "Erin", "predicate": "city", "object": "Berlin"}]
    assert ext.extract("I work at Acme.", speaker="Erin") == [
        {"subject": "Erin", "predicate": "employer", "object": "Acme"}]
    assert ext.extract("I'm a designer.", speaker="Erin") == [
        {"subject": "Erin", "predicate": "role", "object": "designer"}]
    assert ext.extract("I've moved to Oslo.", speaker="Erin") == [
        {"subject": "Erin", "predicate": "city", "object": "Oslo"}]
    # no speaker declared -> first-person text stays unparsed (no invented subject)
    assert ext.extract("I moved to Berlin.") == []


def test_resolve_first_person_is_pure_text():
    assert resolve_first_person("My city is Oslo and I am happy.", "kim") == \
        "Kim's city is Oslo and Kim is happy."


def test_speaker_flows_through_write_metadata():
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb.write("My email address is erin@example.com.", session_id="s", timestamp=0.0,
             metadata={"speaker": "Erin"})
    assert cb.answer_structured("Erin", "email address") == "erin@example.com"
    # without a declared speaker the same text asserts nothing (opt-in, not guessed)
    cb2 = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb2.write("My email address is erin@example.com.", session_id="s", timestamp=0.0)
    assert cb2.answer_structured("Erin", "email address") == ""


# ------------------------------------------------------------ one predicate vocabulary --
def test_verb_and_possessive_forms_share_one_predicate():
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb.write("Erin's city is Berlin.", session_id="s", timestamp=0.0)
    cb.write("Erin has moved to Vienna.", session_id="s", timestamp=1.0)  # verb form
    # one key, superseded once — not two parallel predicates ("city" vs "location")
    assert cb.answer_structured("Erin", "city") == "Vienna"
    assert len(cb.structured.history("Erin", "city")) == 2
    assert cb.answer_structured("Erin", "location") == ""  # the second schema is gone


def test_echo_of_a_superseded_statement_does_not_resurrect_it():
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    cb.write("Alice lives in Riga.", session_id="s", timestamp=0.0)
    cb.write("Alice has moved to Tallinn.", session_id="s", timestamp=1.0)
    # a later VERBATIM restatement of the old line (quote/redelivery) must not flip the
    # current value back — it is reinforcement (Pillar B), not a new assertion
    cb.write("Alice lives in Riga.", session_id="s", timestamp=2.0)
    assert cb.answer_structured("Alice", "city") == "Tallinn"
    # a genuinely NEW assertion (fresh phrasing) still updates normally
    cb.write("Alice has moved to Riga.", session_id="s", timestamp=3.0)
    assert cb.answer_structured("Alice", "city") == "Riga"


# ----------------------------------------------------------- AC-9 without spoon-feeding --
def test_ac9_holds_with_extraction_instead_of_spoonfed_facts():
    """The headline honesty fix: CB derives its own facts from the same raw text the
    baselines see, and still strictly beats all three on every category."""
    scores, _ = run_harness(seed=0, extraction=True)
    cur = scores["curated"]
    for cat in ("C1", "C2", "C3", "C4", "C5", "C6"):
        rival_best = max(scores[b][cat] for b in ("naive", "long_context", "no_memory"))
        assert cur[cat] > rival_best, f"{cat}: curated {cur[cat]} vs best rival {rival_best}"
    # and the extraction path gives up nothing material vs the spoon-fed wiring on this
    # suite (tolerance covers C3's token-count jitter from slightly different phrasing)
    spoon, _ = run_harness(seed=0, extraction=False)
    for cat in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert cur[cat] >= spoon["curated"][cat] - 0.01
