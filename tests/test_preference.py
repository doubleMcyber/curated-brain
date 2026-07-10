"""Preference predicate family — heuristic extraction + preference-intent aggregation.

Preference facts share a ``preference:<topic>`` predicate family and encode polarity in the
object ("like"/"dislike") for bare verb forms, or carry the value verbatim for the
category ("favorite <category>") form. Keying the predicate on the object's head (bare form)
or the explicit category (favorite form) makes the structured tier's supersede fire per topic:
"likes jazz" then "hates jazz" flips polarity, while "likes hiking" stays open alongside; the
bi-temporal history keeps the earlier value queryable as-of.

The query side recognizes preference intent (like/favorite/prefer/... tokens naming a known
entity) and aggregates every open ``preference:*`` fact for that entity, reusing the
structured tier's ``predicates_for`` open-fact listing.
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor
from curated_brain.fakes import DeterministicEmbedder


def _brain() -> CuratedBrain:
    return CuratedBrain(embedder=DeterministicEmbedder(64), dim=64, seed=0,
                        extractor=HeuristicExtractor(extract_preferences=True))


# --------------------------------------------------------------- extraction forms -------
def test_bare_preference_verbs_extract_polarity():
    ext = HeuristicExtractor(extract_preferences=True)
    # every "like" verb -> predicate keyed on the object head, polarity "like"
    for verb in ("likes", "loves", "enjoys", "prefers", "favors"):
        assert ext.extract(f"Erin {verb} jazz.") == [
            {"subject": "Erin", "predicate": "preference:jazz", "object": "like"}]
    # negative forms -> polarity "dislike"
    for verb in ("dislikes", "hates"):
        assert ext.extract(f"Erin {verb} crowds.") == [
            {"subject": "Erin", "predicate": "preference:crowds", "object": "dislike"}]


def test_favorite_possessive_lands_in_preference_family():
    ext = HeuristicExtractor(extract_preferences=True)
    # the possessive "favorite <category>" is UNIFIED into preference:<category>, not a
    # disjoint "favorite cuisine" predicate.
    assert ext.extract("Erin's favorite cuisine is Thai.") == [
        {"subject": "Erin", "predicate": "preference:cuisine", "object": "Thai"}]
    assert ext.extract("Erin's favourite music is jazz.") == [
        {"subject": "Erin", "predicate": "preference:music", "object": "jazz"}]


def test_first_person_preference_via_speaker():
    ext = HeuristicExtractor(extract_preferences=True)
    assert ext.extract("I love hiking.", speaker="erin") == [
        {"subject": "Erin", "predicate": "preference:hiking", "object": "like"}]


# --------------------------------------------------------------- supersede semantics ----
def test_polarity_flip_supersedes_and_history_survives():
    cb = _brain()
    cb.write("Erin likes jazz.", session_id="s1", timestamp=1.0)
    cb.write("Erin hates jazz.", session_id="s2", timestamp=3.0)
    # current shows the flip; the value history keeps the earlier "like" queryable as-of
    assert cb.answer_structured("Erin", "preference:jazz") == "dislike"
    assert cb.answer_structured("Erin", "preference:jazz", at=1.5) == "like"
    hist = cb.structured.history("Erin", "preference:jazz")
    assert [f.object for f in hist] == ["like", "dislike"]


def test_different_objects_do_not_clobber():
    cb = _brain()
    cb.write("Erin likes jazz.", session_id="s1", timestamp=1.0)
    cb.write("Erin likes hiking.", session_id="s1", timestamp=2.0)
    # distinct topics -> distinct predicates -> both open
    assert cb.answer_structured("Erin", "preference:jazz") == "like"
    assert cb.answer_structured("Erin", "preference:hiking") == "like"


def test_category_form_supersedes_within_category_only():
    cb = _brain()
    cb.write("Erin's favorite cuisine is Thai.", session_id="s1", timestamp=1.0)
    cb.write("Erin likes jazz.", session_id="s1", timestamp=2.0)
    cb.write("Erin's favorite cuisine is Italian.", session_id="s2", timestamp=3.0)
    # the cuisine value flips; the unrelated jazz preference is untouched
    assert cb.answer_structured("Erin", "preference:cuisine") == "Italian"
    assert cb.answer_structured("Erin", "preference:cuisine", at=1.5) == "Thai"
    assert cb.answer_structured("Erin", "preference:jazz") == "like"


# --------------------------------------------------------------- end-to-end query -------
def test_query_aggregates_open_preferences():
    cb = _brain()
    cb.write("Erin likes jazz.", session_id="s1", timestamp=1.0)
    cb.write("Erin likes hiking.", session_id="s1", timestamp=2.0)
    cb.write("Erin hates crowds.", session_id="s2", timestamp=3.0)
    r = cb.query("What does Erin like?", session_id="q", timestamp=5.0)
    # every open preference surfaces, rendered readably
    assert "Erin likes jazz." in r.context
    assert "Erin likes hiking." in r.context
    assert "Erin dislikes crowds." in r.context


def test_query_answers_favorite_category():
    cb = _brain()
    cb.write("Erin's favorite cuisine is Thai.", session_id="s1", timestamp=1.0)
    cb.write("Erin's favorite cuisine is Italian.", session_id="s2", timestamp=2.0)
    r = cb.query("What is Erin's favorite cuisine?", session_id="q", timestamp=5.0)
    # the current category value is surfaced; the superseded one is not the current answer
    assert "Erin's favorite cuisine is Italian." in r.context


def test_preference_intent_requires_a_known_entity():
    cb = _brain()
    cb.write("Erin likes jazz.", session_id="s1", timestamp=1.0)
    # an entity-less preference question does not route to the aggregation
    plan = cb.planner.plan("What does everyone like?", entities=cb._entities,
                           predicates=frozenset(), session_ts=cb._session_ts)
    assert plan.preference is False
    # a preference question naming the entity does — but only when preference facts are
    # actually stored (schema-driven gate)
    plan = cb.planner.plan("What does Erin like?", entities=cb._entities,
                           predicates=frozenset({"preference:jazz"}), session_ts=cb._session_ts)
    assert plan.entity == "erin" and plan.preference is True
    # with NO stored preference facts the same question keeps its exact old plan
    plan = cb.planner.plan("What does Erin like?", entities=cb._entities,
                           predicates=frozenset(), session_ts=cb._session_ts)
    assert plan.preference is False


# Comparative preferences fail closed: head-noun keying would merge both alternatives into
# one topic, so a later genuine reversal would silently no-op — better to store nothing.
def test_comparative_preference_fails_closed():
    ext = HeuristicExtractor(extract_preferences=True)
    assert ext.extract("Erin prefers window seats to aisle seats.") == []
    assert ext.extract("Erin prefers tea over coffee.") == []
    assert ext.extract("Erin likes tea rather than coffee.") == []


# Preference extraction is opt-in: the DEFAULT extractor must not fire on preference verbs
# at all (always-on extraction was rejected by the held-out diagnostic gate).
def test_preference_extraction_is_opt_in():
    default = HeuristicExtractor()
    assert default.extract("Erin likes jazz.") == []
    # the favorite-possessive sentence falls back to the LEGACY possessive path (predicate
    # "favorite cuisine"), exactly as before this feature existed — no preference:* fact.
    facts = default.extract("Erin's favorite cuisine is Thai.")
    assert all(not f["predicate"].startswith("preference:") for f in facts)
