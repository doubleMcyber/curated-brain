"""Deliverable A: HeuristicExtractor(resolve_dates=True) sets a fact's valid_from to the
event date stated in the text ("...two months ago", "on 2023-03-15"), so bi-temporal
valid-time is correct for retrospectively-stated events. Opt-in: default off is byte-identical."""

from __future__ import annotations

from datetime import UTC, datetime

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor

REF = datetime(2023, 5, 28, tzinfo=UTC).timestamp()
MAR_28 = datetime(2023, 3, 28, tzinfo=UTC).timestamp()  # two months before REF


def test_default_off_is_unchanged():
    ex = HeuristicExtractor()  # resolve_dates defaults False
    facts = ex.extract("Erin moved to Vienna two months ago.", ref_ts=REF)
    assert facts == [{"subject": "Erin", "predicate": "city", "object": "Vienna two months ago"}]
    assert "valid_from" not in facts[0]


def test_resolve_dates_sets_valid_from_and_cleans_object():
    ex = HeuristicExtractor(resolve_dates=True)
    facts = ex.extract("Erin moved to Vienna two months ago.", ref_ts=REF)
    assert facts[0]["subject"] == "Erin" and facts[0]["predicate"] == "city"
    assert facts[0]["object"] == "Vienna"          # trailing date phrase stripped
    assert facts[0]["valid_from"] == MAR_28        # lifted into valid_from


def test_date_at_front_of_clause():
    ex = HeuristicExtractor(resolve_dates=True)
    facts = ex.extract("Two months ago, Erin moved to Vienna.", ref_ts=REF)
    assert facts[0]["object"] == "Vienna" and facts[0]["valid_from"] == MAR_28


def test_no_date_leaves_valid_from_unset():
    ex = HeuristicExtractor(resolve_dates=True)
    facts = ex.extract("Erin moved to Vienna.", ref_ts=REF)
    assert facts[0]["object"] == "Vienna" and "valid_from" not in facts[0]


def test_no_ref_ts_is_a_noop():
    ex = HeuristicExtractor(resolve_dates=True)
    facts = ex.extract("Erin moved to Vienna two months ago.")  # ref_ts=None
    assert "valid_from" not in facts[0]


def test_end_to_end_valid_time_decoupled_from_transaction_time():
    # The fact is RECORDED at REF but VALID from two months earlier — the bi-temporal split.
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor(resolve_dates=True))
    cb.write("Two months ago, Erin moved to Vienna.", session_id="s", timestamp=REF)
    f = cb.structured.resolve("erin", "city")
    assert f is not None and f.object == "Vienna"
    assert f.valid_from == MAR_28   # true event time
    assert f.created_at == REF      # transaction time = when we recorded it
