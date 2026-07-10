"""Unit tests for the deterministic event-date resolver (curated_brain.dates)."""

from __future__ import annotations

from datetime import UTC, datetime

from curated_brain.dates import resolve_event_date, strip_dates

REF = datetime(2023, 5, 28, tzinfo=UTC).timestamp()  # a Sunday in late May 2023


def _ymd(ts):
    d = datetime.fromtimestamp(ts, tz=UTC)
    return (d.year, d.month, d.day)


def test_absolute_iso():
    assert _ymd(resolve_event_date("It happened on 2023-03-15.", REF)) == (2023, 3, 15)


def test_absolute_numeric_mdy():
    assert _ymd(resolve_event_date("Signed 3/15/2023 in the office.", REF)) == (2023, 3, 15)


def test_month_day_with_year():
    assert _ymd(resolve_event_date("We met on March 3rd, 2022.", REF)) == (2022, 3, 3)


def test_month_day_no_year_takes_most_recent_past():
    # ref is May 28 2023 -> "March 3" means this year's March
    assert _ymd(resolve_event_date("Back in March 3 we launched.", REF)) == (2023, 3, 3)
    # a month later in the year than ref -> previous year
    assert _ymd(resolve_event_date("On December 25 we celebrated.", REF)) == (2022, 12, 25)


def test_relative_months_ago_is_calendar_aware():
    assert _ymd(resolve_event_date("I moved two months ago.", REF)) == (2023, 3, 28)


def test_relative_weeks_and_years_ago():
    assert _ymd(resolve_event_date("Started 3 weeks ago.", REF)) == (2023, 5, 7)
    assert _ymd(resolve_event_date("A year ago I quit.", REF)) == (2022, 5, 28)


def test_last_month_name():
    assert _ymd(resolve_event_date("We spoke last February.", REF)) == (2023, 2, 1)
    assert _ymd(resolve_event_date("It was last December.", REF)) == (2022, 12, 1)


def test_last_unit_and_yesterday():
    assert _ymd(resolve_event_date("I saw it last week.", REF)) == (2023, 5, 21)
    assert _ymd(resolve_event_date("It broke last month.", REF)) == (2023, 4, 28)
    assert _ymd(resolve_event_date("Fixed it yesterday.", REF)) == (2023, 5, 27)


def test_month_end_clamp():
    ref = datetime(2023, 3, 31, tzinfo=UTC).timestamp()
    # one month before Mar 31 clamps to Feb 28 (2023 is not a leap year)
    assert _ymd(resolve_event_date("one month ago", ref)) == (2023, 2, 28)


def test_no_date_returns_none():
    assert resolve_event_date("I really like strong coffee.", REF) is None
    assert resolve_event_date("", REF) is None


def test_non_finite_ref_returns_none():
    assert resolve_event_date("two months ago", float("inf")) is None
    assert resolve_event_date("two months ago", float("nan")) is None


def test_absolute_beats_relative_when_both_present():
    # an explicit date takes priority over a relative phrase in the same text
    assert _ymd(resolve_event_date("On 2022-01-10, about two months ago-ish.", REF)) \
        == (2022, 1, 10)


def test_deterministic():
    a = resolve_event_date("two months ago", REF)
    b = resolve_event_date("two months ago", REF)
    assert a == b


def test_strip_dates_removes_phrase_and_dangling_connective():
    assert strip_dates("Vienna two months ago") == "Vienna"
    assert strip_dates("Vienna on March 3") == "Vienna"        # dangling "on" removed
    assert strip_dates("moved back on 2023-03-15") == "moved"  # "back on" both stripped


def test_strip_dates_does_not_over_strip():
    # no date phrase -> unchanged; a hyphen-joined "in"/"on" is not a dangling connective
    assert strip_dates("Vienna") == "Vienna"
    assert strip_dates("check-in") == "check-in"
    assert strip_dates("Room 12") == "Room 12"
