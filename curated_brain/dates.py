"""Deterministic, offline event-date resolution.

Longitudinal memory needs to know *when an event happened*, which is often stated in the
turn text ("I moved two months ago", "we met last February", "on 2023-03-15") rather than
being the session's own clock. :func:`resolve_event_date` turns such an expression into an
absolute UTC timestamp, relative to a caller-supplied reference clock (never the real clock,
so runs stay reproducible). It returns ``None`` when the text carries no date expression, so
callers fall back to the observation timestamp unchanged.

This is a focused resolver for the common English forms, not a general date parser — it is
hand-rolled on purpose to keep the library offline and dependency-free (numpy-only core)."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}  # for "N <unit> ago" fallback
_NUMWORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))  # longest-first so "sept" wins

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_NUMERIC_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")  # M/D/Y (US order)
_MONTHDAY_RE = re.compile(
    rf"\b({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b", re.I)
_LAST_MONTH_RE = re.compile(rf"\b(?:last|this|in)\s+({_MONTH_ALT})\b", re.I)
_AGO_RE = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(day|week|month|year)s?\s+ago\b", re.I)
_LAST_UNIT_RE = re.compile(r"\b(?:last|past|previous)\s+(week|month|year)\b", re.I)
_YESTERDAY_RE = re.compile(r"\byesterday\b", re.I)


_DATE_PHRASE_RES = (_ISO_RE, _NUMERIC_RE, _MONTHDAY_RE, _AGO_RE, _LAST_MONTH_RE,
                    _LAST_UNIT_RE, _YESTERDAY_RE)
_TIDY_RE = re.compile(r"\s{2,}")
# A temporal connective left dangling once the date it introduced is removed ("Vienna on
# March 3" -> "Vienna on" -> "Vienna"). Stripped only from the trailing edge, iteratively so
# "back on <date>" -> "back" -> "".
_TRAILING_CONN_RE = re.compile(r"[\s,]+(?:on|in|at|around|back|since|by|from|until)$", re.I)


def strip_dates(text: str) -> str:
    """Remove recognized date expressions from ``text`` and tidy leftover whitespace/commas
    (incl. a now-dangling trailing connective). Used to keep an extracted object value clean
    once its date has been lifted into ``valid_from`` ("Vienna two months ago" -> "Vienna")."""
    for rex in _DATE_PHRASE_RES:
        text = rex.sub(" ", text)
    text = _TIDY_RE.sub(" ", text).strip(" ,;")
    while (trimmed := _TRAILING_CONN_RE.sub("", text)) != text:
        text = trimmed
    return text


def _to_ts(dt: datetime) -> float:
    return dt.timestamp()


def _shift_months(dt: datetime, months: int) -> datetime:
    """Subtract/add whole calendar months, clamping the day to the target month's length
    (e.g. Mar 31 minus one month -> Feb 28/29). Calendar-aware, deterministic."""
    total = (dt.year * 12 + (dt.month - 1)) + months
    year, month = divmod(total, 12)
    month += 1
    # days in the target month (next-month day-0 trick)
    if month == 12:
        last = 31
    else:
        last = (datetime(year, month + 1, 1, tzinfo=UTC) - timedelta(days=1)).day
    return dt.replace(year=year, month=month, day=min(dt.day, last))


def resolve_event_date(text: str, ref_ts: float) -> float | None:
    """Resolve an event date stated in ``text`` to an absolute UTC timestamp, relative to
    ``ref_ts`` (the observation's wall clock). Returns ``None`` if no date expression is
    found. Deterministic and offline. Priority: explicit absolute dates first, then relative
    ("N units ago" / "last <month|unit>" / "yesterday"). Only the first match is used."""
    if not isinstance(text, str) or not math.isfinite(ref_ts):
        return None
    ref = datetime.fromtimestamp(ref_ts, tz=UTC)

    # --- absolute forms (most specific / unambiguous) ---
    if m := _ISO_RE.search(text):
        y, mo, d = (int(g) for g in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                return _to_ts(datetime(y, mo, d, tzinfo=UTC))
            except ValueError:
                return None
    if m := _NUMERIC_RE.search(text):
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                return _to_ts(datetime(y, mo, d, tzinfo=UTC))
            except ValueError:
                return None
    if m := _MONTHDAY_RE.search(text):
        mo = _MONTHS[m.group(1).lower()]
        d = int(m.group(2))
        if m.group(3):  # explicit year
            try:
                return _to_ts(datetime(int(m.group(3)), mo, d, tzinfo=UTC))
            except ValueError:
                return None
        # no year: the most recent occurrence of this month/day on-or-before the reference
        for y in (ref.year, ref.year - 1):
            try:
                cand = datetime(y, mo, d, tzinfo=UTC)
            except ValueError:
                return None
            if cand <= ref:
                return _to_ts(cand)
        return _to_ts(datetime(ref.year - 1, mo, d, tzinfo=UTC))

    # --- relative forms ---
    if m := _AGO_RE.search(text):
        n_raw = m.group(1).lower()
        n = int(n_raw) if n_raw.isdigit() else _NUMWORDS[n_raw]
        unit = m.group(2).lower()
        if unit == "month":
            return _to_ts(_shift_months(ref, -n))
        if unit == "year":
            return _to_ts(_shift_months(ref, -12 * n))
        return _to_ts(ref - timedelta(days=n * _UNIT_DAYS[unit]))
    if m := _LAST_MONTH_RE.search(text):
        mo = _MONTHS[m.group(1).lower()]
        # the most recent occurrence of that month before/at the reference month
        y = ref.year if mo <= ref.month else ref.year - 1
        return _to_ts(datetime(y, mo, 1, tzinfo=UTC))
    if m := _LAST_UNIT_RE.search(text):
        unit = m.group(1).lower()
        if unit == "month":
            return _to_ts(_shift_months(ref, -1))
        if unit == "year":
            return _to_ts(_shift_months(ref, -12))
        return _to_ts(ref - timedelta(days=7))
    if _YESTERDAY_RE.search(text):
        return _to_ts(ref - timedelta(days=1))
    return None
