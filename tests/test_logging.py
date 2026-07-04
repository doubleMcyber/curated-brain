"""Track-G observability: the library emits structured logs at key decision/degradation points
(silent by default via NullHandler), and never logs observation content (PII) or secrets."""

from __future__ import annotations

import logging

from curated_brain.backend import CuratedBrain


def _fact(s, p, o):
    return {"fact": {"subject": s, "predicate": p, "object": o}}


def test_package_logger_is_silent_by_default(caplog):
    # No handler configured by the app -> NullHandler swallows everything (no stderr noise).
    log = logging.getLogger("curated_brain")
    assert any(isinstance(h, logging.NullHandler) for h in log.handlers)


def test_write_decision_is_logged_without_content(caplog):
    cb = CuratedBrain(seed=0)
    with caplog.at_level(logging.DEBUG, logger="curated_brain"):
        cb.write("Erin's secret diary entry about Vienna.", session_id="s", timestamp=0.0,
                 metadata=_fact("Erin", "city", "Vienna"))
    recs = [r for r in caplog.records if "write decision" in r.getMessage()]
    assert recs, "expected a write-decision log record"
    msg = recs[0].getMessage()
    assert "decision=stored" in msg
    # the observation TEXT must never appear in the log (no PII leakage)
    assert "secret diary" not in msg


def test_consolidate_is_logged(caplog):
    cb = CuratedBrain(seed=0)
    cb.write("Erin's city is Vienna.", session_id="s", timestamp=0.0,
             metadata=_fact("Erin", "city", "Vienna"))
    with caplog.at_level(logging.INFO, logger="curated_brain"):
        cb.consolidate()
    assert any("consolidate:" in r.getMessage() for r in caplog.records)


def test_logging_does_not_perturb_determinism():
    # Logging is purely observational: a store built with logging enabled must snapshot
    # byte-identically to one built with it disabled.
    def build():
        cb = CuratedBrain(seed=0)
        for i, (s, o) in enumerate([("Erin", "Vienna"), ("Bob", "Berlin"), ("Erin", "Oslo")]):
            cb.write(f"{s} note {i}", session_id="s", timestamp=float(i),
                     metadata=_fact(s, "city", o))
        cb.consolidate()
        return cb.snapshot()

    logging.getLogger("curated_brain").setLevel(logging.DEBUG)
    a = build()
    logging.getLogger("curated_brain").setLevel(logging.CRITICAL)
    b = build()
    assert a == b  # log level does not change computed state
