"""Time-scoped retrieval: query(window=(from_ts, to_ts)) scopes episodic recall to a time
range — a core memory capability ('what did I note last spring?'). Additive + AC-1 safe:
default window=None is byte-identical to before."""

from __future__ import annotations

import pytest

from curated_brain.backend import CuratedBrain


def _cb_with_timeline() -> CuratedBrain:
    cb = CuratedBrain(seed=0)
    # three dated notes about the same topic at t=100, 200, 300
    cb.write("Project Falcon kicked off in January.", session_id="s", timestamp=100.0)
    cb.write("Project Falcon hit a milestone in April.", session_id="s", timestamp=200.0)
    cb.write("Project Falcon shipped in July.", session_id="s", timestamp=300.0)
    return cb


def test_window_scopes_recall_to_the_time_range():
    cb = _cb_with_timeline()
    # scope to [150, 250] -> only the April note is eligible
    r = cb.query("Falcon", session_id="q", timestamp=1000.0, window=(150.0, 250.0))
    assert "April" in r.context
    assert "January" not in r.context and "July" not in r.context


def test_no_window_returns_unscoped():
    cb = _cb_with_timeline()
    r = cb.query("Falcon", session_id="q", timestamp=1000.0)  # no window
    # all three eras reachable within the k budget (no time filter)
    ctx = r.context
    assert sum(m in ctx for m in ("January", "April", "July")) >= 2


def test_window_default_is_byte_identical():
    # AC-1: adding the optional param must not change the default retrieval path.
    def run():
        cb = _cb_with_timeline()
        cb.write("x", session_id="s", timestamp=400.0,
                 metadata={"fact": {"subject": "Falcon", "predicate": "status",
                                    "object": "shipped"}})
        return (cb.query("Falcon", session_id="q", timestamp=1000.0).context,
                cb.snapshot())
    a_ctx, a_snap = run()
    b_ctx, b_snap = run()
    assert a_ctx == b_ctx and a_snap == b_snap


def test_malformed_window_raises():
    cb = _cb_with_timeline()
    # reversed, non-finite, wrong arity, non-tuple scalar, a list, and non-numeric all
    # raise a clear ValueError (not a downstream TypeError)
    for bad in ((250.0, 150.0), (float("inf"), 200.0), (1.0, 2.0, 3.0), 5.0,
                [1.0, 2.0], ("a", "b")):
        with pytest.raises(ValueError, match="window"):
            cb.query("Falcon", session_id="q", timestamp=1000.0, window=bad)
