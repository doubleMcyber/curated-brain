"""Durable persistence (Track C): a populated store survives a process restart.

`save()` writes the canonical snapshot to disk; a fresh `CuratedBrain` `load()`s it and is
byte-identical and answers identically — so an agent can stop and resume without loss.
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.dataset import generate


def test_save_load_roundtrip_survives_a_fresh_instance(tmp_path):
    ds = generate(seed=0)
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day
    cb = CuratedBrain(seed=0)
    for o in ds.observations:
        cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                 metadata={"fact": o.fact} if o.fact else None)

    path = tmp_path / "brain.cb"
    cb.save(str(path))
    assert path.exists() and path.stat().st_size > 0

    reopened = CuratedBrain(seed=0)  # a fresh "process"
    reopened.load(str(path))

    # state is byte-identical, and answers agree across a representative probe sample
    assert reopened.snapshot() == cb.snapshot()
    for p in ds.by_category("C1")[:5] + ds.by_category("C2")[:5] + ds.by_category("C6")[:5]:
        assert (reopened.query(p.question, session_id="q", timestamp=last).context
                == cb.query(p.question, session_id="q", timestamp=last).context)
