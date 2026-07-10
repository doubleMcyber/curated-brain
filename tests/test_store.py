"""Durable SQLite store with an incremental journal (Track C).

An attached :class:`SqliteStore` journals each write as one O(1) row and only periodically
rewrites a full snapshot (compaction). Reopening replays the last snapshot + committed journal,
reproducing byte-identical state — so attaching a store never changes in-memory behavior and a
crash mid-journal loses nothing that committed.
"""

from __future__ import annotations

from curated_brain.backend import CuratedBrain
from curated_brain.dataset import generate
from curated_brain.mcp_server import MemoryService
from curated_brain.store import SqliteStore


def _fill(cb: CuratedBrain, observations) -> None:
    for o in observations:
        cb.write(o.content, session_id=o.session_id, timestamp=o.wall_ts,
                 metadata={"fact": o.fact} if o.fact else None)


def test_reopen_from_disk_is_byte_identical(tmp_path):
    ds = generate(seed=0)
    obs = ds.observations[:120]
    last = ds.base_ts + (ds.n_sessions - 1) * ds.day

    ref = CuratedBrain(seed=0)  # a store-free control for the expected state
    _fill(ref, obs)

    path = str(tmp_path / "brain.sqlite")
    store = SqliteStore(path)
    cb = CuratedBrain(seed=0)
    cb.attach_store(store, compact_every=1000)  # no compaction: pure-journal path
    _fill(cb, obs)
    # (a) attaching a store must not change in-memory behavior.
    assert cb.snapshot() == ref.snapshot()
    store.close()

    # Reopen into a fresh brain from disk alone.
    store2 = SqliteStore(path)
    reopened = CuratedBrain(seed=0)
    reopened.attach_store(store2)
    assert reopened.snapshot() == ref.snapshot()
    for p in ds.by_category("C1")[:5] + ds.by_category("C2")[:5] + ds.by_category("C6")[:5]:
        assert (reopened.query(p.question, session_id="q", timestamp=last).context
                == ref.query(p.question, session_id="q", timestamp=last).context)
    store2.close()


def test_crash_mid_journal_replays_committed_writes(tmp_path):
    ds = generate(seed=0)
    obs = ds.observations[:80]

    path = str(tmp_path / "brain.sqlite")
    store = SqliteStore(path)
    cb = CuratedBrain(seed=0)
    cb.attach_store(store, compact_every=1000)  # never compacts -> every write is journal-only
    _fill(cb, obs)
    # Simulate a crash: abandon cb without any compaction/close-flush. The rows already committed
    # to the journal survive; a fresh connection reads exactly them.
    del cb
    store.close()  # closing just releases the fd — the committed rows are durable

    reopened = CuratedBrain(seed=0)
    store2 = SqliteStore(path)
    reopened.attach_store(store2)

    # State == a serial re-run of the committed writes.
    control = CuratedBrain(seed=0)
    _fill(control, obs)
    assert reopened.snapshot() == control.snapshot()
    store2.close()


def test_compaction_truncates_journal_and_reopen_still_matches(tmp_path):
    ds = generate(seed=0)
    obs = ds.observations[:100]

    path = str(tmp_path / "brain.sqlite")
    store = SqliteStore(path)
    cb = CuratedBrain(seed=0)
    cb.attach_store(store, compact_every=10)  # compact every 10 journal rows
    _fill(cb, obs)

    # After the last multiple-of-10 write the journal is empty and the snapshot row is current.
    blob, journal = store.load()
    assert blob is not None
    assert len(journal) == len(obs) % 10  # only rows since the last compaction remain
    store.close()

    reopened = CuratedBrain(seed=0)
    store2 = SqliteStore(path)
    reopened.attach_store(store2)
    control = CuratedBrain(seed=0)
    _fill(control, obs)
    assert reopened.snapshot() == control.snapshot()
    store2.close()


def test_store_load_matches_pure_json_roundtrip(tmp_path):
    ds = generate(seed=0)
    obs = ds.observations[:90]

    # Pure JSON save/load round-trip (the existing seam).
    json_cb = CuratedBrain(seed=0)
    _fill(json_cb, obs)
    json_path = str(tmp_path / "brain.cb")
    json_cb.save(json_path)
    json_reopened = CuratedBrain(seed=0)
    json_reopened.load(json_path)

    # Store-loaded brain at the same logical point.
    store_path = str(tmp_path / "brain.sqlite")
    store = SqliteStore(store_path)
    store_cb = CuratedBrain(seed=0)
    store_cb.attach_store(store)
    _fill(store_cb, obs)
    store.close()
    store2 = SqliteStore(store_path)
    store_reopened = CuratedBrain(seed=0)
    store_reopened.attach_store(store2)

    assert store_reopened.snapshot() == json_reopened.snapshot()
    store2.close()


def test_mcp_store_path_writes_do_not_rebuild_full_snapshot(tmp_path, monkeypatch):
    path = str(tmp_path / "svc.sqlite")
    svc = MemoryService(store_path=path, compact_every=1000)

    # snapshot() is O(store-size); in store_path mode it must NOT be called per write (only on
    # compaction, which compact_every=1000 defers past this test). Count real invocations.
    calls = {"n": 0}
    real_snapshot = svc.cb.snapshot

    def counting_snapshot():
        calls["n"] += 1
        return real_snapshot()

    monkeypatch.setattr(svc.cb, "snapshot", counting_snapshot)

    for i in range(50):
        svc.write(f"note number {i}", session_id="s", timestamp=1700000000.0 + i)
    assert calls["n"] == 0  # zero full-snapshot rebuilds across 50 writes

    svc.close()


def test_mcp_store_path_persists_across_restart(tmp_path):
    path = str(tmp_path / "svc.sqlite")
    svc = MemoryService(store_path=path, compact_every=1000)
    for i in range(20):
        svc.write(f"my favorite color is blue variant {i}", session_id="s",
                  timestamp=1700000000.0 + i)
    snap_before = svc.cb.snapshot()
    svc.close()

    reopened = MemoryService(store_path=path)
    assert reopened.cb.snapshot() == snap_before
    reopened.close()


def test_forget_with_store_reopens_forgotten(tmp_path):
    ds = generate(seed=0)
    obs = ds.observations[:120]

    path = str(tmp_path / "brain.sqlite")
    store = SqliteStore(path)
    cb = CuratedBrain(seed=0)
    cb.attach_store(store, compact_every=1000)
    _fill(cb, obs)
    cb.forget("Alice")  # state change outside the write stream -> immediate compaction
    after_forget = cb.snapshot()
    assert cb.answer_structured("Alice", "email") == ""  # gone in memory
    store.close()

    reopened = CuratedBrain(seed=0)
    store2 = SqliteStore(path)
    reopened.attach_store(store2)
    # Reopen reflects the forget: byte-identical to the post-forget state, and Alice stays gone.
    assert reopened.snapshot() == after_forget
    assert reopened.answer_structured("Alice", "email") == ""
    store2.close()
