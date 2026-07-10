"""Durable SQLite persistence with an incremental write journal (Track C).

The full-snapshot ``save()``/``load()`` seam rewrites the entire canonical JSON blob on every
persist — O(store-size) per write. This store replaces that cost on the hot path: each write
appends one small journal row (O(1) amortized) and only periodically rewrites a full snapshot
(compaction). On reopen, state is reconstructed by restoring the last snapshot and replaying
the journal rows committed after it.

Determinism is preserved because every write is reproducible from its raw args
(observation, session_id, timestamp, metadata) through the normal write path — the journal
stores exactly those args, so a replay reconstructs byte-identical state (AC-1).

Torn-write safety: sqlite3 in WAL mode commits each ``append``/``compact`` atomically. A crash
between commits leaves a consistent database whose journal holds every row that committed;
``load`` replays exactly those, so state == the serial re-run of the committed writes.
"""

from __future__ import annotations

import json
import sqlite3


class SqliteStore:
    """Snapshot + journal persistence over a single SQLite file.

    Two tables: ``snapshot`` holds at most one full canonical blob plus the journal ``seq`` it
    was taken at; ``journal`` holds the per-write op rows appended since that snapshot. The
    backend owns the semantics — this class is a thin, transactional durability layer and knows
    nothing about what an op means.

    Ownership: one store file belongs to ONE attached brain at a time (the library's
    single-writer-across-processes contract). Attaching two brains to the same path is not
    guarded and interleaves their journals/compactions into state matching neither writer."""

    def __init__(self, path: str) -> None:
        # ``isolation_level=None`` -> autocommit: each single-statement ``append`` commits
        # atomically on its own, and ``compact`` opens an EXPLICIT transaction (BEGIN IMMEDIATE
        # ... COMMIT below) so its delete+insert+clear is one atomic unit. (``with conn:`` does
        # NOT begin a transaction in autocommit mode — that was a real torn-compact bug: a crash
        # between the DELETE and INSERT autocommits lost the snapshot.)
        # ``check_same_thread=False`` because the attached brain's RLock already serializes every
        # store access; sqlite's per-thread guard would otherwise crash multi-threaded writers.
        self._conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            "CREATE TABLE IF NOT EXISTS snapshot("
            "  id INTEGER PRIMARY KEY, blob BLOB NOT NULL, created_seq INTEGER NOT NULL);"
            "CREATE TABLE IF NOT EXISTS journal("
            "  seq INTEGER PRIMARY KEY, op TEXT NOT NULL, payload TEXT NOT NULL);")

    def append(self, op: str, payload: dict) -> None:
        """Append one journal row for a successful write. Autoincrementing ``seq`` orders the
        replay; the payload is the raw JSON write args (no second serialization schema)."""
        self._conn.execute("INSERT INTO journal(op, payload) VALUES(?, ?)",
                            (op, json.dumps(payload, separators=(",", ":"), allow_nan=False)))

    def compact(self, blob: bytes) -> None:
        """Replace the snapshot with ``blob`` and clear the journal, atomically. After this the
        snapshot alone reconstructs current state, so the journal restarts empty. Records the
        current max journal ``seq`` as the snapshot's watermark (informational)."""
        # Explicit transaction — in autocommit mode ``with conn:`` would NOT begin one, and each
        # statement would commit independently (a crash mid-compact then loses the snapshot).
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute("SELECT MAX(seq) FROM journal").fetchone()
            created_seq = row[0] if row and row[0] is not None else 0
            self._conn.execute("DELETE FROM snapshot")
            self._conn.execute("INSERT INTO snapshot(id, blob, created_seq) VALUES(1, ?, ?)",
                               (blob, created_seq))
            self._conn.execute("DELETE FROM journal")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def load(self) -> tuple[bytes | None, list[tuple[str, dict]]]:
        """Return ``(snapshot_blob_or_None, [(op, payload_dict), ...])`` — the last snapshot and
        every journal row committed after it, in seq order. The backend restores the blob then
        re-executes the ops through its normal write path."""
        row = self._conn.execute("SELECT blob FROM snapshot WHERE id = 1").fetchone()
        blob: bytes | None = row[0] if row else None
        journal = [(op, json.loads(payload)) for op, payload in
                   self._conn.execute("SELECT op, payload FROM journal ORDER BY seq").fetchall()]
        return blob, journal

    def close(self) -> None:
        self._conn.close()
