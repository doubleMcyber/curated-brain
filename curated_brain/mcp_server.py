"""MCP server — mount The Curated Brain on any agent (Track E adoption surface).

Exposes the memory layer as MCP tools (`write` / `query` / `answer` / `consolidate` /
`stats`) so a Claude / agent host can use it as long-term memory. The operations live in a
plain :class:`MemoryService` (unit-testable without an MCP transport); :func:`build_server`
just wraps them as FastMCP tools. The ``mcp`` package is a soft dependency imported lazily,
so importing this module (or the rest of the library) never requires it.

Run it:  ``curated-brain-mcp``  (stdio transport; set ``CB_MCP_PATH`` to persist across runs).
"""

from __future__ import annotations

import os
import threading
import time

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor


class MemoryService:
    """The Curated Brain operations exposed over MCP, as plain (testable) methods.

    Defaults to the deterministic no-LLM :class:`HeuristicExtractor` so raw text sent by an
    agent populates the structured tier with no spoon-fed facts. Pass ``path`` to persist the
    store to disk (atomic tmp+rename; ``persist_every`` batches writes to amortize the
    O(store) snapshot cost — the store is also flushed on ``consolidate``).

    Timestamps: the core is clock-free by contract (the harness supplies time), but a live
    agent host has no harness — so THIS boundary defaults to wall-clock. The old defaults
    (write at t=0.0) stamped every fact at the epoch, silently disabling recency scoring,
    supersede ordering, and all as-of semantics for the flagship integration.

    Thread safety: a single lock serializes all operations — CuratedBrain has no internal
    locking, and MCP hosts may issue concurrent tool calls."""

    def __init__(self, cb: CuratedBrain | None = None, *, path: str | None = None,
                 persist_every: int = 1) -> None:
        self.cb = cb or CuratedBrain(seed=0, extractor=HeuristicExtractor())
        self._path = path
        self._persist_every = max(1, persist_every)
        self._dirty = 0
        self._lock = threading.Lock()

    def write(self, observation: str, session_id: str = "default",
              timestamp: float | None = None, speaker: str = "User") -> dict:
        with self._lock:
            ts = time.time() if timestamp is None else timestamp
            # Declare the speaker so first-person text ("My email is …") extracts as facts
            # about them — a single-user server defaults to "User".
            r = self.cb.write(observation, session_id=session_id, timestamp=ts,
                              metadata={"speaker": speaker})
            self._dirty += 1
            if self._dirty >= self._persist_every:
                self._persist()
            return {"stored": r.stored, "reason": r.reason, "record_id": r.record_id}

    def query(self, question: str, session_id: str = "default",
              timestamp: float | None = None, k: int = 8) -> str:
        with self._lock:
            ts = time.time() if timestamp is None else timestamp
            return self.cb.query(question, session_id=session_id, timestamp=ts, k=k).context

    def answer(self, subject: str, predicate: str) -> str:
        """Exact structured answer for (subject, predicate), or "" if unknown."""
        with self._lock:
            return self.cb.answer_structured(subject, predicate)

    def consolidate(self) -> dict:
        with self._lock:
            rep = self.cb.consolidate()
            self._persist()
            return {"episodes_in": rep.episodes_in, "claims_out": rep.claims_out,
                    "pruned": rep.pruned}

    def stats(self) -> dict:
        with self._lock:
            s = self.cb.stats()
            return {"episodic": s.episodic_count, "structured": s.structured_count,
                    "semantic": s.semantic_count, "bytes": s.bytes}

    def _persist(self) -> None:
        """Atomic durable save (tmp + rename): a crash mid-write can no longer truncate the
        store file. Callers hold the lock."""
        if not self._path:
            return
        tmp = f"{self._path}.tmp"
        self.cb.save(tmp)
        os.replace(tmp, self._path)
        self._dirty = 0


def build_server(service: MemoryService | None = None, *, name: str = "curated-brain"):
    """Wrap a :class:`MemoryService` as a FastMCP server (``mcp`` imported lazily)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("MCP server requires the extra: pip install 'curated-brain[mcp]'") from e

    svc = service or MemoryService()
    mcp = FastMCP(name)

    @mcp.tool(description="Store an observation in memory; a surprise gate decides whether to "
                          "keep it, and facts are extracted from the raw text. timestamp "
                          "defaults to the current time.")
    def write(observation: str, session_id: str = "default",
              timestamp: float | None = None) -> dict:
        return svc.write(observation, session_id, timestamp)

    @mcp.tool(description="Retrieve a small, curated context to answer a question (stale "
                          "values are supersede-filtered). timestamp defaults to now.")
    def query(question: str, session_id: str = "default", timestamp: float | None = None,
              k: int = 8) -> str:
        return svc.query(question, session_id, timestamp, k)

    @mcp.tool(description="Exact structured answer for a (subject, predicate) fact, or empty.")
    def answer(subject: str, predicate: str) -> str:
        return svc.answer(subject, predicate)

    @mcp.tool(description="Run background consolidation (dedupe, prune, resolve contradictions).")
    def consolidate() -> dict:
        return svc.consolidate()

    @mcp.tool(description="Store statistics (record counts and serialized size).")
    def stats() -> dict:
        return svc.stats()

    return mcp


def main() -> None:
    """Console entry point: run the server over stdio (optionally persisting to CB_MCP_PATH)."""
    path = os.environ.get("CB_MCP_PATH")
    cb = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    if path and os.path.exists(path):
        cb.load(path)
    build_server(MemoryService(cb, path=path)).run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
