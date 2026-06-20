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

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor

# Far-future default query time: with no explicit clock, "now" sees everything written.
_NOW = 1e12


class MemoryService:
    """The Curated Brain operations exposed over MCP, as plain (testable) methods.

    Defaults to the deterministic no-LLM :class:`HeuristicExtractor` so raw text sent by an
    agent populates the structured tier with no spoon-fed facts. Pass ``path`` to persist the
    store to disk after each mutating call (durable across server restarts)."""

    def __init__(self, cb: CuratedBrain | None = None, *, path: str | None = None) -> None:
        self.cb = cb or CuratedBrain(seed=0, extractor=HeuristicExtractor())
        self._path = path

    def write(self, observation: str, session_id: str = "default",
              timestamp: float = 0.0) -> dict:
        r = self.cb.write(observation, session_id=session_id, timestamp=timestamp)
        self._persist()
        return {"stored": r.stored, "reason": r.reason, "record_id": r.record_id}

    def query(self, question: str, session_id: str = "default",
              timestamp: float = _NOW, k: int = 8) -> str:
        return self.cb.query(question, session_id=session_id, timestamp=timestamp, k=k).context

    def answer(self, subject: str, predicate: str) -> str:
        """Exact structured answer for (subject, predicate), or "" if unknown."""
        return self.cb.answer_structured(subject, predicate)

    def consolidate(self) -> dict:
        rep = self.cb.consolidate()
        self._persist()
        return {"episodes_in": rep.episodes_in, "claims_out": rep.claims_out,
                "pruned": rep.pruned}

    def stats(self) -> dict:
        s = self.cb.stats()
        return {"episodic": s.episodic_count, "structured": s.structured_count,
                "semantic": s.semantic_count, "bytes": s.bytes}

    def _persist(self) -> None:
        if self._path:
            self.cb.save(self._path)


def build_server(service: MemoryService | None = None, *, name: str = "curated-brain"):
    """Wrap a :class:`MemoryService` as a FastMCP server (``mcp`` imported lazily)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("MCP server requires the extra: pip install 'curated-brain[mcp]'") from e

    svc = service or MemoryService()
    mcp = FastMCP(name)

    @mcp.tool(description="Store an observation in memory; a surprise gate decides whether to "
                          "keep it, and facts are extracted from the raw text.")
    def write(observation: str, session_id: str = "default", timestamp: float = 0.0) -> dict:
        return svc.write(observation, session_id, timestamp)

    @mcp.tool(description="Retrieve a small, curated context to answer a question (stale "
                          "values are supersede-filtered).")
    def query(question: str, session_id: str = "default", timestamp: float = _NOW,
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
