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
from curated_brain.namespace import DEFAULT_NAMESPACE, NamespacedMemory
from curated_brain.store import SqliteStore


class MemoryService:
    """The Curated Brain operations exposed over MCP, as plain (testable) methods.

    Multi-tenant by construction: it holds a :class:`NamespacedMemory`, so every method takes a
    ``namespace`` (defaulting to ``"default"``) that routes to a fully-isolated per-tenant
    :class:`CuratedBrain`. Single-namespace behavior is exactly the ``"default"`` namespace, so
    callers that never pass ``namespace`` see the old single-store behavior unchanged.

    Defaults to the deterministic no-LLM :class:`HeuristicExtractor` so raw text sent by an
    agent populates the structured tier with no spoon-fed facts. The same config (seed 0 +
    heuristic extractor) is reproduced by the factory for every lazily-created namespace.

    Two persistence modes, mutually exclusive:

    * ``store_path`` — durable SQLite with an incremental journal (:class:`SqliteStore` +
      ``attach_store``). Each write appends one O(1) journal row; a full snapshot is only
      rewritten periodically (compaction). This is the recommended mode: it fixes the
      O(store-size) full-snapshot cost paid on every write in the ``path`` mode.

      **Single-namespace only.** Journal rows carry no namespace and ``attach_store`` is
      per-brain, so this mode attaches one store to the ``"default"`` namespace and refuses any
      other (a clear :class:`ValueError` from every method when ``namespace != "default"``).
      Honest and safe beats clever: a pre-existing single-namespace store file keeps loading
      into ``"default"`` untouched. Use ``path`` mode for multi-tenant persistence — its blob
      already covers every namespace.
    * ``path`` — full-JSON snapshot of *all* namespaces via :meth:`NamespacedMemory.save`
      (atomic tmp+rename; ``persist_every`` batches writes to amortize the O(store) snapshot
      cost; also flushed on ``consolidate``). A legacy single-store blob loads into ``"default"``.

    Timestamps: the core is clock-free by contract (the harness supplies time), but a live
    agent host has no harness — so THIS boundary defaults to wall-clock. The old defaults
    (write at t=0.0) stamped every fact at the epoch, silently disabling recency scoring,
    supersede ordering, and all as-of semantics for the flagship integration.

    Thread safety: a single lock serializes all operations — kept even though CuratedBrain
    now carries its own RLock, because this service's read-modify-persist sequences must stay
    atomic as a unit, and MCP hosts may issue concurrent tool calls."""

    def __init__(self, cb: CuratedBrain | None = None, *, path: str | None = None,
                 persist_every: int = 1, store_path: str | None = None,
                 compact_every: int = 256) -> None:
        if path and store_path:
            raise ValueError("pass path= OR store_path=, not both")
        # Factory reproduces the service's default brain config for every new namespace.
        self._mem = NamespacedMemory(
            lambda: CuratedBrain(seed=0, extractor=HeuristicExtractor()))
        # The cb ctor arg becomes the "default" namespace's brain (back-compat); otherwise
        # space() builds it lazily from the factory.
        if cb is not None:
            self._mem.put(DEFAULT_NAMESPACE, cb)
        self.cb = self._mem.space(DEFAULT_NAMESPACE)  # back-compat alias for the default brain
        self._path = path
        self._persist_every = max(1, persist_every)
        self._dirty = 0
        self._lock = threading.Lock()
        # Durable journaled store: attach_store restores existing state then journals each write
        # (O(1)) instead of the full-JSON snapshot the path= mode rewrites. Owned here so it is
        # closed with the service; the brain journals to it inside its own locked write path.
        # store_path is single-namespace: the store attaches only to the default brain.
        self._store = SqliteStore(store_path) if store_path else None
        if self._store is not None:
            self.cb.attach_store(self._store, compact_every=compact_every)

    def _space(self, namespace: str) -> CuratedBrain:
        """The namespace's brain, guarding the store_path single-namespace restriction."""
        if self._store is not None and namespace != DEFAULT_NAMESPACE:
            raise ValueError(
                f"store_path mode supports only the {DEFAULT_NAMESPACE!r} namespace; "
                f"got {namespace!r}. Use path= mode for multi-tenant persistence.")
        return self._mem.space(namespace)

    def write(self, observation: str, session_id: str = "default",
              timestamp: float | None = None, speaker: str = "User",
              namespace: str = DEFAULT_NAMESPACE) -> dict:
        with self._lock:
            ts = time.time() if timestamp is None else timestamp
            # Declare the speaker so first-person text ("My email is …") extracts as facts
            # about them — a single-user server defaults to "User".
            r = self._space(namespace).write(observation, session_id=session_id, timestamp=ts,
                                             metadata={"speaker": speaker})
            self._dirty += 1
            if self._dirty >= self._persist_every:
                self._persist()
            return {"stored": r.stored, "reason": r.reason, "record_id": r.record_id}

    def query(self, question: str, session_id: str = "default",
              timestamp: float | None = None, k: int = 8,
              namespace: str = DEFAULT_NAMESPACE) -> str:
        with self._lock:
            ts = time.time() if timestamp is None else timestamp
            return self._space(namespace).query(question, session_id=session_id,
                                                timestamp=ts, k=k).context

    def answer(self, subject: str, predicate: str,
               namespace: str = DEFAULT_NAMESPACE) -> str:
        """Exact structured answer for (subject, predicate), or "" if unknown."""
        with self._lock:
            return self._space(namespace).answer_structured(subject, predicate)

    def consolidate(self, namespace: str = DEFAULT_NAMESPACE) -> dict:
        with self._lock:
            rep = self._space(namespace).consolidate()
            self._persist()
            return {"episodes_in": rep.episodes_in, "claims_out": rep.claims_out,
                    "pruned": rep.pruned}

    def stats(self, namespace: str = DEFAULT_NAMESPACE) -> dict:
        with self._lock:
            s = self._space(namespace).stats()
            return {"episodic": s.episodic_count, "structured": s.structured_count,
                    "semantic": s.semantic_count, "bytes": s.bytes}

    def drop(self, namespace: str = DEFAULT_NAMESPACE) -> dict:
        """Erase an entire namespace (tenant off-boarding / full GDPR erasure). Idempotent:
        returns ``{"existed": bool}`` — dropping an unknown namespace is a no-op, not an error.
        Dropping any namespace (including ``"default"``) is legitimate. In store_path mode only
        the default namespace exists; dropping it detaches and closes the durable store."""
        with self._lock:
            if self._store is not None and namespace != DEFAULT_NAMESPACE:
                raise ValueError(
                    f"store_path mode supports only the {DEFAULT_NAMESPACE!r} namespace; "
                    f"got {namespace!r}. Use path= mode for multi-tenant persistence.")
            if self._store is not None and namespace == DEFAULT_NAMESPACE:
                self.cb.detach_store()
                self._store.close()
                self._store = None
            existed = self._mem.drop(namespace)
            self._persist()
            return {"existed": existed}

    def close(self) -> None:
        """Release the durable store (if any). Idempotent; safe after the process is done."""
        with self._lock:
            if self._store is not None:
                self.cb.detach_store()
                self._store.close()
                self._store = None

    def _persist(self) -> None:
        """Atomic durable save (tmp + rename): a crash mid-write can no longer truncate the
        store file. Callers hold the lock.

        Covers every namespace. When only the ``"default"`` namespace exists the blob is the
        raw single-brain snapshot (so a legacy reader — including a raw ``CuratedBrain.load`` —
        keeps working); the moment a second namespace appears it switches to the multi-namespace
        blob. ``NamespacedMemory.restore`` reads both, so a service reopen is namespace-aware
        regardless.

        No-op in store_path mode — the attached SqliteStore journals every write itself, so we
        must NOT also rebuild the full snapshot here (that is exactly the O(store-size) per-write
        cost this backend removes)."""
        if self._store is not None or not self._path:
            return
        blob = (self._mem.space(DEFAULT_NAMESPACE).snapshot()
                if self._mem.namespaces() == [DEFAULT_NAMESPACE]
                else self._mem.snapshot())
        tmp = f"{self._path}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, self._path)
        self._dirty = 0

    def load(self, path: str) -> None:
        """Reopen the whole NamespacedMemory (all namespaces) from a ``path``-mode blob,
        replacing current state. Accepts both the raw single-brain blob (loads into
        ``"default"``) and the multi-namespace blob."""
        with self._lock:
            self._mem.load(path)
            self.cb = self._mem.space(DEFAULT_NAMESPACE)


def build_server(service: MemoryService | None = None, *, name: str = "curated-brain"):
    """Wrap a :class:`MemoryService` as a FastMCP server (``mcp`` imported lazily)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("MCP server requires the extra: pip install 'curated-brain[mcp]'") from e

    svc = service or MemoryService()
    mcp = FastMCP(name)

    _NS = ("namespace scopes memory to one tenant (default \"default\"); namespaces are "
           "hard-isolated, so facts in one are invisible to the others.")

    @mcp.tool(description="Store an observation in memory; a surprise gate decides whether to "
                          "keep it, and facts are extracted from the raw text. timestamp "
                          "defaults to the current time. " + _NS)
    def write(observation: str, session_id: str = "default",
              timestamp: float | None = None, namespace: str = DEFAULT_NAMESPACE) -> dict:
        return svc.write(observation, session_id, timestamp, namespace=namespace)

    @mcp.tool(description="Retrieve a small, curated context to answer a question (stale "
                          "values are supersede-filtered). timestamp defaults to now. " + _NS)
    def query(question: str, session_id: str = "default", timestamp: float | None = None,
              k: int = 8, namespace: str = DEFAULT_NAMESPACE) -> str:
        return svc.query(question, session_id, timestamp, k, namespace=namespace)

    @mcp.tool(description="Exact structured answer for a (subject, predicate) fact, or empty. "
                          + _NS)
    def answer(subject: str, predicate: str, namespace: str = DEFAULT_NAMESPACE) -> str:
        return svc.answer(subject, predicate, namespace=namespace)

    @mcp.tool(description="Run background consolidation (dedupe, prune, resolve contradictions). "
                          + _NS)
    def consolidate(namespace: str = DEFAULT_NAMESPACE) -> dict:
        return svc.consolidate(namespace=namespace)

    @mcp.tool(description="Store statistics (record counts and serialized size). " + _NS)
    def stats(namespace: str = DEFAULT_NAMESPACE) -> dict:
        return svc.stats(namespace=namespace)

    @mcp.tool(description="DANGER: permanently erase an entire namespace (all of that tenant's "
                          "memory — GDPR-style off-boarding). Idempotent: erasing an unknown "
                          "namespace is a no-op. " + _NS)
    def drop(namespace: str = DEFAULT_NAMESPACE) -> dict:
        return svc.drop(namespace)

    return mcp


def main() -> None:
    """Console entry point: run the server over stdio (optionally persisting to CB_MCP_PATH)."""
    path = os.environ.get("CB_MCP_PATH")
    svc = MemoryService(path=path)
    if path and os.path.exists(path):
        # namespace-aware: restores every persisted namespace (a legacy blob -> default)
        svc.load(path)
    build_server(svc).run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
