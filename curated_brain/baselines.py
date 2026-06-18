"""Baseline backends to beat (PRD §9.1): naive RAG, long-context, no-memory.

All three implement the same ``MemoryBackend`` contract so the harness can score them
against ``CuratedBrain`` under identical conditions. Each is intentionally faithful to its
real-world failure mode:

* **NaiveRAG** — log every utterance, answer by top-k vector similarity. No structure, no
  supersede filtering: noise crowds the top-k, stale facts resurface, multi-hop is
  impossible.
* **LongContext** — paste the most recent history up to a fixed token window; older facts
  fall out of the window entirely (the long-range failure of context stuffing).
* **NoMemory** — the frozen model alone; nothing is retained.

All retrieval respects causality: a query at time ``t`` only sees writes with
``wall_ts <= t``.
"""

from __future__ import annotations

import numpy as np

from curated_brain.backend import MemoryBackend
from curated_brain.fakes import DeterministicEmbedder
from curated_brain.models import Citation, ConsolidationReport, Retrieval, StoreStats, WriteReceipt
from curated_brain.util import count_tokens


class _Stored:
    __slots__ = ("rid", "content", "wall_ts", "embedding")

    def __init__(self, rid: str, content: str, wall_ts: float, embedding: np.ndarray) -> None:
        self.rid = rid
        self.content = content
        self.wall_ts = wall_ts
        self.embedding = embedding


class NaiveRAG(MemoryBackend):
    """Log-everything + top-k semantic retrieval. No curation."""

    def __init__(self, embedder: DeterministicEmbedder | None = None, *, dim: int = 256) -> None:
        self.embedder = embedder or DeterministicEmbedder(dim)
        self.reset()

    def write(self, observation, *, session_id, timestamp, metadata=None) -> WriteReceipt:
        self._n += 1
        rid = f"nv-{self._n:012d}"
        self._records.append(_Stored(rid, observation, timestamp, self.embedder.embed(observation)))
        return WriteReceipt(stored=True, reason="stored", record_id=rid, surprise=1.0)

    def query(self, question, *, session_id, timestamp, k=8) -> Retrieval:
        visible = [r for r in self._records if r.wall_ts <= timestamp]
        if not visible:
            return Retrieval(context="", citations=[], tokens_in=0)
        q = self.embedder.embed(question)
        scored = sorted(visible, key=lambda r: (float(q @ r.embedding), r.wall_ts, r.rid),
                        reverse=True)[:k]
        lines = [f"[{i}] {r.content}" for i, r in enumerate(scored, 1)]
        cites = [Citation(record_id=r.rid, provenance={"source": "naive"},
                          valid_interval=(r.wall_ts, float("inf"))) for r in scored]
        ctx = "\n".join(lines)
        return Retrieval(context=ctx, citations=cites, tokens_in=count_tokens(ctx))

    def consolidate(self) -> ConsolidationReport:
        return ConsolidationReport(len(self._records), 0, 0, 0, 0)

    def stats(self) -> StoreStats:
        return StoreStats(len(self._records), 0, 0, 0, self.embedder.model_id)

    def reset(self) -> None:
        self._n = 0
        self._records: list[_Stored] = []

    def snapshot(self) -> bytes:
        return b""  # baselines need not be deterministic-snapshotable

    def restore(self, blob: bytes) -> None:
        self.reset()


class LongContext(MemoryBackend):
    """Paste the most-recent history up to a fixed token window. Facts older than the
    window fall out entirely — the long-range failure of context stuffing (PRD §9.1)."""

    def __init__(self, *, window_tokens: int = 800) -> None:
        self.window_tokens = window_tokens
        self.reset()

    def write(self, observation, *, session_id, timestamp, metadata=None) -> WriteReceipt:
        self._n += 1
        rid = f"lc-{self._n:012d}"
        self._records.append((rid, observation, timestamp))
        return WriteReceipt(stored=True, reason="stored", record_id=rid, surprise=1.0)

    def query(self, question, *, session_id, timestamp, k=8) -> Retrieval:
        visible = sorted((r for r in self._records if r[2] <= timestamp),
                         key=lambda r: (r[2], r[0]), reverse=True)  # most recent first
        lines, cites, used = [], [], 0
        for rid, content, ts in visible:
            tt = count_tokens(content)
            if used + tt > self.window_tokens:
                break
            lines.append(content)
            used += tt
            cites.append(Citation(record_id=rid, provenance={"source": "long_context"},
                                  valid_interval=(ts, float("inf"))))
        ctx = "\n".join(lines)
        return Retrieval(context=ctx, citations=cites, tokens_in=count_tokens(ctx))

    def consolidate(self) -> ConsolidationReport:
        return ConsolidationReport(len(self._records), 0, 0, 0, 0)

    def stats(self) -> StoreStats:
        return StoreStats(len(self._records), 0, 0, 0, "long_context")

    def reset(self) -> None:
        self._n = 0
        self._records: list[tuple[str, str, float]] = []

    def snapshot(self) -> bytes:
        return b""

    def restore(self, blob: bytes) -> None:
        self.reset()


class NoMemory(MemoryBackend):
    """The frozen model alone — nothing is retained between turns (PRD §9.1)."""

    def write(self, observation, *, session_id, timestamp, metadata=None) -> WriteReceipt:
        return WriteReceipt(stored=False, reason="discarded", record_id=None, surprise=0.0)

    def query(self, question, *, session_id, timestamp, k=8) -> Retrieval:
        return Retrieval(context="", citations=[], tokens_in=0)

    def consolidate(self) -> ConsolidationReport:
        return ConsolidationReport(0, 0, 0, 0, 0)

    def stats(self) -> StoreStats:
        return StoreStats(0, 0, 0, 0, "no_memory")

    def reset(self) -> None:
        pass

    def snapshot(self) -> bytes:
        return b""

    def restore(self, blob: bytes) -> None:
        pass
