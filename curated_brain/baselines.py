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
