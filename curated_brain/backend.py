"""The harness backend adapter (PRD §9) and the ``CuratedBrain`` implementation.

This module defines the ``MemoryBackend`` ABC the eval harness drives, plus the concrete
curated-memory layer. The implementation grows stage by stage; at every stage it stays
byte-deterministic so ``snapshot -> restore`` round-trips exactly (AC-1).

Stage 1 wires only the episodic store + a lexical query, enough to satisfy adapter
conformance and determinism. Later stages plug in the structured tier, vector tier,
surprise gate and consolidation worker behind the same interface.
"""

from __future__ import annotations

import abc
import json

from curated_brain.fakes import DeterministicEmbedder
from curated_brain.models import (
    Citation,
    ConsolidationReport,
    EpisodicRecord,
    Retrieval,
    StoreStats,
    WriteReceipt,
)
from curated_brain.structured import StructuredTier
from curated_brain.util import count_tokens, from_jsonable, to_jsonable
from curated_brain.vector import VectorTier

SNAPSHOT_VERSION = 1


class MemoryBackend(abc.ABC):
    """The contract the Longitudinal Memory Eval Harness drives (PRD §9).

    ``timestamp`` is always supplied by the caller — memory must never read the real
    wall-clock — so runs are reproducible. ``consolidate`` lets the harness simulate
    "sleep" between sessions; ``snapshot``/``restore`` make runs deterministic.
    """

    @abc.abstractmethod
    def write(self, observation: str, *, session_id: str, timestamp: float,
              metadata: dict | None = None) -> WriteReceipt: ...

    @abc.abstractmethod
    def query(self, question: str, *, session_id: str, timestamp: float,
              k: int = 8) -> Retrieval: ...

    @abc.abstractmethod
    def consolidate(self) -> ConsolidationReport: ...

    @abc.abstractmethod
    def stats(self) -> StoreStats: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    @abc.abstractmethod
    def snapshot(self) -> bytes: ...

    @abc.abstractmethod
    def restore(self, blob: bytes) -> None: ...


class CuratedBrain(MemoryBackend):
    def __init__(self, embedder: DeterministicEmbedder | None = None, *,
                 dim: int = 256, seed: int = 0) -> None:
        self.embedder = embedder or DeterministicEmbedder(dim)
        self.seed = seed
        self.reset()

    # ------------------------------------------------------------------ identifiers --
    def _next_id(self, kind: str) -> str:
        self._counter += 1
        return f"{kind}-{self._counter:012d}"

    # ------------------------------------------------------------------ write path ---
    def write(self, observation: str, *, session_id: str, timestamp: float,
              metadata: dict | None = None) -> WriteReceipt:
        meta = metadata or {}
        fact = meta.get("fact")
        rec = EpisodicRecord(
            id=self._next_id("ep"),
            session_id=session_id,
            seq=len(self._episodes),
            wall_ts=timestamp,
            actor=meta.get("actor", "user"),
            content=observation,
            embed_model_id=self.embedder.model_id,
            surprise=0.0,
            provenance={"source": "write", "session_id": session_id},
            last_seen_ts=timestamp,
        )
        self._episodes.append(rec)
        self.vector.add(rid=rec.id, text=observation, wall_ts=timestamp,
                        session_id=session_id, tier="episodic",
                        entities=[fact["subject"]] if fact else [])
        self._route_fact(fact, rec, timestamp)
        return WriteReceipt(stored=True, reason="stored", record_id=rec.id, surprise=0.0)

    def _route_fact(self, fact: dict | None, rec: EpisodicRecord, timestamp: float) -> None:
        """Route an extracted triple into the bi-temporal structured tier (PRD §6)."""
        if not fact:
            return
        self.structured.assert_fact(
            fact_id=self._next_id("fact"),
            subject=fact["subject"], predicate=fact["predicate"], object=fact["object"],
            valid_from=timestamp, created_at=timestamp,
            provenance={"episode_id": rec.id, "session_id": rec.session_id,
                        "wall_ts": rec.wall_ts},
        )

    # ----------------------------------------------------------- structured answers --
    def answer_structured(self, subject: str, predicate: str, *, at: float | None = None) -> str:
        """Exact / as-of-time answer from the structured tier (empty string if unknown)."""
        f = self.structured.resolve(subject, predicate, at)
        return f.object if f else ""

    def answer_path(self, subject: str, predicates: list[str], *,
                    at: float | None = None) -> str:
        """Multi-hop relational answer from the structured tier."""
        f = self.structured.resolve_path(subject, predicates, at)
        return f.object if f else ""

    # ------------------------------------------------------------------ query path ---
    def query(self, question: str, *, session_id: str, timestamp: float,
              k: int = 8) -> Retrieval:
        """Stage-3 semantic recall: top-k vector search over episodic memory (causal).

        Stage 4 layers the structured tier, fusion re-rank and supersede-filtering on top.
        """
        hits = self.vector.search(question, k=k, t=timestamp)
        lines, citations = [], []
        for i, (r, _score) in enumerate(hits, 1):
            lines.append(f"[{i}] {r.text}")
            citations.append(Citation(record_id=r.rid, provenance={"session_id": r.session_id},
                                      valid_interval=(r.wall_ts, float("inf"))))
        context = "\n".join(lines)
        return Retrieval(context=context, citations=citations, tokens_in=count_tokens(context))

    # ------------------------------------------------------------------ maintenance --
    def consolidate(self) -> ConsolidationReport:
        return ConsolidationReport(
            episodes_in=len(self._episodes), claims_out=0, dupes_merged=0,
            contradictions_resolved=0, pruned=0,
        )

    def stats(self) -> StoreStats:
        return StoreStats(
            episodic_count=len(self._episodes),
            structured_count=len(self.structured.facts),
            semantic_count=0,
            bytes=len(self.snapshot()),
            embed_model_id=self.embedder.model_id,
        )

    def reset(self) -> None:
        self._counter = 0
        self._episodes: list[EpisodicRecord] = []
        self.structured = StructuredTier()
        self.vector = VectorTier(self.embedder)

    # ------------------------------------------------------------------ persistence --
    def _state(self) -> dict:
        """Canonical, JSON-able state. The shape is fixed across stages; later stages
        fill the currently-empty subsystem keys rather than changing the schema."""
        return {
            "version": SNAPSHOT_VERSION,
            "config": {"embed_model_id": self.embedder.model_id, "dim": self.embedder.dim,
                       "seed": self.seed},
            "counter": self._counter,
            "episodic": [vars(r) for r in self._episodes],
            "structured": self.structured.to_dict(),
            "vector": self.vector.to_dict(),
            "gate": {},
        }

    def snapshot(self) -> bytes:
        payload = to_jsonable(self._state())
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        state = from_jsonable(json.loads(blob.decode("utf-8")))
        self._counter = state["counter"]
        self._episodes = [EpisodicRecord(**d) for d in state["episodic"]]
        self.structured = StructuredTier()
        self.structured.load(state.get("structured", []))
        self.vector = VectorTier(self.embedder)
        if state.get("vector"):
            self.vector.load(state["vector"])
