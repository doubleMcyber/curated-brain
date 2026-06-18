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

from curated_brain.consolidation import cluster_by_similarity, representative
from curated_brain.fakes import DeterministicEmbedder
from curated_brain.models import (
    Citation,
    ConsolidationReport,
    EpisodicRecord,
    Retrieval,
    StoreStats,
    WriteReceipt,
)
from curated_brain.retrieval import Planner, fuse, render_fact
from curated_brain.structured import StructuredTier
from curated_brain.surprise import REINFORCE, STORE, SurpriseGate
from curated_brain.util import count_tokens, from_jsonable, normalize, to_jsonable
from curated_brain.vector import VectorTier

MAX_CONTEXT_ITEMS = 4  # curated payloads stay tiny (PRD §7: far below long-context)
FREE_DEDUP_THRESHOLD = 0.85  # only genuine near-duplicate free-text episodes are merged

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
                 dim: int = 256, seed: int = 0, gate: SurpriseGate | None = None) -> None:
        self.embedder = embedder or DeterministicEmbedder(dim)
        self.seed = seed
        self.planner = Planner()
        self._gate_cfg = (gate or SurpriseGate()).to_dict()
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
        embedding = self.embedder.embed(observation)

        nearest = self.vector.nearest(embedding)
        max_cos = max(0.0, min(1.0, nearest[1])) if nearest else 0.0
        novelty = 1.0 - max_cos
        contradiction = self._is_contradiction(fact)
        decision = self.gate.decide(novelty, contradiction=contradiction)
        surprise = 1.0 if contradiction else novelty

        rec_id = None
        if decision == STORE:
            rec = EpisodicRecord(
                id=self._next_id("ep"),
                session_id=session_id,
                seq=len(self._episodes),
                wall_ts=timestamp,
                actor=meta.get("actor", "user"),
                content=observation,
                embed_model_id=self.embedder.model_id,
                surprise=surprise,
                provenance={"source": "write", "session_id": session_id},
                last_seen_ts=timestamp,
                fact_key=(f"{normalize(fact['subject'])}|{normalize(fact['predicate'])}"
                          f"|{normalize(fact['object'])}") if fact else None,
            )
            self._episodes.append(rec)
            self._ep_by_id[rec.id] = rec
            self.vector.add(rid=rec.id, text=observation, wall_ts=timestamp,
                            session_id=session_id, tier="episodic",
                            entities=[fact["subject"]] if fact else [], embedding=embedding)
            rec_id = rec.id
        elif decision == REINFORCE and nearest is not None:
            tgt = self._ep_by_id.get(nearest[0].rid)
            if tgt is not None:
                tgt.support_count += 1
                tgt.last_seen_ts = timestamp

        # Facts are ALWAYS captured by the structured tier (idempotent assert), so the
        # surprise gate can drop the raw episodic record without ever losing a salient fact.
        self._route_fact(fact, rec_id, session_id, timestamp)
        self._note_session(session_id, timestamp)
        return WriteReceipt(stored=decision == STORE, reason=decision,
                            record_id=rec_id, surprise=surprise)

    def _is_contradiction(self, fact: dict | None) -> bool:
        if not fact:
            return False
        cur = self.structured.current(fact["subject"], fact["predicate"])
        return cur is not None and normalize(cur.object) != normalize(fact["object"])

    def _note_session(self, session_id: str, timestamp: float) -> None:
        """Record the earliest timestamp seen per session index so the planner can map a
        natural-language "as of session N" back to an as-of time."""
        if session_id.startswith("s") and session_id[1:].isdigit():
            idx = int(session_id[1:])
            prev = self._session_ts.get(idx)
            if prev is None or timestamp < prev:
                self._session_ts[idx] = timestamp

    def _route_fact(self, fact: dict | None, episode_id: str | None,
                    session_id: str, timestamp: float) -> None:
        """Route an extracted triple into the bi-temporal structured tier (PRD §6).

        Called for every fact-bearing observation regardless of the surprise gate's
        decision; ``assert_fact`` is idempotent (reasserting a value is a no-op) so this
        never duplicates rows but guarantees salient facts are never lost to the gate."""
        if not fact:
            return
        self.structured.assert_fact(
            fact_id=self._next_id("fact"),
            subject=fact["subject"], predicate=fact["predicate"], object=fact["object"],
            valid_from=timestamp, created_at=timestamp,
            provenance={"episode_id": episode_id, "session_id": session_id,
                        "wall_ts": timestamp},
        )
        self._entities.add(normalize(fact["subject"]))

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
        """Hybrid retrieval (PRD §7): plan -> fetch structured + vector -> fuse, re-rank,
        supersede-filter. The exact current/as-of fact is surfaced first; superseded values
        are dropped so stale contradictions never reach the agent."""
        plan = self.planner.plan(question, entities=self._entities,
                                 session_ts=self._session_ts)
        lines: list[str] = []
        citations: list[Citation] = []

        if not plan.open_ended:
            f = (self.structured.resolve_path(plan.entity, plan.hops, plan.as_of)
                 if plan.hops else
                 self.structured.resolve(plan.entity, plan.predicate, plan.as_of))
            if f is not None:
                lines.append(render_fact(plan, f))
                citations.append(Citation(record_id=f.id, provenance=f.provenance,
                                          valid_interval=(f.valid_from, f.valid_to)))

        vhits = self.vector.search(question, k=k, t=timestamp, entity=plan.entity)
        for it in fuse(vhits, now=timestamp, stale_objs=self._stale_objs()):
            if len(lines) >= MAX_CONTEXT_ITEMS:
                break
            lines.append(it.text)
            citations.append(Citation(record_id=it.rid, provenance=it.provenance,
                                      valid_interval=it.valid_interval))

        context = "\n".join(f"[{i}] {ln}" for i, ln in enumerate(lines, 1))
        return Retrieval(context=context, citations=citations, tokens_in=count_tokens(context))

    def _stale_objs(self) -> set[str]:
        """Normalized object values to filter out of fused vector context: those that have
        been superseded (closed in valid time) and are **not currently true for any
        entity**. Subtracting the open values keeps this safe even if a value is later
        re-asserted (A→B→A) or is current for one entity while stale for another — so we
        never drop a statement of a value that is presently true."""
        open_vals = {normalize(f.object) for f in self.structured.facts if f.is_open}
        return {normalize(f.object) for f in self.structured.facts if not f.is_open} - open_vals

    # ------------------------------------------------------------------ maintenance --
    def reembed(self, new_embedder) -> dict:
        """Migrate the store to a new embedding model (PRD §12 re-embed-on-upgrade).

        Non-lossy: source text, structured facts and provenance are untouched — only the
        vectors are recomputed and each record's ``embed_model_id`` is stamped with the new
        model. Lets the layer survive an embedder upgrade without losing recall.
        """
        old = self.embedder.model_id
        self.embedder = new_embedder
        n = self.vector.reembed(new_embedder)
        for r in self._episodes:
            r.embed_model_id = new_embedder.model_id
        return {"reembedded": n, "from": old, "to": new_embedder.model_id}

    def consolidate(self) -> ConsolidationReport:
        """Compress the episodic tier (PRD §8). Fact-bearing episodes are grouped by their
        exact ``(subject, predicate)``: current-value paraphrases merge into one semantic
        claim (dedup), and outdated (superseded-value) raw episodes are pruned — the value
        history lives non-lossily in the structured tier. Remaining free-text episodes have
        only true near-duplicates merged. Structured facts and provenance are never touched,
        so accuracy is preserved and the audit trail is intact."""
        episodes_in = len(self._episodes)
        current_obj = {(normalize(f.subject), normalize(f.predicate)): normalize(f.object)
                       for f in self.structured.open_facts}

        groups: dict[tuple[str, str], list[EpisodicRecord]] = {}
        free: list[EpisodicRecord] = []
        for r in self._episodes:
            if r.fact_key:
                subj, pred, _ = r.fact_key.split("|")
                groups.setdefault((subj, pred), []).append(r)
            else:
                free.append(r)

        new_eps: list[EpisodicRecord] = []
        dupes_merged = claims_out = pruned = 0

        for (subj, pred), members in groups.items():
            cur = current_obj.get((subj, pred))
            current_eps = [r for r in members if r.fact_key.split("|")[2] == cur]
            for r in members:  # outdated raw episodes are compacted away
                if r not in current_eps:
                    self.vector.remove_by_rid(r.id)
                    pruned += 1
            if len(current_eps) > 1:
                new_eps.append(self._merge_to_claim(current_eps))
                dupes_merged += len(current_eps) - 1
                claims_out += 1
            elif current_eps:
                new_eps.append(current_eps[0])

        # Free-text episodes: only merge genuine near-duplicates (high threshold avoids
        # collapsing distinct content); keep everything else.
        for cluster in cluster_by_similarity([(self.embedder.embed(r.content), r) for r in free],
                                             FREE_DEDUP_THRESHOLD):
            if len(cluster) > 1:
                new_eps.append(self._merge_to_claim(cluster))
                dupes_merged += len(cluster) - 1
                claims_out += 1
            else:
                new_eps.append(cluster[0])

        self._episodes = new_eps
        self._ep_by_id = {r.id: r for r in new_eps}
        contradictions_resolved = sum(1 for f in self.structured.facts if not f.is_open)
        return ConsolidationReport(episodes_in=episodes_in, claims_out=claims_out,
                                   dupes_merged=dupes_merged,
                                   contradictions_resolved=contradictions_resolved, pruned=pruned)

    def _merge_to_claim(self, members: list[EpisodicRecord]) -> EpisodicRecord:
        rep = representative(members)
        for r in members:
            self.vector.remove_by_rid(r.id)
        claim = EpisodicRecord(
            id=self._next_id("claim"), session_id=rep.session_id, seq=len(self._episodes),
            wall_ts=rep.wall_ts, actor="system", content=rep.content,
            embed_model_id=self.embedder.model_id, surprise=rep.surprise,
            provenance={"source": "consolidation", "merged_from": [r.id for r in members]},
            support_count=sum(r.support_count for r in members),
            last_seen_ts=max(r.last_seen_ts for r in members),
            tier="semantic", supports=[r.id for r in members],
        )
        self.vector.add(rid=claim.id, text=claim.content, wall_ts=claim.wall_ts,
                        session_id=claim.session_id, tier="semantic", entities=[])
        return claim

    def stats(self) -> StoreStats:
        return StoreStats(
            episodic_count=sum(1 for r in self._episodes if r.tier == "episodic"),
            structured_count=len(self.structured.facts),
            semantic_count=sum(1 for r in self._episodes if r.tier == "semantic"),
            bytes=len(self.snapshot()),
            embed_model_id=self.embedder.model_id,
        )

    def reset(self) -> None:
        self._counter = 0
        self._episodes: list[EpisodicRecord] = []
        self._ep_by_id: dict[str, EpisodicRecord] = {}
        self.structured = StructuredTier()
        self.vector = VectorTier(self.embedder)
        self.gate = SurpriseGate.from_dict({**self._gate_cfg, "seen": 0, "stored": 0})
        self._entities: set[str] = set()
        self._session_ts: dict[int, float] = {}

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
            "gate": self.gate.to_dict(),
        }

    def snapshot(self) -> bytes:
        payload = to_jsonable(self._state())
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        state = from_jsonable(json.loads(blob.decode("utf-8")))
        self._counter = state["counter"]
        self._episodes = [EpisodicRecord(**d) for d in state["episodic"]]
        self._ep_by_id = {r.id: r for r in self._episodes}
        self.structured = StructuredTier()
        self.structured.load(state.get("structured", []))
        self.vector = VectorTier(self.embedder)
        if state.get("vector"):
            self.vector.load(state["vector"])
        self.gate = SurpriseGate.from_dict(state["gate"]) if state.get("gate") else SurpriseGate()
        # Rebuild derived planner state from the restored stores.
        self._entities = {normalize(f.subject) for f in self.structured.facts}
        self._session_ts = {}
        for r in self._episodes:
            self._note_session(r.session_id, r.wall_ts)
