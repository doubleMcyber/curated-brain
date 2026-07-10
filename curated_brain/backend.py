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
import inspect
import json
import logging
import math
import threading
from dataclasses import MISSING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from curated_brain.store import SqliteStore

from curated_brain.config import CBConfig
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
from curated_brain.resolve import EntityResolver
from curated_brain.retrieval import Planner, fuse, render_fact
from curated_brain.structured import StructuredTier
from curated_brain.surprise import REINFORCE, STORE, PredictiveSurprise, SurpriseGate
from curated_brain.util import (
    count_tokens,
    from_jsonable,
    normalize,
    to_jsonable,
    tokenize,
)
from curated_brain.vector import VectorTier

SNAPSHOT_VERSION = 1

# Library logger with a NullHandler (stdlib best practice): silent unless the application
# configures logging. Logs decisions/degradations only — never secrets, never observation
# content (which could carry PII). Purely observational: emits no state, so AC-1 determinism
# and byte-identical snapshots are unaffected.
logger = logging.getLogger("curated_brain")
logger.addHandler(logging.NullHandler())


def _fact_key_parts(rec: EpisodicRecord) -> list[str]:
    """The (subject, predicate, object) parts of an episode's fact link. New records store a
    3-item list; legacy snapshots stored "subject|predicate|object", which crashed
    ``consolidate()`` whenever a value contained "|" — split capped at 2 so legacy keys with
    pipes in the OBJECT (the caller-supplied value, the realistic case) parse correctly."""
    if isinstance(rec.fact_key, list):
        return rec.fact_key
    assert rec.fact_key is not None
    return rec.fact_key.split("|", 2)


_EPISODIC_FIELDS = frozenset(EpisodicRecord.__dataclass_fields__)
_EP_REQUIRED = frozenset(
    n for n, f in EpisodicRecord.__dataclass_fields__.items()
    if f.default is MISSING and f.default_factory is MISSING)


def _validate_snapshot(state: dict) -> None:
    """Validate a decoded snapshot's top-level shape before restore consumes it, so an
    untrusted/corrupt blob fails with a clear ValueError instead of an opaque crash deep in
    dataclass/index construction. Bounds are structural (types + episodic-record keys); the
    vector-size bound lives in ``BruteForceIndex.from_dict``."""
    if "counter" not in state or not isinstance(state["counter"], int):
        raise ValueError("snapshot missing an integer 'counter'")
    episodic = state.get("episodic", [])
    if not isinstance(episodic, list):
        raise ValueError(f"snapshot 'episodic' must be a list, got {type(episodic).__name__}")
    for i, d in enumerate(episodic):
        if not isinstance(d, dict):
            raise ValueError(f"episodic[{i}] must be an object, got {type(d).__name__}")
        extra = set(d) - _EPISODIC_FIELDS
        if extra:
            raise ValueError(f"episodic[{i}] has unknown fields {sorted(extra)}")
        missing = _EP_REQUIRED - set(d)
        if missing:
            raise ValueError(f"episodic[{i}] missing required fields {sorted(missing)}")
    for key in ("structured", "vector"):
        if key in state and not isinstance(state[key], (list, dict)):
            raise ValueError(f"snapshot '{key}' has wrong type {type(state[key]).__name__}")
    if "config" in state and not isinstance(state["config"], dict):
        raise ValueError(f"snapshot 'config' must be an object, "
                         f"got {type(state['config']).__name__}")
    if "asserted_texts" in state and not isinstance(state["asserted_texts"], list):
        raise ValueError(f"snapshot 'asserted_texts' must be a list, "
                         f"got {type(state['asserted_texts']).__name__}")


def _validate_fact(fact) -> None:
    """Reject a malformed caller-supplied fact at the boundary (fail-loud, clear message)
    rather than letting it crash deep in routing or silently corrupt the structured tier."""
    if not isinstance(fact, dict):
        raise ValueError(f"metadata fact must be a dict, got {type(fact).__name__}")
    for key in ("subject", "predicate", "object"):
        if not isinstance(fact.get(key), str) or not fact[key]:
            raise ValueError(f"metadata fact needs a non-empty str '{key}': {fact!r}")
    vf = fact.get("valid_from")
    if vf is not None and (not isinstance(vf, (int, float)) or isinstance(vf, bool)
                           or not math.isfinite(vf)):
        raise ValueError(f"metadata fact 'valid_from' must be a finite number: {vf!r}")


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
    # Core mutable state is (re)initialized in reset()/restore(), not __init__, so it is
    # declared here for the type checker (annotations only — no runtime effect).
    structured: StructuredTier
    vector: VectorTier
    gate: SurpriseGate
    _resolver: EntityResolver
    _episodes: list[EpisodicRecord]
    _ep_by_id: dict[str, EpisodicRecord]
    _session_ts: dict[int, float]
    _asserted_texts: set[str]
    _decisions: dict[str, int]
    _cost: dict[str, int]
    _derived: tuple | None

    def __init__(self, embedder: DeterministicEmbedder | None = None, *,
                 dim: int = 256, seed: int = 0, gate: SurpriseGate | None = None,
                 extractor=None, max_context_items: int | None = None,
                 summarizer=None, surprise_llm=None, pricing=None,
                 config: CBConfig | None = None,
                 index_factory=None, ann_path: str | None = None) -> None:
        # Coarse-grained reentrant lock: every public entry point acquires it, so all state
        # reads/mutations serialize within one process. RLock (not Lock) because public methods
        # call one another (e.g. stats() -> snapshot()); private helpers stay lock-free. It is
        # runtime-only state, never serialized (see _state()), so determinism is unaffected.
        self._lock = threading.RLock()
        # Durable-store attachment (see attach_store). None until attached. `_replaying` guards
        # the journal-replay path so replayed writes are not re-journaled; `_journal_since`
        # counts journal rows since the last compaction to fire compact_every.
        self._store: SqliteStore | None = None
        self._compact_every = 0
        self._journal_since = 0
        self._replaying = False
        self.embedder = embedder or DeterministicEmbedder(dim)
        self.seed = seed
        self.planner = Planner()
        # Frozen tuning config: default CBConfig() re-derives today's constants, so config=None
        # (and a default config) is a behavioral no-op. An explicit gate/max_context_items ctor
        # arg still wins over the config field (per-arg precedence).
        self.config = config or CBConfig()
        if gate is None:
            gate = SurpriseGate(budget=self.config.budget, theta0=self.config.theta0,
                                theta_floor=self.config.theta_floor,
                                theta_max=self.config.theta_max, lr=self.config.lr)
        self._gate_cfg = gate.to_dict()
        # Context budget per query. The default (CBConfig.max_context_items=4) keeps payloads
        # tiny (PRD §7) but silently overrode the caller's k — now configurable so an adopter
        # asking for more context actually gets it.
        self._max_ctx = (max_context_items if max_context_items is not None
                         else self.config.max_context_items)
        if self._max_ctx < 1:
            raise ValueError(f"max_context_items must be >= 1, got {max_context_items!r}")
        # Optional Track-B extractor: when set, facts are derived from raw text for
        # observations that arrive without a pre-extracted `metadata.fact`. Default None
        # preserves the spoon-fed-fact behavior (and AC-1 determinism) exactly.
        self.extractor = extractor
        # Extractors MAY accept a `speaker` kwarg (first-person resolution); detected once
        # here so third-party extractors with the plain extract(text) shape keep working.
        self._extractor_takes_speaker = (
            extractor is not None
            and "speaker" in inspect.signature(extractor.extract).parameters)
        # Extractors MAY accept a `ref_ts` kwarg (event-date resolution against the write
        # clock); same soft-detection so plain extract(text) extractors are unaffected.
        self._extractor_takes_ref_ts = (
            extractor is not None
            and "ref_ts" in inspect.signature(extractor.extract).parameters)
        # Extractors MAY accept a `resolve_role` kwarg (definite-NP coreference): the backend
        # supplies a "role noun -> sole current holder or None" lookup over the structured
        # tier, so "the manager" resolves to the unique person with an open (·, role, manager)
        # fact. Same soft-detection so plain extract(text) extractors are unaffected.
        self._extractor_takes_resolve_role = (
            extractor is not None
            and "resolve_role" in inspect.signature(extractor.extract).parameters)
        # Optional consolidation summarizer (an `LLM` protocol object): merged claims get a
        # real one-sentence summary instead of the most-reinforced member verbatim (PRD §8
        # "cluster & summarize"). Default None keeps consolidation model-free + deterministic.
        self.summarizer = summarizer
        # Optional predictive (log-prob) surprise (PRD §6 #2): an LLM exposing
        # complete_with_logprobs. When set, the write gate fuses lexical novelty with the
        # model's prediction error on the observation given its nearest memories, catching the
        # paraphrased-update dead zone. Default None keeps the gate model-free + byte-identical.
        self._predictive = (PredictiveSurprise(surprise_llm, k_context=self._max_ctx)
                            if surprise_llm is not None else None)
        # Optional Pricing (token→$): when set, metrics()["cost"]["estimated_usd"] reports the
        # dollar cost of the metered hot-path tokens. Default None → cost stays token-only.
        self.pricing = pricing
        # Opt-in ANN scale seam (Track C). ``index_factory`` (dim -> VectorIndex) makes the
        # vector tier remember its index TYPE so restore()/reset() rebuild it as itself instead
        # of demoting to BruteForce; ``ann_path`` is the optional on-disk sidecar so a large
        # store reloads the saved graph instead of rebuilding it. Both default to None, which
        # is byte-identical to the exact BruteForce default (AC-1 determinism unaffected).
        self._index_factory = index_factory
        self._ann_path = ann_path
        self.reset()

    def _new_vector_tier(self) -> VectorTier:
        """Construct the vector tier with this brain's config + optional ANN seam. One place so
        reset() and restore() stay consistent (the restore path is where demotion used to
        bite)."""
        return VectorTier(self.embedder, w_sem=self.config.w_sem, w_lex=self.config.w_lex,
                          overfetch=self.config.overfetch, index_factory=self._index_factory,
                          ann_path=self._ann_path)

    # ------------------------------------------------------------------ identifiers --
    def _next_id(self, kind: str) -> str:
        self._counter += 1
        return f"{kind}-{self._counter:012d}"

    # ------------------------------------------------------------------ write path ---
    def write(self, observation: str, *, session_id: str, timestamp: float,
              metadata: dict | None = None) -> WriteReceipt:
        with self._lock:
            receipt = self._write(observation, session_id=session_id, timestamp=timestamp,
                                  metadata=metadata)
            # Journal only AFTER _write returns — a write whose validation raised never reaches
            # here, so the journal holds successful writes only and a replay can never re-hit a
            # validation error. Skipped while replaying (the replayed op is already in the
            # journal — re-journaling would duplicate it and desync compaction counting).
            if self._store is not None and not self._replaying:
                self._store.append("write", {"observation": observation,
                                             "session_id": session_id, "timestamp": timestamp,
                                             "metadata": metadata})
                self._journal_since += 1
                if self._journal_since >= self._compact_every:
                    self._compact_store()
            return receipt

    def _write(self, observation: str, *, session_id: str, timestamp: float,
               metadata: dict | None = None) -> WriteReceipt:
        if not isinstance(observation, str):
            raise TypeError(f"observation must be str, got {type(observation).__name__}")
        if not math.isfinite(timestamp):
            # A non-finite wall_ts silently breaks every bi-temporal `valid_from <= t`
            # comparison, leaving an "open" but un-queryable fact — reject it at the door.
            raise ValueError(f"timestamp must be finite, got {timestamp!r}")
        meta = metadata or {}
        self._derived = None  # any write may change the derived planner/stale state
        fact = meta.get("fact")
        if fact is not None:
            _validate_fact(fact)
        # Track B: derive facts from raw text when none were supplied. The first is the
        # "primary" fact driving contradiction/gate bookkeeping; every extracted fact is
        # routed to the structured tier below.
        facts = [fact] if fact else []
        if not facts and self.extractor is not None:
            # Echo suppression (Pillar B): a VERBATIM restatement of a statement whose facts
            # were already asserted is reinforcement, not new information — re-asserting it
            # would resurrect a superseded value with a fresh valid_from (a later "Alice
            # lives in Riga" echo would flip the current city back after her move). A
            # genuine revert needs new phrasing; an exact byte-duplicate is an echo.
            if normalize(observation) in self._asserted_texts:
                pass  # facts stay [] -> the gate reinforces the near-duplicate below
            else:
                self._cost["extract_calls"] += 1
                # First-person resolution is OPT-IN: it fires only when the caller declares
                # who is speaking (metadata "speaker"/"actor"), so ingestion without that
                # signal is byte-identical to before.
                speaker = meta.get("speaker") or meta.get("actor")
                ex_kwargs: dict = {}
                if speaker and self._extractor_takes_speaker:
                    ex_kwargs["speaker"] = speaker
                if self._extractor_takes_ref_ts:
                    ex_kwargs["ref_ts"] = timestamp
                if self._extractor_takes_resolve_role:
                    ex_kwargs["resolve_role"] = self._resolve_role
                facts = self.extractor.extract(observation, **ex_kwargs)
                if facts:
                    self._asserted_texts.add(normalize(observation))
        # Entity resolution (P1): canonicalize each fact's SUBJECT to a stable key — copy first
        # so the caller's metadata dict is never mutated. A byte no-op on single-token,
        # honorific-free subjects (the synthetic harness), so AC-1/AC-9 stay byte-identical.
        facts = [{**f, "subject": self._canonical_subject(f["subject"])} for f in facts]
        fact = facts[0] if facts else None
        embedding = self.embedder.embed(observation)
        self._cost["embed_calls"] += 1
        self._cost["embed_tokens"] += count_tokens(observation)

        nearest = self.vector.nearest(embedding)
        max_cos = max(0.0, min(1.0, nearest[1])) if nearest else 0.0
        novelty = 1.0 - max_cos
        # Opt-in predictive (log-prob) surprise: fuse the lexical novelty with the model's
        # prediction error on this observation given its nearest-k stored memories. Context
        # comes from the SAME embedding already computed above (nearest_k, no extra embed
        # pass). Fusion is max — a paraphrase with low lexical novelty but an unpredictable
        # buried update still clears the gate. Default None -> novelty is unchanged.
        novelty_effective = novelty
        if self._predictive is not None:
            ctx = [r.text for r, _ in self.vector.nearest_k(embedding, self._predictive.k_context)]
            predictive = self._predictive.score(observation, ctx)
            self._cost["surprise_calls"] += 1
            novelty_effective = max(novelty, predictive)
        contradiction = self._is_contradiction(fact)
        decision = self.gate.decide(novelty_effective, contradiction=contradiction)
        self._decisions[decision] += 1
        surprise = 1.0 if contradiction else novelty_effective
        # Write-decision trace (the Pillar-B selectivity signal). Content is NOT logged — only
        # the decision + scores — so no observation text/PII reaches the log stream.
        logger.debug("write decision=%s novelty=%.3f contradiction=%s session=%s",
                     decision, novelty, contradiction, session_id)

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
                fact_key=[normalize(fact["subject"]), normalize(fact["predicate"]),
                          normalize(fact["object"])] if fact else None,
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
        for f in facts:
            self._route_fact(f, rec_id, session_id, timestamp)
        self._note_session(session_id, timestamp)
        return WriteReceipt(stored=decision == STORE, reason=decision,
                            record_id=rec_id, surprise=surprise)

    def _resolve_role(self, noun: str) -> str | None:
        """Definite-NP coreference lookup for a fact-extractor: the SOLE entity that currently
        holds the role ``noun`` (an open ``(subject, role, noun)`` fact), or ``None`` when
        there are zero or more than one — fail-closed on ambiguity. Called inside ``_write``
        (already under the lock), so it reads the structured tier directly."""
        holders = self.structured.subjects_where("role", noun)
        return holders[0] if len(holders) == 1 else None

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
        # Retroactive facts: a caller may state WHEN the fact became true in the world
        # (fact["valid_from"]) separately from when it was recorded (`timestamp`) — the
        # bi-temporal distinction (PRD §5.2). Default: valid time == transaction time.
        self.structured.assert_fact(
            fact_id=self._next_id("fact"),
            subject=fact["subject"], predicate=fact["predicate"], object=fact["object"],
            valid_from=fact.get("valid_from", timestamp), created_at=timestamp,
            provenance={"episode_id": episode_id, "session_id": session_id,
                        "wall_ts": timestamp},
        )
        # The entity set is owned by the resolver (registered when the subject was
        # canonicalized in write()); _entities reads through to it — nothing to add here.

    # ----------------------------------------------------------- structured answers --
    def answer_structured(self, subject: str, predicate: str, *, at: float | None = None) -> str:
        """Exact / as-of-time answer from the structured tier (empty string if unknown)."""
        with self._lock:
            f = self.structured.resolve(self._resolver.canonical(subject), predicate, at)
            return f.object if f else ""

    def answer_path(self, subject: str, predicates: list[str], *,
                    at: float | None = None) -> str:
        """Multi-hop relational answer from the structured tier."""
        with self._lock:
            f = self.structured.resolve_path(self._resolver.canonical(subject), predicates, at)
            return f.object if f else ""

    def answer_who(self, predicate: str, object: str, *, at: float | None = None) -> list[str]:
        """Inverse / set query: every subject for which (predicate, object) currently holds
        (or held as believed at ``at``) — "who lives in Berlin?", "who reports to Bob?"."""
        with self._lock:
            return self.structured.subjects_where(predicate, object, at)

    # ------------------------------------------------------------------ erasure ------
    def forget(self, subject: str, *, predicate: str | None = None) -> dict:
        """Hard-erase an entity's data (e.g. a GDPR erasure request) — the ONE deliberate
        exception to the never-hard-delete invariant, on explicit request only.

        Removes: the subject's structured facts, open AND superseded history (one predicate
        if given; on a full-subject forget, also facts naming the subject as OBJECT — other
        entities' facts embedding this entity's data); episodic/vector records ASSERTING a
        removed fact; vector records entity-tagged with the subject; the echo-guard entries
        of removed texts; and the resolver vocabulary entry (full forget).

        Documented limit: a free-text record that merely MENTIONS the entity with no
        extracted fact link is not traceable to it and remains — full text-level erasure
        needs a scan the caller can do via the returned report + their own audit.

        Store semantics: forget mutates state OUTSIDE the write stream (it deletes, not
        appends), so a journal replay could not reproduce it. When a store is attached we
        compact immediately after — the new snapshot captures the post-forget state and the
        journal restarts empty."""
        with self._lock:
            out = self._forget(subject, predicate=predicate)
            self._compact_if_attached()
            return out

    def _forget(self, subject: str, *, predicate: str | None = None) -> dict:
        self._derived = None
        ns = self._resolver.canonical(subject)
        np_ = normalize(predicate) if predicate is not None else None
        facts_removed = self.structured.forget(ns, predicate, as_object=predicate is None)

        doomed_rids: set[str] = set()
        for r in self._episodes:
            if r.fact_key:
                s, p, o = _fact_key_parts(r)
                if (s == ns and (np_ is None or p == np_)) or (np_ is None and o == ns):
                    doomed_rids.add(r.id)
        if np_ is None:  # full forget: entity-tagged vector records go too
            for vr in self.vector.meta.values():
                if ns in vr.entities_norm:
                    doomed_rids.add(vr.rid)

        kept: list[EpisodicRecord] = []
        episodes_removed = 0
        for r in self._episodes:
            if r.id in doomed_rids:
                episodes_removed += 1
                self._asserted_texts.discard(normalize(r.content))
            else:
                kept.append(r)
        self._episodes = kept
        self._ep_by_id = {r.id: r for r in kept}
        for rid in sorted(doomed_rids):
            self.vector.remove_by_rid(rid)
        if np_ is None:
            self._resolver.forget(ns)
        return {"facts": facts_removed, "episodes": episodes_removed,
                "vector_records": len(doomed_rids)}

    # ------------------------------------------------------------------ query path ---
    def query(self, question: str, *, session_id: str, timestamp: float,
              k: int = 8, window: tuple[float, float] | None = None) -> Retrieval:
        """Hybrid retrieval (PRD §7): plan -> fetch structured + vector -> fuse, re-rank,
        supersede-filter. The exact current/as-of fact is surfaced first; superseded values
        are dropped so stale contradictions never reach the agent.

        ``window`` optionally scopes episodic recall to a ``(from_ts, to_ts)`` time range —
        time-scoped memory ("what did I note last spring?"), threaded to the vector tier's
        window filter. Default None → unscoped (byte-identical to before; AC-1/AC-9 intact).
        """
        with self._lock:
            return self._query(question, session_id=session_id, timestamp=timestamp,
                               k=k, window=window)

    def _query(self, question: str, *, session_id: str, timestamp: float,
               k: int = 8, window: tuple[float, float] | None = None) -> Retrieval:
        if not isinstance(question, str):
            raise TypeError(f"question must be str, got {type(question).__name__}")
        if not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be finite, got {timestamp!r}")
        if window is not None and not (
                isinstance(window, tuple) and len(window) == 2
                and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                        and math.isfinite(x) for x in window)
                and window[0] <= window[1]):
            raise ValueError(f"window must be a (from_ts, to_ts) tuple with from<=to, "
                             f"got {window!r}")
        pred_vocab, relation_preds, (stale_rids, stale_pairs) = self._derived_state()
        plan = self.planner.plan(question, entities=self._entities, predicates=pred_vocab,
                                 relation_preds=relation_preds, session_ts=self._session_ts,
                                 fuzzy_cutoff=self.config.fuzzy_entity_cutoff)
        # Entity resolution: map the matched entity token to its canonical subject key, so a
        # question naming "Erin" or "Ms. Smith" reaches the "erin smith" facts (identity on the
        # synthetic harness's single-token names).
        cent = self._resolver.canonical(plan.entity) if plan.entity else None
        lines: list[str] = []
        citations: list[Citation] = []

        hit = False
        if not plan.open_ended and plan.hops and cent is not None:
            # Multi-hop: surface the final answer line, but cite EVERY fact in the chain so
            # the whole support set is attributable (not just the last hop). cent is non-None
            # here — a non-open-ended plan always names an entity (normalize() needs a str).
            chain = self.structured.resolve_path_chain(cent, plan.hops, plan.as_of)
            if chain:
                lines.append(render_fact(plan, chain[-1]))
                for hf in chain:
                    citations.append(Citation(record_id=hf.id, provenance=hf.provenance,
                                              valid_interval=(hf.valid_from, hf.valid_to)))
                hit = True
        elif not plan.open_ended and cent is not None and plan.predicate is not None:
            f = self.structured.resolve(cent, plan.predicate, plan.as_of)
            if f is not None:
                lines.append(render_fact(plan, f))
                citations.append(Citation(record_id=f.id, provenance=f.provenance,
                                          valid_interval=(f.valid_from, f.valid_to)))
                hit = True

        if not hit:
            # Backstop: the question was open-domain, OR it routed to a predicate the entity
            # doesn't have (a mis-keyworded plan, e.g. "works" -> role on a company question).
            # Either way, don't fall straight to vector-only — surface high-precision facts
            # for EVERY known entity named in the question (round-robin so a multi-entity
            # question surfaces each), reserving budget for vector recall. This is the main
            # reason a hybrid store loses to plain RAG on open benchmarks.
            # Reserve HALF the context for vector recall: the old all-but-one reservation
            # let the backstop flood the payload with structured facts, starving raw
            # episodic recall exactly on the open-domain questions that need it.
            budget = max(1, (self._max_ctx + 1) // 2)
            qtoks = set(tokenize(question, drop_stop=False))
            # Canonicalize matched tokens and dedupe (so "erin" and "smith" -> one "erin smith").
            mentioned = sorted({self._resolver.canonical(e)
                                for e in self._entities if e in qtoks})
            preds_by_entity = [self.structured.predicates_for(e) for e in mentioned]
            depth = max((len(p) for p in preds_by_entity), default=0)
            for i in range(depth):  # round-robin: each entity's i-th fact before any (i+1)-th
                for ent, preds in zip(mentioned, preds_by_entity, strict=True):
                    if i >= len(preds) or len(lines) >= budget:
                        continue
                    f = self.structured.resolve(ent, preds[i], plan.as_of)
                    if f is not None:
                        lines.append(render_fact(plan, f))
                        citations.append(Citation(record_id=f.id, provenance=f.provenance,
                                                  valid_interval=(f.valid_from, f.valid_to)))
                if len(lines) >= budget:
                    break

        vhits = self.vector.search(question, k=k, t=timestamp, entity=cent, window=window)
        for it in fuse(vhits, now=timestamp, stale_rids=stale_rids, stale_pairs=stale_pairs,
                       w_rel=self.config.w_rel, w_rec=self.config.w_rec, w_imp=self.config.w_imp,
                       half_life_seconds=self.config.half_life_seconds):
            if len(lines) >= self._max_ctx:
                break
            lines.append(it.text)
            citations.append(Citation(record_id=it.rid, provenance=it.provenance,
                                      valid_interval=it.valid_interval))

        context = "\n".join(f"[{i}] {ln}" for i, ln in enumerate(lines, 1))
        tokens_in = count_tokens(context)
        self._cost["embed_calls"] += 1  # search() embeds the question once
        self._cost["embed_tokens"] += count_tokens(question)
        self._cost["queries"] += 1
        self._cost["context_tokens_served"] += tokens_in
        return Retrieval(context=context, citations=citations, tokens_in=tokens_in)

    def _derived_state(self):
        """Query-path state derived purely from the stores — predicate vocabulary, relation
        predicates (object is itself a known entity — auto-detected, not hardcoded to
        "manager"), and the supersede filters. Previously recomputed with full fact scans on
        EVERY query; now cached and invalidated by any mutation (write/consolidate/forget/
        restore/reembed)."""
        if self._derived is None:
            pred_vocab = frozenset(normalize(f.predicate) for f in self.structured.facts)
            relation_preds = frozenset(normalize(f.predicate) for f in self.structured.facts
                                       if normalize(f.object) in self._entities)
            self._derived = (pred_vocab, relation_preds, self._stale_filters())
        return self._derived

    def _stale_filters(self) -> tuple[set[str], list[tuple[frozenset[str], frozenset[str]]]]:
        """Supersede-filtering inputs for :func:`fuse` — provenance-linked first, entity-scoped
        token match as the free-text fallback.

        Returns ``(stale_rids, stale_pairs)``:

        * ``stale_rids`` — episode ids whose asserted triple (``fact_key``) is now superseded
          (its object differs from the key's current value). Exact identity — a record that
          *asserted* a stale fact is dropped, one that merely mentions similar words is not.
        * ``stale_pairs`` — ``(subject_tokens, value_tokens)`` per superseded value, for
          records with NO fact link: dropped only when the text contains the subject AND the
          full stale value. The old filter matched value tokens alone, store-wide — once any
          role "manager" was superseded, every record containing the word "manager"
          ("Erin's manager is Bob") was silently filtered forever.

        Staleness is judged PER (subject, predicate) key: a value can be stale for Alice while
        current for Bob (the old global open-values subtraction hid Alice's stale record)."""
        current_obj = {(normalize(f.subject), normalize(f.predicate)): normalize(f.object)
                       for f in self.structured.open_facts}
        stale_keys: set[tuple[str, str, str]] = set()
        stale_pairs: list[tuple[frozenset[str], frozenset[str]]] = []
        seen_pairs: set[tuple[frozenset[str], frozenset[str]]] = set()
        for f in self.structured.facts:
            if f.is_open:
                continue
            ns, np_, no = normalize(f.subject), normalize(f.predicate), normalize(f.object)
            if current_obj.get((ns, np_)) == no:
                continue  # value is (again) current for this key — not stale
            stale_keys.add((ns, np_, no))
            pair = (frozenset(tokenize(f.subject)), frozenset(tokenize(f.object)))
            if pair[1] and pair not in seen_pairs:
                seen_pairs.add(pair)
                stale_pairs.append(pair)
        stale_rids = {r.id for r in self._episodes
                      if r.fact_key is not None and tuple(_fact_key_parts(r)) in stale_keys}
        return stale_rids, stale_pairs

    # ------------------------------------------------------------------ maintenance --
    def reembed(self, new_embedder) -> dict:
        """Migrate the store to a new embedding model (PRD §12 re-embed-on-upgrade).

        Non-lossy: source text, structured facts and provenance are untouched — only the
        vectors are recomputed and each record's ``embed_model_id`` is stamped with the new
        model. Lets the layer survive an embedder upgrade without losing recall.
        """
        with self._lock:
            self._derived = None
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
        so accuracy is preserved and the audit trail is intact.

        Store semantics: consolidation rewrites the episodic tier in place (prune/merge), which
        a journal replay of raw writes could not reproduce — so when a store is attached we
        compact immediately after, snapshotting the consolidated state."""
        with self._lock:
            rep = self._consolidate()
            self._compact_if_attached()
            return rep

    def _consolidate(self) -> ConsolidationReport:
        self._derived = None
        episodes_in = len(self._episodes)
        current_obj = {(normalize(f.subject), normalize(f.predicate)): normalize(f.object)
                       for f in self.structured.open_facts}

        groups: dict[tuple[str, str], list[EpisodicRecord]] = {}
        free: list[EpisodicRecord] = []
        for r in self._episodes:
            if r.fact_key:
                subj, pred, _ = _fact_key_parts(r)
                groups.setdefault((subj, pred), []).append(r)
            else:
                free.append(r)

        new_eps: list[EpisodicRecord] = []
        dupes_merged = claims_out = pruned = 0

        for (subj, pred), members in groups.items():
            cur = current_obj.get((subj, pred))
            current_eps = [r for r in members
                           if r.fact_key and _fact_key_parts(r)[2] == cur]
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
                                             self.config.free_dedup_threshold):
            if len(cluster) > 1:
                new_eps.append(self._merge_to_claim(cluster))
                dupes_merged += len(cluster) - 1
                claims_out += 1
            else:
                new_eps.append(cluster[0])

        self._episodes = new_eps
        self._ep_by_id = {r.id: r for r in new_eps}
        contradictions_resolved = sum(1 for f in self.structured.facts if not f.is_open)
        logger.info("consolidate: episodes_in=%d claims_out=%d dupes_merged=%d pruned=%d "
                    "contradictions_resolved=%d", episodes_in, claims_out, dupes_merged,
                    pruned, contradictions_resolved)
        return ConsolidationReport(episodes_in=episodes_in, claims_out=claims_out,
                                   dupes_merged=dupes_merged,
                                   contradictions_resolved=contradictions_resolved, pruned=pruned)

    def _merge_to_claim(self, members: list[EpisodicRecord]) -> EpisodicRecord:
        rep = representative(members)
        content = rep.content
        if self.summarizer is not None and len(members) > 1:
            # Real summarization (PRD §8 "cluster & summarize"): one compact claim over the
            # cluster. Falls back to the representative if the model returns nothing.
            notes = "\n".join(f"- {r.content}" for r in members)
            out = self.summarizer.complete(
                "Merge these related memory notes into ONE short factual sentence. "
                "Keep every name and value exactly as written; add nothing new.\n"
                f"{notes}\nSummary:").strip()
            if out:
                content = out.splitlines()[0].strip()
        for r in members:
            self.vector.remove_by_rid(r.id)
        # Propagate the members' subjects to the claim so entity-filtered vector search can
        # still find it (previously entities=[] made every merged claim invisible to
        # entity-scoped recall — consolidation silently degraded the store it was compacting).
        entities = sorted({_fact_key_parts(r)[0] for r in members if r.fact_key})
        claim = EpisodicRecord(
            id=self._next_id("claim"), session_id=rep.session_id, seq=len(self._episodes),
            wall_ts=rep.wall_ts, actor="system", content=content,
            embed_model_id=self.embedder.model_id, surprise=rep.surprise,
            provenance={"source": "consolidation", "merged_from": [r.id for r in members]},
            support_count=sum(r.support_count for r in members),
            last_seen_ts=max(r.last_seen_ts for r in members),
            tier="semantic", supports=[r.id for r in members],
            fact_key=rep.fact_key,  # keep the fact link so a LATER supersede still filters it
        )
        self.vector.add(rid=claim.id, text=claim.content, wall_ts=claim.wall_ts,
                        session_id=claim.session_id, tier="semantic", entities=entities)
        return claim

    def stats(self) -> StoreStats:
        with self._lock:
            return StoreStats(
                episodic_count=sum(1 for r in self._episodes if r.tier == "episodic"),
                structured_count=len(self.structured.facts),
                semantic_count=sum(1 for r in self._episodes if r.tier == "semantic"),
                bytes=len(self.snapshot()),
                embed_model_id=self.embedder.model_id,
            )

    def metrics(self) -> dict:
        """Operational metrics for observability (PRD §G): the write-decision breakdown and
        store size. ``discard_rate`` is the selectivity signal — it should stay high on a
        redundant stream (Pillar B), evidence the gate is doing its job rather than logging
        everything. Cheap, side-effect-free, and safe to poll between writes."""
        with self._lock:
            return self._metrics()

    def _metrics(self) -> dict:
        d = self._decisions
        total = d["stored"] + d["reinforced"] + d["discarded"]
        # Count records directly (don't call stats(), which serializes the whole store just
        # to measure bytes) so metrics() stays genuinely cheap and safe to poll.
        episodic = sum(1 for r in self._episodes if r.tier == "episodic")
        semantic = sum(1 for r in self._episodes if r.tier == "semantic")
        c = self._cost
        return {
            "writes_total": total,
            "stored": d["stored"], "reinforced": d["reinforced"], "discarded": d["discarded"],
            "discard_rate": (d["discarded"] / total) if total else 0.0,
            "store_size": episodic + semantic,
            "structured_facts": len(self.structured.facts),
            # Cost axis (deterministic; no wall-clock). `avg_context_tokens` is the headline
            # retrieval-cost number — tokens of context served per query — directly comparable
            # to a rival's, for the Track-D "≤ its cost" claim.
            "cost": {**c,
                     "avg_context_tokens": (c["context_tokens_served"] / c["queries"])
                     if c["queries"] else 0.0,
                     # $ estimate of the metered hot-path tokens when a Pricing is configured
                     # (embeddings + served context; see curated_brain.pricing for scope).
                     **({"estimated_usd": self.pricing.estimate(c),
                         "usd_per_query": (self.pricing.estimate(c) / c["queries"])
                         if c["queries"] else 0.0}
                        if self.pricing is not None else {})},
        }

    def reset(self) -> None:
        # Store semantics: reset clears state outside the write stream; compact immediately so a
        # reopen reflects the empty store (not the pre-reset journal). See _compact_if_attached.
        with self._lock:
            self._reset()
            self._compact_if_attached()

    def _reset(self) -> None:
        self._derived = None
        self._counter = 0
        self._episodes = []
        self._ep_by_id = {}
        if self.extractor is not None and hasattr(self.extractor, "reset"):
            self.extractor.reset()  # clear any coreference context tied to the old store
        self.structured = StructuredTier()
        self.vector = self._new_vector_tier()
        self.gate = SurpriseGate.from_dict({**self._gate_cfg, "seen": 0, "stored": 0})
        self._resolver = EntityResolver()  # owns the entity vocabulary (`_entities` reads through)
        self._session_ts = {}
        self._asserted_texts = set()  # normalized fact-bearing texts (extraction echo guard)
        # Operational counters (observability, not core state — kept out of the snapshot).
        self._decisions = {"stored": 0, "reinforced": 0, "discarded": 0}
        # Cost accounting for the write + query hot path (the comparable Track-D axes:
        # cost-per-write and cost-per-query) and extraction. Deterministic (token counts,
        # not wall-clock), so it's the "cost" axis for the benchmark table ("accuracy AND
        # cost") and the "≤ its cost" DONE clause. Scope note: consolidation's internal
        # re-embeds (amortized maintenance) and latency (wall-clock — excluded by design to
        # keep the core deterministic) are intentionally not metered here.
        self._cost = {"embed_calls": 0, "embed_tokens": 0, "extract_calls": 0,
                      "surprise_calls": 0, "queries": 0, "context_tokens_served": 0}

    @property
    def _entities(self) -> set[str]:
        """Entity vocabulary the planner matches questions against — owned by the resolver
        (canonical names + standalone single tokens + non-ambiguous component tokens)."""
        return self._resolver.entities

    def _canonical_subject(self, raw: str) -> str:
        """Canonicalize a subject (entity resolution), preserving the original surface when no
        merge to a different entity occurred — so non-resolved subjects keep their case (a byte
        no-op for the common write), while a resolved partial takes the canonical key."""
        c = self._resolver.resolve_and_register(raw)
        return raw if normalize(raw) == c else c

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
            # The session->timestamp map drives as-of-by-session queries and is built from
            # ALL writes (incl. discarded), so it cannot be faithfully rebuilt from the stored
            # episodes alone — persist it or restore() silently shifts C6 answers.
            "session_ts": sorted(self._session_ts.items()),
            # Resolver ambiguity history (poisoned tokens, singletons) is likewise not
            # derivable from the final canonical subjects — persist it or a token refused
            # before snapshot can promote after restore (same query, different answer).
            "resolver": self._resolver.to_dict(),
            "asserted_texts": sorted(self._asserted_texts),
        }

    def snapshot(self) -> bytes:
        with self._lock:
            payload = to_jsonable(self._state())
            return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        # Store semantics: restore replaces state wholesale; compact immediately so the snapshot
        # matches the restored state and the old journal is discarded. Skipped during attach's
        # own replay (guarded by _replaying inside _compact_if_attached).
        with self._lock:
            self._restore(blob)
            self._compact_if_attached()

    def _restore(self, blob: bytes) -> None:
        # Snapshots carry data, not tuning: CBConfig is not persisted, so retrieval/vector
        # knobs come from THIS brain's config — but the gate's runtime state (thresholds
        # included) is restored from the snapshot, overriding the config's gate knobs.
        # restore() takes UNTRUSTED bytes (a file, a network blob). Validate structure up
        # front and fail with a clear ValueError, rather than letting a malformed blob crash
        # deep inside with an opaque KeyError/TypeError or splat unexpected fields into a
        # dataclass. (No behavior change for a valid snapshot — happy path is unchanged.)
        try:
            raw = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"snapshot is not valid UTF-8 JSON: {e}") from e
        if not isinstance(raw, dict):
            raise ValueError(f"snapshot must be a JSON object, got {type(raw).__name__}")
        state = from_jsonable(raw)
        _validate_snapshot(state)
        snap_dim = state.get("config", {}).get("dim")
        if snap_dim is not None and snap_dim != self.embedder.dim:
            # A dim mismatch used to surface only later, as a shape error mid-query (or,
            # worse, silently wrong similarities if the dims happened to agree elsewhere).
            raise ValueError(f"snapshot embedding dim {snap_dim} != live embedder dim "
                             f"{self.embedder.dim}; restore with the matching embedder, "
                             f"then migrate via reembed()")
        self._derived = None
        self._counter = state["counter"]
        self._episodes = [EpisodicRecord(**d) for d in state["episodic"]]
        self._ep_by_id = {r.id: r for r in self._episodes}
        self.structured = StructuredTier()
        self.structured.load(state.get("structured", []))
        self.vector = self._new_vector_tier()
        if state.get("vector"):
            self.vector.load(state["vector"])
        self.gate = SurpriseGate.from_dict(state["gate"]) if state.get("gate") else SurpriseGate()
        if state.get("resolver"):  # faithful restore incl. ambiguity/singleton history
            self._resolver = EntityResolver.from_dict(state["resolver"])
        else:
            # Legacy snapshot: rebuild from stored subjects (lossy — cannot recover which
            # tokens were poisoned as ambiguous by non-fact registrations).
            self._resolver = EntityResolver()
            for f in self.structured.facts:
                self._resolver.register_canonical(f.subject)
        self._asserted_texts = set(state.get("asserted_texts", []))
        if "session_ts" in state:  # faithful restore of the as-of-by-session map
            self._session_ts = {int(k): v for k, v in state["session_ts"]}
        else:  # legacy snapshot: rebuild from stored episodes (lossy for discarded sessions)
            self._session_ts = {}
            for r in self._episodes:
                self._note_session(r.session_id, r.wall_ts)
        # Operational counters describe *this instance's* activity, not the restored store,
        # so reset them — otherwise metrics() would report counts unrelated to the snapshot.
        self._decisions = {"stored": 0, "reinforced": 0, "discarded": 0}
        self._cost = {"embed_calls": 0, "embed_tokens": 0, "extract_calls": 0,
                      "surprise_calls": 0, "queries": 0, "context_tokens_served": 0}

    def save(self, path: str) -> None:
        """Durably persist the whole store to ``path`` (survives a process restart). Thin,
        deterministic wrapper over :meth:`snapshot` — the bytes are the canonical state."""
        with self._lock:
            with open(path, "wb") as fh:
                fh.write(self.snapshot())
            # Opt-in fast path: when an ANN sidecar is configured and the live index supports a
            # native save, persist the built graph next to the snapshot so load() can skip the
            # rebuild. A no-op for the BruteForce default (its vectors live in the snapshot).
            self.vector.save_sidecar()

    def load(self, path: str) -> None:
        """Reopen a store previously written by :meth:`save`, replacing current state."""
        with self._lock:
            with open(path, "rb") as fh:
                self.restore(fh.read())

    # -------------------------------------------------------- durable store (journal) --
    def attach_store(self, store: SqliteStore, *, compact_every: int = 256) -> None:
        """Attach a durable :class:`~curated_brain.store.SqliteStore` for incremental, O(1)
        per-write persistence (replacing the O(store-size) full-snapshot ``save`` on every
        write).

        On attach: if the store already holds state, this brain is rebuilt from it — the last
        snapshot via the normal :meth:`restore` (validators and all), then every journaled write
        re-executed through the normal write path. Because writes are deterministic in their raw
        args, the replay reproduces byte-identical state. Afterwards each successful
        :meth:`write` appends one journal row; every ``compact_every`` rows (and each
        consolidate/forget/reset/restore) triggers a full-snapshot compaction.

        Attaching must not change in-memory behavior: replay runs through the same code that
        produced the original state, so :meth:`snapshot` bytes and query results are unchanged.
        """
        if compact_every < 1:
            raise ValueError(f"compact_every must be >= 1, got {compact_every!r}")
        with self._lock:
            blob, journal = store.load()
            self._replaying = True  # guard: replayed writes/restore must not re-journal
            try:
                if blob is not None:
                    self._restore(blob)
                for op, payload in journal:
                    # Only "write" ops are journaled (see write()); assert keeps an unknown op
                    # from silently corrupting the rebuild.
                    assert op == "write", f"unknown journal op {op!r}"
                    self._write(payload["observation"], session_id=payload["session_id"],
                                timestamp=payload["timestamp"], metadata=payload["metadata"])
            finally:
                self._replaying = False
            # Bind the store, then snapshot the freshly-rebuilt state as the new baseline so the
            # journal starts empty (a torn crash after this replays nothing until the next write).
            self._store = store
            self._compact_every = compact_every
            self._journal_since = 0
            store.compact(self.snapshot())

    def detach_store(self) -> None:
        """Stop journaling to the attached store (leaves it as-is on disk). No-op if none."""
        with self._lock:
            self._store = None
            self._compact_every = 0
            self._journal_since = 0

    def _compact_store(self) -> None:
        """Write a fresh full snapshot to the store and reset the journal counter. Callers hold
        the lock and have already checked a store is attached."""
        assert self._store is not None
        self._store.compact(self.snapshot())
        self._journal_since = 0

    def _compact_if_attached(self) -> None:
        """Compact after a state-replacing op (consolidate/forget/reset/restore) that a journal
        replay could not reproduce. No-op during attach's own replay (the replay guard) so the
        rebuild is not snapshotted mid-flight."""
        if self._store is not None and not self._replaying:
            self._compact_store()
