# The Curated Brain — Product Requirements Document

**Status:** Draft v0.1 · **Date:** 2026-06-17 · **Owner:** Marlon · **Target stack:** Python 3.11+

---

## 0. TL;DR

The Curated Brain is a **persistent memory layer for an LLM agent**, built entirely on **frozen-model APIs** — no training, no fine-tuning. All intelligence comes from how we *orchestrate* a frozen LLM plus an embedding model.

The core bet: **memory is a curation problem, not a search problem.** Transformers forget everything between sessions. The common patch — RAG — fails at *write* time, not query time: retrieval quality is upper-bounded by what got stored and how it's organized. Log everything and the signal-to-noise ratio collapses, so retrieval drowns. We invest at write time instead.

Four hard requirements:

1. **Two-tier store** — a structured tier (entities, relations, timestamps, provenance) for exact queries, and a vector tier for fuzzy semantic recall, with retrieval that fuses both. Provenance and time are first-class.
2. **Selective by default** — store what the model did *not* predict, gated on prediction error / surprise. Most input is discarded.
3. **Self-organizing** — a background consolidation process compresses raw episodic records into a semantic tier: summarize, deduplicate, extract recurring patterns, prune noise, and resolve contradictions between old and new facts.
4. **Harness-scorable** — implements the Longitudinal Memory Eval Harness backend adapter interface, so it can be scored directly.

**Success** (§10): an agent using this layer recalls facts from arbitrarily far back, updates beliefs when facts change (no stale contradictions surface), retrieves accurately and cheaply, and does **not** grow unboundedly as noise accumulates — beating naive RAG, long-context stuffing, and no-memory baselines on *every* eval category. The win condition is a memory layer that gets **more useful the longer it runs**, not more bloated.

> **Scope note on the harness:** This PRD treats the Longitudinal Memory Eval Harness as an **external, given** scorer. We specify the *backend adapter interface* we implement against it (§9) and enumerate the eval categories and baselines as load-bearing acceptance criteria (§10). We do **not** build the harness runner itself. If we later decide to own the harness, §9–§10 are the seam to extend.

---

## 1. Problem & thesis

### 1.1 The failure mode

A frozen transformer is **stateless across sessions**: nothing from yesterday's conversation survives into today's context unless something external puts it there. RAG is the standard external patch — embed everything, retrieve top-k at query time, stuff it into context.

RAG's ceiling is set at **write time**, not query time. Three compounding failures:

- **Indiscriminate writes.** Logging every utterance fills the store with low-value tokens. Top-k retrieval then competes against a huge denominator of noise; precision falls as the corpus grows.
- **No structure.** A flat vector store answers "what's *similar*" but not "what is *true now*." Exact lookups ("what is X's current email?"), relational queries ("who reports to Y?"), and temporal queries ("what did we believe in March?") degrade into fuzzy similarity matches.
- **No reconciliation.** When a fact changes, the old embedding stays. Both the stale and current version are retrievable, and the agent surfaces contradictions.

### 1.2 The reframe

Treat memory as **curation**: decide *what to keep*, impose *structure*, and *continuously reconcile* old and new. This mirrors how biological memory works — fast episodic capture in the hippocampus, slow extraction of semantic structure in the neocortex, with replay-driven consolidation between them (Complementary Learning Systems theory: [McClelland, McNaughton & O'Reilly 1995](https://pubmed.ncbi.nlm.nih.gov/7624455/), updated by [Kumaran, Hassabis & McClelland 2016](https://pubmed.ncbi.nlm.nih.gov/27315762/)). It also picks up the strongest threads from recent agent-memory systems (§13): tiered, self-editing memory ([MemGPT](https://arxiv.org/abs/2310.08560)), reflection over a memory stream ([Generative Agents](https://arxiv.org/abs/2304.03442)), and bi-temporal knowledge graphs ([Zep / Graphiti](https://arxiv.org/abs/2501.13956)).

### 1.3 Hard constraint

**Frozen-model APIs only.** No weight updates. Every mechanism below — surprise estimation, consolidation, contradiction resolution — must be implementable with (a) an embedding endpoint and (b) a chat/completion endpoint, optionally exposing token logprobs. This keeps the layer model-portable and cheap to operate.

---

## 2. Goals & non-goals

### Goals

- A drop-in memory layer an agent calls to `write` observations and `query` for context.
- Retrieval that is **accurate** (high recall of relevant facts, low contradiction rate) and **cheap** (small context payloads, bounded latency).
- A store whose size grows **sublinearly** with input volume and whose quality is **stable or improving** as noise accumulates.
- Direct scorability via the harness adapter, beating all three baselines on every category.

### Non-goals (v1)

- No model training, fine-tuning, or adapter weights.
- Not a general-purpose vector database product; we depend on one, we don't ship one.
- No multi-tenant infrastructure, auth, or UI. Single logical agent, single store.
- No human-in-the-loop curation UI (provenance is retained so one *could* be built later).

---

## 3. Users & operating context

The **consumer is an agent loop**, not a human. The layer sits between the agent and the frozen LLM:

```
            observation                 query(question)
 agent ─────────────────▶  Curated Brain  ◀───────────────── agent
                                │  │
                  write path ◀──┘  └──▶ retrieval (context + citations)
                                │
                     background consolidation worker
```

Operating profile: a long-running assistant spanning **hundreds to thousands of sessions**, where facts drift (jobs change, preferences update, projects close) and most input is chit-chat or repetition that should never be persisted.

---

## 4. Core requirements (the four pillars)

### Pillar A — Two-tier store with hybrid retrieval

A **structured tier** and a **vector tier**, queried together.

- **Structured tier** holds entities, relations, and atomic facts with **bi-temporal** time and **provenance** on every record. Answers exact, relational, and as-of-time queries.
- **Vector tier** holds embeddings of episodic records and consolidated semantic claims, with metadata filters. Answers fuzzy semantic recall.
- **Retrieval fuses both** and filters out facts that are no longer valid.

*Provenance and time are first-class* — every stored item carries where it came from (source episode id, ingestion timestamp, embedding-model id) and when it is true (valid interval).

### Pillar B — Selective by default (surprise-gated writes)

The default action on input is **discard**. An item is persisted only when its **surprise** (prediction error / novelty) exceeds an adaptive threshold. Redundant input reinforces existing memory (a counter/recency bump) rather than creating duplicates. Target: the gate discards the large majority of a redundant stream while losing no salient facts.

### Pillar C — Self-organizing (background consolidation)

A periodic worker compresses the raw **episodic** tier into a durable **semantic** tier:

- **Summarize** clusters of related episodes into compact claims.
- **Deduplicate** near-identical records.
- **Extract** recurring entities, relations, and patterns into the structured tier.
- **Resolve contradictions** between old and new facts (supersede, don't delete).
- **Prune** low-value / expired noise and compact the episodic log.

This is what keeps the store from growing unboundedly and what makes it *more* useful over time.

### Pillar D — Harness adapter interface

Implements the Longitudinal Memory Eval Harness backend adapter (§9) so the whole layer can be scored directly against baselines, deterministically.

---

## 5. Architecture & data model

### 5.1 Components

| Component | Responsibility |
|---|---|
| **Ingestion** | Normalize an incoming observation into a candidate record; attach provenance + timestamps. |
| **Surprise gate** | Score novelty / prediction error; decide store vs. discard vs. reinforce (§6). |
| **Episodic store** | Append-only log of stored raw records (the "hippocampus"). |
| **Structured tier** | Entities, relations, atomic facts; bi-temporal; provenance. Exact + relational + as-of queries. |
| **Vector tier** | Embeddings over episodic + semantic records; metadata-filtered ANN search. |
| **Semantic tier** | Consolidated claims/summaries with support links back to episodes (the "neocortex"). |
| **Retrieval planner** | Routes a query across tiers, fuses + re-ranks, filters superseded facts (§7). |
| **Consolidation worker** | Background compaction / reconciliation (§8). |
| **Adapter** | Wraps the above behind the harness interface (§9). |

### 5.2 Data model (logical)

**Episodic record**

| Field | Notes |
|---|---|
| `id` | ULID (sortable by creation). |
| `session_id`, `seq` | Session + logical clock for ordering. |
| `wall_ts` | Provided/observed wall-clock time. |
| `actor` | user / agent / tool / system. |
| `content` | Raw text. |
| `embedding`, `embed_model_id` | Vector + model version (for re-embedding on upgrade). |
| `surprise` | Score that admitted it (§6). |
| `provenance` | Source ref(s); how it entered. |
| `support_count`, `last_seen_ts` | Reinforcement signal from deduped repeats. |

**Structured tier — entity**

| Field | Notes |
|---|---|
| `id`, `type`, `canonical_name` | Resolved entity. |
| `attributes` | Key/value facts. |
| `valid_from`, `valid_to` | **Valid time** (when true in the world). |
| `created_at`, `expired_at` | **Transaction time** (when recorded / invalidated in the system). |
| `confidence`, `provenance` | Support + source episodes. |

**Structured tier — relation / atomic fact**: `(subject, predicate, object)` carrying the same bi-temporal interval, `confidence`, and `provenance`. Superseding a fact sets `valid_to` / `expired_at` and links to the replacement — **never a hard delete** (provenance and history are preserved).

**Semantic claim**: `id`, `text`, `supports` (episode ids), `entities`, `confidence`, `last_reviewed_ts`. Lives in the vector tier (embedded) and links into the structured tier.

### 5.3 Storage choices (Python, pluggable)

- **Structured + bi-temporal + metadata:** SQLite (single-file, transactional, trivially snapshot/restorable for deterministic eval).
- **Vector index:** start with an in-process ANN (`hnswlib` or `faiss`); abstract behind a `VectorIndex` protocol so it can be swapped for `sqlite-vec` or a managed store.
- **Frozen models:** an `Embedder` and an `LLM` protocol (provider-agnostic), each recording a model id into provenance.

Everything sits behind narrow interfaces so no concrete vendor leaks into the core logic.

---

## 6. Write path — surprise gating

For each candidate record, estimate **surprise** = how much the new content was *not* predictable from existing memory. Three estimators, combinable, all frozen-model-friendly:

1. **Semantic novelty (primary, cheap):** `1 − max cosine similarity` to the k nearest existing memories. High novelty ⇒ likely worth keeping.
2. **Predictive surprise (secondary):** prompt the frozen LLM to predict the answer/next fact given currently-retrieved memory; compare to the actual observation. Large divergence ⇒ high surprise. If the API exposes **logprobs**, use token-level perplexity of the observation under a memory-conditioned prompt as a sharper proxy.
3. **Contradiction signal (override):** if the candidate conflicts with a stored fact, surprise is forced high — we *must* store it and trigger reconciliation (§8).

**Decision policy**

- `surprise ≥ θ` → **store** (episodic; route entities/relations to structured tier).
- `surprise < θ` and matches an existing memory → **reinforce** (bump `support_count` / `last_seen_ts`, no new row).
- otherwise → **discard**.

θ is **adaptive**, calibrated to a target write-rate budget (e.g. persist ≤ X% of a redundant stream) so the store stays selective regardless of input volume. Most input is discarded by design.

---

## 7. Retrieval — hybrid

1. **Plan.** Classify the query: exact/relational/temporal → structured tier; open-ended → vector tier; most queries hit **both**.
2. **Fetch.** Structured tier returns currently-valid facts (filtered by `valid_*` / `expired_*`); vector tier returns top-k episodic + semantic records under metadata filters (time window, entity, tier).
3. **Fuse & re-rank.** Combine candidates and score by **relevance × recency × importance/confidence** — the recency/importance/relevance weighting validated in [Generative Agents](https://arxiv.org/abs/2304.03442). **Superseded facts are dropped** so stale contradictions never surface.
4. **Return** a compact context payload **with citations** (provenance + timestamps) and a token budget far below long-context stuffing.

---

## 8. Consolidation — self-organizing

A background worker runs **periodically or when accumulated surprise crosses a threshold** (the reflection-trigger pattern from Generative Agents). It never sits on the critical path of a query.

Operations:

- **Cluster & summarize** related episodes into semantic claims (with support links retained).
- **Deduplicate** near-identical records; merge into a single claim with summed support.
- **Extract** recurring entities/relations/patterns into the structured tier.
- **Resolve contradictions:** when two facts conflict, keep the one with higher (recency, confidence, provenance quality); **supersede** the loser via bi-temporal invalidation, preserving history — the non-lossy update model from [Zep / Graphiti](https://arxiv.org/abs/2501.13956).
- **Prune** expired, low-surprise, unsupported noise; compact the episodic log.

Invariant: consolidation may compress and reorganize but **never destroys provenance** and **never hard-deletes the audit trail** of a superseded fact.

---

## 9. Eval-harness backend adapter interface (the contract)

The integration seam. The layer implements a single adapter class the harness drives. Sketch (Python, illustrative):

```python
class MemoryBackend(Protocol):
    def write(self, observation: str, *, session_id: str,
              timestamp: float, metadata: dict | None = None) -> WriteReceipt: ...
    # WriteReceipt: {stored: bool, reason: "stored"|"reinforced"|"discarded", record_id|None, surprise}

    def query(self, question: str, *, session_id: str, timestamp: float,
              k: int = 8) -> Retrieval: ...
    # Retrieval: {context: str, citations: [{record_id, provenance, valid_interval}], tokens_in}

    def consolidate(self) -> ConsolidationReport: ...
    # report: {episodes_in, claims_out, dupes_merged, contradictions_resolved, pruned}

    def stats(self) -> StoreStats: ...
    # {episodic_count, structured_count, semantic_count, bytes, embed_model_id}

    def reset(self) -> None: ...
    def snapshot(self) -> bytes: ...          # deterministic eval
    def restore(self, blob: bytes) -> None: ...
```

`timestamp` is supplied by the harness (memory must not depend on real wall-clock). `consolidate()` lets the harness simulate "sleep" between sessions. `stats()` exposes the size signal the **bounded-growth** category scores against. `snapshot/restore` make runs reproducible.

### 9.1 Eval categories & baselines

| # | Category | What it probes |
|---|---|---|
| C1 | **Long-range recall** | Facts injected many sessions ago still retrievable. |
| C2 | **Belief updating / contradiction** | After a fact changes, only the current value surfaces; zero stale contradictions. |
| C3 | **Retrieval accuracy & cost** | Precision/recall of relevant context vs. tokens-in + latency. |
| C4 | **Bounded growth / noise resistance** | Store size grows sublinearly; accuracy stable as noise accumulates. |
| C5 | **Relational / multi-hop** | Exact + multi-step queries over the structured tier. |
| C6 | **Temporal reasoning** | As-of-time queries ("what did we believe on date D?"). |

**Baselines to beat on every category:** (a) **naive RAG** (log-everything + top-k), (b) **long-context stuffing** (paste full history until the window fills), (c) **no-memory** (frozen model alone).

---

## 10. Acceptance criteria (the gate)

These are the load-bearing, pass/fail checks. **Stage 1 encodes them as the test suite** (the gate command). Numeric thresholds marked *(calibrate)* are finalized in Stage 1 against a seeded synthetic longitudinal dataset we ship; the **directional** requirement (beat all baselines) is hard and non-negotiable. Each criterion maps to a build stage (§11).

| ID | Criterion | Pass condition | Stage |
|---|---|---|---|
| **AC-1** | Adapter conformance | All `MemoryBackend` methods implemented; `snapshot→restore` is byte-deterministic; identical input + seed ⇒ identical outputs. | 1 |
| **AC-2** | Long-range recall (C1) | recall@k of injected facts ≥ **0.90** *(calibrate)* even when injected ≥ **50 sessions** prior; **>** naive-RAG and no-memory by a fixed margin. | 4 |
| **AC-3** | Belief updating (C2) | After a fact change, **0** probes return the stale value; superseded facts never returned as current; current value returned ≥ **0.95** *(calibrate)*. | 4 |
| **AC-4** | Retrieval cost (C3) | Mean `tokens_in` per query ≤ **25%** *(calibrate)* of long-context baseline **while** answer accuracy ≥ that baseline. | 4 |
| **AC-5** | Selectivity (Pillar B) | On a redundancy-heavy stream, gate **discards ≥ 80%** *(calibrate)* of input with **no loss** of salient facts (AC-2 still holds). | 5 |
| **AC-6** | Bounded growth (C4) | After injecting **K** noisy items, stored-item count grows **sublinearly** in K (target ≤ **20%** *(calibrate)* of raw input) **and** C1–C3 accuracy does not degrade beyond noise floor. | 6 |
| **AC-7** | Consolidation quality | After `consolidate()`: duplicate claims reduced ≥ **70%** *(calibrate)*; all seeded contradictions resolved to the correct current value; **provenance retained** for every surviving + superseded fact. | 6 |
| **AC-8** | Relational + temporal (C5, C6) | Multi-hop and as-of-time query accuracy ≥ **0.85** *(calibrate)* and **>** naive RAG. | 2 |
| **AC-9** | **Overall harness** | **Strictly** beats naive RAG, long-context, and no-memory on **every** category C1–C6 on the full longitudinal run. A tie or loss on *any* category means the project is **not** done. | 7 |

> **Definition of done for the whole project:** AC-1 through AC-9 pass, i.e. the gate command (§ CLAUDE.md `Done`) exits clean with all stage test suites green.

---

## 11. Build stages (7, commit-gated)

Each stage flips a defined subset of the §10 tests green and earns exactly one commit.

| Stage | Deliverable | Gate (tests that must pass) |
|---|---|---|
| **1** | Test scaffold + `MemoryBackend` ABC stub + **seeded synthetic longitudinal dataset generator** + `pyproject` (pytest, ruff). | AC-1 (conformance, determinism); dataset-generator tests; suite collects & runs (non-empty). |
| **2** | Structured tier: entities/relations, **bi-temporal** model, provenance; exact / relational / as-of queries. | AC-8. |
| **3** | Vector tier: embedder protocol, ANN index, semantic recall. | Vector recall unit tests (subset of C1). |
| **4** | Hybrid retrieval: planner, fusion re-rank, supersede-filtering. | AC-2, AC-3, AC-4. |
| **5** | Surprise-gated selective write path (novelty + contradiction + optional logprob). | AC-5 (recall from AC-2 preserved). |
| **6** | Consolidation worker: summarize, dedupe, extract, resolve, prune. | AC-6, AC-7. |
| **7** | Full longitudinal run vs. all three baselines across C1–C6. | AC-9. |

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Frozen-model surprise estimates are noisy. | Use embedding-novelty + contradiction as primary signals; LLM/logprob as secondary; calibrate θ to a write-budget rather than an absolute. |
| Consolidation LLM hallucinates merges or wrong claims. | Require support links for every claim; keep confidence; never hard-delete; consolidation is reversible because provenance is retained. |
| Contradiction resolver picks the wrong winner. | Bi-temporal history means resolution is reversible; tie-break on recency → confidence → provenance quality; log every supersede. |
| Consolidation cost / latency. | Fully async, off the query critical path, batched, triggered by surprise accumulation. |
| Overfitting to the synthetic dataset. | Generator is diverse + seedable; the real harness is the source of truth; thresholds are *(calibrate)* not hard-coded. |
| Embedding model upgrade invalidates vectors. | `embed_model_id` stored per record; re-embed on version change. |

---

## 13. Related work / prior art

- **MemGPT — Towards LLMs as Operating Systems** (Packer et al., 2023): tiered main/external context with self-editing memory via tool calls. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- **Generative Agents — Interactive Simulacra of Human Behavior** (Park et al., 2023): a memory stream, periodic reflection, and retrieval scored by recency × importance × relevance. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- **Zep — A Temporal Knowledge Graph Architecture for Agent Memory** (Rasmussen et al., 2025): the Graphiti engine; bi-temporal validity; superseded facts invalidated, not deleted. [arXiv:2501.13956](https://arxiv.org/abs/2501.13956)
- **Complementary Learning Systems** — fast pattern-separated hippocampal episodic memory + slow generalizing neocortical semantic memory, with replay-driven consolidation. Origin: [McClelland, McNaughton & O'Reilly 1995, *Psychological Review* 102:419–457](https://pubmed.ncbi.nlm.nih.gov/7624455/); updated: [Kumaran, Hassabis & McClelland 2016, *Trends in Cognitive Sciences* 20:512–534](https://pubmed.ncbi.nlm.nih.gov/27315762/).
- **Predictive coding / free-energy principle** (Friston): the conceptual basis for gating storage on *prediction error* — store what surprises the model. (Concept, not an implementation dependency.)

---

## 14. Open questions

1. **Harness ground truth.** Confirm the real harness's exact category list, scoring functions, and dataset format against §9.1; reconcile any deltas.
2. **Provider choice.** Which embedding + chat models for v1? Does the chosen API expose logprobs (affects estimator #2)?
3. **Threshold calibration.** Lock the *(calibrate)* values in §10 from the Stage-1 dataset.
4. **Single vs. multi-agent.** v1 is single-store; when do we need namespacing / multi-tenant?
5. **Consolidation cadence.** Time-based, volume-based, or surprise-accumulation trigger — or a hybrid?
