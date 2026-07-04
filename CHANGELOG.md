# Changelog

Notable changes. Format loosely follows [Keep a Changelog](https://keepachangelog.com);
the project is pre-1.0, so the API may still change.

## [Unreleased]

### Added
- Real local-model providers — `SentenceTransformerEmbedder` (bge/e5) and `TransformersLLM`
  (local chat model) — behind the frozen-model protocols; the deterministic fakes remain the
  default test doubles so the offline gate needs no model stack. **bge-small verified offline**.
- **OpenAI-compatible providers** (`OpenAICompatEmbedder` / `OpenAICompatLLM`, stdlib HTTP) so
  the layer (and rival systems) can share one hosted/vLLM endpoint.
- **Heuristic (no-LLM) extractor** (`HeuristicExtractor`): deterministic possessive/verb triple
  extraction, predicate canonicalization, and recency-based pronoun coreference.
- **Entity resolution** (`curated_brain.resolve.EntityResolver`): conservative subject
  canonicalization — "Erin" / "Erin Smith" / "Ms. Smith" resolve to one entity via honorific
  stripping + exact component subsumption, gated on store-provable uniqueness, with NO fuzzy
  matching (a false merge is structurally impossible).
- Record/replay **cassette** layer for reproducible real-model runs in CI.
- **Raw-text fact extraction** (`LLMExtractor`) with a groundedness anti-hallucination guard,
  wired into the write path (optional `extractor=`, off by default).
- **Schema-driven + multi-entity retrieval:** the planner recognizes any stored predicate
  (incl. multi-word), and `query()` surfaces facts for every named entity when a plan
  open-domains or mis-routes; multi-hop now cites the whole support chain.
- **Real ANN backend** (`HnswIndex`, `[scale]` extra) behind `VectorIndex` — ~20× faster top-k
  at scale; brute force stays the deterministic default. Now drops into `VectorTier` via
  `VectorTier(embedder, index=HnswIndex(...))`: `search`/`nearest` use the index's `topk`
  fast path (over-fetch + filter + hybrid re-rank), keeping the exact default byte-identical.
- **MCP server** (`curated_brain.mcp_server`, `[mcp]` extra) + `curated-brain-mcp` console
  script — mount the memory layer on any agent host.
- **LangChain Retriever** (`curated_brain.langchain.build_retriever`, `[langchain]` extra) —
  drop the curated context into any retrieval chain/agent via the standard `.invoke()` API.
- Re-embed-on-model-upgrade migration (`CuratedBrain.reembed`).
- Observability metrics (`CuratedBrain.metrics`): write-decision breakdown, discard rate, size,
  and **cost accounting** (embed/extract calls + tokens, `avg_context_tokens`).
- Durable persistence (`CuratedBrain.save` / `load`).
- **Complete PyPI package metadata** (`readme`, `license`, `authors`, `keywords`, `classifiers`,
  `project.urls`) — the README now renders as the project long-description; `twine check` passes.
- **Typed public API** — the package is now `mypy`-clean and ships a `py.typed` marker, so
  downstream type checkers see the annotations. `mypy` runs in the CI gate to keep it honest.
- Robustness / property test suite; runnable `examples/`; Apache-2.0 LICENSE, README, CI.
- **Soak/scale test** — 5000 redundant observations → ~174 episodic records + all 500 distinct
  facts (the store grows with distinct facts, not observations), with full recall after
  consolidation and a byte-identical snapshot round-trip at scale.
- Complete top-level public API in `curated_brain/__init__` (imports stay lazy).
- **Benchmarks** on our companion offline diagnostic harness (same-author — not an independent
  benchmark; see `benchmark/README.md` for the full provenance and tuning disclosures), now
  **reproducible from a clean checkout** via `benchmark/run_offline.sh`: CB leads precision +
  contradiction-resolution + staleness vs the harness's RAG references; the offline Mem0
  comparison is n=3 and mixed (answer ties on plain recall; provenance-metric caveats apply).
  Not yet the full named-rival claim — the doc states the exact endpoint/throughput needed.

### Added (Track G observability/cost, 2026-07-03)
- **Structured logging** on the `curated_brain` logger (stdlib `NullHandler` — silent unless the
  app configures logging): write-decision trace (DEBUG), consolidation summary (INFO), and the
  previously-silent HNSW degenerate-graph degradation (WARNING). Purely observational — no
  observation content/PII or secrets are ever logged, and it changes no computed state (AC-1/AC-9
  byte-identical).
- **Token→dollar pricing** (`curated_brain.Pricing`): `CuratedBrain(pricing=Pricing(...))` makes
  `metrics()["cost"]` report `estimated_usd` + `usd_per_query` from the metered hot-path tokens
  (embeddings computed + context served), so the "≤ cost" comparison is expressible in dollars,
  not just tokens/wall-time. Deterministic; honestly scoped (excludes an LLM extractor's
  completion tokens + consolidation re-embeds — documented). Default off → `metrics()` shape
  unchanged.

### Security (Track H, 2026-07-03)
- **Hardened `restore()`/`load()` against untrusted snapshots.** A malformed or hostile blob now
  fails with a clear `ValueError` instead of an opaque `KeyError`/`TypeError` (or a large
  allocation): `restore()` validates it is UTF-8 JSON of an object with an integer `counter` and
  well-formed `episodic`/`structured` records (no injected/unknown fields, all required fields
  present); `BruteForceIndex.from_dict` enforces each vector hex is exactly `dim*16` chars,
  bounding `np.frombuffer` allocation before it happens. The full restore path now fails loud:
  `StructuredTier.load`, `VectorTier.load`, `SurpriseGate.from_dict`, `EntityResolver.from_dict`,
  and the top-level `config`/`asserted_texts` fields all reject malformed input with clear
  `ValueError`s instead of opaque `KeyError`/`TypeError`/`AttributeError`. No behavior change on
  valid snapshots (byte-identical round-trip through every loader; AC-9 determinism hash unchanged).
- **Fixed a red CI type gate**: `retrieval.py` `fuse` had immutable defaults under mutable
  annotations; `mypy` (a CI step) now passes and is in the documented gate.

### Added (Phase-3b scale & tenancy, 2026-07-02)
- **Namespacing** (`curated_brain.namespace.NamespacedMemory`): hard-isolated per-tenant
  stores (one CuratedBrain per namespace) — cross-tenant bleed is structurally impossible
  (own vector index, structured tier, resolver/coreference state, echo guard per tenant);
  `drop(namespace)` erases a whole tenant in one call. Legacy single-store files load as the
  `default` namespace.
- **LLM-driven consolidation** (`CuratedBrain(summarizer=...)`): merged clusters get a real
  one-sentence model summary instead of the most-reinforced member verbatim (PRD §8
  "cluster & summarize"); default None keeps consolidation model-free and deterministic.
- **ANN filter-pushdown**: with an `HnswIndex` tier, selective metadata filters escalate the
  over-fetch until k survivors are found (or the live set is exhausted) instead of silently
  under-recalling past a fixed k×8 window; degenerate duplicate-heavy graphs degrade to
  what is reachable instead of crashing the query.
- **≥1e5 load test** (`CB_SLOW=1`): 100k records on the `[scale]` backend meet the stated
  bar — recall@10 ≥ 0.90 vs exact brute force and p95 query latency < 50 ms (config m=32,
  ef=400; measured locally 2026-07-02).

### Added (Phase-3a production essentials, 2026-07-02)
- **Hard erasure** (`CuratedBrain.forget(subject, predicate=None)`) — the GDPR path and the
  one deliberate exception to never-hard-delete: removes the subject's facts (open AND
  superseded history; object-side facts too on a full-subject forget), the episodic/vector
  records asserting them, entity-tagged vector records, echo-guard entries, and the resolver
  entry. Documented limit: free-text records that merely mention the entity with no fact link
  are not traceable and remain.
- **Inverse / set queries** (`CuratedBrain.answer_who(predicate, object, at=None)`,
  `StructuredTier.subjects_where`) — "who lives in Berlin?", "who reports to Bob?" — backed
  by a new (predicate, object) index; as-of variants supported.
- **Snapshots with the production ANN**: `snapshot()`/`save()`/`stats()` now work with an
  `HnswIndex`-backed tier (records-only payload; vectors re-derived from text on load, which
  demotes to the exact index — previously all three CRASHED with the `[scale]` backend).
- **Derived-state caching**: the planner vocabulary, relation predicates, and supersede
  filters are computed once and invalidated on mutation instead of full-store scans on every
  query.

### Changed (Phase-1 extraction-default pass, 2026-07-02)
- **One predicate vocabulary.** The heuristic extractor's verb patterns now emit the same
  canonical predicates as the possessive path / planner / dataset (`location`→`city` via
  `PREDICATE_ALIASES`), so "Erin moved to Vienna" supersedes "Erin's city is Berlin" instead
  of living in a parallel schema. (Breaking for callers who queried the old `location` key.)
- **First-person extraction** (`resolve_first_person`): "My email is X" / "I moved to Berlin"
  extract as facts about a declared speaker (`metadata={"speaker": ...}`; the MCP server
  defaults to "User"). Opt-in — without a declared speaker, first-person text asserts nothing
  (and a bare capitalized pronoun is no longer mistaken for a name).
- **Echo suppression (Pillar B).** A verbatim restatement of an already-asserted statement
  reinforces instead of re-asserting — a late echo of "Alice lives in Riga" no longer flips
  her city back after "Alice has moved to Tallinn". A genuine revert needs fresh phrasing.
- **AC-9 without spoon-feeding.** `run_harness(extraction=True)` runs Curated Brain on the
  same raw text as every baseline (no `metadata.fact`); it scores identically to the spoon-fed
  wiring (C1–C6 = 1.0/1.0/0.99/0.91/1.0/1.0) and strictly beats all three baselines on every
  category — locked in CI (`tests/test_extraction_default.py`). External harness results are
  bit-identical (same determinism hash).
- PRD §6: the logprob/predictive-surprise estimator is explicitly **deferred post-v1** (cost
  multiplier on the write path; unvalidatable without a capable endpoint; no claim depends on
  it). Planner: predicates actually stored now outrank keyword-mapped guesses.

### Fixed
- **Silent bi-temporal corruption:** a non-finite timestamp created an "open" fact invisible
  to as-of queries — now rejected with a clear error at the write boundary.
- **Restore fidelity:** the session→timestamp map driving as-of-by-session (C6) queries was not
  persisted and was rebuilt lossily on restore, silently shifting answers — now persisted.
- **(Phase-2 correctness pass)** `consolidate()` no longer crashes on fact values containing
  `"|"` (fact links stored structured; legacy snapshots still load). Bi-temporality is real:
  `metadata.fact["valid_from"]` records retroactive facts, and an out-of-order older assertion
  becomes closed history instead of inverting the open fact's interval. Non-Latin text is
  storable/retrievable (unicode tokenizer + NFKC/casefold; previously zero-vector — ASCII
  unchanged). Supersede-filtering is provenance-linked + entity-scoped per (subject, predicate)
  — superseding one entity's value no longer filters store-wide records sharing its words
  (tradeoff: a stale free-text record with neither fact link nor subject name is no longer
  caught). Snapshots persist the resolver's ambiguity history; `restore()` rejects
  dim-mismatched snapshots. MCP: lock-serialized ops, wall-clock default timestamps (was 0.0 —
  silently disabled temporal semantics), atomic batched persistence. Consolidated claims keep
  entity tags + fact link (visible to entity-filtered search and later supersedes). New
  `CuratedBrain(max_context_items=)` knob so callers asking for more context get it (default 4
  unchanged).

## [0.1.0]

- Initial two-tier memory layer: a bi-temporal structured tier (entities/relations, valid +
  transaction time, non-lossy supersede, multi-hop and as-of queries) and a vector tier, with
  surprise-gated selective writes, hybrid retrieval, and self-organizing consolidation.
- Acceptance tests AC-1…AC-9 on a seeded synthetic longitudinal dataset (beats naive-RAG,
  long-context, and no-memory baselines on every category C1–C6).
