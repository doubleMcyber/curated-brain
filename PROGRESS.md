# PROGRESS — The Curated Brain

> Cross-session state tracker. **Update this after every workstream / session** so a fresh
> session can resume with full context. Companion roadmap (detail + rationale):
> `~/.claude/plans/cosmic-watching-giraffe.md`.

Last updated: 2026-07-03 (Track D EXECUTED on BOTH LongMemEval variants — regime-split result:
oracle Letta 0.471 > CB 0.261 ~ Mem0 0.203 > Zep 0.065; `_s` CB 0.167 = Mem0 0.167, Letta
0.083(partial n=12, ties CB 1/12 on the shared questions), Zep DNF, CB 8–24× cheaper. CB ≥ Mem0
and CB ≥ Zep hold both variants; CB never posts an accuracy win over Letta (loses oracle, ties
shared `_s`) → unconditional DONE clause 1 NOT met, but CB is accuracy co-leader + cost leader
in the context-overflow regime. Red-team plan phases 0/1/2/3a/3b/5 all DONE)
· Branch: `claude/heuristic-extractor`
Published: `github.com/doubleMcyber/curated-brain` (public; `main` = the 21 build commits).
**Active work:** preliminary benchmark on the user's harness `doubleMcyber/longitudinal-memory-eval-harness`
(runs fully offline) — **RAN + IMPROVED 2026-06-19.** After 3 general-capability levers (multi-entity
routing, relational patterns, recency coreference), CB now **wins or ties every quality metric vs
`temporal_rag` EXCEPT overall recall** (0.88 vs 0.92): precision **0.79 (best)**, contradiction-resolution
**1.00 (>0.80)**, staleness **0.00 (<0.06)**, answer 0.76 (tie), multi_hop/needle/contradiction-recall all
tie at 1.00, long-range recall 0.83 (>0.67), cheapest cost. The lone recall deficit is the
`recency_relevance` category, which needs definite-NP/ellipsis coreference we deliberately did NOT add
(don't-tune rule). Harness branch `claude/curated-brain-adapter` → `RESULTS_curated_brain.md`; CB work on
`claude/heuristic-extractor`.

---

## TL;DR for a resuming session

The 7-stage PRD build is **done and green** — but it's a *hermetic architecture proof*:
it never touches a real model (deterministic hashing embedder + a "longest-line" rule LLM),
uses a brute-force index, has no persistence, and beats in-house toy baselines on a
self-authored dataset. The next phase turns that proof into a **credible open-source
contender** (vs Mem0 / Letta / Zep) via real local models, real text→triple extraction,
and an external benchmark (LongMemEval). Work it **risk-first: PROVE → PRODUCTIONIZE → SHIP.**

## Conventions (unchanged from the build)
- **Gate:** `pytest -q` exits 0 **and** `ruff check .` clean. Keep it green at all times.
- **Branch:** work on `claude/*`; **never** commit/push `main`; no force-push; stay in-repo.
- **Commits:** one conventional-message commit per passing workstream; end the message with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Verify:** before marking any workstream done, run a **separate Opus 4.8 reviewer pass**
  against that workstream's acceptance bar. Never self-certify.
- **Pattern:** keep each deterministic fake (`DeterministicEmbedder`, `RuleBasedLLM`,
  `BruteForceIndex`) as the **test double**; real implementations go behind the same protocol.
- **Stop rule:** if a workstream can't pass after a fair attempt, stop and write the blocker
  to the "Blockers" section below rather than looping.

---

## DONE — Stage 1–7 (PRD §11), all gates green

| Stage | Deliverable | AC | Commit |
|---|---|---|---|
| 1 | Adapter scaffold, deterministic fakes, seeded dataset, pyproject | AC-1 | `7930e13` |
| 2 | Bi-temporal structured tier (valid+txn time, non-lossy supersede, multi-hop) | AC-8 | `51bc138` |
| 3 | Vector tier (cosine search, metadata filters, hex-deterministic serialization) | C1 units | `2a9ba43` |
| 4 | Hybrid retrieval (planner, fusion re-rank, supersede-filter) | AC-2/3/4 | `16e7ecb` |
| 5 | Surprise-gated write path (novelty + contradiction + adaptive θ) | AC-5 | `e9912ce` |
| 6 | Fact-aware consolidation (dedup, prune, resolve, provenance) | AC-6/7 | `e92b685` |
| 7 | Full longitudinal run vs all 3 baselines, C1–C6 | AC-9 | `69b9445` |

State: 50 tests pass, ruff clean, `git status` clean, 9 source modules + 9 test files (~2.5k LOC).

## The gap — what makes this a *proof*, not yet a *contender* (verified 2026-06-18)
1. **No real model anywhere.** Hashing-trick embedder + longest-line rule LLM; zero provider code.
2. **The hard NLP is bypassed.** Triples are spoon-fed via `metadata={"fact": …}`; no extraction, entity resolution, or coreference from raw text.
3. **Predictive/logprob surprise (PRD §6 #2) not implemented** — novelty + contradiction only.
4. **No real ANN** (brute-force O(n)); **no persistence** (in-memory; no SQLite / hnswlib / faiss).
5. **Self-refereed eval** — dataset, baselines, and answer-reader all in-repo; no public benchmark, no real competitor.
6. **No release surface** — no README, LICENSE, docs, CI, examples; bare `pyproject`.

---

## DECISIONS (locked 2026-06-18)
- **Scope:** full release push — proof + product + packaging, all 9 workstreams in scope for v1.0.
- **Model stack:** local/open first — `bge`/`e5` embeddings + Llama/Qwen via Ollama/vLLM;
  hosted provider (Anthropic/OpenAI) added later as an optional extra. Greedy/temp-0 decoding.
- **Benchmark:** **LongMemEval** headline, head-to-head vs **Mem0 / Letta (MemGPT) / Zep**;
  LoCoMo optional secondary.

## ROADMAP — risk-first, three tracks (9 MECE workstreams)

Detail/rationale in `plans/cosmic-watching-giraffe.md`. Acceptance bar per workstream below.

### Track 1 — PROVE (do first; most likely to invalidate the thesis)
- [~] **A. Real local-model integration.** *(2 increments landed — both reviewer PASS 2026-06-18)*
      - [x] Real providers behind the protocols: `providers.py` — `SentenceTransformerEmbedder`
        (bge/e5) + `TransformersLLM` (cached Qwen/Mistral); lazy, soft-dependency, unit-norm,
        greedy. Fakes remain the **default** test doubles, so the offline gate needs no model stack.
      - [x] Cassette record/replay (`cassette.py`) for reproducible real-model runs in CI.
      - [x] `[local]` extra; `tests/test_providers.py` (6 offline always-run + 3 `CB_LIVE`-gated live).
      - [x] **Re-embed-on-upgrade migration** — `VectorTier.reembed` + `CuratedBrain.reembed`;
        non-lossy, deterministic (byte-identical snapshot); offline-tested.
      - [x] **Live LLM run VERIFIED** — real cached `Qwen3.5-0.8B` (CPU) extracted
        `Erin | moved | to Vienna` from raw text → also proves **Track B is feasible here**.
        Captured as `test_live_llm_extracts_a_triple` (CB_LIVE-gated, passes in 24s).
      - [x] Offline gate green (56 passed / 3 skipped), ruff clean, Opus-4.8 reviewer PASS ×2.
      - [x] **OpenAI-compatible remote providers** (`OpenAICompatEmbedder`/`OpenAICompatLLM`,
        2026-06-18, reviewer PASS): stdlib-only HTTP (no new dep) behind the same protocols, with an
        injectable `post` transport so they're offline-tested (exact wire format + L2-norm + protocol
        conformance verified). **Lets CB + Mem0/Letta/Zep share ONE endpoint+model for a fair Track-D
        run — and dodges the local-CPU bottleneck** (point everything at a hosted/vLLM endpoint).
      - [x] **Live bge embedder run — WORKS (2026-06-19; the "blocked" note was STALE).** bge-small
        weights are cached (127MB) and load offline; `tests/test_providers.py -k live` → 3 passed
        (semantics + end-to-end). **Finding:** plugging real bge into the CB harness adapter
        (`CB_EMBEDDER=bge`) FIXES the paraphrase/lexgap category (0.00→1.00) the deterministic
        double can't, and raises precision (0.79→0.84), but TRADES aggregate recall (0.88→0.84) on
        the mixed standard suite (semantic spread displaces lexical gold) — honest: a real embedder
        is not a free win on this suite. So Track A's real-embedder milestone is DONE.
      - [ ] Remaining A scope: **logprob surprise estimator** (PRD §6 #2); wire the **real LLM into
        consolidation** (replace the `RuleBasedLLM` summarizer path).
      *Bar:* non-faked end-to-end run (real embedder + real LLM) green; fakes retained as doubles.
      *Status:* LLM half live-verified; embedder half code-complete + offline-proven, live-blocked by env egress.
- [~] **B. Ingestion intelligence.** *(extractor landed — reviewer PASS 2026-06-18)*
      - [x] `extraction.py` → `LLMExtractor`: schema-constrained few-shot prompt → parse
        `subject | predicate | object`, dedup, cap. **Groundedness filter** (subject+object must
        be word-level present in the source) as an anti-hallucination guard (PRD §12).
      - [x] **Honest offline tests** (`test_extraction.py`): replay a committed cassette of
        REAL recorded model completions (`tests/fixtures/extract_cassette.json`) — replay-miss
        raises, so assertions are pinned to genuine output, not a regex. + `CB_LIVE` live test.
      - [x] **Real finding (kept visible):** the 0.8B model is a mediocre extractor — few-shot
        leaks exemplars and chit-chat hallucinates, both *killed by grounding*; predicate mapping
        is imperfect (`Dan leads Apollo` → `role|leads`). Motivates a bigger model / better prompt for D.
      - [x] **Wired into `CuratedBrain.write`** — optional `extractor=` ctor arg (default None →
        unchanged behavior + AC-1 intact); raw text with no `metadata.fact` now derives + routes
        facts. End-to-end test (raw text → structured tier → exact answer) + N>1 routing test.
        Reviewer PASS (non-breaking, multi-fact routing sound). Gate 61 passed / 4 skipped.
      - [x] **Scoped B-eval (capability A/B)** — `test_beval.py` + `beval_cassette.json`:
        extraction-ON (real recorded Qwen output) **matches the spoon-fed baseline on
        C1/C2/C5/C6** (recall, belief-update, multi-hop, as-of, and multi-hop×as-of) on a
        controlled 6-fact world. 1 honest miss surfaced (model dropped Frank's project).
        Reviewer PASS — "the plumbing is real and correct," appropriately scoped (not a vs-rivals claim).
      - [ ] Entity resolution / canonicalization + coreference; extraction confidence → gate.
      - [ ] **Full-harness B-eval** (still pending): extraction ON over the *whole* longitudinal
        stream — needs a stream-wide cassette (hours of CPU) or a faster runtime; try Qwen3.5-2B
        for better predicate mapping (the 0.8B drops `project` from "leads the Apollo project").
      *Bar:* structured tier populated from **raw text only** (no `metadata.fact`), extraction F1 ≥ target.
- [~] **D. External evaluation.** LongMemEval harness; head-to-head vs Mem0/Letta/Zep on the
      same local model; report accuracy **and** cost/latency/size; ablations; reproducible.
      *Bar:* **Curated Brain ≥ each competitor on the headline metric at ≤ its cost**, reproducible
      from a clean checkout. ← **the contender claim**
      - [x] **Reproducible-from-checkout benchmark landed (2026-06-20, reviewer PASS).** `benchmark/`
        in THIS repo: `run_offline.sh` (offline, ~1-2 min, harness pinned to commit `5e1e342`)
        reproduces CB vs the shipped references exactly (recall 0.88, precision **0.79**, contradiction
        **1.00**, staleness **0.00**, answer 0.76 — CB wins every quality axis but aggregate recall,
        where it trails 0.88 vs 0.92 = the one `recency_relevance` category, deliberately not
        special-cased). `benchmark/README.md` carries the table, the n=3 Mem0 head-to-head (CB beats
        Mem0 on answer/precision/contradiction), honest caveats, and the exact path to finish the
        full claim. Reviewer verified every number + that the loss is disclosed, not hidden.
      - [ ] **Residual blocker — now MEASURED (2026-06-20), not assumed.** Probed the actual
        environment instead of asserting "offline": network egress **is** available (`pip` reaches
        PyPI; `letta` 0.16.8 installable), Mem0 runs locally, but a credible full named-rival run is
        still out of reach, for quantified reasons:
        - **Throughput/quality bind:** Mem0 issues *many* LLM calls per add. On a fast-enough tiny
          model (`Qwen3-0.6B` + `/no_think`) one 2-turn scenario took ~350 s/add → **~11.5 h** for
          the 118-add `small` suite, AND Mem0 scored **answer/contradiction 0.00** (too weak to be a
          fair rival → an unfair strawman). A *fair* model (≥2B) is ~5 h for Mem0 **alone**.
        - **Zep — CORRECTED (was wrongly "needs Docker"):** Zep's own engine **Graphiti** runs
          in-process over an **embedded Kuzu** graph DB — no Docker. Built `adapters/zep_graphiti.py`
          (`KuzuDriver(":memory:")`, LLM→OpenAI endpoint, deterministic embedder + reranker).
        - **Letta:** pip-installable (0.16.8); heavy MemGPT framework needing an LLM backend; not wired.
        - **MPS GPU works (was wrongly "broken"):** but every local-inference avenue is now
          conclusively non-viable on this box (measured): CPU ~5–11 h; MPS SDPA crashes on the GQA
          matmul (Ministral-8B + Qwen3); bf16-on-MPS <2 tok/s; fp16+SDPA hangs on load; **fp16+eager
          is the only runnable config (~4.5 tok/s). Initial SIGSEGV-under-load was **fixed** by a
          generation lock (concurrent MPS `generate()` isn't thread-safe), enabling a full stable
          **end-to-end run** — which proved the deeper blocker: Mem0 scored **0.0 across 3 scenarios**
          despite successful calls (722 s / **3125 s** / timeouts) because Qwen3-1.7B emits markdown
          prose, not the JSON Mem0's extractor needs → stores no usable memory. A weak-model strawman
          (~52 min/scenario), **not** a fair Mem0. The clean tradeoff with no middle: the capable model
          (**Ministral-8B + eager runs and emits PERFECT JSON** — exactly what Mem0 needs) is **0.03
          tok/s** (50 tok in 25 min, ~150× too slow for one scenario); the only fast model (1.7B) is
          too weak. Large file transfers are all blocked/truncated (Ollama registry, HF-LFS GGUF,
          GitHub-raw convert script), so no capable-model download / llama.cpp workaround either.
          Forced-JSON lever (WORKED — Mem0 made functional, but throughput-capped): `outlines`
          grammar-constrained decoding (after a `tf-keras` shim) forces the fast 1.7B to emit mem0's
          exact JSON schemas (`MEM0_OUTLINES=1`, harness). On `contra-2` (same 1.7B for both): **mem0
          answer_accuracy 1.00 — no longer the 0.0 strawman**; CB 1.00 too → an answer-accuracy TIE.
          Honest limits: (a) mem0 recall/precision 0.00 is an **adapter provenance artifact** (mem0
          answered correctly → has the info; gold-turn provenance just isn't surfaced for its
          consolidated memories) — so NOT claimed as a CB win; (b) outlines FSM decoding is **~28 min
          for one 2-turn scenario** → a multi-scenario × multi-rival run is infeasible (and Zep/Letta,
          with more LLM calls, flatly so). Real breakthrough (functional same-model Mem0), but the
          throughput ceiling means clause 1's *meaningful, all-three-rivals, reproducible* bar still
          needs a hosted endpoint. The capable
          path is endpoint-bound — and the env has **no usable hosted key either** (only a
          `GEMINI_CLI_IDE_AUTH_TOKEN`, not a generative-API key; no ANTHROPIC/OPENAI/GEMINI key), so
          neither local nor hosted capable inference is reachable from here. The int8 chase also
          degraded the local model stack (torchao/transformers churn → a torch/torchvision
          `nms` op mismatch that breaks transformers *model* loading) — **CB is unaffected** (uses
          numpy + deterministic fakes; gate green, core verified `answer_structured→Vienna`); only
          the harness's already-non-viable local-loading path is hit. Not restoring it (high-risk
          churn vs a green gate, for an abandoned path). Built
          `tools/mps_openai_server.py` + `bench_endpoint_subset.py`. (CB 0.67/**1.00**
          vs temporal_rag 0.67/**0.53** ran fine — matches the full references result; only the
          named-rival LLM path is blocked.) **Conclusion proven end-to-end, not projected.**
        So the headline "≥ each of Mem0/Letta/Zep" is **endpoint-bound, not impossible**: the adapters
        (Mem0 via `MEM0_OPENAI_BASE`, Zep via `ZEP_OPENAI_BASE`) + CB (`OpenAICompatLLM`) all target one
        OpenAI-compatible endpoint. Point any **hosted** endpoint at them and the run is a one-liner per
        backend + `compare`. Three earlier "blocked" beliefs (offline / Zep-needs-Docker / MPS-broken)
        were overturned by probing. Detail: harness `RESULTS_curated_brain.md` (§Measured feasibility +
        §Update). **Local fast inference not provisionable on this box; a hosted endpoint is.**

### Track 2 — PRODUCTIONIZE (make the numbers hold under load)
- [~] **C. Storage & scale.** *(durable persistence landed — reviewer PASS 2026-06-18)*
      - [x] `CuratedBrain.save(path)`/`load(path)` — durable across process restarts; reopen test.
      - [x] **Fixed a real restore-fidelity bug** the persistence test exposed: `_session_ts`
        (the as-of-by-session map driving C6) was rebuilt only from *stored* episodes on restore,
        losing 34/64 sessions + shifting 14 → C6 answers silently diverged after restore. Now
        persisted in the snapshot (legacy fallback retained). Reviewer confirmed no other field
        has the same latent issue (`_entities` is safe — facts route to structured regardless of gate).
      - [x] **Real ANN backend (`HnswIndex`, hnswlib) — landed 2026-06-19, reviewer PASS.** Conforms
        to `VectorIndex`; `topk` is ~20× faster than brute force at n=5000 with recall@10 ≈ 0.998;
        deletion + resize correct. OPT-IN only (`[scale]` extra) — `BruteForceIndex` stays the default
        so AC-1 byte-determinism is untouched (gate 94 passed). Honest scope: swapping it into the tier
        as-is gives no speedup (tier `search` calls full `rank`); the real win needs `topk`-with-
        over-fetch wired into `VectorTier.search` (the documented follow-up).
      - [x] **P3 — `topk`-over-fetch wired into `VectorTier` — landed 2026-06-20, reviewer PASS.**
        `VectorTier(embedder, index=HnswIndex(...))` now uses the index's `topk` fast path
        (over-fetch `k*8`, then filter + hybrid re-rank) in `search`/`nearest`; `BruteForceIndex`
        (no `topk`) stays the default so AC-1/AC-9 are byte-identical (C1–C6 unchanged). Tier-level
        test: recall@10 ≥ 0.85 vs exact tier + faster at n=5000. Reviewer fixes applied: `to_dict`
        on an Hnsw-backed tier now raises a clear `TypeError` (not a raw `AttributeError`), and
        `reembed`'s BruteForce-demotion is documented. **Honest open gap:** very selective metadata
        filters can under-recall on the ANN path (filter happens after top-k truncation) — the
        documented `filter-pushdown` follow-up; the exact default is unaffected.
      - [ ] Remaining: filter-pushdown for selective filters on the ANN path; durable on-disk ANN;
        concurrency/async; namespacing; the ≥1e5 load test.
      *Bar:* load test ≥1e5 records meeting a stated recall@k + p95-latency bar.
- [~] **G. Observability, ops & cost.** *(metrics landed — reviewer PASS 2026-06-18)*
      - [x] `CuratedBrain.metrics()`: write-decision breakdown (stored/reinforced/discarded),
        `discard_rate` (Pillar-B selectivity signal), store size, structured-fact count. Cheap
        (no snapshot), deterministic, reset on restore. Tied to real behavior in `test_observability.py`.
      - [x] **Cost/token accounting** (`metrics()["cost"]`, 2026-06-18, reviewer PASS): deterministic
        write+query hot-path meter — embed calls/tokens, extract calls, queries, `context_tokens_served`,
        and the headline `avg_context_tokens` (tokens served per query) for the Track-D "≤ its cost"
        comparison. Operational (reset on restore); rejected calls cost nothing (verified). Scope:
        consolidation re-embeds + latency intentionally unmetered (documented). Reviewer hand-verified
        all counts; confirmed cost-neutral boundary rejects + 0-query avg guard.
      - [ ] Remaining: decision tracing/log; per-provider **$** pricing (token→$); wall-clock latency
        (outside the deterministic core); budgets + backpressure; graceful provider failure/degradation.
      *Bar:* dashboards/metrics emitted for a full run.
- [~] **H. Robustness & hardening.** *(suite + boundary validation landed — reviewer PASS 2026-06-18)*
      - [x] `test_robustness.py`: seeded fuzz (unicode/control/oversized/empty) — never crashes;
        deterministic under fuzz (incl. consolidate); snapshot/restore round-trips; supersede invariant.
      - [x] **Reviewer found 5 real bugs by probing past the suite; all fixed + locked:** a non-finite
        timestamp used to create an "open" fact invisible to as-of (**silent bi-temporal corruption**)
        → now rejected; malformed `metadata.fact` / non-str observation/question → clear typed errors
        instead of opaque KeyError/AttributeError. Boundary validation in `CuratedBrain.write/query`.
      - [x] **Soak/scale test (`tests/test_soak.py`) — landed 2026-06-19, reviewer PASS.** 5000
        redundant observations → store_size **174** episodic + **500** structured (all distinct facts),
        discard_rate 0.66, full recall after consolidation, byte-identical snapshot at scale — the
        curation thesis validated at volume (deterministic; reviewer forced gate→STORE and confirmed
        the bound catches the regression). Scope honest: spoon-fed facts → exercises gate+structured
        curation, not extraction.
      - [ ] Remaining: security review of blob-restore / key handling; determinism story under real
        (nondeterministic) models. **Latent minor (reviewer-noted, pre-existing, out of scope):** a
        caller-passed `gate=` SUBCLASS is dropped on `reset()` (ctor stores `gate.to_dict()`, reset
        rebuilds a vanilla `SurpriseGate`) — only affects custom-gate-subclass injection.
      *Bar:* fuzz suite green; soak holds.

### Track 3 — SHIP (adoption surface + narrative)
- [~] **E. Public API & DX.** *(API + examples landed — reviewer PASS 2026-06-18)*
      - [x] Complete top-level API (`curated_brain/__init__.py` exports core + protocols + fakes +
        real providers + extractor; imports stay lazy — no torch at import). 3 runnable `examples/`
        (basic memory, belief-update, real-models), 2 offline ones smoke-tested in CI (`test_examples.py`).
      - [x] **MCP server (`curated_brain/mcp_server.py`) — landed 2026-06-19, reviewer PASS.** Mounts
        the memory layer on any agent host: tools `write`/`query`/`answer`/`consolidate`/`stats` over
        a plain testable `MemoryService` (defaults to the heuristic extractor → raw text in; optional
        on-disk persistence). FastMCP imported lazily (`[mcp]` extra; not in the default gate); console
        script `curated-brain-mcp`. +4 tests. Gate 98 passed/4 skipped.
      - [x] **LangChain Retriever (`curated_brain/langchain.py`) — landed 2026-06-19, reviewer PASS (P6).**
        `build_retriever()` wraps a CuratedBrain as a LangChain `BaseRetriever` (standard `.invoke()`
        API; `.cb` writable); plain `memories()` core is langchain-free + tested. langchain-core lazy
        (`[langchain]` extra; not in default gate). +3 tests. Gate 105 passed/4 skipped.
      - [ ] Remaining: LlamaIndex memory adapter, typed config object.
      *Bar:* quickstart runs from a clean env in <10 min.
- [~] **F. Release engineering.** *(core surface landed — reviewer PASS 2026-06-18)*
      - [x] **LICENSE** (Apache-2.0, matches Mem0/Letta/Zep), **README** (honest — explicitly does
        NOT claim benchmark wins), **CI** (`.github/workflows/ci.yml`: ruff + pytest on py3.11/3.12).
      - [x] `pip install -e .` verified (curated-brain 0.1.0); README quickstart runs as written.
      - [x] CHANGELOG.md + CONTRIBUTING.md.
      - [x] **pip-install VERIFIED end-to-end (2026-06-19).** `python -m build` → wheel+sdist;
        installed the wheel in a CLEAN venv (no source on path) → `import curated_brain` + write/answer
        works, the `curated-brain-mcp` console script installs and its entry point resolves, and ONLY
        `numpy` is pulled in (torch/hnswlib/mcp stay optional — the lazy-import discipline holds in a
        real install). DONE clause 2 ("pip-installs") is satisfiable from a clean build; only the actual
        PyPI upload (needs the maintainer's token) is outstanding.
      - [x] **CHANGELOG completed (2026-06-19)** — now reflects bge/OpenAI-compat providers, heuristic
        extractor + coreference, schema/multi-entity retrieval, HnswIndex ANN, MCP server, cost
        metrics, and the honest benchmark scope. SHIP/F surface verified present: LICENSE (Apache-2.0),
        README (+ benchmark section), CI, CHANGELOG, CONTRIBUTING; `console_scripts` entry point added.
      - [x] **CI now tests the adoption extras (2026-06-20).** A planning audit caught that the `[dev]`-only
        gate **silently skipped** `test_mcp`/`test_langchain`/`test_ann` — three README-advertised features
        ran untested in CI (a regression in `mcp_server.py`/`langchain.py`/HnswIndex would have passed). Added
        an `integrations` job installing `[dev,mcp,langchain,scale]` that runs those suites and FAILS on any
        skip, plus the full suite. The `[dev]`-only determinism gate is unchanged (AC-1 stays model-free).
      - [x] **PyPI metadata completed (2026-06-20).** `pyproject` gained `readme`, `license`,
        `authors`, `keywords`, `classifiers`, and `[project.urls]` — previously the PKG-INFO had no
        long-description (PyPI page would render blank) and no license/author/URL metadata. README now
        embeds as the long-description; `python -m build` + `twine check dist/*` → PASSED.
      - [x] **mypy-clean + `py.typed` (2026-06-20, reviewer PASS).** Did the real typing pass: 61
        errors → 0. Root cause (the `reset()`/`restore()`-assigned core attrs) fixed with class-level
        annotations (zero runtime effect); the remaining real holes closed with behavior-preserving
        guards (`cent`/`predicate` non-None in the query path — `normalize()` would have crashed on
        None anyway; `fact_key` short-circuit in consolidate) and honest scoped `# type: ignore`s.
        `py.typed` ships in the wheel; `mypy` is now a CI gate step (scoped `ignore_missing_imports`
        per heavy-dep module, so an internal typo is still flagged). AC-9 byte-identical; reviewer
        verified every line behavior-equivalent and `py.typed` honest (no blanket ignores).
      - [ ] Remaining: **PyPI publish** (needs the maintainer's account/token — can't be done by the
        agent); docs site; coverage.
      *Bar:* `pip install` works ✅; CI green (workflow added; gate is green locally).
- [ ] **I. Positioning.** Write-up of the "memory is curation" thesis + benchmark table, comparison
      matrix vs rivals, demo, launch post. *Bar:* publishable report with reproducible numbers.

---

## CURRENT POSITION (after 2026-06-18 session — 7 reviewed commits)
- **DONE-criteria status: 2 of 3 met.** ✅ `pip install` works · ✅ docs (README) + CI + LICENSE
  + **published public on GitHub** (`doubleMcyber/curated-brain`) · ✅ gate green (85 passed/4
  skipped) on a clean tree. ❌ **LongMemEval ≥ Mem0/Letta/Zep** — Track D, environment-blocked
  (see BLOCKERS). Since (all reviewer-verified, toward D): open-domain backstop + schema-driven planner
  (improvement-plan #1) + cost accounting ("≤ cost" clause) + **OpenAI-compat providers** (fair shared-endpoint run).
- Done this session: **A** (real local providers + cassette + re-embed; live LLM verified),
  **B** (extraction implemented + wired into write path; spoon-feeding crutch removed),
  scoped **B-eval** (extraction-ON matches spoon-fed on C1/C2/C5/C6), **F core** (LICENSE/README/CI/pip).
- **Next when unblocked (capable env — GPU + egress + competitor systems):** Track **D** first
  (the only remaining DONE clause). Then full-harness B-eval, remaining Track A (logprob; real-LLM
  consolidation), Tracks C/G/H, and the rest of E/F/I.
- **Next concrete step → Track B (extraction), designed this session:**
  1. `extraction.py` → `LLMExtractor(llm)`: prompt the LLM for `subject | predicate | object`
     lines, parse → `[{subject,predicate,object}]`. This is the *real, general* path (proven
     feasible: Qwen on CPU already emits clean triples).
  2. Wire into `CuratedBrain.write`: when `metadata.fact` is absent and an extractor is
     configured, derive facts from raw text via the extractor (so the structured tier no
     longer depends on the dataset's spoon-fed `metadata.fact`).
  ✅ Steps 1 & 3 DONE this session (`LLMExtractor` + honest cassette test). **Next:**
  2. **Wire the extractor into `CuratedBrain.write`** (the core-write-path change): when
     `metadata.fact` is absent and an extractor is configured, derive facts from raw text.
     Add an optional `extractor=` ctor arg (default None → unchanged behavior, all 59 stay green).
     Then run the *whole longitudinal harness* with extraction ON (no `metadata.fact`) and see
     how the C-category scores hold up vs the spoon-fed baseline — the real test of the thesis.
  4. Entity resolution/canonicalization for `subject`/`object`; coreference (stretch).
  - Then **B-eval** → unblocks **D (LongMemEval)**. Consider Qwen3.5-2B for better extraction.
  - Remaining Track A (can interleave): logprob surprise estimator; real-LLM consolidation summarizer.
  - Env reminder: LLM runs need `device="cpu"` (MPS bug) + `HF_HUB_OFFLINE=1` + a cached model;
    record cassettes for any CI-facing real-model behavior (HF weight egress is blocked here).

## IMPROVEMENT PLAN — planning pass (2026-06-18, Opus reviewer-architect)

**Critical insight: the path to winning Track D starts with LOCAL, offline code fixes, not a
GPU.** Two components are silently coupled to the synthetic dataset and would make Curated
Brain *lose* to the rivals on open-domain LongMemEval even on a capable box:

1. **`retrieval.py` `Planner.plan` is dataset-coupled (#1 losing condition).** It recognizes
   only 5 hardcoded predicates (email/city/role/project/manager) and hardwires multi-hop to
   `manager`. On open-domain questions `plan.predicate` is `None` → `open_ended` → the
   structured tier (CB's whole advantage) is **never consulted** → CB degrades to NaiveRAG and
   loses. **Fix first, offline:** schema-driven planner (derive predicates from `self.structured`,
   or LLM-route the question) + a fail-soft backstop (`query` tries a structured `resolve` for any
   recognized entity even when predicate is None).
   **[x] Fail-soft backstop DONE (2026-06-18, reviewer PASS).** `query()` now surfaces a known
   entity's high-precision structured facts when `open_ended` (capped, reserving a vector slot);
   `StructuredTier.predicates_for`. Reviewer instrumented the harness: backstop fires **0×** on the
   54 synthetic probes (all carry a recognized keyword) → AC-9 unchanged (C1–C6 1.0/1.0/.987/.906/1.0/1.0).
   **[x] Schema-driven planner DONE (2026-06-18, reviewer PASS).** `Planner.plan` now also recognizes
   any predicate ACTUALLY STORED whose (single-word) name appears in the question (`query()` derives the
   vocab from `self.structured.facts`), so un-hardcoded predicates ("hobby") route *precisely* instead of
   falling to the backstop. Reviewer diffed every harness `QueryPlan` with/without the vocab → **0
   differences** (the 5 dataset predicates are all already keyword-matched), AC-9 unchanged.
   **Still open:** LLM-routing for multi-word/paraphrased predicates + coreference (single-token exact
   match only today); auto-detect relation predicates for arbitrary multi-hop (only `manager` hardwired);
   and the open-domain reader in `eval.py` (#2 below).
2. **`eval.py` `candidates_for`/`extract_value` is a closed-set reader over `ds.people`** — it
   *cannot* score an open-domain benchmark and using it on LongMemEval is a methodological error.
   Track D must feed `Retrieval.context` to the **same shared judge LLM** used for every system
   and score with LongMemEval's official metric. Keep the closed reader only for in-repo tests.

**Accuracy lever (do 2nd, offline):** entity resolution + coreference. `normalize` is just
`strip().lower()`, so `Erin`/`she`/`Ms. Smith` are distinct subjects → structured lookups miss.
Worse, `extraction._supported` grounding *fights* coreference ("she moved to Vienna" → no fact).
Add a per-session alias map + canonicalize subject/object against `self._entities`. Then move
extraction to open-schema + constrained/JSON decoding + confidence (PRD "extraction confidence → gate").

**Other ranked items:** real ANN behind `VectorIndex` with **over-fetch-then-filter** (a naive
top-k ANN under-recalls under the entity/as-of filters — correctness, not just speed);
entity-scope `_stale_objs`/`fuse` (a value stale for A but in B's context can be wrongly dropped —
the `- open_vals` subtraction only partially mitigates); LLM-driven consolidation (`_merge_to_claim`
seam exists, unused); concurrency/namespacing; SQLite-index the structured tier (linear scans today).

**Highest-ROI quick wins:** an **OpenAI-compatible provider** in `providers.py` so CB + rivals
share ONE model endpoint (fairest Track D, dodges local-CPU); **cost/token accounting** in
`metrics()` (needed for the "≤ its cost" clause); mypy/pyright + coverage in CI.

**Reprioritized order:** (0) de-couple planner+reader [local] *(backstop + schema-driven planner landed;
LLM-routing/relation-autodetect + open-domain reader still open)* → (1) entity resolution + open-schema extraction [local] →
(2) ✅ OpenAI-compat provider + cost metrics **(both landed 2026-06-18)** → (3) Track D harness on a capable
box → (4) ANN + structured indexing + stale-scope → (5) LLM consolidation, concurrency, namespacing.

## ENVIRONMENT NOTES (for resuming sessions — verified 2026-06-18)
- Python 3.12.7; `torch` 2.5.1 with **MPS** (Apple GPU); `transformers`, `huggingface_hub`,
  `httpx`, `requests`, `sentence-transformers` (installed this session) present. `faiss`/
  `hnswlib`/`sqlite_vec` and **Ollama** are **absent** (use `transformers` directly, not Ollama).
- **Network quirk:** `huggingface.co` reachable from Python (model_info works) but **weight
  downloads are very slow**; `curl` to HF times out while `pip`/pypi works. Prefer **already-cached**
  models. `curl` is not a reliable HF reachability probe — use `huggingface_hub` from Python.
- **Cached models with weights** (`~/.cache/huggingface/hub`): `Ministral-8B-Instruct-2410` (15G),
  `Qwen3.5-2B` (4.2G), `Qwen3.5-0.8B` (1.6G), `Qwen3-0.6B-Base` (1.2G), `EXAONE-4.0-1.2B` (2.4G).
  `bge-small-en-v1.5` config cached; weights were downloading (slow). For a live LLM run, point
  `TransformersLLM(model_name=...)` at a cached instruct model to avoid a fresh download.
- **Run live model tests:** `CB_LIVE=1 pytest tests/test_providers.py -k live` (default gate skips them).

## BLOCKERS
- **Named-rival run — FIRST RESULT 2026-06-19 (offline, preliminary).** Built an offline Mem0
  adapter (mem0 driven by cached `Qwen3.5-2B` behind its `LLMBase` + the SAME deterministic
  embedder as CB + in-memory qdrant) and ran a 3-query subset head-to-head. **CB beats Mem0**:
  answer 1.00 vs 0.67, precision 1.00 vs 0.37, contradiction 1.00 vs 0.00; ties recall (1.00);
  ~0s vs **~70 min** ingest. **Fairness adversarially reviewed** (caught + fixed a `top_k` vs
  `limit` defect before quoting precision/contradiction).
  **HONEST CORRECTION (broader partial run):** that n=3 was the smallest-scenario-per-category
  subset = CB-FAVORABLE. A broader run (long-0/1, lexgap-0) — killed because the full suite is
  infeasible offline (~2h/large scenario, ~15-20h) — shows: on PLAIN longitudinal recall all three
  TIE on answer (1.00); CB still **dominates precision** (~1.00 vs Mem0 ~0.13) and contradiction;
  **paraphrase (lexgap) fails for ALL three** (shared offline-embedder limit, not CB-specific). So
  the honest claim is "CB beats Mem0 on precision + contradiction, ties on plain-recall answer," NOT
  a sweep. **Caveats:** tiny n; Mem0 on a SMALL local model via an OpenAI-shaped shim with frequent
  JSON-parse errors (NOT its cloud best); different extractors. A CREDIBLE FULL run needs a capable
  shared endpoint + a real semantic embedder (mem0 is ~2.7 min/add on CPU → infeasible here). Harness branch
  `claude/curated-brain-adapter`: `mem_eval/adapters/mem0_local.py`, `bench_mem0_h2h.py`,
  `RESULTS_curated_brain.md`.
- **Mem0 install + offline path (TESTED 2026-06-19).** `pip install mem0ai` SUCCEEDS
  here (mem0 2.0.7; CB gate still green after the dep bump to numpy 2.2.6). Inspected mem0's factories:
  an **offline path exists** — LLM via `langchain` provider + a HuggingFacePipeline over the cached
  Qwen (ollama/lmstudio/vllm providers all need an absent server; the rest are cloud), embedder via
  `langchain`/custom (so CB + mem0 can SHARE one deterministic embedder) or `fastembed`, vector store
  via in-memory `chroma`. **BUT the only usable local model is the weak 0.8B Qwen on CPU.** Running
  mem0 on that would (a) be a slow multi-hour CPU run and (b) produce a **misleading** "CB beats mem0"
  — mem0 hamstrung by a tiny model, not an architecture result. A CREDIBLE/fair run needs a **capable
  shared model** (an OpenAI-compatible endpoint+key, which CB's `OpenAICompatLLM` already supports, or a
  GPU box). Crippling the rival to claim a win is exactly what the goal's integrity forbids — so this is
  the genuine unblock: a capable shared model, then the run is fast, fair, and credible.
- **Track D (LongMemEval head-to-head vs Mem0/Letta/Zep) — the one remaining DONE clause — is
  environment-blocked (substantiated 2026-06-18), not a code problem.** Three independent reasons:
  1. **Compute.** This is a laptop with a broken-for-these-models MPS, so LLM inference runs on
     CPU (~15–25s/call). LongMemEval is ~500 questions over long multi-session histories; running
     4 systems (Curated Brain + 3 rivals) over it = many hours–days. Infeasible interactively.
  2. **Competitor infra.** Mem0/Letta/Zep each need their own backend (Letta = a server, Zep =
     Graphiti + a graph DB) and default to a cloud LLM API key that this environment does not have.
     A fair run requires wiring all three to the same local model — substantial per-system setup.
  3. **Data egress.** The dataset exists (`xiaowu0162/longmemeval` on HF) but its history files are
     large/LFS, and this env's HF **LFS egress is blocked** (same reason bge weights wouldn't download).
- **How to do Track D on a capable machine (GPU + network + ability to run the rivals):**
  1. `pip install` the LongMemEval harness + dataset; pick the headline model (a strong instruct
     model — NOT the 0.8B; it drops facts. Use the hosted/large local model the rivals also use).
  2. Implement a thin LongMemEval adapter over `MemoryBackend` (write/query/consolidate already fit).
  3. Stand up Mem0, Letta, Zep, each on the **same** model + token budget (the memory layer must be
     the only variable). Record accuracy AND cost/latency/store-size.
  4. Gate: Curated Brain ≥ each rival on the headline metric at ≤ its cost, reproducible from a clean
     checkout. That closes the last DONE clause.
- Non-blocking note: live *bge embedder* run also needs HF LFS egress (cached LLMs work on CPU; the
  live LLM path + extraction are already verified). Use a cached/hosted embedder where egress works.

## RED-TEAM + IMPROVEMENT PLAN v2 (2026-07-02, user-requested; two adversarial subagent passes)

Two independent red-team reviews (core implementation; eval credibility) produced a severity-ranked
finding set and a 6-phase plan. **The controlling insight: the project benchmarks the configuration
that can't work on real data (regex extractor) and ships unbenchmarked the configuration that could
(LLM extractor).** Top findings, in brief:

- **Eval credibility (most damaging):** the harness is same-author, same-week; "independent /
  third-party / never saw during development" in the public READMEs was false; PROGRESS documents a
  lose→lever→win cycle (selection pressure even if each lever is general); provenance-based
  recall/precision structurally penalize consolidators (Mem0) — internally admitted but the public
  n=3 table printed the artifact wins anyway; suite is ~25 queries, no CIs (0.04 recall gap = 1
  query); in-repo AC-9 feeds gold triples to CB only, scored by a closed-set oracle reader.
- **Implementation (top cut):** token-subset stale filtering false-positives store-wide on common
  words ("manager", "Nice"); `valid_from`≡`created_at` (retroactive facts unreachable; out-of-order
  writes create inverted intervals); `|` in a fact value crashes all future `consolidate()`;
  tokenizer `[a-z0-9]+` makes non-ASCII text unstorable/unretrievable (zero vector); consolidation
  destroys merged members' content, never calls an LLM, re-adds claims with `entities=[]` (degrades
  entity-scoped recall); no namespacing (cross-session/user bleed incl. via MCP), no delete/GDPR, no
  locks; MCP defaults timestamp=0.0 (kills temporal semantics) + full-snapshot write per observation;
  HnswIndex crashes `snapshot()/save()/stats()`; `k` ignored (hard cap `MAX_CONTEXT_ITEMS=4`);
  extractor predicates (`location`/`employer`) don't match planner/dataset vocab (`city`/`role`) so
  extracted updates don't supersede; surprise gate = lexical novelty (paraphrased updates fall in a
  discard dead-zone; logprob estimator never built); restore doesn't reconstruct resolver ambiguity
  history (determinism violation across restart).

**Plan (sequenced):** Phase 0 credibility triage of public docs (✅ DONE 2026-07-02 — see changelog);
Phase 1 make LLM extraction the benchmarked default (unify predicate schemas, first-person
extraction, re-run suite with `metadata.fact` withheld, implement-or-retire logprob surprise);
Phase 2 correctness fixes (provenance-linked stale filter, tuple fact keys, decouple valid_from,
unicode tokens, resolver-state in snapshot, MCP lock/timestamps, honor `k`); Phase 3 product
essentials (namespacing, delete/retract, inverse queries, persistent ANN + snapshot, real LLM
consolidation); Phase 4 credible external eval (LongMemEval frozen-config vs rivals at recommended
configs on a hosted endpoint — still endpoint-bound); Phase 5 reposition around what CB uniquely
owns (deterministic, auditable, offline, bi-temporal + provenance).

## FORWARD ROADMAP — credible non-gaming path to clause 1 + production punch-list (planning pass, 2026-07-03)

Two Opus planning/audit subagents produced this. **The firewall governs everything:** every
capability change is accepted/rejected on TWO LongMemEval-BLIND gates *before* any LongMemEval
number is looked at — **Gate A** (diagnostic harness `run_offline.sh`: determinism hash
673a25c7… held where the change should be a no-op; precision 0.79 / contradiction 1.00 /
staleness 0.00 must NOT regress) and **Gate B** (unit test of the mechanism + `pytest -q` +
`ruff` + `mypy` green). Only then is a FROZEN LongMemEval re-run permitted, config locked
*before* the run. One change at a time, each with its own frozen re-run — no compound tuning.

**The honest reframe (do first, free):** CB ≥ Letta on the ORACLE variant is NOT achievable for
a curation layer — oracle histories fit the 32k context, so Letta wins by full-context reading,
not memory; a k=10 curation layer structurally discards what Letta keeps. Reframe DONE clause 1
to the regime it already meets: **CB ≥ Mem0 and CB ≥ Zep on both variants (tie-or-better, always
cheaper); CB ≥ Letta in the context-overflow (`_s`) regime a memory layer exists for.** This
matches the measured data and removes the unwinnable oracle target that caused two reverts.

Prioritized by (credibility)/(effort):
1. **Soft entity scoping** — **TRIED 2026-07-03, REJECTED by Gate A (precision 0.79→0.573).**
   Implemented exactly as designed (`VectorTier.search` `entity_soft`: entity matches fill
   first, unscoped semantic fill only remaining slots when scoping under-fills; default-off →
   Gate A part 1 byte-identical 673a25c7 confirmed). But the "no-op when scoping fills k"
   assumption is FALSE at realistic k: on the diagnostic suite each entity has only ~5-7 facts,
   so at k=10 scoping under-fills for MOST queries → the fill fires constantly → precision
   collapses 0.79→0.573. The firewall worked: rejected on blind data before any LongMemEval
   look; reverted (three attempts now, all fail their held-out gate). **The design direction is
   still right but "fill all remaining slots" is too aggressive.** Refined next attempts to try
   (each gated the same way): (a) fill ONLY when entity-scoped returns ZERO (not merely <k);
   (b) relevance-threshold the fillers (only add unscoped hits above a similarity floor);
   (c) the more fundamental fix — tag raw turns with mentioned entities AT INGEST so entity
   scoping actually includes them (needs the adapter to extract session entities before writing
   turns, resolving the ordering problem). (c) is the most promising + most work.
2. **GPT-4o judge** in `bench_longmemeval.py` (route only the JUDGE call to GPT-4o, systems stay
   on the shared local model) — re-score existing frozen outputs, no re-run; cheapest
   leaderboard-credibility win.
3. **Temporal-reasoning path** (`Planner.plan` new intent + wire `VectorTier.search`'s unused
   `window` arg into `query()`; surface DATED lines, let the answer model do the arithmetic):
   0.043 vs Letta 0.435. Gate A: QueryPlan diff = 0 on diagnostic probes.
4. **Hosted endpoint + n≈140 `_s` run to completion** (adapters already wired: CB
   OpenAICompatEmbedder, MEM0/ZEP_OPENAI_BASE, Letta handles): removes the throughput cap that
   caused Zep-DNF + Letta-partial; report CIs + McNemar. **Provisioning task, not engineering —
   the project's long-standing endpoint block.**
5. **Preference path** (`extraction.py` preference predicate family + aggregation in `query()`):
   0.087 vs 0.217; reuses structured tier + supersede. Gate A hash-identical.
6. *(optional)* Adaptive "history-fits → read it all" mode (adapter, triggered on
   token-budget-fit, NOT question type) — only if a reviewer insists oracle must not look like a
   loss; won't make CB *beat* Letta on oracle, only approach parity.

**Production punch-list (audit pass):** P0 — [x] mypy gate fixed (was red: retrieval.py fuse
mutable-default type mismatch) 2026-07-03; [x] mypy added to documented gate (README/CONTRIBUTING).
P1 — cut CHANGELOG `[Unreleased]`→dated version at publish; ship CHANGELOG in sdist; harden
`restore()` against untrusted blobs (schema-validate keys + bound vector sizes — currently splats
`EpisodicRecord(**d)`); structured logging on silent degradation paths (HNSW retry); document
end-user `pip install curated-brain` (only editable shown); RLock on CuratedBrain/NamespacedMemory
or document single-writer contract at top level. P2 — HNSW-survives-restore (currently demotes to
brute force); nightly CB_SLOW/CB_LIVE CI lane. **Genuinely done:** packaging metadata + twine, import
hygiene, deterministic snapshot/restore, fuzz+soak, no unsafe deserialization, boundary validation,
cost metrics, the 1e5 load bar, an honest README. PyPI upload = maintainer-token only.

## CHANGELOG OF THIS FILE
- 2026-07-03 (later¹⁰) — **Per-category ceiling proof: oracle gap is arithmetically bounded, and
  CB beats Letta on its OWN thesis.** Computed CB-vs-Letta oracle by category (n=23 each): CB
  **wins knowledge-update 0.478 vs 0.435** (belief revision — CB's bi-temporal thesis working);
  loses single-session-assistant 0.261 vs **0.957** (the answer IS the assistant's verbatim
  output — full-context reading, curation can't reproduce a table), temporal 0.043 vs 0.435
  (genuinely fixable — date arithmetic over stored timestamps), preference 0.087 vs 0.217,
  multi-session/user close. **Theoretical ceiling** (match Letta on laggards + keep CB's win):
  0.478 — barely above Letta's 0.471. **Realistic ceiling** (single-session-assistant capped ~0.6
  since curation can't full-context-read it): **~0.42 < Letta 0.471** — so even a perfect campaign
  on fixable categories does NOT yield CB ≥ Letta on oracle; single-session-assistant dominates
  and is a full-context-reading category. **The one worth-doing lever = temporal-reasoning**
  (0.043→~0.4 achievable, +~0.06 overall, firewall-compliant library work, does NOT beat Letta
  but is a genuine capability). This converts "empirically blocked" → "arithmetically bounded":
  clause 1's Letta component is not a tuning target. Documented in harness `RESULTS_longmemeval.md`
  §"arithmetically bounded". No code changed; gate green, repos clean. **Next real capability item
  (future session, not clause-1-satisfying): temporal-reasoning path** (planner date-intent + wire
  the unused `VectorTier.search` window arg; test_temporal.py; QueryPlan-diff=0 on diagnostic).
- 2026-07-03 (later⁹) — **Probed the LAST unblock (hosted inference) — conclusively unavailable in
  this environment.** The credible clause-1 close needs a fast hosted model (rivals at `_s`, n≥100).
  Probed every credential/endpoint present, not assumed: `GEMINI_CLI_IDE_AUTH_TOKEN` → 400/401 on
  the generativelanguage API (it's an IDE-server token, not a generative key); `gemini` CLI is
  installed + authenticated but **tier-ineligible for generation** ("no longer supported for Gemini
  Code Assist for individuals → migrate to Antigravity"); no gcloud/ADC; no OpenAI/Anthropic keys;
  only local Ollama (the throughput that made the rivals infeasible). **So the hosted-endpoint
  unblock is not discoverable in-session — it requires the USER to provide an API key / endpoint.**
  With that, clause 1 is EXHAUSTIVELY blocked: every avenue probed + closed (local retrieval fixes
  fail held-out gates; extraction can't close the structural oracle gap; local `_s`-completion is a
  within-noise tie needing ~5 infeasible days; hosted inference unreachable). Terminal state; the
  blocked report is complete. What unblocks it, precisely: set `OPENAI_BASE_URL`+`OPENAI_API_KEY`
  (or any OpenAI-compatible endpoint) — CB `OpenAICompatLLM`/`OpenAICompatEmbedder`,
  `MEM0_OPENAI_BASE`, `ZEP_OPENAI_BASE`, Letta handles all already target it — then
  `bench_longmemeval.py --data data/longmemeval_s --n 140` per backend + compare. No code changed.
- 2026-07-03 (later⁸) — **Ruled out the last "just run the benchmark" avenue with statistics.**
  The `_s` variant IS the headline LongMemEval benchmark (oracle is a diagnostic), and it's where
  CB co-leads — so the natural question is "complete the cut Letta/Zep `_s` runs and claim CB ≥
  each." Examined Letta's `_s` rows: on the 12 questions it completed, **Letta 1/12 and CB 1/12 —
  the SAME single question (q4)**; CB's 0.167 comes only from 3 questions in the other 12 Letta
  never ran. Completing Letta = fresh ~28 h run (no resume); best case ~CB 4/24 vs Letta 2/24 is
  **within noise at n=24 (±0.15)** on a subset where they exactly tied → not a credible win, and
  presenting it as one would be the overclaiming this project was red-teamed for. A credible
  separation needs n≈100+ = ~5 days of Letta alone (Zep infeasible even at n=1) → requires a fast
  hosted endpoint, not local hardware. **Terminal finding (now statistically closed): CB posts no
  credible accuracy win over Letta on either variant** (clear oracle loss; within-noise `_s` tie),
  beats/ties Mem0+Zep, leads all on cost. Clause 1 is a provisioning + offline-capability task, not
  an in-session tweak. Documented in harness `RESULTS_longmemeval.md` §"Why just complete the runs
  would not credibly change this". No code changed; repos clean, gate green.
- 2026-07-03 (later⁷) — **Track G cost: token→$ pricing (reviewer PASS).** `Pricing` dataclass +
  `CuratedBrain(pricing=)` → `metrics()["cost"]` reports `estimated_usd`/`usd_per_query` from the
  metered hot-path tokens (embeddings + served context); makes the DONE-clause "≤ its cost"
  expressible in dollars, not just tokens/wall-time. Deterministic, honestly scoped (excludes
  extractor completion tokens + consolidation re-embeds — documented), default-off so metrics()
  shape is unchanged; AC-9 hash 673a25c7 unchanged. +5 tests incl. one documenting the honest
  semantic that a gate-REJECTED write still costs its embedding (cost = work performed, not
  storage outcome). 182 passed, ruff+mypy clean. Reviewer PASS. Advances Track G; does NOT
  address clause 1 (Letta) — still blocked as documented. **Session-terminal note:** clause 1's
  Letta sub-clause is structurally unmeetable in-session (oracle rewards full-context reading;
  4 gated retrieval fixes all failed held-out precision; the real close needs a hosted-endpoint
  `_s`-at-scale rerun or offline extraction work — both out-of-session). Clauses 2&3 met. This
  session hardened production-readiness (mypy gate fix, complete untrusted-restore hardening,
  $ cost accounting) while the core clause stays honestly blocked.
- 2026-07-03 (later⁶) — **Track H security COMPLETE: entire untrusted-restore path now fails loud
  (reviewer PASS).** Closed the three loader surfaces the prior reviewer flagged + two more it
  found: `VectorTier.load` (VectorRecord field validation + int-key/list-meta checks),
  `SurpriseGate.from_dict` (all-8-keys + numbers-not-bool), `EntityResolver.from_dict` (dict +
  list-of-pairs), plus top-level `config` (dict) and `asserted_texts` (list) in `_validate_snapshot`.
  Every restore-path loader now raises a clear ValueError on malformed/hostile input instead of an
  opaque KeyError/TypeError/AttributeError. +6 tests (177 passed), ruff+mypy clean. **No-op on
  valid snapshots**: byte-identical round-trip through every loader (verified with multi-token
  entities + poisoned resolver tokens) + AC-9 hash 673a25c7 unchanged. Reviewer PASS (probed each
  loader's malformed-input classes; confirmed int(seen/stored) + bool-rejection are harmless on
  valid input; happy path untouched). Track H (fuzz + soak + security) is now substantively done:
  seeded fuzz + 5k soak (earlier) + untrusted-blob restore hardening (now). Does NOT address
  clause 1 (Letta) — that remains blocked as documented; this advances the PRODUCTIONIZE/H track.
- 2026-07-03 (later⁵) — **Track H security: hardened `restore()` against untrusted snapshots
  (reviewer PASS).** Pivoted off the (closed) Letta-gap fix loop to a real, non-benchmark
  production gap the audit flagged: `restore()` splatted untrusted JSON into `EpisodicRecord(**d)`
  / `Fact(**d)` and hex into `np.frombuffer` with no validation (opaque-crash + unbounded-alloc
  surface). Added: UTF-8/JSON/object validation + `_validate_snapshot` (counter int; episodic
  records reject injected/unknown fields, require all required fields); `BruteForceIndex.from_dict`
  enforces vector hex == dim*16 chars (bounds allocation BEFORE frombuffer — verified via
  tracemalloc by the reviewer: ~0 MB on a 40M-char hostile hex); same fact-field validation in
  `StructuredTier.load`. +9 tests (173 passed), ruff+mypy clean. **No-op on valid snapshots**:
  byte-identical round-trip + AC-9 diagnostic hash 673a25c7 unchanged. Reviewer PASS (probed every
  malformed-input class, confirmed the alloc bound fires pre-frombuffer, verified happy path
  untouched). Documented follow-ups: same fail-loud for VectorTier.load/SurpriseGate.from_dict/
  EntityResolver.from_dict (lower priority). Advances the "credible/production-ready" goal on the
  H track; does NOT address clause 1 (Letta), which remains blocked as documented.
- 2026-07-03 (later⁴) — **Fourth Letta-gap attempt (ingest-time entity-mention tagging), Gate A
  REJECTED — and it converges to a FUNDAMENTAL conclusion.** The genuinely-different approach
  from the roadmap #1(c): tag each stored record with the resolver entities its text MENTIONS
  (precise by construction — only surfaces records naming the entity), a library change validatable
  on the diagnostic suite. Gate B green (164, mypy clean). Gate A: precision **0.79→0.76** — and
  it produced the EXACT same hash + precision (a9c72df2 / 0.76) as the "untagged eligible" revert,
  proving that on realistic data, scoping a record to everyone it mentions is behaviorally
  equivalent to admitting untagged content: **broader entity recall inescapably trades diagnostic
  precision.** Reverted; gate + mypy green, AC-9 hash restored. **CONCLUSION (4 distinct
  implementations, all held-out-rejected):** the single-session-assistant recall gain and tight
  diagnostic precision are in genuine tension — there is no clean retrieval tweak that gets the
  recall without the precision cost. Combined with the oracle gap being STRUCTURAL (Letta wins by
  full-context reading when history fits), the honest verdict is firm: **the unconditional
  "CB ≥ Letta" clause is not achievable through in-session retrieval changes.** Real closes need a
  different axis (better extraction so assistant facts become STRUCTURED, not episodic; or a
  hosted-model rerun where the regime shifts) — deliberate offline work, not benchmark patching.
  Per the global contract's 3-strikes rule (now 4), the fix loop is CLOSED; the blocked report is
  this + the forward roadmap above.
- 2026-07-03 (later³) — **Executed the planning roadmap's #1 item (soft entity scoping) under the
  firewall; Gate A REJECTED it — third held-out-validated rejection.** Implemented the
  precision-preserving design exactly (entity_soft two-pass fill, unit-tested: no-op when scoping
  fills k, fills only when under-k; 165 passed). Gate A part 1 (default-off) byte-identical
  (673a25c7 ✓). Gate A part 2 (soft ON, diagnostic suite): precision **0.79→0.573** — the fill
  fires on nearly every query because at k=10 each entity has too few tagged records, flooding
  slots with irrelevant unscoped hits. Rejected on LongMemEval-BLIND data *before* any benchmark
  run (the firewall working as intended — proceeding to LongMemEval with a precision-regressing
  change would be the gaming this guards against). Reverted; gate green + mypy clean + AC-9 hash
  restored. Net: three fixes tried, all fail their held-out precision gate → strong evidence the
  Letta-gap close is a real recall/precision tradeoff needing a smarter design (see roadmap #1
  a/b/c — ingest-time entity tagging of raw turns is the promising path). No benchmark chased.
- 2026-07-03 (later²) — **Ran the goal's planning + production-audit subagents (both Opus); fixed
  a real gate blocker.** The production audit caught that **CI's mypy step was RED** (2 errors in
  `retrieval.py` `fuse`: `stale_rids: set[str] = frozenset()` / `stale_pairs: list[...] = ()` —
  immutable defaults under mutable annotations, introduced in the Phase-2 stale-filter work). Local
  pytest+ruff were green but CI's Type-check would have failed → clause 3 ("gate green") was only
  half-true. Fixed by widening to the read-only abstract types the function needs (`AbstractSet`/
  `Sequence`, keeping the safe immutable defaults); mypy now clean, gate green (164 passed, ruff
  clean). Added mypy to the documented gate so it can't hide again. Recorded the planning agent's
  credible non-gaming roadmap (above) + the production punch-list. No benchmark chasing this step.
- 2026-07-03 (later) — **Second attempt at the Letta gap, reverted: the entity-filter change is a
  TRADEOFF, not a free fix.** Implemented the properly-scoped LIBRARY fix + unit test:
  `VectorTier.search` no longer excludes records with NO entity tags under an entity filter (so
  raw conversation turns stay retrievable when the planner resolves an entity — architecturally
  correct in isolation; gate went 164→165 passed). BUT it **broke the AC-9 byte-identical
  invariant**: diagnostic-harness determinism hash 673a25c7→a9c72df2 and precision 0.79→0.76 —
  the synthetic suite *does* contain untagged raw turns, so surfacing them is a real
  recall↑/precision↓ tradeoff, and that held-out precision regression is the tell. Reverted;
  hash + gate + tree confirmed restored (673a25c7…, 164 passed, clean). **Lesson for next
  session:** the entity-filter behavior IS the right lever for single-session-assistant, but it
  needs a precision-preserving design (e.g. tag raw turns with mentioned entities at ingest so
  scoping still works, OR soft-boost instead of hard-exclude) evaluated OFFLINE on held-out
  data — not an in-session flip validated by whether it beats Letta. Two documented reverts now
  confirm: closing clause 1 needs deliberate capability work, not benchmark patching.
- 2026-07-03 — **DONE-clause status closed out honestly; declined in-session benchmark tuning.**
  Evidenced clauses 2 & 3 fresh: `pip install .` into a clean venv imports + write/answer works
  (v0.1.0), LICENSE/README/CHANGELOG/CONTRIBUTING/CI/pyproject all present, gate green (164
  passed, ruff clean) on a clean tree. Clause 1 (CB ≥ each rival) stands as **regime-split, not
  unconditionally met** (CB ≥ Mem0/Zep hold; CB posts no accuracy win over Letta). Diagnosed the
  largest oracle gap — single-session-assistant 0.261 vs Letta 0.957 — to a real mechanism: raw
  conversation turns are stored with NO entity tags, so the query's entity-filtered vector search
  excludes them and the structured backstop fills context with "User's X is Y" fact-echoes; the
  answer-bearing assistant turn never surfaces. Attempted a general adapter fix (drop synthetic
  fact-sentences from the vector tier) — it REGRESSED on a 6-question smoke (0/6) because the
  deeper cause is the entity-filter exclusion of untagged turns, not the echoes. **Chose to REVERT
  rather than keep iterating**: closing this in-session, after seeing Letta win, against these exact
  numbers is the post-hoc iterate-until-win pattern this project was red-teamed for twice — even
  "general" fixes accepted under that selection pressure aren't credible. The honest path to
  overtaking Letta is general capability work (entity-tag raw turns / unfiltered-vector fallback
  for open-ended queries; date-aware temporal reasoning; a preference-summary path) developed +
  validated on HELD-OUT data, then re-run frozen — NOT same-session tuning. Logged as the concrete
  next-session lever. Repos clean; no CB code changed this step.
- 2026-07-02 — **TRACK D EXECUTED: first full LongMemEval head-to-head vs ALL THREE named
  rivals.** All three environment blockers found GONE on re-probe (ollama newly installed →
  pulls work, qwen2.5:7b at ~24+ tok/s on Metal; HF LFS egress restored → real LongMemEval
  downloaded). Runner `bench_longmemeval.py` (harness repo): same local model for every
  system's LLM calls + answers + judging, same nomic embedder; oracle variant, stratified
  n=138 seed 42; per-question fresh backend; crash-safe checkpoints. Substantial rival-infra
  work: Graphiti/Kuzu FTS indexes (upstream driver never creates them), Mem0 2.x API,
  Letta 0.16.8 stood up WITHOUT Docker (pip-embedded postgres via pgserver + pgvector +
  schema from letta's ORM + sequence-identity fix). **Result (frozen run, no tuning):
  CB 0.261 (20 min) vs Mem0 0.203 (77 min) — within noise at n=138 (McNemar p=0.24), a
  statistical tie on accuracy + clear CB cost win; CB beats Zep 0.065 (254 min) decisively
  (p<0.0001); Letta 0.471 (159 min) BEATS CB on accuracy (p=0.0002) at ~8× cost, so the
  DONE clause is NOT met on this variant.**
  Context: oracle questions have ~2 evidence sessions → Letta answers agentically with
  transcripts effectively in-context (memory machinery barely engages); the `_s` variant
  (~115k-token haystacks) is where that collapses — the follow-up run. CB is the strongest
  of all four on knowledge-update (0.478 — bi-temporal supersede working); weakest on
  temporal-reasoning (0.043) and preference (0.087) — general improvement targets, to be
  fixed and re-run frozen. Full write-up: harness `RESULTS_longmemeval.md`. One CB core
  change shipped during smoke (pre-freeze): backstop reserves half the context for vector
  recall (bit-identical on the diagnostic harness, hash 673a25c7…).
- 2026-07-03 — **Track D _s-variant (context-overflow regime) DONE — the decisive test.**
  `longmemeval_s`: mean 50 sessions / 494 turns / ~490k chars per question (overflows the 7B's
  32k window — the regime a memory layer actually targets). n=24 seed 42, same shared
  model/embedder/judge. **CB 0.167 (24/24, 3.1 min/q) EXACT-TIES Mem0 0.167 (24/24, 25.1 min/q)
  — CB 8× cheaper; Letta 0.083 (n=12 partial, cut per pre-registered rule at ~70 min/q) ties
  CB 1/12 on the questions it finished — its win is completeness+cost, not accuracy; Letta
  COLLAPSES from its oracle 0.471 once history can't fit context (Letta-vs-itself, unambiguous);
  Zep DNF (0 questions in ~2 h; ~200 graphiti LLM calls/question infeasible on a local 7B).** So the
  oracle picture INVERTS at scale: Letta's win was full-context reading, not memory. Combined
  verdict: CB ≥ Mem0 (accuracy tie both variants, always cheaper) and CB ≥ Zep hold; CB ≥ Letta
  holds only at `_s` → unconditional clause NOT met, but CB is the accuracy co-leader + runaway
  cost leader in the fits-nowhere regime. Rival-infra work this session: Letta stood up
  Docker-free (pgserver+pgvector), its SDK default timeout raised to 1800 (first `_s` letta run
  was all-timeout — archived .bak); Mem0 empty-extraction embed crash patched (archived .bak);
  Zep/Graphiti Kuzu FTS indexes hand-created; episode truncation 8k→24k for `_s`. Three failed
  attempts archived for audit, two legs disclosed partials, all in harness `RESULTS_longmemeval.md`.
  CB gate stays green (164 passed); no CB tuning against any of these numbers (frozen). **Next
  lever to overtake Letta at `_s`: CB temporal-reasoning (0.0 — needs date arithmetic over
  retrieved memories) + preference paths.**
- 2026-07-02 — **Phase 5 repositioning (docs; reviewer PASS) — improvement-plan complete on
  the local axis.** README now leads with the four properties CB verifiably owns
  (deterministic/replayable, bi-temporal + provenance, offline-capable, hard isolation +
  zero-residue erasure — reviewer verified every claim against code/tests) and explicitly
  concedes the rivals are more established / the head-to-head has not run. Feature list
  refreshed. Bonus reviewer catch fixed: `TransformersLLM.complete` imported torch before the
  guarded extras error (clean env got ImportError instead of the actionable message).
  **SESSION SUMMARY (6 reviewed commits):** Phase 0 credibility triage `168ea01` → Phase 2
  correctness `7398003` → Phase 1 extraction-default `3880c2d` → Phase 3a essentials
  `d06a54c` → Phase 3b scale & tenancy `fa77425` → Phase 5 repositioning `92f6c1e`. Gate
  124→164 passed; external harness bit-identical (673a25c7…) through every code change.
  **Remaining:** MCP namespace arg (small follow-up); definite-NP coreference; Phase 4
  credible external eval — STILL ENDPOINT-BOUND (needs a capable hosted OpenAI-compatible
  endpoint; adapters pre-wired, protocol written in benchmark/README.md §Scope); PyPI upload
  (needs maintainer token).
- 2026-07-02 — **Phase 3b scale & tenancy (gate 164 passed; harness hash bit-identical
  673a25c7… — fourth consecutive workstream).** (1) `NamespacedMemory` — hard per-tenant
  isolation (store-per-namespace: own index/tier/resolver/coreference/echo state; `drop()` =
  one-call tenant erasure; legacy single-store files upgrade to the `default` namespace).
  Closes the biggest rival-parity gap (user/agent scoping) with stronger isolation than
  filter-based scoping. MCP still serves a single namespace — threading a namespace arg
  through the MCP tools is a follow-up. (2) `CuratedBrain(summarizer=LLM)` — consolidation
  now actually summarizes merged clusters (PRD §8) when a model is supplied; deterministic
  default unchanged. (3) ANN filter-pushdown — escalating over-fetch under selective filters
  (fixes the documented under-recall follow-up from P3); hnswlib duplicate-degenerate graphs
  now degrade gracefully (shrink k) instead of raising mid-query; `ef` auto-raised when the
  escalation outgrows it. (4) Track-C load bar MET: 1e5 records — reviewer-measured recall@10
  **1.000** (bar 0.90) and p95 **2.4 ms** (bar 50 ms), mean 1.9 ms (m=32, ef=400; `CB_SLOW=1`,
  ~68 s, kept out of the default gate). Reviewer PASS: tenant isolation probed across
  structured/vector/coreference/echo-guard/resolver-poison axes — nothing crosses; drop()
  zero-residue; BruteForce path character-identical; shrink path terminates on all-duplicate
  corpora. Non-blocking: a raising summarizer propagates out of consolidate() (fail-loud,
  judged acceptable).
- 2026-07-02 — **Phase 3a production essentials (gate 157 passed; harness hash bit-identical
  673a25c7… for the third consecutive workstream).** (1) `forget()` hard erasure — GDPR path,
  the one documented exception to never-hard-delete: facts incl. superseded history +
  object-side facts on full forget, asserting episodes/vector records, echo-guard entries,
  resolver entry; free-text mere-mentions documented as untraceable. (2) inverse/set queries
  (`answer_who`, `subjects_where`, new (predicate, object) index) — "who lives in Berlin?"
  was structurally unanswerable before. (3) Hnsw-backed `snapshot`/`save`/`stats` no longer
  crash (records-only payload, re-embed on load with documented demotion to exact).
  (4) derived planner/stale state cached, invalidated on mutation (was full-store scans per
  query). Remaining Phase-3 scope → 3b: namespacing (user/agent/session scoping — the
  biggest rival-parity gap), real LLM consolidation, filter-pushdown, ≥1e5 load test.
- 2026-07-02 — **Phase 1: extraction is now the proven default path (reviewer PASS, gate
  147 passed).** Reviewer independently verified: bit-identical harness hash, extraction-ON
  beats-all-baselines holds on seeds 0/1/2 (not cherry-picked), first-person path opt-in +
  injection-safe, echo-suppression judged a defensible general principle with the tradeoff
  documented. Reviewer's coupling judgment on the 2 new verb patterns ("has moved to",
  "works on project"): in-distribution but general English; the extractor still misses 4 of
  the dataset's 5 project templates — evidence of good faith, not template-hardcoding.
  Noted limitations: multi-word/non-ASCII speakers fail closed (subject patterns are
  single-token ASCII); "manager's name" tokenizes to a rough predicate key. The red-team's controlling insight ("benchmarks the config that can't work on
  real data, ships unbenchmarked the config that could") is closed on the in-repo suite:
  (1) predicate schemas unified (`location`→`city` alias; verb + possessive + planner + dataset
  now one vocabulary — verb-form updates finally supersede possessive-form facts);
  (2) first-person extraction (`resolve_first_person` + opt-in `metadata.speaker`; MCP defaults
  speaker="User"; bare capitalized pronouns no longer parsed as names) — the dominant real-data
  form previously extracted to NOTHING; (3) echo suppression: a verbatim restatement of an
  already-asserted statement reinforces instead of re-asserting (found live: the dataset's
  redundancy stream restated pre-update lines late, flipping extraction-ON C2 back to stale —
  the general Pillar-B fix, not a dataset patch); (4) **AC-9 now passes with `metadata.fact`
  withheld** — `run_harness(extraction=True)`: CB on the same raw text as the baselines scores
  1.0/1.0/0.99/0.91/1.0/1.0 (identical to spoon-fed) and strictly beats all three on every
  category, locked in CI; (5) planner prefers stored predicates over keyword guesses;
  (6) PRD §6 logprob surprise formally deferred post-v1 with rationale. External harness:
  determinism hash BIT-IDENTICAL before/after (673a25c7…). Still open toward a real-data claim:
  definite-NP coreference, LLM-extractor benchmarking on real logs (endpoint-bound).
- 2026-07-02 — **Phase 2 correctness fixes (red-team v2), reviewer PASS, harness numbers
  byte-identical** (re-ran the standard suite: 0.88/0.79/1.00/0.00/0.76 — unchanged). Fixed:
  (1) pipe-in-fact-value crash in consolidate (fact_key now a list; legacy strings parsed
  `split("|", 2)`); (2) bi-temporal decoupling — `metadata.fact["valid_from"]` records
  retroactive facts, and an out-of-order older assertion becomes closed HISTORY instead of
  inverting the open fact's interval; (3) unicode tokenization (`[^\W_]+` + casefold, NFKC
  normalize — identity on ASCII; non-Latin text now embeds + retrieves instead of zero-vector);
  (4) supersede-filtering rebuilt: provenance-linked (superseded fact_key → drop by rid) +
  entity-scoped token fallback, staleness judged per (subject,predicate) — a superseded role
  "manager" no longer filters every record containing the word store-wide. *Honest tradeoff
  (reviewer-noted): a free-text record with NO fact link and NO subject token stating a stale
  value now leaks where the old global filter caught it — accepted; the old filter's store-wide
  false positives were worse.* (5) resolver state persisted in snapshots (ambiguity history
  restore-faithful) + restore rejects dim-mismatched snapshots up front; (6) MCP: lock (hosts
  issue concurrent calls), wall-clock default timestamps (was t=0.0 — silently killed all
  temporal semantics), atomic batched persistence; (7) `max_context_items=` ctor knob (default
  4 unchanged) + merged claims keep entities/fact_key so consolidation no longer hides them from
  entity-filtered search. +17 tests (141 passed). *Reviewer latent note: successive out-of-order
  historical inserts store loose (overlapping) valid_to bounds — as_of answers stay correct;
  optional future hardening.*
- 2026-07-02 — **Phase 0 credibility triage (docs-only).** Red-team v2 recorded (section above).
  Public claims corrected: README + benchmark/README no longer say "independent / third-party /
  never saw during development" (harness relabeled same-author diagnostic suite); AC-9 spoon-feeding
  disclosed next to the claim; suite size (~25 queries, 0.04 = 1 query) stated; tuning disclosure
  (lose→3 levers→win) added; Mem0 n=3 table now carries the corrections block (CB-favorable subset,
  broader-run answer ties, provenance artifact NOT claimed as CB wins, handicapped rival); stale
  "Zep needs Docker" claim fixed (Graphiti+Kuzu adapter exists); external-benchmark + frozen-config
  requirement added to the named-rival path. No code changed.
- 2026-06-18 — Created. Recorded Stage 1–7 done, the proof→contender gap, locked decisions, and the 9-workstream roadmap.
- 2026-06-18 — Track A foundation: real `providers.py` (bge/e5 + Transformers LLM) + `cassette.py`
  reproducibility layer + `test_providers.py` + `[local]` extra. Gate 55 passed/2 skipped, ruff
  clean, Opus-4.8 reviewer PASS. Added Environment Notes. (commit `a080143`)
- 2026-06-18 — Track A increment 2: re-embed-on-upgrade migration (`VectorTier.reembed` +
  `CuratedBrain.reembed`) + captured a real live-LLM extraction proof (cached Qwen on CPU →
  `Erin | moved | to Vienna`). Gate 56 passed/3 skipped, ruff clean, Opus-4.8 reviewer PASS.
  Confirmed Track B feasible on CPU; confirmed bge-embedder live run blocked by HF egress + MPS
  bug for some models (use device="cpu").
- 2026-06-18 — Track B groundwork: `extraction.py` (`LLMExtractor`, few-shot + groundedness
  anti-hallucination filter) + honest replay-cassette tests (`test_extraction.py`,
  `tests/fixtures/extract_cassette.json`). Gate 59 passed/4 skipped, ruff clean, Opus-4.8 reviewer
  PASS (verified fixture is genuine model output, tests non-vacuous). Surfaced the 0.8B model's
  extraction weaknesses honestly.
- 2026-06-18 — Track B wiring: extractor wired into `CuratedBrain.write` (optional `extractor=`,
  default None → byte-identical legacy behavior). Raw text → structured tier proven end-to-end;
  grounding keeps leaked exemplars out of the store; N>1 routing covered. Gate 61 passed/4 skipped,
  ruff clean, Opus-4.8 reviewer PASS (non-breaking, AC-1 intact). **Milestone: the spoon-fed-fact
  crutch is removed when an extractor is supplied.**
- 2026-06-18 — Scoped B-eval: `test_beval.py` + `beval_cassette.json` (genuine recorded Qwen
  output). Extraction-ON **matches spoon-fed on C1/C2/C5/C6** on a controlled world; 1 model miss
  surfaced honestly. Gate 63 passed/4 skipped, ruff clean, Opus-4.8 reviewer PASS (cassette
  verified genuine, A/B non-vacuous, bi-temporal correct, scoping honest). Full-harness B-eval +
  Track D (LongMemEval vs Mem0/Letta/Zep) remain — need external systems / heavy compute.
- 2026-06-18 — Track F core surface: LICENSE (Apache-2.0), honest README (no overclaim of
  benchmark wins), CI workflow (ruff+pytest on py3.11/3.12). `pip install -e .` verified; README
  quickstart runs. Gate 63 passed/4 skipped, ruff clean, Opus-4.8 reviewer PASS (README accurate,
  LICENSE valid, CI correct). Pivoted to unblocked SHIP groundwork because Track D is env-blocked.
  Remaining F: PyPI publish (needs maintainer creds), docs/CHANGELOG/CONTRIBUTING, type-check, coverage.
- 2026-06-18 — Track H: robustness/property suite (`test_robustness.py`). The reviewer PASSed the
  suite then found 5 real bugs by probing beyond it — all fixed + locked, incl. a **silent
  bi-temporal corruption** (non-finite timestamp → un-queryable "open" fact). Added boundary
  validation to `write`/`query` (fail-loud, clear typed errors). Gate 69 passed/4 skipped, ruff clean.
- 2026-06-18 — Track G: `CuratedBrain.metrics()` observability (decision breakdown, discard_rate,
  store size). Reviewer PASS; addressed 2 nice-to-haves (cheap metrics without snapshot; reset on
  restore). Gate 73 passed/4 skipped, ruff clean.
- 2026-06-18 — Track C: durable `save`/`load`. The reopen test exposed a real **restore-fidelity
  bug** (`_session_ts` rebuilt only from stored episodes → C6 as-of answers diverged after restore);
  fixed by persisting it in the snapshot. Reviewer PASS (quantified 34/64 sessions lost pre-fix;
  confirmed no other field affected). Gate 74 passed/4 skipped, ruff clean.
- 2026-06-18 — Track E: complete top-level public API + 3 runnable `examples/` (2 offline,
  smoke-tested). Lazy imports verified (no torch at `import curated_brain`). Reviewer PASS.
  Gate 76 passed/4 skipped, ruff clean.
- 2026-06-18 — Published to `github.com/doubleMcyber/curated-brain` (public, default branch `main`);
  local branch `claude/curated-brain` renamed `main`. All 21 build commits pushed; CI workflow live.
- 2026-06-18 — Track B/D PROVE: **open-domain planner backstop** (improvement-plan #1, the "#1
  reason CB would lose Track D"). `query()` now consults the structured tier for a known entity
  even when no predicate keyword matches (was bypassed → vector-only), capped at MAX_CONTEXT_ITEMS-1
  to reserve vector recall; new `StructuredTier.predicates_for`. +3 tests. Gate 79 passed/4 skipped,
  ruff clean. **Opus-4.8 reviewer PASS** — independently instrumented `run_harness`: backstop fires
  **0×** on all 54 probes → AC-9 provably unchanged; zero bugs across as-of/render/budget/determinism
  edge cases. Branch `claude/open-domain-backstop`.
- 2026-06-18 — Track B/D PROVE: **schema-driven planner** (improvement-plan #1, remaining half).
  `Planner.plan` recognizes any stored predicate whose single-word name appears in the question
  (vocab derived in `query()` from `self.structured.facts`); un-hardcoded predicates now route
  precisely instead of hitting the backstop. +1 test (`test_schema_driven_planner_routes_an_unhardcoded_predicate`).
  Gate 80 passed/4 skipped, ruff clean. **Opus-4.8 reviewer PASS** — diffed every harness `QueryPlan`
  with/without the vocab: **0 differences**, C1–C6 identical (1.0/1.0/.987/.906/1.0/1.0); zero bugs.
  Branch `claude/open-domain-backstop`.
- 2026-06-18 — Track G: **cost/token accounting** in `metrics()["cost"]` (deterministic write+query
  hot-path meter: embed calls/tokens, extract calls, queries, context_tokens_served, avg_context_tokens).
  Fills the "≤ its cost" reporting prerequisite for Track D. +1 test (+1 restore-reset assertion).
  Gate 81 passed/4 skipped, ruff clean. **Opus-4.8 reviewer PASS** — hand-verified every count, confirmed
  rejected-call cost-neutrality, 0-query avg guard, restore reset, determinism; consolidation exclusion
  documented (not a silent undercount); zero bugs. Branch `claude/open-domain-backstop`.
- 2026-06-19 — **BENCHMARK Track A** (toward a preliminary head-to-head on the user's harness
  `doubleMcyber/longitudinal-memory-eval-harness`, which runs fully offline): `HeuristicExtractor`
  (deterministic, no-LLM `(subject,predicate,object)` parsing — possessive copula + general verb
  patterns; predicate canonicalization strips temporal markers so updates supersede) + generalized
  the schema-driven planner match from single-token to "all non-stop predicate tokens ⊆ question"
  (multi-word predicates like "mailing address" now route precisely). +3 tests. Gate 88 passed/4
  skipped, ruff clean. **Opus-4.8 reviewer PASS** — 20-sentence generality probe (general, not
  overfit), 54/54 harness QueryPlans identical (AC-9 unchanged), over-match guards verified, zero
  bugs. Branch `claude/heuristic-extractor`. **DISCOVERED (needed before the benchmark shows strong
  staleness/precision):** `_stale_objs`/`fuse` supersede-filter is token-based, so MULTI-WORD stale
  values (e.g. "14 Rua das Flores, Lisbon") are NOT dropped from vector context — the structured
  answer is clean but a stale episode can still surface as an item. Pre-existing; fix in the adapter
  phase (tokenize-subset or substring match, entity-scoped).
- 2026-06-19 — **BENCHMARK Track B (harness adapter) + FIRST REAL HEAD-TO-HEAD.** Implemented the
  `curated_brain` adapter in the harness (provenance threading via two ingest maps, NO CB core change;
  superseded items dropped via CB's own bi-temporal state, not gold). Adapter is contract-clean (17/17),
  deterministic, **adversarial-review PASS** (faithful + fair: 0 unmapped citations / 0 gold turns
  wrongly excluded / no gold peeking; answer switched to top-1 to match baselines). **Result (standard,
  seed 42, fully offline):** recall CB 0.76 vs temporal_rag 0.92; precision CB **0.63 (best of all)**;
  contradiction CB 0.80 = TR 0.80 (RAG backends 0.10); staleness 0.20 vs 0.06; answer 0.64 vs 0.76;
  cost CB **lowest**; long-range-recall by category CB **0.83 vs 0.67**. **Headline bar NOT met** (CB
  loses recall) → **STOPPED per the plan's "don't tune to the benchmark" rule.** The losses are GENERAL,
  non-tuning gaps: (1) multi-hop cites only the final hop (CB answers multi-hop *better*); (2) no
  "originally/first" history-/recency-intent path; (3) extractor canonicalization misses on some
  contradiction phrasings; (4) storage slope is a snapshot-JSON-verbosity artifact, not real bloat.
  Harness branch `claude/curated-brain-adapter` → `RESULTS_curated_brain.md`. This is the project's
  first REAL external head-to-head (vs strong RAG references, not yet Mem0/Letta/Zep): CB is
  competitive, cheaper, more precise, contradiction-strong — with a clear general path to overtake.
- 2026-06-19 — **Multi-hop full-chain provenance** (general correctness fix): `query()` now cites
  EVERY fact in a resolved hop chain (`StructuredTier.resolve_path_chain`), not just the final one,
  so the whole support set is attributable. +1 test. Gate 89 passed/4 skipped, ruff clean. **Opus-4.8
  reviewer PASS** (AC-9 C1–C6 unchanged incl. C5; resolvable→2 cites, unresolvable→0, no crash; zero
  bugs). **Honest note:** this does NOT move the harness `multi_hop` recall — the bottleneck is
  UPSTREAM: the planner hardwires the only relation to "manager" (`_RELATION_PREDS`), so hop chains
  never form for the harness's other relations. Closing that needs GENERAL relation auto-detection
  (a predicate whose object is itself a known entity) — deferred deliberately to avoid drifting into
  benchmark-coupling; tracked as the next general lever alongside the history/recency-intent path.
- 2026-06-19 — **Beat-the-references push (user-chosen): two general-capability commits, both reviewer
  PASS, AC-9 exact throughout.** (1) Multi-entity backstop + relational extraction ("works at/for",
  "is headquartered in", "located in"): `query()` now falls back to surfacing facts for EVERY named
  entity when a plan is open-domain OR mis-keyworded (round-robin, budget-reserved) → `multi_hop`
  recall 0.67→1.00, precision 0.63→0.71. (2) Recency-based pronoun coreference ("Their/His/Her current
  X" → most-recent subject; stateful, reset-cleared) → `contradiction` recall 0.80→1.00,
  contradiction-resolution 0.80→1.00 (beats TR), staleness 0.20→0.00 (beats TR), answer 0.68→0.76 (tie).
  **Net: CB overall recall 0.76→0.88; CB now wins/ties every quality metric vs temporal_rag except
  overall recall.** Remaining 0.04 deficit = `recency_relevance` (definite-NP/ellipsis coreference) —
  NOT added (would be benchmark-coupling). Branch `claude/heuristic-extractor` (f01b654, 3075c28);
  harness results on `claude/curated-brain-adapter`. Gate 92 passed/4 skipped, ruff clean.
- 2026-06-18 — Track A: **OpenAI-compatible remote providers** (`OpenAICompatEmbedder`/`OpenAICompatLLM`).
  Stdlib-only HTTP, injectable `post` transport → offline-tested (exact wire format, L2-norm, protocol
  conformance, drives a CuratedBrain write/query). Enables a FAIR Track-D run (CB + rivals on one
  endpoint+model) and sidesteps local-CPU. +4 tests. Gate 85 passed/4 skipped, ruff clean. **Opus-4.8
  reviewer PASS** — verified request bodies, zero-vector/empty-batch/float64 contract, no network/heavy-dep
  at import, temperature=0.0 default; zero bugs (1 nitpick: bare KeyError on malformed responses).
  Branch `claude/open-domain-backstop`.
- 2026-06-19 — **Planning pass (Opus architect) + hybrid retrieval (P0).** Strategic review produced
  a prioritized offline plan (P0–P9). Headline insight: the references use the SAME lexical embedder
  class, so CB's recall comparison is FAIR; hybrid lexical+semantic retrieval is the #1 lever.
  **P0 LANDED, reviewer PASS:** `VectorTier.search` now fuses 0.5·embedding-sim + 0.5·jaccard before
  truncating to k (general, deterministic, AC-9 exact, `nearest`/novelty stay pure-embedding). It
  FIXES bge's paraphrase regression (longitudinal/lexgap → 1.00) but does NOT close the headline recall
  gap — the remaining bge loss is `needle`, a surprise-gate/storage interaction, not retrieval (so a
  ranking change can't fix it). Honestly framed, not over-claimed. Gate 99 passed.
  **Remaining priorities (offline, general, NOT benchmark-tuning):** P1 entity resolution/canonicalization
  (alias map — biggest general accuracy lever); **[x] P2 DONE (2026-06-19, reviewer PASS)** — relation
  auto-detection (predicate whose object is a known entity → relational; passed to `Planner.plan` as
  `relation_preds`) so multi-hop forms for ANY relation, not just "manager"; no-op on synthetic (AC-9
  byte-identical, git-stash-confirmed). **[x] P5 DONE (2026-06-19, reviewer PASS)** — multi-word supersede-filtering moved INTO core
  (`_stale_token_sets`/`fuse` token-subset); the harness adapter's `_superseded_turns` workaround removed
  (staleness=0.00 byte-identical → core, not adapter — honesty wrinkle closed); AC-9 exact. P3 wire
  `HnswIndex.topk`+over-fetch into the tier (real end-to-end ANN speedup) + P7 index the structured tier →
  P4 ≥1e5 soak test; P6 LangChain adapter; CI: run cassette tests in-gate + coverage. Defer P8 (Mem0
  speedup — low ROI on a weak model) and P9 (logprob surprise — no metric movement). Branch `claude/heuristic-extractor`.
  **[x] P6 DONE** (LangChain Retriever). **[x] P7 DONE (2026-06-19, reviewer PASS)** — structured tier
  indexed by `(subject,predicate)` (`_by_key`, same Fact objects as the list so in-place supersede stays
  consistent; rebuilt on `load`; `to_dict` unchanged → snapshots byte-identical). Reads are O(1)-per-key
  (10k-fact resolve in ~0.017s vs old O(n) scan); AC-9 byte-identical; index==scan verified (incl. 12
  supersedes, mixed-case, restore). Soak/scale test landed (Track H).
  **[x] P1 DONE (2026-06-20, ULTRACODE workflows — design panel + 4-lens adversarial verify).**
  `curated_brain/resolve.py EntityResolver`: conservative subject canonicalization ("Erin"/"Erin Smith"/
  "Ms. Smith" -> one entity) via honorific-strip + exact component subsumption, gated on store-provable
  uniqueness, NO fuzzy matching. Hooks: write canonicalizes subject (copy-first, case-preserved when no
  merge), query canonicalizes plan.entity + backstop, answer_structured/path, restore rebuilds the resolver.
  AC-9 byte-identical (no-op on single-token harness names). **The verify workflow caught TWO real
  false-merge bugs** (same-index-only poison missed cross-index; then poisoned-one-index still promoted via
  the other) — both fixed; a 4392-permutation brute-force final check found 0 false merges. +13 tests.
  Gate 123 passed/4 skipped. **Still open: P3 ANN-into-tier over-fetch (interacts with P0 hybrid);
  object-side relation canonicalization (deferred behind the same gate).**
