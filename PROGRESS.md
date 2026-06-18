# PROGRESS — The Curated Brain

> Cross-session state tracker. **Update this after every workstream / session** so a fresh
> session can resume with full context. Companion roadmap (detail + rationale):
> `~/.claude/plans/cosmic-watching-giraffe.md`.

Last updated: 2026-06-18 · Branch: `claude/curated-brain`

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
- [~] **A. Real local-model integration.** *(foundation done — reviewer PASS 2026-06-18)*
      - [x] Real providers behind the protocols: `providers.py` — `SentenceTransformerEmbedder`
        (bge/e5) + `TransformersLLM` (cached Qwen/Mistral); lazy, soft-dependency, unit-norm,
        greedy. Fakes remain the **default** test doubles, so the offline gate needs no model stack.
      - [x] Cassette record/replay (`cassette.py`) for reproducible real-model runs in CI.
      - [x] `tests/test_providers.py` (5 offline always-run + 2 `CB_LIVE`-gated live); `[local]` extra.
      - [x] Offline gate green (55 passed / 2 skipped), ruff clean, Opus-4.8 reviewer PASS.
      - [ ] Live bge end-to-end *executed* (code+tests ready; weight download slow in this env — see Env notes).
      - [ ] Remaining A scope: **logprob surprise estimator** (PRD §6 #2), **re-embed-on-upgrade**
        migration, wire the **real LLM into consolidation** (replace `RuleBasedLLM` path).
      *Bar:* non-faked end-to-end run (real embedder + real LLM) green; fakes retained as doubles.
- [ ] **B. Ingestion intelligence.** Text→triple extraction, entity resolution/coreference,
      schema, extraction confidence; schema/grammar-constrained decoding to offset small-model
      weakness. *Bar:* structured tier populated from **raw text only** (no `metadata.fact`),
      extraction F1 ≥ target on a held-out set.
- [ ] **D. External evaluation.** LongMemEval harness; head-to-head vs Mem0/Letta/Zep on the
      same local model; report accuracy **and** cost/latency/size; ablations; reproducible.
      *Bar:* **Curated Brain ≥ each competitor on the headline metric at ≤ its cost**, reproducible
      from a clean checkout. ← **the contender claim**

### Track 2 — PRODUCTIONIZE (make the numbers hold under load)
- [ ] **C. Storage & scale.** Real ANN (hnswlib/faiss/sqlite-vec) + durable persistence
      (SQLite + on-disk index) behind existing protocols; concurrency/async; namespacing.
      *Bar:* load test ≥1e5 records meeting a stated recall@k + p95-latency bar.
- [ ] **G. Observability, ops & cost.** Decision logging/tracing, metrics (size/discard/recall/$),
      budgets + backpressure, graceful provider failure. *Bar:* dashboards/metrics emitted for a full run.
- [ ] **H. Robustness & hardening.** Property/fuzz tests, determinism story under real models,
      long-horizon soak, security review of key handling + blob restore. *Bar:* fuzz suite green; soak holds.

### Track 3 — SHIP (adoption surface + narrative)
- [ ] **E. Public API & DX.** Clean SDK facade, LangChain/LlamaIndex + **MCP server**, examples,
      quickstart, typed config. *Bar:* quickstart runs from a clean env in <10 min.
- [ ] **F. Release engineering.** LICENSE (Apache-2.0), README, docs site, CHANGELOG, CONTRIBUTING,
      PyPI publish, CI matrix (pytest+ruff+type-check), coverage. *Bar:* `pip install` works; CI green.
- [ ] **I. Positioning.** Write-up of the "memory is curation" thesis + benchmark table, comparison
      matrix vs rivals, demo, launch post. *Bar:* publishable report with reproducible numbers.

---

## CURRENT POSITION
- Phase: **Track 1 PROVE, in progress.** Track A foundation landed (real provider seam +
  cassette + reviewer PASS); the pipeline is no longer architecturally faked.
- **Next concrete step:** finish Track A then start B — wire the **real LLM into consolidation**
  + add the **logprob surprise estimator**, then **B**: extract `(subject,predicate,object)`
  triples + entity resolution from **raw text** so the structured tier no longer depends on the
  dataset's `metadata.fact` spoon-feeding. That unblocks D (LongMemEval).

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
- _(none blocking. Note: live bge weight download is slow in this env; live embedder test is
  written + gated, executes once weights finish — not a code blocker.)_

## CHANGELOG OF THIS FILE
- 2026-06-18 — Created. Recorded Stage 1–7 done, the proof→contender gap, locked decisions, and the 9-workstream roadmap.
- 2026-06-18 — Track A foundation: real `providers.py` (bge/e5 + Transformers LLM) + `cassette.py`
  reproducibility layer + `test_providers.py` + `[local]` extra. Gate 55 passed/2 skipped, ruff
  clean, Opus-4.8 reviewer PASS. Added Environment Notes. Remaining A: logprob estimator,
  re-embed migration, real-LLM consolidation.
