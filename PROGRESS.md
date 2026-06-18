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
      - [ ] **Live bge embedder run** — BLOCKED: HF LFS weight egress is blocked in this env
        (config downloads, weights don't; bge not pre-cached). Code+test ready; runs where egress works.
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
      - [ ] Entity resolution / canonicalization + coreference; extraction confidence → gate.
      - [ ] **B-eval:** run the *full longitudinal harness with extraction ON* (needs a recorded
        cassette over the whole stream — hours of CPU inference, or a faster runtime) and compare
        C-category scores vs the spoon-fed baseline. Try Qwen3.5-2B for better predicate mapping.
      *Bar:* structured tier populated from **raw text only** (no `metadata.fact`), extraction F1 ≥ target.
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
- Phase: **Track 1 PROVE, well advanced.** Track A real-provider seam + cassette + re-embed
  migration landed (live LLM verified on CPU); **Track B extraction is implemented AND wired into
  the write path** — raw text → structured tier proven end-to-end. The pipeline is no longer
  architecturally faked *and* no longer depends on spoon-fed facts (when an extractor is supplied).
  Env-blocked: live bge embedder run (HF weight egress). 5 reviewed commits this session.
- **Next:** B-eval — run the full longitudinal harness with extraction ON (record a stream-wide
  cassette, or use a faster runtime) and compare C-category scores vs the spoon-fed baseline.
  Then D (LongMemEval vs Mem0/Letta/Zep). Interleave remaining Track A (logprob; real-LLM consolidation).
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
  crutch is removed when an extractor is supplied.** Next: B-eval on the full harness.
