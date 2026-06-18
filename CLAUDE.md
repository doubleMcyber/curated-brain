# The Curated Brain — build controller

**Done** — `pytest -q` exits 0 and `ruff check .` is clean. No tests exist yet, so the gate is the acceptance tests written from `PRD.md` §10; writing them is Stage 1.

**Workflow** — Build `PRD.md` in 7 stages, in order. After each stage run the gate; commit only when it passes.
1. Test scaffold + `MemoryBackend` ABC stub + seeded synthetic longitudinal dataset generator + `pyproject` (pytest, ruff) → AC-1
2. Structured tier: entities/relations, bi-temporal, provenance; exact/relational/as-of queries → AC-8
3. Vector tier: embedder protocol, ANN index, semantic recall → C1 recall units
4. Hybrid retrieval: planner, fusion re-rank, supersede-filtering → AC-2, AC-3, AC-4
5. Surprise-gated selective write path → AC-5
6. Consolidation worker: summarize, dedupe, extract, resolve contradictions, prune → AC-6, AC-7
7. Full longitudinal run beating naive-RAG / long-context / no-memory on every category → AC-9

**Verify** — Before marking any stage done, run a separate reviewer pass on Opus 4.8 against that stage's `PRD.md` §10 acceptance criteria. Never mark a stage done on your own say-so.

**Git** — Work on `claude/<feature>`; never commit to or push `main`; one conventional-message commit per passing stage.

**Guardrails** — Stay inside this repo; no `git push`, no force-push, no deleting files outside the working tree.

**Stop** — If the gate can't pass after a fair attempt, stop and write the blocker to `PROGRESS.md` rather than looping.

`/goal` — All 7 stages done when `pytest -q` exits 0 and `ruff check .` is clean and `git status` is clean on a `claude/*` branch with one commit per passing stage; cap 60 turns, then stop and write `PROGRESS.md`.
