---
name: Feature request
about: Propose a capability or integration
title: ""
labels: enhancement
---

**The problem / use case** (what are you trying to do that the memory layer doesn't support today?)

**Proposed shape** (API sketch, or which seam it plugs into — extractor, embedder/`VectorIndex`,
retrieval planner, consolidation, or an adoption surface like MCP/LangChain)

**Constraints it must respect**: stays reproducible offline on the deterministic fakes; doesn't
break AC-1 byte-determinism or AC-9; real-model deps stay optional behind an extra.
