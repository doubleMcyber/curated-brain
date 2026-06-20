---
name: Bug report
about: Something doesn't behave as documented
title: ""
labels: bug
---

**What happened vs. what you expected**

**Minimal repro** (ideally on the deterministic fakes — `DeterministicEmbedder` / `RuleBasedLLM` —
so it reproduces offline):

```python
from curated_brain import CuratedBrain
# ...
```

**Environment**: `curated_brain.__version__`, Python version, OS, and which extras are installed
(`[local]`, `[scale]`, `[mcp]`, `[langchain]`).
