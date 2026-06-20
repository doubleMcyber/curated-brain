<!-- Thanks for contributing to The Curated Brain. -->

## What & why
<!-- What does this change and why? Link any issue: Closes #123 -->

## Checklist
- [ ] `pytest -q` passes (and `pip install -e ".[dev,mcp,langchain,scale]"` if you touched an adoption surface)
- [ ] `ruff check .` is clean
- [ ] AC-1 (byte-deterministic snapshot/restore) and AC-9 (curated beats the baselines on every
      category) are unchanged — the default path uses the deterministic fakes, no model stack
- [ ] New behavior has a focused test; docs/CHANGELOG updated if user-facing
