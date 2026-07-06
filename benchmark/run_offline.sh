#!/usr/bin/env bash
# Reproduce the offline Curated-Brain-vs-references benchmark from a clean checkout.
#
# Deterministic & offline at EVAL time (token-cosine embedder + a no-model normalizing judge
# ship in the harness). No GPU, no API keys. Network is used only on first run to clone the
# harness and pip-install deps (numpy/setuptools); set LMEH_PATH to skip the clone. ~1-2 min.
#
# Usage:
#   benchmark/run_offline.sh                # clones the harness next to this repo if absent
#   LMEH_PATH=/path/to/harness benchmark/run_offline.sh   # use an existing harness checkout
#
# What it does: installs this library editable, runs Curated Brain + the shipped references on
# the standard suite (seed 42, k=10), and prints the `compare` scoreboard (recall/precision/
# contradiction/staleness/answer-acc/cost). See benchmark/README.md for the expected numbers.
set -euo pipefail

CB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_URL="https://github.com/doubleMcyber/longitudinal-memory-eval-harness.git"
HARNESS_BRANCH="claude/curated-brain-adapter"   # carries the curated_brain adapter
HARNESS_COMMIT="5e1e342"                         # pinned reviewed commit (branches can move)
LMEH_PATH="${LMEH_PATH:-$(dirname "$CB_DIR")/longitudinal-memory-eval-harness}"

if [ ! -d "$LMEH_PATH" ]; then
  echo "==> cloning harness into $LMEH_PATH @ $HARNESS_COMMIT"
  git clone --branch "$HARNESS_BRANCH" "$HARNESS_URL" "$LMEH_PATH"
  git -C "$LMEH_PATH" checkout "$HARNESS_COMMIT"   # pin for byte-reproducibility
fi

echo "==> installing curated-brain (editable) so the harness adapter can import it"
python3 -m pip install -e "$CB_DIR" >/dev/null

cd "$LMEH_PATH"
python3 -m pip install -e . >/dev/null
# Dedicated output dir so the compare globs only this run's scorecards (not stale artifacts).
OUT="results/offline_repro"
rm -rf "$OUT" && mkdir -p "$OUT"
BACKENDS="curated_brain temporal_rag naive_rag semantic_rag long_context no_memory"
echo "==> running backends: $BACKENDS"
for b in $BACKENDS; do
  PYTHONPATH=src python3 -m mem_eval.runner.cli run \
    --backend "$b" --suite v1 --seed 42 --k 10 --scale standard --out "$OUT/"
done

echo "==> scoreboard"
PYTHONPATH=src python3 -m mem_eval.runner.cli compare "$OUT"/*.json
