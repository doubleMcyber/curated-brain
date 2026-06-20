"""Consolidation worker (PRD §8, Pillar C) — the self-organizing background pass.

Run periodically (off the query critical path), it compresses the raw episodic tier into
a durable semantic tier:

* **Cluster & summarize** near-duplicate episodes into a single semantic claim that keeps
  support links back to its source episodes.
* **Deduplicate** — merging a cluster removes the redundant copies (summed support).
* **Resolve contradictions** — already non-lossily superseded in the structured tier at
  write time; consolidation reports them and preserves the audit trail.
* **Prune** stale/unsupported noise and compact the episodic log.

Invariant (PRD §8): consolidation may compress and reorganize but never destroys a
structured fact or its provenance — the audit trail of every superseded fact survives.

Pure, order-deterministic helpers live here; the backend orchestrates the tier updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from curated_brain.models import EpisodicRecord

CLUSTER_THRESHOLD = 0.6  # cosine; same-fact paraphrases cluster, distinct facts do not


def cluster_by_similarity(pairs: list[tuple[np.ndarray, object]],
                          threshold: float = CLUSTER_THRESHOLD) -> list[list]:
    """Greedy single-pass clustering by cosine similarity to each cluster's first member.

    Deterministic in the input order. ``pairs`` is ``[(unit_embedding, record), ...]``.
    """
    clusters: list[dict] = []
    for emb, rec in pairs:
        for c in clusters:
            if float(emb @ c["rep"]) >= threshold:
                c["members"].append(rec)
                break
        else:
            clusters.append({"rep": emb, "members": [rec]})
    return [c["members"] for c in clusters]


def representative(members: list[EpisodicRecord]) -> EpisodicRecord:
    """Pick the cluster's representative: most-reinforced, then earliest id (deterministic).
    Its text becomes the extractive summary of the merged claim."""
    return min(members, key=lambda r: (-r.support_count, r.id))
