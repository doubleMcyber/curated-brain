"""Token→dollar pricing for the cost axis (PRD §G / the Track-D "≤ its cost" claim).

CB's core meters token *counts* deterministically (embeddings computed, context tokens served
per query). This turns those counts into an estimated USD figure given per-1k-token rates, so
"CB is cheaper" can be stated in dollars, not just tokens/wall-time.

Honest scope: this prices the two things the core meters — the embedding calls CB makes and
the context tokens it serves to a downstream model. It does NOT include an LLM extractor's
completion tokens (those live behind the ``LLM`` protocol and aren't metered in the core) or
consolidation's amortized re-embeds. Set ``extract_usd_per_call`` to attribute a flat per-call
cost to extraction if you want a rough all-in figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """Per-1k-token (and per-call) USD rates. All default to 0.0 — set what applies to your
    provider. ``embed`` prices tokens CB embeds; ``context`` prices tokens CB serves to the
    answering model; ``extract_usd_per_call`` is an optional flat charge per extraction call."""

    embed_usd_per_1k: float = 0.0
    context_usd_per_1k: float = 0.0
    extract_usd_per_call: float = 0.0

    def estimate(self, cost: dict) -> float:
        """USD estimate from a ``metrics()["cost"]`` block (embed_tokens, context_tokens_served,
        extract_calls). Deterministic; unmetered components (see module docstring) are excluded."""
        embed = cost.get("embed_tokens", 0) / 1000.0 * self.embed_usd_per_1k
        served = cost.get("context_tokens_served", 0) / 1000.0 * self.context_usd_per_1k
        extract = cost.get("extract_calls", 0) * self.extract_usd_per_call
        return embed + served + extract


__all__ = ["Pricing"]
