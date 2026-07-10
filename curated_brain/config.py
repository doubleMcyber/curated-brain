"""Frozen, typed tuning config (``CBConfig``) — one place to see and override every
constant that shapes recall, selectivity and consolidation.

Defaults are bound to the existing module-level constants (single source of truth), so a
default ``CBConfig()`` reproduces today's behavior byte-for-byte. This module imports only
the lightweight tier modules (no ``torch``/heavy deps, and never ``backend`` — that would
cycle), keeping it cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass

from curated_brain.retrieval import HALF_LIFE_SECONDS, W_IMP, W_REC, W_REL
from curated_brain.surprise import SurpriseGate
from curated_brain.vector import _OVERFETCH, _W_LEX, _W_SEM

# Context budget + free-text dedup threshold. This module is their single source of truth
# (backend reads them off self.config): they live here, not in backend, to avoid a cycle
# (backend imports config). 4 = tiny curated payload (PRD §7); 0.85 = only genuine near-dups.
_MAX_CONTEXT_ITEMS = 4
_FREE_DEDUP_THRESHOLD = 0.85

# SurpriseGate ships its defaults on __init__; read them off a default instance so CBConfig
# never restates the literals (still the gate's single source of truth).
_GATE_DEFAULTS = SurpriseGate().to_dict()


@dataclass(frozen=True)
class CBConfig:
    """Immutable tuning knobs. All fields default to today's constants, so ``CBConfig()`` is
    a behavioral no-op. Threaded into ``CuratedBrain`` via ``config=``; an explicit per-arg
    ctor value (e.g. ``max_context_items=``) still wins over the config field.

    Tuning is a property of the live process, not the data: ``CBConfig`` is NOT persisted in
    ``snapshot()``. On ``restore()`` the retrieval/vector knobs come from the destination
    brain's config, while the surprise gate's runtime state (thresholds included) comes from
    the snapshot — restoring into a brain with different gate knobs keeps the snapshot's gate."""

    # Retrieval — context budget + fusion weights (retrieval.fuse) + recency half-life.
    max_context_items: int = _MAX_CONTEXT_ITEMS
    free_dedup_threshold: float = _FREE_DEDUP_THRESHOLD
    w_rel: float = W_REL
    w_rec: float = W_REC
    w_imp: float = W_IMP
    half_life_seconds: float = HALF_LIFE_SECONDS

    # QUERY-side fuzzy entity fallback (opt-in). None = off, today's behavior byte-identical.
    # When set to a difflib cutoff (e.g. 0.85), a question naming no exact known entity may
    # match a close variant (typo/diminutive); it never overrides an exact match and fails
    # closed on ambiguity. WRITE-side resolution stays exact-only (see resolve.py) — a false
    # merge on ingest corrupts the store, whereas a query-time miss only costs one answer.
    fuzzy_entity_cutoff: float | None = None

    # Surprise gate (surprise.SurpriseGate defaults).
    budget: float = _GATE_DEFAULTS["budget"]
    theta0: float = _GATE_DEFAULTS["theta"]
    theta_floor: float = _GATE_DEFAULTS["theta_floor"]
    theta_max: float = _GATE_DEFAULTS["theta_max"]
    lr: float = _GATE_DEFAULTS["lr"]

    # Vector hybrid scoring (vector._W_SEM/_W_LEX) + ANN over-fetch factor (vector._OVERFETCH).
    w_sem: float = _W_SEM
    w_lex: float = _W_LEX
    overfetch: int = _OVERFETCH
