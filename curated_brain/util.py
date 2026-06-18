"""Deterministic, network-free helpers shared across tiers.

Everything here avoids wall-clock, ``random`` and salted ``hash()`` so that identical
input + seed always yields byte-identical state (the AC-1 determinism requirement).
"""

from __future__ import annotations

import hashlib
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A tiny stop-word set: dropping these sharpens lexical/embedding discrimination without
# pulling in a dependency. Intentionally small to stay predictable.
_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been do does did "
    "i you he she it we they this that with as by from".split()
)


def tokenize(text: str, *, drop_stop: bool = True) -> list[str]:
    """Lowercase alphanumeric tokens; optionally drop common stop-words."""
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stop:
        toks = [t for t in toks if t not in _STOP]
    return toks


def stable_hash(s: str) -> int:
    """A deterministic 64-bit hash (blake2b), unlike the salted built-in ``hash``."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def count_tokens(text: str) -> int:
    """Cheap token count used for retrieval-cost accounting (`tokens_in`)."""
    return len(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    """Lexical overlap of two strings in [0, 1]; used before the vector tier exists."""
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalize(s: str) -> str:
    """Canonical form for entity/value matching (PRD §5.1 entity resolution)."""
    return s.strip().lower()


# --- JSON sanitization for non-finite floats --------------------------------------
# Bi-temporal "open" intervals use float("inf"); plain json would emit the non-standard
# token `Infinity`. We wrap non-finite floats so snapshots stay strict, byte-stable JSON.

def to_jsonable(obj):
    """Recursively replace non-finite floats with tagged dicts so json stays strict."""
    if isinstance(obj, float):
        if obj == float("inf"):
            return {"__f__": "inf"}
        if obj == float("-inf"):
            return {"__f__": "-inf"}
        if obj != obj:  # NaN
            return {"__f__": "nan"}
        return obj
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def from_jsonable(obj):
    """Inverse of :func:`to_jsonable`."""
    if isinstance(obj, dict):
        if set(obj) == {"__f__"}:
            return {"inf": float("inf"), "-inf": float("-inf"), "nan": float("nan")}[obj["__f__"]]
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(v) for v in obj]
    return obj
