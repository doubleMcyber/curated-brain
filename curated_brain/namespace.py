"""Namespacing — hard-isolated per-tenant memory (multi-user / multi-agent scoping).

Every rival system scopes memory by user/agent as a core primitive; without it, one
store serves user A's facts to user B (and, via pronoun coreference, can even resolve
user B's "her" to user A's last subject). This layer closes that gap with the strongest
isolation available: **one CuratedBrain per namespace**. Nothing is shared — no vector
index, no structured tier, no resolver/coreference state, no echo guard — so cross-tenant
bleed is structurally impossible rather than filter-suppressed, and a whole tenant can be
erased in one call (``drop``), the strongest GDPR story.

Deterministic: namespaces serialize sorted; each sub-store snapshot is the CuratedBrain
byte format. ``load``/``restore`` accept a legacy single-store blob (it becomes the
``default`` namespace), so existing persisted stores upgrade in place.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

from curated_brain.backend import CuratedBrain

DEFAULT_NAMESPACE = "default"


class NamespacedMemory:
    """A family of fully-isolated :class:`CuratedBrain` stores, keyed by namespace.

    ``factory`` builds a fresh store for a new namespace (defaults to a deterministic
    ``CuratedBrain(seed=0)``); pass your own to configure embedder/extractor/gate per
    deployment. All per-store methods take the namespace first and lazily create it.
    """

    def __init__(self, factory: Callable[[], CuratedBrain] | None = None) -> None:
        self._factory = factory or (lambda: CuratedBrain(seed=0))
        self._spaces: dict[str, CuratedBrain] = {}
        # Guards the namespace registry (`_spaces`) only. Reentrant because registry ops call
        # one another (snapshot() iterates while space() may create). Each CuratedBrain owns
        # its own lock, so per-store work is not serialized behind this one.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ spaces -------
    def space(self, namespace: str = DEFAULT_NAMESPACE) -> CuratedBrain:
        """The namespace's store, created on first use."""
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(f"namespace must be a non-empty str, got {namespace!r}")
        with self._lock:
            if namespace not in self._spaces:
                self._spaces[namespace] = self._factory()
            return self._spaces[namespace]

    def namespaces(self) -> list[str]:
        with self._lock:
            return sorted(self._spaces)

    def drop(self, namespace: str) -> bool:
        """Erase an entire namespace (tenant off-boarding / full GDPR erasure).
        Returns whether it existed."""
        with self._lock:
            return self._spaces.pop(namespace, None) is not None

    # ------------------------------------------------------- scoped store operations --
    def write(self, namespace: str, observation: str, *, session_id: str,
              timestamp: float, metadata: dict | None = None):
        return self.space(namespace).write(observation, session_id=session_id,
                                           timestamp=timestamp, metadata=metadata)

    def query(self, namespace: str, question: str, *, session_id: str,
              timestamp: float, k: int = 8):
        return self.space(namespace).query(question, session_id=session_id,
                                           timestamp=timestamp, k=k)

    def forget(self, namespace: str, subject: str, *, predicate: str | None = None) -> dict:
        return self.space(namespace).forget(subject, predicate=predicate)

    def consolidate(self, namespace: str):
        return self.space(namespace).consolidate()

    # ------------------------------------------------------------------ persistence --
    def snapshot(self) -> bytes:
        """Deterministic multi-store blob: ``{"namespaces": {ns: <sub-blob utf-8>}}``."""
        with self._lock:
            payload = {"namespaces": {ns: self._spaces[ns].snapshot().decode("utf-8")
                                      for ns in sorted(self._spaces)}}
            return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        state = json.loads(blob.decode("utf-8"))
        with self._lock:
            self._spaces = {}
            if "namespaces" in state:
                for ns, sub in state["namespaces"].items():
                    cb = self._factory()
                    cb.restore(sub.encode("utf-8"))
                    self._spaces[ns] = cb
            else:  # legacy single-store blob -> it becomes the default namespace
                cb = self._factory()
                cb.restore(blob)
                self._spaces[DEFAULT_NAMESPACE] = cb

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self.snapshot())

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            self.restore(fh.read())


__all__ = ["NamespacedMemory", "DEFAULT_NAMESPACE"]
