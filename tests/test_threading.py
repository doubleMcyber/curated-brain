"""In-process thread safety (see README "Concurrency").

Each ``CuratedBrain`` holds one coarse RLock acquired at every public entry point, so
concurrent calls serialize instead of corrupting state. These tests hammer the public API
from many threads and assert (a) no exception and a store size matching the serial run,
(b) mixed write/query concurrency is safe, and (c) a fixed submit order reproduces the
serial snapshot byte-for-byte — proving the lock serializes without changing behavior.
"""

from __future__ import annotations

import json
import threading

from curated_brain.backend import CuratedBrain
from curated_brain.surprise import SurpriseGate

N_THREADS = 16
PER_THREAD = 20


def _store_all_gate() -> SurpriseGate:
    """A gate that stores every write, so the store size is independent of write order
    (the selectivity gate's decisions depend on prior vectors, hence on order)."""
    return SurpriseGate(budget=1.0, theta0=0.0, theta_floor=0.0)


def _obs(i: int) -> tuple[str, dict]:
    """A distinct observation with a distinct-subject fact, so it always stores as new."""
    return (f"note number {i} about topic {i}",
            {"fact": {"subject": f"e{i}", "predicate": "p", "object": f"v{i}"}})


def _serial_size(n: int) -> int:
    cb = CuratedBrain(seed=0, gate=_store_all_gate())
    for i in range(n):
        content, meta = _obs(i)
        cb.write(content, session_id="s0", timestamp=float(i), metadata=meta)
    return cb.metrics()["store_size"]


def test_concurrent_writes_no_exception_and_size_matches_serial():
    cb = CuratedBrain(seed=0, gate=_store_all_gate())
    total = N_THREADS * PER_THREAD
    errors: list[BaseException] = []

    def worker(base: int) -> None:
        try:
            for j in range(PER_THREAD):
                i = base + j
                content, meta = _obs(i)
                cb.write(content, session_id="s0", timestamp=float(i), metadata=meta)
        except BaseException as e:  # noqa: BLE001 - surface any thread failure to the test
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t * PER_THREAD,))
               for t in range(N_THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, errors
    assert cb.metrics()["store_size"] == total == _serial_size(total)
    # The snapshot is well-formed JSON regardless of interleaving.
    assert json.loads(cb.snapshot().decode("utf-8"))["counter"] > 0


def test_concurrent_write_and_query_no_exception():
    cb = CuratedBrain(seed=0, gate=_store_all_gate())
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for i in range(PER_THREAD * 4):
                content, meta = _obs(i)
                cb.write(content, session_id="s0", timestamp=float(i), metadata=meta)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def reader() -> None:
        try:
            for _ in range(PER_THREAD * 4):
                cb.query("topic", session_id="s0", timestamp=1000.0)
                cb.stats()
                cb.metrics()
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(4)] + \
              [threading.Thread(target=reader) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, errors


def test_barrier_ordered_writes_reproduce_serial_snapshot():
    """A fixed submit order enforced by a turn lock (not a race): each thread waits for its
    turn, writes one observation, then hands off. The final snapshot must byte-equal a plain
    serial run over the same order — the lock serializes cleanly without altering behavior."""
    total = N_THREADS * PER_THREAD

    serial = CuratedBrain(seed=0, gate=_store_all_gate())
    for i in range(total):
        content, meta = _obs(i)
        serial.write(content, session_id="s0", timestamp=float(i), metadata=meta)
    expected = serial.snapshot()

    concurrent = CuratedBrain(seed=0, gate=_store_all_gate())
    turn = threading.Lock()
    next_i = [0]
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            while True:
                with turn:  # serialize submission in strict index order
                    i = next_i[0]
                    if i >= total:
                        return
                    next_i[0] = i + 1
                    content, meta = _obs(i)
                    concurrent.write(content, session_id="s0", timestamp=float(i),
                                     metadata=meta)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, errors
    assert concurrent.snapshot() == expected
