"""Smoke-test the runnable examples so they never silently rot.

Only the offline examples (no model download) run here; example 03 uses the real providers,
whose behavior is covered by the `CB_LIVE` provider/extraction tests.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run(name: str) -> str:
    result = subprocess.run([sys.executable, str(EXAMPLES / name)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"{name} failed:\n{result.stderr}"
    return result.stdout


def test_basic_memory_example_runs():
    out = _run("01_basic_memory.py")
    assert "Vienna" in out          # Erin's city, via exact lookup and the multi-hop join
    assert "discard_rate" in out    # the metrics line printed


def test_belief_update_example_runs():
    out = _run("02_belief_update.py")
    assert "Current city        : Berlin" in out          # current value surfaces
    assert "As-of t=50 (earlier): Vienna" in out          # earlier value still retrievable
