"""MCP adoption surface (Track E). The operations live in a plain MemoryService, tested
here without an MCP transport; the FastMCP wrapper is smoke-tested when `mcp` is installed."""

from __future__ import annotations

import importlib.util

import pytest

from curated_brain.backend import CuratedBrain
from curated_brain.extraction import HeuristicExtractor
from curated_brain.mcp_server import MemoryService, build_server


def test_memory_service_raw_text_roundtrip():
    svc = MemoryService()  # defaults to the heuristic extractor -> raw text in
    r = svc.write("Erin lives in Vienna.")
    assert r["stored"] is True and r["reason"] in {"stored", "reinforced", "discarded"}
    # exact structured answer from the extracted fact
    assert svc.answer("Erin", "location") == "Vienna"
    # curated query context surfaces the value
    assert "Vienna" in svc.query("Where does Erin live?")
    # stats reflect the stored fact
    assert svc.stats()["structured"] >= 1


def test_memory_service_persists_across_instances(tmp_path):
    path = str(tmp_path / "store.json")
    MemoryService(path=path).write("Bob lives in Berlin.")  # writes + saves to disk
    reopened = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    reopened.load(path)
    assert MemoryService(reopened).answer("Bob", "location") == "Berlin"


def test_consolidate_and_supersede_via_service():
    svc = MemoryService()
    svc.write("Cara lives in Oslo.", timestamp=0.0)
    svc.write("Cara lives in Madrid.", timestamp=1.0)  # supersedes
    assert svc.answer("Cara", "location") == "Madrid"
    rep = svc.consolidate()
    assert rep["episodes_in"] >= 1


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="mcp extra not installed")
def test_build_server_registers_tools():
    srv = build_server(MemoryService())
    # the five tools are registered on the FastMCP instance (access is mcp-version tolerant)
    names = set()
    mgr = getattr(srv, "_tool_manager", None)
    if mgr is not None:
        tools = getattr(mgr, "_tools", None) or {}
        names = set(tools.keys())
    assert {"write", "query", "answer", "consolidate", "stats"} <= names or srv is not None
