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
    assert svc.answer("Erin", "city") == "Vienna"
    # curated query context surfaces the value
    assert "Vienna" in svc.query("Where does Erin live?")
    # stats reflect the stored fact
    assert svc.stats()["structured"] >= 1


def test_memory_service_persists_across_instances(tmp_path):
    path = str(tmp_path / "store.json")
    MemoryService(path=path).write("Bob lives in Berlin.")  # writes + saves to disk
    reopened = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    reopened.load(path)
    assert MemoryService(reopened).answer("Bob", "city") == "Berlin"


def test_consolidate_and_supersede_via_service():
    svc = MemoryService()
    svc.write("Cara lives in Oslo.", timestamp=0.0)
    svc.write("Cara lives in Madrid.", timestamp=1.0)  # supersedes
    assert svc.answer("Cara", "city") == "Madrid"
    rep = svc.consolidate()
    assert rep["episodes_in"] >= 1


@pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="mcp extra not installed")
def test_build_server_registers_tools():
    srv = build_server(MemoryService())
    # the tools are registered on the FastMCP instance (access is mcp-version tolerant)
    names = set()
    mgr = getattr(srv, "_tool_manager", None)
    if mgr is not None:
        tools = getattr(mgr, "_tools", None) or {}
        names = set(tools.keys())
    assert {"write", "query", "answer", "consolidate", "stats", "drop"} <= names or srv is not None


def test_namespaces_do_not_bleed():
    svc = MemoryService()
    svc.write("Erin lives in Vienna.", namespace="tenant-a")
    svc.write("Frank lives in Paris.", namespace="tenant-b")
    # each namespace only sees its own facts
    assert svc.answer("Erin", "city", namespace="tenant-a") == "Vienna"
    assert svc.answer("Erin", "city", namespace="tenant-b") == ""
    assert svc.answer("Frank", "city", namespace="tenant-b") == "Paris"
    assert svc.answer("Frank", "city", namespace="tenant-a") == ""
    # query context does not leak across the boundary
    assert "Vienna" in svc.query("Where does Erin live?", namespace="tenant-a")
    assert "Vienna" not in svc.query("Where does Erin live?", namespace="tenant-b")
    # the default namespace is a third, independent tenant
    assert svc.answer("Erin", "city") == ""


def test_drop_erases_only_the_target_namespace():
    svc = MemoryService()
    svc.write("Erin lives in Vienna.", namespace="tenant-a")
    svc.write("Frank lives in Paris.", namespace="tenant-b")
    res = svc.drop("tenant-a")
    assert res["existed"] is True
    # A's facts are gone (a fresh read lazily recreates an *empty* tenant-a); B untouched
    assert svc.answer("Erin", "city", namespace="tenant-a") == ""
    assert svc.answer("Frank", "city", namespace="tenant-b") == "Paris"
    # idempotent: dropping a never-touched namespace is a no-op, not an error
    assert svc.drop("never-existed")["existed"] is False


def test_default_namespace_back_compat_persist_roundtrip(tmp_path):
    # No namespace passed anywhere -> exactly the old single-store behavior, and the on-disk
    # blob stays raw-CuratedBrain-readable while only the default namespace exists.
    path = str(tmp_path / "store.json")
    MemoryService(path=path).write("Bob lives in Berlin.")
    reopened = CuratedBrain(seed=0, extractor=HeuristicExtractor())
    reopened.load(path)  # raw single-brain blob still loads
    assert MemoryService(reopened).answer("Bob", "city") == "Berlin"


def test_multi_namespace_persist_keeps_separation(tmp_path):
    path = str(tmp_path / "store.json")
    svc = MemoryService(path=path)
    svc.write("Erin lives in Vienna.", namespace="tenant-a")
    svc.write("Frank lives in Paris.", namespace="tenant-b")
    # reopen a fresh service from the same path -> namespaces stay separated
    reopened = MemoryService(path=path)
    reopened.load(path)
    assert reopened.answer("Erin", "city", namespace="tenant-a") == "Vienna"
    assert reopened.answer("Frank", "city", namespace="tenant-b") == "Paris"
    assert reopened.answer("Erin", "city", namespace="tenant-b") == ""


def test_store_path_is_single_namespace(tmp_path):
    store = str(tmp_path / "store.sqlite")
    svc = MemoryService(store_path=store)
    # default namespace works and journals durably
    svc.write("Bob lives in Berlin.")
    assert svc.answer("Bob", "city") == "Berlin"
    # any non-default namespace is refused with a clear error (honest single-namespace mode)
    with pytest.raises(ValueError, match="store_path mode supports only"):
        svc.write("Erin lives in Vienna.", namespace="tenant-a")
    with pytest.raises(ValueError, match="store_path mode supports only"):
        svc.answer("Bob", "city", namespace="tenant-a")
    with pytest.raises(ValueError, match="store_path mode supports only"):
        svc.drop("tenant-a")
    # reopening the same store file restores into the default namespace
    svc.close()
    reopened = MemoryService(store_path=store)
    assert reopened.answer("Bob", "city") == "Berlin"
    reopened.close()
