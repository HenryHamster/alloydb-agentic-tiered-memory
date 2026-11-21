"""Offline checks for the MCP server: scope from env, lazy build, tool wiring."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from mcp_server import MemoryService, create_server, scope_from_env  # noqa: E402
from retain import Scope  # noqa: E402

SCOPE = Scope("alice", "demo", "s1")


class FakeSystem:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.recaller = SimpleNamespace()

    def context(self, scope, query, *, max_tokens):
        self.calls.append(("context", scope, query, max_tokens))
        recall = SimpleNamespace(items=[1, 2])
        return SimpleNamespace(recall=recall, format=lambda: f"[LONG-TERM MEMORY]\n{query}")

    def record_turn(self, scope, user_text, assistant_text, *, user_name, assistant_name):
        self.calls.append(("record", scope, user_text, assistant_text, user_name))
        return SimpleNamespace(episode_id="ep1", skipped=False, extracted=3)

    def finish_session(self, scope):
        self.calls.append(("finish", scope))
        return ["obs"], None

    def close(self):
        self.closed = True


def service():
    system = FakeSystem()
    builds = []

    def build():
        builds.append(1)
        return system, ()

    return MemoryService(SCOPE, build), system, builds


def test_scope_requires_user_and_generates_a_session():
    with pytest.raises(RuntimeError, match="MEMORY_USER_ID"):
        scope_from_env({})
    scope = scope_from_env({"MEMORY_USER_ID": "alice"})
    assert scope.user_id == "alice" and scope.project_id is None
    assert scope.session_id.startswith("mcp-")
    fixed = scope_from_env(
        {"MEMORY_USER_ID": "a", "MEMORY_PROJECT_ID": "p", "MEMORY_SESSION_ID": "s"}
    )
    assert fixed == Scope("a", "p", "s")


def test_system_is_built_once_on_first_use():
    svc, system, builds = service()
    assert not builds
    assert svc.recall_context("cats?")["memories_found"] == 2
    svc.remember_turn("I have a cat", "Noted", "Alice", None)
    assert svc.finish_session() == {"session_id": "s1", "observations_built": 1}
    assert len(builds) == 1
    assert [c[0] for c in system.calls] == ["context", "record", "finish"]
    assert all(c[1] == SCOPE for c in system.calls)
    svc.close()
    assert system.closed


def test_remember_turn_reports_retain_result():
    svc, _, _ = service()
    assert svc.remember_turn("u", "a") == {
        "episode_id": "ep1",
        "already_stored": False,
        "facts_extracted": 3,
    }


def test_search_memory_drops_trace(monkeypatch):
    svc, _, _ = service()
    import recall

    def tools(recaller, scope):
        assert scope == SCOPE
        return {"search_memory": lambda q, t, e: {"items": [], "context": q, "trace": ["x"]}}

    monkeypatch.setattr(recall, "make_recall_tool_functions", tools)
    assert svc.search_memory("q") == {"items": [], "context": "q"}


def test_server_exposes_memory_tools_without_scope_arguments():
    svc, _, _ = service()
    tools = asyncio.run(create_server(svc).list_tools())
    names = {tool.name for tool in tools}
    assert names == {
        "recall_context",
        "search_memory",
        "get_episode",
        "get_supporting",
        "remember_turn",
        "finish_session",
    }
    for tool in tools:
        params = set(tool.inputSchema.get("properties", {}))
        assert not params & {"user_id", "project_id", "session_id", "scope"}


def test_tool_call_reaches_the_service():
    svc, system, _ = service()
    asyncio.run(create_server(svc).call_tool("recall_context", {"query": "hello"}))
    assert system.calls[0][:3] == ("context", SCOPE, "hello")


def test_missing_configuration_is_a_tool_error_not_a_crash():
    def build():
        raise RuntimeError("Set PROJECT_ID")

    server = create_server(MemoryService(SCOPE, build))
    with pytest.raises(Exception, match="PROJECT_ID"):
        asyncio.run(server.call_tool("recall_context", {"query": "x"}))
