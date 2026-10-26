"""Offline checks using real library imports and fake storage/model boundaries."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import redis
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from adk_guardrails import ADKPermissionGuardrail
from agent_orchestrator import build_agent_prompt
from async_worker import AsyncMemoryWorker
from enterprise_engine import AgentMemoryEngine
from valkey_buffer import (
    IN_MEMORY_VALKEY_FALLBACK,
    append_session_turn_with_rolling_summary,
    get_session_context_buffer,
)


class MemoryValkey:
    def __init__(self):
        self.values = {}
        self.events = []

    def rpush(self, key, value):
        self.values.setdefault(key, []).append(value.encode())

    def lrange(self, key, start, end):
        values = self.values.get(key, [])
        return values[start:] if end == -1 else values[start : end + 1]

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.events.append(("set", key))
        self.values[key] = value.encode()

    def ltrim(self, key, start, end):
        self.events.append(("trim", key))
        self.values[key] = self.lrange(key, start, end)

    def delete(self, key):
        self.values.pop(key, None)


def test_library_modules_import_without_cloud_connections():
    for module in [
        "db_clients",
        "valkey_buffer",
        "async_worker",
        "hybrid_retriever",
        "agent_orchestrator",
        "enterprise_engine",
        "adk_memory_provider",
        "adk_memory_tools",
        "adk_guardrails",
        "turn_context",
        "recall",
        "recall.alloydb_store",
        "retain",
        "reflect",
        "shared.chunking",
    ]:
        importlib.import_module(module)


def test_summary_is_saved_before_history_is_trimmed():
    store = MemoryValkey()
    llm = MagicMock()
    llm.models.generate_content.return_value.text = "Durable summary"
    for number in range(3):
        append_session_turn_with_rolling_summary(
            store,
            llm,
            "session",
            f"user {number}",
            f"answer {number}",
            trigger_limit=2,
            window_size=1,
        )

    prompt = llm.models.generate_content.call_args.kwargs["contents"]
    assert "user 0" in prompt and "user 1" in prompt
    assert "user 2" not in prompt
    assert [event[0] for event in store.events] == ["set", "trim"]
    context = get_session_context_buffer(store, "session")
    assert context == {
        "rolling_summary": "Durable summary",
        "recent_turns": [
            {"role": "user", "content": "user 2"},
            {"role": "assistant", "content": "answer 2"},
        ],
    }


def test_short_sessions_remain_uncompressed():
    store, llm = MemoryValkey(), MagicMock()
    append_session_turn_with_rolling_summary(store, llm, "short", "hello", "hi")
    assert len(get_session_context_buffer(store, "short")["recent_turns"]) == 2
    llm.models.generate_content.assert_not_called()


def test_storage_failure_has_an_in_process_fallback():
    IN_MEMORY_VALKEY_FALLBACK.clear()
    store, llm = MagicMock(), MagicMock()
    store.rpush.side_effect = redis.exceptions.ConnectionError("offline")
    store.get.side_effect = redis.exceptions.ConnectionError("offline")
    try:
        append_session_turn_with_rolling_summary(store, llm, "fallback", "hello", "hi")
        assert get_session_context_buffer(store, "fallback")["recent_turns"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    finally:
        IN_MEMORY_VALKEY_FALLBACK.clear()


def test_background_worker_processes_a_turn_and_closes_its_connection():
    conn, llm = MagicMock(), MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    llm.models.generate_content.return_value.text = json.dumps(
        [
            {
                "entity_name": "language",
                "project_id": "global",
                "scope": "global",
                "summary": "Python",
            }
        ]
    )
    worker = AsyncMemoryWorker(lambda: conn, llm)
    job = worker.enqueue_extraction("user", "session", "I prefer Python", "Okay", "prior")
    assert job.result(timeout=5) is None
    worker.close(timeout=5)
    assert not worker.worker_thread.is_alive(), "Worker failed to finish the queued exchange"
    assert cursor.execute.call_count == 1
    sql, params = cursor.execute.call_args.args
    assert "ON CONFLICT" in sql
    assert params == ("user", "language", "global", "session", "global", "Python")
    assert "prior" in llm.models.generate_content.call_args.kwargs["contents"]
    conn.close.assert_called_once()


@pytest.mark.parametrize(
    "rules,expected",
    [
        ([("pytest.*", "ALLOW", "project")], "ALLOW"),
        ([("pytest.*", "BLOCK", "project"), (".*", "ALLOW", "global")], "BLOCK"),
        ([], "PROMPT_USER"),
    ],
)
def test_permission_outcomes_with_project_rules_first(rules, expected):
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = rules
    engine = AgentMemoryEngine(MemoryValkey(), conn)
    action, _ = engine.evaluate_tool_permission(
        "user", "project", "session", "run_command", {"CommandLine": "pytest --timeout=30"}
    )
    assert action == expected


def test_adk_callback_blocks_execution_pending_approval():
    engine = MagicMock()
    engine.evaluate_tool_permission.return_value = ("PROMPT_USER", "No policy")
    guardrail = ADKPermissionGuardrail(engine)
    callback = guardrail.create_before_tool_callback("user", "project")
    result = callback(
        SimpleNamespace(name="run_command"),
        {"CommandLine": "echo hello"},
        SimpleNamespace(session=SimpleNamespace(id="session")),
    )
    assert result["status"] == "DENIED"


def test_adk_can_register_the_callback_and_function_tool():
    def fake_terminal(CommandLine: str) -> str:
        """Return a simulated tool response."""
        return CommandLine

    agent = Agent(
        name="offline_agent",
        model="gemini-3.5-flash",
        instruction="Demo",
        tools=[FunctionTool(fake_terminal)],
        before_tool_callback=ADKPermissionGuardrail(MagicMock()).create_before_tool_callback(
            "user"
        ),
    )
    assert agent.tools[0].name == "fake_terminal"
    assert callable(agent.before_tool_callback)


def test_agent_prompt_contains_all_context_components():
    prompt = build_agent_prompt(
        "Earlier goal",
        [{"role": "user", "content": "Recent message"}],
        {"language": {"summary": "Python"}},
        "Current question",
    )
    for component in ["Earlier goal", "Recent message", "language: Python", "Current question"]:
        assert component in prompt
