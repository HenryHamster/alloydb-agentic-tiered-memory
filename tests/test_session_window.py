"""Offline checks of the Valkey sliding window and how it feeds Recall."""

import json
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
import redis
from test_offline import MemoryValkey
from test_recall import NOW, FakeStore, embed, fact

from recall import Memory
from recall_tools import RecallDeps
from turn_context import MIN_RECALL_TOKENS, build_turn_context
from valkey_buffer import (
    IN_MEMORY_VALKEY_FALLBACK,
    SessionBuffer,
    get_session_context_buffer,
)

TURNS_KEY = "session:{s}:turns"


@pytest.fixture(autouse=True)
def clear_fallback():
    IN_MEMORY_VALKEY_FALLBACK.clear()
    yield
    IN_MEMORY_VALKEY_FALLBACK.clear()


def summarizer(text="summary"):
    return MagicMock(return_value=text)


def test_each_round_gets_one_id_and_timestamp():
    buffer = SessionBuffer(MemoryValkey(), summarizer())
    first = buffer.append("s", "hello", "hi", at=NOW)
    second = buffer.append("s", "next", "ok", at=NOW + timedelta(minutes=1))

    window = buffer.read("s")
    assert window.round_ids == [first, second]
    assert [turn.round_id for turn in window.turns] == [first, first, second, second]
    assert window.turns[2].at == NOW + timedelta(minutes=1)
    assert not window.degraded


def test_compaction_summarizes_old_rounds_then_keeps_the_window():
    generate = summarizer("rolled up")
    buffer = SessionBuffer(MemoryValkey(), generate, trigger_limit=2, window_size=1)
    ids = [buffer.append("s", f"user {n}", f"answer {n}") for n in range(3)]

    prompt = generate.call_args.args[0]
    assert "user 0" in prompt and "answer 1" in prompt and "user 2" not in prompt
    window = buffer.read("s")
    assert window.summary == "rolled up"
    assert window.round_ids == [ids[2]]


def test_trim_removes_only_what_was_summarized():
    """A turn appended while the summary was being written must survive the trim."""
    store = MemoryValkey()

    def generate(prompt):
        store.rpush(TURNS_KEY, json.dumps({"role": "user", "content": "late arrival"}))
        return "summary"

    buffer = SessionBuffer(store, generate, trigger_limit=2, window_size=1)
    for n in range(3):
        buffer.append("s", f"user {n}", f"answer {n}")

    assert [turn.content for turn in buffer.read("s").turns] == [
        "user 2",
        "answer 2",
        "late arrival",
    ]


def test_window_size_zero_trims_everything_after_summarizing():
    buffer = SessionBuffer(MemoryValkey(), summarizer(), trigger_limit=1, window_size=0)
    buffer.append("s", "a", "b")
    buffer.append("s", "c", "d")
    window = buffer.read("s")
    assert window.turns == [] and window.summary == "summary"


def test_failed_summary_keeps_turns_and_does_not_fall_back():
    generate = MagicMock(side_effect=RuntimeError("model down"))
    buffer = SessionBuffer(MemoryValkey(), generate, trigger_limit=1, window_size=0)
    buffer.append("s", "a", "b")
    buffer.append("s", "c", "d")

    window = buffer.read("s")
    assert [turn.content for turn in window.turns] == ["a", "b", "c", "d"]
    assert window.summary == ""
    assert IN_MEMORY_VALKEY_FALLBACK == {}


def test_valkey_outage_uses_the_fallback_with_the_same_compaction():
    store = MagicMock()
    store.rpush.side_effect = redis.exceptions.ConnectionError("offline")
    store.get.side_effect = redis.exceptions.ConnectionError("offline")
    buffer = SessionBuffer(store, summarizer("kept"), trigger_limit=1, window_size=1)
    buffer.append("s", "a", "b")
    last = buffer.append("s", "c", "d")

    window = buffer.read("s")
    assert window.degraded
    assert window.summary == "kept"
    assert window.round_ids == [last]


def test_turns_stored_before_round_ids_still_load():
    store = MemoryValkey()
    store.rpush(TURNS_KEY, json.dumps({"role": "user", "content": "old"}))
    store.rpush(TURNS_KEY, "not json")
    window = SessionBuffer(store).read("s")
    assert [turn.content for turn in window.turns] == ["old"]
    assert window.round_ids == []
    assert get_session_context_buffer(store, "s")["recent_turns"] == [
        {"role": "user", "content": "old"}
    ]


def test_recall_skips_memories_from_rounds_still_in_the_window():
    buffer = SessionBuffer(MemoryValkey(), summarizer())
    round_id = buffer.append("s", "We picked Postgres", "Noted", at=NOW)
    in_window = fact("f1", source_episode_id=round_id)
    episode = Memory(round_id, "episode", "We picked Postgres / Noted")
    older = fact("f2", source_episode_id="older-round")
    store = FakeStore({"vector:0": [in_window, episode, older]})

    context = build_turn_context(
        buffer, RecallDeps(store, embed), "user", "project", "s", "db?", query_time=NOW
    )

    assert [item.memory.id for item in context.recall.items] == ["f2"]
    fuse = next(stage for stage in context.recall.trace if stage.stage == "fuse_rrf")
    assert fuse.detail["in_window"] == 2


def test_window_is_charged_to_the_budget_before_recall():
    buffer = SessionBuffer(MemoryValkey(), summarizer())
    buffer.append("s", "x" * 400, "y" * 400, at=NOW)
    deps = RecallDeps(FakeStore({"vector:0": [fact("f1")]}), embed)

    context = build_turn_context(
        buffer, deps, "user", None, "s", "q", query_time=NOW, max_tokens=1000
    )
    assert context.window_tokens > 200
    assert context.recall_budget == 1000 - context.window_tokens
    assert context.recall.max_tokens == context.recall_budget

    tight = build_turn_context(buffer, deps, "user", None, "s", "q", query_time=NOW, max_tokens=10)
    assert tight.recall_budget == MIN_RECALL_TOKENS


def test_new_session_carries_the_previous_summary():
    buffer = SessionBuffer(MemoryValkey(), summarizer("old goals"), trigger_limit=1, window_size=0)
    buffer.append("prev", "a", "b")
    buffer.append("prev", "c", "d")

    context = build_turn_context(
        buffer,
        RecallDeps(FakeStore(), embed),
        "user",
        None,
        "new",
        "q",
        prev_session_id="prev",
        query_time=NOW,
    )
    assert context.summary == "[From prior session prev]:\nold goals"
    assert "old goals" in context.format()


def test_recall_failure_leaves_the_window_context():
    buffer = SessionBuffer(MemoryValkey(), summarizer())
    buffer.append("s", "hello", "hi", at=NOW)
    store = FakeStore()
    store.search = MagicMock(side_effect=RuntimeError("db down"))

    context = build_turn_context(buffer, RecallDeps(store, embed), "user", None, "s", "q")
    assert context.recall is None
    text = context.format()
    assert "No relevant long-term memories." in text and "user: hello" in text
