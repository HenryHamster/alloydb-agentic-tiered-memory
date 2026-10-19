"""Offline checks of the Recall pipeline and tools with a fake store and model boundaries."""

import inspect
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from recall import (
    MAX_BOOST,
    Memory,
    Scored,
    TimeWindow,
    analyze_query,
    apply_boosts,
    fuse_rrf,
    parse_time_phrase,
    recall,
)
from recall_tools import RecallDeps, make_recall_tool_functions, make_recall_tools

NOW = datetime(2026, 10, 18, 15, 30, tzinfo=timezone.utc)


def fact(memory_id, text=None, **fields):
    return Memory(memory_id, "fact", text or f"text of {memory_id}", **fields)


class FakeStore:
    def __init__(self, lists=None, neighbors=None):
        self.lists = lists or {}
        self.neighbors = neighbors or []
        self.search_calls = []
        self.expand_calls = []

    def search(self, scope, queries, embeddings, window, types, as_of, include_invalid, limit):
        self.search_calls.append(dict(locals()))
        return {name: list(results) for name, results in self.lists.items()}

    def expand_neighbors(self, scope, seed_ids, types, as_of, include_invalid, limit):
        self.expand_calls.append(dict(locals()))
        return list(self.neighbors)

    def get_episode(self, scope, episode_id):
        return {"episode_id": episode_id, "scope": scope.user_id} if episode_id == "ep1" else None

    def get_supporting(self, scope, observation_id):
        return [fact("f1"), fact("f2")]


def embed(texts):
    return [[0.1] * 3 for _ in texts]


def scores_by_id(table, default=0.0):
    def reranker(query, texts):
        return [table.get(text, default) for text in texts]

    return reranker


def run(store, **kwargs):
    kwargs.setdefault("query_time", NOW)
    kwargs.setdefault("embedder", embed)
    return recall(
        "user", "project", "session", kwargs.pop("query", "what stack?"), store=store, **kwargs
    )


def test_rrf_sums_reciprocal_ranks_and_ignores_repeats_within_a_list():
    a, b, c = fact("a"), fact("b"), fact("c")
    fused = fuse_rrf({"vector:0": [a, b, b], "keyword:0": [b, c]})
    assert [item.memory.id for item in fused] == ["b", "a", "c"]
    scores = {item.memory.id: item.relevance for item in fused}
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 62)


def test_abstains_when_every_reranked_score_is_below_the_floor():
    store = FakeStore({"vector:0": [fact("a"), fact("b")]})
    result = run(store, reranker=scores_by_id({}, default=0.1))
    assert result.abstained and result.items == [] and result.context == ""


def test_floor_uses_relevance_before_boosts():
    recent = fact("a", "recent", occurred_at=NOW, evidence_count=5)
    store = FakeStore({"vector:0": [recent]})
    result = run(store, reranker=scores_by_id({"recent": 0.28}), score_floor=0.3)
    assert result.abstained


def test_unreranked_results_are_not_abstained_on_score():
    store = FakeStore({f"vector:{i}": [fact(f"m{i}")] for i in range(12)})
    result = run(store)
    assert not result.abstained and len(result.items) == 12
    rerank_stage = next(stage for stage in result.trace if stage.stage == "rerank")
    assert rerank_stage.detail["reranked"] is False


def test_superseded_facts_hidden_by_default_and_flagged_when_requested():
    old = fact(
        "old", "used MySQL", valid_from=NOW - timedelta(days=400), valid_to=NOW - timedelta(days=30)
    )
    new = fact("new", "uses Postgres", valid_from=NOW - timedelta(days=30))
    store = FakeStore({"vector:0": [old, new]})
    reranker = scores_by_id({"used MySQL": 0.9, "uses Postgres": 0.8})

    hidden = run(store, reranker=reranker)
    assert [item.memory.id for item in hidden.items] == ["new"]
    assert store.search_calls[-1]["include_invalid"] is False

    shown = run(store, reranker=reranker, include_invalid=True)
    stale = {item.memory.id: item.stale for item in shown.items}
    assert stale == {"old": True, "new": False}
    assert store.search_calls[-1]["include_invalid"] is True
    assert "superseded 2026-09-18" in shown.context


def test_boosts_are_capped_and_cannot_overturn_a_clear_relevance_gap():
    window = TimeWindow(NOW - timedelta(days=7), NOW)
    relevant = fact("relevant", occurred_at=NOW - timedelta(days=900))
    boosted = fact("boosted", occurred_at=NOW - timedelta(days=1), evidence_count=9)
    results = apply_boosts([Scored(relevant, 0.9), Scored(boosted, 0.5)], NOW, window)
    assert [item.memory.id for item in results] == ["relevant", "boosted"]
    assert all(-MAX_BOOST <= item.boost <= MAX_BOOST for item in results)
    assert results[0].boost < 0 < results[1].boost


def test_recency_boost_is_skipped_when_the_question_has_a_window():
    old = fact("old", occurred_at=NOW - timedelta(days=3000))
    [no_window] = apply_boosts([Scored(old, 1.0)], NOW, None)
    window = TimeWindow(NOW - timedelta(days=3100), NOW - timedelta(days=2900))
    [in_window] = apply_boosts([Scored(old, 1.0)], NOW, window)
    assert no_window.boost < 0 and in_window.boost > 0


def test_pack_fills_the_budget_and_reports_truncation():
    store = FakeStore({"vector:0": [fact(f"m{i}") for i in range(5)]})
    result = run(store, max_tokens=25, count_tokens=lambda line: 10)
    assert len(result.items) == 2 and result.tokens_used == 20 and result.truncated
    assert [item.rank for item in result.items] == [1, 2]
    assert result.context.count("\n") == 1


def test_result_items_carry_structured_fields_and_serialize_to_json():
    memory = fact(
        "f1",
        "Alice prefers Python",
        occurred_at=NOW - timedelta(days=2),
        speaker="user",
        claim_type="preference",
        source_episode_id="ep1",
        entity_ids=("alice",),
    )
    result = run(FakeStore({"vector:0": [memory]}))
    payload = json.loads(json.dumps(result.to_dict()))
    item = payload["items"][0]
    for key in [
        "id", "type", "text", "dates", "score", "rank", "speaker", "claim_type",
        "stale", "source_episode_id", "supporting_ids", "entity_ids",
    ]:  # fmt: skip
        assert key in item
    assert payload["context"] == "[f1] (fact | 2026-10-16 | user | preference) Alice prefers Python"
    assert [stage["stage"] for stage in payload["trace"]] == [
        "analyze_query", "embed", "search", "graph_expand", "fuse_rrf",
        "rerank", "apply_boosts", "filter_and_abstain", "pack",
    ]  # fmt: skip


def test_decompose_false_never_calls_the_llm():
    generate = MagicMock()
    plan = analyze_query(
        "what did I use last month and why?", NOW, decompose=False, generate=generate
    )
    generate.assert_not_called()
    assert plan.sub_queries == ["what did I use last month and why?"]
    assert plan.window == TimeWindow(
        datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 10, 1, tzinfo=timezone.utc)
    )


def test_llm_decomposition_keeps_the_original_and_supplies_a_window():
    generate = MagicMock(
        return_value='```json\n{"sub_queries": ["backend stack", "timeout rules", "Backend Stack"],'
        ' "time_window": {"start": "2026-03-01", "end": "2026-04-01"}}\n```'
    )
    plan = analyze_query("stack and timeouts before launch", NOW, generate=generate)
    assert plan.sub_queries == [
        "stack and timeouts before launch",
        "backend stack",
        "timeout rules",
    ]
    assert plan.window_source == "llm" and plan.window.start.month == 3


def test_explicit_window_overrides_parsed_phrases():
    explicit = TimeWindow(datetime(2025, 1, 1), datetime(2025, 2, 1))
    generate = MagicMock(
        return_value='{"sub_queries": [], "time_window": {"start": "2026-01-01", "end": "2026-02-01"}}'
    )
    plan = analyze_query("what happened yesterday", NOW, explicit, generate=generate)
    assert plan.window == explicit and plan.window_source == "explicit"


def test_bad_llm_output_falls_back_to_the_original_query():
    plan = analyze_query("anything", NOW, generate=lambda prompt: "not json")
    assert plan.sub_queries == ["anything"] and plan.window is None


@pytest.mark.parametrize(
    "phrase,start,end",
    [
        ("what did I do yesterday", datetime(2026, 10, 17), datetime(2026, 10, 18)),
        ("notes from last week", datetime(2026, 10, 5), datetime(2026, 10, 12)),
        ("decisions last month", datetime(2026, 9, 1), datetime(2026, 10, 1)),
        ("what changed in 2025", datetime(2025, 1, 1), datetime(2026, 1, 1)),
        ("what did we ship in november", datetime(2025, 11, 1), datetime(2025, 12, 1)),
        ("past 3 days", NOW - timedelta(days=3), NOW),
    ],
)
def test_time_phrases_resolve_to_windows(phrase, start, end):
    assert parse_time_phrase(phrase, NOW) == TimeWindow(start, end)


def test_graph_expansion_seeds_from_fused_hits_and_adds_neighbors():
    neighbor = fact("linked", "linked fact")
    store = FakeStore({"vector:0": [fact("a"), fact("b")]}, neighbors=[neighbor])
    result = run(store)
    assert store.expand_calls[0]["seed_ids"] == ["a", "b"]
    assert "linked" in [item.memory.id for item in result.items]


def test_embedding_failure_skips_the_vector_channel_instead_of_using_a_zero_vector():
    def broken(texts):
        raise RuntimeError("quota")

    store = FakeStore({"keyword:0": [fact("a")]})
    result = run(store, embedder=broken)
    assert store.search_calls[0]["embeddings"] is None
    assert [item.memory.id for item in result.items] == ["a"]
    embed_stage = next(stage for stage in result.trace if stage.stage == "embed")
    assert "quota" in embed_stage.detail["error"]


def test_reranker_sees_at_most_the_rerank_pool_and_scores_against_each_sub_query():
    memories = [fact(f"m{i}") for i in range(200)]
    seen = []

    def reranker(query, texts):
        seen.append((query, len(texts)))
        return [0.9 if query == "second" and text == "text of m149" else 0.5 for text in texts]

    generate = lambda prompt: '{"sub_queries": ["second"]}'  # noqa: E731
    result = run(
        FakeStore({"vector:0": memories}), depth="mid", reranker=reranker, generate=generate
    )
    assert seen == [("what stack?", 150), ("second", 150)]
    assert result.items[0].memory.id == "m149"
    assert len(result.items) == 30


def test_depth_sets_the_candidate_pool_and_rejects_unknown_values():
    store = FakeStore()
    run(store, depth="high")
    assert store.search_calls[0]["limit"] == 1000
    with pytest.raises(ValueError):
        run(store, depth="deep")
    with pytest.raises(ValueError):
        run(store, types=["fact", "opinion"])


def test_search_memory_hides_scope_and_presets_a_deeper_recall():
    store = FakeStore({"vector:0": [fact("a")]})
    tools = make_recall_tool_functions(RecallDeps(store, embed), "user", "project", "session")
    for function in tools.values():
        params = set(inspect.signature(function).parameters)
        assert not params & {"user_id", "project_id", "session_id", "scope"}
    assert list(inspect.signature(tools["search_memory"]).parameters) == [
        "query",
        "time_range",
        "include_episodes",
    ]

    result = tools["search_memory"](
        "stack", time_range="2026-01-01/2026-01-31", include_episodes=True
    )
    call = store.search_calls[0]
    assert call["limit"] == 300 and result["max_tokens"] == 4096
    assert "episode" in call["types"] and call["scope"].user_id == "user"
    assert result["window"] == {
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-02-01T00:00:00+00:00",
    }


def test_search_memory_reports_an_unusable_time_range():
    store = FakeStore()
    tools = make_recall_tool_functions(RecallDeps(store, embed), "user", None, None)
    result = tools["search_memory"]("stack", time_range="sometime")
    assert store.search_calls[0]["window"] is None
    assert "time_range ignored" in result["warnings"][0]


def test_lookup_tools_use_the_bound_scope():
    tools = make_recall_tool_functions(RecallDeps(FakeStore(), embed), "user", "project", "session")
    assert tools["get_episode"]("ep1") == {"episode_id": "ep1", "scope": "user"}
    assert tools["get_episode"]("missing")["error"] == "not_found"
    assert [f["id"] for f in tools["get_supporting"]("obs1")["facts"]] == ["f1", "f2"]


def test_adk_tool_names():
    tools = make_recall_tools(RecallDeps(FakeStore(), embed), "user", "project", "session")
    assert [tool.name for tool in tools] == ["search_memory", "get_episode", "get_supporting"]
