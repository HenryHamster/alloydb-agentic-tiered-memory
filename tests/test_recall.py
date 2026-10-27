"""Offline checks of Recall behaviour over Retain's in-memory tables."""

import inspect
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from recall_fixtures import NOW, SCOPE, Memories, embed, overlap_reranker
from test_retain import ExactResolver, NoLinks, ScriptedExtractor

from recall import MemoryType, RecallConfig, RecallDepth, TimeWindow, make_recall_tool_functions
from recall.models import Memory, Scored
from recall.query import analyze_query, parse_time_phrase
from recall.ranking import apply_boosts, fuse_rrf
from retain import Retainer, Scope
from retain.models import Visibility


def ids(result):
    return [item.memory.id for item in result.items]


@pytest.fixture
def memories():
    return Memories()


# Finding the right memories ----------------------------------------------------------------


def test_the_relevant_memory_ranks_above_unrelated_ones(memories):
    memories.fact("The office coffee machine is on the third floor")
    wanted = memories.fact("The team stores orders in a Postgres database")
    memories.fact("Quarterly planning happens every January")

    result = memories.recaller().recall(SCOPE, "Which database holds the orders?")
    assert ids(result)[0] == wanted.fact_id


@pytest.mark.parametrize(
    "visibility,writer,reader,visible",
    [
        (Visibility.GLOBAL, Scope("other_user", "project_a", "session_1"), SCOPE, False),
        (Visibility.GLOBAL, Scope("user_1", "project_b", "session_9"), SCOPE, True),
        (Visibility.PROJECT, Scope("user_1", "project_b", "session_9"), SCOPE, False),
        (Visibility.PROJECT, Scope("user_1", "project_a", "session_9"), SCOPE, True),
        (Visibility.SESSION, Scope("user_1", "project_a", "session_9"), SCOPE, False),
        (Visibility.SESSION, SCOPE, SCOPE, True),
    ],
)
def test_only_memories_the_reader_may_see_are_returned(
    memories, visibility, writer, reader, visible
):
    fact = memories.fact("Deploys need two approvals", scope=writer, visibility=visibility)
    result = memories.recaller().recall(reader, "deploy approvals", include_invalid=True)
    assert (fact.fact_id in ids(result)) == visible


def test_superseded_memories_are_hidden_unless_requested(memories):
    old = memories.fact("The team database is MySQL", valid_from=NOW - timedelta(days=400))
    new = memories.fact(
        "The team database is now Postgres", valid_from=NOW - timedelta(days=30), supersedes=old
    )
    recaller = memories.recaller()

    current = recaller.recall(SCOPE, "team database")
    assert new.fact_id in ids(current) and old.fact_id not in ids(current)

    history = recaller.recall(SCOPE, "team database", include_invalid=True)
    stale = {item.memory.id: item.stale for item in history.items}
    assert stale == {old.fact_id: True, new.fact_id: False}


def test_superseded_memories_do_not_crowd_out_current_ones(memories):
    previous = None
    for n in range(5):
        previous = memories.fact(
            f"Release owner is person {n} for the release train release",
            valid_from=NOW - timedelta(days=100 - n),
            supersedes=previous,
        )
    current = memories.fact("Release owner is Dana", supersedes=previous)
    small_pool = RecallConfig(pool_by_depth={depth: 2 for depth in RecallDepth})

    result = memories.recaller(config=small_pool).recall(SCOPE, "release train release owner")
    assert ids(result) == [current.fact_id]


def test_a_dated_question_prefers_memories_from_that_period(memories):
    recent = memories.fact(
        "We deployed the billing service", occurred_at=datetime(2026, 10, 8, tzinfo=timezone.utc)
    )
    memories.fact(
        "We deployed the billing service", occurred_at=datetime(2025, 10, 8, tzinfo=timezone.utc)
    )

    result = memories.recaller().recall(SCOPE, "what billing deploys happened last week")
    assert result.window is not None
    assert ids(result)[0] == recent.fact_id


def test_naming_a_known_entity_finds_facts_about_it(memories):
    falcon = memories.entity("Project Falcon", "falcon")
    fact = memories.fact("The launch moved to March", entity_ids=(falcon,))

    result = memories.recaller().recall(SCOPE, "any news on falcon?")
    assert fact.fact_id in ids(result)


def test_linked_memories_are_reachable_through_the_graph(memories):
    hit = memories.fact("Alice leads the payments team")
    cause = memories.fact("A reorg merged two groups in spring")
    query = "who leads payments?"

    assert cause.fact_id not in ids(memories.recaller().recall(SCOPE, query))
    memories.link(hit, cause)
    assert cause.fact_id in ids(memories.recaller().recall(SCOPE, query))


def test_memories_from_rounds_the_caller_already_shows_are_skipped(memories):
    shown = memories.fact("We picked Postgres for orders", episode_id="round-shown")
    older = memories.fact("Postgres replaced MySQL for orders", episode_id="round-older")

    result = memories.recaller().recall(
        SCOPE, "Postgres orders", exclude_episode_ids=["round-shown"]
    )
    assert older.fact_id in ids(result) and shown.fact_id not in ids(result)


def test_episodes_are_returned_only_when_requested(memories):
    memories.fact("User wants nightly backups", episode_id="round-7")
    recaller = memories.recaller()

    default = recaller.recall(SCOPE, "nightly backups")
    assert all(item.memory.type != MemoryType.EPISODE for item in default.items)

    with_episodes = recaller.recall(SCOPE, "nightly backups", types=["fact", "episode"])
    episodes = [item.memory for item in with_episodes.items if item.memory.type == "episode"]
    assert [episode.id for episode in episodes] == ["round-7"]
    assert "nightly backups" in episodes[0].text


def test_end_to_end_a_retained_turn_can_be_recalled(memories):
    retainer = Retainer(
        store=memories.retain_store,
        extractor=ScriptedExtractor(
            {"text": "User prefers dark mode in the editor", "evidence": "dark mode"}
        ),
        entity_resolver=ExactResolver(),
        embedder=embed,
        linker=NoLinks(),
        clock=lambda: NOW,
    )
    retainer.retain_turn(SCOPE, "I like dark mode in my editor", "Noted.", episode_id="round-1")

    result = memories.recaller().recall(SCOPE, "editor theme")
    assert [item.memory.source_episode_id for item in result.items] == ["round-1"]


# Ranking, abstention and packing -----------------------------------------------------------


def test_abstains_when_nothing_stored_is_relevant(memories):
    memories.fact("The team stores orders in a Postgres database")
    recaller = memories.recaller(reranker=overlap_reranker)

    unrelated = recaller.recall(SCOPE, "favourite holiday destination")
    assert unrelated.abstained and unrelated.items == [] and unrelated.context == ""
    assert not recaller.recall(SCOPE, "orders database").abstained


def test_low_reranker_scores_keep_the_best_few_instead_of_abstaining(memories):
    for n in range(5):
        memories.fact(f"Movie note {n}: the user rated a film")

    def low(query, texts):
        return [0.1] * len(texts)

    result = memories.recaller(reranker=low).recall(SCOPE, "favourite movie")
    assert len(result.items) == RecallConfig().floor_fallback_top and not result.abstained
    strict = memories.recaller(reranker=low, config=RecallConfig(floor_fallback_top=0))
    assert strict.recall(SCOPE, "favourite movie").abstained


@pytest.mark.parametrize("budget", [1, 15, 60, 10_000])
def test_packed_context_never_exceeds_the_token_budget(memories, budget):
    for n in range(20):
        memories.fact(f"Deployment pipeline note number {n} about release checks")
    recaller = memories.recaller()

    everything = recaller.recall(SCOPE, "deployment pipeline", max_tokens=10_000)
    result = recaller.recall(SCOPE, "deployment pipeline", max_tokens=budget)
    assert result.tokens_used == sum(item.tokens for item in result.items) <= budget
    assert result.truncated == (len(result.items) < len(everything.items))


def test_items_are_ranked_best_first_and_labelled_in_the_context(memories):
    for n in range(5):
        memories.fact(
            f"Release {n} shipped with the new search page", occurred_at=NOW - timedelta(days=n + 1)
        )

    result = memories.recaller().recall(SCOPE, "search page release")
    assert [item.rank for item in result.items] == list(range(1, len(result.items) + 1))
    scores = [item.score for item in result.items]
    assert scores == sorted(scores, reverse=True)
    lines = result.context.splitlines()
    for item, line in zip(result.items, lines, strict=True):
        assert item.memory.id in line and item.memory.primary_date().date().isoformat() in line


def test_boosts_cannot_overturn_a_clear_relevance_gap():
    config = RecallConfig()
    high = (1 + config.max_boost) / (1 - config.max_boost) * 0.5 + 0.01  # beats any boost on 0.5
    window = TimeWindow(NOW - timedelta(days=7), NOW)
    old = Memory("old", MemoryType.FACT, "old", occurred_at=NOW - timedelta(days=900))
    fresh = Memory("fresh", MemoryType.FACT, "fresh", occurred_at=NOW, evidence_count=10)
    for win in (None, window):
        results = apply_boosts([Scored(fresh, 0.5), Scored(old, high)], NOW, win, config)
        assert [item.memory.id for item in results] == ["old", "fresh"]
        assert all(abs(item.boost) <= config.max_boost for item in results)


def test_fusion_favours_memories_found_by_more_channels():
    a, b = (Memory(i, MemoryType.FACT, i) for i in "ab")
    fused = fuse_rrf({"vector:0": [a, b], "keyword:0": [b], "entity:0": [b]})
    assert [item.memory.id for item in fused] == ["b", "a"]


# Resilience and arguments ------------------------------------------------------------------


def test_keeps_answering_when_the_embedder_fails(memories):
    fact = memories.fact("Backups run nightly at two")

    def broken(texts):
        raise RuntimeError("quota")

    result = memories.recaller(embedder=broken).recall(SCOPE, "nightly backups")
    assert fact.fact_id in ids(result)


def test_keeps_answering_when_the_reranker_fails(memories):
    fact = memories.fact("Backups run nightly at two")
    reranker = MagicMock(side_effect=RuntimeError("ranker offline"))

    result = memories.recaller(reranker=reranker).recall(SCOPE, "nightly backups")
    assert fact.fact_id in ids(result)


@pytest.mark.parametrize(
    "kwargs", [{"depth": "deep"}, {"types": ["fact", "opinion"]}, {"max_tokens": 0}]
)
def test_rejects_invalid_arguments(memories, kwargs):
    with pytest.raises(ValueError):
        memories.recaller().recall(SCOPE, "anything", **kwargs)


def test_results_serialize_to_json(memories):
    memories.fact("Backups run nightly at two", occurred_at=NOW - timedelta(days=2))
    payload = json.loads(json.dumps(memories.recaller().recall(SCOPE, "backups").to_dict()))
    assert payload["items"] and payload["context"]


# Query analysis ----------------------------------------------------------------------------


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


def test_questions_without_a_time_phrase_get_no_window():
    assert parse_time_phrase("what database do we use", NOW) is None


def test_decomposition_keeps_the_original_question_and_drops_repeats():
    generate = MagicMock(
        return_value='{"sub_queries": ["backend stack", "Backend Stack", "timeout rules"],'
        ' "time_window": {"start": "2026-03-01", "end": "2026-04-01"}}'
    )
    plan = analyze_query("stack and timeouts before launch", NOW, generate=generate)
    assert plan.sub_queries[0] == "stack and timeouts before launch"
    assert sorted(q.lower() for q in plan.sub_queries[1:]) == ["backend stack", "timeout rules"]
    assert plan.window is not None and plan.window.start.month == 3


def test_decompose_false_never_calls_the_llm():
    generate = MagicMock()
    plan = analyze_query("what and why last month?", NOW, decompose=False, generate=generate)
    generate.assert_not_called()
    assert plan.sub_queries == ["what and why last month?"] and plan.window is not None


def test_an_explicit_window_wins_over_phrases_and_the_llm():
    explicit = TimeWindow(datetime(2025, 1, 1), datetime(2025, 2, 1))
    generate = MagicMock(
        return_value='{"sub_queries": [], "time_window": {"start": "2026-01-01", "end": "2026-02-01"}}'
    )
    assert (
        analyze_query("what happened yesterday", NOW, explicit, generate=generate).window
        == explicit
    )


def test_unusable_llm_output_falls_back_to_the_original_question():
    plan = analyze_query("anything", NOW, generate=lambda prompt: "not json")
    assert plan.sub_queries == ["anything"] and plan.window is None


# Agent tools -------------------------------------------------------------------------------


def test_tools_never_expose_scope_arguments(memories):
    tools = make_recall_tool_functions(memories.recaller(), SCOPE)
    for function in tools.values():
        params = set(inspect.signature(function).parameters)
        assert not params & {"user_id", "project_id", "session_id", "scope"}


def test_search_memory_accepts_a_date_range(memories):
    inside = memories.fact(
        "Shipped dark mode", occurred_at=datetime(2026, 1, 15, tzinfo=timezone.utc)
    )
    tools = make_recall_tool_functions(memories.recaller(), SCOPE)

    result = tools["search_memory"]("dark mode", time_range="2026-01-01/2026-01-31")
    assert result["window"]["start"].startswith("2026-01-01")
    assert result["window"]["end"].startswith("2026-02-01")  # the end date is inclusive
    assert inside.fact_id in [item["id"] for item in result["items"]]


def test_search_memory_reports_an_unusable_time_range(memories):
    tools = make_recall_tool_functions(memories.recaller(), SCOPE)
    result = tools["search_memory"]("anything", time_range="sometime")
    assert result["window"] is None and result["warnings"]


def test_get_episode_returns_the_original_round_only_to_its_owner(memories):
    memories.fact("User wants nightly backups", episode_id="round-7")
    mine = make_recall_tool_functions(memories.recaller(), SCOPE)
    theirs = make_recall_tool_functions(
        memories.recaller(), Scope("other_user", "project_a", "session_1")
    )

    episode = mine["get_episode"]("round-7")
    assert "nightly backups" in episode["messages"][0]["text"]
    assert theirs["get_episode"]("round-7")["error"] == "not_found"


def test_observations_lead_back_to_their_supporting_facts(memories):
    facts = [
        memories.fact("Alice reviews every schema change"),
        memories.fact("Alice owns the migration tooling"),
    ]
    observation = memories.observation("Alice is the database migration owner", facts)
    tools = make_recall_tool_functions(memories.recaller(), SCOPE)

    result = tools["search_memory"]("who owns database migrations?")
    assert observation.observation_id in [item["id"] for item in result["items"]]
    supporting = tools["get_supporting"](observation.observation_id)["facts"]
    assert sorted(f["id"] for f in supporting) == sorted(f.fact_id for f in facts)

    stranger = make_recall_tool_functions(memories.recaller(), Scope("other_user"))
    assert stranger["get_supporting"](observation.observation_id)["facts"] == []


def test_adk_tool_wrappers_build():
    from recall import make_recall_tools

    tools = make_recall_tools(Memories().recaller(), SCOPE)
    assert {tool.name for tool in tools} == {"search_memory", "get_episode", "get_supporting"}
