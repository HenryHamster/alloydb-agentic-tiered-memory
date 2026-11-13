"""Offline checks of Recall behaviour over Retain's in-memory tables."""

import inspect
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from recall_fixtures import NOW, SCOPE, Memories, embed, overlap_reranker
from test_retain import ExactResolver, NoLinks, ScriptedExtractor
from test_retain_extraction import gemini_returning

from recall import MemoryType, RecallConfig, RecallDepth, TimeWindow, make_recall_tool_functions
from recall.config import ROUND_RELATIONS
from recall.models import Memory, Scored
from recall.query import analyze_query, parse_time_phrase
from recall.ranking import apply_boosts, fuse_rrf
from retain import Retainer, Scope
from retain.extraction import GeminiFactExtractor
from retain.models import Relation, Speaker, Visibility


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


def test_explaining_links_are_followed_before_stronger_similar_links(memories):
    seed = memories.fact("The assistant restarted server psql-002")
    similar = memories.fact("Server psql-003 was restarted last month")
    request = memories.fact("The user asked to restart psql-002")
    memories.link(seed, similar, weight=0.9, relation=Relation.SIMILAR)
    memories.link(seed, request, weight=0.5, relation=Relation.EVIDENCED_BY)

    neighbors = memories.store.expand_neighbors(
        SCOPE, [seed.fact_id], [MemoryType.FACT], NOW, False, 5
    )
    assert [m.id for m in neighbors][:2] == [request.fact_id, similar.fact_id]
    only_round = memories.store.expand_neighbors(
        SCOPE, [seed.fact_id], [MemoryType.FACT], NOW, False, 5, relations=ROUND_RELATIONS
    )
    assert [m.id for m in only_round] == [request.fact_id]


def test_follow_up_adds_the_request_linked_to_a_kept_tool_call(memories):
    call = memories.fact('The assistant called the get_server_status tool with server = "psql-001"')
    request = memories.fact("The user requested to check the status of psql-001")
    memories.link(call, request, relation=Relation.EVIDENCED_BY)

    def names_psql(query, texts):
        return [0.8 if "server" in text else 0.4 if "psql-001" in text else 0.0 for text in texts]

    question = "how is that server doing?"  # shares no word with the request fact
    # graph_seeds=0 turns off the expansion before reranking, so only the follow-up can link.
    no_followup = RecallConfig(graph_seeds=0, followup_seeds=0)
    without = memories.recaller(reranker=names_psql, config=no_followup)
    assert request.fact_id not in ids(without.recall(SCOPE, question))
    with_followup = memories.recaller(reranker=names_psql, config=RecallConfig(graph_seeds=0))
    result = with_followup.recall(SCOPE, question)
    assert request.fact_id in ids(result)
    added = next(s for s in result.trace if s.stage == "graph_followup").to_dict()["added"]
    assert [entry["id"] for entry in added] == [request.fact_id]


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


def test_recall_reads_retains_tool_facts_ids_and_round_links(memories):
    # Retain's production extractor settings; the model returns no facts of its own, so every
    # fact below is built by Retain's code from the round itself.
    extractor = GeminiFactExtractor(
        gemini_returning("[]"), lean=True, tool_result_facts=True, link_rounds=True
    )
    retainer = Retainer(
        store=memories.retain_store,
        extractor=extractor,
        entity_resolver=ExactResolver(),
        embedder=embed,
        linker=NoLinks(),
        clock=lambda: NOW,
    )
    call = {"name": "get_server_status", "arguments": json.dumps({"server_id": "psql-001"})}
    events = [
        {"role": "assistant", "content": "Checking.", "tool_calls": [{"function": call}]},
        {"role": "tool", "content": json.dumps({"server_id": "psql-001", "status": "running"})},
    ]
    retainer.retain_turn(
        SCOPE, "Check the status of server psql-001.", json.dumps(events), episode_id="round-1"
    )
    facts = list(memories.retain_store.facts.values())
    request = next(f for f in facts if f.text.startswith("The user said:"))
    call_fact = next(f for f in facts if f.text.startswith("The assistant called"))
    result = next(f for f in facts if f.text.startswith("A tool returned:"))

    # Naming the ID finds all three facts through Retain's entity for psql-001.
    named = memories.recaller().recall(SCOPE, "psql-001")
    assert {request.fact_id, call_fact.fact_id, result.fact_id} <= set(ids(named))
    assert any("entity:0" in item.channels for item in named.items)

    # A question that only matches the result still brings in its tool call, one round
    # link away, through the follow-up step.
    def result_only(query, texts):
        return [0.9 if "returned" in t else 0.3 if "psql-001" in t else 0.0 for t in texts]

    vague = memories.recaller(reranker=result_only, config=RecallConfig(graph_seeds=0))
    found = vague.recall(SCOPE, "is it running?")
    assert {call_fact.fact_id, result.fact_id} <= set(ids(found))
    followup = next(s for s in found.trace if s.stage == "graph_followup").to_dict()
    assert [entry["id"] for entry in followup["added"]] == [call_fact.fact_id]


# Ranking, abstention and packing -----------------------------------------------------------


def test_abstains_when_nothing_stored_is_relevant(memories):
    memories.fact("The team stores orders in a Postgres database")
    recaller = memories.recaller(reranker=overlap_reranker)

    unrelated = recaller.recall(SCOPE, "favourite holiday destination")
    assert unrelated.abstained and unrelated.items == [] and unrelated.context == ""
    assert not recaller.recall(SCOPE, "orders database").abstained


def test_low_but_positive_reranker_scores_are_kept_up_to_max_items(memories):
    for n in range(12):
        memories.fact(f"Movie note {n}: the user rated a film")

    def low(query, texts):
        return [0.15] * len(texts)  # under the old 0.3 floor, over min_relevance

    result = memories.recaller(reranker=low).recall(SCOPE, "favourite movie")
    facts = [i for i in result.items if i.memory.type == MemoryType.FACT]
    assert len(facts) == RecallConfig().max_items and not result.abstained
    # Under min_relevance only the min_items floor survives; min_items=0 abstains.
    stricter = memories.recaller(reranker=low, config=RecallConfig(min_relevance=0.2))
    floor = [i for i in stricter.recall(SCOPE, "favourite movie").items if i.memory.type == "fact"]
    assert len(floor) == RecallConfig().min_items
    strict = RecallConfig(min_relevance=0.2, min_items=0)
    assert memories.recaller(reranker=low, config=strict).recall(SCOPE, "favourite movie").abstained


def test_weak_results_bring_their_verbatim_source_round(memories):
    fact = memories.fact("The user is organizing a film club event", episode_id="round-film")
    memories.fact("The user booked a dentist visit", episode_id="round-dentist")

    def weak(query, texts):
        return [0.07 if "film" in text else 0.01 for text in texts]

    result = memories.recaller(reranker=weak).recall(SCOPE, "the film we picked")
    rounds = [i for i in result.items if i.memory.type == MemoryType.EPISODE]
    assert fact.fact_id in ids(result)
    assert [r.memory.id for r in rounds][0] == "round-film"
    assert "film club event" in rounds[0].memory.text and "source_round" in rounds[0].channels

    def strong(query, texts):
        return [0.9 if "film" in text else 0.0 for text in texts]

    clear = memories.recaller(reranker=strong).recall(SCOPE, "film club event")
    assert all(i.memory.type != MemoryType.EPISODE for i in clear.items)


def test_items_record_the_search_lists_that_found_them(memories):
    fact = memories.fact("The team stores orders in a Postgres database")

    result = memories.recaller(reranker=overlap_reranker).recall(SCOPE, "orders database")
    item = next(item for item in result.items if item.memory.id == fact.fact_id)
    assert {"vector:0", "keyword:0"} <= set(item.channels)
    assert item.to_dict()["channels"] == list(item.channels)


def test_the_trace_lists_dropped_candidates_with_scores(memories):
    memories.fact("The team stores orders in a Postgres database")
    dropped = memories.fact("Orders ship from the Berlin warehouse")

    def scores(query, texts):
        return [0.05 if "Berlin" in text else 0.9 for text in texts]

    one = RecallConfig(min_items=1)
    result = memories.recaller(reranker=scores, config=one).recall(SCOPE, "orders database")
    stage = next(s for s in result.trace if s.stage == "filter_and_abstain").to_dict()
    assert stage["below_min"] == 1
    assert stage["dropped"][0]["id"] == dropped.fact_id
    assert stage["dropped"][0]["relevance"] == 0.05 and stage["dropped"][0]["channels"]


def five_pride_facts_a_talk_and_a_support_group(memories):
    """LoCoMo q032: five Pride facts filled the list, so the talk and support group were cut."""
    pride = [
        memories.fact(f"Caroline went to an LGBTQ pride parade, mention {n}") for n in range(5)
    ]
    for i, fact in enumerate(pride):
        for other in pride[i + 1 :]:
            memories.link(fact, other, relation=Relation.SIMILAR)
    talk = memories.fact("Caroline gave an LGBTQ talk at a school event")
    group = memories.fact("Caroline went to an LGBTQ support group")
    return pride, talk, group


def pride_first(query, texts):
    return [0.9 if "pride" in text else 0.5 for text in texts]


def test_near_repeats_leave_room_for_distinct_answers_to_a_list_question(memories):
    pride, talk, group = five_pride_facts_a_talk_and_a_support_group(memories)
    question = "What LGBTQ events has Caroline been to?"

    off = RecallConfig(max_items=4, similar_cap=0)
    crowded = memories.recaller(reranker=pride_first, config=off).recall(SCOPE, question)
    assert set(ids(crowded)) <= {f.fact_id for f in pride}

    on = RecallConfig(max_items=4)
    result = memories.recaller(reranker=pride_first, config=on).recall(SCOPE, question)
    assert {talk.fact_id, group.fact_id} <= set(ids(result))
    assert sum(f.fact_id in ids(result) for f in pride) == on.similar_cap
    stage = next(s for s in result.trace if s.stage == "filter_and_abstain").to_dict()
    assert stage["repeats"] == 3


def test_counting_questions_keep_every_near_repeat(memories):
    pride, _, _ = five_pride_facts_a_talk_and_a_support_group(memories)
    recaller = memories.recaller(reranker=pride_first, config=RecallConfig(max_items=5))

    result = recaller.recall(SCOPE, "How many pride parades has Caroline been to?")
    assert {f.fact_id for f in pride} <= set(ids(result))


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


@pytest.mark.parametrize(
    "question,counting",
    [
        ("How many times has Melanie gone to the beach?", True),
        ("How often does she paint?", True),
        ("What is the number of kids she has?", True),
        ("What LGBTQ events has Caroline been to?", False),
    ],
)
def test_counting_questions_are_recognised_without_the_llm(question, counting):
    assert analyze_query(question, NOW, decompose=False).counting is counting


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
    assert plan.refers_back is False


@pytest.mark.parametrize(
    "question", ["how is the server doing right now?", "what's the latest on X?"]
)
def test_now_and_latest_never_become_a_date_window(question):
    today = '{"sub_queries": [], "time_window": {"start": "2026-10-18", "end": "2026-10-19"}}'
    plan = analyze_query(question, NOW, generate=lambda prompt: today)
    assert plan.window is None and plan.window_source is None


def test_a_past_llm_window_is_still_used():
    past = '{"sub_queries": [], "time_window": {"start": "2026-09-01", "end": "2026-09-08"}}'
    plan = analyze_query("what did we decide before the launch?", NOW, generate=lambda p: past)
    assert plan.window_source == "llm" and plan.window.start.day == 1


def test_back_references_and_abbreviations_add_queries():
    reply = (
        '{"sub_queries": ["problems the user asked about", "artificial intelligence issues"],'
        ' "time_window": null, "refers_back": true}'
    )
    plan = analyze_query("how are those AI issues being tackled?", NOW, generate=lambda p: reply)
    assert plan.refers_back is True
    assert "artificial intelligence issues" in plan.sub_queries


@pytest.mark.parametrize("value", ["false", '"yes"', "null"])
def test_refers_back_is_false_unless_the_llm_says_true(value):
    reply = f'{{"sub_queries": [], "time_window": null, "refers_back": {value}}}'
    assert analyze_query("anything", NOW, generate=lambda p: reply).refers_back is False


def test_user_facts_get_the_request_boost_only_for_back_references():
    config = RecallConfig()
    user = Memory("user", MemoryType.FACT, "asked", occurred_at=NOW, speaker=Speaker.USER)
    other = Memory("other", MemoryType.FACT, "said", occurred_at=NOW, speaker=Speaker.ASSISTANT)
    scored = [Scored(user, 0.5), Scored(other, 0.5)]

    plain = {s.memory.id: s.boost for s in apply_boosts(scored, NOW, None, config)}
    back = {s.memory.id: s.boost for s in apply_boosts(scored, NOW, None, config, refers_back=True)}
    assert back["user"] == pytest.approx(plain["user"] + config.user_request_boost)
    assert back["other"] == plain["other"]


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
