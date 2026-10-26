"""Cross-component checks for the candidate application and lifecycle reads."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from recall_fixtures import NOW, SCOPE, Memories, embed
from test_offline import MemoryValkey
from test_recall_alloydb import FakeConnection, queries
from test_reflect import (
    OTHER_ENTITY,
    T0,
    USER,
    FakeEmbedder,
    FakeWriter,
    duckdb_story,
    fact,
    redundant_judge,
    seeded_store,
    similar,
    subject,
)
from test_retain_extraction import chunks, fact_row, gemini_returning

from bench.metering import BudgetStop, Ledger, Models
from recall import InMemoryRecallStore, Recaller
from recall.alloydb_store import AlloyDBRecallStore
from recall.config import MemoryType
from reflect import Reflector
from reflect.models import FactState, Tier
from reflect.store import InMemoryReflectStore
from retain import Scope
from retain.extraction import GeminiFactExtractor
from tiered_memory import TieredMemory
from valkey_buffer import SessionBuffer


def test_reflected_macro_and_merged_fact_reach_recall_without_original_duplicates():
    (s1, s3, s4), links = duckdb_story()
    duplicate = fact("dup", "User switched to DuckDB.", days=15)
    store = seeded_store(
        s1,
        s3,
        s4,
        duplicate,
        links=[*links, similar(s3, duplicate)],
        rows=subject(s3, duplicate, entity=OTHER_ENTITY),
    )
    reflector = Reflector(
        store=store,
        embedder=FakeEmbedder(),
        judge=redundant_judge,
        macro_writer=FakeWriter(),
        clock=lambda: T0,
    )
    result = reflector.reflect(USER)
    recaller = Recaller(
        store=InMemoryRecallStore(store.retain_store, store),
        reflect_store=store,
        embedder=FakeEmbedder(),
        clock=lambda: T0,
        strict=True,
    )
    recalled = recaller.recall(Scope(USER, "proj", "new"), "DuckDB pipeline", decompose=False)
    ids = {item.memory.id for item in recalled.items}
    assert result.batch.consolidations[0].merged.fact_id in ids
    assert s3.fact_id not in ids and duplicate.fact_id not in ids
    macro = store.user_macros(USER)[0]
    assert macro.macro_id in ids and macro.access_count == 1
    assert not recaller.recall(Scope("other"), "DuckDB pipeline", decompose=False).items


def test_archived_fallback_promotes_only_returned_facts_and_keeps_conflict_warning():
    m = Memories()
    row = m.fact("Favorite color is blue")
    lifecycle = InMemoryReflectStore(m.retain_store)
    lifecycle.states[row.fact_id] = FactState(
        row.fact_id, tier=Tier.ARCHIVED, conflicts_with=("other-fact",), confidence=0.7
    )
    recaller = Recaller(
        store=InMemoryRecallStore(m.retain_store, lifecycle),
        reflect_store=lifecycle,
        embedder=embed,
        clock=lambda: NOW,
        reranker=lambda q, texts: [1.0] * len(texts),
        strict=True,
    )
    result = recaller.recall(SCOPE, "Favorite color", decompose=False)
    assert len(result.items) == 1
    assert "other-fact" in result.context
    assert lifecycle.states[row.fact_id].tier == Tier.WARM
    assert lifecycle.states[row.fact_id].access_count == 1


def test_candidate_writes_share_round_id_and_retention_failure_is_visible():
    retainer = Mock()
    system = TieredMemory(
        retainer=retainer,
        recaller=Mock(),
        reflector=Mock(),
        buffer=SessionBuffer(MemoryValkey(), lambda p: "summary", strict=True),
        generate=lambda p: "answer",
    )
    system.record_turn(SCOPE, "question", "answer", timestamp=NOW, episode_id="stable-id")
    assert retainer.retain_turn.call_args.kwargs["episode_id"] == "stable-id"
    assert system.buffer.read(SCOPE.session_id).round_ids == ["stable-id"]
    retainer.retain_turn.side_effect = RuntimeError("storage unavailable")
    with pytest.raises(RuntimeError, match="storage unavailable"):
        system.record_turn(SCOPE, "next", "reply")


def test_reflect_enabled_sql_binds_tiers_and_includes_macros():
    conn = FakeConnection()
    store = AlloyDBRecallStore(lambda: conn, reflect_enabled=True)
    store.search(
        SCOPE, ["query"], [[0.1] * 768], None, list(MemoryType), NOW, False, 10, tiers=["hot"]
    )
    sql, params = queries(conn)[0]
    assert "macro_memories" in sql and "consolidated_into IS NULL" in sql
    assert params["tiers"] == ["hot"]


@pytest.mark.parametrize("known", [True, False])
def test_relative_day_counts_and_unknown_source_dates(known):
    chunk = chunks("The project starts in 3 days.")[0]
    row = fact_row(chunk, text=chunk.text, evidence=chunk.text, category="event", entities=[])
    row["details"]["when"] = "in 3 days"
    client = gemini_returning(json.dumps([row]))
    result = GeminiFactExtractor(client, source_dates_known=known).extract([chunk])
    assert len(result) == 1
    assert (result[0].occurred_at is not None) == known
    prompt = client.models.generate_content.call_args.kwargs["contents"]
    assert ('"timestamp": null' in prompt) != known


def test_budget_counts_schema_before_provider_call_and_survives_restart(tmp_path):
    ledger = Ledger(tmp_path, 1, 0)
    client = Mock()
    model = Models(client, client, ledger)
    with pytest.raises(BudgetStop):
        model.generate_content(
            model="gemini-3.5-flash",
            contents="tiny",
            config={"response_json_schema": {"description": "x" * 400000}},
        )
    client.models.generate_content.assert_not_called()
    ledger.reserve("uncertain-call", 0.8)
    restarted = Ledger(tmp_path, 1, 0)
    with pytest.raises(BudgetStop):
        restarted.reserve("retry", 0.3)


def test_budget_settles_thinking_and_persists_provider_receipt(tmp_path):
    ledger = Ledger(tmp_path, 1, 0)
    usage = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=50,
        thoughts_token_count=200,
        model_dump=lambda **k: {"thoughts_token_count": 200},
    )
    response = SimpleNamespace(
        usage_metadata=usage,
        text="ok",
        model_version="pinned",
        model_dump=lambda **k: {"text": "ok"},
    )
    client = Mock()
    client.models.generate_content.return_value = response
    Models(client, client, ledger).generate_content(model="gemini-3.5-flash", contents="hello")
    assert ledger.state["accounted_usd"] == pytest.approx(0.0024)
    assert (tmp_path / "calls/00001.json").exists()
