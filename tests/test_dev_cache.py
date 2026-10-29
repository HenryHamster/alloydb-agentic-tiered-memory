import json
from unittest.mock import Mock

import pytest
from google.genai import types

from devtools.cache import Budget, BudgetStop, CachedCursor, CachedModels, CacheMiss, RequestCache


def generated(text="hello"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=4, thoughts_token_count=7
        ),
        model_version="test-version",
    )


def model(tmp_path, *, mode="cached", limit=1, output=1024, generation=None, run="run"):
    budget = Budget(tmp_path / run, limit)
    cache = RequestCache(tmp_path / "cache", budget, mode=mode)
    client = generation or Mock()
    client.models.generate_content.return_value = generated()
    embedding = Mock()
    embedding.models.embed_content.return_value = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.1] * 768)]
    )
    return (
        CachedModels(client, embedding, cache, max_output_tokens=output),
        budget,
        client,
        embedding,
    )


def test_cache_reuses_real_sdk_response_and_embedding_vectors_without_new_charges(tmp_path):
    first, paid, client, embedder = model(tmp_path)
    assert first.generate_content(model="gemini-3.5-flash", contents="hello").text == "hello"
    assert (
        first.embed_content(model="text-embedding-005", contents=["hello"]).embeddings[0].values
        == [0.1] * 768
    )
    assert len(paid.calls) == 2 and paid.accounted > 0
    second, replay, _, _ = model(tmp_path, mode="replay", run="second")
    assert second.generate_content(model="gemini-3.5-flash", contents="hello").text == "hello"
    assert (
        second.embed_content(model="text-embedding-005", contents=["hello"]).embeddings[0].values
        == [0.1] * 768
    )
    assert replay.accounted == 0 and len(replay.hits) == 2
    assert client.models.generate_content.call_count == 1
    assert embedder.models.embed_content.call_count == 1


def test_budget_stop_precedes_provider_and_default_output_bound_is_conservative(tmp_path):
    client, budget, provider, _ = model(tmp_path, limit=0.1, output=65536)
    with pytest.raises(BudgetStop):
        client.generate_content(model="gemini-3.5-flash", contents="hello")
    provider.models.generate_content.assert_not_called()
    assert budget.accounted == 0


def test_transport_failure_keeps_reserve_and_retry_cannot_reset_allowance(tmp_path):
    client, budget, provider, _ = model(tmp_path, limit=0.02)
    provider.models.generate_content.side_effect = TimeoutError("unknown completion")
    with pytest.raises(TimeoutError):
        client.generate_content(model="gemini-3.5-flash", contents="hello")
    reserved = budget.accounted
    assert reserved > 0.01
    restored = Budget(tmp_path / "run", 0.02)
    assert restored.accounted == reserved
    with pytest.raises(BudgetStop):
        restored.reserve("generate", reserved, "again")
    with pytest.raises(ValueError):
        Budget(tmp_path / "run", 2)


def test_nonbillable_rejection_is_not_cached_and_replay_never_falls_through(tmp_path):
    client, budget, provider, _ = model(tmp_path)
    error = RuntimeError("quota")
    error.code = 429
    provider.models.generate_content.side_effect = error
    with pytest.raises(RuntimeError):
        client.generate_content(model="gemini-3.5-flash", contents="hello")
    assert budget.accounted == 0
    replay, _, untouched, _ = model(tmp_path, mode="replay", run="second")
    with pytest.raises(CacheMiss):
        replay.generate_content(model="gemini-3.5-flash", contents="hello")
    untouched.models.generate_content.assert_not_called()


def test_request_config_changes_miss_and_fresh_does_not_overwrite_previous_receipt(tmp_path):
    client, _, _, _ = model(tmp_path)
    client.generate_content(model="gemini-3.5-flash", contents="hello")
    replay, _, _, _ = model(tmp_path, mode="replay", run="second")
    with pytest.raises(CacheMiss):
        replay.generate_content(
            model="gemini-3.5-flash", contents="hello", config={"temperature": 0.1}
        )
    fresh, fresh_budget, provider, _ = model(tmp_path, mode="fresh", run="third")
    provider.models.generate_content.return_value = generated("different realization")
    assert (
        fresh.generate_content(model="gemini-3.5-flash", contents="hello").text
        == "different realization"
    )
    assert fresh_budget.accounted > 0
    assert replay.generate_content(model="gemini-3.5-flash", contents="hello").text == "hello"


def test_corrupt_cache_refuses_reuse(tmp_path):
    client, _, _, _ = model(tmp_path)
    client.generate_content(model="gemini-3.5-flash", contents="hello")
    path = next((tmp_path / "cache/generate").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["response"]["model_version"] = "tampered"
    path.write_text(json.dumps(payload))
    replay, _, _, _ = model(tmp_path, mode="replay", run="second")
    with pytest.raises(ValueError, match="integrity"):
        replay.generate_content(model="gemini-3.5-flash", contents="hello")


def test_ranking_replays_rows_and_preserves_nonranking_cursor_behavior(tmp_path):
    budget = Budget(tmp_path / "run", 1)
    cache = RequestCache(tmp_path / "cache", budget)
    inner = Mock()
    inner.fetchall.return_value = [(2, -0.2), (1, 0.6)]
    cursor = CachedCursor(inner, cache)
    query, params = (
        "SELECT index, score FROM ai.rank(%s,%s,%s)",
        ("semantic-ranker-default-003", "sky", ["a", "b"]),
    )
    cursor.execute(query, params)
    assert cursor.fetchall() == [(2, -0.2), (1, 0.6)]
    assert budget.accounted == 0.001
    other = CachedCursor(Mock(), RequestCache(tmp_path / "cache", budget, mode="replay"))
    other.execute(query, params)
    assert other.fetchall() == cursor.fetchall()
    other.inner.execute.assert_not_called()
    other.execute("SELECT 1")
    other.inner.fetchall.return_value = [(1,)]
    assert other.fetchall() == [(1,)]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_api_allowances_rejected(tmp_path, value):
    with pytest.raises(ValueError):
        Budget(tmp_path, value)


def test_multiple_candidates_are_rejected_before_a_paid_call(tmp_path):
    client, _, provider, _ = model(tmp_path)
    with pytest.raises(ValueError, match="one candidate"):
        client.generate_content(
            model="gemini-3.5-flash", contents="hello", config={"candidate_count": 2}
        )
    provider.models.generate_content.assert_not_called()
