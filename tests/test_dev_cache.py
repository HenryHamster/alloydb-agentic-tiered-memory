from unittest.mock import Mock

import pytest
from google.genai import types

from devtools.cache import CachedCursor, CachedModels, CacheMiss, RequestCache, Usage
from devtools.files import read_json, save_json


def model(tmp_path, mode="cached", run="run"):
    usage = Usage(tmp_path / run)
    cache = RequestCache(tmp_path / "cache", usage, mode=mode)
    provider = Mock()
    provider.models.generate_content.return_value = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="hello")]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=4, thoughts_token_count=7
        ),
    )
    provider.models.embed_content.return_value = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.1] * 768)]
    )
    return CachedModels(provider, provider, cache), usage, provider


def test_provider_cache_replay_preserves_responses_and_reports_usage(tmp_path):
    first, usage, provider = model(tmp_path)
    # Cache keys contain model names; the cache imposes no pricing/model allowlist.
    request = dict(model="developer-selected-model", contents="hello")
    assert first.generate_content(**request).text == "hello"
    first.embed_content(model="developer-selected-embedding", contents=["hello"])
    assert usage.calls[0]["usage_metadata"]["thoughts_token_count"] == 7
    assert all(c["status"] == "completed" and c["seconds"] >= 0 for c in usage.calls)
    replay, recorded, untouched = model(tmp_path, "replay", "second")
    assert replay.generate_content(**request).text == "hello"
    assert (
        replay.embed_content(model="developer-selected-embedding", contents=["hello"])
        .embeddings[0]
        .values
        == [0.1] * 768
    )
    assert not recorded.calls and len(recorded.hits) == 2
    untouched.models.generate_content.assert_not_called()
    untouched.models.embed_content.assert_not_called()
    assert provider.models.generate_content.call_count == 1
    with pytest.raises(CacheMiss):
        replay.generate_content(**{**request, "contents": "different"})


def test_config_invalidation_and_fresh_preserve_previous_receipt(tmp_path):
    first, _, provider = model(tmp_path)
    request = dict(model="model", contents="hello")
    first.generate_content(**request)
    replay, _, _ = model(tmp_path, "replay", "second")
    with pytest.raises(CacheMiss):
        replay.generate_content(**request, config={"temperature": 0.1})
    fresh, _, provider = model(tmp_path, "fresh", "third")
    provider.models.generate_content.return_value.candidates[0].content.parts[0].text = "new"
    assert fresh.generate_content(**request).text == "new"
    assert replay.generate_content(**request).text == "hello"
    path = next((tmp_path / "cache/generate").glob("*.json"))
    entry = read_json(path)
    entry["response"]["model_version"] = "tampered"
    save_json(path, entry)
    with pytest.raises(ValueError, match="integrity"):
        replay.generate_content(**request)


def test_failed_call_is_recorded_and_not_cached(tmp_path):
    first, usage, provider = model(tmp_path)
    provider.models.generate_content.side_effect = TimeoutError("unknown completion")
    with pytest.raises(TimeoutError):
        first.generate_content(model="model", contents="hello")
    assert usage.calls[0]["status"] == "failed"
    assert usage.calls[0]["error_type"] == "TimeoutError"
    replay, _, untouched = model(tmp_path, "replay", "second")
    with pytest.raises(CacheMiss):
        replay.generate_content(model="model", contents="hello")
    untouched.models.generate_content.assert_not_called()


def test_ranking_cache_preserves_raw_scores_and_sql_passthrough(tmp_path):
    usage = Usage(tmp_path / "run")
    cache = RequestCache(tmp_path / "cache", usage)
    inner = Mock()
    inner.fetchall.return_value = [(2, -0.2), (1, 0.6)]
    cursor = CachedCursor(inner, cache)
    query = "SELECT index, score FROM ai.rank(%s,%s,%s)"
    params = ("developer-selected-ranker", "sky", ["a", "b"])
    cursor.execute(query, params)
    assert cursor.fetchall() == [(2, -0.2), (1, 0.6)]
    replay = CachedCursor(Mock(), RequestCache(tmp_path / "cache", usage, mode="replay"))
    replay.execute(query, params)
    assert replay.fetchall() == cursor.fetchall()
    replay.inner.execute.assert_not_called()
    replay.execute("SELECT 1")
    replay.inner.fetchall.return_value = [(1,)]
    assert replay.fetchall() == [(1,)]
