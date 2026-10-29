import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from dev_fixture import FixtureModels, MemoryBackend

from devtools.files import read_json
from devtools.worker import run_job

SOURCE = Path(__file__).resolve().parents[1]


def job():
    return {
        "run_id": "a" * 32,
        "budget": 2,
        "mode": "cached",
        "max_output_tokens": 1024,
        "config": {
            "project": "test-project",
            "region": "us-east1",
            "generation_location": "global",
        },
        "history": read_json(SOURCE / "fixtures/dev/history.json"),
        "questions": [{"id": "movie", "question": "Which movie did we pick?"}],
    }


def execute(tmp_path, request, output="run", fixture=None, source=SOURCE):
    fixture = fixture or FixtureModels()
    client = SimpleNamespace(models=fixture)
    result = run_job(
        request,
        source,
        tmp_path / "cache",
        tmp_path / output,
        client,
        client,
        backend_factory=MemoryBackend,
    )
    return result, fixture


def test_real_retain_reflect_recall_answer_and_prepared_replay(tmp_path):
    first, fixture = execute(tmp_path, job())
    assert first["status"] == "completed", first
    assert first["accounted_api_usd"] > 0
    assert not first["prepared_cache_hit"]
    answer = read_json(tmp_path / "run/answers/movie.json")
    assert answer["answer"] == "Inception"
    assert answer["context"]["recall"]["items"]
    assert any(
        r.get("config", {}).get("response_json_schema", {}).get("properties", {}).get("title")
        for r in fixture.requests
    )  # Real Reflect synthesized a macro through the cached SDK.
    count = len(fixture.requests)
    request = job()
    request["mode"] = "replay"
    second, _ = execute(tmp_path, request, "again", fixture)
    assert second["status"] == "completed", second
    assert second["prepared_cache_hit"] and second["accounted_api_usd"] == 0
    assert len(fixture.requests) == count


def test_changed_retain_source_reexecutes_pipeline_but_reuses_identical_provider_requests(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("tiered_memory.py", "valkey_buffer.py"):
        (source / name).write_text("source version 1")
    first, fixture = execute(tmp_path, job(), source=source)
    assert first["status"] == "completed"
    count = len(fixture.requests)
    (source / "tiered_memory.py").write_text("source version 2")
    second, _ = execute(tmp_path, job(), "changed", fixture, source)
    assert second["status"] == "completed", second
    assert not second["prepared_cache_hit"]
    assert second["accounted_api_usd"] == 0
    assert (tmp_path / "changed/retention/00000.json").exists()
    assert len(fixture.requests) == count


def test_replay_miss_and_budget_stop_are_visible_terminal_results(tmp_path):
    request = job()
    request["mode"] = "replay"
    replay, fixture = execute(tmp_path, request)
    assert replay["status"] == "cache_miss"
    assert replay["accounted_api_usd"] == 0 and not fixture.requests
    request["mode"], request["budget"] = "cached", 0.00001
    stopped, fixture = execute(tmp_path, request, "stopped")
    assert stopped["status"] == "budget_stopped"
    assert stopped["accounted_api_usd"] == 0 and not fixture.requests
    assert read_json(tmp_path / "stopped/status.json")["status"] == "budget_stopped"


def test_worker_rejects_scoring_labels_before_constructing_any_client_backend(tmp_path):
    request = job()
    request["questions"][0]["expected_answer"] = "SECRET_LABEL"
    with pytest.raises(ValueError, match="Scoring labels"):
        execute(tmp_path, request)


def test_missing_source_dates_do_not_create_future_facts(tmp_path):
    result, _ = execute(tmp_path, job())
    assert result["status"] == "completed"
    receipt = read_json(tmp_path / "run/retention/00000.json")
    for fact in receipt["batch"]["facts"]:
        assert fact["valid_from"] <= result["logical_clock"]
    prompt = json.dumps(read_json(tmp_path / "run/calls/00001.json"))
    assert "Source dates are unknown" in prompt
