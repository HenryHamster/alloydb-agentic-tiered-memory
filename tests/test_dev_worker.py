from pathlib import Path
from types import SimpleNamespace

from dev_fixture import FixtureModels, MemoryBackend

from devtools.files import read_json
from devtools.worker import run_job

SOURCE = Path(__file__).resolve().parents[1]


def job():
    return {
        "run_id": "a" * 32,
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
    assert not read_json(tmp_path / "changed/usage.json")["calls"]
    assert (tmp_path / "changed/retention/00000.json").exists()
    assert len(fixture.requests) == count


def test_replay_miss_stops_without_provider_calls(tmp_path):
    request = job()
    request["mode"] = "replay"
    replay, fixture = execute(tmp_path, request)
    assert replay["status"] == "cache_miss"
    assert not fixture.requests
    assert not read_json(tmp_path / "run/usage.json")["calls"]
