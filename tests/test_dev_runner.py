"""Developer runner: cache, CLI, changed preparation, and optional real SQL integration."""

import base64
import hashlib
import io
import json
import os
import pickle
import tarfile
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psycopg2
import pytest
from google.genai import types
from test_offline import MemoryValkey

from devtools.bootstrap import extract_bundle
from devtools.cache import (
    CachedConnection,
    CachedModels,
    CacheMiss,
    RequestCache,
    Usage,
)
from devtools.cli import collect_archive
from devtools.files import read_json, save_json
from devtools.state import RunSchema
from devtools.worker import run_job
from recall import InMemoryRecallStore, Recaller
from reflect import Reflector
from reflect.consolidation import LLMConflictJudge
from reflect.macros import LLMMacroWriter
from reflect.store import InMemoryReflectStore
from retain import RetainConfig
from retain.embeddings import GeminiEmbedder
from retain.entities import FuzzyEntityResolver
from retain.extraction import GeminiFactExtractor
from retain.linking import GeminiSupersedeClassifier, StoreLinker
from retain.observations import GeminiObservationBuilder
from retain.pipeline import Retainer
from retain.store import InMemoryRetainStore
from shared.llm import GeminiGenerator, GeminiStructuredGenerator
from tiered_memory import TieredMemory, build_memory_system
from valkey_buffer import SessionBuffer

SOURCE = Path(__file__).resolve().parents[1]
DSN = os.environ.get("DEV_TEST_DSN")


class FixtureModels:
    def __init__(self):
        self.requests = []

    def generate_content(self, **request):
        self.requests.append(request)
        prompt = request["contents"]
        schema = request.get("config", {}).get("response_json_schema", {})
        if prompt.startswith("Extract useful"):
            payload = json.loads(prompt.split("SOURCE JSON:\n", 1)[1])
            chunk = payload["chunks"][0]
            rows = []
            for key, text in enumerate(
                (
                    "The club picked Inception.",
                    "Inception's movie ID is tt1375666.",
                    "The club will watch Inception on Saturday.",
                )
            ):
                rows.append(
                    {
                        "key": key,
                        "chunk_id": chunk["chunk_id"],
                        "text": text,
                        "evidence": chunk["text"],
                        "fact_type": "world",
                        "category": "attribute",
                        "claim_type": "asserted",
                        "visibility": "session",
                        "occurred_at": None,
                        "valid_from": None,
                        "details": {k: None for k in ("who", "what", "when", "where", "why")},
                        "entities": [
                            {"name": "Inception", "entity_type": "other", "role": "subject"}
                        ],
                        "relations": [],
                        "caused_by": [],
                        "evidenced_by": [],
                    }
                )
            text = json.dumps(rows)
        elif "judgments" in schema.get("properties", {}):
            text = '{"judgments": []}'
        elif "title" in schema.get("properties", {}):
            text = '{"title": "Film club", "summary": "The club picked Inception, movie tt1375666, for Saturday."}'
        elif "fact_ids" in schema.get("properties", {}):
            text = '{"summary": "", "fact_ids": []}'
        elif prompt.startswith("You prepare questions"):
            text = '{"sub_queries": [], "time_window": null}'
        elif "one JSON object with keys name and arguments" in prompt:
            text = '{"name": "Get Movie Reviews", "arguments": {"movie_id": 1375666}}'
        else:
            text = "Inception"
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=text)]))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=80, candidates_token_count=20, thoughts_token_count=10
            ),
            model_version="fixture-not-a-live-model",
        )

    def embed_content(self, **request):
        self.requests.append(request)
        contents = request["contents"]
        if isinstance(contents, str):
            contents = [contents]
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.1] * 768) for _ in contents]
        )


RETAIN_ATTRS = (
    "documents",
    "retained",
    "facts",
    "entities",
    "fact_entities",
    "entity_relations",
    "links",
    "observations",
)


class MemorySchema:
    def __init__(self):
        self.created = False
        self.retain = InMemoryRetainStore()
        self.reflect = InMemoryReflectStore(self.retain)

    def create(self):
        self.created = True

    def snapshot(self):
        # Test double only: production snapshots are typed PostgreSQL JSON rows, never pickle.
        value = (
            [getattr(self.retain, k) for k in RETAIN_ATTRS],
            self.reflect.states,
            self.reflect.macros,
            self.reflect.reviews,
        )
        return {"fixture": base64.b64encode(pickle.dumps(value)).decode()}

    def restore(self, snapshot):
        values, states, macros, reviews = pickle.loads(base64.b64decode(snapshot["fixture"]))
        for key, value in zip(RETAIN_ATTRS, values):
            setattr(self.retain, key, value)
        self.reflect.states, self.reflect.macros, self.reflect.reviews = states, macros, reviews

    def close(self):
        self.created = False


class MemoryBackend:
    def __init__(self, config, cache, models, source, clock, run_id):
        self.cache, self.models, self.clock = cache, models, clock
        self.schemas = []

    def schema(self):
        schema = MemorySchema()
        self.schemas.append(schema)
        return schema

    def system(self, schema):
        client = SimpleNamespace(models=self.models)
        embedder = GeminiEmbedder(client)
        structured = GeminiStructuredGenerator(client)
        generate = GeminiGenerator(client)
        retainer = Retainer(
            store=schema.retain,
            extractor=GeminiFactExtractor(client),
            entity_resolver=FuzzyEntityResolver(schema.retain),
            embedder=embedder,
            linker=StoreLinker(schema.retain),
            classifier=GeminiSupersedeClassifier(client, schema.retain),
            observation_builder=GeminiObservationBuilder(client, schema.retain, embedder),
            config=RetainConfig(maintenance_every=0),
            clock=lambda: self.clock,
        )
        reflector = Reflector(
            store=schema.reflect,
            embedder=embedder,
            judge=LLMConflictJudge(structured),
            macro_writer=LLMMacroWriter(structured),
            clock=lambda: self.clock,
        )

        def ranking(query, texts):
            result = self.cache.call(
                "ranking",
                {"query": query, "documents": list(texts)},
                lambda key: {"scores": [0.8] * len(texts)},
            )
            return result["scores"]

        recaller = Recaller(
            store=InMemoryRecallStore(schema.retain, schema.reflect),
            reflect_store=schema.reflect,
            embedder=embedder,
            generate=generate,
            reranker=ranking,
            clock=lambda: self.clock,
            strict=True,
        )
        return TieredMemory(
            retainer=retainer,
            reflector=reflector,
            recaller=recaller,
            buffer=SessionBuffer(MemoryValkey(), generate, strict=True),
            generate=generate,
        )

    def close(self):
        assert not any(s.created for s in self.schemas)


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
        replay.generate_content(**request, config={"temperature": 0.1})


def test_archive_traversal_rejected(tmp_path):
    name = "../escape"
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        member.size = 1
        stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        extract_bundle(archive, tmp_path / "destination")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        collect_archive(archive, digest, tmp_path / "results", "a" * 32)


def test_collect_worker_pipeline_and_local_scoring_protocol(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from devtools.cli import run_remote
    from devtools.worker import run_job

    class FakeCloud:
        config = dict(
            project="fixture",
            vm="fixture-vm",
            zone="us-east1-b",
            region="us-east1",
            db_cluster="cluster",
            db_instance="db",
            generation_location="global",
            max_hours=4,
        )

        def status(self):
            return {"vm": "RUNNING", "database": {"state": "READY"}}

        def control(self, action):
            assert action == "touch"

        def gcloud(self, args, **kwargs):
            if args[3].startswith("fixture-vm:"):
                self.upload = Path(args[2])
            else:
                Path(args[3]).write_bytes(self.export.read_bytes())

        def ssh(self, command, **kwargs):
            source = tmp_path / "uploaded"
            extract_bundle(self.upload, source)
            job = read_json(source / "job.json")
            assert "budget" not in job
            assert not any(
                n.endswith(".json") or ".env" in n
                for n in read_json(source / "manifest.json")["files"]
            )
            assert all(
                "expected_answer" not in q and "expected_tool_call" not in q
                for q in job["questions"]
            )
            output = tmp_path / "remote" / job["run_id"]
            client = SimpleNamespace(models=FixtureModels())
            result = run_job(
                job,
                SOURCE,
                tmp_path / "cache",
                output,
                client,
                client,
                backend_factory=MemoryBackend,
            )
            assert result["status"] == "completed", result
            self.export = tmp_path / "remote.tar.gz"
            with tarfile.open(self.export, "w:gz") as stream:
                stream.add(output, arcname=job["run_id"])
            return (
                "DEV_RESULT /home/fixture/.tiered-dev/runs/"
                + job["run_id"]
                + ".tar.gz "
                + hashlib.sha256(self.export.read_bytes()).hexdigest()
            )

    questions = tmp_path / "questions.json"
    save_json(
        questions,
        {
            "questions": [
                {"id": "movie", "question": "Which movie?", "expected_answer": "Inception"}
            ]
        },
    )
    args = SimpleNamespace(
        history=SOURCE / "fixtures/dev/history.json",
        question=None,
        questions=questions,
        question_id="movie",
        name="protocol",
        output=tmp_path / "results",
        mode="cached",
        max_output_tokens=1024,
        cache_epoch="1",
    )
    assert run_remote(FakeCloud(), args, SOURCE) == 0
    report = read_json(next((tmp_path / "results").glob("*/report.json")))
    assert report["questions"][0]["score"]["correct"]
    assert report["api_usage"]["calls"]
    assert any(
        c.get("usage_metadata", {}).get("thoughts_token_count")
        for c in report["api_usage"]["calls"]
        if c["kind"] == "generate"
    )


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


def connect():
    return psycopg2.connect(DSN)


@pytest.fixture
def extensions():
    connection = connect()
    try:
        with connection.cursor() as cur:
            cur.execute(
                "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_trgm"
            )
            cur.execute("CREATE SCHEMA IF NOT EXISTS ai")
            cur.execute("""CREATE OR REPLACE FUNCTION ai.rank(model_id text, search_string text, documents text[])
                RETURNS TABLE(index int, score real) LANGUAGE SQL AS
                $$ SELECT (n-1)::int, 0.8::real FROM generate_subscripts(documents,1) AS n $$""")
        connection.commit()
    finally:
        connection.close()


class PostgresBackend:
    def __init__(self, config, cache, models, source, clock, run_id):
        self.cache, self.models, self.source, self.clock = cache, models, source, clock

    def schema(self):
        return RunSchema(connect, uuid.uuid4().hex, self.source)

    def system(self, schema):
        client = SimpleNamespace(models=self.models)
        system = build_memory_system(
            lambda: CachedConnection(schema.connection(), self.cache),
            MemoryValkey(),
            client,
            client,
        )
        for component in (system.retainer, system.reflector, system.recaller):
            component.clock = lambda: self.clock
        return system

    def close(self):
        pass


@pytest.mark.skipif(not DSN, reason="Set DEV_TEST_DSN to a disposable pgvector database")
def test_real_sql_pipeline_preparation_restore_answer_replay(tmp_path, extensions):
    fixture = FixtureModels()
    client = SimpleNamespace(models=fixture)
    job = {
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
    first = run_job(
        job,
        SOURCE,
        tmp_path / "cache",
        tmp_path / "first",
        client,
        client,
        backend_factory=PostgresBackend,
    )
    assert first["status"] == "completed", first
    assert read_json(tmp_path / "first/answers/movie.json")["answer"] == "Inception"
    assert read_json(tmp_path / "first/answers/movie.json")["context"]["recall"]["items"]
    count = len(fixture.requests)
    job["mode"] = "replay"
    second = run_job(
        job,
        SOURCE,
        tmp_path / "cache",
        tmp_path / "second",
        client,
        client,
        backend_factory=PostgresBackend,
    )
    assert second["status"] == "completed", second
    assert not read_json(tmp_path / "second/usage.json")["calls"]
    assert len(fixture.requests) == count
