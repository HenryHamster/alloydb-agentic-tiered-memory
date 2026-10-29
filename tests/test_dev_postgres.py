"""Opt-in, unpaid PostgreSQL/pgvector integration: DEV_TEST_DSN points to a disposable DB."""

import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import psycopg2
import pytest
from dev_fixture import FixtureModels
from test_offline import MemoryValkey

from devtools.cache import CachedConnection
from devtools.files import read_json
from devtools.state import RunSchema
from devtools.worker import run_job
from tiered_memory import build_memory_system

SOURCE = Path(__file__).resolve().parents[1]
DSN = os.environ.get("DEV_TEST_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="Set DEV_TEST_DSN to a disposable pgvector database"
)


def connect():
    return psycopg2.connect(DSN)


@pytest.fixture(autouse=True)
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


def test_typed_snapshot_and_isolation(tmp_path):
    schemas = [RunSchema(connect, uuid.uuid4().hex, SOURCE) for _ in range(3)]
    try:
        for schema in schemas:
            schema.create()
        first, second, empty = schemas
        con = first.connection()
        with con.cursor() as cur:
            ids = [str(uuid.uuid4()) for _ in range(2)]
            vector = "[" + ",".join(["0.1"] * 768) + "]"
            for number, fid in enumerate(ids):
                cur.execute(
                    """INSERT INTO memory_facts(fact_id,user_id,visibility,fact_type,category,speaker,
                    claim_type,text,evidence,source_episode_id,valid_from,superseded_by,embedding,details,
                    access_count,tier) VALUES (%s,'same-user','global','world','attribute','user','asserted',
                    'movie Inception','exact source','episode','2026-01-01',%s,%s,'{"who":"club"}',7,'warm')""",
                    (fid, ids[1 - number], vector),
                )
            cur.execute(
                """INSERT INTO macro_memories(macro_id,user_id,visibility,macro_title,macro_summary,
                supporting_fact_ids,temporal_span,macro_embedding) VALUES (%s,'same-user','global','Movie',
                'Inception',%s::uuid[],tstzrange('2025-01-01','2026-01-01','[]'),%s)""",
                (str(uuid.uuid4()), ids, vector),
            )
        con.commit()
        con.close()
        snapshot = first.snapshot()
        second.restore(snapshot)
        restored = second.snapshot()
        assert sorted(restored["memory_facts"], key=lambda r: r["fact_id"]) == sorted(
            snapshot["memory_facts"], key=lambda r: r["fact_id"]
        )
        assert restored["macro_memories"] == snapshot["macro_memories"]
        assert all(not rows for rows in empty.snapshot().values())
        first.close()
        assert second.snapshot()["memory_facts"]
    finally:
        for schema in schemas:
            schema.close()


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


def test_real_sql_pipeline_preparation_restore_answer_replay(tmp_path):
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
