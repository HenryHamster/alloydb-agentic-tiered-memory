"""Optional real queue/transaction tests; CI supplies a disposable pgvector DSN."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest
from test_dream import NOW, SCOPE, ScriptedGenerator, document

from bench.embedders import HashingEmbedder
from reflect.dream import Dreamer, DreamRequest, LostLease, StaleDream
from reflect.dream_store import AlloyDBDreamStore
from retain.alloydb_store import AlloyDBRetainStore
from retain.models import WriteBatch

DSN = os.getenv("DEV_TEST_DSN")
pytestmark = pytest.mark.skipif(
    not DSN, reason="Set DEV_TEST_DSN to a disposable pgvector database"
)


@pytest.fixture
def database():
    schema = "dream_test_" + uuid4().hex
    root = Path(__file__).resolve().parents[1]
    admin = psycopg2.connect(DSN)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute(f'CREATE SCHEMA "{schema}"')

    def connect():
        return psycopg2.connect(DSN, options=f"-c search_path={schema},public")

    try:
        conn = connect()
        with conn, conn.cursor() as cur:
            for name in ("004_retain_schema.sql", "005_reflect_schema.sql", "006_dream_schema.sql"):
                cur.execute((root / "sql" / name).read_text(encoding="utf-8"))
        conn.close()
        retain = AlloyDBRetainStore(connect)
        try:
            retain.write(WriteBatch(documents=(document(), document("b", session="s2"))))
        finally:
            retain.close()
        yield connect
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP SCHEMA "{schema}" CASCADE')
        admin.close()


def dreamer(connect):
    return Dreamer(
        store=AlloyDBDreamStore(connect),
        generate=ScriptedGenerator(),
        embedder=HashingEmbedder(),
        clock=lambda: NOW,
    )


def test_queue_survives_new_instances_and_publishes_atomically(database):
    first = dreamer(database)
    job = first.start(DreamRequest(SCOPE, ("s1", "s2")))
    second = dreamer(database)
    assert second.worker_once()["status"] == "ready"
    assert not first.insights(SCOPE)
    first.apply(job["run_id"], SCOPE)
    assert len(second.insights(SCOPE)) == 1
    assert second.get(job["run_id"], SCOPE)["fresh"]
    assert second.apply(job["run_id"], SCOPE)["status"] == "applied"


def test_parallel_enqueue_is_idempotent_and_claim_is_exclusive(database):
    def enqueue(_):
        return dreamer(database).start(DreamRequest(SCOPE, ("s1", "s2"), "parallel"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = list(pool.map(enqueue, range(4)))
    assert len({j["run_id"] for j in jobs}) == 1

    def claim(_):
        with AlloyDBDreamStore(database).transaction() as tx:
            return tx.claim(NOW)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(claim, range(2)))
    assert sum(j is not None for j in claimed) == 1


def test_old_worker_cannot_overwrite_reclaimed_job(database):
    worker = dreamer(database)
    worker.start(DreamRequest(SCOPE, ("s1", "s2")))
    with worker.store.transaction() as tx:
        old = tx.claim(NOW)
    with AlloyDBDreamStore(database).transaction() as tx:
        new = tx.claim(NOW + timedelta(seconds=601))
    assert new["lease_token"] != old["lease_token"]
    with pytest.raises(LostLease):
        worker._checkpoint(old, old["lease_token"])


def test_source_change_during_embedding_prevents_partial_publish(database):
    worker = dreamer(database)
    job = worker.start(DreamRequest(SCOPE, ("s1", "s2")))
    worker.worker_once()

    def embed(texts):
        conn = database()
        with conn, conn.cursor() as cur:
            cur.execute("UPDATE documents SET text = 'changed source'")
        conn.close()
        return HashingEmbedder()(texts)

    worker.embedder = embed
    with pytest.raises(StaleDream):
        worker.apply(job["run_id"], SCOPE)
    conn = database()
    with conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM memory_facts")
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM dream_insights")
        assert cur.fetchone()[0] == 0
    conn.close()
