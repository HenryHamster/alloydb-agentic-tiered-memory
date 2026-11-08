"""Dream transactions: offline reference implementation and durable PostgreSQL queue.

Run sql/006_dream_schema.sql explicitly before using AlloyDBDreamStore. No
background thread or migration starts merely by importing this module.
"""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from uuid import uuid4

from psycopg2.extras import Json, RealDictCursor

from reflect.dream import encoded, snapshot
from retain.alloydb_store import _FACT_COLUMNS, _fact, _insert_fact
from retain.models import Document, Scope, Speaker, Visibility, is_visible


def _sources(documents, facts, job, now):
    user, project, sessions = job["user_id"], job["project_id"], set(job["session_ids"])
    docs = [
        d
        for d in documents
        if d.scope.user_id == user
        and d.scope.project_id == project
        and d.scope.session_id in sessions
        and d.occurred_at <= now
        and (d.expires_at is None or d.expires_at > now)
    ]
    allowed = [
        f
        for f in facts
        if f.scope.user_id == user
        and f.recorded_at <= now
        and (
            f.visibility == Visibility.GLOBAL
            or f.scope.project_id == project
            and (f.visibility == Visibility.PROJECT or f.scope.session_id in sessions)
        )
    ]
    return (
        sorted(docs, key=lambda d: (d.occurred_at, d.episode_id, d.position, d.document_id)),
        sorted(allowed, key=lambda f: f.fact_id),
    )


def _claim(job, now):
    if job["status"] == "running" and datetime.fromisoformat(job["lease_until"]) > now:
        return None
    if job["status"] not in ("queued", "running"):
        return None
    if job["attempts"] >= job["config"]["max_attempts"]:
        job.update(status="failed", error="Maximum worker attempts exhausted")
        return None
    job.update(
        status="running",
        attempts=job["attempts"] + 1,
        error=None,
        lease_token=str(uuid4()),
        updated_at=now.isoformat(),
        lease_until=(now + timedelta(seconds=job["config"]["lease_seconds"])).isoformat(),
    )
    return job


def _visible(tx, records, scope, now):
    fresh, visible = {}, {}
    for run_id, insight in records:
        origin = Scope(**insight["scope"])
        if not is_visible(Visibility(insight["visibility"]), origin, scope):
            continue
        if datetime.fromisoformat(insight["created_at"]) > now:
            continue
        if run_id not in fresh:
            job = tx.get_job(run_id)
            fresh[run_id] = job["status"] == "applied" and (
                snapshot(tx, job, now)[2] == job.get("fresh_hash")
            )
        if fresh[run_id]:
            key = insight["insight_id"]
            if key not in visible or insight["created_at"] > visible[key]["created_at"]:
                visible[key] = insight
    return deepcopy(list(visible.values()))


class InMemoryDreamStore:
    """Reference store; rolls back active facts and staged jobs on any failure."""

    def __init__(self, retain_store):
        self.retain_store = retain_store
        self.jobs = {}
        self.insight_rows = {}

    @contextmanager
    def transaction(self):
        with self.retain_store._lock:
            before = (
                deepcopy(self.jobs),
                deepcopy(self.insight_rows),
                dict(self.retain_store.facts),
            )
            try:
                yield self
            except BaseException:
                self.jobs, self.insight_rows, self.retain_store.facts = before
                raise

    def protect_sources(self):
        pass  # the shared Retain lock already protects this transaction

    def get_job(self, run_id):
        return deepcopy(self.jobs.get(run_id))

    def put_job(self, job):
        self.jobs[job["run_id"]] = deepcopy(job)

    def create_job(self, job):
        self.jobs.setdefault(job["run_id"], deepcopy(job))
        return self.get_job(job["run_id"])

    def claim(self, now):
        for job in sorted(self.jobs.values(), key=lambda j: (j["created_at"], j["run_id"])):
            claimed = _claim(job, now)
            if claimed:
                return deepcopy(claimed)
        return None

    def sources(self, job, now):
        return _sources(
            self.retain_store.documents.values(), self.retain_store.facts.values(), job, now
        )

    def put_facts(self, facts):
        for fact in facts:
            old = self.retain_store.facts.get(fact.fact_id)
            if old is not None and old != fact:
                raise ValueError("Conflicting recovered fact ID")
            self.retain_store.facts[fact.fact_id] = fact

    def put_insights(self, job):
        for insight in job["insights"]:
            self.insight_rows[(job["run_id"], insight["insight_id"])] = deepcopy(insight)

    def visible_insights(self, scope, now):
        return _visible(self, [(r, i) for (r, _), i in self.insight_rows.items()], scope, now)


class AlloyDBDreamStore:
    """Connection-per-transaction queue. Independent workers use SKIP LOCKED.

    No provider calls occur in these transactions. Apply briefly locks source
    tables against writes to validate and publish one atomic snapshot.
    """

    def __init__(self, conn_factory):
        self.conn_factory = conn_factory

    @contextmanager
    def transaction(self):
        conn = self.conn_factory()
        try:
            with conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    yield _PostgresTransaction(cur)
        finally:
            conn.close()


class _PostgresTransaction:
    def __init__(self, cur):
        self.cur = cur

    def protect_sources(self):
        # Acquired before reading the job, with the same order for every apply.
        # SHARE ROW EXCLUSIVE avoids lock-upgrade deadlocks between dream applies.
        self.cur.execute("LOCK TABLE documents, memory_facts IN SHARE ROW EXCLUSIVE MODE")

    def get_job(self, run_id):
        self.cur.execute(
            "SELECT payload FROM dream_runs WHERE run_id = %s::uuid FOR UPDATE", (run_id,)
        )
        row = self.cur.fetchone()
        return row["payload"] if row else None

    def put_job(self, job):
        self.cur.execute(
            "INSERT INTO dream_runs (run_id, user_id, project_id, status, created_at, lease_until, payload) "
            "VALUES (%s::uuid, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status, "
            "lease_until = EXCLUDED.lease_until, payload = EXCLUDED.payload",
            (
                job["run_id"],
                job["user_id"],
                job["project_id"],
                job["status"],
                job["created_at"],
                job["lease_until"],
                Json(job, dumps=encoded),
            ),
        )

    def create_job(self, job):
        self.cur.execute(
            "INSERT INTO dream_runs (run_id, user_id, project_id, status, created_at, payload) "
            "VALUES (%s::uuid, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (
                job["run_id"],
                job["user_id"],
                job["project_id"],
                job["status"],
                job["created_at"],
                Json(job, dumps=encoded),
            ),
        )
        return self.get_job(job["run_id"])

    def claim(self, now):
        while True:
            self.cur.execute(
                "SELECT payload FROM dream_runs WHERE status = 'queued' OR "
                "(status = 'running' AND lease_until <= %s) "
                "ORDER BY created_at, run_id LIMIT 1 FOR UPDATE SKIP LOCKED",
                (now,),
            )
            row = self.cur.fetchone()
            if row is None:
                return None
            job = row["payload"]
            claimed = _claim(job, now)
            self.put_job(job)
            if claimed:
                return claimed

    def sources(self, job, now):
        self.cur.execute(
            "SELECT * FROM documents WHERE user_id = %s AND project_id = %s "
            "AND session_id = ANY(%s) AND occurred_at <= %s "
            "AND (expires_at IS NULL OR expires_at > %s)",
            (job["user_id"], job["project_id"], job["session_ids"], now, now),
        )
        documents = [
            Document(
                document_id=str(r["document_id"]),
                episode_id=r["episode_id"],
                position=r["position"],
                scope=Scope(r["user_id"], r["project_id"], r["session_id"]),
                speaker=Speaker(r["speaker"]),
                text=r["text"],
                occurred_at=r["occurred_at"],
                recorded_at=r["recorded_at"],
                expires_at=r["expires_at"],
            )
            for r in self.cur.fetchall()
        ]
        self.cur.execute(
            f"SELECT {_FACT_COLUMNS} FROM memory_facts f "
            "LEFT JOIN memory_entities me ON me.fact_id = f.fact_id "
            "WHERE f.user_id = %s AND (f.visibility = 'global' OR (f.project_id = %s "
            "AND (f.visibility = 'project' OR f.session_id = ANY(%s)))) "
            "GROUP BY f.fact_id",
            (job["user_id"], job["project_id"], job["session_ids"]),
        )
        return _sources(documents, [_fact(r) for r in self.cur.fetchall()], job, now)

    def put_facts(self, facts):
        for fact in facts:
            _insert_fact(self.cur, fact)

    def put_insights(self, job):
        for insight in job["insights"]:
            self.cur.execute(
                "INSERT INTO dream_insights (run_id, insight_id, user_id, project_id, payload) "
                "VALUES (%s::uuid, %s::uuid, %s, %s, %s) ON CONFLICT DO NOTHING",
                (
                    job["run_id"],
                    insight["insight_id"],
                    job["user_id"],
                    job["project_id"],
                    Json(insight, dumps=encoded),
                ),
            )

    def visible_insights(self, scope, now):
        self.cur.execute(
            "SELECT run_id::text, payload FROM dream_insights WHERE user_id = %s AND project_id = %s",
            (scope.user_id, scope.project_id),
        )
        return _visible(
            self, [(r["run_id"], r["payload"]) for r in self.cur.fetchall()], scope, now
        )
