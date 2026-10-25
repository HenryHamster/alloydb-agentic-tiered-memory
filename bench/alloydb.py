"""Benchmark writer for AlloyDB: the documents and facts the verbatim baseline produces.

AlloyDBRetainStore.write is still Retain's stub. This writes the subset of a WriteBatch the
benchmark creates (documents and facts; no entities, links or supersessions) into the real
sql/004_retain_schema.sql tables, so AlloyDBRecallStore is measured on them. Swap in Retain's
store once it is filled.
"""

import threading

from psycopg2.extras import Json, execute_values

from retain.models import WriteBatch

_DOCUMENT_COLUMNS = (
    "document_id, episode_id, position, user_id, project_id, session_id, speaker, text,"
    " occurred_at, recorded_at, expires_at, retained_at"
)
_FACT_COLUMNS = (
    "fact_id, user_id, project_id, session_id, visibility, fact_type, category, speaker,"
    " claim_type, text, evidence, details, source_document_id, source_episode_id, embedding,"
    " occurred_at, recorded_at, valid_from, valid_to, superseded_by, evidence_count"
)
_FACT_TEMPLATE = (
    "(%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid, %s, %s::vector,"
    " %s, %s, %s, %s, %s::uuid, %s)"
)


class BenchAlloyDBWriter:
    """RetainStore subset Retainer needs for the baseline: retained_document_ids and write."""

    def __init__(self, conn_factory):
        self.conn_factory = conn_factory
        self._conn = None
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def retained_document_ids(self, document_ids):
        rows = self._run(
            "SELECT document_id::text FROM documents"
            " WHERE document_id = ANY(%s::uuid[]) AND retained_at IS NOT NULL",
            (list(document_ids),),
            fetch=True,
        )
        return {row[0] for row in rows}

    def write(self, batch: WriteBatch) -> None:
        unsupported = (
            batch.entities,
            batch.fact_entities,
            batch.entity_relations,
            batch.links,
            batch.supersessions,
            batch.reinforced_fact_ids,
        )
        if any(unsupported):
            raise NotImplementedError("BenchAlloyDBWriter writes documents and facts only")
        documents = [
            (
                d.document_id,
                d.episode_id,
                d.position,
                d.scope.user_id,
                d.scope.project_id,
                d.scope.session_id,
                str(d.speaker),
                d.text,
                d.occurred_at,
                d.recorded_at,
                d.expires_at,
            )
            for d in batch.documents
        ]
        facts = [
            (
                f.fact_id,
                f.scope.user_id,
                f.scope.project_id,
                f.scope.session_id,
                str(f.visibility),
                str(f.fact_type),
                str(f.category),
                str(f.speaker),
                str(f.claim_type),
                f.text,
                f.evidence,
                Json(f.details),
                f.source_document_id,
                f.source_episode_id,
                "[" + ",".join(repr(float(x)) for x in f.embedding) + "]" if f.embedding else None,
                f.occurred_at,
                f.recorded_at,
                f.valid_from,
                f.valid_to,
                f.superseded_by,
                f.evidence_count,
            )
            for f in batch.facts
        ]

        def apply(cur):
            if documents:
                execute_values(
                    cur,
                    f"INSERT INTO documents ({_DOCUMENT_COLUMNS}) VALUES %s ON CONFLICT DO NOTHING",
                    documents,
                    template="(%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                )
            if facts:
                execute_values(
                    cur,
                    f"INSERT INTO memory_facts ({_FACT_COLUMNS}) VALUES %s ON CONFLICT DO NOTHING",
                    facts,
                    template=_FACT_TEMPLATE,
                )

        self._run(apply)

    def delete_users(self, user_id_prefix: str) -> None:
        """Removes every benchmark row whose user_id starts with the prefix."""

        def apply(cur):
            for table in ("memory_facts", "documents"):
                cur.execute(
                    f"DELETE FROM {table} WHERE starts_with(user_id, %s)", (user_id_prefix,)
                )

        self._run(apply)

    def _run(self, statement, params=None, *, fetch=False):
        with self._lock:  # benchmark workers share the connection
            if self._conn is None or self._conn.closed:
                self._conn = self.conn_factory()
            conn = self._conn
            try:
                with conn.cursor() as cur:
                    if callable(statement):
                        statement(cur)
                        rows = None
                    else:
                        cur.execute(statement, params)
                        rows = cur.fetchall() if fetch else None
                conn.commit()
                return rows
            except Exception:
                if not conn.closed:
                    conn.rollback()
                raise
