"""AlloyDB implementation of RecallStore over sql/004_retain_schema.sql."""

import logging
import re
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID

from recall.config import MemoryType
from recall.models import Memory
from recall.store import _STOPWORDS
from retain.entities import normalize_name
from retain.models import (
    Category,
    ClaimType,
    Document,
    FactType,
    MemoryFact,
    Observation,
    Scope,
    Speaker,
    Visibility,
)

logger = logging.getLogger(__name__)

# Retain's VISIBLE_SQL (retain/alloydb_store.py) with its validity check swapped: Recall reads
# as of a point in time and may ask for superseded rows, where Retain reads current rows only.
CANDIDATE_SQL = """
    f.user_id = %(user_id)s
    AND (%(include_invalid)s OR f.valid_to IS NULL OR f.valid_to > %(as_of)s)
    AND (f.visibility = 'global'
         OR (f.visibility = 'project' AND f.project_id = %(project_id)s)
         OR (f.visibility = 'session' AND f.session_id = %(session_id)s))
"""

# Documents have no visibility column; mirrors recall.store.episode_visible.
EPISODE_SQL = """
    d.user_id = %(user_id)s
    AND CASE
        WHEN d.project_id IS NOT NULL THEN d.project_id = %(project_id)s
        WHEN d.session_id IS NOT NULL THEN d.session_id = %(session_id)s
        ELSE TRUE
    END
"""

# Row payloads, decoded by _fact / _observation / _episode. Memory carries neither a fact's
# evidence quote nor its details, so they are not read.
FACT_JSON = """jsonb_build_object(
    'fact_id', f.fact_id::text, 'user_id', f.user_id, 'project_id', f.project_id,
    'session_id', f.session_id, 'visibility', f.visibility, 'fact_type', f.fact_type,
    'category', f.category, 'speaker', f.speaker, 'claim_type', f.claim_type, 'text', f.text,
    'source_document_id', f.source_document_id::text, 'source_episode_id', f.source_episode_id,
    'occurred_at', f.occurred_at, 'recorded_at', f.recorded_at, 'valid_from', f.valid_from,
    'valid_to', f.valid_to, 'superseded_by', f.superseded_by::text,
    'evidence_count', f.evidence_count,
    'entity_ids', ARRAY(SELECT DISTINCT me.entity_id::text FROM memory_entities me
                        WHERE me.fact_id = f.fact_id ORDER BY 1)
)"""

OBSERVATION_JSON = """jsonb_build_object(
    'observation_id', f.observation_id::text, 'user_id', f.user_id, 'project_id', f.project_id,
    'session_id', f.session_id, 'visibility', f.visibility, 'entity_id', f.entity_id::text,
    'text', f.text, 'fact_ids', f.fact_ids::text[], 'valid_from', f.valid_from,
    'valid_to', f.valid_to
)"""

DOCUMENT_JSON = """jsonb_build_object(
    'document_id', d.document_id::text, 'episode_id', d.episode_id, 'position', d.position,
    'user_id', d.user_id, 'project_id', d.project_id, 'session_id', d.session_id,
    'speaker', d.speaker, 'text', d.text, 'occurred_at', d.occurred_at,
    'recorded_at', d.recorded_at, 'expires_at', d.expires_at
)"""

# websearch_to_tsquery ANDs every word, so a natural question rarely matches anything. OR the
# query's lexemes instead and let ts_rank reward the rows matching more of them, like the word
# overlap of InMemoryRecallStore.
OR_TSQUERY = "replace(plainto_tsquery('english', %(q_{i})s)::text, ' & ', ' | ')::tsquery"

# Entities named in query i: exact canonical name, name or alias among the query's word n-grams,
# or a pg_trgm match on the canonical name for longer n-grams.
ENTITY_SQL = """
    SELECT e.entity_id FROM entities e
    WHERE e.user_id = %(user_id)s
      AND (e.canonical_name = ANY(%(grams_{i})s::text[])
           OR lower(e.name) = ANY(%(grams_{i})s::text[])
           OR e.aliases && %(grams_{i})s::text[]
           OR EXISTS (SELECT 1 FROM unnest(%(fuzzy_{i})s::text[]) g
                      WHERE similarity(e.canonical_name, g) >= %(min_entity_similarity)s))
"""

# Per memory table: (kind, table, id column, primary date, interval end). Dates mirror
# Memory.primary_date and Memory.interval so channel order matches InMemoryRecallStore.
_TABLES = {
    MemoryType.FACT: (
        "fact",
        "memory_facts",
        "fact_id",
        "COALESCE(f.occurred_at, f.valid_from)",
        "GREATEST(COALESCE(f.occurred_at, f.valid_from),"
        " COALESCE(f.occurred_at, f.valid_to, %(as_of)s))",
    ),
    MemoryType.OBSERVATION: (
        "observation",
        "observations",
        "observation_id",
        "f.valid_from",
        "GREATEST(f.valid_from, COALESCE(f.valid_to, %(as_of)s))",
    ),
}

_MAX_GRAM_WORDS = 4
_MIN_FUZZY_CHARS = 4


class AlloyDBRecallStore:
    """One reused connection from `conn_factory` (db_clients.get_db_connection).

    Read-only: each method runs one statement in a read-only transaction. All values are bound
    parameters (never formatted into SQL text); vectors are passed as '[...]'::vector literals.
    Rows map to recall.models.Memory through Memory.from_fact / from_observation / from_episode.
    """

    def __init__(self, conn_factory, *, min_entity_similarity: float = 0.6):
        self.conn_factory = conn_factory
        self.min_entity_similarity = min_entity_similarity
        self._conn = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def search(self, scope, queries, embeddings, window, types, as_of, include_invalid, limit):
        params = self._params(scope, as_of, include_invalid)
        params["limit"] = limit
        params["min_entity_similarity"] = self.min_entity_similarity
        for i, query in enumerate(queries):
            grams = _query_grams(query)
            params[f"q_{i}"] = query
            params[f"grams_{i}"] = grams
            params[f"fuzzy_{i}"] = [g for g in grams if len(g) >= _MIN_FUZZY_CHARS]
            if embeddings is not None:
                params[f"qvec_{i}"] = _vector(embeddings[i])
        if window is not None:
            params["win_start"], params["win_end"] = window.start, window.end

        sql = _search_sql(len(queries), embeddings is not None, window is not None, types)
        if sql is None:
            return {}
        lists: dict[str, list[Memory]] = {}
        memories: dict[tuple[str, str], Memory] = {}
        for channel, kind, row_id, row in self._fetch(sql, params):
            key = (kind, row_id)
            if key not in memories:
                memories[key] = _DECODERS[kind](row)
            lists.setdefault(channel, []).append(memories[key])
        return lists

    def expand_neighbors(self, scope, seed_ids, types, as_of, include_invalid, limit):
        seeds = [i for i in dict.fromkeys(seed_ids) if _is_uuid(i)]  # episode ids are not facts
        if MemoryType.FACT not in types or not seeds:
            return []
        params = self._params(scope, as_of, include_invalid)
        params.update(seeds=seeds, limit=limit)
        return [_fact(row) for (row,) in self._fetch(_NEIGHBORS_SQL, params)]

    def get_episode(self, scope, episode_id):
        params = self._params(scope, datetime.now(timezone.utc), False)
        params["episode_id"] = episode_id
        rows = self._fetch(_EPISODE_GET_SQL, params)
        if not rows:
            return None
        return {
            "episode_id": episode_id,
            "messages": [
                {"speaker": speaker, "text": text, "occurred_at": occurred_at.isoformat()}
                for speaker, text, occurred_at in rows
            ],
        }

    def get_supporting(self, scope, observation_id):
        if not _is_uuid(observation_id):
            return []
        # Like InMemoryRecallStore: supporting facts are returned even once superseded, so
        # the caller can see how the observation's evidence changed.
        params = self._params(scope, datetime.now(timezone.utc), True)
        params["id"] = observation_id
        return [_fact(row) for (row,) in self._fetch(_SUPPORTING_SQL, params)]

    # Connection handling.

    @staticmethod
    def _params(scope: Scope, as_of: datetime, include_invalid: bool) -> dict:
        return {
            "user_id": scope.user_id,
            "project_id": scope.project_id,
            "session_id": scope.session_id,
            "as_of": as_of,
            "include_invalid": include_invalid,
        }

    def _connection(self):
        if self._conn is None or self._conn.closed:
            conn = self.conn_factory()
            conn.set_session(readonly=True)
            with conn.cursor() as cur:
                cur.execute("SET TIME ZONE 'UTC';")
            conn.commit()
            try:
                # pgvector >= 0.8: keep scanning the HNSW graph until LIMIT rows pass the
                # user/visibility filter, instead of filtering a fixed ef_search candidate set.
                with conn.cursor() as cur:
                    cur.execute("SET hnsw.iterative_scan = relaxed_order;")
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.warning(
                    "hnsw.iterative_scan unavailable; filtered vector search may miss rows: %s", e
                )
            self._conn = conn
        return self._conn

    def _fetch(self, sql: str, params: dict) -> list[tuple]:
        with self._lock:
            for attempt in range(2):
                conn = self._connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall()
                    conn.commit()
                    return rows
                except Exception:
                    if conn.closed:
                        # Connection dropped: reads are safe to retry once on a new one.
                        self._conn = None
                        if attempt == 0:
                            continue
                    else:
                        conn.rollback()
                    raise
        raise AssertionError("unreachable")


# SQL builders. Only fixed fragments and internal names are formatted into SQL text.


def _search_sql(n_queries: int, vectors: bool, window: bool, types) -> str | None:
    """One statement: a CTE per channel ranking (kind, id) across memory types, then one join
    back to each table for the rows that made any channel."""
    ctes, hits = [], []
    tables = [_TABLES[t] for t in (MemoryType.FACT, MemoryType.OBSERVATION) if t in types]
    episodes = MemoryType.EPISODE in types

    def channel(name: str, parts: list[str]) -> None:
        if not parts:
            return
        cte = f"c{len(ctes)}"
        union = "\n  UNION ALL\n  ".join(parts)
        ctes.append(
            f"{cte} AS (SELECT kind, id, row_number() OVER (ORDER BY key DESC, kind, id) AS rank"
            f"\n  FROM ({union}) s)"
        )
        hits.append(
            f"SELECT '{name}'::text AS channel, kind, id, rank FROM {cte} WHERE rank <= %(limit)s"
        )

    prelude = []
    for i in range(n_queries):
        prelude.append(f"kq{i} AS (SELECT {OR_TSQUERY.format(i=i)} AS query)")
        if tables:
            prelude.append(f"ent{i} AS ({ENTITY_SQL.format(i=i)})")

    for i in range(n_queries):
        if vectors:
            channel(
                f"vector:{i}",
                [
                    f"(SELECT '{kind}'::text AS kind, f.{pk}::text AS id,"
                    f" -(f.embedding <=> %(qvec_{i})s::vector) AS key"
                    f" FROM {table} f WHERE {CANDIDATE_SQL} AND f.embedding IS NOT NULL"
                    f" ORDER BY f.embedding <=> %(qvec_{i})s::vector LIMIT %(limit)s)"
                    for kind, table, pk, _, _ in tables
                ],
            )
        keyword = [
            f"(SELECT '{kind}'::text AS kind, f.{pk}::text AS id,"
            f" ts_rank(f.text_tsv, q.query) AS key"
            f" FROM {table} f, kq{i} q WHERE {CANDIDATE_SQL} AND f.text_tsv @@ q.query"
            f" ORDER BY key DESC LIMIT %(limit)s)"
            for kind, table, pk, _, _ in tables
        ]
        if episodes:
            episode_tsv = "to_tsvector('english', string_agg(d.text, ' ' ORDER BY d.position))"
            keyword.append(
                f"(SELECT 'episode'::text AS kind, d.episode_id AS id,"
                f" ts_rank({episode_tsv}, (SELECT query FROM kq{i})) AS key"
                f" FROM documents d WHERE {EPISODE_SQL} GROUP BY d.episode_id"
                f" HAVING {episode_tsv} @@ (SELECT query FROM kq{i})"
                f" ORDER BY key DESC LIMIT %(limit)s)"
            )
        channel(f"keyword:{i}", keyword)
        entity = []
        for kind, table, pk, date, _ in tables:
            if kind == "fact":
                match = (
                    "EXISTS (SELECT 1 FROM memory_entities me"
                    f" WHERE me.fact_id = f.fact_id AND me.entity_id IN (SELECT entity_id FROM ent{i}))"
                )
            else:
                match = f"f.entity_id IN (SELECT entity_id FROM ent{i})"
            entity.append(
                f"(SELECT '{kind}'::text AS kind, f.{pk}::text AS id, {date} AS key"
                f" FROM {table} f WHERE {CANDIDATE_SQL} AND {match}"
                f" ORDER BY key DESC LIMIT %(limit)s)"
            )
        channel(f"entity:{i}", entity)

    if window:
        temporal = [
            f"(SELECT '{kind}'::text AS kind, f.{pk}::text AS id, {date} AS key"
            f" FROM {table} f WHERE {CANDIDATE_SQL}"
            f" AND {date} < %(win_end)s AND {end} >= %(win_start)s"
            f" ORDER BY key DESC LIMIT %(limit)s)"
            for kind, table, pk, date, end in tables
        ]
        if episodes:
            # An episode is a point at its first message's time (Memory.from_episode).
            temporal.append(
                "(SELECT 'episode'::text AS kind, id, key FROM ("
                " SELECT d.episode_id AS id, (array_agg(d.occurred_at ORDER BY d.position))[1] AS key"
                f" FROM documents d WHERE {EPISODE_SQL} GROUP BY d.episode_id) e"
                " WHERE key < %(win_end)s AND key >= %(win_start)s"
                " ORDER BY key DESC LIMIT %(limit)s)"
            )
        channel("temporal", temporal)

    if not hits:
        return None
    payload = []
    for kind, table, pk, _, _ in tables:
        row_json = FACT_JSON if kind == "fact" else OBSERVATION_JSON
        payload.append(
            f"SELECT '{kind}'::text AS kind, f.{pk}::text AS id, {row_json} AS data FROM {table} f"
            f" WHERE f.{pk} = ANY(ARRAY(SELECT h.id::uuid FROM hits h WHERE h.kind = '{kind}'))"
        )
    if episodes:
        payload.append(
            f"SELECT 'episode'::text AS kind, d.episode_id AS id,"
            f" jsonb_agg({DOCUMENT_JSON} ORDER BY d.position) AS data FROM documents d"
            " WHERE d.episode_id = ANY(ARRAY(SELECT h.id FROM hits h WHERE h.kind = 'episode'))"
            f" AND {EPISODE_SQL} GROUP BY d.episode_id"
        )
    ctes.append("hits AS (" + "\n  UNION ALL\n  ".join(hits) + ")")
    ctes.append("payload AS (" + "\n  UNION ALL\n  ".join(payload) + ")")
    return (
        "WITH " + ",\n".join(prelude + ctes) + "\n"
        "SELECT h.channel, h.kind, h.id, p.data FROM hits h"
        " JOIN payload p ON p.kind = h.kind AND p.id = h.id"
        " ORDER BY h.channel, h.rank"
    )


# Links in both directions by weight, then facts sharing an entity with a readable seed, newest
# first. Mirrors InMemoryRecallStore.expand_neighbors.
_NEIGHBORS_SQL = f"""
WITH seed AS (
    SELECT f.fact_id FROM memory_facts f
    WHERE f.fact_id = ANY(%(seeds)s::uuid[]) AND {CANDIDATE_SQL}
),
edge AS (
    SELECT l.to_fact_id AS fact_id, l.weight FROM memory_links l
    WHERE l.from_fact_id = ANY(%(seeds)s::uuid[])
    UNION ALL
    SELECT l.from_fact_id, l.weight FROM memory_links l
    WHERE l.to_fact_id = ANY(%(seeds)s::uuid[])
),
linked AS (
    SELECT f.fact_id, max(e.weight) AS weight
    FROM edge e JOIN memory_facts f ON f.fact_id = e.fact_id
    WHERE {CANDIDATE_SQL} AND f.fact_id <> ALL(%(seeds)s::uuid[])
    GROUP BY f.fact_id
),
shared AS (
    SELECT DISTINCT f.fact_id, COALESCE(f.occurred_at, f.valid_from) AS at
    FROM memory_entities se
    JOIN memory_entities me ON me.entity_id = se.entity_id
    JOIN memory_facts f ON f.fact_id = me.fact_id
    WHERE se.fact_id IN (SELECT fact_id FROM seed) AND {CANDIDATE_SQL}
      AND f.fact_id <> ALL(%(seeds)s::uuid[])
      AND f.fact_id NOT IN (SELECT fact_id FROM linked)
),
picked AS (
    SELECT fact_id, 0 AS grp, row_number() OVER (ORDER BY weight DESC, fact_id) AS pos FROM linked
    UNION ALL
    SELECT fact_id, 1, row_number() OVER (ORDER BY at DESC, fact_id) FROM shared
)
SELECT {FACT_JSON}
FROM (SELECT * FROM picked ORDER BY grp, pos LIMIT %(limit)s) p
JOIN memory_facts f ON f.fact_id = p.fact_id
ORDER BY p.grp, p.pos
"""

_EPISODE_GET_SQL = f"""
SELECT d.speaker, d.text, d.occurred_at FROM documents d
WHERE d.episode_id = %(episode_id)s AND {EPISODE_SQL}
ORDER BY d.position
"""

_SUPPORTING_SQL = f"""
SELECT {FACT_JSON}
FROM observations o JOIN memory_facts f ON f.fact_id = ANY(o.fact_ids)
WHERE o.observation_id = %(id)s::uuid AND {CANDIDATE_SQL.replace("f.", "o.")} AND {CANDIDATE_SQL}
ORDER BY array_position(o.fact_ids, f.fact_id)
"""


# Query text -> parameters.


def _query_grams(query: str) -> list[str]:
    """Word n-grams of the normalized query that may name an entity: no stopword at either end."""
    words = re.findall(r"\w+", normalize_name(query))
    grams = []
    for n in range(1, _MAX_GRAM_WORDS + 1):
        for start in range(len(words) - n + 1):
            gram = words[start : start + n]
            if gram[0] not in _STOPWORDS and gram[-1] not in _STOPWORDS:
                grams.append(" ".join(gram))
    return list(dict.fromkeys(grams))


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in values) + "]"


def _is_uuid(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except ValueError:
        return False


# Payload rows -> Memory.


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _scope(row: dict) -> Scope:
    return Scope(row["user_id"], row["project_id"], row["session_id"])


def _category(value: str) -> Category:
    try:
        return Category(value)
    except ValueError:  # category has no CHECK constraint in the schema
        return Category.OTHER


def _fact(row: dict) -> Memory:
    return Memory.from_fact(
        MemoryFact(
            fact_id=row["fact_id"],
            scope=_scope(row),
            visibility=Visibility(row["visibility"]),
            fact_type=FactType(row["fact_type"]),
            category=_category(row["category"]),
            speaker=Speaker(row["speaker"]),
            claim_type=ClaimType(row["claim_type"]),
            text=row["text"],
            evidence="",
            source_document_id=row["source_document_id"] or "",
            source_episode_id=row["source_episode_id"],
            recorded_at=_ts(row["recorded_at"]),
            valid_from=_ts(row["valid_from"]),
            occurred_at=_ts(row["occurred_at"]),
            valid_to=_ts(row["valid_to"]),
            superseded_by=row["superseded_by"],
            evidence_count=row["evidence_count"],
            entity_ids=tuple(row["entity_ids"] or ()),
        )
    )


def _observation(row: dict) -> Memory:
    return Memory.from_observation(
        Observation(
            observation_id=row["observation_id"],
            scope=_scope(row),
            visibility=Visibility(row["visibility"]),
            text=row["text"],
            fact_ids=tuple(row["fact_ids"] or ()),
            valid_from=_ts(row["valid_from"]),
            entity_id=row["entity_id"],
            valid_to=_ts(row["valid_to"]),
        )
    )


def _episode(rows: list[dict]) -> Memory:
    return Memory.from_episode(
        [
            Document(
                document_id=row["document_id"],
                episode_id=row["episode_id"],
                position=row["position"],
                scope=_scope(row),
                speaker=Speaker(row["speaker"]),
                text=row["text"],
                occurred_at=_ts(row["occurred_at"]),
                recorded_at=_ts(row["recorded_at"]),
                expires_at=_ts(row["expires_at"]),
            )
            for row in rows
        ]
    )


_DECODERS = {"fact": _fact, "observation": _observation, "episode": _episode}
