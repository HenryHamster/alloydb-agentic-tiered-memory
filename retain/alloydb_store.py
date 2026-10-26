"""AlloyDB implementation of RetainStore over sql/004_retain_schema.sql."""

import json
import math
from contextlib import contextmanager

from psycopg2.extras import RealDictCursor

from retain.models import (
    Category,
    ClaimType,
    Entity,
    EntityType,
    FactType,
    MemoryFact,
    Observation,
    Scope,
    Speaker,
    Visibility,
    is_broader,
    resolve_visibility,
)

VISIBLE_SQL = """
    f.user_id = %(user_id)s
    AND (f.valid_to IS NULL OR f.valid_to > %(at)s)
    AND (f.visibility = 'global'
         OR (f.visibility = 'project' AND f.project_id = %(project_id)s)
         OR (f.visibility = 'session' AND f.session_id = %(session_id)s))
"""

_FACT_COLUMNS = """
    f.fact_id::text, f.user_id, f.project_id, f.session_id, f.visibility, f.fact_type,
    f.category, f.speaker, f.claim_type, f.text, f.evidence,
    f.source_document_id::text, f.source_episode_id, f.recorded_at, f.valid_from,
    f.details, f.occurred_at, f.valid_to, f.superseded_by::text, f.evidence_count,
    f.embedding::text,
    COALESCE(array_agg(DISTINCT me.entity_id::text) FILTER (WHERE me.entity_id IS NOT NULL), '{}')
        AS entity_ids
"""


class AlloyDBRetainStore:
    """RetainStore backed by one lazily-created psycopg2 connection."""

    def __init__(self, conn_factory):
        self.conn_factory = conn_factory
        self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def retained_document_ids(self, document_ids):
        if not document_ids:
            return set()
        with self._transaction() as cur:
            cur.execute(
                "SELECT document_id::text FROM documents "
                "WHERE document_id = ANY(%s::uuid[]) AND retained_at IS NOT NULL",
                (list(document_ids),),
            )
            return {row["document_id"] for row in cur.fetchall()}

    def validate_documents(self, documents):
        """Raise if deterministic source IDs are replayed with different source data."""
        if not documents:
            return
        with self._transaction() as cur:
            for document in documents:
                cur.execute(
                    "SELECT document_id::text, episode_id, position, user_id, project_id, "
                    "session_id, speaker, text, occurred_at, recorded_at, expires_at "
                    "FROM documents WHERE document_id = %s::uuid FOR KEY SHARE",
                    (document.document_id,),
                )
                row = cur.fetchone()
                if row is not None and not _same_document(row, document):
                    raise ValueError(
                        f"Conflicting replay for source document {document.document_id}"
                    )

    def find_entities(self, user_id, canonical_names):
        names = list(dict.fromkeys(name for name in canonical_names if name))
        if not names:
            return []
        with self._transaction() as cur:
            cur.execute(
                "SELECT entity_id::text, user_id, name, canonical_name, entity_type, aliases "
                "FROM entities WHERE user_id = %s AND (canonical_name = ANY(%s) "
                "OR aliases && %s OR EXISTS (SELECT 1 FROM unnest(%s::text[]) n "
                "WHERE canonical_name %% n)) ORDER BY canonical_name, entity_id",
                (user_id, names, names, names),
            )
            return [_entity(row) for row in cur.fetchall()]

    def similar_facts(
        self, reader, embedding, *, limit, min_similarity=0.0, category: Category | None = None
    ):
        if limit <= 0:
            return []
        vector = _vector(embedding)
        params = _reader_params(reader)
        params.update(vector=vector, min_similarity=min_similarity, limit=limit)
        category_sql = " AND f.category = %(category)s" if category is not None else ""
        if category is not None:
            params["category"] = category.value
        query = f"""
            SELECT {_FACT_COLUMNS}, 1 - (f.embedding <=> %(vector)s::vector) AS similarity
            FROM memory_facts f LEFT JOIN memory_entities me USING (fact_id)
            WHERE {VISIBLE_SQL} AND f.embedding IS NOT NULL
              AND 1 - (f.embedding <=> %(vector)s::vector) >= %(min_similarity)s
              {category_sql}
            GROUP BY f.fact_id
            ORDER BY f.embedding <=> %(vector)s::vector, f.fact_id
            LIMIT %(limit)s
        """
        with self._transaction() as cur:
            cur.execute(query, params)
            return [(_fact(row), float(row["similarity"])) for row in cur.fetchall()]

    def facts_near_time(self, reader, at, window, *, limit):
        if limit <= 0:
            return []
        params = _reader_params(reader, at)
        params.update(at=at, start=at - window, end=at + window, limit=limit)
        query = f"""
            SELECT {_FACT_COLUMNS}
            FROM memory_facts f LEFT JOIN memory_entities me USING (fact_id)
            WHERE {VISIBLE_SQL} AND f.occurred_at BETWEEN %(start)s AND %(end)s
            GROUP BY f.fact_id
            ORDER BY abs(extract(epoch FROM f.occurred_at - %(at)s)), f.fact_id
            LIMIT %(limit)s
        """
        with self._transaction() as cur:
            cur.execute(query, params)
            return [_fact(row) for row in cur.fetchall()]

    def get_facts(self, fact_ids):
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return []
        with self._transaction() as cur:
            cur.execute(
                f"SELECT {_FACT_COLUMNS} FROM memory_facts f "
                "LEFT JOIN memory_entities me USING (fact_id) "
                "WHERE f.fact_id = ANY(%s::uuid[]) GROUP BY f.fact_id ORDER BY f.fact_id",
                (ids,),
            )
            return [_fact(row) for row in cur.fetchall()]

    def facts_with_entities(self, reader, entity_ids, *, limit):
        ids = list(dict.fromkeys(entity_ids))
        if not ids or limit <= 0:
            return []
        params = _reader_params(reader)
        params.update(entity_ids=ids, limit=limit)
        query = f"""
            SELECT {_FACT_COLUMNS}
            FROM memory_facts f JOIN memory_entities wanted USING (fact_id)
            LEFT JOIN memory_entities me USING (fact_id)
            WHERE {VISIBLE_SQL} AND wanted.entity_id = ANY(%(entity_ids)s::uuid[])
            GROUP BY f.fact_id
            ORDER BY f.recorded_at DESC, f.fact_id
            LIMIT %(limit)s
        """
        with self._transaction() as cur:
            cur.execute(query, params)
            return [_fact(row) for row in cur.fetchall()]

    def facts_for_observations(self, reader):
        query = f"""
            SELECT {_FACT_COLUMNS}
            FROM memory_facts f JOIN memory_entities me USING (fact_id)
            WHERE {VISIBLE_SQL}
            GROUP BY f.fact_id
            ORDER BY f.recorded_at, f.fact_id
        """
        with self._transaction() as cur:
            cur.execute(query, _reader_params(reader))
            return [_fact(row) for row in cur.fetchall()]

    def observations_for(self, reader):
        params = _reader_params(reader)
        query = """
            SELECT observation_id::text, user_id, project_id, session_id, visibility,
                   entity_id::text, text, fact_ids, valid_from, valid_to, embedding::text
            FROM observations o
            WHERE o.user_id = %(user_id)s
              AND (o.visibility = 'global'
                   OR (o.visibility = 'project' AND o.project_id = %(project_id)s)
                   OR (o.visibility = 'session' AND o.session_id = %(session_id)s))
            ORDER BY o.observation_id
        """
        with self._transaction() as cur:
            cur.execute(query, params)
            return [_observation(row) for row in cur.fetchall()]

    def write(self, batch):
        if not batch.documents and not any(
            (
                batch.facts,
                batch.entities,
                batch.fact_entities,
                batch.entity_relations,
                batch.links,
                batch.supersessions,
                batch.reinforced_fact_ids,
            )
        ):
            return
        with self._transaction() as cur:
            doc_states = [_insert_document(cur, doc) for doc in batch.documents]
            if doc_states and all(state == "retained" for state in doc_states):
                return
            if "retained" in doc_states:
                raise ValueError("Inconsistent partially retained episode")

            for entity in batch.entities:
                _upsert_entity(cur, entity)
            for fact in batch.facts:
                _insert_fact(cur, fact)

            facts_by_id = _load_facts(cur, _batch_fact_ids(batch))
            _validate_fact_sources(batch.facts, cur)
            entities_by_id = _load_entities(
                cur,
                {
                    *(row.entity_id for row in batch.fact_entities),
                    *(row.subject_entity_id for row in batch.entity_relations),
                    *(row.object_entity_id for row in batch.entity_relations),
                },
            )
            _validate_entity_rows(
                batch.fact_entities, batch.entity_relations, facts_by_id, entities_by_id
            )
            _validate_links(batch.links, facts_by_id)
            # Reinforce first: a round may repeat a fact and then supersede it.
            _reinforce(batch.reinforced_fact_ids, batch.documents, cur)
            _apply_supersessions(batch.supersessions, facts_by_id, cur)

            for row in batch.fact_entities:
                cur.execute(
                    "INSERT INTO memory_entities (fact_id, entity_id, role) "
                    "VALUES (%s::uuid, %s::uuid, %s) ON CONFLICT DO NOTHING",
                    (row.fact_id, row.entity_id, row.role.value),
                )
            for row in batch.entity_relations:
                cur.execute(
                    "INSERT INTO entity_relations "
                    "(subject_entity_id, predicate, object_entity_id, fact_id) "
                    "VALUES (%s::uuid, %s, %s::uuid, %s::uuid) ON CONFLICT DO NOTHING",
                    (row.subject_entity_id, row.predicate, row.object_entity_id, row.fact_id),
                )
            for link in batch.links:
                cur.execute(
                    "INSERT INTO memory_links (from_fact_id, to_fact_id, link_type, relation, weight) "
                    "VALUES (%s::uuid, %s::uuid, %s, %s, %s) "
                    "ON CONFLICT (from_fact_id, to_fact_id, relation) DO UPDATE "
                    "SET weight = GREATEST(memory_links.weight, EXCLUDED.weight)",
                    (
                        link.from_fact_id,
                        link.to_fact_id,
                        link.link_type.value,
                        link.relation.value,
                        link.weight,
                    ),
                )
            if batch.documents:
                cur.execute(
                    "UPDATE documents SET retained_at = NOW() "
                    "WHERE document_id = ANY(%s::uuid[]) AND retained_at IS NULL",
                    ([doc.document_id for doc in batch.documents],),
                )

    def write_observations(self, observations):
        if not observations:
            return
        with self._transaction() as cur:
            ordered = sorted(observations, key=lambda observation: observation.valid_to is None)
            lock_keys = sorted({_observation_group_lock_key(item) for item in ordered})
            for lock_key in lock_keys:
                cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,))
            for observation in ordered:
                _validate_observation(observation)
                if observation.valid_to is None:
                    _insert_observation_version(cur, observation)
                else:
                    _close_observation_version(cur, observation)

    def purge_expired_documents(self, now):
        with self._transaction() as cur:
            cur.execute(
                "DELETE FROM documents WHERE expires_at IS NOT NULL AND expires_at <= %s", (now,)
            )
            return cur.rowcount

    @contextmanager
    def _transaction(self):
        conn = self._connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            if getattr(conn, "closed", False):
                conn.close()
                self._conn = None
            raise

    def _connection(self):
        if self._conn is None or getattr(self._conn, "closed", False):
            self._conn = self.conn_factory()
        return self._conn


def _reader_params(reader, at=None):
    from datetime import datetime, timezone

    return {
        "user_id": reader.user_id,
        "project_id": reader.project_id,
        "session_id": reader.session_id,
        "at": at or datetime.now(timezone.utc),
    }


def _vector(values):
    vector = tuple(float(value) for value in values)
    if len(vector) != 768 or any(not math.isfinite(value) for value in vector):
        raise ValueError("Embeddings must be 768-dimensional finite vectors")
    return "[" + ",".join(repr(value) for value in vector) + "]"


def _fact(row):
    embedding = row["embedding"]
    if embedding is not None:
        embedding = tuple(float(value) for value in embedding.strip("[]").split(","))
    details = row["details"]
    if isinstance(details, str):
        details = json.loads(details)
    return MemoryFact(
        fact_id=row["fact_id"],
        scope=Scope(row["user_id"], row["project_id"], row["session_id"]),
        visibility=Visibility(row["visibility"]),
        fact_type=FactType(row["fact_type"]),
        category=Category(row["category"]),
        speaker=Speaker(row["speaker"]),
        claim_type=ClaimType(row["claim_type"]),
        text=row["text"],
        evidence=row["evidence"],
        source_document_id=row["source_document_id"],
        source_episode_id=row["source_episode_id"],
        recorded_at=row["recorded_at"],
        valid_from=row["valid_from"],
        details=details or {},
        occurred_at=row["occurred_at"],
        valid_to=row["valid_to"],
        superseded_by=row["superseded_by"],
        evidence_count=row["evidence_count"],
        embedding=embedding,
        entity_ids=tuple(sorted(row["entity_ids"] or ())),
    )


def _entity(row):
    return Entity(
        entity_id=row["entity_id"],
        user_id=row["user_id"],
        name=row["name"],
        canonical_name=row["canonical_name"],
        entity_type=EntityType(row["entity_type"]),
        aliases=tuple(row["aliases"] or ()),
    )


def _observation(row):
    embedding = row["embedding"]
    if embedding is not None:
        embedding = tuple(float(value) for value in embedding.strip("[]").split(","))
    return Observation(
        observation_id=row["observation_id"],
        scope=Scope(row["user_id"], row["project_id"], row["session_id"]),
        visibility=Visibility(row["visibility"]),
        text=row["text"],
        fact_ids=tuple(str(fact_id) for fact_id in row["fact_ids"]),
        valid_from=row["valid_from"],
        entity_id=row["entity_id"],
        valid_to=row["valid_to"],
        embedding=embedding,
    )


def _insert_document(cur, document):
    cur.execute(
        "INSERT INTO documents "
        "(document_id, episode_id, position, user_id, project_id, session_id, speaker, text, "
        "occurred_at, recorded_at, expires_at) "
        "VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (document_id) DO NOTHING RETURNING document_id::text",
        (
            document.document_id,
            document.episode_id,
            document.position,
            document.scope.user_id,
            document.scope.project_id,
            document.scope.session_id,
            document.speaker.value,
            document.text,
            document.occurred_at,
            document.recorded_at,
            document.expires_at,
        ),
    )
    if cur.fetchone() is not None:
        return "new"
    cur.execute(
        "SELECT document_id::text, episode_id, position, user_id, project_id, session_id, "
        "speaker, text, occurred_at, recorded_at, expires_at, retained_at "
        "FROM documents WHERE document_id = %s::uuid FOR UPDATE",
        (document.document_id,),
    )
    row = cur.fetchone()
    if row is None or not _same_document(row, document):
        raise ValueError(f"Conflicting replay for source document {document.document_id}")
    return "retained" if row["retained_at"] is not None else "pending"


def _same_document(row, doc):
    return (
        row["document_id"] == doc.document_id
        and row["episode_id"] == doc.episode_id
        and row["position"] == doc.position
        and row["user_id"] == doc.scope.user_id
        and row["project_id"] == doc.scope.project_id
        and row["session_id"] == doc.scope.session_id
        and row["speaker"] == doc.speaker.value
        and row["text"] == doc.text
        and row["occurred_at"] == doc.occurred_at
    )


def _upsert_entity(cur, entity):
    cur.execute(
        "INSERT INTO entities (entity_id, user_id, name, canonical_name, entity_type, aliases) "
        "VALUES (%s::uuid, %s, %s, %s, %s, %s) "
        "ON CONFLICT (entity_id) DO UPDATE SET "
        "aliases = ARRAY(SELECT DISTINCT value FROM unnest(entities.aliases || EXCLUDED.aliases) value), "
        "updated_at = NOW() "
        "WHERE entities.user_id = EXCLUDED.user_id "
        "AND entities.entity_type = EXCLUDED.entity_type "
        "AND entities.canonical_name = EXCLUDED.canonical_name",
        (
            entity.entity_id,
            entity.user_id,
            entity.name,
            entity.canonical_name,
            entity.entity_type.value,
            list(entity.aliases),
        ),
    )
    if cur.rowcount == 0:
        raise ValueError(f"Conflicting entity ID {entity.entity_id}")


def _insert_fact(cur, fact):
    embedding = _vector(fact.embedding) if fact.embedding is not None else None
    cur.execute(
        "INSERT INTO memory_facts "
        "(fact_id, user_id, project_id, session_id, visibility, fact_type, category, speaker, "
        "claim_type, text, evidence, details, source_document_id, source_episode_id, embedding, "
        "occurred_at, recorded_at, valid_from, valid_to, superseded_by, evidence_count) "
        "VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::uuid, %s, "
        "%s::vector, %s, %s, %s, %s, %s::uuid, %s) "
        "ON CONFLICT (fact_id) DO NOTHING RETURNING fact_id",
        (
            fact.fact_id,
            fact.scope.user_id,
            fact.scope.project_id,
            fact.scope.session_id,
            fact.visibility.value,
            fact.fact_type.value,
            fact.category.value,
            fact.speaker.value,
            fact.claim_type.value,
            fact.text,
            fact.evidence,
            json.dumps(fact.details),
            fact.source_document_id,
            fact.source_episode_id,
            embedding,
            fact.occurred_at,
            fact.recorded_at,
            fact.valid_from,
            fact.valid_to,
            fact.superseded_by,
            fact.evidence_count,
        ),
    )
    if cur.fetchone() is None:
        existing = _load_facts(cur, {fact.fact_id}).get(fact.fact_id)
        if existing is None or not _same_fact(existing, fact):
            raise ValueError(f"Conflicting replay for fact {fact.fact_id}")


def _same_fact(left, right):
    return all(
        getattr(left, key) == getattr(right, key)
        for key in (
            "scope",
            "visibility",
            "fact_type",
            "category",
            "speaker",
            "claim_type",
            "text",
            "evidence",
            "source_document_id",
            "source_episode_id",
            "recorded_at",
            "valid_from",
            "details",
            "occurred_at",
            "valid_to",
            "superseded_by",
            "evidence_count",
            "embedding",
            "entity_ids",
        )
    )


def _batch_fact_ids(batch):
    return {
        *(fact.fact_id for fact in batch.facts),
        *(change.old_fact_id for change in batch.supersessions),
        *(change.new_fact_id for change in batch.supersessions),
        *(link.from_fact_id for link in batch.links),
        *(link.to_fact_id for link in batch.links),
        *(row.fact_id for row in batch.fact_entities),
        *(row.fact_id for row in batch.entity_relations),
    }


def _load_facts(cur, fact_ids):
    if not fact_ids:
        return {}
    cur.execute(
        f"SELECT {_FACT_COLUMNS} FROM memory_facts f "
        "LEFT JOIN memory_entities me USING (fact_id) "
        "WHERE f.fact_id = ANY(%s::uuid[]) GROUP BY f.fact_id",
        (list(fact_ids),),
    )
    return {row["fact_id"]: _fact(row) for row in cur.fetchall()}


def _validate_fact_sources(facts, cur):
    if not facts:
        return
    document_ids = {fact.source_document_id for fact in facts}
    cur.execute(
        "SELECT document_id::text, episode_id, user_id, project_id, session_id, speaker "
        "FROM documents WHERE document_id = ANY(%s::uuid[])",
        (list(document_ids),),
    )
    documents = {row["document_id"]: row for row in cur.fetchall()}
    for fact in facts:
        document = documents.get(fact.source_document_id)
        if (
            document is None
            or document["episode_id"] != fact.source_episode_id
            or document["user_id"] != fact.scope.user_id
            or document["project_id"] != fact.scope.project_id
            or document["session_id"] != fact.scope.session_id
            or document["speaker"] != fact.speaker.value
        ):
            raise ValueError(f"Fact source does not match document {fact.source_document_id}")


def _load_entities(cur, entity_ids):
    if not entity_ids:
        return {}
    cur.execute(
        "SELECT entity_id::text, user_id, name, canonical_name, entity_type, aliases "
        "FROM entities WHERE entity_id = ANY(%s::uuid[])",
        (list(entity_ids),),
    )
    return {row["entity_id"]: _entity(row) for row in cur.fetchall()}


def _validate_entity_rows(fact_entities, relations, facts, entities):
    for row in fact_entities:
        fact, entity = facts.get(row.fact_id), entities.get(row.entity_id)
        if fact is None or entity is None or fact.scope.user_id != entity.user_id:
            raise ValueError("Fact entity endpoint missing or belongs to another user")
    for row in relations:
        fact = facts.get(row.fact_id)
        subject, obj = entities.get(row.subject_entity_id), entities.get(row.object_entity_id)
        if (
            fact is None
            or subject is None
            or obj is None
            or fact.scope.user_id != subject.user_id
            or fact.scope.user_id != obj.user_id
        ):
            raise ValueError("Entity relation endpoint missing or belongs to another user")


def _validate_links(links, facts):
    for link in links:
        source, target = facts.get(link.from_fact_id), facts.get(link.to_fact_id)
        if source is None or target is None:
            raise ValueError("Link endpoint missing")
        if not _same_visibility_domain(source, target):
            raise ValueError("Link endpoints do not share a visibility domain")


def _same_visibility_domain(left, right):
    if left.scope.user_id != right.scope.user_id:
        return False
    if left.visibility == Visibility.GLOBAL or right.visibility == Visibility.GLOBAL:
        return True
    if left.visibility == Visibility.PROJECT and right.visibility == Visibility.PROJECT:
        return left.scope.project_id == right.scope.project_id
    if left.visibility == Visibility.SESSION and right.visibility == Visibility.SESSION:
        return left.scope.session_id == right.scope.session_id
    project_fact = left if left.visibility == Visibility.PROJECT else right
    session_fact = left if left.visibility == Visibility.SESSION else right
    return session_fact.scope.project_id in (None, project_fact.scope.project_id)


def _apply_supersessions(changes, facts, cur):
    for change in changes:
        old, new = facts.get(change.old_fact_id), facts.get(change.new_fact_id)
        if (
            old is None
            or new is None
            or not old.is_current
            or not new.is_current
            or old.fact_id == new.fact_id
        ):
            raise ValueError(f"Cannot supersede {change.old_fact_id}: missing or closed")
        if not _same_visibility_domain(old, new):
            raise ValueError("Supersession target is outside the visibility domain")
        if is_broader(old.visibility, new.visibility):
            raise ValueError("A narrower fact cannot supersede a broader fact")
        end = max(change.valid_to, old.valid_from)
        cur.execute(
            "UPDATE memory_facts SET valid_to = %s, superseded_by = %s::uuid "
            "WHERE fact_id = %s::uuid AND valid_to IS NULL",
            (end, new.fact_id, old.fact_id),
        )
        if cur.rowcount != 1:
            raise ValueError(f"Cannot supersede {change.old_fact_id}: concurrent update")


def _reinforce(fact_ids, documents, cur):
    if not fact_ids:
        return
    document_scopes = [doc.scope for doc in documents]
    cur.execute(
        "SELECT fact_id::text, user_id, project_id, session_id, visibility, valid_to "
        "FROM memory_facts WHERE fact_id = ANY(%s::uuid[]) FOR UPDATE",
        (list(set(fact_ids)),),
    )
    rows = {row["fact_id"]: row for row in cur.fetchall()}
    if len(rows) != len(set(fact_ids)):
        raise ValueError("Cannot reinforce missing fact")
    for fact_id in set(fact_ids):
        row = rows[fact_id]
        origin = Scope(row["user_id"], row["project_id"], row["session_id"])
        visibility = Visibility(row["visibility"])
        if row["valid_to"] is not None or not any(
            _scope_can_read(scope, visibility, origin) for scope in document_scopes
        ):
            raise ValueError("Cannot reinforce fact outside the current visibility domain")
    cur.execute(
        "UPDATE memory_facts SET evidence_count = evidence_count + 1 "
        "WHERE fact_id = ANY(%s::uuid[]) AND valid_to IS NULL",
        (list(set(fact_ids)),),
    )
    if cur.rowcount != len(set(fact_ids)):
        raise ValueError("Cannot reinforce missing or closed fact")


def _scope_can_read(reader, visibility, origin):
    if reader.user_id != origin.user_id:
        return False
    if visibility == Visibility.GLOBAL:
        return True
    if visibility == Visibility.PROJECT:
        return reader.project_id == origin.project_id
    return reader.session_id == origin.session_id


def _observation_can_include(
    observation_visibility, observation_scope, fact_visibility, fact_scope
):
    if observation_scope.user_id != fact_scope.user_id:
        return False
    if observation_visibility == Visibility.GLOBAL:
        return fact_visibility == Visibility.GLOBAL
    if fact_visibility == Visibility.GLOBAL:
        return True
    if observation_visibility == Visibility.PROJECT:
        return (
            fact_visibility == Visibility.PROJECT
            and fact_scope.project_id == observation_scope.project_id
        )
    if fact_visibility == Visibility.PROJECT:
        return fact_scope.project_id == observation_scope.project_id
    return fact_scope.session_id == observation_scope.session_id


def _validate_observation(observation):
    if resolve_visibility(observation.visibility, observation.scope) != observation.visibility:
        raise ValueError("Observation visibility needs the matching scope")
    if not observation.text.strip():
        raise ValueError("Observation text cannot be empty")
    if observation.valid_to is not None and observation.valid_to < observation.valid_from:
        raise ValueError("Observation valid_to precedes valid_from")
    if observation.valid_from.utcoffset() is None or (
        observation.valid_to is not None and observation.valid_to.utcoffset() is None
    ):
        raise ValueError("Observation timestamps must be timezone-aware")
    _vector(observation.embedding) if observation.embedding is not None else None


def _validate_observation_references(cur, observation):
    cur.execute(
        "SELECT user_id FROM entities WHERE entity_id = %s::uuid",
        (observation.entity_id,),
    ) if observation.entity_id else None
    if observation.entity_id:
        row = cur.fetchone()
        if row is None or row["user_id"] != observation.scope.user_id:
            raise ValueError("Observation entity is missing or belongs to another user")
    if observation.fact_ids:
        cur.execute(
            "SELECT fact_id::text, user_id, project_id, session_id, visibility "
            "FROM memory_facts WHERE fact_id = ANY(%s::uuid[])",
            (list(observation.fact_ids),),
        )
        rows = cur.fetchall()
        if len(rows) != len(set(observation.fact_ids)):
            raise ValueError("Observation references a missing fact")
        for row in rows:
            origin = Scope(row["user_id"], row["project_id"], row["session_id"])
            if not _observation_can_include(
                observation.visibility, observation.scope, Visibility(row["visibility"]), origin
            ):
                raise ValueError("Observation references a fact outside its visibility domain")


_OBSERVATION_COLUMNS = (
    "observation_id::text, user_id, project_id, session_id, visibility, entity_id::text, "
    "text, fact_ids, valid_from, valid_to, embedding::text"
)


def _observation_version(cur, observation_id):
    cur.execute(
        f"SELECT {_OBSERVATION_COLUMNS} FROM observations "
        "WHERE observation_id = %s::uuid FOR UPDATE",
        (observation_id,),
    )
    row = cur.fetchone()
    return _observation(row) if row is not None else None


def _observation_group_lock_key(observation):
    return json.dumps(
        [
            observation.scope.user_id,
            observation.scope.project_id,
            observation.scope.session_id,
            observation.visibility.value,
            observation.entity_id,
        ],
        separators=(",", ":"),
    )


def _insert_observation_version(cur, observation):
    _validate_observation_references(cur, observation)
    cur.execute(
        "SELECT observation_id::text FROM observations "
        "WHERE user_id = %s AND project_id IS NOT DISTINCT FROM %s "
        "AND session_id IS NOT DISTINCT FROM %s AND visibility = %s "
        "AND entity_id IS NOT DISTINCT FROM %s::uuid AND valid_to IS NULL FOR UPDATE",
        (
            observation.scope.user_id,
            observation.scope.project_id,
            observation.scope.session_id,
            observation.visibility.value,
            observation.entity_id,
        ),
    )
    current_ids = {row["observation_id"] for row in cur.fetchall()}
    if current_ids - {observation.observation_id}:
        raise ValueError("Observation group already has another current version")
    embedding = _vector(observation.embedding) if observation.embedding is not None else None
    cur.execute(
        "INSERT INTO observations "
        "(observation_id, user_id, project_id, session_id, visibility, entity_id, text, embedding, "
        "fact_ids, valid_from, valid_to) "
        "VALUES (%s::uuid, %s, %s, %s, %s, %s::uuid, %s, %s::vector, %s::uuid[], %s, %s) "
        "ON CONFLICT (observation_id) DO NOTHING",
        (
            observation.observation_id,
            observation.scope.user_id,
            observation.scope.project_id,
            observation.scope.session_id,
            observation.visibility.value,
            observation.entity_id,
            observation.text,
            embedding,
            list(observation.fact_ids),
            observation.valid_from,
            observation.valid_to,
        ),
    )
    if cur.rowcount == 0:
        existing = _observation_version(cur, observation.observation_id)
        if (
            existing is None
            or existing.valid_to is not None
            or not existing.same_version(observation)
        ):
            raise ValueError(f"Conflicting observation ID {observation.observation_id}")


def _close_observation_version(cur, observation):
    existing = _observation_version(cur, observation.observation_id)
    if existing is None or not existing.same_version(observation):
        raise ValueError(
            f"Cannot close missing or changed observation {observation.observation_id}"
        )
    if existing.valid_to is not None:
        return
    cur.execute(
        "UPDATE observations SET valid_to = %s, updated_at = NOW() "
        "WHERE observation_id = %s::uuid AND valid_to IS NULL",
        (observation.valid_to, observation.observation_id),
    )
    if cur.rowcount != 1:
        raise ValueError(f"Observation version changed concurrently: {observation.observation_id}")
