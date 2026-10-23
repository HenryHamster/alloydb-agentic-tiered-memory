"""AlloyDB implementation of RetainStore over sql/004_retain_schema.sql."""

from retain.models import Category

# Read rule for every query; mirrors retain.models.is_visible.
VISIBLE_SQL = """
    f.user_id = %(user_id)s
    AND (f.valid_to IS NULL OR f.valid_to > NOW())
    AND (f.visibility = 'global'
         OR (f.visibility = 'project' AND f.project_id = %(project_id)s)
         OR (f.visibility = 'session' AND f.session_id = %(session_id)s))
"""


class AlloyDBRetainStore:
    """One reused connection from `conn_factory` (db_clients.get_db_connection).

    Each method runs in its own transaction: commit on success, rollback on error, reconnect
    if the connection is closed. All values are bound parameters; vectors are passed as
    '[...]'::vector literals.
    """

    def __init__(self, conn_factory):
        self.conn_factory = conn_factory
        self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def retained_document_ids(self, document_ids):
        # TODO(retain): SELECT document_id FROM documents
        #               WHERE document_id = ANY(%s::uuid[]) AND retained_at IS NOT NULL
        raise NotImplementedError

    def find_entities(self, user_id, canonical_names):
        # TODO(retain): WHERE user_id = %s AND (canonical_name = ANY(%s) OR aliases && %s
        #               OR EXISTS (SELECT 1 FROM unnest(%s) n WHERE canonical_name %% n))
        raise NotImplementedError

    def similar_facts(
        self, reader, embedding, *, limit, min_similarity=0.0, category: Category | None = None
    ):
        # TODO(retain): SELECT f.*, 1 - (f.embedding <=> %(vec)s::vector) AS similarity
        #               WHERE VISIBLE_SQL AND f.embedding IS NOT NULL [AND f.category = ...]
        #               ORDER BY f.embedding <=> %(vec)s::vector LIMIT %(limit)s  (HNSW index)
        raise NotImplementedError

    def facts_near_time(self, reader, at, window, *, limit):
        # TODO(retain): WHERE VISIBLE_SQL AND f.occurred_at BETWEEN at - window AND at + window
        #               ORDER BY abs(extract(epoch FROM f.occurred_at - at)) LIMIT %(limit)s
        raise NotImplementedError

    def get_facts(self, fact_ids):
        # TODO(retain): SELECT f.* (+ entity_ids via memory_entities)
        #               FROM memory_facts f WHERE f.fact_id = ANY(%s::uuid[])
        raise NotImplementedError

    def facts_with_entities(self, reader, entity_ids, *, limit):
        # TODO(retain): JOIN memory_entities USING (fact_id) WHERE VISIBLE_SQL
        #               AND entity_id = ANY(%s::uuid[]) ORDER BY recorded_at DESC
        raise NotImplementedError

    def write(self, batch):
        # TODO(retain): one transaction, in FK order:
        #   documents ON CONFLICT DO NOTHING
        #   entities ON CONFLICT (entity_id) DO UPDATE aliases = distinct union
        #   memory_facts ON CONFLICT DO NOTHING
        #   UPDATE evidence_count + 1 for reinforced_fact_ids
        #   UPDATE valid_to = GREATEST(new valid_to, valid_from), superseded_by
        #     WHERE fact_id = old AND valid_to IS NULL
        #     (rowcount 0 -> raise, rollback)
        #   memory_entities, entity_relations ON CONFLICT DO NOTHING
        #   memory_links ON CONFLICT DO UPDATE weight = GREATEST(weight, EXCLUDED.weight)
        #   UPDATE documents SET retained_at = NOW()
        raise NotImplementedError

    def write_observations(self, observations):
        # TODO(retain): observations ON CONFLICT (observation_id) DO UPDATE text, embedding,
        #               fact_ids, valid_to, updated_at = NOW()
        raise NotImplementedError

    def purge_expired_documents(self, now):
        # TODO(retain): DELETE FROM documents WHERE expires_at <= %s  (facts keep evidence)
        raise NotImplementedError
