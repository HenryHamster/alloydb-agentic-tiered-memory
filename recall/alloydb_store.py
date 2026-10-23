"""AlloyDB implementation of RecallStore over sql/004_retain_schema.sql."""

# Retain's VISIBLE_SQL (retain/alloydb_store.py) with its validity check swapped: Recall reads
# as of a point in time and may ask for superseded rows, where Retain filters on NOW().
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


class AlloyDBRecallStore:
    """One reused connection from `conn_factory` (db_clients.get_db_connection).

    Read-only: each method runs one statement in a read-only transaction. All values are bound
    parameters (never formatted into SQL text); vectors are passed as '[...]'::vector literals.
    Rows map to recall.models.Memory through Memory.from_fact / from_observation / from_episode.
    """

    def __init__(self, conn_factory):
        self.conn_factory = conn_factory
        self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def search(self, scope, queries, embeddings, window, types, as_of, include_invalid, limit):
        # TODO(recall): one statement, one round trip. Each channel is a CTE returning
        # (channel, rank, id); rows are joined back to their table once at the end.
        #   vector:i   ORDER BY f.embedding <=> %(qvec_i)s::vector LIMIT %(limit)s  (HNSW)
        #   keyword:i  f.text_tsv @@ websearch_to_tsquery('english', %(q_i)s)
        #              ORDER BY ts_rank(f.text_tsv, q) DESC
        #   entity:i   JOIN memory_entities USING (fact_id) WHERE entity_id IN (entities whose
        #              canonical_name or alias matches the query: aliases && words, or
        #              canonical_name %% phrase via pg_trgm), ORDER BY recorded_at DESC
        #   temporal   COALESCE(occurred_at, valid_from) < %(win_end)s
        #              AND COALESCE(occurred_at, valid_to, %(as_of)s) >= %(win_start)s
        # Each fact CTE filters WHERE CANDIDATE_SQL. Union the same CTEs over `observations f`
        # when "observation" is in types, and keyword/temporal over `documents d` grouped by
        # episode_id (WHERE EPISODE_SQL) when "episode" is.
        raise NotImplementedError

    def expand_neighbors(self, scope, seed_ids, types, as_of, include_invalid, limit):
        # TODO(recall): neighbours of %(seeds)s::uuid[] through memory_links in either direction
        #   (ORDER BY weight DESC), then facts sharing a memory_entities.entity_id with a seed;
        #   both WHERE CANDIDATE_SQL AND fact_id <> ALL(%(seeds)s) LIMIT %(limit)s.
        raise NotImplementedError

    def get_episode(self, scope, episode_id):
        # TODO(recall): SELECT speaker, text, occurred_at FROM documents d
        #               WHERE d.episode_id = %(episode_id)s AND EPISODE_SQL ORDER BY position
        #               (no rows -> None: unknown, expired or not readable)
        raise NotImplementedError

    def get_supporting(self, scope, observation_id):
        # TODO(recall): SELECT f.* FROM observations o JOIN memory_facts f
        #               ON f.fact_id = ANY(o.fact_ids)
        #               WHERE o.observation_id = %(id)s AND <CANDIDATE_SQL on o> AND CANDIDATE_SQL
        raise NotImplementedError
