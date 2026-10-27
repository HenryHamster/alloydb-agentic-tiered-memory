"""Cross-encoder reranking inside AlloyDB with ai.rank()."""

import threading
from collections.abc import Sequence


class AlloyDBReranker:
    """Scores texts against a query with AlloyDB's ai.rank(); one call per sub-query.

    Reuses one connection (get_db_connection shells out to gcloud for a token, too slow per
    call) and reconnects once if it drops. Rows ai.rank() does not return score 0.

    Signature, model ID and the 1-based `index` follow the AlloyDB docs; not yet run live. It
    needs the Discovery Engine API and is a preview function. The ranking API takes at most 200
    unique records per call, which RecallConfig.rerank_pool stays under.
    """

    def __init__(self, conn_factory, *, model_id: str = "semantic-ranker-default-003"):
        self.conn_factory = conn_factory
        self.model_id = model_id
        self._conn = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        # ai.rank() takes unique strings; repeated texts (common in chat) share one score.
        keys = [text if text.strip() else " " for text in texts]
        documents = list(dict.fromkeys(keys))
        with self._lock:
            rows = self._rank(query, documents)
        by_text = {documents[int(index) - 1]: float(score) for index, score in rows}  # 1-based
        return [by_text.get(key, 0.0) for key in keys]

    def _connection(self):
        if self._conn is None or self._conn.closed:
            conn = self.conn_factory()
            with conn.cursor() as cur:
                cur.execute("SET google_ml_integration.enable_preview_ai_functions = true;")
            conn.commit()
            self._conn = conn
        return self._conn

    def _rank(self, query: str, documents: list[str]) -> list[tuple]:
        for attempt in range(2):
            conn = self._connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT index, score FROM ai.rank("
                        "model_id => %s, search_string => %s, documents => %s::text[]);",
                        (self.model_id, query, documents),
                    )
                    rows = cur.fetchall()
                conn.commit()
                return rows
            except Exception:
                if conn.closed:
                    self._conn = None
                    if attempt == 0:
                        continue
                else:
                    conn.rollback()
                raise
        raise AssertionError("unreachable")
