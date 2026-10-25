"""Cross-encoder reranking inside AlloyDB with ai.rank()."""

from collections.abc import Sequence


class AlloyDBReranker:
    """Scores texts against a query with AlloyDB's ai.rank(); one call per sub-query.

    TODO(recall): verify the ai.rank() signature, the model ID, and that `index` is 1-based on
    the live instance. It needs the Discovery Engine API and is a preview function.
    """

    def __init__(self, conn_factory, *, model_id: str = "semantic-ranker-default-003"):
        self.conn_factory = conn_factory
        self.model_id = model_id

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        conn = self.conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("SET google_ml_integration.enable_preview_ai_functions = true;")
                cur.execute(
                    "SELECT index, score FROM ai.rank("
                    "model_id => %s, search_string => %s, documents => %s::text[]);",
                    (self.model_id, query, list(texts)),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        scores = [0.0] * len(texts)
        for index, score in rows:
            scores[int(index) - 1] = float(score)  # positions into the documents array
        return scores
