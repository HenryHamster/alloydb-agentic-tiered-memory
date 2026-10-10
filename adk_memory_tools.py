from typing import Any, Dict, List, Optional

from google.adk.tools import FunctionTool


def make_long_term_memory_tool(db_conn_factory, embed_client: Any, user_id: str) -> FunctionTool:
    """Creates an ADK FunctionTool for vector search over AlloyDB historical records."""

    def search_archived_memory(
        query: str, project_id: Optional[str] = None, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Searches past architecture decisions, historical notes, and old discussions."""
        # 1. Embed query with text-embedding-005
        emb_resp = embed_client.models.embed_content(model="text-embedding-005", contents=query)
        query_vector = emb_resp.embeddings[0].values

        # 2. Cosine distance search on AlloyDB HNSW vector index
        sql = """
            SELECT document, cmetadata, created_at, 1 - (embedding <=> %s::vector) AS similarity
            FROM episodic_memory_embeddings
            WHERE cmetadata->>'user_id' = %s
              AND (%s IS NULL OR cmetadata->>'project_id' = %s)
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """
        conn = db_conn_factory()
        results = []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (str(query_vector), user_id, project_id, project_id, str(query_vector), limit),
                )
                for doc, meta, created_at, similarity in cur.fetchall():
                    results.append(
                        {
                            "document": doc,
                            "metadata": meta,
                            "timestamp": created_at.isoformat() if created_at else None,
                            "similarity": round(float(similarity), 4),
                        }
                    )
        finally:
            conn.close()
        return results

    return FunctionTool(search_archived_memory)
