"""AlloyDB helpers for benchmark runs over sql/004_retain_schema.sql.

Ingestion writes through Retain's own AlloyDBRetainStore, so AlloyDBRecallStore is measured on
the rows production would hold. This module only removes a run's rows afterwards.
"""

# Children before parents: deleting facts cascades to memory_entities, memory_links and
# entity_relations; observations and documents are only SET NULL targets, so go explicitly.
_TABLES = ("observations", "memory_facts", "entities", "documents")


def delete_users(conn_factory, user_id_prefix: str) -> None:
    """Removes every Retain row whose user_id starts with the prefix, in one transaction."""
    conn = conn_factory()
    try:
        with conn.cursor() as cur:
            for table in _TABLES:
                cur.execute(
                    f"DELETE FROM {table} WHERE starts_with(user_id, %s)", (user_id_prefix,)
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
