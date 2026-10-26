"""Reflect persistence contract and AlloyDB implementation over sql/004 + sql/005."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from reflect.models import FactState, MacroMemory, ReflectBatch
from retain.models import FactEntity, MemoryFact, MemoryLink


class ReflectStore(Protocol):
    """Reads are per user (Reflect is a per-user batch job); writes are one transaction."""

    def user_facts(self, user_id: str) -> list[MemoryFact]:
        """Every stored fact of the user, current or closed, in any tier."""
        ...

    def fact_states(self, fact_ids: Sequence[str]) -> dict[str, FactState]:
        """Lifecycle columns per fact; defaults (hot, rank 0, confidence 1) for untouched rows."""
        ...

    def links_among(self, fact_ids: Sequence[str]) -> list[MemoryLink]:
        """memory_links rows whose both endpoints are in `fact_ids`."""
        ...

    def entity_rows(self, fact_ids: Sequence[str]) -> list[FactEntity]: ...

    def user_macros(self, user_id: str) -> list[MacroMemory]: ...

    def record_access(
        self, fact_ids: Sequence[str], macro_ids: Sequence[str], at: datetime
    ) -> None:
        """Called by Recall for every row it returns: access_count + 1, last_accessed_at = at,
        and an archived row is promoted back immediately (tiers.promote_on_hit)."""
        ...

    def write_reflect(self, batch: ReflectBatch) -> None:
        """Apply atomically. Rejects the whole batch if a referenced fact is missing or a
        supersession target is already closed. Purges delete rows (links and entity rows
        cascade; superseded_by / consolidated_into pointing at them become NULL)."""
        ...


class AlloyDBReflectStore:
    """Same connection handling as retain.alloydb_store.AlloyDBRetainStore.

    Reflect runs as a per-user batch job; take `pg_advisory_xact_lock(hashtext(user_id))` in
    write_reflect so two passes for one user never interleave.
    """

    def __init__(self, conn_factory):
        self.conn_factory = conn_factory
        self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def user_facts(self, user_id):
        # TODO(reflect): SELECT f.*, array_agg(me.entity_id) FROM memory_facts f
        #               LEFT JOIN memory_entities me USING (fact_id) WHERE f.user_id = %s GROUP BY
        raise NotImplementedError

    def fact_states(self, fact_ids):
        # TODO(reflect): SELECT fact_id, access_count, last_accessed_at, rank_score, tier,
        #               tier_changed_at, confidence, consolidated_into, conflicts_with
        #               FROM memory_facts WHERE fact_id = ANY(%s::uuid[])
        raise NotImplementedError

    def links_among(self, fact_ids):
        # TODO(reflect): WHERE from_fact_id = ANY(%(ids)s) AND to_fact_id = ANY(%(ids)s)
        raise NotImplementedError

    def entity_rows(self, fact_ids):
        # TODO(reflect): SELECT fact_id, entity_id, role FROM memory_entities
        #               WHERE fact_id = ANY(%s::uuid[])
        raise NotImplementedError

    def user_macros(self, user_id):
        # TODO(reflect): SELECT * FROM macro_memories WHERE user_id = %s
        raise NotImplementedError

    def record_access(self, fact_ids, macro_ids, at):
        # TODO(reflect): UPDATE memory_facts SET access_count = access_count + 1,
        #               last_accessed_at = %(at)s,
        #               tier = CASE WHEN tier = 'archived' THEN 'warm' ELSE tier END
        #               WHERE fact_id = ANY(%(ids)s); same for macro_memories.
        #               Recall calls this off its hot path (fire-and-forget).
        raise NotImplementedError

    def write_reflect(self, batch):
        # TODO(reflect): one transaction, advisory lock per user, in FK order:
        #   memory_facts INSERT merged facts ON CONFLICT DO NOTHING
        #   memory_entities INSERT ... SELECT merged_id, entity_id, role FROM originals
        #   UPDATE memory_facts SET valid_to, superseded_by WHERE valid_to IS NULL
        #     (rowcount 0 -> raise, rollback)
        #   UPDATE memory_facts SET rank_score, tier, tier_changed_at, confidence,
        #     consolidated_into, conflicts_with FROM unnest(...) (one statement for all states)
        #   memory_links ON CONFLICT DO NOTHING
        #   memory_conflict_reviews ON CONFLICT (review_id) DO NOTHING
        #   macro_memories ON CONFLICT (macro_id) DO UPDATE every column, updated_at = NOW()
        #   DELETE FROM memory_facts WHERE fact_id = ANY(purged) (links/entities cascade)
        #   DELETE FROM macro_memories WHERE macro_id = ANY(purged_macros)
        raise NotImplementedError
