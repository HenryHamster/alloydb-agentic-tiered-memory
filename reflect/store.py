"""Reflect persistence contract over the Retain tables plus sql/005_reflect_schema.sql."""

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
