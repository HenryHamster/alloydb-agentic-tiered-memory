"""Persistence boundary; intentionally no SQL until the schema is agreed."""

from typing import Protocol

from retain_models import WriteBatch


class RetainStore(Protocol):
    def write(self, batch: WriteBatch) -> None:
        """Atomically persist facts, cited source messages, replacements and links.

        Compare expected_previous with stored rows under a lock/version check; reject
        stale updates, missing/cross-scope link endpoints and duplicate insertion IDs.
        Recheck previous/new scope.can_reconcile_with(), not origin Scope equality.
        Store MemoryLink as memory_links(source_id, target_id, relationship, scope);
        ordinary PostgreSQL rows/foreign keys suffice, with no graph extension.
        Future traversal must check visible_to(context) for each edge and both
        endpoints at every hop, in addition to temporal/type filters. No weights
        are required; their representation and ranking semantics remain deferred.
        Preserve cited source text, not just the extracted sentence. This batch does
        not contain uncited messages; full episodes require separate source ingestion.
        """
        ...


class AlloyDBRetainStore:
    def write(self, batch: WriteBatch) -> None:
        raise NotImplementedError("AlloyDB writes await an agreed schema and transaction adapter")


class InMemoryRetainStore:
    """Atomic offline reference store; deliberately not durable or a vector index."""

    def __init__(self):
        from threading import RLock

        self.memories = {}
        self.links = set()
        self._lock = RLock()

    def candidates(self, scope):
        with self._lock:
            return tuple(m for m in self.memories.values() if m.scope.can_reconcile_with(scope))

    def write(self, batch):
        from dataclasses import replace

        with self._lock:
            rows = dict(self.memories)
            if len({m.id for m in batch.insertions}) != len(batch.insertions):
                raise ValueError("Duplicate insertion IDs")
            for previous in batch.expected_previous:
                if rows.get(previous.id) != previous:
                    raise ValueError("Stale or missing previous memory")
            expected = {m.id: m for m in batch.expected_previous}
            for memory in batch.replacements:
                previous = expected.get(memory.id)
                if previous is None or replace(previous, valid_to=memory.valid_to) != memory:
                    raise ValueError("Replacement must only close an expected validity interval")
                rows[memory.id] = memory
            for memory in batch.insertions:
                if memory.id in rows:
                    old = rows[memory.id]
                    # Retry/chunk-size changes may alter citation spans and ingestion time.
                    comparable = replace(
                        memory,
                        recorded_at=old.recorded_at,
                        sources=old.sources,
                        fact=replace(memory.fact, source_chunk_ids=old.fact.source_chunk_ids),
                    )
                    if comparable != old:
                        raise ValueError("Conflicting insertion ID")
                else:
                    rows[memory.id] = memory
            links = self.links | set(batch.links)
            for link in links:
                source, target = rows.get(link.source_id), rows.get(link.target_id)
                if source is None or target is None:
                    raise ValueError("Missing link endpoint")
                if not (
                    source.scope.can_reconcile_with(target.scope)
                    and source.scope.can_reconcile_with(link.scope)
                ):
                    raise ValueError("Cross-scope link")
            self.memories, self.links = rows, links
