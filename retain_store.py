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
