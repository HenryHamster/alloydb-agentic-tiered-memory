"""Persistence boundary; intentionally no SQL until the schema is agreed."""

from typing import Protocol

from retain_models import WriteBatch


class RetainStore(Protocol):
    def write(self, batch: WriteBatch) -> None:
        """Atomically persist facts, cited source messages, replacements and links.

        Compare expected_previous with stored rows under a lock/version check; reject
        stale updates, missing/cross-scope link endpoints and duplicate insertion IDs.
        Preserve cited source text, not just the extracted sentence. This batch does
        not contain uncited messages; full episodes require separate source ingestion.
        """
        ...


class AlloyDBRetainStore:
    def write(self, batch: WriteBatch) -> None:
        raise NotImplementedError("AlloyDB writes await an agreed schema and transaction adapter")
