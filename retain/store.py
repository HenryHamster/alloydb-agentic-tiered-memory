"""Retain persistence contract and an in-memory reference store for offline use."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from threading import RLock
from typing import Protocol

from retain.models import (
    Category,
    Document,
    Entity,
    EntityRelation,
    FactEntity,
    MemoryFact,
    MemoryLink,
    Observation,
    Scope,
    WriteBatch,
)


class RetainStore(Protocol):
    """Every read filters by user and visibility; every write is one transaction."""

    def retained_document_ids(self, document_ids: Sequence[str]) -> set[str]:
        """IDs among `document_ids` whose retain already committed (ingestion ledger)."""
        ...

    def find_entities(self, user_id: str, canonical_names: Sequence[str]) -> list[Entity]:
        """Candidate entities whose canonical name or alias may match any of the names."""
        ...

    def similar_facts(
        self,
        reader: Scope,
        embedding: Sequence[float],
        *,
        limit: int,
        min_similarity: float = 0.0,
        category: Category | None = None,
    ) -> list[tuple[MemoryFact, float]]:
        """Current, visible facts by descending cosine similarity."""
        ...

    def facts_near_time(
        self, reader: Scope, at: datetime, window: timedelta, *, limit: int
    ) -> list[MemoryFact]:
        """Current, visible facts with occurred_at within `window` of `at`, nearest first."""
        ...

    def facts_with_entities(
        self, reader: Scope, entity_ids: Sequence[str], *, limit: int
    ) -> list[MemoryFact]:
        """Current, visible facts mentioning any of the entities, newest first."""
        ...

    def write(self, batch: WriteBatch) -> None:
        """Apply the batch atomically and mark its documents retained.

        Rejects the whole batch if a link endpoint is missing or a supersession target is
        already closed. Inserts are idempotent on primary keys.
        """
        ...

    def write_observations(self, observations: Sequence[Observation]) -> None: ...

    def purge_expired_documents(self, now: datetime) -> int:
        """Delete documents past expires_at; facts keep their evidence quote."""
        ...


class InMemoryRetainStore:
    """Reference implementation of RetainStore. Not durable; brute-force search."""

    def __init__(self):
        self.documents: dict[str, Document] = {}
        self.retained: set[str] = set()
        self.facts: dict[str, MemoryFact] = {}
        self.entities: dict[str, Entity] = {}
        self.fact_entities: set[FactEntity] = set()
        self.entity_relations: set[EntityRelation] = set()
        self.links: dict[tuple, MemoryLink] = {}
        self.observations: dict[str, Observation] = {}
        self._lock = RLock()

    def retained_document_ids(self, document_ids):
        with self._lock:
            return self.retained & set(document_ids)

    def find_entities(self, user_id, canonical_names):
        with self._lock:
            return [e for e in self.entities.values() if e.user_id == user_id]

    def similar_facts(self, reader, embedding, *, limit, min_similarity=0.0, category=None):
        scored = [
            (fact, _cosine(embedding, fact.embedding))
            for fact in self._readable(reader)
            if fact.embedding and (category is None or fact.category == category)
        ]
        scored = [pair for pair in scored if pair[1] >= min_similarity]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0].fact_id))[:limit]

    def facts_near_time(self, reader, at, window, *, limit):
        near = [
            fact
            for fact in self._readable(reader)
            if fact.occurred_at and abs(fact.occurred_at - at) <= window
        ]
        return sorted(near, key=lambda fact: (abs(fact.occurred_at - at), fact.fact_id))[:limit]

    def facts_with_entities(self, reader, entity_ids, *, limit):
        wanted = set(entity_ids)
        hits = [fact for fact in self._readable(reader) if wanted & set(fact.entity_ids)]
        return sorted(hits, key=lambda fact: fact.recorded_at, reverse=True)[:limit]

    def write(self, batch):
        with self._lock:
            facts = dict(self.facts)
            for fact in batch.facts:
                facts.setdefault(fact.fact_id, fact)
            for fact_id in batch.reinforced_fact_ids:
                fact = facts.get(fact_id)
                if fact is None:
                    raise ValueError(f"Cannot reinforce missing fact {fact_id}")
                facts[fact_id] = replace(fact, evidence_count=fact.evidence_count + 1)
            for change in batch.supersessions:
                old = facts.get(change.old_fact_id)
                if old is None or not old.is_current:
                    raise ValueError(f"Cannot supersede {change.old_fact_id}: missing or closed")
                facts[old.fact_id] = replace(
                    old,
                    valid_to=max(change.valid_to, old.valid_from),
                    superseded_by=change.new_fact_id,
                )
            links = dict(self.links)
            for link in batch.links:
                source, target = facts.get(link.from_fact_id), facts.get(link.to_fact_id)
                if source is None or target is None:
                    raise ValueError("Link endpoint missing")
                if source.scope.user_id != target.scope.user_id:
                    raise ValueError("Links cannot cross users")
                links[(link.from_fact_id, link.to_fact_id, link.relation)] = link
            entities = dict(self.entities)
            for entity in batch.entities:
                old = entities.get(entity.entity_id)
                aliases = tuple(dict.fromkeys((*(old.aliases if old else ()), *entity.aliases)))
                entities[entity.entity_id] = replace(entity, aliases=aliases)

            # Validation done: commit.
            for document in batch.documents:
                self.documents.setdefault(document.document_id, document)
            self.retained |= {document.document_id for document in batch.documents}
            self.facts, self.links, self.entities = facts, links, entities
            self.fact_entities |= set(batch.fact_entities)
            self.entity_relations |= set(batch.entity_relations)

    def write_observations(self, observations):
        with self._lock:
            self.observations |= {o.observation_id: o for o in observations}

    def purge_expired_documents(self, now):
        with self._lock:
            expired = [
                doc_id
                for doc_id, doc in self.documents.items()
                if doc.expires_at and doc.expires_at <= now
            ]
            for doc_id in expired:
                del self.documents[doc_id]
            return len(expired)

    def _readable(self, reader: Scope) -> Iterable[MemoryFact]:
        with self._lock:
            facts = list(self.facts.values())
        return [fact for fact in facts if fact.is_current and fact.visible_to(reader)]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right, strict=True)) / norm if norm else 0.0
