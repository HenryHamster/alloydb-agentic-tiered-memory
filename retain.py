"""Small synchronous Retain pipeline with injected extraction and atomic persistence."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from retain_models import (
    AtomicFact,
    ChangeKind,
    ConversationChunk,
    MemoryFact,
    MemoryLink,
    MemoryUpdate,
    Message,
    RelationshipType,
    UpdateCategory,
    WriteBatch,
)
from retain_store import RetainStore
from shared.chunking import chunk_conversation


class FactExtractor(Protocol):
    def extract(self, chunks: Sequence[ConversationChunk]) -> Sequence[AtomicFact]:
        """Extract self-contained facts, targeting 8–15 words without a hard limit.

        Preserve claim modality and source IDs. Assistant suggestions remain assistant
        claims, never established user preferences. Resolve relative dates only when
        supported by source timestamps; leave unknown dates absent.
        """
        ...


class GeminiFactExtractor:
    def extract(self, chunks: Sequence[ConversationChunk]) -> Sequence[AtomicFact]:
        raise NotImplementedError("Gemini structured-output adapter and prompt remain to be built")


def extract_atomic_facts(
    chunks: Sequence[ConversationChunk], *, extractor: FactExtractor
) -> list[AtomicFact]:
    return list(extractor.extract(chunks)) if chunks else []


def validate_facts(
    facts: Sequence[AtomicFact],
    chunks: Sequence[ConversationChunk],
    *,
    recorded_at: datetime | None = None,
) -> list[MemoryFact]:
    """Bind claims to trusted source metadata; semantic truth checking remains future work."""
    by_id = {chunk.id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    memories = []
    for fact in facts:
        try:
            sources = tuple(by_id[source_id] for source_id in fact.source_chunk_ids)
        except KeyError as error:
            raise ValueError("Fact references an unknown source chunk") from error
        source = sources[0].message
        memories.append(
            MemoryFact(
                id=str(uuid4()),
                fact=fact,
                scope=source.scope,
                speaker=source.speaker,
                source_episode_id=source.episode_id,
                sources=sources,
                recorded_at=recorded_at or datetime.now(timezone.utc),
            )
        )
    return memories


def resolve_entities(
    memories: Sequence[MemoryFact],
    *,
    resolver: Callable[[MemoryFact], Sequence[str]] | None = None,
) -> list[MemoryFact]:
    """Optional scoped entity resolver; omission preserves existing entity IDs."""
    return [
        replace(memory, entity_ids=tuple(resolver(memory))) if resolver else memory
        for memory in memories
    ]


def create_memory_links(updates: Sequence[MemoryUpdate]) -> list[MemoryLink]:
    """Only explicit update relationships; semantic/entity/causal discovery is deferred."""
    relationships = {
        UpdateCategory.CONFIRM: RelationshipType.CONFIRMS,
        UpdateCategory.REFINE: RelationshipType.REFINES,
        UpdateCategory.SUPERSEDE: RelationshipType.SUPERSEDES,
        UpdateCategory.CONTEST: RelationshipType.CONTESTS,
        UpdateCategory.CONTEXT: RelationshipType.CONTEXT,
    }
    return [
        MemoryLink(
            update.memory.id,
            update.previous.id,
            relationships[update.category],
            update.memory.scope,
        )
        for update in updates
        if update.previous is not None
    ]


def reconcile_memory_updates(updates: Sequence[MemoryUpdate]) -> WriteBatch:
    """Apply explicit decisions; this does not infer duplicates or conflicts using an LLM.

    CONFIRM/REFINE/CONTEST/CONTEXT insert attributed evidence and a typed link without
    mutating the old claim. Historical SUPERSEDE plans closure of the old interval;
    atomic persistence is the store's responsibility.
    """
    insertions, replacements, expected = [], [], []
    ids = [update.memory.id for update in updates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate new memory IDs")
    previous_ids = [update.previous.id for update in updates if update.previous is not None]
    if set(ids) & set(previous_ids):
        raise ValueError("Batch cannot reference a newly inserted fact as previous")
    for update in updates:
        memory = update.memory
        if update.previous is not None:
            expected.append(update.previous)
        if update.category == UpdateCategory.SUPERSEDE:
            if update.change_kind == ChangeKind.CORRECTION:
                raise NotImplementedError("Corrections need an agreed invalidation/audit model")
            previous = update.previous
            effective = update.effective_at
            if previous.valid_to is not None or previous.fact.valid_from is None:
                raise ValueError("Supersession requires an open previous interval with a start")
            if memory.valid_to is not None:
                raise ValueError("New version must have an open validity interval")
            if memory.fact.valid_from not in (None, effective):
                raise ValueError("New validity start must match effective_at")
            if any(row.id == previous.id for row in replacements):
                raise ValueError("Cannot supersede a fact twice in one batch")
            replacements.append(replace(previous, valid_to=effective))
            memory = replace(memory, fact=replace(memory.fact, valid_from=effective))
        insertions.append(memory)
    return WriteBatch(
        tuple(insertions), tuple(replacements), tuple(expected), tuple(create_memory_links(updates))
    )


def store_memories(batch: WriteBatch, *, store: RetainStore) -> None:
    store.write(batch)


def retain(
    messages: Sequence[Message],
    *,
    extractor: FactExtractor,
    store: RetainStore,
    max_chars: int = 2000,
    entity_resolver: Callable[[MemoryFact], Sequence[str]] | None = None,
    classify_updates: Callable[[Sequence[MemoryFact]], Sequence[MemoryUpdate]] | None = None,
) -> WriteBatch:
    """Add-only by default; a caller may inject scoped conflict lookup/classification.

    No cloud client is constructed here. Retries/idempotency are not implemented yet.
    """
    chunks = chunk_conversation(messages, max_chars=max_chars)
    facts = extract_atomic_facts(chunks, extractor=extractor)
    memories = resolve_entities(validate_facts(facts, chunks), resolver=entity_resolver)
    validated = {memory.id: memory for memory in memories}
    updates = (
        list(classify_updates(memories))
        if classify_updates
        else [MemoryUpdate(memory) for memory in memories]
    )
    if {update.memory.id for update in updates} != set(validated):
        raise ValueError("Classifier must return one decision for every extracted memory")
    if any(update.memory != validated[update.memory.id] for update in updates):
        raise ValueError("Classifier must preserve validated memory content and provenance")
    batch = reconcile_memory_updates(updates)
    if batch.insertions:
        store_memories(batch, store=store)
    return batch
