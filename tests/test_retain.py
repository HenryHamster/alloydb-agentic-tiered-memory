"""Retain checks with no cloud imports, credentials or network access."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from retain import (
    GeminiFactExtractor,
    reconcile_memory_updates,
    resolve_entities,
    retain,
    validate_facts,
)
from retain_models import (
    AtomicFact,
    Category,
    ChangeKind,
    ClaimType,
    MemoryLink,
    MemoryScope,
    MemoryUpdate,
    Message,
    RelationshipType,
    Scope,
    Speaker,
    UpdateCategory,
)
from retain_store import AlloyDBRetainStore
from shared.chunking import chunk_conversation

NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)
SCOPE = Scope("user", "project", "session")


def message(text="I prefer Python.", speaker=Speaker.USER):
    return Message("source", "episode", text, speaker, NOW, SCOPE)


def fact(chunks, text="User prefers Python.", **kwargs):
    return AtomicFact(
        text,
        Category.PREFERENCE,
        ClaimType.ASSERTED,
        tuple(chunk.id for chunk in chunks),
        **kwargs,
    )


def memory(text="User prefers Python."):
    chunks = chunk_conversation([message()])
    return validate_facts([fact(chunks, text, valid_from=NOW)], chunks, recorded_at=NOW)[0]


def test_chunking_is_bounded_lossless_and_preserves_metadata():
    source = message("First topic. " * 12 + "\n\n" + "z" * 65)
    chunks = chunk_conversation([source], max_chars=40)
    assert "".join(chunk.text for chunk in chunks) == source.text
    assert all(0 < len(chunk.text) <= 40 and chunk.message is source for chunk in chunks)
    assert chunks == chunk_conversation([source], max_chars=40)
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert chunks[0].end == chunks[1].start


def test_message_boundaries_and_empty_input():
    messages = [message(), replace(message(), source_id="other", speaker=Speaker.TOOL)]
    assert len(chunk_conversation(messages)) == 2
    assert chunk_conversation([]) == []
    with pytest.raises(ValueError):
        chunk_conversation(messages, max_chars=0)
    with pytest.raises(ValueError):
        chunk_conversation([message(), message()])


@pytest.mark.parametrize("text", ["Python.", "A carefully qualified preference " * 15])
def test_fact_length_is_flexible(text):
    assert memory(text).fact.text == text


@pytest.mark.parametrize("speaker", list(Speaker))
@pytest.mark.parametrize("claim", list(ClaimType))
def test_claim_and_source_are_preserved(speaker, claim):
    chunks = chunk_conversation([message(speaker=speaker)])
    extracted = replace(fact(chunks), claim_type=claim)
    stored = validate_facts([extracted], chunks, recorded_at=NOW)[0]
    assert stored.speaker == speaker
    assert stored.fact.claim_type == claim
    assert stored.source_episode_id == "episode"
    assert stored.evidence_count == 1
    assert stored.recorded_at == NOW


def test_models_reject_bad_scope_dates_categories_and_empty_facts():
    with pytest.raises(ValueError):
        Scope("user", scope=MemoryScope.PROJECT)
    with pytest.raises(ValueError):
        Scope("user", scope="unknown")
    with pytest.raises(ValueError):
        replace(message(), timestamp=NOW.replace(tzinfo=None))
    chunks = chunk_conversation([message()])
    for changes in ({"text": " "}, {"category": "unknown"}, {"source_chunk_ids": ()}):
        with pytest.raises(ValueError):
            replace(fact(chunks), **changes)
    with pytest.raises(ValueError):
        replace(memory(), valid_to=NOW)
    with pytest.raises(ValueError):
        replace(memory(), embedding=(float("nan"),))


def test_validation_rejects_unknown_and_mixed_provenance():
    chunks = chunk_conversation([message()])
    with pytest.raises(ValueError, match="unknown source"):
        validate_facts([replace(fact(chunks), source_chunk_ids=("missing",))], chunks)
    for changed in (
        replace(message(), speaker=Speaker.ASSISTANT),
        replace(message(), scope=Scope("another-user", "project", "session")),
        replace(message(), episode_id="another-episode"),
    ):
        mixed = chunk_conversation([message(), replace(changed, source_id="other")])
        with pytest.raises(ValueError, match="consistent"):
            validate_facts([fact(mixed)], mixed)


@pytest.mark.parametrize("category", list(UpdateCategory))
def test_update_categories_and_explicit_links(category):
    old, new = memory(), memory("User prefers Go.")
    kwargs = {}
    if category == UpdateCategory.SUPERSEDE:
        kwargs = {"change_kind": ChangeKind.HISTORICAL, "effective_at": NOW + timedelta(days=1)}
        new = replace(new, fact=replace(new.fact, valid_from=None))
    update = MemoryUpdate(new, category, old if category != UpdateCategory.ADD else None, **kwargs)
    batch = reconcile_memory_updates([update])
    assert len(batch.insertions) == 1
    assert len(batch.links) == (category != UpdateCategory.ADD)
    if category == UpdateCategory.SUPERSEDE:
        assert batch.replacements[0].valid_to == kwargs["effective_at"]
        assert batch.insertions[0].fact.valid_from == kwargs["effective_at"]
        assert batch.expected_previous == (old,)
        assert old.valid_to is None
        assert batch.links[0].relationship == RelationshipType.SUPERSEDES
    else:
        assert batch.replacements == ()


def test_corrections_are_not_disguised_as_historical_changes():
    update = MemoryUpdate(memory(), UpdateCategory.SUPERSEDE, memory(), ChangeKind.CORRECTION, NOW)
    with pytest.raises(NotImplementedError, match="Corrections"):
        reconcile_memory_updates([update])


def test_invalid_updates_are_rejected():
    old, new = memory(), memory()
    with pytest.raises(ValueError):
        MemoryUpdate(new, UpdateCategory.CONFIRM)
    with pytest.raises(ValueError):
        MemoryUpdate(new, UpdateCategory.ADD, old)
    with pytest.raises(ValueError):
        MemoryUpdate(old, UpdateCategory.CONTEST, old)
    with pytest.raises(ValueError):
        MemoryUpdate(new, UpdateCategory.SUPERSEDE, old)
    other_chunks = chunk_conversation(
        [replace(message(), scope=Scope("other", "project", "session"))]
    )
    other_memory = validate_facts([fact(other_chunks)], other_chunks)[0]
    with pytest.raises(ValueError):
        MemoryUpdate(other_memory, UpdateCategory.CONTEST, old)
    with pytest.raises(ValueError):
        reconcile_memory_updates([MemoryUpdate(new), MemoryUpdate(new)])
    with pytest.raises(ValueError):
        reconcile_memory_updates(
            [MemoryUpdate(new, UpdateCategory.SUPERSEDE, old, ChangeKind.HISTORICAL, NOW)]
        )


@pytest.mark.parametrize("relationship", list(RelationshipType))
def test_relationship_types(relationship):
    assert MemoryLink("a", "b", relationship, SCOPE).target_id == "b"
    with pytest.raises(ValueError):
        MemoryLink("a", "a", relationship, SCOPE)
    with pytest.raises(ValueError):
        MemoryLink("a", "b", "unknown", SCOPE)


def test_pipeline_and_entity_injection():
    class Extractor:
        def extract(self, chunks):
            return [replace(fact(chunks), claim_type=ClaimType.PROPOSED)]

    store = Mock()
    batch = retain(
        [message(speaker=Speaker.ASSISTANT)],
        extractor=Extractor(),
        store=store,
        entity_resolver=lambda memory: ["language:python"],
    )
    store.write.assert_called_once_with(batch)
    stored = batch.insertions[0]
    assert stored.entity_ids == ("language:python",)
    assert stored.speaker == Speaker.ASSISTANT
    assert stored.fact.claim_type == ClaimType.PROPOSED
    assert resolve_entities([stored]) == [stored]


def test_empty_pipeline_does_not_write_or_extract():
    store, extractor = Mock(), Mock()
    assert retain([], extractor=extractor, store=store).insertions == ()
    store.write.assert_not_called()
    extractor.extract.assert_not_called()


def test_cloud_adapters_are_explicit_stubs():
    with pytest.raises(NotImplementedError):
        GeminiFactExtractor().extract([])
    with pytest.raises(NotImplementedError):
        AlloyDBRetainStore().write(reconcile_memory_updates([]))


def test_injected_classification_and_store_failure():
    old = memory()
    extractor = Mock()
    extractor.extract.side_effect = lambda chunks: [fact(chunks, occurred_at=NOW)]
    store = Mock()

    def classify(memories):
        return [MemoryUpdate(memories[0], UpdateCategory.CONFIRM, old)]

    batch = retain([message()], extractor=extractor, store=store, classify_updates=classify)
    assert batch.insertions[0].fact.occurred_at == NOW
    assert batch.links[0].target_id == old.id
    assert batch.links[0].relationship == RelationshipType.CONFIRMS
    store.write.assert_called_once_with(batch)
    store.write.side_effect = RuntimeError("Transaction failed")
    with pytest.raises(RuntimeError, match="Transaction failed"):
        retain([message()], extractor=extractor, store=store)


def test_incomplete_classification_does_not_write():
    extractor, store = Mock(), Mock()
    extractor.extract.side_effect = lambda chunks: [fact(chunks)]
    with pytest.raises(ValueError, match="every extracted memory"):
        retain([message()], extractor=extractor, store=store, classify_updates=lambda _: [])
    store.write.assert_not_called()


@pytest.mark.parametrize("change", ["text", "scope"])
def test_classifier_cannot_replace_validated_memory(change):
    extractor, store = Mock(), Mock()
    extractor.extract.side_effect = lambda chunks: [fact(chunks)]

    def classify(memories):
        original = memories[0]
        if change == "text":
            altered = replace(original, fact=replace(original.fact, text="A different claim."))
        else:
            scope = Scope("another-user", "project", "session")
            sources = tuple(
                replace(source, message=replace(source.message, scope=scope))
                for source in original.sources
            )
            altered = replace(original, scope=scope, sources=sources)
        return [MemoryUpdate(altered)]

    with pytest.raises(ValueError, match="preserve validated memory"):
        retain([message()], extractor=extractor, store=store, classify_updates=classify)
    store.write.assert_not_called()
