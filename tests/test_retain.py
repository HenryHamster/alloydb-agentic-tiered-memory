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
SCOPE = Scope("account-42", "memory-service", "session-2026-10-19")


def message(text="I prefer Python.", speaker=Speaker.USER):
    return Message("message-001", "episode-architecture-review", text, speaker, NOW, SCOPE)


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
    assert stored.fact.claim_type == (
        ClaimType.PROPOSED
        if speaker == Speaker.ASSISTANT and claim == ClaimType.ASSERTED
        else claim
    )
    assert stored.source_episode_id == "episode-architecture-review"
    assert stored.evidence_count == 1
    assert stored.recorded_at == NOW


def test_models_reject_bad_scope_dates_categories_and_empty_facts():
    with pytest.raises(ValueError):
        Scope("account-42", scope=MemoryScope.PROJECT)
    with pytest.raises(ValueError):
        Scope("account-42", scope="unknown")
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
        replace(message(), scope=Scope("another-user", "memory-service", "session-2026-10-19")),
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
    expected_links = {
        UpdateCategory.CONFIRM: RelationshipType.CONFIRMS,
        UpdateCategory.REFINE: RelationshipType.REFINES,
        UpdateCategory.SUPERSEDE: RelationshipType.SUPERSEDES,
        UpdateCategory.CONTEST: RelationshipType.CONTESTS,
        UpdateCategory.CONTEXT: RelationshipType.CONTEXT,
    }
    if category == UpdateCategory.ADD:
        assert batch.links == ()
    else:
        assert batch.links == (MemoryLink(new.id, old.id, expected_links[category], new.scope),)
    if category == UpdateCategory.SUPERSEDE:
        assert batch.replacements[0].valid_to == kwargs["effective_at"]
        assert batch.insertions[0].fact.valid_from == kwargs["effective_at"]
        assert batch.expected_previous == (old,)
        assert old.valid_to is None
        assert batch.links[0].relationship == RelationshipType.SUPERSEDES
    else:
        assert batch.replacements == ()


def test_corrections_are_not_disguised_as_historical_changes():
    update = MemoryUpdate(
        memory("Corrected claim."), UpdateCategory.SUPERSEDE, memory(), ChangeKind.CORRECTION, NOW
    )
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
        [replace(message(), scope=Scope("other", "memory-service", "session-2026-10-19"))]
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


def test_links_reject_self_reference_and_unknown_relationships():
    with pytest.raises(ValueError, match="distinct"):
        MemoryLink("fact-python", "fact-python", RelationshipType.SEMANTIC, SCOPE)
    with pytest.raises(ValueError):
        MemoryLink("fact-python", "fact-go", "unknown", SCOPE)


def test_pipeline_and_entity_injection():
    class Extractor:
        def extract(self, chunks):
            return [replace(fact(chunks), claim_type=ClaimType.PROPOSED)]

    store = Mock(spec=["write"])
    batch = retain(
        [message(speaker=Speaker.ASSISTANT)],
        extractor=Extractor(),
        store=store,
        entity_resolver=lambda memory: ["language:python"],
        link_top_k=0,
    )
    store.write.assert_called_once_with(batch)
    stored = batch.insertions[0]
    assert stored.entity_ids == ("language:python",)
    assert stored.speaker == Speaker.ASSISTANT
    assert stored.fact.claim_type == ClaimType.PROPOSED
    assert resolve_entities([stored])[0].entity_ids == stored.entity_ids


def test_empty_pipeline_does_not_write_or_extract():
    store, extractor = Mock(spec=["write"]), Mock()
    assert retain([], extractor=extractor, store=store).insertions == ()
    store.write.assert_not_called()
    extractor.extract.assert_not_called()


def test_cloud_adapters_are_explicit_stubs():
    assert GeminiFactExtractor(Mock()).extract([]) == []
    with pytest.raises(NotImplementedError):
        AlloyDBRetainStore().write(reconcile_memory_updates([]))


def test_injected_classification_and_store_failure():
    old = memory()
    extractor = Mock()
    extractor.extract.side_effect = lambda chunks: [fact(chunks, occurred_at=NOW)]
    store = Mock(spec=["write"])

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
    extractor, store = Mock(), Mock(spec=["write"])
    extractor.extract.side_effect = lambda chunks: [fact(chunks)]
    with pytest.raises(ValueError, match="every extracted memory"):
        retain([message()], extractor=extractor, store=store, classify_updates=lambda _: [])
    store.write.assert_not_called()


@pytest.mark.parametrize("change", ["text", "scope"])
def test_classifier_cannot_replace_validated_memory(change):
    extractor, store = Mock(), Mock(spec=["write"])
    extractor.extract.side_effect = lambda chunks: [fact(chunks)]

    def classify(memories):
        original = memories[0]
        if change == "text":
            altered = replace(original, fact=replace(original.fact, text="A different claim."))
        else:
            scope = Scope("another-user", "memory-service", "session-2026-10-19")
            sources = tuple(
                replace(source, message=replace(source.message, scope=scope))
                for source in original.sources
            )
            altered = replace(original, scope=scope, sources=sources)
        return [MemoryUpdate(altered)]

    with pytest.raises(ValueError, match="preserve validated memory"):
        retain([message()], extractor=extractor, store=store, classify_updates=classify)
    store.write.assert_not_called()


def scoped_memory(scope):
    chunks = chunk_conversation([replace(message(), scope=scope)])
    return validate_facts([fact(chunks, valid_from=NOW)], chunks, recorded_at=NOW)[0]


@pytest.mark.parametrize(
    "context,visible",
    [
        (
            Scope("account-42", "memory-service", "session-2026-10-19"),
            {"global", "project", "session"},
        ),
        (Scope("account-42", "memory-service", "session-next"), {"global", "project"}),
        (Scope("account-42", "billing-service", "session-2026-10-19"), {"global"}),
        (Scope("account-other", "memory-service", "session-2026-10-19"), set()),
        (Scope("account-42", scope=MemoryScope.GLOBAL), {"global"}),
    ],
)
def test_scope_visibility(context, visible):
    for visibility in MemoryScope:
        assert replace(SCOPE, scope=visibility).visible_to(context) == (visibility.value in visible)


@pytest.mark.parametrize("visibility", [MemoryScope.GLOBAL, MemoryScope.PROJECT])
def test_cross_session_reconciliation_preserves_history_and_origin(visibility):
    old = scoped_memory(replace(SCOPE, scope=visibility))
    new = scoped_memory(replace(SCOPE, session_id="new-session", scope=visibility))
    effective = NOW + timedelta(days=1)
    new = replace(new, fact=replace(new.fact, valid_from=effective))
    batch = reconcile_memory_updates(
        [MemoryUpdate(new, UpdateCategory.SUPERSEDE, old, ChangeKind.HISTORICAL, effective)]
    )
    assert batch.replacements == (replace(old, valid_to=effective),)
    assert batch.insertions == (new,)
    assert batch.expected_previous == (old,)
    assert batch.links[0].scope == new.scope
    assert old.scope.session_id == "session-2026-10-19"
    assert new.scope.session_id == "new-session"
    assert batch.replacements[0].sources == old.sources
    assert batch.insertions[0].sources == new.sources


@pytest.mark.parametrize("old_scope", list(MemoryScope))
@pytest.mark.parametrize("new_scope", list(MemoryScope))
def test_cross_visibility_updates_rejected(old_scope, new_scope):
    if old_scope == new_scope:
        return
    old = scoped_memory(replace(SCOPE, scope=old_scope))
    new = scoped_memory(replace(SCOPE, scope=new_scope))
    for category in set(UpdateCategory) - {UpdateCategory.ADD}:
        with pytest.raises(ValueError, match="visibility domain"):
            MemoryUpdate(new, category, old)


@pytest.mark.parametrize("visibility", list(MemoryScope))
@pytest.mark.parametrize("field", ["user_id", "project_id", "session_id"])
def test_reconciliation_isolation(visibility, field):
    old = scoped_memory(replace(SCOPE, scope=visibility))
    other = replace(old.scope, **{field: "other"})
    allowed = field != "user_id" and (
        visibility == MemoryScope.GLOBAL
        or visibility == MemoryScope.PROJECT
        and field == "session_id"
    )
    assert old.scope.can_reconcile_with(other) == allowed
    if not allowed:
        with pytest.raises(ValueError, match="visibility domain"):
            MemoryUpdate(scoped_memory(other), UpdateCategory.CONFIRM, old)


@pytest.mark.parametrize("field", ["project_id", "session_id"])
def test_global_facts_allow_missing_optional_origin(field):
    scope = replace(SCOPE, scope=MemoryScope.GLOBAL, **{field: None})
    assert scoped_memory(scope).scope == scope


@pytest.mark.parametrize("limit", [1, 2, 7, 20, 2000])
def test_unicode_chunk_offsets_reconstruct_every_source(limit):
    sources = [message("é🙂漢字e\u0301. \n\n" * 30), replace(message("x" * 81), source_id="second")]
    chunks = chunk_conversation(sources, max_chars=limit)
    for source in sources:
        spans = [chunk for chunk in chunks if chunk.message is source]
        assert spans[0].start == 0
        assert spans[-1].end == len(source.text)
        assert all(left.end == right.start for left, right in zip(spans, spans[1:]))
        assert all(0 < chunk.end - chunk.start <= limit for chunk in spans)
        assert "".join(source.text[chunk.start : chunk.end] for chunk in spans) == source.text


def test_sentence_boundary_keeps_preference_qualification_together():
    text = (
        "I prefer Python for prototypes. This excludes production services. "
        "For production I require Go and explicit request deadlines."
    )
    chunks = chunk_conversation([message(text)], max_chars=75)
    assert chunks[0].text == "I prefer Python for prototypes. This excludes production services. "
    assert "".join(c.text for c in chunks) == text


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_chunk_limit(limit):
    with pytest.raises(ValueError, match="positive"):
        chunk_conversation([message()], max_chars=limit)


def gemini_row(**changes):
    return {
        "text": "User prefers Python.",
        "category": "preference",
        "claim_type": "asserted",
        "who": "User",
        "what": "prefers Python",
        "when": None,
        "where": None,
        "why": None,
        "occurred_at": None,
        "valid_from": None,
        "entities": [],
        "evidence": "I prefer Python.",
        **changes,
    }


def gemini_client(rows):
    import json

    client = Mock()
    client.models.generate_content.return_value.text = json.dumps(rows)
    return client


def test_gemini_structured_extraction_and_context():
    import json

    source = replace(message("I started a project in September. It uses AlloyDB."), timestamp=None)
    chunks = chunk_conversation([source], max_chars=36)
    client = gemini_client(
        [
            gemini_row(
                text="User started a project in September.",
                category="project",
                what="started a project",
                when="September",
                evidence=chunks[0].text.strip(),
                source_chunk_ids=[chunks[0].id],
            )
        ]
    )
    extracted = GeminiFactExtractor(client).extract(chunks[:1])
    assert extracted[0].what == "started a project"
    assert extracted[0].occurred_at is None
    call = client.models.generate_content.call_args.kwargs
    assert call["config"]["response_json_schema"]["type"] == "array"
    assert json.dumps(source.text) in call["contents"]
    assert '"timestamp": null' in call["contents"]


@pytest.mark.parametrize(
    "rows",
    [
        None,
        {},
        [42],
        [gemini_row(evidence="invented")],
        [gemini_row(occurred_at="2026-09-01")],
        [gemini_row(entities=[{}])],
        [gemini_row(who=12)],
        [gemini_row(text=" ")],
        [gemini_row(category="bad")],
    ],
)
def test_malformed_gemini_output_is_rejected(rows):
    extractor = GeminiFactExtractor(gemini_client(rows))
    with pytest.raises(ValueError, match="Malformed"):
        extractor.extract(chunk_conversation([message()]))


def test_empty_gemini_output_and_invalid_json_never_write():
    from retain_store import InMemoryRetainStore

    client = gemini_client([])
    store = InMemoryRetainStore()
    assert not retain(
        [message("Hello!")], extractor=GeminiFactExtractor(client), store=store
    ).insertions
    client.models.generate_content.return_value.text = "```not json```"
    with pytest.raises(ValueError):
        retain([message()], extractor=GeminiFactExtractor(client), store=store)
    assert not store.memories


def test_duplicate_chunks_and_message_replay_are_idempotent():
    from retain_store import InMemoryRetainStore

    extractor = Mock()
    extractor.extract.side_effect = lambda chunks: [fact([chunk]) for chunk in chunks]
    store = InMemoryRetainStore()
    source = message("I prefer Python. " * 70)
    batch = retain([source], extractor=extractor, store=store, max_chars=350)
    assert len(batch.insertions) == 1
    retain([source], extractor=extractor, store=store, max_chars=700)
    assert len(store.memories) == 1


def test_scoped_entity_and_semantic_links_are_bounded_and_non_destructive():
    from retain_models import Entity
    from retain_store import InMemoryRetainStore

    store = InMemoryRetainStore()
    extractor = Mock()

    def extract(chunks):
        return [
            replace(
                fact(chunks, text=chunks[0].text),
                entities=(Entity("Berkeley", "city", "California"),),
            )
        ]

    extractor.extract.side_effect = extract
    for i in range(5):
        retain(
            [replace(message(f"Event {i}."), source_id=str(i))],
            extractor=extractor,
            store=store,
            embedder=lambda texts: [(1.0, 0.0) for _ in texts],
            link_top_k=1,
        )
    assert len(store.memories) == 5
    assert len(store.links) == 8
    assert all(
        link.relationship in (RelationshipType.ENTITY, RelationshipType.SEMANTIC)
        for link in store.links
    )
    hidden = replace(message("Hidden preference."), scope=replace(SCOPE, project_id="other"))
    batch = retain(
        [hidden],
        extractor=extractor,
        store=store,
        embedder=lambda texts: [(1.0, 0.0) for _ in texts],
    )
    assert not batch.links
    assert all(m.valid_to is None for m in store.memories.values())


def test_contextual_alias_resolution_does_not_merge_city_and_university():
    from retain import canonical_entity
    from retain_models import Entity

    aliases = {
        ("organization", "university california", name): "university:ucb"
        for name in ("berkeley", "uc berkeley", "university of california, berkeley")
    }
    ids = {
        canonical_entity(Entity(name, "organization", "university California"), aliases)
        for name in ("Berkeley", "UC Berkeley", "University of California, Berkeley")
    }
    assert ids == {"university:ucb"}
    assert canonical_entity(Entity("Berkeley", "city", "California"), aliases) not in ids


def test_similarity_does_not_overwrite_negation_conditions_or_global_preferences():
    from retain_store import InMemoryRetainStore

    store, extractor = InMemoryRetainStore(), Mock()
    extractor.extract.side_effect = lambda chunks: [fact(chunks, chunks[0].text)]
    inputs = [
        replace(
            message("User prefers Python globally."), scope=replace(SCOPE, scope=MemoryScope.GLOBAL)
        ),
        message("User avoids Python for this project."),
        replace(message("User prefers Python only for prototypes."), source_id="condition"),
    ]
    for source in inputs:
        retain(
            [source],
            extractor=extractor,
            store=store,
            embedder=lambda texts: [(1.0, 0.0) for _ in texts],
        )
    assert len(store.memories) == 3
    assert all(m.valid_to is None for m in store.memories.values())
    assert len(store.links) == 1
    assert next(iter(store.links)).relationship == RelationshipType.SEMANTIC


def test_store_rolls_back_missing_endpoints_and_stale_updates():
    from retain_models import WriteBatch
    from retain_store import InMemoryRetainStore

    store, old = InMemoryRetainStore(), memory()
    with pytest.raises(ValueError, match="endpoint"):
        store.write(
            WriteBatch(
                (old,), links=(MemoryLink(old.id, "missing", RelationshipType.ENTITY, SCOPE),)
            )
        )
    assert not store.memories
    store.write(WriteBatch((old,)))
    new = memory("User prefers Go.")
    new = replace(new, fact=replace(new.fact, valid_from=NOW + timedelta(days=1)))
    batch = reconcile_memory_updates(
        [
            MemoryUpdate(
                new, UpdateCategory.SUPERSEDE, old, ChangeKind.HISTORICAL, NOW + timedelta(days=1)
            )
        ]
    )
    store.write(batch)
    assert store.memories[old.id].valid_to == NOW + timedelta(days=1)
    assert store.memories[old.id].fact.text == old.fact.text
    with pytest.raises(ValueError, match="Stale"):
        store.write(batch)


def test_embedding_validation_and_client_adapter():
    from retain import GeminiEmbedder, cosine
    from retain_store import InMemoryRetainStore

    client = Mock()
    client.models.embed_content.return_value.embeddings = [Mock(values=[1.0, 0.0])]
    assert GeminiEmbedder(client)(["a", "b"]) == [(1.0, 0.0), (1.0, 0.0)]
    assert client.models.embed_content.call_args.kwargs["model"] == "text-embedding-005"
    for vector in ((), (float("inf"),), (0.0, 0.0)):
        with pytest.raises(ValueError):
            if vector == (0.0, 0.0):
                cosine(vector, vector)
            else:
                replace(memory(), embedding=vector)
    with pytest.raises(ValueError):
        cosine((1.0,), (1.0, 2.0))
    extractor = Mock()
    extractor.extract.side_effect = lambda chunks: [fact(chunks)]
    with pytest.raises(ValueError, match="one vector"):
        retain([message()], extractor=extractor, store=InMemoryRetainStore(), embedder=lambda _: [])


def test_realistic_chunk_sizes_and_source_boundaries():
    text = "\n\n".join(
        [
            "I started an AI memory project at UC Berkeley last month. "
            "It explores long-term memory because our agents forget earlier work. "
            "We use AlloyDB and Gemini, and keep source timestamps for every claim. "
            "The team will evaluate retrieval carefully before changing storage. "
            "I prefer Python for experiments but require SQL for database migrations.",
            "My garden has tomatoes and basil near the kitchen window. "
            "They need watering twice a week during warm weather. "
            "I planted them in September and have not harvested anything yet. "
            "The balcony is shaded, so I avoid plants that require full sun. "
            "I may add mint later, but that is only an idea.",
            "Our travel plan includes a train journey through the mountains. "
            "We rejected renting a car because parking would cost too much. "
            "The departure date remains unknown until the hotel confirms availability. "
            "Please remember that I need a quiet room when traveling for work. "
            "That preference does not apply when camping with friends.",
            "The service deploys with a strict timeout of thirty seconds. "
            "This applies only to production requests that involve external APIs. "
            "Local experiments can run longer because they do not affect customers. "
            "We used to prefer synchronous processing but now prefer a queue. "
            "This is a change in architecture rather than an error in the earlier record. "
            "I will review the next deployment after the offline checks finish tomorrow. "
            "No final deployment decision has been made yet.",
            "I joined a community orchestra in August and practice violin on weekends. "
            "Our rehearsals happen in a school hall near the railway station. "
            "I cannot attend Tuesday rehearsals because I teach an evening class. "
            "The orchestra director suggested moving practice to Thursday, but nobody has confirmed that change. "
            "I previously preferred printed sheet music; for outdoor performances I now use a tablet. "
            "That change happened in September after rain damaged my paper copies. "
            "I still prefer paper when practicing at home because annotations are easier. "
            "Our next concert might happen in December if the hall becomes available. "
            "Please keep that possibility separate from the confirmed rehearsal schedule.",
        ]
    )
    counts = []
    for size in (350, 700, 2000):
        chunks = chunk_conversation([message(text)], max_chars=size)
        counts.append(len(chunks))
        assert "".join(c.text for c in chunks) == text
        assert all(len(c.text) <= size for c in chunks)
    assert counts[0] > counts[1] > counts[2]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. X uses PostgreSQL 16.2. See https://example.com/a.b?q=x. Then continue.",
        "```python\nvalue = 'a.b'\nprint(value)\n```\n- Use SQL\n- Keep dates 2026-10-19\n",
        "one " * 300,
        "🙂 e\u0301\t\r\n" * 90,
        "x" * 1000,
        "I liked coffee. I no longer like coffee.",
    ],
)
def test_realistic_chunk_content_is_lossless(text):
    chunks = chunk_conversation([message(text)], max_chars=350)
    assert "".join(c.text for c in chunks) == text
    assert all(c.text == text[c.start : c.end] for c in chunks)


def test_gemini_structured_extraction_and_evidence():
    import json

    client = Mock()
    chunks = chunk_conversation([message("I moved to Berkeley on May 12, 2026.")])
    row = dict(
        text="User moved to Berkeley on May 12, 2026.",
        category="event",
        claim_type="asserted",
        source_chunk_ids=[chunks[0].id],
        evidence="I moved to Berkeley on May 12, 2026.",
        who="User",
        what="moved",
        when="May 12, 2026",
        where="Berkeley",
        why=None,
        occurred_at="2026-05-12T00:00:00+00:00",
        valid_from=None,
        entities=[],
    )
    client.models.generate_content.return_value.text = json.dumps([row])
    facts = GeminiFactExtractor(client).extract(chunks)
    stored = validate_facts(facts, chunks)[0]
    assert stored.fact.occurred_at.month == 5
    assert stored.recorded_at != stored.fact.occurred_at
    assert stored.fact.why is None
    assert "2026-10-19" in client.models.generate_content.call_args.kwargs["contents"]
    with pytest.raises(ValueError, match="Evidence"):
        validate_facts([replace(facts[0], evidence="invented")], chunks)


@pytest.mark.parametrize("response", ["{}", "null", "broken", "[{}]", '[{"text": 3}]'])
def test_malformed_gemini_fails_before_write(response):
    client = Mock()
    client.models.generate_content.return_value.text = response
    from retain_store import InMemoryRetainStore

    store = InMemoryRetainStore()
    with pytest.raises(ValueError):
        retain([message()], extractor=GeminiFactExtractor(client), store=store)
    assert store.memories == {}


def test_retries_duplicate_chunks_and_scoped_links():
    from retain_models import Entity
    from retain_store import InMemoryRetainStore

    class Extractor:
        def extract(self, chunks):
            return [
                replace(
                    fact(chunks, text=c.message.text),
                    source_chunk_ids=(c.id,),
                    entities=(Entity("Python", "language", "programming"),),
                )
                for c in chunks
                for _ in range(2)
            ]

    store = InMemoryRetainStore()
    first = retain(
        [message()],
        extractor=Extractor(),
        store=store,
        embedder=lambda texts: [(1.0, 0.0) for _ in texts],
    )
    assert len(first.insertions) == 1
    assert not retain([message()], extractor=Extractor(), store=store).insertions
    source = replace(message("Python only for backend tasks."), source_id="second")
    second = retain(
        [source],
        extractor=Extractor(),
        store=store,
        embedder=lambda texts: [(1.0, 0.0) for _ in texts],
    )
    assert {link.relationship for link in second.links} == {
        RelationshipType.ENTITY,
        RelationshipType.SEMANTIC,
    }
    global_source = replace(
        source, source_id="third", scope=replace(SCOPE, scope=MemoryScope.GLOBAL)
    )
    isolated = retain(
        [global_source],
        extractor=Extractor(),
        store=store,
        embedder=lambda texts: [(1.0, 0.0) for _ in texts],
    )
    assert not isolated.links
    assert len(store.memories) == 3


def test_entity_aliases_require_type_and_context():
    from retain import canonical_entity
    from retain_models import Entity

    aliases = {
        ("university", "education", "uc berkeley"): "university:berkeley",
        ("university", "education", "university of california, berkeley"): "university:berkeley",
    }
    assert canonical_entity(
        Entity("UC Berkeley", "university", "education"), aliases
    ) == canonical_entity(
        Entity("University of California, Berkeley", "university", "education"), aliases
    )
    assert (
        canonical_entity(Entity("Berkeley", "city", "California"), aliases) != "university:berkeley"
    )


@pytest.mark.parametrize("limit", [350, 700, 2000])
def test_realistic_multitopic_message_chunk_comparison(limit):
    topics = [
        "I use AlloyDB for durable memory and Gemini for extraction. ",
        "My trip to Berkeley happened in May, months before this conversation. ",
        "I prefer Python for backend work. This preference excludes browser code. ",
        "I rejected Redis as the primary database but still use it for caching. ",
        "The build uses `v1.2.3` and https://example.com/path?q=a.b safely. ",
    ]
    text = "\n\n".join(topic * 5 for topic in topics)
    chunks = chunk_conversation([message(text)], max_chars=limit)
    assert "".join(c.text for c in chunks) == text
    assert all(len(c.text) <= limit for c in chunks)
    assert len(chunks) > 1 if limit < len(text) else len(chunks) == 1


def test_missing_timestamp_and_no_useful_facts():
    client = Mock()
    client.models.generate_content.return_value.text = "[]"
    from retain_store import InMemoryRetainStore

    store = InMemoryRetainStore()
    assert not retain(
        [replace(message("Hello 🙂"), timestamp=None)],
        extractor=GeminiFactExtractor(client),
        store=store,
    ).insertions
    assert not store.memories


def test_fact_spanning_chunks_has_complete_context_and_citations():
    import json

    text = "I prefer Python for prototypes. This does not apply to production."
    chunks = chunk_conversation([message(text)], max_chars=35)
    client = gemini_client(
        [
            gemini_row(
                text="User prefers Python for prototypes, excluding production.",
                evidence=text,
                source_chunk_ids=[c.id for c in chunks],
            )
        ]
    )
    facts = GeminiFactExtractor(client).extract(chunks)
    assert len(facts) == 1
    assert len(validate_facts(facts, chunks)[0].sources) == len(chunks)
    assert json.dumps(text) in client.models.generate_content.call_args.kwargs["contents"]
    client.models.generate_content.assert_called_once()


def test_assistant_preference_cannot_become_asserted_user_preference():
    chunks = chunk_conversation([message("Try Python.", speaker=Speaker.ASSISTANT)])
    stored = validate_facts([fact(chunks)], chunks)[0]
    assert stored.speaker == Speaker.ASSISTANT
    assert stored.fact.claim_type == ClaimType.PROPOSED


def test_unrelated_equal_timestamps_do_not_create_temporal_or_causal_links():
    from retain_store import InMemoryRetainStore

    store, extractor = InMemoryRetainStore(), Mock()
    extractor.extract.side_effect = lambda chunks: [
        fact([c], c.text, occurred_at=NOW) for c in chunks
    ]
    sources = [
        message("User planted basil."),
        replace(message("User updated a database."), source_id="second"),
    ]
    retain(sources, extractor=extractor, store=store)
    assert len(store.memories) == 2
    assert not store.links


def test_bounded_extraction_owns_neighbor_facts_once_and_keeps_offsets():
    import json

    from retain_models import ConversationChunk

    sentences = [
        "I use AlloyDB for durable project memory.\n",
        "I use Gemini to extract atomic project facts.\n",
        "I require source evidence for stored facts.\n",
        "I retain previous preferences as history.\n",
    ]
    source = message("".join(sentences))
    chunks, offset = [], 0
    for index, sentence in enumerate(sentences):
        chunks.append(ConversationChunk(f"chunk-{index}", source, offset, offset + len(sentence)))
        offset += len(sentence)
    client = Mock()
    payloads = []

    def generate(**kwargs):
        payload = json.loads(kwargs["contents"].split("SOURCE JSON:\n")[1])
        payloads.append(payload)
        rows = []
        for chunk in chunks:
            if payload["source_start"] <= chunk.start and chunk.end <= (
                payload["source_start"] + len(payload["source"])
            ):
                rows.append(
                    gemini_row(
                        text=chunk.text.strip(),
                        evidence=chunk.text.strip(),
                        source_chunk_ids=[chunk.id],
                    )
                )
        return Mock(text=json.dumps(rows))

    client.models.generate_content.side_effect = generate
    extractor = GeminiFactExtractor(client, max_context_chars=80, neighbor_chars=50)
    facts = extractor.extract(chunks)
    assert {f.text for f in facts} == {sentence.strip() for sentence in sentences}
    assert len(facts) == len(sentences)
    assert len(payloads) > 1
    for payload in payloads:
        start = payload["source_start"]
        assert len(payload["source"]) <= 180
        assert payload["source"] == source.text[start : start + len(payload["source"])]
    assert all(
        m.scope == SCOPE and m.sources[0].message is source for m in validate_facts(facts, chunks)
    )


def test_bounded_extraction_rejects_hidden_evidence_and_oversized_chunks():
    chunks = chunk_conversation([message("I prefer Python. " * 30)], max_chars=80)
    client = gemini_client([gemini_row(source_chunk_ids=[chunks[-1].id])])
    extractor = GeminiFactExtractor(client, max_context_chars=80, neighbor_chars=0)
    with pytest.raises(ValueError, match="Malformed"):
        extractor.extract(chunks)
    client.models.generate_content.reset_mock()
    with pytest.raises(ValueError, match="Chunk exceeds"):
        extractor.extract(chunk_conversation([message("x" * 81)]))
    client.models.generate_content.assert_not_called()


def test_classifier_duplicate_decisions_fail_before_storage():
    extractor, store = Mock(), Mock(spec=["write"])
    extractor.extract.side_effect = lambda chunks: [fact(chunks)]
    with pytest.raises(ValueError, match="every extracted memory"):
        retain(
            [message()],
            extractor=extractor,
            store=store,
            classify_updates=lambda memories: [MemoryUpdate(memories[0])] * 2,
        )
    store.write.assert_not_called()


def test_evidence_cannot_join_nonadjacent_spans_or_cross_messages():
    chunks = chunk_conversation([message("I prefer Python. " * 20)], max_chars=50)
    selected = [chunks[0], chunks[-1]]
    with pytest.raises(ValueError, match="contiguous"):
        validate_facts([fact(selected, evidence=selected[0].text)], chunks)
    other = chunk_conversation([replace(message(), source_id="message-002")])
    selected = [chunks[0], other[0]]
    with pytest.raises(ValueError, match="contiguous"):
        validate_facts([fact(selected, evidence="I prefer Python.")], [*chunks, *other])


def test_scoped_link_write_cannot_leak_another_user():
    from retain_models import WriteBatch
    from retain_store import InMemoryRetainStore

    store = InMemoryRetainStore()
    own = memory()
    foreign = scoped_memory(replace(SCOPE, user_id="account-other"))
    with pytest.raises(ValueError, match="Cross-scope"):
        store.write(
            WriteBatch(
                (own, foreign),
                links=(MemoryLink(own.id, foreign.id, RelationshipType.ENTITY, own.scope),),
            )
        )
    assert not store.memories and not store.links


def test_repeated_quote_requires_unambiguous_citations():
    from retain_models import ConversationChunk

    prototype = "For prototypes, I said: I prefer Python.\n"
    production = "For production, I said: I prefer Python.\n"
    source = message(prototype + production)
    chunks = [
        ConversationChunk("prototype", source, 0, len(prototype)),
        ConversationChunk("production", source, len(prototype), len(source.text)),
    ]
    row = gemini_row(
        text="User expressed a Python preference for production.",
        evidence="I prefer Python.",
        source_chunk_ids=[c.id for c in chunks],
    )
    client = gemini_client([row])
    with pytest.raises(ValueError, match="Ambiguous"):
        validate_facts([fact(chunks, evidence=row["evidence"])], chunks)
    with pytest.raises(ValueError, match="Malformed"):
        GeminiFactExtractor(client).extract(chunks)
    row["source_chunk_ids"] = [chunks[1].id]
    extracted = GeminiFactExtractor(gemini_client([row])).extract(chunks)
    stored = validate_facts(extracted, chunks)[0]
    assert stored.sources == (chunks[1],)
    assert stored.sources[0].start == len(prototype)
    row["source_chunk_ids"] = [c.id for c in chunks]
    row["evidence"] = production.strip()
    assert len(GeminiFactExtractor(gemini_client([row])).extract(chunks)) == 1


def test_qualified_fact_crosses_window_and_is_owned_once():
    import json

    from retain_models import ConversationChunk

    sentences = [
        "I am planning the next release. ",
        "I prefer Python for internal prototypes. ",
        "This excludes all production services. ",
        "Production services must use Go.",
    ]
    source = message("".join(sentences))
    chunks, offset = [], 0
    for index, sentence in enumerate(sentences):
        chunks.append(ConversationChunk(f"release-{index}", source, offset, offset + len(sentence)))
        offset += len(sentence)
    evidence = sentences[1] + sentences[2].rstrip()
    row = gemini_row(
        text="User prefers Python for internal prototypes, excluding production services.",
        what="prefers Python only for internal prototypes",
        evidence=evidence,
        source_chunk_ids=[chunks[1].id, chunks[2].id],
    )
    client = Mock()
    requests = []

    def generate(**kwargs):
        payload = json.loads(kwargs["contents"].split("SOURCE JSON:\n")[1])
        requests.append(payload)
        return Mock(text=json.dumps([row] if evidence in payload["source"] else []))

    client.models.generate_content.side_effect = generate
    facts = GeminiFactExtractor(client, max_context_chars=45, neighbor_chars=100).extract(chunks)
    assert len(requests) == 4
    assert sum(evidence in r["source"] for r in requests) > 1
    assert requests[1]["core_start"] == chunks[1].start
    assert requests[1]["core_end"] < chunks[2].end
    assert len(facts) == 1
    assert facts[0].text == row["text"]
    assert validate_facts(facts, chunks)[0].sources == (chunks[1], chunks[2])


def test_global_visibility_preserves_known_origin_across_projects():
    origin = replace(SCOPE, scope=MemoryScope.GLOBAL)
    stored = scoped_memory(origin)
    other_project = Scope(origin.user_id, "billing-service", "session-import-review")
    assert stored.scope.visible_to(other_project)
    assert stored.scope.project_id == SCOPE.project_id
    assert stored.scope.session_id == SCOPE.session_id
    assert stored.sources[0].message.scope == origin
    assert not stored.scope.visible_to(replace(other_project, user_id="account-other"))
