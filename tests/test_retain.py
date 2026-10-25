"""Offline checks of Retain contracts and pipeline wiring with fake stages."""

import hashlib
import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from async_worker import AsyncMemoryWorker
from retain import InMemoryRetainStore, Retainer, Scope, build_retainer
from retain.entities import mention_key, normalize_name
from retain.models import (
    Category,
    ClaimType,
    Entity,
    EntityMention,
    EntityResolution,
    EntityRole,
    EntityType,
    ExtractedFact,
    FactType,
    LinkType,
    MemoryFact,
    MemoryLink,
    Relation,
    RelationMention,
    Speaker,
    UpdateAction,
    UpdateDecision,
    Visibility,
    WriteBatch,
    is_visible,
    resolve_visibility,
    stable_id,
)
from shared.chunking import chunk_spans

T0 = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
SCOPE = Scope("user_dev_42", "CloudRetail", "sess_2026_01")


# Fakes -------------------------------------------------------------------------------------


def embed(texts):
    """Bag-of-words hash vectors: shared words give cosine > 0."""
    vectors = []
    for text in texts:
        vector = [0.0] * 32
        for word in normalize_name(text).split():
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 32] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


class ScriptedExtractor:
    """Returns facts whose evidence occurs in a chunk, keyed by evidence text."""

    def __init__(self, *facts):
        self.facts = facts
        self.calls = 0

    def extract(self, chunks):
        self.calls += 1
        found = []
        for key, spec in enumerate(self.facts):
            chunk = next(c for c in chunks if spec["evidence"] in c.text)
            defaults = dict(
                key=key,
                chunk_id=chunk.chunk_id,
                fact_type=FactType.WORLD,
                category=Category.PREFERENCE,
                claim_type=ClaimType.ASSERTED,
                visibility=Visibility.GLOBAL,
            )
            found.append(ExtractedFact(**(defaults | spec)))
        return found


class ExactResolver:
    def resolve(self, user_id, mentions):
        ids = {
            mention_key(m): stable_id(user_id, m.entity_type, normalize_name(m.name))
            for m in mentions
        }
        upserts = tuple(
            Entity(entity_id, user_id, name, name, entity_type)
            for (name, entity_type), entity_id in ids.items()
        )
        return EntityResolution(ids, upserts)


class NoLinks:
    def link(self, facts):
        return []


class ScriptedClassifier:
    def __init__(self, decide):
        self.decide = decide

    def classify(self, facts):
        return [self.decide(fact) for fact in facts]


def retainer(extractor, store=None, **kwargs):
    return Retainer(
        store=store or InMemoryRetainStore(),
        extractor=extractor,
        entity_resolver=ExactResolver(),
        embedder=embed,
        linker=NoLinks(),
        clock=lambda: T0,
        **kwargs,
    )


def stored_fact(text, scope=SCOPE, visibility=Visibility.GLOBAL, **kwargs):
    defaults = dict(
        fact_id=stable_id(text),
        scope=scope,
        visibility=visibility,
        fact_type=FactType.WORLD,
        category=Category.PREFERENCE,
        speaker=Speaker.USER,
        claim_type=ClaimType.ASSERTED,
        text=text,
        evidence=text,
        source_document_id=stable_id("doc", text),
        source_episode_id="ep",
        recorded_at=T0,
        valid_from=T0,
        embedding=tuple(embed([text])[0]),
    )
    return MemoryFact(**(defaults | kwargs))


# Chunking ----------------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 7, 40, 700])
def test_chunks_are_bounded_contiguous_and_lossless(limit):
    text = (
        "Café ☕ naïve.\n\nSecond paragraph, e.g. with `code. here` and https://x.io/a.b ok. " * 5
    )
    spans = chunk_spans(text, limit)
    assert "".join(text[a:b] for a, b in spans) == text
    assert all(0 < b - a <= limit for a, b in spans)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))


def test_chunks_prefer_sentence_ends_and_reject_bad_limits():
    text = "I prefer Python for scripts. " + "Only when the team agrees on it."
    (_, first), *_ = chunk_spans(text, 40)
    assert text[:first].endswith("scripts. ")
    with pytest.raises(ValueError):
        chunk_spans(text, 0)


# Models ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "visibility,reader,expected",
    [
        (Visibility.GLOBAL, Scope("user_dev_42", "Other", "s9"), True),
        (Visibility.GLOBAL, Scope("someone_else"), False),
        (Visibility.PROJECT, Scope("user_dev_42", "CloudRetail", "s9"), True),
        (Visibility.PROJECT, Scope("user_dev_42", "Other", "sess_2026_01"), False),
        (Visibility.SESSION, Scope("user_dev_42", "CloudRetail", "sess_2026_01"), True),
        (Visibility.SESSION, Scope("user_dev_42", "CloudRetail", "s9"), False),
    ],
)
def test_visibility_rule(visibility, reader, expected):
    assert is_visible(visibility, SCOPE, reader) is expected


def test_visibility_narrows_to_available_context_never_to_global():
    no_project = Scope("u", session_id="s")
    assert resolve_visibility(Visibility.PROJECT, no_project) == Visibility.SESSION
    assert resolve_visibility(Visibility.SESSION, Scope("u", "p")) == Visibility.PROJECT
    assert resolve_visibility(Visibility.PROJECT, Scope("u")) == Visibility.GLOBAL
    with pytest.raises(ValueError):
        stored_fact("x", scope=Scope("u"), visibility=Visibility.PROJECT)


def test_every_relation_has_one_link_type_and_decisions_validate_targets():
    assert {r.link_type for r in Relation} == set(LinkType)
    with pytest.raises(ValueError):
        UpdateDecision("f", UpdateAction.DUPLICATE, ("a", "b"))
    with pytest.raises(ValueError):
        UpdateDecision("f", UpdateAction.ADD, ("a",))
    with pytest.raises(ValueError):
        MemoryLink("a", "a", Relation.SIMILAR)


# Pipeline ----------------------------------------------------------------------------------


def test_turn_stores_documents_facts_entities_relations_and_causal_links():
    extractor = ScriptedExtractor(
        dict(
            text="User prefers dark mode across all projects.",
            evidence="dark mode",
            entities=(EntityMention("Dark Mode", EntityType.CONCEPT, EntityRole.OBJECT),),
        ),
        dict(
            text="CloudRetail backend uses FastAPI on AlloyDB.",
            evidence="FastAPI on AlloyDB",
            category=Category.DECISION,
            visibility=Visibility.PROJECT,
            entities=(
                EntityMention("CloudRetail", EntityType.PROJECT, EntityRole.SUBJECT),
                EntityMention("FastAPI", EntityType.TECHNOLOGY, EntityRole.OBJECT),
            ),
            relations=(RelationMention("CloudRetail", "uses", "FastAPI"),),
            caused_by=(0,),
        ),
        dict(text="Assistant suggests light mode instead.", evidence="light mode"),
    )
    store = InMemoryRetainStore()
    result = retainer(extractor, store).retain_turn(
        SCOPE, "I like dark mode. CloudRetail runs FastAPI on AlloyDB.", "Try light mode?"
    )

    assert len(store.documents) == 2 and store.retained == set(store.documents)
    facts = {f.text: f for f in store.facts.values()}
    assert facts["CloudRetail backend uses FastAPI on AlloyDB."].visibility == Visibility.PROJECT
    suggestion = facts["Assistant suggests light mode instead."]
    assert (suggestion.speaker, suggestion.claim_type) == (Speaker.ASSISTANT, ClaimType.PROPOSED)
    assert all(f.embedding and f.valid_from == T0 for f in facts.values())
    assert len(store.entities) == 3 and len(store.fact_entities) == 3
    (relation,) = store.entity_relations
    assert relation.predicate == "uses"
    (link,) = store.links.values()
    assert (link.relation, link.link_type) == (Relation.CAUSED_BY, LinkType.CAUSAL)
    assert result.extracted == 3 and not result.skipped


def test_replayed_episode_is_skipped_without_calling_the_model():
    extractor = ScriptedExtractor(dict(text="User prefers Python 3.11.", evidence="Python"))
    pipeline = retainer(extractor)
    pipeline.retain_turn(SCOPE, "I use Python.", "Noted.", timestamp=T0)
    again = pipeline.retain_turn(SCOPE, "I use Python.", "Noted.", timestamp=T0)
    assert again.skipped and extractor.calls == 1


def test_reusing_episode_id_with_changed_source_is_rejected():
    pipeline = retainer(ScriptedExtractor())
    pipeline.retain_turn(SCOPE, "I use Python.", "Noted.", timestamp=T0, episode_id="source-1")
    with pytest.raises(ValueError, match="Conflicting replay"):
        pipeline.retain_turn(SCOPE, "I use Go.", "Noted.", timestamp=T0, episode_id="source-1")


def test_duplicate_reinforces_and_supersede_closes_validity():
    store = InMemoryRetainStore()
    old = stored_fact("User prefers PostgreSQL.")
    store.write(WriteBatch(documents=(), facts=(old,)))

    def decide(fact):
        if "MySQL" in fact.text:
            return UpdateDecision(fact.fact_id, UpdateAction.SUPERSEDE, (old.fact_id,))
        return UpdateDecision(fact.fact_id, UpdateAction.DUPLICATE, (old.fact_id,))

    later = T0 + timedelta(days=1)
    pipeline = retainer(
        ScriptedExtractor(
            dict(text="User prefers PostgreSQL again.", evidence="Postgres"),
            dict(text="User now prefers MySQL.", evidence="MySQL", valid_from=later),
        ),
        store,
        classifier=ScriptedClassifier(decide),
    )
    pipeline.retain_turn(SCOPE, "Postgres still. Actually MySQL now.", "Ok.")

    closed = store.facts[old.fact_id]
    assert closed.evidence_count == 2 and closed.valid_to == later
    new_id = closed.superseded_by
    assert store.facts[new_id].is_current and len(store.facts) == 2
    assert (new_id, old.fact_id, Relation.SUPERSEDES) in store.links


def test_failed_stage_writes_nothing_and_leaves_episode_retryable():
    extractor = ScriptedExtractor(dict(text="User prefers tabs.", evidence="tabs"))
    store = InMemoryRetainStore()
    wrong_id = ScriptedClassifier(lambda fact: UpdateDecision("not-a-new-fact"))
    with pytest.raises(ValueError, match="one decision per new fact"):
        retainer(extractor, store, classifier=wrong_id).retain_turn(
            SCOPE, "I use tabs.", "Ok.", timestamp=T0
        )
    assert not store.documents and not store.facts
    retry = retainer(extractor, store).retain_turn(SCOPE, "I use tabs.", "Ok.", timestamp=T0)
    assert not retry.skipped and len(store.facts) == 1


@pytest.mark.parametrize(
    "action,target,error",
    [
        (UpdateAction.DUPLICATE, "missing", "Invalid duplicate target"),
        (UpdateAction.SUPERSEDE, "other_user", "Invalid supersede target"),
        (UpdateAction.SUPERSEDE, "other_project", "Invalid supersede target"),
        (UpdateAction.SUPERSEDE, "closed", "Invalid supersede target"),
        (UpdateAction.SUPERSEDE, "global", "project fact cannot supersede global"),
    ],
)
def test_classifier_targets_are_checked_before_writing(action, target, error):
    targets = {
        "other_user": stored_fact("Intruder prefers Go.", scope=Scope("intruder", "CloudRetail")),
        "other_project": stored_fact(
            "Other project uses Go.",
            scope=Scope("user_dev_42", "Other"),
            visibility=Visibility.PROJECT,
        ),
        "closed": stored_fact("User preferred Go.", valid_to=T0),
        "global": stored_fact("User prefers Go everywhere."),
    }
    store = InMemoryRetainStore()
    store.write(WriteBatch((), facts=tuple(targets.values())))
    target_id = targets[target].fact_id if target in targets else target
    classifier = ScriptedClassifier(lambda fact: UpdateDecision(fact.fact_id, action, (target_id,)))
    extractor = ScriptedExtractor(
        dict(text="CloudRetail uses Rust.", evidence="Rust", visibility=Visibility.PROJECT)
    )
    with pytest.raises(ValueError, match=error):
        retainer(extractor, store, classifier=classifier).retain_turn(SCOPE, "Rust here.", "Ok.")
    assert not store.documents
    assert not store.documents
    assert store.facts == {fact.fact_id: fact for fact in targets.values()}


def test_global_duplicate_cannot_target_a_narrower_project_fact():
    project_fact = stored_fact(
        "Project rule uses Go.",
        visibility=Visibility.PROJECT,
        category=Category.PREFERENCE,
    )
    store = InMemoryRetainStore()
    store.write(WriteBatch((), facts=(project_fact,)))
    classifier = ScriptedClassifier(
        lambda new: UpdateDecision(new.fact_id, UpdateAction.DUPLICATE, (project_fact.fact_id,))
    )
    extractor = ScriptedExtractor(
        dict(
            text="User prefers Go globally.",
            evidence="Go",
            category=Category.PREFERENCE,
            visibility=Visibility.GLOBAL,
        )
    )

    with pytest.raises(ValueError, match="global fact cannot use narrower project target"):
        retainer(extractor, store, classifier=classifier).retain_turn(SCOPE, "Go", "Ok.")
    assert not store.documents


def test_store_rejects_bad_batches_atomically():
    store = InMemoryRetainStore()
    fact = stored_fact("User prefers vim.")
    with pytest.raises(ValueError, match="endpoint"):
        store.write(
            WriteBatch(
                (), facts=(fact,), links=(MemoryLink(fact.fact_id, "missing", Relation.SIMILAR),)
            )
        )
    assert not store.facts


def test_store_reads_respect_user_and_visibility():
    store = InMemoryRetainStore()
    mine = stored_fact("Project rule uses FastAPI.", visibility=Visibility.PROJECT)
    other_user = stored_fact(
        "Project rule uses FastAPI too.", scope=Scope("intruder", "CloudRetail")
    )
    store.write(WriteBatch((), facts=(mine, other_user)))
    query = embed(["FastAPI project rule"])[0]
    same_project = store.similar_facts(Scope("user_dev_42", "CloudRetail"), query, limit=5)
    other_project = store.similar_facts(Scope("user_dev_42", "Other"), query, limit=5)
    assert [f.fact_id for f, _ in same_project] == [mine.fact_id]
    assert other_project == []


def test_worker_runs_retain_with_trusted_scope_and_message_time():
    pipeline = MagicMock()
    worker = AsyncMemoryWorker(MagicMock(), MagicMock(), retainer=pipeline)
    worker.enqueue_extraction(
        "user_dev_42", "sess_2026_01", "hi", "hello", project_id="CloudRetail"
    )
    worker.work_queue.put(None)
    worker.worker_thread.join(timeout=5)
    (scope, user_msg, ai_msg), kwargs = pipeline.retain_turn.call_args
    assert scope == SCOPE and (user_msg, ai_msg) == ("hi", "hello")
    assert kwargs["timestamp"].tzinfo is not None


def test_production_wiring_uses_extraction_embedding_and_alloydb_store():
    client = MagicMock()
    client.models.embed_content.return_value.embeddings = [MagicMock(values=[0.25] * 768)]
    production = build_retainer(MagicMock(), client)
    assert production.extractor.extract([]) == []
    assert len(production.embedder(["fact"])[0]) == 768
    from retain.alloydb_store import AlloyDBRetainStore

    assert isinstance(production.store, AlloyDBRetainStore)
