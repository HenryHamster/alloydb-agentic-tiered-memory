"""Retain pipeline: documents -> chunks -> facts -> entities -> embeddings -> updates -> links."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from retain.entities import EntityResolution, EntityResolver, mention_key, normalize_name
from retain.extraction import FactExtractor, chunk_documents
from retain.linking import Embedder, Linker, SupersedeClassifier, causal_links
from retain.models import (
    Category,
    ClaimType,
    Document,
    EntityRelation,
    ExtractedFact,
    FactEntity,
    MemoryFact,
    MemoryLink,
    Message,
    Observation,
    Relation,
    RetainResult,
    Scope,
    Speaker,
    Supersession,
    UpdateAction,
    UpdateDecision,
    WriteBatch,
    resolve_visibility,
    stable_id,
)
from retain.observations import ObservationBuilder
from retain.store import RetainStore


@dataclass(frozen=True)
class RetainConfig:
    max_chunk_chars: int = 700
    document_ttl: timedelta | None = timedelta(days=90)  # None keeps documents forever


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Retainer:
    """Runs Retain for one conversation round. Stages are injected; see build_retainer."""

    def __init__(
        self,
        *,
        store: RetainStore,
        extractor: FactExtractor,
        entity_resolver: EntityResolver,
        embedder: Embedder,
        linker: Linker,
        classifier: SupersedeClassifier | None = None,
        observation_builder: ObservationBuilder | None = None,
        config: RetainConfig = RetainConfig(),
        clock: Callable[[], datetime] = _utcnow,
    ):
        self.store = store
        self.extractor = extractor
        self.entity_resolver = entity_resolver
        self.embedder = embedder
        self.linker = linker
        self.classifier = classifier
        self.observation_builder = observation_builder
        self.config = config
        self.clock = clock

    def retain_turn(
        self,
        scope: Scope,
        user_msg: str,
        ai_msg: str,
        *,
        timestamp: datetime | None = None,
        episode_id: str | None = None,
    ) -> RetainResult:
        at = timestamp or self.clock()
        messages = [Message(Speaker.USER, user_msg, at), Message(Speaker.ASSISTANT, ai_msg, at)]
        return self.retain(scope, messages, episode_id=episode_id)

    def retain(
        self, scope: Scope, messages: Sequence[Message], *, episode_id: str | None = None
    ) -> RetainResult:
        """Store the round and its facts. Idempotent: a committed episode is skipped.

        Model or store errors propagate and leave the episode unretained, so a retry
        reprocesses it.
        """
        if not messages:
            raise ValueError("retain needs at least one message")
        now = self.clock()
        episode_id = episode_id or _episode_id(scope, messages)
        documents = self._documents(scope, messages, episode_id, now)
        document_ids = [document.document_id for document in documents]
        if self.store.retained_document_ids(document_ids) == set(document_ids):
            return RetainResult(episode_id, None)

        chunks = chunk_documents(documents, self.config.max_chunk_chars)
        documents_by_chunk = {chunk.chunk_id: chunk.document for chunk in chunks}
        extracted = [f for f in self.extractor.extract(chunks) if f.chunk_id in documents_by_chunk]
        mentions = [mention for fact in extracted for mention in fact.entities]
        resolution = (
            self.entity_resolver.resolve(scope.user_id, mentions)
            if mentions
            else EntityResolution({})
        )

        facts_by_key = {
            fact.key: _build_fact(fact, documents_by_chunk[fact.chunk_id], resolution, now)
            for fact in extracted
        }
        new_facts = list({fact.fact_id: fact for fact in facts_by_key.values()}.values())
        if new_facts:
            vectors = self.embedder([fact.text for fact in new_facts])
            if len(vectors) != len(new_facts):
                raise ValueError("Embedder must return one vector per fact")
            new_facts = [replace(f, embedding=tuple(v)) for f, v in zip(new_facts, vectors)]

        decisions = self._decide(new_facts)
        kept, reinforced, supersessions, redirect = _apply_decisions(new_facts, decisions)
        kept_ids = {fact.fact_id for fact in kept}
        stored_id = {key: redirect.get(f.fact_id, f.fact_id) for key, f in facts_by_key.items()}

        links = [
            MemoryLink(change.new_fact_id, change.old_fact_id, Relation.SUPERSEDES)
            for change in supersessions
        ]
        links += causal_links(extracted, stored_id)
        links += self.linker.link(kept) if kept else []
        own = [fact for fact in extracted if stored_id[fact.key] in kept_ids]

        batch = WriteBatch(
            documents=tuple(documents),
            facts=tuple(kept),
            entities=resolution.upserts,
            fact_entities=_fact_entities(own, stored_id, resolution),
            entity_relations=_entity_relations(own, stored_id, resolution),
            links=tuple({(x.from_fact_id, x.to_fact_id, x.relation): x for x in links}.values()),
            supersessions=tuple(supersessions),
            reinforced_fact_ids=tuple(reinforced),
        )
        self.store.write(batch)
        return RetainResult(episode_id, batch, extracted=len(extracted))

    def consolidate(self, reader: Scope) -> list[Observation]:
        """Periodic observation pass (proposal step 4); a no-op without a builder."""
        return self.observation_builder.consolidate(reader) if self.observation_builder else []

    def _documents(self, scope, messages, episode_id, now) -> list[Document]:
        ttl = self.config.document_ttl
        return [
            Document(
                document_id=stable_id(episode_id, position),
                episode_id=episode_id,
                position=position,
                scope=scope,
                speaker=message.speaker,
                text=message.text,
                occurred_at=message.timestamp,
                recorded_at=now,
                expires_at=now + ttl if ttl else None,
            )
            for position, message in enumerate(messages)
        ]

    def _decide(self, facts: list[MemoryFact]) -> list[UpdateDecision]:
        if not facts:
            return []
        if self.classifier is None:
            return [UpdateDecision(fact.fact_id) for fact in facts]
        decisions = list(self.classifier.classify(facts))
        if sorted(d.fact_id for d in decisions) != sorted(f.fact_id for f in facts):
            raise ValueError("Classifier must return exactly one decision per new fact")
        return decisions


def _episode_id(scope: Scope, messages: Sequence[Message]) -> str:
    parts = [(m.speaker, m.timestamp.isoformat(), m.text) for m in messages]
    return stable_id(scope.user_id, scope.project_id, scope.session_id, *parts)


def _build_fact(
    fact: ExtractedFact, document: Document, resolution: EntityResolution, now: datetime
) -> MemoryFact:
    claim_type = fact.claim_type
    # Trust boundary: an assistant can propose a preference, never assert the user's.
    if document.speaker != Speaker.USER and fact.category == Category.PREFERENCE:
        if claim_type == ClaimType.ASSERTED:
            claim_type = ClaimType.PROPOSED
    entity_ids = {resolution.entity_ids[mention_key(m)] for m in fact.entities}
    return MemoryFact(
        fact_id=stable_id(document.document_id, normalize_name(fact.text), claim_type),
        scope=document.scope,
        visibility=resolve_visibility(fact.visibility, document.scope),
        fact_type=fact.fact_type,
        category=fact.category,
        speaker=document.speaker,
        claim_type=claim_type,
        text=fact.text,
        evidence=fact.evidence,
        details=fact.details,
        source_document_id=document.document_id,
        source_episode_id=document.episode_id,
        recorded_at=now,
        occurred_at=fact.occurred_at,
        valid_from=fact.valid_from or fact.occurred_at or document.occurred_at,
        entity_ids=tuple(sorted(entity_ids)),
    )


def _apply_decisions(facts, decisions):
    """Split decisions into rows to insert, facts to reinforce and intervals to close."""
    by_id = {fact.fact_id: fact for fact in facts}
    kept, reinforced, supersessions, redirect = [], [], [], {}
    for decision in decisions:
        fact = by_id[decision.fact_id]
        if decision.action == UpdateAction.DUPLICATE:
            (target,) = decision.target_ids
            reinforced.append(target)
            redirect[fact.fact_id] = target
            continue
        kept.append(fact)
        if decision.action == UpdateAction.SUPERSEDE:
            supersessions += [
                Supersession(old, fact.fact_id, fact.valid_from) for old in decision.target_ids
            ]
    return kept, reinforced, supersessions, redirect


def _fact_entities(facts, stored_id, resolution) -> tuple[FactEntity, ...]:
    rows = {
        FactEntity(stored_id[fact.key], resolution.entity_ids[mention_key(m)], m.role)
        for fact in facts
        for m in fact.entities
    }
    return tuple(sorted(rows, key=lambda row: (row.fact_id, row.entity_id, row.role)))


def _entity_relations(facts, stored_id, resolution) -> tuple[EntityRelation, ...]:
    rows = set()
    for fact in facts:
        ids = {normalize_name(m.name): resolution.entity_ids[mention_key(m)] for m in fact.entities}
        for relation in fact.relations:
            subject = ids.get(normalize_name(relation.subject))
            obj = ids.get(normalize_name(relation.object))
            if subject and obj and subject != obj:
                rows.add(EntityRelation(subject, relation.predicate, obj, stored_id[fact.key]))
    return tuple(sorted(rows, key=lambda row: (row.fact_id, row.subject_entity_id, row.predicate)))


def build_retainer(conn_factory, genai_client, *, config: RetainConfig = RetainConfig()):
    """Production wiring: AlloyDB store with Gemini stages. Stages are skeletons for now."""
    from retain.alloydb_store import AlloyDBRetainStore
    from retain.entities import FuzzyEntityResolver
    from retain.extraction import GeminiFactExtractor
    from retain.linking import GeminiEmbedder, GeminiSupersedeClassifier, StoreLinker
    from retain.observations import GeminiObservationBuilder

    store = AlloyDBRetainStore(conn_factory)
    embedder = GeminiEmbedder(genai_client)
    return Retainer(
        store=store,
        extractor=GeminiFactExtractor(genai_client),
        entity_resolver=FuzzyEntityResolver(store),
        embedder=embedder,
        linker=StoreLinker(store),
        classifier=GeminiSupersedeClassifier(genai_client, store),
        observation_builder=GeminiObservationBuilder(genai_client, store, embedder),
        config=config,
    )
