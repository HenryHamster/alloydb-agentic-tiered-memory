"""Retain pipeline: documents -> chunks -> facts -> entities -> embeddings -> updates -> links."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from retain.embeddings import Embedder
from retain.entities import EntityResolver, mention_key, normalize_name
from retain.extraction import FactExtractor, chunk_documents
from retain.linking import Linker, SupersedeClassifier, causal_links
from retain.models import (
    Category,
    ClaimType,
    Document,
    EntityRelation,
    EntityResolution,
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
    is_broader,
    resolve_visibility,
    stable_id,
)
from retain.observations import ObservationBuilder
from retain.store import RetainStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetainConfig:
    max_chunk_chars: int = 700
    document_ttl: timedelta | None = timedelta(days=90)  # None keeps documents forever
    # Every N retained rounds per scope: consolidate observations, purge expired documents.
    maintenance_every: int = 20  # 0 disables


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
        self._rounds_since_maintenance: dict[Scope, int] = {}

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
        validate_documents = getattr(self.store, "validate_documents", None)
        if validate_documents is not None:
            validate_documents(documents)
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

        built = {
            fact.key: _build_fact(fact, documents_by_chunk[fact.chunk_id], resolution, now)
            for fact in extracted
        }  # key -> (fact, entity rows, relation rows)
        new_facts = list({fact.fact_id: fact for fact, _, _ in built.values()}.values())
        if new_facts:
            vectors = self.embedder([fact.text for fact in new_facts])
            if len(vectors) != len(new_facts):
                raise ValueError("Embedder must return one vector per fact")
            new_facts = [replace(f, embedding=tuple(v)) for f, v in zip(new_facts, vectors)]

        kept, reinforced, supersessions, redirect = _apply_decisions(
            new_facts, self._decide(new_facts)
        )
        stored_id = {key: redirect.get(f.fact_id, f.fact_id) for key, (f, _, _) in built.items()}
        kept_ids = {fact.fact_id for fact in kept}
        entity_rows = [r for f, rows, _ in built.values() if f.fact_id in kept_ids for r in rows]
        relation_rows = [r for f, _, rows in built.values() if f.fact_id in kept_ids for r in rows]

        links = [
            MemoryLink(change.new_fact_id, change.old_fact_id, Relation.SUPERSEDES)
            for change in supersessions
        ]
        links += causal_links(extracted, stored_id)
        links += self.linker.link(kept) if kept else []

        batch = WriteBatch(
            documents=tuple(documents),
            facts=tuple(kept),
            entities=resolution.upserts,
            fact_entities=tuple(dict.fromkeys(entity_rows)),
            entity_relations=tuple(dict.fromkeys(relation_rows)),
            links=tuple({(x.from_fact_id, x.to_fact_id, x.relation): x for x in links}.values()),
            supersessions=tuple(supersessions),
            reinforced_fact_ids=tuple(reinforced),
        )
        self.store.write(batch)
        self._maybe_maintain(scope, now)
        return RetainResult(episode_id, batch, extracted=len(extracted))

    def _maybe_maintain(self, scope: Scope, now: datetime) -> None:
        """Proposal step 4 and document retention, run after the round is committed."""
        every = self.config.maintenance_every
        if every <= 0:
            return
        count = self._rounds_since_maintenance.get(scope, 0) + 1
        self._rounds_since_maintenance[scope] = count % every
        if count < every:
            return
        try:
            self.consolidate(scope)
            self.store.purge_expired_documents(now)
        except Exception:
            # The round is already stored; maintenance retries after the next N rounds.
            logger.warning("Retain maintenance failed for %s", scope, exc_info=True)

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
        """Classify new facts, then enforce the trust boundary on the model's targets.

        One decision per new fact. A missing, repeated or invalid decision becomes ADD
        (see _target_problem), so one bad model answer never loses the rest of the turn.
        """
        if self.classifier is None or not facts:
            return [UpdateDecision(fact.fact_id) for fact in facts]
        by_id = {fact.fact_id: fact for fact in facts}
        proposed = {}
        for decision in self.classifier.classify(facts):
            if decision.fact_id not in by_id or decision.fact_id in proposed:
                logger.warning("Ignoring classifier decision for unknown or repeated fact")
                continue
            proposed[decision.fact_id] = decision
        target_ids = sorted({target for d in proposed.values() for target in d.target_ids})
        stored = {fact.fact_id: fact for fact in self.store.get_facts(target_ids)}
        decisions, superseded = [], set()
        for fact in facts:
            decision = proposed.get(fact.fact_id, UpdateDecision(fact.fact_id))
            problem = _target_problem(decision, fact, stored)
            is_supersede = decision.action == UpdateAction.SUPERSEDE
            if not problem and is_supersede and superseded & set(decision.target_ids):
                # A stored fact can be closed only once per round.
                problem = "target already superseded by another fact in this round"
            if problem:
                # Model output is untrusted: keep the fact as new rather than lose the turn.
                logger.warning("Treating fact %s as new: %s", fact.fact_id, problem)
                decision = UpdateDecision(fact.fact_id)
            if decision.action == UpdateAction.SUPERSEDE:
                superseded |= set(decision.target_ids)
            decisions.append(decision)
        return decisions


def _target_problem(decision: UpdateDecision, new: MemoryFact, stored) -> str | None:
    """Why `decision` cannot be applied, or None.

    Targets must be stored, current, same speaker, visible from the new fact's scope (so same
    user) and not narrower than it; a supersede target must also be no broader than it.
    """
    for target_id in decision.target_ids:
        old = stored.get(target_id)
        if (
            old is None
            or not old.is_current
            or not old.visible_to(new.scope)
            or old.speaker != new.speaker
        ):
            return f"invalid {decision.action} target {target_id}"
        if is_broader(new.visibility, old.visibility):
            return f"{new.visibility} fact cannot use narrower {old.visibility} target"
        if decision.action == UpdateAction.SUPERSEDE and is_broader(old.visibility, new.visibility):
            return f"{new.visibility} fact cannot supersede {old.visibility}"
    return None


def _episode_id(scope: Scope, messages: Sequence[Message]) -> str:
    parts = [(m.speaker, m.timestamp.isoformat(), m.text) for m in messages]
    return stable_id(scope.user_id, scope.project_id, scope.session_id, *parts)


def _build_fact(
    fact: ExtractedFact, document: Document, resolution: EntityResolution, now: datetime
) -> tuple[MemoryFact, list[FactEntity], list[EntityRelation]]:
    """Stored fact plus its memory_entities and entity_relations rows."""
    claim_type = fact.claim_type
    # Trust boundary: an assistant can propose a preference, never assert the user's.
    if document.speaker != Speaker.USER and fact.category == Category.PREFERENCE:
        if claim_type == ClaimType.ASSERTED:
            claim_type = ClaimType.PROPOSED
    fact_id = stable_id(document.document_id, normalize_name(fact.text), claim_type)
    by_name = {normalize_name(m.name): resolution.entity_ids[mention_key(m)] for m in fact.entities}
    entity_rows = [
        FactEntity(fact_id, resolution.entity_ids[mention_key(m)], m.role) for m in fact.entities
    ]
    relation_rows = [
        EntityRelation(subject, r.predicate, obj, fact_id)
        for r in fact.relations
        if (subject := by_name.get(normalize_name(r.subject)))
        and (obj := by_name.get(normalize_name(r.object)))
        and subject != obj
    ]
    memory = MemoryFact(
        fact_id=fact_id,
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
        entity_ids=tuple(sorted(set(by_name.values()))),
    )
    return memory, entity_rows, relation_rows


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


def build_retainer(
    conn_factory, genai_client, *, embedding_client=None, config: RetainConfig = RetainConfig()
):
    """Production wiring for AlloyDB storage and Gemini stages."""
    from retain.alloydb_store import AlloyDBRetainStore
    from retain.embeddings import GeminiEmbedder
    from retain.entities import FuzzyEntityResolver
    from retain.extraction import GeminiFactExtractor
    from retain.linking import GeminiSupersedeClassifier, StoreLinker
    from retain.observations import GeminiObservationBuilder

    store = AlloyDBRetainStore(conn_factory)
    embedder = GeminiEmbedder(embedding_client or genai_client)
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
