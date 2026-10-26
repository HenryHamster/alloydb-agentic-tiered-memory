"""Shared Recall test fixtures: Retain rows written straight into the in-memory stores."""

import re
from datetime import datetime, timedelta, timezone

from recall import InMemoryRecallStore, Recaller
from retain import InMemoryRetainStore, Scope
from retain.models import (
    Category,
    ClaimType,
    Document,
    Entity,
    EntityType,
    FactType,
    MemoryFact,
    MemoryLink,
    Observation,
    Relation,
    Speaker,
    Supersession,
    Visibility,
    WriteBatch,
    stable_id,
)

NOW = datetime(2026, 10, 18, 15, 30, tzinfo=timezone.utc)
SCOPE = Scope("user_1", "project_a", "session_1")

_STOPWORDS = set(
    "a an and are did do does for how in is it of on the to was we what which who".split()
)
_VOCAB: dict[str, int] = {}


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]


def embed(texts):
    """Bag-of-words vectors: cosine > 0 exactly when two texts share a content word."""
    vectors = []
    for text in texts:
        vector = [0.0] * 512
        for word in words(text):
            vector[_VOCAB.setdefault(word, len(_VOCAB) % 512)] += 1.0
        vectors.append(vector)
    return vectors


def overlap_reranker(query, texts):
    """Fraction of the query's content words found in each text, as a stand-in cross-encoder."""
    wanted = set(words(query))
    return [len(wanted & set(words(text))) / (len(wanted) or 1) for text in texts]


class Memories:
    """A Retain store plus the Recall store reading it, with helpers to write rows."""

    def __init__(self):
        self.retain_store = InMemoryRetainStore()
        self.store = InMemoryRecallStore(self.retain_store)

    def recaller(self, **kwargs) -> Recaller:
        kwargs.setdefault("embedder", embed)
        return Recaller(store=self.store, clock=lambda: NOW, **kwargs)

    def fact(
        self,
        text: str,
        *,
        scope: Scope = SCOPE,
        visibility: Visibility = Visibility.GLOBAL,
        episode_id: str | None = None,
        occurred_at: datetime | None = None,
        valid_from: datetime | None = None,
        entity_ids: tuple[str, ...] = (),
        supersedes: MemoryFact | None = None,
    ) -> MemoryFact:
        said_at = occurred_at or valid_from or NOW - timedelta(days=1)
        episode_id = episode_id or stable_id("episode", scope.user_id, text, said_at)
        document = Document(
            stable_id(episode_id, 0), episode_id, 0, scope, Speaker.USER, text, said_at, said_at
        )
        fact = MemoryFact(
            fact_id=stable_id("fact", episode_id, text),
            scope=scope,
            visibility=visibility,
            fact_type=FactType.WORLD,
            category=Category.OTHER,
            speaker=Speaker.USER,
            claim_type=ClaimType.ASSERTED,
            text=text,
            evidence=text,
            source_document_id=document.document_id,
            source_episode_id=episode_id,
            recorded_at=said_at,
            occurred_at=occurred_at,
            valid_from=valid_from or said_at,
            embedding=tuple(embed([text])[0]),
            entity_ids=entity_ids,
        )
        changes, links = (), ()
        if supersedes is not None:
            changes = (Supersession(supersedes.fact_id, fact.fact_id, fact.valid_from),)
            links = (MemoryLink(fact.fact_id, supersedes.fact_id, Relation.SUPERSEDES),)
        self.retain_store.write(
            WriteBatch(documents=(document,), facts=(fact,), supersessions=changes, links=links)
        )
        return fact

    def link(self, source: MemoryFact, target: MemoryFact, weight: float = 1.0) -> None:
        link = MemoryLink(source.fact_id, target.fact_id, Relation.CAUSED_BY, weight)
        self.retain_store.write(WriteBatch(documents=(), links=(link,)))

    def entity(self, name: str, *aliases: str, user_id: str = SCOPE.user_id) -> str:
        entity_id = stable_id(user_id, name)
        entity = Entity(entity_id, user_id, name, name.lower(), EntityType.PROJECT, aliases)
        self.retain_store.write(WriteBatch(documents=(), entities=(entity,)))
        return entity_id

    def observation(self, text: str, facts: list[MemoryFact], scope: Scope = SCOPE) -> Observation:
        observation = Observation(
            observation_id=stable_id("observation", text),
            scope=scope,
            visibility=Visibility.GLOBAL,
            text=text,
            fact_ids=tuple(fact.fact_id for fact in facts),
            valid_from=min(fact.valid_from for fact in facts),
            embedding=tuple(embed([text])[0]),
        )
        self.retain_store.write_observations([observation])
        return observation
