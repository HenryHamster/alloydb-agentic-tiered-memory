"""Recall read contract over Retain's tables, and an in-memory reference store for offline use."""

import re
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

from recall.config import MemoryType
from recall.models import Memory, TimeWindow
from retain.entities import normalize_name
from retain.models import Document, Scope, Visibility, is_visible, resolve_visibility
from retain.store import InMemoryRetainStore
from retain.store import _cosine as cosine  # same similarity Retain links with


class RecallStore(Protocol):
    """Every method filters by the reader's scope with the read rule in retain.models.is_visible."""

    def search(
        self,
        scope: Scope,
        queries: Sequence[str],
        embeddings: Sequence[Sequence[float]] | None,
        window: TimeWindow | None,
        types: Sequence[MemoryType],
        as_of: datetime,
        include_invalid: bool,
        limit: int,
        *,
        tiers: Sequence[str] | None = None,
    ) -> dict[str, list[Memory]]:
        """Runs vector, keyword, query-entity and temporal searches in one round trip.

        Returns best-first lists keyed "<channel>:<query index>" (temporal is "temporal").
        Skips the vector channel when `embeddings` is None and temporal when `window` is None.
        Unless include_invalid, excludes rows with valid_to <= as_of inside the query so stale
        rows never take pool slots.
        """
        ...

    def expand_neighbors(
        self,
        scope: Scope,
        seed_ids: Sequence[str],
        types: Sequence[MemoryType],
        as_of: datetime,
        include_invalid: bool,
        limit: int,
        *,
        tiers: Sequence[str] | None = None,
    ) -> list[Memory]:
        """One hop from the seeds over memory_links (by weight) and shared entities."""
        ...

    def get_episode(self, scope: Scope, episode_id: str) -> dict[str, Any] | None:
        """The original conversation round, or None if unknown, expired or not readable."""
        ...

    def get_supporting(self, scope: Scope, observation_id: str) -> list[Memory]:
        """The readable facts an observation was built from."""
        ...


def episode_visible(origin: Scope, reader: Scope) -> bool:
    """Documents carry no visibility, so read them in the narrowest domain they belong to:
    the project if there is one, else the session, else every project of the same user."""
    return is_visible(resolve_visibility(Visibility.PROJECT, origin), origin, reader)


_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how i in is it me my of on or "
    "our that the their this to was we were what when where which who why with you your".split()
)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", normalize_name(text)) if w not in _STOPWORDS}


def _phrase_in(phrase: str, text: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


class InMemoryRecallStore:
    """Reference implementation of RecallStore over an InMemoryRetainStore. Brute force.

    Keyword search is word overlap and query entities are matched by name or alias appearing
    in the query; AlloyDBRecallStore uses tsvector ranking and pg_trgm instead.
    """

    def __init__(self, retain_store: InMemoryRetainStore, reflect_store=None):
        self.retain_store = retain_store
        self.reflect_store = reflect_store
        self.reflect_enabled = reflect_store is not None

    def search(
        self,
        scope,
        queries,
        embeddings,
        window,
        types,
        as_of,
        include_invalid,
        limit,
        *,
        tiers=None,
    ):
        candidates = self._candidates(scope, types, as_of, include_invalid, tiers)
        lists: dict[str, list[Memory]] = {}
        for index, query in enumerate(queries):
            if embeddings is not None:
                lists[f"vector:{index}"] = self._vector(candidates, embeddings[index], limit)
            lists[f"keyword:{index}"] = self._keyword(candidates, query, limit)
            lists[f"entity:{index}"] = self._entity(candidates, scope, query, limit)
        if window is not None:
            hits = [m for m, _ in candidates if (i := m.interval(as_of)) and window.overlaps(*i)]
            lists["temporal"] = sorted(hits, key=_newest_first)[:limit]
        return {name: results for name, results in lists.items() if results}

    def expand_neighbors(
        self, scope, seed_ids, types, as_of, include_invalid, limit, *, tiers=None
    ):
        if MemoryType.FACT not in types:
            return []
        seeds = set(seed_ids)
        readable = {
            memory.id: memory
            for memory, _ in self._candidates(
                scope, [MemoryType.FACT], as_of, include_invalid, tiers
            )
        }
        by_link: dict[str, float] = {}
        for link in list(self.retain_store.links.values()):
            ends = ((link.from_fact_id, link.to_fact_id), (link.to_fact_id, link.from_fact_id))
            for here, there in ends:  # links are followed in both directions
                if here in seeds and there in readable and there not in seeds:
                    by_link[there] = max(by_link.get(there, 0.0), link.weight)
        linked = sorted(by_link, key=lambda fact_id: -by_link[fact_id])
        seed_entities = {e for i in seeds if i in readable for e in readable[i].entity_ids}
        shared = sorted(
            (
                m
                for m in readable.values()
                if m.id not in seeds and m.id not in by_link and seed_entities & set(m.entity_ids)
            ),
            key=_newest_first,
        )
        return ([readable[fact_id] for fact_id in linked] + shared)[:limit]

    def get_episode(self, scope, episode_id):
        documents = self._episodes(scope).get(episode_id)
        if not documents:
            return None
        return {
            "episode_id": episode_id,
            "messages": [
                {
                    "speaker": str(document.speaker),
                    "text": document.text,
                    "occurred_at": document.occurred_at.isoformat(),
                }
                for document in sorted(documents, key=lambda document: document.position)
            ],
        }

    def get_supporting(self, scope, observation_id):
        observation = self.retain_store.observations.get(observation_id)
        if observation is None and self.reflect_store is not None:
            macro = self.reflect_store.macros.get(observation_id)
            if macro and is_visible(macro.visibility, macro.scope, scope):
                return [
                    Memory.from_fact(fact)
                    for fact in self.retain_store.get_facts(macro.supporting_fact_ids)
                    if fact.visible_to(scope)
                ]
        if observation is None or not is_visible(observation.visibility, observation.scope, scope):
            return []
        facts = self.retain_store.get_facts(observation.fact_ids)
        return [Memory.from_fact(fact) for fact in facts if fact.visible_to(scope)]

    # Candidate rows: (memory, embedding) readable by `scope`, filtered by type and validity.

    def _candidates(self, scope, types, as_of, include_invalid, tiers=None):
        rows: list[tuple[Memory, Sequence[float] | None]] = []
        if MemoryType.FACT in types:
            rows += [
                (
                    Memory.from_fact(
                        fact,
                        state=self.reflect_store.states.get(fact.fact_id)
                        if self.reflect_store
                        else None,
                    ),
                    fact.embedding,
                )
                for fact in list(self.retain_store.facts.values())
                if fact.visible_to(scope)
            ]
        if MemoryType.OBSERVATION in types:
            rows += [
                (Memory.from_observation(obs), obs.embedding)
                for obs in list(self.retain_store.observations.values())
                if is_visible(obs.visibility, obs.scope, scope)
            ]
        if MemoryType.EPISODE in types:
            # Episodes have no embedding; they are found by keyword and time only.
            rows += [(Memory.from_episode(docs), None) for docs in self._episodes(scope).values()]
        if MemoryType.MENTAL_MODEL in types and self.reflect_store is not None:
            rows += [
                (Memory.from_macro(macro), macro.macro_embedding)
                for macro in self.reflect_store.user_macros(scope.user_id)
                if is_visible(macro.visibility, macro.scope, scope)
            ]
        return [
            (memory, embedding)
            for memory, embedding in rows
            if (include_invalid or not memory.is_stale(as_of))
            and not memory.consolidated_into
            and (tiers is None or (memory.tier or "hot") in tiers)
        ]

    def _episodes(self, scope) -> dict[str, list[Document]]:
        episodes = defaultdict(list)
        for document in list(self.retain_store.documents.values()):
            if episode_visible(document.scope, scope):
                episodes[document.episode_id].append(document)
        return episodes

    @staticmethod
    def _vector(candidates, query_vector, limit):
        scored = [
            (cosine(query_vector, embedding), memory)
            for memory, embedding in candidates
            if embedding
        ]
        scored = [pair for pair in scored if pair[0] > 0]
        return [memory for _, memory in sorted(scored, key=lambda pair: -pair[0])[:limit]]

    @staticmethod
    def _keyword(candidates, query, limit):
        wanted = _words(query)
        scored = [(len(wanted & _words(memory.text)), memory) for memory, _ in candidates]
        scored = [pair for pair in scored if pair[0] > 0]
        return [memory for _, memory in sorted(scored, key=lambda pair: -pair[0])[:limit]]

    def _entity(self, candidates, scope, query, limit):
        text = " ".join(re.findall(r"\w+", normalize_name(query)))
        matched = {
            entity.entity_id
            for entity in list(self.retain_store.entities.values())
            if entity.user_id == scope.user_id
            and any(
                _phrase_in(" ".join(re.findall(r"\w+", normalize_name(name))), text)
                for name in (entity.name, entity.canonical_name, *entity.aliases)
            )
        }
        hits = [memory for memory, _ in candidates if matched & set(memory.entity_ids)]
        return sorted(hits, key=_newest_first)[:limit]


def _newest_first(memory: Memory) -> float:
    date = memory.primary_date()
    return -date.timestamp() if date else 0.0
