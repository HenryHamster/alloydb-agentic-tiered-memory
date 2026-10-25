"""Proposal step 3, entity linking: resolve mentions to canonical per-user entities."""

import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Protocol

from retain.models import Entity, EntityMention, EntityResolution, EntityType, stable_id


def normalize_name(name: str) -> str:
    """Unicode-normalized, case-folded, whitespace-collapsed form used for matching."""
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def mention_key(mention: EntityMention) -> tuple[str, EntityType]:
    return normalize_name(mention.name), mention.entity_type


class EntityResolver(Protocol):
    def resolve(self, user_id: str, mentions: Sequence[EntityMention]) -> EntityResolution:
        """Map every mention to exactly one entity of the same user."""
        ...


class FuzzyEntityResolver:
    """Exact name/alias match first, then fuzzy alias match within the same entity type.

    Rules: types must agree (OTHER may match an exact name of any type); strip legal suffixes
    (Inc, LLC, Ltd) before comparing; accept ratio >= `threshold`; a fuzzy hit adds the surface
    form to the entity's aliases; unmatched mentions become new entities with
    entity_id = stable_id(user_id, entity_type, canonical_name). Mentions in one batch that
    resolve to the same entity share it.
    """

    def __init__(self, store, *, threshold: float = 0.9):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.store = store  # RetainStore.find_entities supplies candidates
        self.threshold = threshold

    def resolve(self, user_id: str, mentions: Sequence[EntityMention]) -> EntityResolution:
        mentions = tuple(dict.fromkeys(mentions))
        if not mentions:
            return EntityResolution({})

        names = list(dict.fromkeys(mention.name for mention in mentions if mention.name.strip()))
        candidates = self.store.find_entities(user_id, names) if names else []
        by_id = {entity.entity_id: entity for entity in candidates if entity.user_id == user_id}
        resolved, upserts = {}, {}

        for mention in mentions:
            key = mention_key(mention)
            if key in resolved:
                continue
            normalized = normalize_name(mention.name)
            if not normalized:
                raise ValueError("Entity mentions need a nonempty name")

            exact = [
                entity
                for entity in by_id.values()
                if _type_matches(mention, entity)
                and _comparison_name(mention.name, mention.entity_type)
                in _entity_names(entity, mention.entity_type)
            ]
            exact_ids = {entity.entity_id for entity in exact}
            if len(exact_ids) > 1:
                raise ValueError(f"Ambiguous exact entity match for {mention.name!r}")
            selected = exact[0] if exact else None

            if selected is None and not exact and mention.entity_type != EntityType.PERSON:
                scored = _fuzzy_candidates(mention, by_id.values(), self.threshold)
                if scored and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.05):
                    selected = scored[0][1]

            if selected is None:
                entity_id = stable_id(user_id, mention.entity_type.value, normalized)
                selected = Entity(
                    entity_id=entity_id,
                    user_id=user_id,
                    name=mention.name.strip(),
                    canonical_name=normalized,
                    entity_type=mention.entity_type,
                )
                by_id[entity_id] = selected
                upserts.setdefault(entity_id, selected)
            elif normalize_name(mention.name) not in _entity_names(selected, mention.entity_type):
                aliases = tuple(dict.fromkeys((*selected.aliases, mention.name.strip())))
                selected = Entity(
                    entity_id=selected.entity_id,
                    user_id=selected.user_id,
                    name=selected.name,
                    canonical_name=selected.canonical_name,
                    entity_type=selected.entity_type,
                    aliases=aliases,
                )
                by_id[selected.entity_id] = selected
                upserts[selected.entity_id] = selected
            resolved[key] = selected.entity_id

        return EntityResolution(resolved, tuple(upserts.values()))


def _type_matches(mention, entity):
    return entity.entity_type == mention.entity_type or (
        mention.entity_type == EntityType.OTHER
        and _comparison_name(mention.name, mention.entity_type)
        in _entity_names(entity, mention.entity_type)
    )


def _entity_names(entity, mention_type):
    values = (entity.canonical_name, entity.name, *entity.aliases)
    return {_comparison_name(value, mention_type) for value in values if value.strip()}


def _comparison_name(name, entity_type):
    value = normalize_name(name)
    if entity_type == EntityType.ORGANIZATION:
        for suffix in (
            " incorporated",
            " corporation",
            " limited",
            " inc",
            " llc",
            " ltd",
            " corp",
        ):
            if value.endswith(suffix):
                return value[: -len(suffix)].strip()
    return value


def _fuzzy_candidates(mention, candidates, threshold):
    source = _comparison_name(mention.name, mention.entity_type)
    scored = []
    for entity in candidates:
        if entity.entity_type != mention.entity_type:
            continue
        score = max(
            (
                SequenceMatcher(None, source, candidate).ratio()
                for candidate in _entity_names(entity, mention.entity_type)
            ),
            default=0.0,
        )
        if score >= threshold:
            scored.append((score, entity))
    return sorted(scored, key=lambda row: (-row[0], row[1].entity_id))
