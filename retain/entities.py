"""Proposal step 3, entity linking: resolve mentions to canonical per-user entities."""

import unicodedata
from collections.abc import Sequence
from typing import Protocol

from retain.models import EntityMention, EntityResolution, EntityType


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
        self.store = store  # RetainStore.find_entities supplies candidates
        self.threshold = threshold

    def resolve(self, user_id: str, mentions: Sequence[EntityMention]) -> EntityResolution:
        # TODO(retain): implement the class rules with difflib (no new dependency); the AlloyDB
        # store prefilters candidates with pg_trgm similarity.
        raise NotImplementedError("Fuzzy entity resolution")
