"""Reflect records. Field names mirror sql/005_reflect_schema.sql."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from retain.models import FactEntity, MemoryFact, MemoryLink, Scope, Supersession, Visibility


class Tier(StrEnum):
    """Search order for Recall. Hottest first."""

    HOT = "hot"  # small, capped, searched on every recall
    WARM = "warm"  # cooling; searched only when hot returns weak results
    ARCHIVED = "archived"  # last fallback; promoted back on a hit
    PURGED = "purged"  # transition target only: the store deletes the row


class Verdict(StrEnum):
    """Judge label for a group of similar facts on the same entity and attribute."""

    DISTINCT = "distinct"  # related but both true: leave alone
    REDUNDANT = "redundant"  # same claim: merge into one consolidated fact
    STATE_CHANGE = "state_change"  # resolvable conflict: newer wins, older keeps history
    CONFLICT = "conflict"  # unresolvable: keep both, flag, lower confidence


class ReflectTrigger(StrEnum):
    NEW_FACTS = "new_facts"
    SESSION_END = "session_end"
    SCHEDULED = "scheduled"


@dataclass(frozen=True)
class FactState:
    """Lifecycle columns Reflect adds to memory_facts, keyed by fact_id."""

    fact_id: str
    access_count: int = 0
    last_accessed_at: datetime | None = None
    rank_score: float = 0.0
    tier: Tier = Tier.HOT
    tier_changed_at: datetime | None = None
    confidence: float = 1.0
    consolidated_into: str | None = None
    conflicts_with: tuple[str, ...] = ()

    def __post_init__(self):
        if self.tier == Tier.PURGED:
            raise ValueError("Purged rows are deleted, not stored")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class Candidate:
    """Facts in one visibility domain sharing a subject entity and category, linked by
    similarity >= threshold. The judge labels the whole group."""

    fact_ids: tuple[str, ...]
    entity_id: str


@dataclass(frozen=True)
class Judgment:
    """Judge output for one candidate. Untrusted: Reflector validates it."""

    fact_ids: tuple[str, ...]
    verdict: Verdict
    merged_text: str | None = None  # REDUNDANT only
    reason: str = ""


@dataclass(frozen=True)
class Consolidation:
    """One merged fact replacing redundant originals (which are archived, not deleted)."""

    merged: MemoryFact
    original_ids: tuple[str, ...]
    state: FactState
    fact_entities: tuple[FactEntity, ...] = ()


@dataclass(frozen=True)
class ConflictReview:
    review_id: str
    user_id: str
    fact_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Cluster:
    """Facts joined by temporal/causal links or shared subject/object entities."""

    scope: Scope  # origin of the visibility domain (user + project/session as needed)
    visibility: Visibility
    fact_ids: tuple[str, ...]
    entity_ids: tuple[str, ...] = ()
    causal_chains: tuple[tuple[str, ...], ...] = ()  # cause -> ... -> effect
    temporal_span: tuple[datetime, datetime] | None = None


@dataclass(frozen=True)
class MacroMemory:
    """One synthesized narrative (proposal: "MacroMemories" table)."""

    macro_id: str
    scope: Scope
    visibility: Visibility
    macro_title: str
    macro_summary: str
    supporting_fact_ids: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    primary_entity_ids: tuple[str, ...] = ()
    causal_chain_ids: tuple[tuple[str, ...], ...] = ()
    temporal_span: tuple[datetime, datetime] | None = None
    version: int = 1
    access_count: int = 0
    last_accessed_at: datetime | None = None
    rank_score: float = 0.0
    tier: Tier = Tier.HOT
    valid_to: datetime | None = None
    macro_embedding: tuple[float, ...] | None = field(default=None, compare=False)

    @property
    def is_current(self) -> bool:
        return self.valid_to is None


@dataclass(frozen=True)
class ReflectBatch:
    """Everything one reflect pass writes; the store applies it in one transaction."""

    consolidations: tuple[Consolidation, ...] = ()
    supersessions: tuple[Supersession, ...] = ()
    reviews: tuple[ConflictReview, ...] = ()
    links: tuple[MemoryLink, ...] = ()
    fact_states: tuple[FactState, ...] = ()  # full rows: rank, tier, confidence, markers
    macros: tuple[MacroMemory, ...] = ()  # inserts, new versions and tier updates
    purged_fact_ids: tuple[str, ...] = ()
    purged_macro_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReflectResult:
    user_id: str
    trigger: ReflectTrigger
    batch: ReflectBatch
