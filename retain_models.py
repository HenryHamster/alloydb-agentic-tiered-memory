"""Provisional Retain records; no database schema is implied by these dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from math import isfinite


class MemoryScope(StrEnum):
    GLOBAL = "global"
    PROJECT = "project"
    SESSION = "session"


class Speaker(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Category(StrEnum):
    PREFERENCE = "preference"
    EVENT = "event"
    CONSTRAINT = "constraint"
    GOAL = "goal"
    PROJECT = "project"
    EXPERIENCE = "experience"


class ClaimType(StrEnum):
    ASSERTED = "asserted"
    PROPOSED = "proposed"
    HYPOTHETICAL = "hypothetical"
    REJECTED = "rejected"


class UpdateCategory(StrEnum):
    ADD = "ADD"
    CONFIRM = "CONFIRM"
    REFINE = "REFINE"
    SUPERSEDE = "SUPERSEDE"
    CONTEST = "CONTEST"
    CONTEXT = "CONTEXT"


class ChangeKind(StrEnum):
    HISTORICAL = "historical_change"
    CORRECTION = "correction"


class RelationshipType(StrEnum):
    ENTITY = "entity"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    EVIDENCE = "evidence"
    SUPERSEDES = "supersedes"
    CONFIRMS = "confirms"
    REFINES = "refines"
    CONTESTS = "contests"
    CONTEXT = "context"


def require_text(*values: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Identifiers and text must be nonempty strings")


def require_aware(*values: datetime | None) -> None:
    if any(value is not None and value.utcoffset() is None for value in values):
        raise ValueError("Timestamps must have a timezone")


@dataclass(frozen=True)
class Scope:
    """Trusted visibility label plus origin metadata, never selected by the model.

    GLOBAL means user-wide, not shared between users. Fact origins are immutable;
    read contexts may omit project/session IDs.
    """

    user_id: str
    project_id: str | None = None
    session_id: str | None = None
    scope: MemoryScope = MemoryScope.SESSION

    def __post_init__(self):
        require_text(self.user_id)
        MemoryScope(self.scope)
        for value in (self.project_id, self.session_id):
            if value is not None:
                require_text(value)
        if self.scope == MemoryScope.PROJECT and not self.project_id:
            raise ValueError("Project scope requires project_id")
        if self.scope == MemoryScope.SESSION and not self.session_id:
            raise ValueError("Session scope requires session_id")

    def visible_to(self, context: "Scope") -> bool:
        """Pure visibility predicate; adapters must apply it to facts and link endpoints."""
        if self.user_id != context.user_id:
            return False
        if self.scope == MemoryScope.GLOBAL:
            return True
        if self.project_id != context.project_id:
            return False
        return self.scope == MemoryScope.PROJECT or self.session_id == context.session_id

    def can_reconcile_with(self, other: "Scope") -> bool:
        """Same visibility domain, independent of origin for project/user-wide facts."""
        return self.scope == other.scope and self.visible_to(other) and other.visible_to(self)


@dataclass(frozen=True)
class Message:
    source_id: str
    episode_id: str
    text: str
    speaker: Speaker
    timestamp: datetime
    scope: Scope

    def __post_init__(self):
        require_text(self.source_id, self.episode_id, self.text)
        Speaker(self.speaker)
        require_aware(self.timestamp)


@dataclass(frozen=True)
class ConversationChunk:
    id: str
    message: Message
    start: int
    end: int

    def __post_init__(self):
        require_text(self.id)
        if not 0 <= self.start < self.end <= len(self.message.text):
            raise ValueError("Chunk offsets must identify a nonempty source span")

    @property
    def text(self) -> str:
        return self.message.text[self.start : self.end]


@dataclass(frozen=True)
class AtomicFact:
    """Extractor output: provenance is supplied as chunk IDs, not model-written scope."""

    text: str
    category: Category
    claim_type: ClaimType
    source_chunk_ids: tuple[str, ...]
    occurred_at: datetime | None = None
    valid_from: datetime | None = None

    def __post_init__(self):
        require_text(self.text, *self.source_chunk_ids)
        Category(self.category)
        ClaimType(self.claim_type)
        if not self.source_chunk_ids or len(set(self.source_chunk_ids)) != len(
            self.source_chunk_ids
        ):
            raise ValueError("Facts require unique source chunk IDs")
        require_aware(self.occurred_at, self.valid_from)


@dataclass(frozen=True)
class MemoryFact:
    id: str
    fact: AtomicFact
    scope: Scope
    speaker: Speaker
    source_episode_id: str
    sources: tuple[ConversationChunk, ...]
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: datetime | None = None
    entity_ids: tuple[str, ...] = ()
    supporting_ids: tuple[str, ...] = ()
    embedding: tuple[float, ...] | None = None

    def __post_init__(self):
        require_text(self.id, self.source_episode_id, *self.entity_ids, *self.supporting_ids)
        require_text(self.scope.project_id, self.scope.session_id)
        Speaker(self.speaker)
        require_aware(self.recorded_at, self.valid_to)
        if self.valid_to is not None and (
            self.fact.valid_from is None or self.valid_to <= self.fact.valid_from
        ):
            raise ValueError("Validity interval must be nonempty and have a start")
        if tuple(source.id for source in self.sources) != self.fact.source_chunk_ids:
            raise ValueError("Sources must match fact provenance in order")
        if any(
            source.message.scope != self.scope
            or source.message.speaker != self.speaker
            or source.message.episode_id != self.source_episode_id
            for source in self.sources
        ):
            raise ValueError("A fact must have consistent scope, speaker and episode")
        if self.embedding is not None and (
            not self.embedding or not all(isfinite(value) for value in self.embedding)
        ):
            raise ValueError("Embedding must be nonempty and finite")

    @property
    def evidence_count(self) -> int:
        return len({source.message.source_id for source in self.sources})


@dataclass(frozen=True)
class MemoryLink:
    source_id: str
    target_id: str
    relationship: RelationshipType
    scope: Scope

    def __post_init__(self):
        require_text(self.source_id, self.target_id)
        RelationshipType(self.relationship)
        if self.source_id == self.target_id:
            raise ValueError("Memory links must connect distinct memory IDs")


@dataclass(frozen=True)
class MemoryUpdate:
    memory: MemoryFact
    category: UpdateCategory = UpdateCategory.ADD
    previous: MemoryFact | None = None
    change_kind: ChangeKind | None = None
    effective_at: datetime | None = None

    def __post_init__(self):
        UpdateCategory(self.category)
        require_aware(self.effective_at)
        if self.category == UpdateCategory.ADD:
            if self.previous is not None:
                raise ValueError("ADD must not reference a previous fact")
        elif self.previous is None:
            raise ValueError("Non-ADD updates require a previous fact")
        if self.previous is not None:
            if self.previous.id == self.memory.id or not self.previous.scope.can_reconcile_with(
                self.memory.scope
            ):
                raise ValueError("Updates need distinct IDs and the same visibility domain")
        if self.category == UpdateCategory.SUPERSEDE:
            if self.change_kind is None or self.effective_at is None:
                raise ValueError("Supersession needs a change kind and effective timestamp")
            ChangeKind(self.change_kind)
        elif self.change_kind is not None or self.effective_at is not None:
            raise ValueError("Change metadata is only supported for SUPERSEDE")


@dataclass(frozen=True)
class WriteBatch:
    """Store must compare previous rows and commit all changes in one transaction."""

    insertions: tuple[MemoryFact, ...]
    replacements: tuple[MemoryFact, ...] = ()
    expected_previous: tuple[MemoryFact, ...] = ()
    links: tuple[MemoryLink, ...] = ()
