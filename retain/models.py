"""Retain records. Field names mirror sql/004_retain_schema.sql."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, uuid5


class Visibility(StrEnum):
    """Who can read a memory. Broadest first."""

    GLOBAL = "global"  # every project of the same user
    PROJECT = "project"
    SESSION = "session"


class Speaker(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FactType(StrEnum):
    WORLD = "world"  # about the user or the world
    EXPERIENCE = "experience"  # what the agent itself did or said


class Category(StrEnum):
    """Narrows semantic search and supersession candidates."""

    PREFERENCE = "preference"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    GOAL = "goal"
    EVENT = "event"
    ATTRIBUTE = "attribute"
    RELATIONSHIP = "relationship"
    OTHER = "other"


class ClaimType(StrEnum):
    ASSERTED = "asserted"
    PROPOSED = "proposed"  # suggestions, including every assistant recommendation
    HYPOTHETICAL = "hypothetical"
    REJECTED = "rejected"


class EntityType(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PLACE = "place"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    PROJECT = "project"
    CONCEPT = "concept"
    OTHER = "other"


class EntityRole(StrEnum):
    SUBJECT = "subject"
    OBJECT = "object"
    MENTIONED = "mentioned"


class LinkType(StrEnum):
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


class Relation(StrEnum):
    """Directed edge label; every relation belongs to exactly one LinkType."""

    SIMILAR = "similar"
    NEAR_IN_TIME = "near_in_time"
    SUPERSEDES = "supersedes"  # new -> old
    CAUSED_BY = "caused_by"  # effect -> cause
    EVIDENCED_BY = "evidenced_by"  # claim -> evidence
    CONSOLIDATED_INTO = "consolidated_into"  # original -> merged fact (written by Reflect)
    CONFLICTS_WITH = "conflicts_with"  # lower fact_id -> higher; symmetric (written by Reflect)

    @property
    def link_type(self) -> LinkType:
        return _LINK_TYPES[self]


_LINK_TYPES = {
    Relation.SIMILAR: LinkType.SEMANTIC,
    Relation.NEAR_IN_TIME: LinkType.TEMPORAL,
    Relation.SUPERSEDES: LinkType.TEMPORAL,
    Relation.CAUSED_BY: LinkType.CAUSAL,
    Relation.EVIDENCED_BY: LinkType.CAUSAL,
    Relation.CONSOLIDATED_INTO: LinkType.SEMANTIC,
    Relation.CONFLICTS_WITH: LinkType.SEMANTIC,
}


class UpdateAction(StrEnum):
    ADD = "add"
    DUPLICATE = "duplicate"  # same claim already stored: reinforce it, insert nothing
    SUPERSEDE = "supersede"  # insert and close the targets' validity


def stable_id(*parts: object) -> str:
    """Deterministic UUID so replays and retries write the same rows."""
    return str(uuid5(NAMESPACE_URL, "\x1f".join(str(part) for part in parts)))


def _require_aware(*values: datetime | None) -> None:
    if any(value is not None and value.utcoffset() is None for value in values):
        raise ValueError("Timestamps must be timezone-aware")


@dataclass(frozen=True)
class Scope:
    """Trusted request context from the caller; never chosen by the model."""

    user_id: str
    project_id: str | None = None
    session_id: str | None = None

    def __post_init__(self):
        if not self.user_id:
            raise ValueError("Scope requires user_id")


def is_visible(visibility: Visibility, origin: Scope, reader: Scope) -> bool:
    """Read rule shared with Recall: same user, then global / same project / same session."""
    if origin.user_id != reader.user_id:
        return False
    if visibility == Visibility.GLOBAL:
        return True
    if visibility == Visibility.PROJECT:
        return origin.project_id is not None and origin.project_id == reader.project_id
    return origin.session_id is not None and origin.session_id == reader.session_id


def is_broader(left: Visibility, right: Visibility) -> bool:
    """True when `left` is readable in strictly more contexts than `right`."""
    order = list(Visibility)
    return order.index(left) < order.index(right)


def resolve_visibility(requested: Visibility, scope: Scope) -> Visibility:
    """Narrow a model-proposed visibility to what the context supports.

    Project without a project_id falls to session, never to global, so project rules never leak.
    """
    if requested == Visibility.PROJECT and scope.project_id is None:
        requested = Visibility.SESSION
    if requested == Visibility.SESSION and scope.session_id is None:
        requested = Visibility.PROJECT if scope.project_id else Visibility.GLOBAL
    return requested


@dataclass(frozen=True)
class Message:
    speaker: Speaker
    text: str
    timestamp: datetime

    def __post_init__(self):
        _require_aware(self.timestamp)


@dataclass(frozen=True)
class Document:
    """One stored source message (proposal: "documents" table)."""

    document_id: str
    episode_id: str
    position: int
    scope: Scope
    speaker: Speaker
    text: str
    occurred_at: datetime
    recorded_at: datetime
    expires_at: datetime | None = None

    def __post_init__(self):
        _require_aware(self.occurred_at, self.recorded_at, self.expires_at)


@dataclass(frozen=True)
class Chunk:
    """Half-open span [start, end) of a document."""

    chunk_id: str
    document: Document
    start: int
    end: int

    @property
    def text(self) -> str:
        return self.document.text[self.start : self.end]


@dataclass(frozen=True)
class EntityMention:
    name: str
    entity_type: EntityType
    role: EntityRole = EntityRole.MENTIONED


@dataclass(frozen=True)
class RelationMention:
    """Entity graph triple stated by a fact, by surface name."""

    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class ExtractedFact:
    """Validated extractor output. `key` is local to one extraction batch."""

    key: int
    chunk_id: str
    text: str
    evidence: str
    fact_type: FactType
    category: Category
    claim_type: ClaimType
    visibility: Visibility
    occurred_at: datetime | None = None
    valid_from: datetime | None = None
    details: dict[str, str] = field(default_factory=dict, compare=False)  # the 5 Ws
    entities: tuple[EntityMention, ...] = ()
    relations: tuple[RelationMention, ...] = ()
    caused_by: tuple[int, ...] = ()
    evidenced_by: tuple[int, ...] = ()

    def __post_init__(self):
        if not self.text.strip() or not self.evidence.strip():
            raise ValueError("Facts need text and evidence")
        _require_aware(self.occurred_at, self.valid_from)


@dataclass(frozen=True)
class Entity:
    entity_id: str
    user_id: str
    name: str
    canonical_name: str
    entity_type: EntityType
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntityResolution:
    """Resolver output: (normalized name, type) -> entity_id, plus rows to create or extend."""

    entity_ids: dict[tuple[str, EntityType], str]
    upserts: tuple[Entity, ...] = ()


@dataclass(frozen=True)
class MemoryFact:
    """A stored fact (proposal: "memory_facts" table)."""

    fact_id: str
    scope: Scope
    visibility: Visibility
    fact_type: FactType
    category: Category
    speaker: Speaker
    claim_type: ClaimType
    text: str
    evidence: str
    source_document_id: str
    source_episode_id: str
    recorded_at: datetime
    valid_from: datetime
    details: dict[str, str] = field(default_factory=dict, compare=False)
    occurred_at: datetime | None = None
    valid_to: datetime | None = None
    superseded_by: str | None = None
    evidence_count: int = 1
    embedding: tuple[float, ...] | None = field(default=None, compare=False)
    entity_ids: tuple[str, ...] = ()

    def __post_init__(self):
        _require_aware(self.recorded_at, self.valid_from, self.occurred_at, self.valid_to)
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise ValueError("valid_to precedes valid_from")
        if resolve_visibility(self.visibility, self.scope) != self.visibility:
            raise ValueError("Visibility needs the matching project_id or session_id")

    @property
    def is_current(self) -> bool:
        return self.valid_to is None

    def visible_to(self, reader: Scope) -> bool:
        return is_visible(self.visibility, self.scope, reader)


@dataclass(frozen=True)
class FactEntity:
    fact_id: str
    entity_id: str
    role: EntityRole


@dataclass(frozen=True)
class EntityRelation:
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    fact_id: str


@dataclass(frozen=True)
class MemoryLink:
    from_fact_id: str
    to_fact_id: str
    relation: Relation
    weight: float = 1.0

    def __post_init__(self):
        if self.from_fact_id == self.to_fact_id:
            raise ValueError("Links must join two different facts")
        if not 0 <= self.weight <= 1:
            raise ValueError("Link weight must be in [0, 1]")

    @property
    def link_type(self) -> LinkType:
        return self.relation.link_type


@dataclass(frozen=True)
class UpdateDecision:
    """Classifier verdict for one new fact against stored facts."""

    fact_id: str
    action: UpdateAction = UpdateAction.ADD
    target_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if (self.action == UpdateAction.ADD) != (not self.target_ids):
            raise ValueError("Only duplicate/supersede decisions name targets")
        if self.action == UpdateAction.DUPLICATE and len(self.target_ids) != 1:
            raise ValueError("A duplicate names exactly one stored fact")


@dataclass(frozen=True)
class Supersession:
    old_fact_id: str
    new_fact_id: str
    valid_to: datetime


@dataclass(frozen=True)
class Observation:
    """Consolidated summary of facts sharing an entity in one visibility domain."""

    observation_id: str
    scope: Scope
    visibility: Visibility
    text: str
    fact_ids: tuple[str, ...]
    valid_from: datetime
    entity_id: str | None = None
    valid_to: datetime | None = None
    embedding: tuple[float, ...] | None = field(default=None, compare=False)

    def same_version(self, other: "Observation") -> bool:
        """Same version apart from valid_to, the only field closing it may change.

        Dataclass equality already ignores the embedding, which is derived from the text.
        """
        return replace(self, valid_to=None) == replace(other, valid_to=None)


@dataclass(frozen=True)
class WriteBatch:
    """Everything one retain call writes; the store applies it in one transaction."""

    documents: tuple[Document, ...]
    facts: tuple[MemoryFact, ...] = ()
    entities: tuple[Entity, ...] = ()
    fact_entities: tuple[FactEntity, ...] = ()
    entity_relations: tuple[EntityRelation, ...] = ()
    links: tuple[MemoryLink, ...] = ()
    supersessions: tuple[Supersession, ...] = ()
    reinforced_fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetainResult:
    episode_id: str
    batch: WriteBatch | None  # None when the episode was already retained
    extracted: int = 0

    @property
    def skipped(self) -> bool:
        return self.batch is None
