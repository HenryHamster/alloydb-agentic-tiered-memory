"""Recall records: the query window, the candidate view of stored memories, and results.

Stored rows are Retain's (retain/models.py). Recall reads facts, observations and episodes
through one flat `Memory` view so every stage ranks them the same way.
"""

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from recall.config import MemoryType
from retain.models import (
    Category,
    ClaimType,
    Document,
    FactType,
    MemoryFact,
    Observation,
    Speaker,
)


def as_aware(value: datetime | None, tz=None) -> datetime | None:
    """Reads naive caller input as UTC (or `tz`); stored rows are always aware."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=tz or timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


@dataclass(frozen=True)
class TimeWindow:
    """Half-open date window [start, end)."""

    start: datetime
    end: datetime

    def __post_init__(self):
        object.__setattr__(self, "start", as_aware(self.start))
        object.__setattr__(self, "end", as_aware(self.end))
        if self.end <= self.start:
            raise ValueError(f"Time window end {self.end} must be after start {self.start}")

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return start < self.end and end >= self.start

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True)
class Memory:
    """One search candidate, whatever table it came from."""

    id: str
    type: MemoryType
    text: str
    recorded_at: datetime | None = None
    occurred_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    speaker: Speaker | None = None
    claim_type: ClaimType | None = None
    fact_type: FactType | None = None
    category: Category | None = None
    source_episode_id: str | None = None
    supporting_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    evidence_count: int = 1

    @classmethod
    def from_fact(cls, fact: MemoryFact) -> "Memory":
        return cls(
            id=fact.fact_id,
            type=MemoryType.FACT,
            text=fact.text,
            recorded_at=fact.recorded_at,
            occurred_at=fact.occurred_at,
            valid_from=fact.valid_from,
            valid_to=fact.valid_to,
            speaker=fact.speaker,
            claim_type=fact.claim_type,
            fact_type=fact.fact_type,
            category=fact.category,
            source_episode_id=fact.source_episode_id,
            entity_ids=fact.entity_ids,
            evidence_count=fact.evidence_count,
        )

    @classmethod
    def from_observation(cls, observation: Observation) -> "Memory":
        return cls(
            id=observation.observation_id,
            type=MemoryType.OBSERVATION,
            text=observation.text,
            valid_from=observation.valid_from,
            valid_to=observation.valid_to,
            supporting_ids=observation.fact_ids,
            entity_ids=(observation.entity_id,) if observation.entity_id else (),
            evidence_count=len(observation.fact_ids),
        )

    @classmethod
    def from_episode(cls, documents: Sequence[Document]) -> "Memory":
        """One conversation round; `documents` share an episode_id."""
        ordered = sorted(documents, key=lambda document: document.position)
        first = ordered[0]
        return cls(
            id=first.episode_id,
            type=MemoryType.EPISODE,
            text="\n".join(f"{d.speaker}: {d.text}" for d in ordered),
            recorded_at=first.recorded_at,
            occurred_at=first.occurred_at,
            source_episode_id=first.episode_id,
        )

    def primary_date(self) -> datetime | None:
        return self.occurred_at or self.valid_from or self.recorded_at

    def interval(self, as_of: datetime) -> tuple[datetime, datetime] | None:
        """Events are points at occurred_at; states run from valid_from until valid_to or now."""
        if self.occurred_at:
            return self.occurred_at, self.occurred_at
        start = self.valid_from or self.recorded_at
        if start is None:
            return None
        return start, max(start, self.valid_to or as_of)

    def is_stale(self, as_of: datetime) -> bool:
        return self.valid_to is not None and self.valid_to <= as_of

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": str(self.type),
            "text": self.text,
            "dates": {
                "occurred_at": _iso(self.occurred_at),
                "recorded_at": _iso(self.recorded_at),
                "valid_from": _iso(self.valid_from),
                "valid_to": _iso(self.valid_to),
            },
            "speaker": self.speaker and str(self.speaker),
            "claim_type": self.claim_type and str(self.claim_type),
            "fact_type": self.fact_type and str(self.fact_type),
            "category": self.category and str(self.category),
            "source_episode_id": self.source_episode_id,
            "supporting_ids": list(self.supporting_ids),
            "entity_ids": list(self.entity_ids),
        }


@dataclass
class QueryPlan:
    sub_queries: list[str]
    window: TimeWindow | None
    window_source: str | None = None  # "explicit", "rules" or "llm"
    llm_used: bool = False


@dataclass
class Scored:
    memory: Memory
    relevance: float
    boost: float = 0.0
    score: float = 0.0


@dataclass
class StageTrace:
    stage: str
    ms: float = 0.0
    count: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "ms": round(self.ms, 3), "count": self.count, **self.detail}


class Tracer:
    """Collects per-stage timings and counts for RecallResult.trace."""

    def __init__(self):
        self.stages: list[StageTrace] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTrace]:
        entry = StageTrace(name)
        started = time.perf_counter()
        try:
            yield entry
        finally:
            entry.ms = (time.perf_counter() - started) * 1000
            self.stages.append(entry)


@dataclass
class RecallItem:
    memory: Memory
    score: float
    relevance: float
    rank: int
    stale: bool
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.memory.to_dict(),
            "score": round(self.score, 6),
            "relevance": round(self.relevance, 6),
            "rank": self.rank,
            "stale": self.stale,
        }


@dataclass
class RecallResult:
    items: list[RecallItem]
    context: str
    tokens_used: int
    max_tokens: int
    truncated: bool
    abstained: bool
    sub_queries: list[str]
    window: TimeWindow | None
    trace: list[StageTrace]

    @classmethod
    def empty(cls, plan: QueryPlan, max_tokens: int, trace: list[StageTrace]) -> "RecallResult":
        return cls([], "", 0, max_tokens, False, True, plan.sub_queries, plan.window, trace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "context": self.context,
            "tokens_used": self.tokens_used,
            "max_tokens": self.max_tokens,
            "truncated": self.truncated,
            "abstained": self.abstained,
            "sub_queries": self.sub_queries,
            "window": self.window.to_dict() if self.window else None,
            "trace": [stage.to_dict() for stage in self.trace],
        }
