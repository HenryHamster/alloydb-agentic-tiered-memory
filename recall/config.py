"""Recall tuning: search depth, memory types and the ranking constants in one place."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType


class RecallDepth(StrEnum):
    """How many candidates each search channel returns before fusion."""

    LOW = "low"  # automatic per-turn context
    MID = "mid"  # the agent's search_memory tool
    HIGH = "high"  # Reflect and offline analysis

    @classmethod
    def parse(cls, value: "RecallDepth | str") -> "RecallDepth":
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"depth must be one of {[d.value for d in cls]}, got {value!r}"
            ) from None


class MemoryType(StrEnum):
    FACT = "fact"  # memory_facts
    OBSERVATION = "observation"  # observations
    EPISODE = "episode"  # documents grouped by episode_id
    MENTAL_MODEL = "mental_model"  # Reflect; no table yet, so stores return none


DEFAULT_TYPES = (MemoryType.FACT, MemoryType.OBSERVATION)


def parse_types(types: Iterable[str] | None) -> tuple[MemoryType, ...]:
    values = tuple(types or DEFAULT_TYPES)
    unknown = sorted(str(t) for t in values if t not in set(MemoryType))
    if unknown:
        raise ValueError(f"Unknown memory types: {unknown}")
    return tuple(dict.fromkeys(MemoryType(t) for t in values))


def _pools() -> Mapping[RecallDepth, int]:
    return MappingProxyType({RecallDepth.LOW: 100, RecallDepth.MID: 300, RecallDepth.HIGH: 1000})


@dataclass(frozen=True)
class RecallConfig:
    pool_by_depth: Mapping[RecallDepth, int] = field(default_factory=_pools)
    rrf_k: int = 60
    # Cross-encoding a whole high-depth pool is too slow; rerank the fused head only.
    rerank_pool: int = 150
    rerank_top: int = 30
    graph_seeds: int = 10
    max_sub_queries: int = 4
    # TODO(recall): calibrate against real ai.rank() scores on a labelled sample.
    score_floor: float = 0.3
    # Boosts are multiplied in as relevance * (1 + boost), total clipped to +/-max_boost.
    max_boost: float = 0.25
    recency_boost: float = 0.10
    window_boost: float = 0.10
    evidence_boost: float = 0.05
    # TODO(recall): tune once real fact ages are available.
    recency_half_life_days: float = 90.0
