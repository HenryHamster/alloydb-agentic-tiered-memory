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
    MENTAL_MODEL = "mental_model"  # Reflect macro_memories


DEFAULT_TYPES = (MemoryType.FACT, MemoryType.OBSERVATION)


def parse_types(types: Iterable[str] | None) -> tuple[MemoryType, ...]:
    values = tuple(types or DEFAULT_TYPES)
    unknown = sorted(str(t) for t in values if t not in set(MemoryType))
    if unknown:
        raise ValueError(f"Unknown memory types: {unknown}")
    return tuple(dict.fromkeys(MemoryType(t) for t in values))


def _pools() -> Mapping[RecallDepth, int]:
    return MappingProxyType({RecallDepth.LOW: 100, RecallDepth.MID: 300, RecallDepth.HIGH: 1000})


# Links that explain a fact (the request behind a tool call, a cause, an earlier version) are
# followed before loose ones (similar, near_in_time). Both stores order neighbours this way.
EXPLAINING_RELATIONS = ("evidenced_by", "caused_by", "supersedes")
# The round links Retain writes in code: tool call -> request, tool result -> call, and so on.
ROUND_RELATIONS = ("evidenced_by", "caused_by")


@dataclass(frozen=True)
class RecallConfig:
    pool_by_depth: Mapping[RecallDepth, int] = field(default_factory=_pools)
    rrf_k: int = 60
    # Cross-encoding a whole high-depth pool is too slow; rerank the fused head only.
    rerank_pool: int = 150
    rerank_top: int = 30
    graph_seeds: int = 10
    max_sub_queries: int = 4
    # What the answer sees: the best max_items whose reranker score is at least min_relevance.
    # A fixed 0.3 floor cut the deciding facts in the Mem2Act runs while the context sat mostly
    # empty, so the count, not the score, is the main limit. Results with no positive score
    # still abstain. TODO(recall): confirm both on the full 10-question set.
    max_items: int = 10
    min_relevance: float = 0.1
    # Reranker scores are not calibrated across questions: a vague back-reference ("the film
    # we picked") scores its own evidence 0.06-0.08. Keep this many best positive results even
    # under min_relevance; 0 restores strict abstention.
    min_items: int = 5
    # Skip a fact once this many facts it shares a `similar` link with are already kept, so
    # near-repeats (five Pride facts) cannot fill a list answer. Counting questions ("how many
    # times") keep every repeat. 0 turns it off.
    similar_cap: int = 2
    # After filtering, follow the round links of the top kept facts so a request, its tool
    # call and the result arrive together.
    followup_seeds: int = 3
    followup_limit: int = 6
    # When the question points back at earlier talk or the best result is weak, add the
    # verbatim source rounds of the top kept facts: they keep exact IDs and turn order that
    # extracted facts can paraphrase away.
    source_rounds: int = 2
    source_round_below: float = 0.3
    source_round_chars: int = 2400
    # Boosts are multiplied in as relevance * (1 + boost), total clipped to +/-max_boost.
    max_boost: float = 0.25
    recency_boost: float = 0.10
    window_boost: float = 0.10
    evidence_boost: float = 0.05
    # For questions that point back at earlier talk ("that server"), the user's own earlier
    # requests are the facts that name the thing.
    user_request_boost: float = 0.05
    # TODO(recall): tune once real fact ages are available.
    recency_half_life_days: float = 90.0
    # With Reflect enabled, widen from hot to warm/archived until this many results are kept.
    # Without a reranker, search all tiers rather than assuming relevance.
    tier_min_hits: int = 3
