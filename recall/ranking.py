"""Ranking stages: fuse the search channels, rerank, boost, then filter and abstain."""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from recall.config import MemoryType, RecallConfig
from recall.models import Memory, Scored, TimeWindow
from retain.models import Speaker

logger = logging.getLogger(__name__)

Reranker = Callable[[str, Sequence[str]], list[float]]

# How many dropped candidates the filter stage lists in the trace.
TRACE_DROPPED = 10


def fuse_rrf(ranked_lists: Mapping[str, Sequence[Memory]], k: int = 60) -> list[Scored]:
    """Reciprocal rank fusion: sum of 1 / (k + rank) across lists, with no training labels.

    Each result keeps the names of the lists it appeared in, for the trace.
    """
    scores: dict[str, float] = {}
    channels: dict[str, list[str]] = {}
    memories: dict[str, Memory] = {}
    for name, results in ranked_lists.items():
        seen = set()
        for memory in results:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            memories.setdefault(memory.id, memory)
            scores[memory.id] = scores.get(memory.id, 0.0) + 1.0 / (k + len(seen))
            channels.setdefault(memory.id, []).append(name)
    ordered = sorted(memories, key=lambda memory_id: scores[memory_id], reverse=True)
    return [
        Scored(memories[memory_id], scores[memory_id], channels=tuple(channels[memory_id]))
        for memory_id in ordered
    ]


def rerank(
    queries: Sequence[str],
    fused: Sequence[Scored],
    reranker: Reranker | None,
    pool: int,
    top: int,
    *,
    strict: bool = False,
) -> tuple[list[Scored], bool]:
    """Cross-encodes the fused head and keeps the top results.

    Relevance is the best score against any sub-query. Without a reranker (or if it fails),
    relevance is the fused score relative to the leader, and the result is marked unreranked.
    """
    head = list(fused[:pool])
    if not head:
        return [], reranker is not None
    if reranker is not None:
        try:
            texts = [scored.memory.text for scored in head]
            # Start from the first sub-query's scores, not 0.0: seeding with zeros would clamp
            # any negative reranker score to 0 and make a real score look like "no signal".
            best: list[float] | None = None
            for query in queries:
                scores = reranker(query, texts)
                if len(scores) != len(head):
                    raise ValueError(f"Reranker returned {len(scores)} scores for {len(head)}")
                scores = [float(score) for score in scores]
                best = scores if best is None else [max(o, n) for o, n in zip(best, scores)]
            if best is None:
                best = [0.0] * len(head)
            reranked = [replace(scored, relevance=score) for scored, score in zip(head, best)]
            reranked.sort(key=lambda scored: scored.relevance, reverse=True)
            return reranked[:top], True
        except Exception as e:
            if strict:
                raise
            logger.warning("Reranker failed; keeping fused order: %s", e)
    leader = head[0].relevance or 1.0
    relative = [replace(scored, relevance=scored.relevance / leader) for scored in head[:top]]
    return relative, False


def apply_boosts(
    scored: Sequence[Scored],
    as_of: datetime,
    window: TimeWindow | None,
    config: RecallConfig = RecallConfig(),
    *,
    refers_back: bool = False,
) -> list[Scored]:
    """Applies small capped boosts after reranking so relevance still decides.

    Recency (only without a window, so dated questions are not pulled toward the present),
    window fit, evidence count and, for questions that refer back to earlier talk, the user's
    own facts together stay within +/-max_boost.
    """
    boosted = []
    for item in scored:
        memory, boost = item.memory, 0.0
        date = memory.primary_date()
        if window is None and date is not None:
            age_days = max(0.0, (as_of - date).total_seconds() / 86400)
            decay = 0.5 ** (age_days / config.recency_half_life_days)
            boost += config.recency_boost * (2 * decay - 1)
        interval = memory.interval(as_of)
        if window is not None and interval is not None:
            boost += config.window_boost if window.overlaps(*interval) else -config.window_boost
        evidence = max(memory.evidence_count, len(memory.supporting_ids), 1)
        boost += config.evidence_boost * min(evidence - 1, 4) / 4
        if refers_back and _is_user_fact(memory):
            boost += config.user_request_boost
        boost = max(-config.max_boost, min(config.max_boost, boost))
        score = item.relevance * (1 + boost) * memory.confidence
        boosted.append(replace(item, boost=boost, score=score))
    boosted.sort(key=lambda item: item.score, reverse=True)
    return boosted


def _is_user_fact(memory: Memory) -> bool:
    return memory.type == MemoryType.FACT and memory.speaker == Speaker.USER


def filter_and_abstain(
    scored: Sequence[Scored],
    as_of: datetime,
    include_invalid: bool,
    min_relevance: float,
    max_items: int,
) -> tuple[list[Scored], dict[str, Any]]:
    """Keeps the best `max_items` with relevance >= `min_relevance`; hides superseded memories.

    `scored` is best-first. A result with no positive relevance is never kept, so a query
    with nothing relevant still abstains. Stores already exclude superseded rows; this check
    also covers stores that do not. The returned detail counts each kind of drop and lists
    the best dropped candidates, so a missing memory can be traced.
    """
    kept, below_min, over_max, stale = [], [], [], 0
    for item in scored:
        if item.memory.consolidated_into or (not include_invalid and item.memory.is_stale(as_of)):
            stale += 1
        elif item.relevance <= 0 or item.relevance < min_relevance:
            below_min.append(item)
        elif len(kept) >= max_items:
            over_max.append(item)
        else:
            kept.append(item)
    dropped = sorted([*below_min, *over_max], key=lambda item: item.relevance, reverse=True)
    return kept, {
        "min_relevance": min_relevance,
        "max_items": max_items,
        "below_min": len(below_min),
        "over_max": len(over_max),
        "stale_hidden": stale,
        "dropped": [item.summary() for item in dropped[:TRACE_DROPPED]],
    }
