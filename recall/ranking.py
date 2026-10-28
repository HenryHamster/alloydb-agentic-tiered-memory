"""Ranking stages: fuse the search channels, rerank, boost, then filter and abstain."""

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

from recall.config import RecallConfig
from recall.models import Memory, Scored, TimeWindow

logger = logging.getLogger(__name__)

Reranker = Callable[[str, Sequence[str]], list[float]]


def fuse_rrf(ranked_lists: Mapping[str, Sequence[Memory]], k: int = 60) -> list[Scored]:
    """Reciprocal rank fusion: sum of 1 / (k + rank) across lists, with no training labels."""
    scores: dict[str, float] = {}
    memories: dict[str, Memory] = {}
    for results in ranked_lists.values():
        seen = set()
        for memory in results:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            memories.setdefault(memory.id, memory)
            scores[memory.id] = scores.get(memory.id, 0.0) + 1.0 / (k + len(seen))
    ordered = sorted(memories, key=lambda memory_id: scores[memory_id], reverse=True)
    return [Scored(memories[memory_id], scores[memory_id]) for memory_id in ordered]


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
            reranked = [Scored(scored.memory, score) for scored, score in zip(head, best)]
            reranked.sort(key=lambda scored: scored.relevance, reverse=True)
            return reranked[:top], True
        except Exception as e:
            if strict:
                raise
            logger.warning("Reranker failed; keeping fused order: %s", e)
    leader = head[0].relevance or 1.0
    return [Scored(scored.memory, scored.relevance / leader) for scored in head[:top]], False


def apply_boosts(
    scored: Sequence[Scored],
    as_of: datetime,
    window: TimeWindow | None,
    config: RecallConfig = RecallConfig(),
) -> list[Scored]:
    """Applies small capped boosts after reranking so relevance still decides.

    Recency (only without a window, so dated questions are not pulled toward the present),
    window fit, and evidence count together stay within +/-max_boost.
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
        boost = max(-config.max_boost, min(config.max_boost, boost))
        boosted.append(
            Scored(memory, item.relevance, boost, item.relevance * (1 + boost) * memory.confidence)
        )
    boosted.sort(key=lambda item: item.score, reverse=True)
    return boosted


def filter_and_abstain(
    scored: Sequence[Scored],
    as_of: datetime,
    include_invalid: bool,
    score_floor: float,
    fallback_top: int = 0,
) -> tuple[list[Scored], dict[str, int]]:
    """Drops results below the relevance floor (before boosts) and hides superseded memories.

    If the floor rejects every live result, the best `fallback_top` with positive relevance are
    kept instead (counted as "fallback"), so only results with no signal at all abstain.
    Stores already exclude superseded rows; this check also covers stores that do not.
    """
    kept, below, stale = [], [], 0
    for item in scored:
        if item.memory.consolidated_into or (not include_invalid and item.memory.is_stale(as_of)):
            stale += 1
        elif item.relevance < score_floor:
            below.append(item)
        else:
            kept.append(item)
    need = max(0, fallback_top - len(kept))
    fallback = [item for item in below if item.relevance > 0][:need]
    if fallback:
        # Keep the combined list best-first so packing spends the budget on the strongest.
        kept = sorted([*kept, *fallback], key=lambda item: item.score, reverse=True)
    return kept, {
        "below_floor": len(below) - len(fallback),
        "stale_hidden": stale,
        "fallback": len(fallback),
    }
