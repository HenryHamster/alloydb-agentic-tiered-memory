"""Rank score and tier lifecycle shared by facts and macro memories.

Reflect maintains lifecycle scores and tiers; search execution belongs to the existing Recall
retrieval path. Archived hits are promoted back; long-dormant rows may be purged. Promotion needs
a higher score than demotion (hysteresis), so rows do not flip between tiers on every pass.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from reflect.models import Tier


@dataclass(frozen=True)
class TierConfig:
    hot_cap: int = 200  # per user; overflow drops one tier, lowest rank first
    use_warm: bool = True  # False for small tables: hot and archived only
    hot_promote: float = 1.0  # enter hot at or above
    hot_demote: float = 0.6  # leave hot below (< hot_promote)
    warm_promote: float = 0.35  # archived -> warm at or above
    archive_below: float = 0.2  # warm (or hot without warm) -> archived below, if dormant
    archive_after: timedelta = timedelta(days=30)  # "no hits for N days"
    purge_after: timedelta = timedelta(days=365)  # archived and dormant this long -> purged
    access_half_life: timedelta = timedelta(days=14)
    fresh_half_life: timedelta = timedelta(days=3)
    fresh_weight: float = 1.0  # new rows start hot, then cool unless recalled
    evidence_weight: float = 0.5

    def __post_init__(self):
        if not self.archive_below < self.warm_promote <= self.hot_demote < self.hot_promote:
            raise ValueError("Thresholds need archive < warm_promote <= hot_demote < hot_promote")


def rank_score(
    *,
    access_count: int,
    last_accessed_at: datetime | None,
    created_at: datetime,
    evidence_count: int,
    confidence: float,
    now: datetime,
    config: TierConfig = TierConfig(),
) -> float:
    """confidence * (usage + evidence + freshness), where

    - usage = log1p(access_count) * 0.5 ** (since last access / access_half_life)
    - evidence = evidence_weight * log1p(evidence_count - 1)
    - freshness = fresh_weight * 0.5 ** (age / fresh_half_life)

    A new, never-recalled fact scores fresh_weight (1.0), so it starts hot.
    """
    since_access = _elapsed(last_accessed_at, now) if last_accessed_at else None
    usage = 0.0
    if access_count > 0 and since_access is not None:
        usage = math.log1p(access_count) * _decay(since_access, config.access_half_life)
    evidence = config.evidence_weight * math.log1p(max(evidence_count - 1, 0))
    freshness = config.fresh_weight * _decay(_elapsed(created_at, now), config.fresh_half_life)
    return max(0.0, min(1.0, confidence)) * (usage + evidence + freshness)


def next_tier(
    current: Tier,
    score: float,
    *,
    last_seen: datetime,
    now: datetime,
    config: TierConfig = TierConfig(),
) -> Tier:
    """One lifecycle step for one row, before the hot cap. `last_seen` = last_accessed_at, or
    creation time for rows never recalled; dormant = now - last_seen.

    - hot: stays while score >= hot_demote; else warm (without warm: archived only if cold).
    - warm: score >= hot_promote -> hot; cold (score < archive_below and dormant >=
      archive_after) -> archived; else stays.
    - archived: dormant >= purge_after -> PURGED; score >= warm_promote -> warm (without warm:
      score >= hot_promote -> hot); else stays.
    """
    if current == Tier.PURGED:
        raise ValueError("Purged rows have no next tier")
    if current == Tier.WARM and not config.use_warm:
        current = Tier.HOT  # warm switched off: treat leftover warm rows as hot
    dormant = _elapsed(last_seen, now)
    cold = score < config.archive_below and dormant >= config.archive_after
    if current == Tier.HOT:
        if score >= config.hot_demote:
            return Tier.HOT
        if config.use_warm:
            return Tier.WARM
        return Tier.ARCHIVED if cold else Tier.HOT
    if current == Tier.WARM:
        if score >= config.hot_promote:
            return Tier.HOT
        return Tier.ARCHIVED if cold else Tier.WARM
    if dormant >= config.purge_after:
        return Tier.PURGED
    if config.use_warm and score >= config.warm_promote:
        return Tier.WARM
    if not config.use_warm and score >= config.hot_promote:
        return Tier.HOT
    return Tier.ARCHIVED


def apply_hot_cap(
    proposed: Mapping[str, Tier], scores: Mapping[str, float], config: TierConfig = TierConfig()
) -> dict[str, Tier]:
    """Keep the `hot_cap` highest-ranked hot rows (ties by id); overflow drops to warm, or to
    archived without warm."""
    capped = dict(proposed)
    hot = sorted(
        (row_id for row_id, tier in proposed.items() if tier == Tier.HOT),
        key=lambda row_id: (-scores.get(row_id, 0.0), row_id),
    )
    overflow = Tier.WARM if config.use_warm else Tier.ARCHIVED
    for row_id in hot[max(config.hot_cap, 0) :]:
        capped[row_id] = overflow
    return capped


def promote_on_hit(tier: Tier, config: TierConfig = TierConfig()) -> Tier:
    """Recall hit on an archived row: warm (hot without warm) immediately, no reflect pass."""
    if tier == Tier.ARCHIVED:
        return Tier.WARM if config.use_warm else Tier.HOT
    return tier


def _elapsed(since: datetime, now: datetime) -> timedelta:
    return max(now - since, timedelta(0))


def _decay(elapsed: timedelta, half_life: timedelta) -> float:
    return 0.5 ** (elapsed / half_life)
