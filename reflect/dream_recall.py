"""Opt-in Recall adapter for approved, fresh Dream insights.

The underlying Recall store still handles facts, macros and episodes. Insights
use the existing mental_model channel, with an explicit proposed/inference label.
Staged, stale and out-of-scope lessons never enter the candidate pool.
"""

from datetime import datetime

from recall.config import MemoryType
from recall.models import Memory
from recall.store import InMemoryRecallStore
from retain.models import ClaimType


def insight_memory(insight):
    return Memory(
        id=insight["insight_id"],
        type=MemoryType.MENTAL_MODEL,
        text="[Dream inference; verify supporting evidence] " + insight["text"],
        recorded_at=datetime.fromisoformat(insight["created_at"]),
        valid_from=datetime.fromisoformat(insight["created_at"]),
        claim_type=ClaimType.PROPOSED,
        supporting_ids=tuple(insight["fact_ids"]),
        evidence_count=len(insight["evidence"]),
        tier="hot",
    )


class DreamRecallStore:
    def __init__(self, base, dreamer):
        self.base, self.dreamer = base, dreamer
        self.reflect_enabled = getattr(base, "reflect_enabled", False)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def search(
        self,
        scope,
        queries,
        embeddings,
        window,
        types,
        as_of,
        include_invalid,
        limit,
        *,
        tiers=None,
    ):
        lists = self.base.search(
            scope, queries, embeddings, window, types, as_of, include_invalid, limit, tiers=tiers
        )
        if MemoryType.MENTAL_MODEL not in types or tiers is not None and "hot" not in tiers:
            return lists
        candidates = [
            (insight_memory(i), i.get("embedding"))
            for i in self.dreamer.insights(scope, as_of=as_of)
        ]
        # Lessons concern a range of events; do not present their generation date
        # as an event date. Date-filtered questions use the base facts/timelines.
        if window is not None:
            return lists
        for index, query in enumerate(queries):
            keyword = InMemoryRecallStore._keyword(candidates, query, limit)
            if keyword:
                lists[f"dream_keyword:{index}"] = keyword
            if embeddings is not None:
                vector = InMemoryRecallStore._vector(candidates, embeddings[index], limit)
                if vector:
                    lists[f"dream_vector:{index}"] = vector
        return lists

    def get_supporting(self, scope, observation_id):
        for insight in self.dreamer.insights(scope):
            if insight["insight_id"] == observation_id:
                # Source quotes remain available even if Recall's base store has
                # no get-by-fact-ID API. These are evidence, not fabricated facts.
                return [
                    Memory(
                        id=f"{observation_id}:evidence:{n}",
                        type=MemoryType.FACT,
                        text=e["quote"],
                        source_episode_id=None,
                    )
                    for n, e in enumerate(insight["evidence"])
                ]
        return self.base.get_supporting(scope, observation_id)
