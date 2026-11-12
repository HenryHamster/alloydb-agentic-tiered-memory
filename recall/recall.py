"""Recall pipeline: find stored evidence and return it ranked, dated, typed and token-fitted.

    analyze_query    time phrases -> date window; multi-part questions, back-references and
                     abbreviations -> sub-queries
    embed            one vector per sub-query (a failure skips the vector channel)
    search           vector, keyword, query-entity and temporal channels in one store call
    graph_expand     one hop over memory_links and shared entities from the fused top hits
    fuse_rrf         reciprocal rank fusion; drops memories from rounds the caller already shows
    rerank           cross-encode the fused head against every sub-query
    apply_boosts     small capped recency / window-fit / evidence / user-request boosts
    filter_and_abstain  the best max_items at or above min_relevance, never fewer than
                        min_items positive ones (both apply to reranked scores only); hides
                        superseded memories
    graph_followup   adds the round links (request, tool call, result) of the top kept facts
    source_rounds    for back-references or weak results, adds the verbatim rounds behind
                     the top kept facts
    pack             best-first into the token budget as dated, typed lines

Every stage is injected or pure, so the pipeline runs offline with InMemoryRecallStore.
See docs/RECALL_NOTES.md for decisions and what is still open.
"""

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone

from recall.config import (
    DEFAULT_TYPES,
    ROUND_RELATIONS,
    MemoryType,
    RecallConfig,
    RecallDepth,
    parse_types,
)
from recall.models import Memory, QueryPlan, RecallResult, Scored, TimeWindow, Tracer, as_aware
from recall.packing import pack
from recall.query import analyze_query, said_on_day
from recall.ranking import Reranker, apply_boosts, filter_and_abstain, fuse_rrf, rerank
from recall.store import RecallStore
from retain.embeddings import Embedder
from retain.models import Scope
from shared.llm import Generate
from shared.tokens import TokenCounter, estimate_tokens

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Recaller:
    """Runs Recall for one query. Stages are injected; see build_recaller."""

    def __init__(
        self,
        *,
        store: RecallStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        generate: Generate | None = None,
        count_tokens: TokenCounter = estimate_tokens,
        config: RecallConfig = RecallConfig(),
        clock: Callable[[], datetime] = _utcnow,
        reflect_store=None,
        strict: bool = False,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.generate = generate
        self.count_tokens = count_tokens
        self.config = config
        self.clock = clock
        self.reflect_store = reflect_store
        self.strict = strict

    def recall(
        self,
        scope: Scope,
        query: str,
        *,
        query_time: datetime | None = None,
        temporal_window: TimeWindow | None = None,
        max_tokens: int = 2048,
        depth: RecallDepth | str = RecallDepth.LOW,
        types: Sequence[str] | None = None,
        include_invalid: bool = False,
        decompose: bool = True,
        exclude_episode_ids: Sequence[str] = (),
    ) -> RecallResult:
        """Finds stored evidence for `query` that `scope` may read.

        min_relevance applies to reranker scores only; fused ranks are not calibrated
        relevance, so unreranked results are never abstained on score.

        exclude_episode_ids are rounds the caller already shows verbatim (the Valkey sliding
        window); memories drawn from them are dropped before reranking.
        """
        # Normalise inputs and fix the reference time; depth sets the candidate pool size.
        depth = RecallDepth.parse(depth)
        if types is None and self.reflect_store is not None:
            types = (*DEFAULT_TYPES, MemoryType.MENTAL_MODEL)
        types = parse_types(types)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        config = self.config
        as_of = as_aware(query_time) or self.clock()
        pool = config.pool_by_depth[depth]
        tracer = Tracer()

        # Blank query: nothing to search for, so return an empty (abstained) result.
        if not query.strip():
            return RecallResult.empty(QueryPlan([query], temporal_window), max_tokens, [])

        # Resolve "last week"-style phrases to a date window and split compound questions.
        with tracer.stage("analyze_query") as stage:
            plan = analyze_query(
                query,
                as_of,
                temporal_window,
                decompose,
                self.generate,
                config.max_sub_queries,
                strict=self.strict,
            )
            stage.count = len(plan.sub_queries)
            stage.detail = {
                "window_source": plan.window_source,
                "llm_used": plan.llm_used,
                "refers_back": plan.refers_back,
            }

        # Embed each sub-query for the vector channel.
        with tracer.stage("embed") as stage:
            try:
                embeddings = self.embedder(plan.sub_queries)
                stage.count = len(embeddings)
            except Exception as e:
                if self.strict:
                    raise
                # A zero vector would return arbitrary neighbours; skip the channel instead.
                logger.warning("Query embedding failed; skipping vector search: %s", e)
                embeddings = None
                stage.detail = {"error": str(e)}

        # Reuse the same channels/ranking when expanding the working set. A reranked
        # hot set may finish early; uncalibrated fusion always searches every tier.
        lists = {}
        visited = []
        passes = ("hot", "warm", "archived") if self.reflect_store is not None else (None,)
        for tier in passes:
            if tier is not None:
                visited.append(tier)
            search_kwargs = {"tiers": [tier]} if tier is not None else {}
            with tracer.stage("search") as stage:
                found = self.store.search(
                    scope,
                    plan.sub_queries,
                    embeddings,
                    plan.window,
                    types,
                    as_of,
                    include_invalid,
                    pool,
                    **search_kwargs,
                )
                for name, hits in found.items():
                    lists.setdefault(name, []).extend(hits)
                stage.count = sum(len(hits) for hits in found.values())
                stage.detail = {"tier": tier, "lists": {k: len(v) for k, v in found.items()}}

            with tracer.stage("graph_expand") as stage:
                seeds = [s.memory.id for s in fuse_rrf(lists, config.rrf_k)[: config.graph_seeds]]
                if seeds:
                    kwargs = {"tiers": visited} if tier is not None else {}
                    neighbors = self.store.expand_neighbors(
                        scope, seeds, types, as_of, include_invalid, pool, **kwargs
                    )
                    if neighbors:
                        lists["graph"] = neighbors
                    stage.count = len(neighbors)
                stage.detail = {"seeds": len(seeds)}

            with tracer.stage("fuse_rrf") as stage:
                fused = fuse_rrf(lists, config.rrf_k)
                if exclude_episode_ids:
                    excluded = set(exclude_episode_ids)
                    before = len(fused)
                    fused = [s for s in fused if s.memory.source_episode_id not in excluded]
                    stage.detail = {"in_window": before - len(fused)}
                fused = fused[:pool]
                stage.count = len(fused)

            with tracer.stage("rerank") as stage:
                ranked, reranked = rerank(
                    plan.sub_queries,
                    fused,
                    self.reranker,
                    config.rerank_pool,
                    config.rerank_top,
                    strict=self.strict,
                )
                stage.count = len(ranked)
                stage.detail = {"reranked": reranked}

            with tracer.stage("apply_boosts") as stage:
                boosted = apply_boosts(
                    ranked, as_of, plan.window, config, refers_back=plan.refers_back
                )
                stage.count = len(boosted)

            with tracer.stage("filter_and_abstain") as stage:
                min_relevance = config.min_relevance if reranked else 0.0
                max_items = config.list_max_items if plan.wants_list else config.max_items
                kept, stage.detail = filter_and_abstain(
                    boosted, as_of, include_invalid, min_relevance, max_items,
                    config.min_items if reranked else 0,
                )  # fmt: skip
                stage.count = len(kept)
            if reranked and len(kept) >= config.tier_min_hits:
                break

        with tracer.stage("graph_followup") as stage:
            added = self._follow_round_links(
                scope, plan, kept, types, as_of, include_invalid, visited, exclude_episode_ids
            )
            kept = sorted([*kept, *added], key=lambda item: item.score, reverse=True)
            stage.count = len(added)
            stage.detail = {"added": [item.summary() for item in added]}

        with tracer.stage("source_rounds") as stage:
            rounds = self._source_rounds(scope, plan, kept, exclude_episode_ids)
            kept = [*kept, *rounds]
            stage.count = len(rounds)
            stage.detail = {"episodes": [item.memory.id for item in rounds]}

        if (
            depth == RecallDepth.LOW
            and config.quote_messages > 0
            and embeddings is not None
            and self.reranker is not None
            and not _ASKS_DATE.match(query)
            # A list needs every matching fact, and quotes would push items out.
            and not plan.wants_list
        ):
            with tracer.stage("quote_messages") as stage:
                quotes, stage.detail = self._quote_messages(
                    scope, plan, kept, embeddings, exclude_episode_ids, said_on_day(query)
                )
                kept = [*kept, *quotes]
                stage.count = len(quotes)

        # Format the survivors as dated, typed lines until the token budget runs out.
        with tracer.stage("pack") as stage:
            items, context, used, truncated = pack(kept, as_of, max_tokens, self.count_tokens)
            stage.count = len(items)
            stage.detail = {"tokens_used": used, "truncated": truncated}

        if self.reflect_store is not None:
            with tracer.stage("record_access") as stage:
                facts = [item.memory.id for item in items if item.memory.type == MemoryType.FACT]
                macros = [
                    item.memory.id for item in items if item.memory.type == MemoryType.MENTAL_MODEL
                ]
                self.reflect_store.record_access(facts, macros, as_of)
                stage.count = len(facts) + len(macros)

        # Return the packed context plus the per-stage trace for debugging.
        return RecallResult(
            items=items,
            context=context,
            tokens_used=used,
            max_tokens=max_tokens,
            truncated=truncated,
            abstained=not items,
            sub_queries=plan.sub_queries,
            window=plan.window,
            trace=tracer.stages,
        )

    def _quote_messages(self, scope, plan, kept, embeddings, exclude_episode_ids, day=None):
        """Original messages that match the question on their own, placed after the facts.

        Searched by meaning, reranked against every sub-query like any candidate, and kept
        only above quote_min_relevance; a message already shown in a source round is skipped.
        A question about what was said on a named day ("on 16 June, 2023") searches only that
        day's messages: facts carry when an event happened, not the day it was talked about.
        """
        config = self.config
        try:
            found = self.store.search_messages(
                scope, embeddings, config.quote_pool, exclude_episode_ids, on_day=day
            )
        except Exception as e:
            if self.strict:
                raise
            logger.warning("Message search failed; answering without quotes: %s", e)
            return [], {"error": str(e)}
        shown = "\n".join(i.memory.text for i in kept if i.memory.type == MemoryType.EPISODE)
        candidates = [Scored(m, 0.0, channels=("quote",)) for m in found if m.text not in shown]
        if not candidates:
            return [], {"candidates": 0}
        n = len(candidates)
        ranked, reranked = rerank(
            plan.sub_queries, candidates, self.reranker, n, n, strict=self.strict
        )
        if not reranked:
            return [], {"candidates": n, "reranked": False}
        quotes = [
            replace(item, score=item.relevance)
            for item in ranked
            if item.relevance >= config.quote_min_relevance
        ][: config.quote_messages]
        return quotes, {
            "day": day.isoformat() if day else None,
            "candidates": n,
            "top": [(item.memory.id, round(item.relevance, 3)) for item in ranked[:5]],
            "quoted": [item.memory.id for item in quotes],
        }

    def _source_rounds(
        self,
        scope: Scope,
        plan: QueryPlan,
        kept: Sequence[Scored],
        exclude_episode_ids: Sequence[str],
    ) -> list[Scored]:
        """The verbatim rounds behind the top kept facts, for weak or back-referring questions.

        Placed after the ranked memories with the score of the fact that led to them, so the
        answer sees exact values (IDs, codes) and the order of the original turns.
        """
        config = self.config
        best = max((item.relevance for item in kept), default=0.0)
        if not kept or config.source_rounds <= 0:
            return []
        if not (plan.refers_back or best < config.source_round_below):
            return []
        excluded, episode_ids = set(exclude_episode_ids), {}
        for item in kept:
            episode_id = item.memory.source_episode_id
            if item.memory.type != MemoryType.FACT or not episode_id or episode_id in excluded:
                continue
            episode_ids.setdefault(episode_id, item)
            if len(episode_ids) >= config.source_rounds:
                break
        rounds = []
        for episode_id, origin in episode_ids.items():
            try:
                episode = self.store.get_episode(scope, episode_id)
            except Exception as e:
                if self.strict:
                    raise
                logger.warning("Source round lookup failed; skipping it: %s", e)
                continue
            if not episode or not episode["messages"]:
                continue
            text = "\n".join(f"{m['speaker']}: {m['text']}" for m in episode["messages"])
            first = episode["messages"][0].get("occurred_at")
            memory = Memory(
                id=episode_id,
                type=MemoryType.EPISODE,
                text=text[: config.source_round_chars],
                occurred_at=datetime.fromisoformat(first) if first else None,
                source_episode_id=episode_id,
            )
            rounds.append(
                Scored(memory, origin.relevance, score=origin.score, channels=("source_round",))
            )
        return rounds

    def _follow_round_links(
        self,
        scope: Scope,
        plan: QueryPlan,
        kept: Sequence[Scored],
        types: Sequence[MemoryType],
        as_of: datetime,
        include_invalid: bool,
        visited_tiers: Sequence[str],
        exclude_episode_ids: Sequence[str],
    ) -> list[Scored]:
        """Relevant facts linked to the top kept facts by Retain's round links.

        Search often finds one end of a request -> tool call -> result chain but not the other
        (the request that names the server). Neighbours are reranked and filtered like every
        other candidate; without a reranker their scores would not compare, so none are added.
        """
        config = self.config
        seeds = [item.memory.id for item in kept if item.memory.type == MemoryType.FACT]
        seeds = seeds[: config.followup_seeds]
        if self.reranker is None or not seeds:
            return []

        neighbors = self.store.expand_neighbors(
            scope, seeds, types, as_of, include_invalid, config.followup_limit,
            tiers=visited_tiers or None, relations=ROUND_RELATIONS,
        )  # fmt: skip
        known = {item.memory.id for item in kept}
        candidates = [
            Scored(memory, 0.0, channels=("graph_followup",))
            for memory in neighbors
            if memory.id not in known and memory.source_episode_id not in exclude_episode_ids
        ]
        n = len(candidates)
        ranked, reranked = rerank(
            plan.sub_queries, candidates, self.reranker, n, n, strict=self.strict
        )
        if not reranked:
            return []
        boosted = apply_boosts(ranked, as_of, plan.window, config, refers_back=plan.refers_back)
        added, _ = filter_and_abstain(
            boosted, as_of, include_invalid, config.min_relevance, config.followup_limit
        )
        return added


# Questions answered by a date or duration, where facts carry the resolved date.
_ASKS_DATE = re.compile(
    r"\s*(?:when|since when|how long|how many (?:days|weeks|months|years)"
    r"|what (?:year|month|date|day|time))\b",
    re.IGNORECASE,
)


def build_recaller(
    conn_factory,
    genai_client,
    *,
    embedding_client=None,
    reflect_store=None,
    strict: bool = False,
    config: RecallConfig = RecallConfig(),
):
    """Production wiring: AlloyDB store, Retain's embedder, ai.rank() and Gemini."""
    from recall.alloydb_store import AlloyDBRecallStore
    from recall.reranker import AlloyDBReranker
    from retain.embeddings import GeminiEmbedder
    from shared.llm import GeminiGenerator

    return Recaller(
        store=AlloyDBRecallStore(conn_factory, reflect_enabled=reflect_store is not None),
        # Same model as the write path, so query and fact vectors are comparable.
        embedder=GeminiEmbedder(embedding_client or genai_client),
        reranker=AlloyDBReranker(conn_factory),
        generate=GeminiGenerator(genai_client),
        config=config,
        reflect_store=reflect_store,
        strict=strict,
    )
