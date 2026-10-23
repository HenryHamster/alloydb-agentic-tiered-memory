"""Recall pipeline: find stored evidence and return it ranked, dated, typed and token-fitted.

    analyze_query    time phrases -> date window; multi-part questions -> sub-queries
    embed            one vector per sub-query (a failure skips the vector channel)
    search           vector, keyword, query-entity and temporal channels in one store call
    graph_expand     one hop over memory_links and shared entities from the fused top hits
    fuse_rrf         reciprocal rank fusion; drops memories from rounds the caller already shows
    rerank           cross-encode the fused head against every sub-query
    apply_boosts     small capped recency / window-fit / evidence boosts
    filter_and_abstain  relevance floor (reranked only) and superseded memories
    pack             best-first into the token budget as dated, typed lines

Every stage is injected or pure, so the pipeline runs offline with InMemoryRecallStore.
See docs/RECALL_NOTES.md for decisions and what is still open.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from recall.config import RecallConfig, RecallDepth, parse_types
from recall.models import QueryPlan, RecallResult, TimeWindow, Tracer, as_aware
from recall.packing import pack
from recall.query import analyze_query
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
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.generate = generate
        self.count_tokens = count_tokens
        self.config = config
        self.clock = clock

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

        The score floor applies to reranker scores only; fused ranks are not calibrated
        relevance, so unreranked results are never abstained on score.

        exclude_episode_ids are rounds the caller already shows verbatim (the Valkey sliding
        window); memories drawn from them are dropped before reranking.
        """
        depth = RecallDepth.parse(depth)
        types = parse_types(types)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        config = self.config
        as_of = as_aware(query_time) or self.clock()
        pool = config.pool_by_depth[depth]
        tracer = Tracer()

        if not query.strip():
            return RecallResult.empty(QueryPlan([query], temporal_window), max_tokens, [])

        with tracer.stage("analyze_query") as stage:
            plan = analyze_query(
                query, as_of, temporal_window, decompose, self.generate, config.max_sub_queries
            )
            stage.count = len(plan.sub_queries)
            stage.detail = {"window_source": plan.window_source, "llm_used": plan.llm_used}

        with tracer.stage("embed") as stage:
            try:
                embeddings = self.embedder(plan.sub_queries)
                stage.count = len(embeddings)
            except Exception as e:
                # A zero vector would return arbitrary neighbours; skip the channel instead.
                logger.warning("Query embedding failed; skipping vector search: %s", e)
                embeddings = None
                stage.detail = {"error": str(e)}

        with tracer.stage("search") as stage:
            lists = self.store.search(
                scope,
                plan.sub_queries,
                embeddings,
                plan.window,
                types,
                as_of,
                include_invalid,
                pool,
            )
            stage.count = sum(len(results) for results in lists.values())
            stage.detail = {"lists": {name: len(results) for name, results in lists.items()}}

        with tracer.stage("graph_expand") as stage:
            seeds = [s.memory.id for s in fuse_rrf(lists, config.rrf_k)[: config.graph_seeds]]
            if seeds:
                neighbors = self.store.expand_neighbors(
                    scope, seeds, types, as_of, include_invalid, pool
                )
                if neighbors:
                    lists = {**lists, "graph": neighbors}
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
                plan.sub_queries, fused, self.reranker, config.rerank_pool, config.rerank_top
            )
            stage.count = len(ranked)
            stage.detail = {"reranked": reranked}

        with tracer.stage("apply_boosts") as stage:
            boosted = apply_boosts(ranked, as_of, plan.window, config)
            stage.count = len(boosted)

        with tracer.stage("filter_and_abstain") as stage:
            floor = config.score_floor if reranked else 0.0
            kept, dropped = filter_and_abstain(boosted, as_of, include_invalid, floor)
            stage.count = len(kept)
            stage.detail = {"score_floor": floor, **dropped}

        with tracer.stage("pack") as stage:
            items, context, used, truncated = pack(kept, as_of, max_tokens, self.count_tokens)
            stage.count = len(items)
            stage.detail = {"tokens_used": used, "truncated": truncated}

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


def build_recaller(conn_factory, genai_client, *, config: RecallConfig = RecallConfig()):
    """Production wiring: AlloyDB store, Retain's embedder, ai.rank() and Gemini."""
    from recall.alloydb_store import AlloyDBRecallStore
    from recall.reranker import AlloyDBReranker
    from retain.embeddings import GeminiEmbedder
    from shared.llm import GeminiGenerator

    return Recaller(
        store=AlloyDBRecallStore(conn_factory),
        # Same model as the write path, so query and fact vectors are comparable.
        embedder=GeminiEmbedder(genai_client),
        reranker=AlloyDBReranker(conn_factory),
        generate=GeminiGenerator(genai_client),
        config=config,
    )
