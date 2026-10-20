"""Recall read path: find stored evidence and return it ranked, dated, typed and token-fitted.

Pipeline: analyze_query -> four searches -> fuse_rrf -> graph expansion -> rerank ->
apply_boosts -> filter_and_abstain -> pack.

Storage, embedding, reranking and the LLM are passed in, so the pipeline runs offline with
fakes. Work that waits on another phase is tagged PLACEHOLDER(RETAIN), PLACEHOLDER(REFLECT) or
PLACEHOLDER(CLOUD); docs/RECALL_NOTES.md lists every one.
"""

import json
import logging
import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

ALL_TYPES = ("fact", "observation", "episode", "mental_model")
# mental_model is returned only when Reflect asks for it through `types`.
# PLACEHOLDER(REFLECT): the store has no mental-model table to read until Reflect defines one,
# and mental-model staleness must be readable as valid_to (see docs/RECALL_NOTES.md).
DEFAULT_TYPES = ("fact", "observation")


class RecallDepth(StrEnum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


POOL_BY_DEPTH = {
    RecallDepth.LOW: 100,
    RecallDepth.MID: 300,
    RecallDepth.HIGH: 1000,
}
RRF_K = 60
# Cross-encoding the whole high-depth pool (1,000) is too slow; rerank the fused head only.
RERANK_POOL = 150
RERANK_TOP = 30
GRAPH_SEEDS = 10
MAX_SUB_QUERIES = 4
# PLACEHOLDER(CLOUD): calibrate against real ai.rank() scores on a labelled sample.
SCORE_FLOOR = 0.3
MAX_BOOST = 0.25
RECENCY_BOOST = 0.10
WINDOW_BOOST = 0.10
EVIDENCE_BOOST = 0.05
# PLACEHOLDER(CLOUD): tune once real fact ages are available.
RECENCY_HALF_LIFE_DAYS = 90.0

Embedder = Callable[[Sequence[str]], List[List[float]]]
Reranker = Callable[[str, Sequence[str]], List[float]]
Generate = Callable[[str], str]
TokenCounter = Callable[[str], int]


def _aware(value: Optional[datetime], tz: Optional[timezone] = None) -> Optional[datetime]:
    """Treats naive datetimes as UTC (or the given zone) so comparisons never mix kinds."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=tz or timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


@dataclass(frozen=True)
class TimeWindow:
    """Half-open date window [start, end)."""

    start: datetime
    end: datetime

    def __post_init__(self):
        object.__setattr__(self, "start", _aware(self.start))
        object.__setattr__(self, "end", _aware(self.end))
        if self.end <= self.start:
            raise ValueError(f"Time window end {self.end} must be after start {self.start}")

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return start < self.end and end >= self.start

    def to_dict(self) -> Dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True)
class Scope:
    """Injected by code, never by the model. The store must enforce it in SQL."""

    user_id: str
    project_id: Optional[str]
    session_id: Optional[str]


@dataclass
class Memory:
    """One stored candidate as the store returns it.

    PLACEHOLDER(RETAIN): field names mirror the Retain plan; columns such as speaker,
    claim_type, source_episode_id and evidence_count are not in Retain's schema yet.
    """

    id: str
    type: str
    text: str
    occurred_at: Optional[datetime] = None
    recorded_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    speaker: Optional[str] = None
    claim_type: Optional[str] = None
    source_episode_id: Optional[str] = None
    supporting_ids: Tuple[str, ...] = ()
    entity_ids: Tuple[str, ...] = ()
    evidence_count: int = 1

    def __post_init__(self):
        for name in ("occurred_at", "recorded_at", "valid_from", "valid_to"):
            setattr(self, name, _aware(getattr(self, name)))
        self.supporting_ids = tuple(self.supporting_ids)
        self.entity_ids = tuple(self.entity_ids)

    def primary_date(self) -> Optional[datetime]:
        return self.occurred_at or self.valid_from or self.recorded_at

    def interval(self, as_of: datetime) -> Optional[Tuple[datetime, datetime]]:
        """Events are points at occurred_at; states run from valid_from until valid_to or now."""
        if self.occurred_at:
            return self.occurred_at, self.occurred_at
        start = self.valid_from or self.recorded_at
        if start is None:
            return None
        return start, max(start, self.valid_to or as_of)

    def is_stale(self, as_of: datetime) -> bool:
        return self.valid_to is not None and self.valid_to <= as_of

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "dates": {
                "occurred_at": _iso(self.occurred_at),
                "recorded_at": _iso(self.recorded_at),
                "valid_from": _iso(self.valid_from),
                "valid_to": _iso(self.valid_to),
            },
            "speaker": self.speaker,
            "claim_type": self.claim_type,
            "source_episode_id": self.source_episode_id,
            "supporting_ids": list(self.supporting_ids),
            "entity_ids": list(self.entity_ids),
        }


class RecallStore(Protocol):
    """Read contract Recall needs from Retain's tables.

    PLACEHOLDER(RETAIN): no AlloyDB implementation exists until Retain's schema lands. See
    docs/RECALL_NOTES.md for the intended single-statement SQL. Every method must filter by
    `scope` with bound parameters, never by formatting IDs into SQL text.
    """

    def search(
        self,
        scope: Scope,
        queries: Sequence[str],
        embeddings: Optional[Sequence[Sequence[float]]],
        window: Optional[TimeWindow],
        types: Sequence[str],
        as_of: datetime,
        include_invalid: bool,
        limit: int,
    ) -> Dict[str, List[Memory]]:
        """Runs vector, keyword, query-entity and temporal searches in one round trip.

        Returns best-first lists keyed "<channel>:<query index>" (temporal is "temporal").
        Skips the vector channel when `embeddings` is None and temporal when `window` is None.
        Unless include_invalid, excludes rows with valid_to <= as_of inside the query so stale
        rows never take pool slots.
        """
        ...

    def expand_neighbors(
        self,
        scope: Scope,
        seed_ids: Sequence[str],
        types: Sequence[str],
        as_of: datetime,
        include_invalid: bool,
        limit: int,
    ) -> List[Memory]:
        """One hop over memory_links and shared entities from the seeds, best-first by weight."""
        ...

    def get_episode(self, scope: Scope, episode_id: str) -> Optional[Dict[str, Any]]:
        """Returns the original conversation round."""
        ...

    def get_supporting(self, scope: Scope, observation_id: str) -> List[Memory]:
        """Returns the facts an observation was built from."""
        ...


@dataclass
class QueryPlan:
    sub_queries: List[str]
    window: Optional[TimeWindow]
    window_source: Optional[str] = None
    llm_used: bool = False


@dataclass
class StageTrace:
    stage: str
    ms: float = 0.0
    count: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "ms": round(self.ms, 3), "count": self.count, **self.detail}


@dataclass
class Scored:
    memory: Memory
    relevance: float
    boost: float = 0.0
    score: float = 0.0


@dataclass
class RecallItem:
    memory: Memory
    score: float
    relevance: float
    rank: int
    stale: bool
    tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.memory.to_dict(),
            "score": round(self.score, 6),
            "relevance": round(self.relevance, 6),
            "rank": self.rank,
            "stale": self.stale,
        }


@dataclass
class RecallResult:
    items: List[RecallItem]
    context: str
    tokens_used: int
    max_tokens: int
    truncated: bool
    abstained: bool
    sub_queries: List[str]
    window: Optional[TimeWindow]
    trace: List[StageTrace]

    def to_dict(self) -> Dict[str, Any]:
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


# ---------------------------------------------------------------- analyze_query

_MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_LAST_N_RE = re.compile(r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?\b")
_MONTH_RE = re.compile(r"\b(?:in|during)\s+(" + "|".join(_MONTHS) + r")(?:\s+((?:19|20)\d{2}))?\b")
_YEAR_RE = re.compile(r"\b(?:in|during)\s+((?:19|20)\d{2})\b")


def _shift_month(year: int, month: int, delta: int) -> Tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _month_window(year: int, month: int, tz) -> TimeWindow:
    next_year, next_month = _shift_month(year, month, 1)
    return TimeWindow(
        datetime(year, month, 1, tzinfo=tz), datetime(next_year, next_month, 1, tzinfo=tz)
    )


def _months_ago(now: datetime, months: int) -> datetime:
    year, month = _shift_month(now.year, now.month, -months)
    next_year, next_month = _shift_month(year, month, 1)
    last_day = (datetime(next_year, next_month, 1) - timedelta(days=1)).day
    return now.replace(year=year, month=month, day=min(now.day, last_day))


def parse_time_phrase(text: str, now: datetime) -> Optional[TimeWindow]:
    """Resolves common relative time phrases without an LLM; returns None when none match."""
    now = _aware(now)
    tz = now.tzinfo
    lowered = text.lower()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = today - timedelta(days=today.weekday())

    match = _LAST_N_RE.search(lowered)
    if match and int(match.group(1)) > 0:
        count, unit = int(match.group(1)), match.group(2)
        if unit == "day":
            start = now - timedelta(days=count)
        elif unit == "week":
            start = now - timedelta(weeks=count)
        else:
            start = _months_ago(now, count * (12 if unit == "year" else 1))
        return TimeWindow(start, now)

    if re.search(r"\byesterday\b", lowered):
        return TimeWindow(today - timedelta(days=1), today)
    if re.search(r"\btoday\b", lowered):
        return TimeWindow(today, today + timedelta(days=1))
    if re.search(r"\b(?:last|previous) week\b", lowered):
        return TimeWindow(monday - timedelta(weeks=1), monday)
    if re.search(r"\bthis week\b", lowered):
        return TimeWindow(monday, monday + timedelta(weeks=1))
    if re.search(r"\b(?:last|previous) month\b", lowered):
        return _month_window(*_shift_month(now.year, now.month, -1), tz)
    if re.search(r"\bthis month\b", lowered):
        return _month_window(now.year, now.month, tz)
    if re.search(r"\b(?:last|previous) year\b", lowered):
        return TimeWindow(
            datetime(now.year - 1, 1, 1, tzinfo=tz), datetime(now.year, 1, 1, tzinfo=tz)
        )
    if re.search(r"\bthis year\b", lowered):
        return TimeWindow(
            datetime(now.year, 1, 1, tzinfo=tz), datetime(now.year + 1, 1, 1, tzinfo=tz)
        )

    match = _MONTH_RE.search(lowered)
    if match:
        month = _MONTHS.index(match.group(1)) + 1
        year = int(match.group(2)) if match.group(2) else now.year - (month > now.month)
        return _month_window(year, month, tz)
    match = _YEAR_RE.search(lowered)
    if match:
        year = int(match.group(1))
        return TimeWindow(datetime(year, 1, 1, tzinfo=tz), datetime(year + 1, 1, 1, tzinfo=tz))
    return None


def _parse_datetime(value: str, tz) -> datetime:
    return _aware(datetime.fromisoformat(value.strip()), tz)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return text


def _decomposition_prompt(query: str, query_time: datetime) -> str:
    return f"""You prepare questions for an AI agent's memory search.

1. If the question asks about more than one distinct thing, split it into at most {MAX_SUB_QUERIES} self-contained search queries; otherwise return it as the only query.
2. If it refers to a time ("last week", "in March", "before the launch"), resolve it to a date window. The current time is {query_time.isoformat()}. The end date is exclusive.
3. Return ONLY JSON: {{"sub_queries": ["..."], "time_window": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null}}

Question: {query}
JSON:"""


def analyze_query(
    query: str,
    query_time: datetime,
    temporal_window: Optional[TimeWindow] = None,
    decompose: bool = True,
    generate: Optional[Generate] = None,
) -> QueryPlan:
    """Turns time phrases into a date window and splits multi-part questions.

    Window precedence: explicit argument, then the rule parser, then the LLM. decompose=False
    (used by Reflect) skips the LLM call entirely and keeps the query whole.
    """
    plan = QueryPlan(sub_queries=[query], window=temporal_window)
    if temporal_window:
        plan.window_source = "explicit"
    else:
        plan.window = parse_time_phrase(query, query_time)
        plan.window_source = "rules" if plan.window else None

    if not decompose or generate is None:
        return plan

    try:
        data = json.loads(_strip_fences(generate(_decomposition_prompt(query, query_time))))
        plan.llm_used = True
        sub_queries = [str(q).strip() for q in data.get("sub_queries") or [] if str(q).strip()]
        seen = {query.lower()}
        for sub_query in sub_queries[:MAX_SUB_QUERIES]:
            if sub_query.lower() not in seen:
                seen.add(sub_query.lower())
                plan.sub_queries.append(sub_query)
        window = data.get("time_window")
        if (
            plan.window is None
            and isinstance(window, dict)
            and window.get("start")
            and window.get("end")
        ):
            tz = query_time.tzinfo
            plan.window = TimeWindow(
                _parse_datetime(window["start"], tz), _parse_datetime(window["end"], tz)
            )
            plan.window_source = "llm"
    except Exception as e:
        logger.warning("Query analysis fell back to the original query: %s", e)
    return plan


# ---------------------------------------------------------------- ranking stages


def fuse_rrf(ranked_lists: Mapping[str, Sequence[Memory]], k: int = RRF_K) -> List[Scored]:
    """Reciprocal rank fusion: sum of 1 / (k + rank) across lists, with no training labels."""
    scores: Dict[str, float] = {}
    memories: Dict[str, Memory] = {}
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
    reranker: Optional[Reranker],
    pool: int = RERANK_POOL,
    top: int = RERANK_TOP,
) -> Tuple[List[Scored], bool]:
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
            best = [0.0] * len(head)
            for query in queries:
                scores = reranker(query, texts)
                if len(scores) != len(head):
                    raise ValueError(
                        f"Reranker returned {len(scores)} scores for {len(head)} texts"
                    )
                best = [max(old, float(new)) for old, new in zip(best, scores)]
            reranked = [Scored(scored.memory, score) for scored, score in zip(head, best)]
            reranked.sort(key=lambda scored: scored.relevance, reverse=True)
            return reranked[:top], True
        except Exception as e:
            logger.warning("Reranker failed; keeping fused order: %s", e)
    leader = head[0].relevance or 1.0
    return [Scored(scored.memory, scored.relevance / leader) for scored in head[:top]], False


def apply_boosts(
    scored: Sequence[Scored], as_of: datetime, window: Optional[TimeWindow]
) -> List[Scored]:
    """Applies small capped boosts after reranking so relevance still decides.

    Recency (only without a window, so dated questions are not pulled toward the present),
    window fit, and evidence count together stay within +/-MAX_BOOST of the relevance score.
    """
    boosted = []
    for item in scored:
        memory, boost = item.memory, 0.0
        date = memory.primary_date()
        if window is None and date is not None:
            age_days = max(0.0, (as_of - date).total_seconds() / 86400)
            boost += RECENCY_BOOST * (2 * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS) - 1)
        interval = memory.interval(as_of)
        if window is not None and interval is not None:
            boost += WINDOW_BOOST if window.overlaps(*interval) else -WINDOW_BOOST
        evidence = max(memory.evidence_count, len(memory.supporting_ids), 1)
        boost += EVIDENCE_BOOST * min(evidence - 1, 4) / 4
        boost = max(-MAX_BOOST, min(MAX_BOOST, boost))
        boosted.append(Scored(memory, item.relevance, boost, item.relevance * (1 + boost)))
    boosted.sort(key=lambda item: item.score, reverse=True)
    return boosted


def filter_and_abstain(
    scored: Sequence[Scored], as_of: datetime, include_invalid: bool, score_floor: float
) -> Tuple[List[Scored], Dict[str, int]]:
    """Drops results below the relevance floor (before boosts) and hides superseded facts.

    The store already excludes superseded rows; this check also covers stores that do not.
    """
    kept, below_floor, stale = [], 0, 0
    for item in scored:
        if item.relevance < score_floor:
            below_floor += 1
        elif not include_invalid and item.memory.is_stale(as_of):
            stale += 1
        else:
            kept.append(item)
    return kept, {"below_floor": below_floor, "stale_hidden": stale}


def estimate_tokens(text: str) -> int:
    """About four characters per token; swap in a model tokenizer for exact counts."""
    return max(1, math.ceil(len(text) / 4))


def format_line(memory: Memory, stale: bool) -> str:
    labels = [memory.type]
    date = memory.primary_date()
    if date:
        labels.append(date.date().isoformat())
    if memory.speaker:
        labels.append(memory.speaker)
    if memory.claim_type:
        labels.append(memory.claim_type)
    if stale and memory.valid_to:
        labels.append(f"superseded {memory.valid_to.date().isoformat()}")
    return f"[{memory.id}] ({' | '.join(labels)}) {memory.text}"


def pack(
    scored: Sequence[Scored],
    as_of: datetime,
    max_tokens: int,
    count_tokens: TokenCounter = estimate_tokens,
) -> Tuple[List[RecallItem], str, int, bool]:
    """Fills the token budget best-first with dated, typed lines; skips items that do not fit."""
    items, lines, used, truncated = [], [], 0, False
    for item in scored:
        stale = item.memory.is_stale(as_of)
        line = format_line(item.memory, stale)
        tokens = count_tokens(line)
        if used + tokens > max_tokens:
            truncated = True
            continue
        used += tokens
        lines.append(line)
        items.append(
            RecallItem(item.memory, item.score, item.relevance, len(items) + 1, stale, tokens)
        )
    return items, "\n".join(lines), used, truncated


# ---------------------------------------------------------------- pipeline


def _from_episode(memory: Memory, episode_ids: set) -> bool:
    if memory.type == "episode":
        return memory.id in episode_ids
    return memory.source_episode_id in episode_ids


class _Tracer:
    def __init__(self):
        self.stages: List[StageTrace] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTrace]:
        entry = StageTrace(name)
        started = time.perf_counter()
        try:
            yield entry
        finally:
            entry.ms = (time.perf_counter() - started) * 1000
            self.stages.append(entry)


def recall(
    user_id: str,
    project_id: Optional[str],
    session_id: Optional[str],
    query: str,
    query_time: Optional[datetime] = None,
    temporal_window: Optional[TimeWindow] = None,
    max_tokens: int = 2048,
    depth: RecallDepth | str = RecallDepth.LOW,
    types: Optional[Sequence[str]] = None,
    include_invalid: bool = False,
    decompose: bool = True,
    *,
    store: RecallStore,
    embedder: Embedder,
    reranker: Optional[Reranker] = None,
    generate: Optional[Generate] = None,
    score_floor: float = SCORE_FLOOR,
    count_tokens: TokenCounter = estimate_tokens,
    exclude_episode_ids: Sequence[str] = (),
) -> RecallResult:
    """Finds stored evidence for a query and returns it ranked, dated, typed and token-fitted.

    The score floor applies to reranker scores only; unreranked results are never abstained
    on score because fused ranks are not calibrated relevance.

    exclude_episode_ids are rounds the caller already shows verbatim (the Valkey sliding
    window). Memories drawn from them are dropped before reranking so they neither repeat
    those turns nor take rerank slots.
    """
    if not isinstance(depth, RecallDepth):
        try:
            depth = RecallDepth(depth)
        except ValueError:
            raise ValueError(
                f"depth must be one of {[d.value for d in RecallDepth]}, got {depth!r}"
            ) from None
    types = tuple(types or DEFAULT_TYPES)
    unknown = set(types) - set(ALL_TYPES)
    if unknown:
        raise ValueError(f"Unknown memory types: {sorted(unknown)}")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    as_of = _aware(query_time) or datetime.now(timezone.utc)
    scope = Scope(user_id, project_id, session_id)
    pool = POOL_BY_DEPTH[depth]
    tracer = _Tracer()

    def empty(plan: QueryPlan) -> RecallResult:
        return RecallResult(
            [], "", 0, max_tokens, False, True, plan.sub_queries, plan.window, tracer.stages
        )

    if not query.strip():
        return empty(QueryPlan([query], temporal_window))

    with tracer.stage("analyze_query") as stage:
        plan = analyze_query(query, as_of, temporal_window, decompose, generate)
        stage.count = len(plan.sub_queries)
        stage.detail = {"window_source": plan.window_source, "llm_used": plan.llm_used}

    with tracer.stage("embed") as stage:
        try:
            embeddings: Optional[List[List[float]]] = embedder(plan.sub_queries)
            stage.count = len(embeddings)
        except Exception as e:
            # A zero vector would return arbitrary neighbours; skip the vector channel instead.
            logger.warning("Query embedding failed; skipping vector search: %s", e)
            embeddings = None
            stage.detail = {"error": str(e)}

    with tracer.stage("search") as stage:
        lists = store.search(
            scope, plan.sub_queries, embeddings, plan.window, types, as_of, include_invalid, pool
        )
        stage.count = sum(len(results) for results in lists.values())
        stage.detail = {"lists": {name: len(results) for name, results in lists.items()}}

    with tracer.stage("graph_expand") as stage:
        seeds = [scored.memory.id for scored in fuse_rrf(lists)[:GRAPH_SEEDS]]
        neighbors = (
            store.expand_neighbors(scope, seeds, types, as_of, include_invalid, pool)
            if seeds
            else []
        )
        if neighbors:
            lists = {**lists, "graph": neighbors}
        stage.count = len(neighbors)
        stage.detail = {"seeds": len(seeds)}

    with tracer.stage("fuse_rrf") as stage:
        fused = fuse_rrf(lists)
        if exclude_episode_ids:
            excluded = set(exclude_episode_ids)
            before = len(fused)
            fused = [scored for scored in fused if not _from_episode(scored.memory, excluded)]
            stage.detail = {"in_window": before - len(fused)}
        fused = fused[:pool]
        stage.count = len(fused)

    with tracer.stage("rerank") as stage:
        ranked, reranked = rerank(plan.sub_queries, fused, reranker)
        stage.count = len(ranked)
        stage.detail = {"reranked": reranked, "pool": min(len(fused), RERANK_POOL)}

    with tracer.stage("apply_boosts") as stage:
        boosted = apply_boosts(ranked, as_of, plan.window)
        stage.count = len(boosted)

    with tracer.stage("filter_and_abstain") as stage:
        floor = score_floor if reranked else 0.0
        kept, dropped = filter_and_abstain(boosted, as_of, include_invalid, floor)
        stage.count = len(kept)
        stage.detail = {"score_floor": floor, **dropped}

    if not kept:
        return empty(plan)

    with tracer.stage("pack") as stage:
        items, context, used, truncated = pack(kept, as_of, max_tokens, count_tokens)
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


# ---------------------------------------------------------------- default backends


def make_genai_embedder(client: Any, model: str = "text-embedding-005") -> Embedder:
    """Embeds sub-queries with the same Vertex AI model the write path uses."""

    def embed(texts: Sequence[str]) -> List[List[float]]:
        response = client.models.embed_content(model=model, contents=list(texts))
        return [list(embedding.values) for embedding in response.embeddings]

    return embed


def make_genai_generate(client: Any, model: str = "gemini-3.5-flash") -> Generate:
    def generate(prompt: str) -> str:
        return client.models.generate_content(model=model, contents=prompt).text

    return generate


def make_alloydb_reranker(
    conn_factory: Callable[[], Any], model_id: str = "semantic-ranker-default-003"
) -> Reranker:
    """Cross-encoder reranking inside AlloyDB with ai.rank().

    PLACEHOLDER(CLOUD): verify the ai.rank() signature and model ID on the live instance; it
    needs the Discovery Engine API enabled and is a preview function.
    """

    def rerank_texts(query: str, texts: Sequence[str]) -> List[float]:
        conn = conn_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("SET google_ml_integration.enable_preview_ai_functions = true;")
                cur.execute(
                    "SELECT index, score FROM ai.rank("
                    "model_id => %s, search_string => %s, documents => %s::text[]);",
                    (model_id, query, list(texts)),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        scores = [0.0] * len(texts)
        for index, score in rows:
            # ai.rank() reports 1-based positions into the documents array.
            scores[int(index) - 1] = float(score)
        return scores

    return rerank_texts
