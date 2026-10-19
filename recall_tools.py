"""Agent-facing Recall tools: search_memory, get_episode and get_supporting.

Scope is bound in code when the tools are built, so no tool schema exposes user, project or
session arguments. search_memory is a preset recall() call, not a second retrieval pipeline.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from recall import (
    DEFAULT_TYPES,
    Embedder,
    Generate,
    RecallStore,
    Reranker,
    Scope,
    TimeWindow,
    parse_time_phrase,
    recall,
)

SEARCH_DEPTH = "mid"
SEARCH_MAX_TOKENS = 4096


@dataclass
class RecallDeps:
    store: RecallStore
    embedder: Embedder
    reranker: Optional[Reranker] = None
    generate: Optional[Generate] = None


def parse_time_range(text: Optional[str], now: datetime) -> Optional[TimeWindow]:
    """Accepts "START/END" ISO dates (end date inclusive) or a phrase such as "last month"."""
    if not text or not text.strip():
        return None
    if "/" in text:
        start_text, end_text = (part.strip() for part in text.split("/", 1))
        start = datetime.fromisoformat(start_text)
        end = datetime.fromisoformat(end_text)
        if len(end_text) == 10:
            end += timedelta(days=1)
        tz = now.tzinfo or timezone.utc
        return TimeWindow(
            start.replace(tzinfo=start.tzinfo or tz), end.replace(tzinfo=end.tzinfo or tz)
        )
    return parse_time_phrase(text, now)


def make_recall_tool_functions(
    deps: RecallDeps, user_id: str, project_id: Optional[str], session_id: Optional[str]
) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Plain callables for ADK or Reflect's loop, bound to one caller's scope."""
    scope = Scope(user_id, project_id, session_id)

    def search_memory(
        query: str, time_range: Optional[str] = None, include_episodes: bool = False
    ) -> Dict[str, Any]:
        """Searches long-term memory more deeply than the automatic per-turn context.

        Args:
            query: What to look for, in natural language.
            time_range: Optional dates as "YYYY-MM-DD/YYYY-MM-DD" (inclusive) or a phrase
                such as "last month".
            include_episodes: Also return the original conversation rounds.
        """
        now = datetime.now(timezone.utc)
        warnings: List[str] = []
        try:
            window = parse_time_range(time_range, now)
        except ValueError as e:
            window = None
            warnings.append(f"time_range ignored: {e}")
        if time_range and window is None and not warnings:
            warnings.append(f"time_range ignored: could not resolve {time_range!r}")
        types = DEFAULT_TYPES + ("episode",) if include_episodes else DEFAULT_TYPES
        result = recall(
            user_id,
            project_id,
            session_id,
            query,
            query_time=now,
            temporal_window=window,
            max_tokens=SEARCH_MAX_TOKENS,
            depth=SEARCH_DEPTH,
            types=types,
            store=deps.store,
            embedder=deps.embedder,
            reranker=deps.reranker,
            generate=deps.generate,
        ).to_dict()
        if warnings:
            result["warnings"] = warnings
        return result

    def get_episode(episode_id: str) -> Dict[str, Any]:
        """Returns the original conversation round a memory came from.

        Args:
            episode_id: The source_episode_id of a search result.
        """
        episode = deps.store.get_episode(scope, episode_id)
        return episode if episode is not None else {"error": "not_found", "episode_id": episode_id}

    def get_supporting(observation_id: str) -> Dict[str, Any]:
        """Returns the facts behind an observation.

        Args:
            observation_id: The id of an observation search result.
        """
        facts = deps.store.get_supporting(scope, observation_id)
        return {"observation_id": observation_id, "facts": [fact.to_dict() for fact in facts]}

    return {
        "search_memory": search_memory,
        "get_episode": get_episode,
        "get_supporting": get_supporting,
    }


def make_recall_tools(
    deps: RecallDeps, user_id: str, project_id: Optional[str], session_id: Optional[str]
) -> List[Any]:
    """Wraps the Recall tools as ADK FunctionTools."""
    from google.adk.tools import FunctionTool

    functions = make_recall_tool_functions(deps, user_id, project_id, session_id)
    return [FunctionTool(function) for function in functions.values()]
