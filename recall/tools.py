"""Agent-facing Recall tools: search_memory, get_episode and get_supporting.

Scope is bound in code when the tools are built, so no tool schema exposes user, project or
session arguments. search_memory is a preset Recaller.recall call, not a second pipeline.
"""

from collections.abc import Callable
from typing import Any

from recall.config import DEFAULT_TYPES, MemoryType, RecallDepth
from recall.query import parse_time_range
from recall.recall import Recaller
from retain.models import Scope

SEARCH_DEPTH = RecallDepth.MID
SEARCH_MAX_TOKENS = 4096


def make_recall_tool_functions(
    recaller: Recaller, scope: Scope
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Plain callables for ADK or Reflect's loop, bound to one caller's scope."""

    def search_memory(
        query: str, time_range: str | None = None, include_episodes: bool = False
    ) -> dict[str, Any]:
        """Searches long-term memory more deeply than the automatic per-turn context.

        Args:
            query: What to look for, in natural language.
            time_range: Optional dates as "YYYY-MM-DD/YYYY-MM-DD" (inclusive) or a phrase
                such as "last month".
            include_episodes: Also return the original conversation rounds.
        """
        now = recaller.clock()
        warnings: list[str] = []
        try:
            window = parse_time_range(time_range, now)
        except ValueError as e:
            window = None
            warnings.append(f"time_range ignored: {e}")
        if time_range and window is None and not warnings:
            warnings.append(f"time_range ignored: could not resolve {time_range!r}")
        types = DEFAULT_TYPES + (MemoryType.EPISODE,) if include_episodes else DEFAULT_TYPES
        result = recaller.recall(
            scope,
            query,
            query_time=now,
            temporal_window=window,
            max_tokens=SEARCH_MAX_TOKENS,
            depth=SEARCH_DEPTH,
            types=types,
        ).to_dict()
        if warnings:
            result["warnings"] = warnings
        return result

    def get_episode(episode_id: str) -> dict[str, Any]:
        """Returns the original conversation round a memory came from.

        Args:
            episode_id: The source_episode_id of a search result.
        """
        episode = recaller.store.get_episode(scope, episode_id)
        return episode if episode is not None else {"error": "not_found", "episode_id": episode_id}

    def get_supporting(observation_id: str) -> dict[str, Any]:
        """Returns the facts behind an observation.

        Args:
            observation_id: The id of an observation search result.
        """
        facts = recaller.store.get_supporting(scope, observation_id)
        return {"observation_id": observation_id, "facts": [fact.to_dict() for fact in facts]}

    return {
        "search_memory": search_memory,
        "get_episode": get_episode,
        "get_supporting": get_supporting,
    }


def make_recall_tools(recaller: Recaller, scope: Scope) -> list[Any]:
    """Wraps the Recall tools as ADK FunctionTools."""
    from google.adk.tools import FunctionTool

    return [FunctionTool(f) for f in make_recall_tool_functions(recaller, scope).values()]
