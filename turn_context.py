"""Per-turn context: the Valkey sliding window plus long-term Recall under one token budget.

The window (recent turns verbatim, older turns as a rolling summary) is charged to the budget
first, because dropping recent turns breaks the dialogue. Recall gets what is left, never less
than MIN_RECALL_TOKENS, and skips memories drawn from rounds still in the window.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from recall import Recaller, RecallResult
from recall.config import RecallDepth
from retain.models import Scope
from valkey_buffer import SessionBuffer, SessionWindow

logger = logging.getLogger(__name__)

TURN_MAX_TOKENS = 4096
MIN_RECALL_TOKENS = 512
TURN_DEPTH = RecallDepth.LOW


@dataclass
class TurnContext:
    window: SessionWindow
    summary: str
    recall: Optional[RecallResult]
    window_tokens: int
    recall_budget: int

    def format(self) -> str:
        memories = self.recall.context if self.recall else ""
        return f"""[LONG-TERM MEMORY]
{memories or "No relevant long-term memories."}

[SHORT-TERM ROLLING SUMMARY]
{self.summary or "No prior summary."}

[RECENT DIALOGUE]
{self.window.history() or "No earlier turns in this session."}"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window.to_dict(),
            "summary": self.summary,
            "recall": self.recall.to_dict() if self.recall else None,
            "window_tokens": self.window_tokens,
            "recall_budget": self.recall_budget,
        }


def build_turn_context(
    buffer: SessionBuffer,
    recaller: Recaller,
    scope: Scope,
    query: str,
    prev_session_id: Optional[str] = None,
    query_time: Optional[datetime] = None,
    max_tokens: int = TURN_MAX_TOKENS,
    *,
    strict: bool = False,
    depth: RecallDepth = TURN_DEPTH,
) -> TurnContext:
    """Reads the session window, then recalls long-term memory into the remaining budget."""
    if not scope.session_id:
        raise ValueError("build_turn_context needs a session_id in scope")
    window = buffer.read(scope.session_id)
    if strict and window.degraded:
        raise RuntimeError("Valkey is unavailable; refusing a degraded candidate turn")
    summary = window.summary
    if not summary and prev_session_id:
        # A fresh session starts from the previous session's summary.
        previous = buffer.read(prev_session_id).summary
        if previous:
            summary = f"[From prior session {prev_session_id}]:\n{previous}"

    count_tokens = recaller.count_tokens
    window_tokens = sum(count_tokens(text) for text in (summary, window.history()) if text)
    recall_budget = max(MIN_RECALL_TOKENS, max_tokens - window_tokens)

    try:
        result: Optional[RecallResult] = recaller.recall(
            scope,
            query,
            query_time=query_time,
            max_tokens=recall_budget,
            depth=depth,
            exclude_episode_ids=window.round_ids,
        )
    except Exception:
        if strict:
            raise
        # Short-term context alone still lets the turn proceed.
        logger.warning("Recall failed; continuing with the session window only", exc_info=True)
        result = None

    return TurnContext(window, summary, result, window_tokens, recall_budget)
