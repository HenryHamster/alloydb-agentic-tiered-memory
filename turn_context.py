"""Per-turn context: the Valkey sliding window plus long-term Recall under one token budget.

The window (recent turns verbatim, older turns as a rolling summary) is charged to the budget
first, because dropping recent turns breaks the dialogue. Recall gets what is left, never less
than MIN_RECALL_TOKENS, and skips memories drawn from rounds still in the window.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from recall import RecallResult, TokenCounter, estimate_tokens, recall
from recall_tools import RecallDeps
from valkey_buffer import SessionBuffer, SessionWindow

logger = logging.getLogger(__name__)

TURN_MAX_TOKENS = 4096
MIN_RECALL_TOKENS = 512
TURN_DEPTH = "low"


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
    deps: RecallDeps,
    user_id: str,
    project_id: Optional[str],
    session_id: str,
    query: str,
    prev_session_id: Optional[str] = None,
    query_time: Optional[datetime] = None,
    max_tokens: int = TURN_MAX_TOKENS,
    count_tokens: TokenCounter = estimate_tokens,
) -> TurnContext:
    """Reads the session window, then recalls long-term memory into the remaining budget."""
    window = buffer.read(session_id)
    summary = window.summary
    if not summary and prev_session_id:
        # A fresh session starts from the previous session's summary.
        previous = buffer.read(prev_session_id).summary
        if previous:
            summary = f"[From prior session {prev_session_id}]:\n{previous}"

    window_tokens = sum(count_tokens(text) for text in (summary, window.history()) if text)
    recall_budget = max(MIN_RECALL_TOKENS, max_tokens - window_tokens)

    try:
        result: Optional[RecallResult] = recall(
            user_id,
            project_id,
            session_id,
            query,
            query_time=query_time or datetime.now(timezone.utc),
            max_tokens=recall_budget,
            depth=TURN_DEPTH,
            store=deps.store,
            embedder=deps.embedder,
            reranker=deps.reranker,
            generate=deps.generate,
            count_tokens=count_tokens,
            exclude_episode_ids=window.round_ids,
        )
    except Exception:
        # Short-term context alone still lets the turn proceed.
        logger.warning("Recall failed; continuing with the session window only", exc_info=True)
        result = None

    return TurnContext(window, summary, result, window_tokens, recall_budget)
