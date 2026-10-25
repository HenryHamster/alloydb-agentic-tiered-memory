"""Short-term tier: a sliding window of recent turns in Valkey plus a rolling summary.

When a session passes `trigger_limit` rounds, the oldest rounds are summarized into the rolling
summary and only then trimmed, leaving the last `window_size` rounds verbatim. Every round gets a
round_id; turn_context.py passes the window's round IDs to recall() so long-term memories that
restate turns still in the window are skipped.

If Valkey is unreachable, turns go to an in-process fallback with the same behaviour.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

from shared.llm import GeminiGenerator, Generate, generate_with_retry
from shared.tokens import TokenCounter, estimate_tokens

logger = logging.getLogger(__name__)

IN_MEMORY_VALKEY_FALLBACK: Dict[str, Any] = {}
VALKEY_COMPACTION_TOKENS = 0
SUMMARY_MODEL = "gemini-3.5-flash"
TRIGGER_LIMIT = 10
WINDOW_SIZE = 4
STORAGE_ERRORS = (redis.exceptions.RedisError, OSError)


def get_valkey_compaction_tokens() -> int:
    return VALKEY_COMPACTION_TOKENS


def _keys(session_id: str) -> tuple[str, str]:
    # Redis hash tags {session_id} put both keys in the same cluster slot.
    return "session:{" + session_id + "}:turns", "session:{" + session_id + "}:summary"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class Turn:
    """One message. Turns stored before round IDs existed load with round_id and at as None."""

    role: str
    content: str
    round_id: Optional[str] = None
    at: Optional[datetime] = None

    @classmethod
    def from_json(cls, raw: Any) -> "Turn":
        data = json.loads(_text(raw))
        at = data.get("at")
        return cls(
            data["role"],
            data["content"],
            data.get("round_id"),
            datetime.fromisoformat(at) if at else None,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "role": self.role,
                "content": self.content,
                "round_id": self.round_id,
                "at": self.at.isoformat() if self.at else None,
            }
        )

    def line(self) -> str:
        return f"{self.role}: {self.content}"


@dataclass
class SessionWindow:
    session_id: str
    summary: str
    turns: List[Turn]
    degraded: bool = False

    @property
    def round_ids(self) -> List[str]:
        return list(dict.fromkeys(turn.round_id for turn in self.turns if turn.round_id))

    def history(self) -> str:
        return "\n".join(turn.line() for turn in self.turns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "rolling_summary": self.summary,
            "recent_turns": [{"role": turn.role, "content": turn.content} for turn in self.turns],
            "round_ids": self.round_ids,
            "degraded": self.degraded,
        }


class _ValkeyLists:
    def __init__(self, client: Any):
        self.client = client

    def push(self, key: str, values: List[str]) -> None:
        for value in values:
            self.client.rpush(key, value)

    def items(self, key: str) -> List[str]:
        return [_text(value) for value in self.client.lrange(key, 0, -1)]

    def get(self, key: str) -> str:
        return _text(self.client.get(key))

    def set(self, key: str, value: str) -> None:
        self.client.set(key, value)

    def drop_head(self, key: str, count: int) -> None:
        # Trim by count from the head, not to the last N, so a turn appended by a concurrent
        # writer after the summary was built is never trimmed unsummarized.
        self.client.ltrim(key, count, -1)


class _FallbackLists:
    def push(self, key: str, values: List[str]) -> None:
        IN_MEMORY_VALKEY_FALLBACK.setdefault(key, []).extend(values)

    def items(self, key: str) -> List[str]:
        return list(IN_MEMORY_VALKEY_FALLBACK.get(key, []))

    def get(self, key: str) -> str:
        return IN_MEMORY_VALKEY_FALLBACK.get(key, "")

    def set(self, key: str, value: str) -> None:
        IN_MEMORY_VALKEY_FALLBACK[key] = value

    def drop_head(self, key: str, count: int) -> None:
        IN_MEMORY_VALKEY_FALLBACK[key] = IN_MEMORY_VALKEY_FALLBACK.get(key, [])[count:]


def _line(raw: str) -> str:
    try:
        return Turn.from_json(raw).line()
    except (ValueError, KeyError, TypeError):
        # Summarize the raw text rather than block compaction on one bad entry.
        return raw


def _summary_prompt(existing_summary: str, pruned_text: str) -> str:
    return f"""Update the existing rolling conversation summary with the old turns about to be trimmed.
Retain key constraints, user choices, technical decisions, and ongoing goals.

Existing Summary:
{existing_summary or "No prior summary."}

Old turns about to be trimmed:
{pruned_text}

Updated Rolling Summary:"""


class SessionBuffer:
    """Valkey sliding window with summarize-before-trim. Sizes are in rounds (user + assistant)."""

    def __init__(
        self,
        client: Any,
        generate: Optional[Generate] = None,
        trigger_limit: int = TRIGGER_LIMIT,
        window_size: int = WINDOW_SIZE,
        count_tokens: TokenCounter = estimate_tokens,
    ):
        if window_size < 0 or trigger_limit < window_size:
            raise ValueError("Need 0 <= window_size <= trigger_limit")
        self.generate = generate
        self.trigger_limit = trigger_limit
        self.window_size = window_size
        self.count_tokens = count_tokens
        self._valkey = _ValkeyLists(client)
        self._fallback = _FallbackLists()

    def append(
        self, session_id: str, user_msg: str, ai_msg: str, at: Optional[datetime] = None
    ) -> str:
        """Stores one round, compacts if the window is over the trigger, and returns its round_id.

        Pass the round_id to Retain as the episode_id (AsyncMemoryWorker.enqueue_extraction) so
        Recall can skip memories drawn from rounds still in the window.
        """
        turns_key, _ = _keys(session_id)
        round_id = uuid.uuid4().hex
        at = at or datetime.now(timezone.utc)
        values = [
            Turn("user", user_msg, round_id, at).to_json(),
            Turn("assistant", ai_msg, round_id, at).to_json(),
        ]
        try:
            self._valkey.push(turns_key, values)
            lists: Any = self._valkey
        except STORAGE_ERRORS as e:
            logger.warning("Valkey write failed (%s); using the in-process fallback buffer.", e)
            self._fallback.push(turns_key, values)
            lists = self._fallback

        try:
            self._compact(lists, session_id)
        except Exception as e:
            # The turn is stored; the window stays untrimmed and compaction retries next turn.
            logger.warning("Rolling summary failed for session %s; not trimming: %s", session_id, e)
        return round_id

    def read(self, session_id: str) -> SessionWindow:
        try:
            return self._read(self._valkey, session_id, degraded=False)
        except STORAGE_ERRORS as e:
            logger.warning("Valkey read failed (%s); using the in-process fallback buffer.", e)
            return self._read(self._fallback, session_id, degraded=True)

    def _read(self, lists: Any, session_id: str, degraded: bool) -> SessionWindow:
        turns_key, summary_key = _keys(session_id)
        summary = lists.get(summary_key)
        turns = []
        for raw in lists.items(turns_key):
            try:
                turns.append(Turn.from_json(raw))
            except (ValueError, KeyError, TypeError) as e:
                logger.warning("Skipping unreadable turn in session %s: %s", session_id, e)
        return SessionWindow(session_id, summary, turns, degraded)

    def _compact(self, lists: Any, session_id: str) -> None:
        global VALKEY_COMPACTION_TOKENS
        turns_key, summary_key = _keys(session_id)
        raw_turns = lists.items(turns_key)
        if len(raw_turns) <= self.trigger_limit * 2:
            return
        if self.generate is None:
            logger.warning("No summarizer configured; session %s is not compacted", session_id)
            return
        pruned = raw_turns[: len(raw_turns) - self.window_size * 2]
        pruned_text = "\n".join(_line(raw) for raw in pruned)
        prompt = _summary_prompt(lists.get(summary_key), pruned_text)
        VALKEY_COMPACTION_TOKENS += self.count_tokens(prompt)

        summary = generate_with_retry(self.generate, prompt).strip()
        lists.set(summary_key, summary)
        lists.drop_head(turns_key, len(pruned))
        logger.info("Updated rolling summary and trimmed the buffer for session %s", session_id)


# ---------------------------------------------------------------- codelab-compatible API


def append_session_turn_with_rolling_summary(
    valkey_client: redis.Redis,
    llm_client: Any,
    session_id: str,
    user_msg: str,
    ai_msg: str,
    trigger_limit: int = TRIGGER_LIMIT,
    window_size: int = WINDOW_SIZE,
) -> str:
    """Appends a turn and summarizes old turns before trimming them. Returns the round_id."""
    generate = GeminiGenerator(llm_client, model=SUMMARY_MODEL)
    buffer = SessionBuffer(valkey_client, generate, trigger_limit, window_size)
    return buffer.append(session_id, user_msg, ai_msg)


def get_session_context_buffer(valkey_client: redis.Redis, session_id: str) -> Dict[str, Any]:
    """Returns {"rolling_summary", "recent_turns"} in the codelab's original shape."""
    window = SessionBuffer(valkey_client).read(session_id)
    return {
        "rolling_summary": window.summary,
        "recent_turns": [{"role": turn.role, "content": turn.content} for turn in window.turns],
    }
