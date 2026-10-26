import json
import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import CancelledError, Future
from concurrent.futures import wait as wait_futures
from datetime import datetime, timezone
from typing import Optional

from retain.models import Scope

logger = logging.getLogger(__name__)

MAX_KEPT_FAILURES = 100


def build_temporal_rules_prompt(
    active_session_id: str, prev_session_id: Optional[str] = None
) -> str:
    """REUSABLE PROMPT HELPER: Defines temporal coreference rules for both Read and Write paths."""
    return f"""TEMPORAL COREFERENCE RULES:
1. Do not use relative temporal words like 'currently', 'now', 'this session', 'previous session', or 'last session'.
2. Replace any relative temporal reference with explicit session identifiers:
   - Active Session: '{active_session_id}'
   - Previous Session: '{prev_session_id if prev_session_id else active_session_id}'
   (e.g., write 'User prefers Python in session {active_session_id}' instead of 'User currently prefers Python')."""


class AsyncMemoryWorker:
    """Off-thread daemon queue worker performing background entity extraction and upserts into AlloyDB.

    With `retainer`, each turn runs the Retain pipeline instead of the codelab's agent_entities
    upsert, so only one writer owns long-term memory.
    """

    def __init__(self, conn_factory, llm_client, retainer=None):
        self.conn_factory = conn_factory
        self.llm_client = llm_client
        self.retainer = retainer
        self.work_queue = queue.Queue()
        self.total_extraction_tokens = 0
        self._lock = threading.Lock()
        self._pending: set[Future] = set()
        # Bounded so an unwatched long-running worker cannot grow without limit; every error
        # also stays on its job's Future.
        self._failures = deque(maxlen=MAX_KEPT_FAILURES)
        self._closed = False
        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self.worker_thread.start()

    def enqueue_extraction(
        self,
        user_id: str,
        session_id: str,
        user_msg: str,
        ai_msg: str,
        prev_session_id: Optional[str] = None,
        project_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> Future:
        """Queues one round. Pass the sliding window's round_id as `episode_id` so Recall can
        skip memories from rounds the window still shows verbatim."""
        sent_at = datetime.now(timezone.utc)  # message time, not processing time
        future = Future()
        future._completion_callback_done = threading.Event()
        with self._lock:
            if self._closed:
                raise RuntimeError("Memory worker is closed")
            self._pending.add(future)
            future.add_done_callback(self._completed)
        self.work_queue.put(
            (
                user_id,
                session_id,
                user_msg,
                ai_msg,
                prev_session_id,
                project_id,
                sent_at,
                episode_id,
                future,
            )
        )
        return future

    def wait(self, timeout: float | None = None) -> None:
        """Wait for queued jobs and re-raise background failures."""
        with self._lock:
            pending = tuple(self._pending)
        if pending:
            deadline = None if timeout is None else time.monotonic() + timeout
            _, unfinished = wait_futures(pending, timeout=timeout)
            if unfinished:
                raise TimeoutError("Memory worker jobs did not finish before timeout")
            for future in pending:
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                if not future._completion_callback_done.wait(remaining):
                    raise TimeoutError("Memory worker completion was not reported before timeout")
        with self._lock:
            if self._failures:
                raise self._failures.popleft()

    def close(self, timeout: float | None = None) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self.work_queue.put(None)
        self.worker_thread.join(timeout)
        if self.worker_thread.is_alive():
            raise TimeoutError("Memory worker did not stop before timeout")
        try:
            self.wait(timeout=0)
        finally:
            store = getattr(self.retainer, "store", None)
            close_store = getattr(store, "close", None)
            if close_store is not None:
                close_store()

    def _completed(self, future: Future) -> None:
        try:
            error = future.exception()
        except CancelledError as cancelled:
            error = cancelled
        with self._lock:
            self._pending.discard(future)
            if error is not None:
                self._failures.append(error)
        future._completion_callback_done.set()

    def _run_worker(self):
        while True:
            item = self.work_queue.get()
            if item is None:
                self.work_queue.task_done()
                break
            (
                user_id,
                session_id,
                user_msg,
                ai_msg,
                prev_session_id,
                project_id,
                sent_at,
                episode_id,
                future,
            ) = item
            if not future.set_running_or_notify_cancel():
                self.work_queue.task_done()
                continue
            try:
                if self.retainer is not None:
                    scope = Scope(user_id=user_id, project_id=project_id, session_id=session_id)
                    result = self.retainer.retain_turn(
                        scope, user_msg, ai_msg, timestamp=sent_at, episode_id=episode_id
                    )
                else:
                    result = self._extract_and_upsert(
                        user_id, session_id, user_msg, ai_msg, prev_session_id
                    )
                future.set_result(result)
            except Exception as error:
                logger.exception("Background retention failed")
                future.set_exception(error)
            finally:
                self.work_queue.task_done()

    def _extract_and_upsert(
        self,
        user_id: str,
        session_id: str,
        user_msg: str,
        ai_msg: str,
        prev_session_id: Optional[str] = None,
    ):
        temporal_rules = build_temporal_rules_prompt(session_id, prev_session_id)
        json_fmt = '[{"entity_name": "...", "project_id": "...", "scope": "global|project|session", "summary": "..."}]'

        prompt = f"""Extract key preferences and named entities from this exchange.
Format output strictly as a JSON array of objects with structure: {json_fmt}.

{temporal_rules}

Rules for Scope:
- 'global': Universal developer preferences.
- 'project': Architecture, database choices, timeouts, and rules for a specific project.
- 'session': Ephemeral task state tied strictly to a single session.

User: {user_msg}
AI: {ai_msg}

Output (JSON array only):"""

        self.total_extraction_tokens += max(1, len(prompt) // 4)

        for attempt in range(4):
            try:
                response = self.llm_client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 3:
                    time.sleep(3 * (2**attempt))
                else:
                    raise
        content = str(response.text).strip()
        if not content:
            return

        extracted_list = json.loads(content)
        if not isinstance(extracted_list, list):
            return

        upsert_sql = """
            INSERT INTO agent_entities (user_id, entity_name, project_id, session_id, scope, summary, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (user_id, entity_name)
            DO UPDATE SET
                project_id = EXCLUDED.project_id,
                session_id = EXCLUDED.session_id,
                scope = EXCLUDED.scope,
                summary = EXCLUDED.summary,
                updated_at = NOW();
        """

        conn = self.conn_factory()
        try:
            with conn:
                with conn.cursor() as cur:
                    for item in extracted_list:
                        e_name = item.get("entity_name", "General_Fact")
                        p_id = item.get("project_id", "global")
                        s_scope = item.get("scope", "project")
                        summary = item.get("summary", "")
                        if e_name and summary:
                            cur.execute(
                                upsert_sql, (user_id, e_name, p_id, session_id, s_scope, summary)
                            )
            logger.info("Saved %d entities for user %s", len(extracted_list), user_id)
        finally:
            conn.close()
