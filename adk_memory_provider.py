from concurrent.futures import Future
from typing import Any, Dict, Optional

import redis

from async_worker import AsyncMemoryWorker
from hybrid_retriever import retrieve_hybrid_entities
from valkey_buffer import append_session_turn_with_rolling_summary, get_session_context_buffer


class ADKTieredMemoryProvider:
    """Provides ambient memory context before turns and saves history after turns."""

    def __init__(
        self,
        db_conn_factory,
        valkey_client: redis.Redis,
        genai_client: Any,
        trigger_limit: int = 10,
        window_size: int = 4,
        retainer=None,
    ):
        self.conn_factory = db_conn_factory
        self.valkey_client = valkey_client
        self.genai_client = genai_client
        self.trigger_limit = trigger_limit
        self.window_size = window_size
        self.async_worker = AsyncMemoryWorker(db_conn_factory, genai_client, retainer)

    def get_context_for_turn(
        self,
        user_id: str,
        project_id: Optional[str],
        session_id: str,
        user_query: str,
        prev_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reads Valkey buffer (<1ms) and AlloyDB hybrid entities before agent reasoning."""
        buffer_data = get_session_context_buffer(self.valkey_client, session_id)
        rolling_summary = buffer_data["rolling_summary"]

        # Carry over prior session summary when starting a fresh session
        if not rolling_summary and prev_session_id:
            prev_buffer = get_session_context_buffer(self.valkey_client, prev_session_id)
            if prev_buffer["rolling_summary"]:
                rolling_summary = (
                    f"[From prior session {prev_session_id}]:\n" + prev_buffer["rolling_summary"]
                )

        conn = self.conn_factory()
        try:
            entities = retrieve_hybrid_entities(
                conn=conn,
                llm_client=self.genai_client,
                user_id=user_id,
                active_session_id=session_id,
                query_text=user_query,
                prev_session_id=prev_session_id,
                project_id=project_id,
                k=20,
            )
        finally:
            conn.close()

        return {
            "rolling_summary": rolling_summary,
            "recent_turns": buffer_data["recent_turns"],
            "retrieved_entities": entities,
        }

    def format_system_instruction(self, context: Dict[str, Any], base_prompt: str = "") -> str:
        """Injects ambient entities and short-term summaries into the agent's prompt."""
        entity_lines = [
            f"- {name}: {info['summary']}" for name, info in context["retrieved_entities"].items()
        ]
        entity_str = "\n".join(entity_lines) if entity_lines else "No relevant long-term entities."
        summary_str = context["rolling_summary"] or "No prior summary."

        return f"""{base_prompt}

[LONG-TERM PREFERENCES & ENTITIES]
{entity_str}

[SHORT-TERM ROLLING SUMMARY]
{summary_str}

[NOTE ON TOOLS & MEMORY]
Memory persistence is managed automatically by the platform behind the scenes. Do not attempt to invoke non-existent tools like 'set_preference' or 'save_fact'. Only invoke explicitly declared tools when needed.
""".strip()

    def record_turn_async(
        self,
        user_id: str,
        project_id: Optional[str],
        session_id: str,
        user_msg: str,
        ai_msg: str,
        prev_session_id: Optional[str] = None,
    ) -> Future:
        """Updates Valkey sliding window and triggers background AlloyDB extraction."""
        append_session_turn_with_rolling_summary(
            valkey_client=self.valkey_client,
            llm_client=self.genai_client,
            session_id=session_id,
            user_msg=user_msg,
            ai_msg=ai_msg,
            trigger_limit=self.trigger_limit,
            window_size=self.window_size,
        )
        return self.async_worker.enqueue_extraction(
            user_id=user_id,
            session_id=session_id,
            user_msg=user_msg,
            ai_msg=ai_msg,
            prev_session_id=prev_session_id,
            project_id=project_id,
        )

    def wait_for_retention(self, timeout: float | None = None) -> None:
        self.async_worker.wait(timeout=timeout)

    def close(self, timeout: float | None = None) -> None:
        self.async_worker.close(timeout=timeout)
