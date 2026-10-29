"""Validate the small public JSON format and keep scoring labels out of inference."""

from datetime import datetime, timezone

from devtools.files import safe_name


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be an ISO string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() is None:
        raise ValueError("Message timestamps must include a timezone")
    return result


def validate_history(history):
    if not isinstance(history, dict):
        raise ValueError("History must be a JSON object")
    safe_name(history.get("id"))
    sessions = history.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("History needs a nonempty sessions array")
    ids = set()
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError("Session must be an object")
        sid = safe_name(session.get("id"))
        if sid in ids:
            raise ValueError("Session IDs must be unique within a history")
        ids.add(sid)
        messages = session.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Each session needs messages")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Message must be an object")
            if message.get("role") not in {"user", "assistant", "tool", "system"}:
                raise ValueError("Message role must be user, assistant, tool or system")
            if "content" not in message and "tool_calls" not in message:
                raise ValueError("Messages need content or tool_calls")
            if message.get("timestamp") is not None:
                timestamp(message["timestamp"])
    return history


def validate_questions(document):
    questions = document.get("questions") if isinstance(document, dict) else None
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions file needs a nonempty questions array")
    ids = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Question must be an object")
        qid = safe_name(question.get("id"))
        if qid in ids:
            raise ValueError("Question IDs must be unique")
        ids.add(qid)
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError("Each question needs nonempty question text")
        if "tools" in question:
            if not isinstance(question["tools"], list) or any(
                not isinstance(t, dict)
                or not isinstance(t.get("name"), str)
                or not isinstance(t.get("parameters"), dict)
                for t in question["tools"]
            ):
                raise ValueError("Tools need name and JSON-schema parameters")
        if "expected_answer" in question and not isinstance(question["expected_answer"], str):
            raise ValueError("expected_answer must be text")
        if "expected_tool_call" in question and not isinstance(
            question["expected_tool_call"], dict
        ):
            raise ValueError("expected_tool_call must be an object")
    return questions


def inference_questions(questions):
    return [{k: q[k] for k in ("id", "question", "tools") if k in q} for q in questions]


def reference_time(history):
    dates = [
        timestamp(m["timestamp"])
        for s in history["sessions"]
        for m in s["messages"]
        if m.get("timestamp")
    ]
    # A fixed execution clock enables exact request reuse and repeatable age/TTL policies.
    # It is never represented as a known source date when the input omitted timestamps.
    return max(dates) if dates else datetime(2026, 1, 1, tzinfo=timezone.utc)


def score(question, answer):
    import json

    if "expected_tool_call" in question:
        try:
            actual = json.loads(answer)
        except ValueError:
            return {"metric": "exact_tool_call", "correct": False}
        # Deliberately strict local development check, not a semantic/official benchmark judge.
        from devtools.files import canonical

        correct = canonical(actual) == canonical(question["expected_tool_call"])
        return {"metric": "exact_tool_call", "correct": correct}
    if "expected_answer" in question:
        return {
            "metric": "exact_text",
            "correct": answer.strip() == question["expected_answer"].strip(),
        }
    return None
