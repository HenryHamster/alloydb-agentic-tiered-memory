"""Generation contracts, Gemini adapter, and retry policy for the memory stages."""

import json
import re
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

from google.genai.errors import APIError

Generate = Callable[[str], str]
JsonValue = dict[str, Any] | list[Any]
T = TypeVar("T")

# The one Gemini model every stage uses unless a caller passes another. Benchmark
# metering prices it in bench/metering.py.
DEFAULT_MODEL = "gemini-4-hydrogen"


class LLMProvider(Protocol):
    """One SDK attempt per call. Memory stages choose their own retry budget."""

    def generate_text(self, prompt: str) -> str: ...

    def generate_json(
        self, prompt: str, schema: Mapping[str, Any], *, thinking_level: str | None = None
    ) -> JsonValue: ...


class ProviderError(RuntimeError):
    """Request failed. The message contains no request body or credentials."""

    def __init__(self, message: str, *, retryable: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class RefusedResponse(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(f"Model refused the request ({reason})")
        self.reason = reason


class IncompleteResponse(RuntimeError):
    """The provider stopped before producing a complete answer."""


class InvalidResponse(ValueError):
    """A missing or invalid response, eligible for stage-specific validation handling."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def response_text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidResponse("empty", "Model returned no text")
    return value


def decode_json(text: str | None, schema: Mapping[str, Any]) -> JsonValue:
    if not isinstance(text, str):
        raise InvalidResponse("empty", "Model returned no text")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        raise InvalidResponse("malformed", "Model returned malformed JSON") from None
    expected = list if schema.get("type") == "array" else dict
    if not isinstance(value, expected):
        kind = "array" if expected is list else "object"
        raise InvalidResponse("shape", f"Model must return a JSON {kind}")
    return value


# Keep Retain's legacy candidate refusals. Other finishes still require valid JSON.
_GEMINI_BLOCK_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}


class GeminiGenerator:
    """Gemini adapter that also preserves the original text callable."""

    def __init__(self, client, *, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def __call__(self, prompt: str) -> str:
        return self.generate_text(prompt)

    def generate_text(self, prompt: str) -> str:
        # Benchmarks and legacy callers consume Gemini's text as returned by the SDK.
        return self._generate(prompt).text

    def generate_json(
        self, prompt: str, schema: Mapping[str, Any], *, thinking_level: str | None = None
    ) -> JsonValue:
        response = self._generate(
            prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": dict(schema),
                **thinking(thinking_level),
            },
        )
        feedback = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        reason = getattr(feedback, "name", feedback)
        if isinstance(reason, str) and reason not in ("", "BLOCKED_REASON_UNSPECIFIED"):
            raise RefusedResponse(reason)
        candidates = getattr(response, "candidates", None)
        for candidate in candidates if isinstance(candidates, (list, tuple)) else []:
            finish = getattr(candidate, "finish_reason", None)
            reason = getattr(finish, "name", finish)
            if isinstance(reason, str) and reason in _GEMINI_BLOCK_REASONS:
                raise RefusedResponse(reason)
        # A capped JSON response still goes through the stage's existing validation.
        return decode_json(getattr(response, "text", None), schema)

    def _generate(self, prompt, **kwargs):
        try:
            return self.client.models.generate_content(model=self.model, contents=prompt, **kwargs)
        except APIError as error:
            message = f"Gemini generation failed (HTTP {error.code})"
            if error.code in (400, 404):
                message += ". Check the Gemini model and GENAI_LOCATION"
            elif error.code in (401, 403):
                message += ". Check Google application credentials and Vertex AI model access"
            raise ProviderError(message, retryable=is_rate_limited(error)) from None


class GeminiStructuredGenerator(GeminiGenerator):
    """Reflect's provider-neutral prompt/schema interface backed by Gemini JSON output."""

    def __call__(self, prompt, response_schema):
        return generate_object_with_retry(self, prompt, response_schema)


def generate_object_with_retry(
    provider: LLMProvider, prompt: str, schema: Mapping[str, Any]
) -> dict[str, Any]:
    value = generate_json_with_retry(provider, prompt, schema)
    if not isinstance(value, dict):
        raise InvalidResponse("shape", "Reflect generation must return a JSON object")
    return value


def thinking(level: str | None) -> dict:
    """Request config for a Gemini thinking level (minimal/low/medium/high); None keeps the
    model default. Thinking tokens are most of Retain's cost, and these calls are short
    structured tasks."""
    return {"thinking_config": {"thinking_level": level}} if level else {}


# Temporary provider errors worth another try: rate limits, and the server being busy or timing
# out. On LoCoMo runs a single 504 or 429 from one of ~800 calls otherwise ended the whole run;
# the same request then succeeded on replay.
_RETRYABLE_CODES = frozenset({429, 500, 503, 504})
_RETRYABLE_STATUSES = frozenset(
    {"RESOURCE_EXHAUSTED", "INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED"}
)
# google-genai formats errors as "<code> <STATUS>. <details>"; the details can hold any digits
# or words ("... 1504 ..."), so only the leading code or status counts.
_LEADING_CODE = re.compile(r"\s*(\d{3})\b")
_LEADING_STATUS = re.compile(r"\s*([A-Z][A-Z_]+)\b")


def _status_code(error: Exception) -> int | None:
    code = getattr(error, "code", None)  # google.genai.errors.APIError
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    status_code = getattr(getattr(error, "response", None), "status_code", None)  # httpx
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        return status_code
    match = _LEADING_CODE.match(str(error))
    return int(match.group(1)) if match else None


def is_rate_limited(error: Exception) -> bool:
    """True for errors a retry can fix: rate limits and temporary server failures, judged by
    the HTTP status code (or, without one, the leading gRPC status name)."""
    if isinstance(error, ProviderError):
        return error.retryable
    if isinstance(error, (RefusedResponse, IncompleteResponse, InvalidResponse)):
        return False
    code = _status_code(error)
    if code is not None:
        return code in _RETRYABLE_CODES
    status = getattr(error, "status", None)
    if not isinstance(status, str):
        match = _LEADING_STATUS.match(str(error))
        status = match.group(1) if match else None
    return status in _RETRYABLE_STATUSES


# Gemini on the global endpoint draws on shared capacity, whose 429 spells can last minutes.
# Background work (Retain stages, offline judging) waits 3, 6, 12, 24, 48, 60, 60 s, about
# 3.5 min, before a request gives up. A user is waiting on interactive calls (answers,
# compaction, query analysis), so they keep 4 attempts, about 21 s.
RETRY_ATTEMPTS = 8
INTERACTIVE_ATTEMPTS = 4


def backoff(attempt: int) -> float:
    """Seconds to wait after failed attempt `attempt` (0-based) of a retryable error."""
    return min(60, 3 * 2**attempt)


def generate_with_retry(
    generate: Callable[[str], T], prompt: str, attempts: int = INTERACTIVE_ATTEMPTS
) -> T:
    """Retry transient errors within the attempt count and the usual total backoff."""
    remaining_wait = sum(backoff(attempt) for attempt in range(attempts - 1))
    for attempt in range(attempts):
        try:
            return generate(prompt)
        except Exception as e:
            if not is_rate_limited(e) or attempt == attempts - 1:
                raise
            delay = backoff(attempt)
            if isinstance(e, ProviderError) and e.retry_after is not None:
                delay = max(delay, e.retry_after)
            if delay > remaining_wait:
                raise
            time.sleep(delay)
            remaining_wait -= delay
    raise AssertionError("unreachable")


def generate_json_with_retry(
    provider: LLMProvider,
    prompt: str,
    schema: Mapping[str, Any],
    *,
    attempts: int = INTERACTIVE_ATTEMPTS,
    thinking_level: str | None = None,
) -> JsonValue:
    return generate_with_retry(
        lambda text: provider.generate_json(text, schema, thinking_level=thinking_level),
        prompt,
        attempts,
    )
