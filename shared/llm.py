"""Text generation boundary shared by the summarizer and Recall's query analysis."""

import json
import re
import time
from collections.abc import Callable

Generate = Callable[[str], str]

# The one Gemini model every stage uses unless a caller passes another. Benchmark
# metering prices it in bench/metering.py.
DEFAULT_MODEL = "gemini-4-hydrogen"


class GeminiGenerator:
    """Plain-text Gemini call: prompt in, response text out."""

    def __init__(self, client, *, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model

    def __call__(self, prompt: str) -> str:
        return self.client.models.generate_content(model=self.model, contents=prompt).text


class GeminiStructuredGenerator(GeminiGenerator):
    """Reflect's provider-neutral prompt/schema interface backed by Gemini JSON output."""

    def __call__(self, prompt, response_schema):
        def generate(text):
            response = self.client.models.generate_content(
                model=self.model,
                contents=text,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": dict(response_schema),
                },
            )
            return response.text

        value = json.loads(generate_with_retry(generate, prompt))
        if not isinstance(value, dict):
            raise ValueError("Reflect generation must return a JSON object")
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
    code = _status_code(error)
    if code is not None:
        return code in _RETRYABLE_CODES
    status = getattr(error, "status", None)
    if not isinstance(status, str):
        match = _LEADING_STATUS.match(str(error))
        status = match.group(1) if match else None
    return status in _RETRYABLE_STATUSES


# Gemini on the global endpoint draws on shared capacity, whose 429 spells can last minutes:
# 8 attempts wait 3, 6, 12, 24, 48, 60, 60 s (about 3.5 min) before a request gives up.
RETRY_ATTEMPTS = 8


def backoff(attempt: int) -> float:
    """Seconds to wait after failed attempt `attempt` (0-based) of a retryable error."""
    return min(60, 3 * 2**attempt)


def generate_with_retry(generate: Generate, prompt: str, attempts: int = RETRY_ATTEMPTS) -> str:
    """Retries rate-limit and temporary server errors with `backoff`; others raise at once."""
    for attempt in range(attempts):
        try:
            return generate(prompt)
        except Exception as e:
            if not is_rate_limited(e) or attempt == attempts - 1:
                raise
            time.sleep(backoff(attempt))
    raise AssertionError("unreachable")
