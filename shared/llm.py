"""Text generation boundary shared by the summarizer and Recall's query analysis."""

import json
import time
from collections.abc import Callable

Generate = Callable[[str], str]


class GeminiGenerator:
    """Plain-text Gemini call: prompt in, response text out."""

    def __init__(self, client, *, model: str = "gemini-3.5-flash"):
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
_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "500 INTERNAL", "503", "UNAVAILABLE", "504",
              "DEADLINE_EXCEEDED")  # fmt: skip


def is_rate_limited(error: Exception) -> bool:
    """True for errors a retry can fix: rate limits and temporary server failures."""
    text = str(error)
    return any(marker in text for marker in _RETRYABLE)


def generate_with_retry(generate: Generate, prompt: str, attempts: int = 4) -> str:
    """Retries rate-limit errors with exponential backoff (3s, 6s, 12s); others raise at once."""
    for attempt in range(attempts):
        try:
            return generate(prompt)
        except Exception as e:
            if not is_rate_limited(e) or attempt == attempts - 1:
                raise
            time.sleep(3 * (2**attempt))
    raise AssertionError("unreachable")
