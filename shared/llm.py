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


def is_rate_limited(error: Exception) -> bool:
    return "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)


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
