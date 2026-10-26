"""Provider-neutral structured generation boundary for Reflect stages.

The caller adapts its SDK to generate(prompt, response_schema) -> parsed JSON object.
Model selection, authentication and provider retries belong to that adapter.
"""

from collections.abc import Callable, Mapping
from typing import Any

StructuredGenerator = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def generate_object(
    generate: StructuredGenerator, prompt: str, schema: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Call the adapter and check it returned a JSON object. Field checks belong to the stage."""
    response = generate(prompt, schema)
    if not isinstance(response, Mapping):
        raise ValueError("Structured generator must return a JSON object")
    return response


def require_text(value: Any, name: str) -> str:
    """A nonempty string from untrusted model output, stripped."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model output needs a nonempty {name}")
    return value.strip()
