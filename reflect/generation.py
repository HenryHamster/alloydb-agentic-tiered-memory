"""Provider-neutral structured generation boundary for Reflect stages.

The caller adapts its SDK to generate(prompt, response_schema) -> parsed JSON object.
Model selection, authentication and provider retries belong to that adapter.
"""

from collections.abc import Callable, Mapping
from typing import Any

StructuredGenerator = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
