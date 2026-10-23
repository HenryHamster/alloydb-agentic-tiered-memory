"""Token estimates for budgeting prompts and packed context."""

import math
from collections.abc import Callable

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """About four characters per token; pass a model tokenizer where exact counts matter."""
    return max(1, math.ceil(len(text) / 4))
