"""Pack ranked memories into a token budget as dated, typed context lines."""

from collections.abc import Sequence
from datetime import datetime

from recall.models import Memory, RecallItem, Scored
from shared.tokens import TokenCounter, estimate_tokens


def format_line(memory: Memory, stale: bool) -> str:
    """`[id] (type | date | speaker | claim_type | superseded date) text`, omitting unknowns."""
    labels = [str(memory.type)]
    date = memory.primary_date()
    if date:
        labels.append(date.date().isoformat())
    if memory.speaker:
        labels.append(str(memory.speaker))
    if memory.claim_type:
        labels.append(str(memory.claim_type))
    if stale and memory.valid_to:
        labels.append(f"superseded {memory.valid_to.date().isoformat()}")
    return f"[{memory.id}] ({' | '.join(labels)}) {memory.text}"


def pack(
    scored: Sequence[Scored],
    as_of: datetime,
    max_tokens: int,
    count_tokens: TokenCounter = estimate_tokens,
) -> tuple[list[RecallItem], str, int, bool]:
    """Fills the budget best-first; skips items that do not fit so smaller ones still can."""
    items, lines, used, truncated = [], [], 0, False
    for item in scored:
        stale = item.memory.is_stale(as_of)
        line = format_line(item.memory, stale)
        tokens = count_tokens(line)
        if used + tokens > max_tokens:
            truncated = True
            continue
        used += tokens
        lines.append(line)
        items.append(
            RecallItem(item.memory, item.score, item.relevance, len(items) + 1, stale, tokens)
        )
    return items, "\n".join(lines), used, truncated
