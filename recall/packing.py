"""Pack ranked memories into a token budget as dated, typed context lines."""

from collections.abc import Sequence
from datetime import datetime

from recall.config import MemoryType
from recall.models import Memory, RecallItem, Scored
from shared.tokens import TokenCounter, estimate_tokens

ORIGINAL_WORDING_HEADING = "Original wording (exact words behind the memories above):"
# Experiment (recall-show-relevance): tell the answer model the order and each memory's score.
RELEVANCE_NOTE = (
    "Memories are sorted most relevant first. rel is each memory's retrieval relevance to the "
    "question (0 to 1, higher is a closer match). It measures topic match, not whether the "
    "memory answers the question. Lines without rel are supporting source text."
)


def format_line(memory: Memory, stale: bool, relevance: float = 0.0) -> str:
    """`[id] (type | date | speaker | claim_type | superseded date | rel) text`, omitting unknowns."""
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
    if memory.conflicts_with:
        labels.append("unresolved conflict: " + ", ".join(memory.conflicts_with))
    if memory.confidence < 1:
        labels.append(f"confidence {memory.confidence:g}")
    if relevance > 0:
        labels.append(f"rel {relevance:.2f}")
    return f"[{memory.id}] ({' | '.join(labels)}) {memory.text}"


def pack(
    scored: Sequence[Scored],
    as_of: datetime,
    max_tokens: int,
    count_tokens: TokenCounter = estimate_tokens,
) -> tuple[list[RecallItem], str, int, bool]:
    """Fills the budget best-first; skips items that do not fit so smaller ones still can.

    The context opens with RELEVANCE_NOTE, whose tokens come out of the budget first.
    """
    items, lines, used, truncated = [], [], 0, False
    max_tokens -= count_tokens(RELEVANCE_NOTE)
    quoting = False
    for item in scored:
        stale = item.memory.is_stale(as_of)
        line = format_line(item.memory, stale, item.relevance)
        heading = item.memory.type == MemoryType.MESSAGE and not quoting
        if heading:
            # Quotes sit under their own heading so the memories above stay the answer's basis.
            line = f"{ORIGINAL_WORDING_HEADING}\n{line}"
        tokens = count_tokens(line)
        if used + tokens > max_tokens:
            truncated = True
            continue
        used += tokens
        quoting = quoting or heading
        lines.append(line)
        items.append(
            RecallItem(
                item.memory,
                item.score,
                item.relevance,
                len(items) + 1,
                stale,
                tokens,
                item.channels,
            )
        )
    context = "\n".join([RELEVANCE_NOTE, *lines]) if lines else ""
    return items, context, used, truncated
