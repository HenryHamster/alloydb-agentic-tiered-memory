"""Evidence check: one LLM call marks which top candidates help answer the question.

The cross-encoder scores how close a memory is to the question's topic. On questions where
that is not enough (see RecallConfig.judge), this asks the model which memories actually
supply the answer, so a request "for cocktail ideas" stops outranking the fact naming the fruit.
"""

import json
from collections.abc import Sequence

from recall.models import Memory
from recall.query import _strip_fences
from shared.llm import Generate, generate_with_retry

# The check is on the answer path: retry a temporary provider error once, like query analysis.
JUDGE_ATTEMPTS = 2
# Long memories are cut in the prompt; the evidence is nearly always in the first part.
JUDGE_TEXT_CHARS = 500


def judge_prompt(question: str, memories: Sequence[Memory]) -> str:
    lines = []
    for number, memory in enumerate(memories, 1):
        date = memory.primary_date()
        label = f"({date.date().isoformat()}) " if date else ""
        text = " ".join(memory.text.split())[:JUDGE_TEXT_CHARS]
        lines.append(f"{number}. {label}{text}")
    listing = "\n".join(lines)
    return f"""You check which memories help answer a question. The memories are numbered.

A memory is evidence when it states something the answer needs: a fact that answers the question, or one item of the answer. A memory that is only about the same topic is not evidence.
- If the question asks what the user did, used, made, bought, visited or attended, a request, question, plan, wish or idea is not evidence of it; only something the user actually did is.
- If the question asks for several items or a count, mark every memory that names a separate item, even when it mentions the item only in passing.
- Dates in parentheses are when the memory happened or was said.

Return ONLY JSON: {{"evidence": [numbers of the evidence memories]}}, with [] when none is evidence.

Question: {question}

Memories:
{listing}

JSON:"""


def judge_evidence(generate: Generate, question: str, memories: Sequence[Memory]) -> set[str]:
    """Ids of the memories the model marks as evidence. Raises ValueError on unusable output,
    so the caller can fall back to the score-only fill."""
    if not memories:
        return set()
    reply = generate_with_retry(generate, judge_prompt(question, memories), JUDGE_ATTEMPTS)
    try:
        data = json.loads(_strip_fences(reply))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Evidence check returned malformed JSON") from error
    numbers = data.get("evidence") if isinstance(data, dict) else None
    if not isinstance(numbers, list):
        raise ValueError("Evidence check reply has no evidence list")
    evidence = set()
    for number in numbers:
        # Booleans are ints in Python; a bare true is not a memory number.
        if (
            isinstance(number, int)
            and not isinstance(number, bool)
            and 1 <= number <= len(memories)
        ):
            evidence.add(memories[number - 1].id)
    return evidence
