"""Length-based chunking with exact source character offsets."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from retain_models import ConversationChunk, Message


def chunk_conversation(
    messages: Sequence[Message], *, max_chars: int = 2000
) -> list[ConversationChunk]:
    """Keep messages separate; prefer paragraph, sentence, then whitespace boundaries.

    Character limits approximate token budgets (roughly four characters per token).
    No overlap or semantic topic detection yet. Even unbroken strings obey the limit.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    keys = [(message.scope, message.source_id) for message in messages]
    if len(set(keys)) != len(keys):
        raise ValueError("Source IDs must be unique within a scope")
    chunks = []
    for message in messages:
        start = 0
        while start < len(message.text):
            end = min(start + max_chars, len(message.text))
            if end < len(message.text):
                window = message.text[start:end]
                for separator in ("\n\n", ". ", "\n", " "):
                    split = window.rfind(separator)
                    if split >= len(window) // 2:
                        end = start + split + len(separator)
                        break
            key = repr((message.scope, message.source_id, message.text, start, end))
            chunks.append(ConversationChunk(str(uuid5(NAMESPACE_URL, key)), message, start, end))
            start = end
    return chunks
