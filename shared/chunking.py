"""Length-based chunking with exact source character offsets."""

import re
from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from retain_models import ConversationChunk, Message


def chunk_conversation(
    messages: Sequence[Message], *, max_chars: int = 700
) -> list[ConversationChunk]:
    """Split messages at punctuation/whitespace with exact offsets and no overlap."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    keys = [(message.scope, message.source_id) for message in messages]
    if len(set(keys)) != len(keys):
        raise ValueError("Source IDs must be unique within a scope")
    chunks = []
    for message in messages:
        protected = [
            m.span()
            for m in re.finditer(r"```[\s\S]*?(?:```|$)|`[^`\n]*`|https?://\S+", message.text)
        ]
        start = 0
        while start < len(message.text):
            end = min(start + max_chars, len(message.text))
            if end < len(message.text):
                window = message.text[start:end]
                sentence_ends = [
                    m.end()
                    for m in re.finditer(r'[.!?]["\')]*\s+', window)
                    if m.start() >= len(window) // 2
                    and not any(a <= start + m.start() < b for a, b in protected)
                    and not re.search(
                        r"(?:\b(?:Mr|Mrs|Dr|Prof|e\.g|i\.e)|\b[A-Z])\.$", window[: m.start() + 1]
                    )
                ]
                for separator in ("\n\n", "sentence", "\n", " "):
                    if separator == "sentence":
                        if sentence_ends:
                            end = start + sentence_ends[-1]
                            break
                        continue
                    split = window.rfind(separator)
                    if split >= len(window) // 2:
                        end = start + split + len(separator)
                        break
            key = repr((message.scope, message.source_id, message.text, start, end))
            chunks.append(ConversationChunk(str(uuid5(NAMESPACE_URL, key)), message, start, end))
            start = end
    return chunks
