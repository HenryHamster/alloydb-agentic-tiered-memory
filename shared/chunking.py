"""Length-bounded text chunking with exact, lossless character offsets."""

import re

_PROTECTED = re.compile(r"```[\s\S]*?(?:```|$)|`[^`\n]*`|https?://\S+")
_SENTENCE_END = re.compile(r'[.!?]["\')]*\s+')
_ABBREVIATION = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|e\.g|i\.e|etc|vs)|\b[A-Z])\.$")


def chunk_spans(text: str, max_chars: int = 700) -> list[tuple[int, int]]:
    """Split text into contiguous [start, end) spans of at most max_chars.

    Cuts in the second half of each window, preferring a paragraph break, then a sentence end
    (outside code and URLs), then a newline, then a space, else a hard cut. Joining the spans
    reproduces the text exactly.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    protected = [m.span() for m in _PROTECTED.finditer(text)]
    spans, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            end = start + _cut(text[start:end], start, protected)
        spans.append((start, end))
        start = end
    return spans


def _cut(window: str, offset: int, protected: list[tuple[int, int]]) -> int:
    half = len(window) // 2
    paragraph = window.rfind("\n\n")
    if paragraph >= half:
        return paragraph + 2
    sentences = [
        m.end()
        for m in _SENTENCE_END.finditer(window)
        if m.start() >= half
        and not any(a <= offset + m.start() < b for a, b in protected)
        and not _ABBREVIATION.search(window[: m.start() + 1])
    ]
    if sentences:
        return sentences[-1]
    for separator in ("\n", " "):
        split = window.rfind(separator)
        if split >= half:
            return split + 1
    return len(window)
