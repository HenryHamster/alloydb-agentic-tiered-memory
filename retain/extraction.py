"""Proposal step 2: chunk source documents and extract atomic facts with an LLM."""

from collections.abc import Sequence
from typing import Protocol

from retain.models import Chunk, Document, ExtractedFact, stable_id
from shared.chunking import chunk_spans


def chunk_documents(documents: Sequence[Document], max_chars: int = 700) -> list[Chunk]:
    """Split each document into lossless chunks; IDs are stable across retries."""
    return [
        Chunk(stable_id(document.document_id, start, end), document, start, end)
        for document in documents
        for start, end in chunk_spans(document.text, max_chars)
    ]


class FactExtractor(Protocol):
    def extract(self, chunks: Sequence[Chunk]) -> list[ExtractedFact]:
        """Return validated facts; every fact cites one chunk and quotes evidence from it."""
        ...


class GeminiFactExtractor:
    """Gemini structured-output extractor.

    Contract for each returned fact:
    - 8-15 word self-contained sentence; known who/what/when/where/why go in `details`.
    - `evidence` is an exact substring of the cited chunk; rows failing this are dropped.
    - Speaker, scope and episode come from the chunk's document, never from the model.
    - Assistant preferences are stored as `proposed`; tool output is attributed, not trusted.
    - Model-proposed visibility is narrowed with `resolve_visibility`.
    - Dates resolve relative phrases against the document timestamp; date-only values become
      midnight UTC; ambiguous dates stay null with the phrase kept in `when`.
    - `caused_by` / `evidenced_by` reference other facts' `key` from the same response.
    - One bad row is logged and dropped; it never discards the rest of the response.
    """

    def __init__(
        self,
        client,
        *,
        model: str = "gemini-3.5-flash",
        max_request_chars: int = 6000,
        context_chars: int = 600,
    ):
        self.client = client
        self.model = model
        self.max_request_chars = max_request_chars
        self.context_chars = context_chars

    def extract(self, chunks: Sequence[Chunk]) -> list[ExtractedFact]:
        # TODO(retain): pack chunks into requests of <= max_request_chars, prepend up to
        # context_chars of preceding episode text, call generate_content with
        # response_mime_type="application/json" and response_json_schema (nullable fields as
        # anyOf [..., {"type": "null"}]), retry 429/RESOURCE_EXHAUSTED with backoff, then parse
        # rows leniently per the class contract.
        raise NotImplementedError("Gemini fact extraction")
