"""Fact and observation embeddings (768 dims, matching VECTOR(768) in the schema)."""

from collections.abc import Callable, Sequence

Embedder = Callable[[Sequence[str]], list[list[float]]]


class GeminiEmbedder:
    """text-embedding-005 (768 dims), batched; failures raise instead of writing zero vectors."""

    def __init__(self, client, *, model: str = "text-embedding-005", batch_size: int = 100):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        # TODO(retain): client.models.embed_content(model=..., contents=list(batch)) per batch,
        # retry 429 with backoff, check one 768-dim vector per input.
        raise NotImplementedError("Gemini embeddings")
