"""Fact and observation embeddings for the schema's 768-dimensional vectors."""

import math
import time
from collections.abc import Callable, Sequence

Embedder = Callable[[Sequence[str]], list[list[float]]]
EMBEDDING_DIMENSIONS = 768


class GeminiEmbedder:
    """Call text-embedding-005 in bounded batches and validate each returned vector."""

    def __init__(self, client, *, model: str = "text-embedding-005", batch_size: int = 100):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        texts = tuple(texts)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be nonempty strings")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._embed(batch)
            embeddings = getattr(response, "embeddings", None)
            if not isinstance(embeddings, (list, tuple)) or len(embeddings) != len(batch):
                raise ValueError("Gemini must return one embedding per input")
            vectors.extend(_validate_vector(getattr(item, "values", None)) for item in embeddings)
        return vectors

    def _embed(self, batch):
        for attempt in range(4):
            try:
                return self.client.models.embed_content(model=self.model, contents=list(batch))
            except Exception as error:
                retryable = "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)
                if not retryable or attempt == 3:
                    raise
                time.sleep(3 * (2**attempt))


def _validate_vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Gemini embeddings must have {EMBEDDING_DIMENSIONS} dimensions")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Gemini embeddings must contain finite numbers")
    vector = [float(value) for value in values]
    if not any(vector):
        raise ValueError("Gemini embeddings must be nonzero")
    return vector
