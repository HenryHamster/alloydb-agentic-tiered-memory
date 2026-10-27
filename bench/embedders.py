"""Embedders for benchmark runs, plus an on-disk cache so reruns do not re-embed.

VertexEmbedder calls the same model and dimensions as retain.embeddings.GeminiEmbedder, but also
sets the RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task type, so benchmark vectors are asymmetric where
production's are not.
"""

import hashlib
import math
import re
import sqlite3
import threading
import time
import zlib
from array import array
from collections.abc import Sequence

from retain.embeddings import Embedder
from shared.llm import is_rate_limited

DIMENSIONS = 768


class HashingEmbedder:
    """Deterministic bag of word unigrams and bigrams, hashed into 768 dims and L2-normalised.

    No model and no network: for smoke runs and tests, not for reporting quality numbers.
    """

    def __init__(self, dims: int = DIMENSIONS):
        self.dims = dims

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = re.findall(r"\w+", text.lower())
            vector = [0.0] * self.dims
            for feature in words + [f"{a} {b}" for a, b in zip(words, words[1:])]:
                vector[zlib.crc32(feature.encode()) % self.dims] += 1.0
            norm = math.sqrt(sum(x * x for x in vector)) or 1.0
            vectors.append([x / norm for x in vector])
        return vectors


class VertexEmbedder:
    """text-embedding-005 through google-genai, batched, 429s retried with backoff.

    task_type is "RETRIEVAL_DOCUMENT" for stored memories and "RETRIEVAL_QUERY" for questions.
    """

    def __init__(
        self,
        client,
        *,
        task_type: str,
        model: str = "text-embedding-005",
        batch_size: int = 50,
        attempts: int = 6,
    ):
        self.client = client
        self.task_type = task_type
        self.model = model
        self.batch_size = batch_size
        self.attempts = attempts

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(
            task_type=self.task_type, output_dimensionality=DIMENSIONS
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._embed(batch, config)
            got = [list(e.values) for e in response.embeddings or []]
            if len(got) != len(batch) or any(len(v) != DIMENSIONS for v in got):
                raise ValueError(f"Expected {len(batch)} vectors of {DIMENSIONS} dims")
            vectors += got
        return vectors

    def _embed(self, batch, config):
        for attempt in range(self.attempts):
            try:
                return self.client.models.embed_content(
                    model=self.model, contents=batch, config=config
                )
            except Exception as e:
                if not is_rate_limited(e) or attempt == self.attempts - 1:
                    raise
                time.sleep(min(60, 2 * 2**attempt))
        raise AssertionError("unreachable")


class CachedEmbedder:
    """Wraps an embedder with a SQLite cache keyed by (namespace, text)."""

    def __init__(self, inner: Embedder, path: str, namespace: str):
        self.inner = inner
        self.namespace = namespace
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, data BLOB)")
        self._lock = threading.Lock()

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        found = self._load(set(keys))
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
        if missing:
            fresh = self.inner(missing)
            rows = [(self._key(t), array("f", v).tobytes()) for t, v in zip(missing, fresh)]
            with self._lock:
                self._db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", rows)
                self._db.commit()
            found.update({key: list(vector) for (key, _), vector in zip(rows, fresh)})
        return [found[key] for key in keys]

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.namespace}\x1f{text}".encode()).hexdigest()

    def _load(self, keys: set[str]) -> dict[str, list[float]]:
        found, wanted = {}, list(keys)
        with self._lock:
            for start in range(0, len(wanted), 500):
                chunk = wanted[start : start + 500]
                marks = ",".join("?" * len(chunk))
                for key, data in self._db.execute(
                    f"SELECT key, data FROM vectors WHERE key IN ({marks})", chunk
                ):
                    found[key] = array("f", data).tolist()
        return found
