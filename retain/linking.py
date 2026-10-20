"""Proposal step 3: semantic, temporal and causal links, plus supersession decisions."""

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import Protocol

from retain.models import ExtractedFact, MemoryFact, MemoryLink, Relation, UpdateDecision

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


class Linker(Protocol):
    def link(self, facts: Sequence[MemoryFact]) -> list[MemoryLink]:
        """Semantic and temporal edges from new facts to visible, current stored facts."""
        ...


class StoreLinker:
    """Edge rules (both endpoints same user; the target must be visible to the new fact's scope):

    - semantic: top `semantic_k` by cosine >= `min_similarity`, same category; weight = cosine.
    - temporal: top `temporal_k` facts whose occurred_at is within `temporal_window` of the new
      fact's occurred_at; weight = exp(-|dt| / temporal_tau). Facts without occurred_at get none,
      so statements made in the same conversation are not linked just for sharing a timestamp.
    New facts in the same batch are also candidates for each other.
    """

    def __init__(
        self,
        store,
        *,
        semantic_k: int = 5,
        min_similarity: float = 0.75,
        temporal_k: int = 3,
        temporal_window: timedelta = timedelta(days=3),
        temporal_tau: timedelta = timedelta(days=1),
    ):
        self.store = store  # uses similar_facts and facts_near_time
        self.semantic_k = semantic_k
        self.min_similarity = min_similarity
        self.temporal_k = temporal_k
        self.temporal_window = temporal_window
        self.temporal_tau = temporal_tau

    def link(self, facts: Sequence[MemoryFact]) -> list[MemoryLink]:
        # TODO(retain): implement the class rules.
        raise NotImplementedError("Semantic and temporal linking")


def causal_links(
    extracted: Sequence[ExtractedFact], fact_ids: Mapping[int, str]
) -> list[MemoryLink]:
    """caused_by / evidenced_by edges stated by the extractor, mapped to stored fact IDs.

    `fact_ids` maps extraction key -> stored fact_id (a duplicate maps to the fact it
    reinforced); keys missing from it, and self-references, are skipped.
    """
    links = {}
    for fact in extracted:
        source = fact_ids.get(fact.key)
        for relation, keys in (
            (Relation.CAUSED_BY, fact.caused_by),
            (Relation.EVIDENCED_BY, fact.evidenced_by),
        ):
            for key in keys:
                target = fact_ids.get(key)
                if source and target and source != target:
                    links[(source, target, relation)] = MemoryLink(source, target, relation)
    return list(links.values())


class SupersedeClassifier(Protocol):
    def classify(self, facts: Sequence[MemoryFact]) -> list[UpdateDecision]:
        """One decision per new fact: add, duplicate of one stored fact, or supersede some."""
        ...


class GeminiSupersedeClassifier:
    """LLM decides whether new facts repeat or replace stored ones.

    Candidates per new fact: current facts visible to its scope, same claim type, either
    cosine >= `min_similarity` or sharing a subject entity in the same category; at most
    `candidates_k`. A new fact may supersede only facts whose visibility is no broader than its
    own (a project rule never closes a global preference). Targets outside the candidate set
    are rejected; facts with no candidates skip the LLM and are added.
    """

    def __init__(
        self,
        client,
        store,
        *,
        model: str = "gemini-3.5-flash",
        candidates_k: int = 6,
        min_similarity: float = 0.6,
    ):
        self.client = client
        self.store = store
        self.model = model
        self.candidates_k = candidates_k
        self.min_similarity = min_similarity

    def classify(self, facts: Sequence[MemoryFact]) -> list[UpdateDecision]:
        # TODO(retain): gather candidates, one structured-output call per batch, validate targets.
        raise NotImplementedError("Gemini supersession classification")
