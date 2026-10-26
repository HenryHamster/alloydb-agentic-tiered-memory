"""Redundant vs conflicting facts: find candidate groups, judge them, apply the verdicts.

Candidates come from Retain's own `similar` links, so Reflect needs no extra vector search.
The judge (LLM) only labels groups; which fact is newer, what gets archived and what is kept
for history is decided by code.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from reflect.generation import StructuredGenerator
from reflect.models import Candidate, ConflictReview, Consolidation, FactState, Judgment
from retain.models import FactEntity, MemoryFact, MemoryLink, Scope, Supersession


@dataclass(frozen=True)
class ConsolidationConfig:
    min_similarity: float = 0.85  # "high embedding similarity"
    conflict_confidence: float = 0.5  # multiplier for both sides of an unresolved conflict
    queue_reviews: bool = True


def domain_key(fact: MemoryFact) -> tuple:
    """Visibility domain (user, visibility, project_id | session_id | None). Facts are only
    merged, superseded or clustered within one domain."""
    # TODO(reflect): project -> project_id, session -> session_id, global -> None.
    raise NotImplementedError("Domain key")


def domain_scope(fact: MemoryFact) -> Scope:
    """Scope that owns the fact's domain: only the IDs a reader must match to see it."""
    # TODO(reflect): Scope(user_id, project_id=...), Scope(user_id, session_id=...) or
    # Scope(user_id).
    raise NotImplementedError("Domain scope")


def find_candidates(
    facts: Mapping[str, MemoryFact],
    states: Mapping[str, FactState],
    links: Sequence[MemoryLink],
    fact_entities: Sequence[FactEntity],
    config: ConsolidationConfig = ConsolidationConfig(),
) -> list[Candidate]:
    """Groups for the judge: connected components over `similar` links >= min_similarity
    between facts that are current, not consolidated, in the same domain, same category (the
    "attribute") and share a subject entity. Pairs already in each other's conflicts_with are
    skipped, so a flagged conflict is judged (and penalized) once."""
    # TODO(reflect): union-find per (domain, category, subject entity).
    raise NotImplementedError("Candidate groups")


# Any callable that labels candidate groups can be injected, including offline fakes.
ConflictJudge = Callable[[Sequence[Candidate], Mapping[str, MemoryFact]], list[Judgment]]


class LLMConflictJudge:
    """One structured-output call per batch of candidates.

    Prompt input per fact: fact_id, text, evidence, speaker, claim_type, occurred_at,
    valid_from, recorded_at, source_episode_id. Output per group: verdict, fact_ids (subset of
    the group, >= 2), merged_text (8-15 words, REDUNDANT only), reason.
    Guidance: REDUNDANT = same claim reworded; STATE_CHANGE = the same attribute moved to a new
    value over time ("works at X" then "works at Y"); CONFLICT = incompatible claims with no
    time order to explain them (different sources, same time); otherwise DISTINCT.
    The model never picks the winner: resolve() orders by valid_from.
    """

    def __init__(self, generate: StructuredGenerator, *, batch_size: int = 20):
        self.generate = generate
        self.batch_size = batch_size

    def __call__(self, candidates, facts):
        # TODO(reflect): build prompt/schema, call self.generate, validate/parse Judgments.
        raise NotImplementedError("LLM conflict judge")


@dataclass(frozen=True)
class Resolution:
    consolidations: tuple[Consolidation, ...] = ()
    supersessions: tuple[Supersession, ...] = ()
    links: tuple[MemoryLink, ...] = ()
    states: tuple[FactState, ...] = ()  # originals archived / conflict markers
    reviews: tuple[ConflictReview, ...] = ()


def resolve(
    judgments: Sequence[Judgment],
    candidates: Sequence[Candidate],
    facts: Mapping[str, MemoryFact],
    states: Mapping[str, FactState],
    fact_entities: Sequence[FactEntity],
    *,
    now: datetime,
    config: ConsolidationConfig = ConsolidationConfig(),
) -> Resolution:
    """Turn verdicts into writes. Each fact is touched by at most one judgment.

    - Validation: every judgment names >= 2 facts inside one candidate group, else ValueError
      (nothing written). STATE_CHANGE whose newest two share valid_from -> CONFLICT.
    - REDUNDANT: consolidate() one merged fact (needs merged_text); its state sums
      access_count, takes max last_accessed_at, best tier, min confidence; memory_entities rows
      of the originals are copied to it. Originals: consolidated_into = merged, tier archived,
      link original -consolidated_into-> merged.
    - STATE_CHANGE: newest valid_from wins. Each older fact: Supersession(old, new,
      new.valid_from) and link new -supersedes-> old. Old rows are kept for history.
    - CONFLICT: keep all; conflicts_with on each side; confidence *= conflict_confidence;
      link lower id -conflicts_with-> higher id; ConflictReview if queue_reviews.
    - DISTINCT: nothing.
    """
    # TODO(reflect): implement the rules above.
    raise NotImplementedError("Resolve judgments")


def consolidate(originals: Sequence[MemoryFact], text: str, *, now: datetime) -> MemoryFact:
    """Merged fact for redundant originals (oldest first).

    fact_id = stable_id("consolidated", *sorted ids); scope, visibility and source from the
    newest; evidence = unique quotes joined; valid_from = earliest; evidence_count = sum;
    entity_ids = union; recorded_at = now; embedding = None (Reflector embeds it).
    """
    # TODO(reflect): build the MemoryFact.
    raise NotImplementedError("Consolidated fact")
