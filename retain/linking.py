"""Proposal step 3: semantic, temporal and causal links, plus supersession decisions."""

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Protocol

from retain.models import (
    ExtractedFact,
    MemoryFact,
    MemoryLink,
    Relation,
    Scope,
    UpdateAction,
    UpdateDecision,
    Visibility,
    is_broader,
)
from shared.llm import is_rate_limited

logger = logging.getLogger(__name__)


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
        if semantic_k < 0 or temporal_k < 0:
            raise ValueError("Link limits must be nonnegative")
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be in [0, 1]")
        if temporal_window < timedelta(0) or temporal_tau <= timedelta(0):
            raise ValueError("Temporal window must be nonnegative and tau must be positive")
        self.store = store  # uses similar_facts and facts_near_time
        self.semantic_k = semantic_k
        self.min_similarity = min_similarity
        self.temporal_k = temporal_k
        self.temporal_window = temporal_window
        self.temporal_tau = temporal_tau

    def link(self, facts: Sequence[MemoryFact]) -> list[MemoryLink]:
        facts = tuple({fact.fact_id: fact for fact in facts}.values())
        if not facts:
            return []

        by_id = {fact.fact_id: fact for fact in facts}
        semantic = {}
        temporal = {}

        for fact in facts:
            if not fact.is_current:
                continue
            if fact.embedding is not None and self.semantic_k:
                candidates = self.store.similar_facts(
                    _reader_for(fact),
                    fact.embedding,
                    limit=self.semantic_k,
                    min_similarity=self.min_similarity,
                    category=fact.category,
                )
                eligible = [
                    (float(score), candidate)
                    for candidate, score in candidates
                    if candidate.fact_id != fact.fact_id
                    and candidate.fact_id not in by_id
                    and candidate.category == fact.category
                    and candidate.is_current
                    and _candidate_visible(fact, candidate)
                    and score >= self.min_similarity
                ]
                eligible.sort(key=lambda row: (-row[0], row[1].fact_id))
                for score, candidate in eligible[: self.semantic_k]:
                    _add_link(semantic, fact.fact_id, candidate.fact_id, Relation.SIMILAR, score)

            if fact.occurred_at is not None and self.temporal_k:
                candidates = self.store.facts_near_time(
                    _reader_for(fact),
                    fact.occurred_at,
                    self.temporal_window,
                    limit=self.temporal_k,
                )
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate.fact_id != fact.fact_id
                    and candidate.fact_id not in by_id
                    and candidate.is_current
                    and candidate.occurred_at is not None
                    and abs(candidate.occurred_at - fact.occurred_at) <= self.temporal_window
                    and _candidate_visible(fact, candidate)
                ]
                eligible.sort(
                    key=lambda candidate: (
                        abs((candidate.occurred_at - fact.occurred_at).total_seconds()),
                        candidate.fact_id,
                    )
                )
                for candidate in eligible[: self.temporal_k]:
                    _add_temporal(temporal, fact, candidate, self.temporal_tau)

        current = [fact for fact in facts if fact.is_current]
        for index, left in enumerate(current):
            for right in current[index + 1 :]:
                if _candidate_visible(left, right) and _candidate_visible(right, left):
                    if left.category == right.category and left.embedding and right.embedding:
                        score = _cosine(left.embedding, right.embedding)
                        if score >= self.min_similarity:
                            source, target = sorted((left.fact_id, right.fact_id))
                            _add_link(semantic, source, target, Relation.SIMILAR, score)
                    if left.occurred_at is not None and right.occurred_at is not None:
                        if abs(left.occurred_at - right.occurred_at) <= self.temporal_window:
                            _add_temporal(temporal, left, right, self.temporal_tau)

        selected_semantic = _top_by_source(semantic, self.semantic_k)
        selected_temporal = _top_by_source(temporal, self.temporal_k)
        return list(
            {
                (link.from_fact_id, link.to_fact_id, link.relation): link
                for link in (*selected_semantic, *selected_temporal)
            }.values()
        )


def _candidate_visible(new, candidate):
    return (
        new.scope.user_id == candidate.scope.user_id
        and candidate.visible_to(new.scope)
        and not is_broader(new.visibility, candidate.visibility)
    )


def _reader_for(fact):
    if fact.visibility == Visibility.GLOBAL:
        return Scope(fact.scope.user_id)
    if fact.visibility == Visibility.PROJECT:
        return Scope(fact.scope.user_id, project_id=fact.scope.project_id)
    return fact.scope


def _cosine(left, right):
    if len(left) != len(right) or not left:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _add_link(links, source, target, relation, weight):
    if source == target:
        return
    key = source, target, relation
    link = MemoryLink(source, target, relation, max(0.0, min(1.0, weight)))
    if key not in links or links[key].weight < link.weight:
        links[key] = link


def _add_temporal(links, left, right, tau):
    if left.occurred_at == right.occurred_at:
        source, target = sorted((left.fact_id, right.fact_id))
    elif left.occurred_at < right.occurred_at:
        source, target = left.fact_id, right.fact_id
    else:
        source, target = right.fact_id, left.fact_id
    delta = abs((left.occurred_at - right.occurred_at).total_seconds())
    _add_link(links, source, target, Relation.NEAR_IN_TIME, math.exp(-delta / tau.total_seconds()))


def _top_by_source(links, limit):
    if limit <= 0:
        return []
    grouped = {}
    for link in links.values():
        grouped.setdefault(link.from_fact_id, []).append(link)
    return [
        link
        for source in sorted(grouped)
        for link in sorted(
            grouped[source],
            key=lambda item: (-item.weight, item.to_fact_id, item.relation.value),
        )[:limit]
    ]


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

    Candidates per new fact: current facts visible to its scope, same speaker and claim type,
    either cosine >= `min_similarity` or sharing an entity in the same category; at most
    `candidates_k`. A new fact may supersede only facts whose visibility is no broader than its
    own (a project rule never closes a global preference); Retainer re-checks both rules.
    Invalid or missing decisions leave that fact as add; facts with no candidates skip the LLM.
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
        if candidates_k < 0:
            raise ValueError("candidates_k must be nonnegative")
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be in [0, 1]")
        self.client = client
        self.store = store
        self.model = model
        self.candidates_k = candidates_k
        self.min_similarity = min_similarity

    def classify(self, facts: Sequence[MemoryFact]) -> list[UpdateDecision]:
        facts = tuple(facts)
        if not facts:
            return []
        if len({fact.fact_id for fact in facts}) != len(facts):
            raise ValueError("Classifier input fact IDs must be unique")
        if self.candidates_k <= 0:
            return [UpdateDecision(fact.fact_id) for fact in facts]

        candidate_map = {fact.fact_id: self._candidates(fact) for fact in facts}
        if not any(candidate_map.values()):
            return [UpdateDecision(fact.fact_id) for fact in facts]

        payload = [
            {
                "fact_id": fact.fact_id,
                "text": fact.text,
                "evidence": fact.evidence,
                "category": fact.category.value,
                "claim_type": fact.claim_type.value,
                "speaker": fact.speaker.value,
                "visibility": fact.visibility.value,
                "occurred_at": fact.occurred_at.isoformat() if fact.occurred_at else None,
                "valid_from": fact.valid_from.isoformat(),
                "candidates": [
                    {
                        "fact_id": candidate.fact_id,
                        "text": candidate.text,
                        "evidence": candidate.evidence,
                        "category": candidate.category.value,
                        "claim_type": candidate.claim_type.value,
                        "speaker": candidate.speaker.value,
                        "visibility": candidate.visibility.value,
                        "occurred_at": candidate.occurred_at.isoformat()
                        if candidate.occurred_at
                        else None,
                        "valid_from": candidate.valid_from.isoformat(),
                    }
                    for candidate in candidate_map[fact.fact_id]
                ],
            }
            for fact in facts
        ]
        prompt = (
            "Classify each new memory against only its listed candidates. Return one decision "
            "per input fact. Use duplicate only when the meaning is materially the same; use "
            "supersede only when the evidence explicitly corrects or replaces a current fact. "
            "Keep contradictions as separate facts unless the source establishes a correction. "
            "Historical events are not superseded merely because a later event occurred. "
            "When uncertain, choose add. Add has no target IDs; duplicate has exactly one; "
            "supersede has one or more listed target IDs. Never invent IDs.\nINPUT:\n"
            + json.dumps(payload, ensure_ascii=False)
        )
        rows = _decision_rows(self._generate(prompt))
        new_by_id = {fact.fact_id: fact for fact in facts}
        decisions = {}
        for row in rows:
            try:
                decision = _parse_decision(row, new_by_id, candidate_map, decisions)
            except (TypeError, ValueError) as error:
                # One bad row only costs that fact its update: it is kept as new.
                logger.warning("Ignoring invalid supersession decision: %s", error)
                continue
            decisions[decision.fact_id] = decision
        return [decisions.get(fact.fact_id, UpdateDecision(fact.fact_id)) for fact in facts]

    def _candidates(self, fact):
        candidates = {}
        reader = _reader_for(fact)
        if fact.embedding is not None:
            for candidate, score in self.store.similar_facts(
                reader,
                fact.embedding,
                limit=max(self.candidates_k * 3, self.candidates_k),
                min_similarity=self.min_similarity,
                category=fact.category,
            ):
                if self._eligible(fact, candidate) and score >= self.min_similarity:
                    candidates[candidate.fact_id] = (float(score), candidate)
        if fact.entity_ids:
            for candidate in self.store.facts_with_entities(
                reader, fact.entity_ids, limit=max(self.candidates_k * 3, self.candidates_k)
            ):
                if self._eligible(fact, candidate) and set(fact.entity_ids) & set(
                    candidate.entity_ids
                ):
                    score = (
                        _cosine(fact.embedding, candidate.embedding)
                        if fact.embedding is not None and candidate.embedding is not None
                        else 0.0
                    )
                    previous = candidates.get(candidate.fact_id)
                    if previous is None or score > previous[0]:
                        candidates[candidate.fact_id] = (score, candidate)
        ranked = sorted(candidates.values(), key=lambda row: (-row[0], row[1].fact_id))
        return [candidate for _, candidate in ranked[: self.candidates_k]]

    @staticmethod
    def _eligible(fact, candidate):
        return (
            candidate.fact_id != fact.fact_id
            and candidate.is_current
            and candidate.claim_type == fact.claim_type
            and candidate.category == fact.category
            and candidate.speaker == fact.speaker
            and _candidate_visible(fact, candidate)
        )

    def _generate(self, prompt):
        for attempt in range(4):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": _decision_schema(),
                    },
                )
            except Exception as error:
                if not is_rate_limited(error) or attempt == 3:
                    raise
                time.sleep(3 * (2**attempt))


def _decision_rows(response):
    """Decision rows from a response; a malformed response yields none (every fact is added)."""
    try:
        rows = json.loads(response.text)
    except (AttributeError, TypeError, json.JSONDecodeError):
        logger.warning("Gemini returned no valid supersession JSON; adding every fact")
        return []
    if not isinstance(rows, list):
        logger.warning("Gemini supersession response is not an array; adding every fact")
        return []
    return rows


def _parse_decision(row, new_by_id, candidate_map, seen):
    if not isinstance(row, dict) or set(row) != {"fact_id", "action", "target_ids"}:
        raise ValueError("invalid supersession fields")
    fact_id, target_ids = row["fact_id"], row["target_ids"]
    if not isinstance(fact_id, str) or fact_id not in new_by_id or fact_id in seen:
        raise ValueError("unknown or repeated fact ID")
    if not isinstance(target_ids, list) or any(not isinstance(t, str) for t in target_ids):
        raise ValueError("target_ids must be an array of fact IDs")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("duplicate target IDs")
    decision = UpdateDecision(fact_id, UpdateAction(row["action"]), tuple(target_ids))
    allowed = {candidate.fact_id: candidate for candidate in candidate_map[fact_id]}
    if any(target not in allowed for target in target_ids):
        raise ValueError("target outside the candidate set")
    if decision.action == UpdateAction.SUPERSEDE and any(
        is_broader(allowed[target].visibility, new_by_id[fact_id].visibility)
        for target in target_ids
    ):
        raise ValueError("attempted to supersede a broader-scope fact")
    return decision


def _decision_schema():
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "fact_id": {"type": "string"},
                "action": {"type": "string", "enum": [action.value for action in UpdateAction]},
                "target_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["fact_id", "action", "target_ids"],
            "additionalProperties": False,
        },
    }
