"""Consolidate scoped fact groups into observations."""

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Protocol

from retain.models import ClaimType, Observation, Scope, Visibility, stable_id


class ObservationBuilder(Protocol):
    def consolidate(self, reader: Scope) -> list[Observation]:
        """Create, refresh or close observations visible to `reader`; return those written."""
        ...


class GeminiObservationBuilder:
    """Summarize current asserted facts that share an entity and visibility domain."""

    def __init__(self, client, store, embedder, *, model="gemini-3.5-flash", min_facts=3):
        if min_facts < 2:
            raise ValueError("Observations require at least two supporting facts")
        self.client = client
        self.store = store
        self.embedder = embedder
        self.model = model
        self.min_facts = min_facts

    def consolidate(self, reader: Scope) -> list[Observation]:
        fact_reader = getattr(self.store, "facts_for_observations", None)
        observation_reader = getattr(self.store, "observations_for", None)
        if fact_reader is None or observation_reader is None:
            raise RuntimeError("Observation store read methods are unavailable")

        groups = defaultdict(list)
        for fact in fact_reader(reader):
            if (
                fact.is_current
                and fact.visible_to(reader)
                and fact.claim_type == ClaimType.ASSERTED
                and fact.entity_ids
            ):
                for entity_id in set(fact.entity_ids):
                    groups[_group_key(fact.scope, fact.visibility, entity_id)].append(fact)

        now = datetime.now(timezone.utc)
        history = defaultdict(list)
        for observation in observation_reader(reader):
            if observation.entity_id:
                history[
                    _group_key(observation.scope, observation.visibility, observation.entity_id)
                ].append(observation)

        pending = []
        closing = {}
        active_ids = set()
        for (scope, visibility, entity_id), group in sorted(
            groups.items(), key=lambda row: row[0][2]
        ):
            facts = list({fact.fact_id: fact for fact in group}.values())
            if len(facts) < self.min_facts:
                continue
            facts.sort(key=lambda fact: fact.fact_id)
            fact_ids = tuple(fact.fact_id for fact in facts)
            group_key = (scope, visibility, entity_id)
            versions = history.get(group_key, [])
            current = [observation for observation in versions if observation.valid_to is None]
            if len(current) > 1:
                raise ValueError("Observation group has multiple current versions")
            previous = current[0] if current else None
            if previous is not None and previous.fact_ids == fact_ids:
                active_ids.add(group_key)
                continue
            summary = self._summarize(facts)
            if summary is None:
                if previous is not None:
                    closing[previous.observation_id] = _close_version(previous, now)
                continue

            valid_from = _version_start(facts, previous, now)
            observation_id = _next_observation_id(
                scope, visibility, entity_id, versions, previous, fact_ids
            )
            if previous is not None:
                closing[previous.observation_id] = _close_version(previous, valid_from)
            active_ids.add(group_key)
            pending.append(
                (
                    observation_id,
                    scope,
                    visibility,
                    entity_id,
                    facts,
                    fact_ids,
                    summary,
                    valid_from,
                )
            )

        for group_key, versions in history.items():
            if group_key in active_ids:
                continue
            for observation in versions:
                if observation.valid_to is None:
                    closing[observation.observation_id] = _close_version(observation, now)

        vectors = self.embedder([row[-2] for row in pending]) if pending else []
        if len(vectors) != len(pending):
            raise ValueError("Embedder must return one vector per observation")
        written = []
        for row, vector in zip(pending, vectors, strict=True):
            (
                observation_id,
                scope,
                visibility,
                entity_id,
                facts,
                fact_ids,
                summary,
                valid_from,
            ) = row
            written.append(
                Observation(
                    observation_id=observation_id,
                    scope=scope,
                    visibility=visibility,
                    text=summary,
                    fact_ids=fact_ids,
                    valid_from=valid_from,
                    entity_id=entity_id,
                    embedding=tuple(vector),
                )
            )

        written.extend(closing.values())

        if written:
            self.store.write_observations(written)
        return written

    def _summarize(self, facts):
        source = [
            {
                "fact_id": fact.fact_id,
                "text": fact.text,
                "evidence": fact.evidence,
                "category": fact.category.value,
                "speaker": fact.speaker.value,
                "occurred_at": fact.occurred_at.isoformat() if fact.occurred_at else None,
                "valid_from": fact.valid_from.isoformat(),
            }
            for fact in facts
        ]
        prompt = (
            "Write a concise observation supported only by these asserted facts. Preserve "
            "qualifiers, dates, and contradictions; do not infer unstated causes or attributes. "
            "If the facts do not support a useful synthesis, return an empty summary and an "
            "empty fact_ids array. Otherwise include every input fact_id that supports the "
            "summary exactly once. Return only the requested JSON object.\n"
            + json.dumps(source, ensure_ascii=False)
        )
        response = self._generate(prompt)
        if not isinstance(getattr(response, "text", None), str):
            raise ValueError("Gemini returned no observation")
        try:
            row = json.loads(response.text)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Gemini returned malformed observation JSON") from error
        if not isinstance(row, dict) or set(row) != {"summary", "fact_ids"}:
            raise ValueError("Gemini returned invalid observation fields")
        summary, fact_ids = row["summary"], row["fact_ids"]
        expected_ids = {fact.fact_id for fact in facts}
        if (
            not isinstance(summary, str)
            or not isinstance(fact_ids, list)
            or any(not isinstance(fact_id, str) for fact_id in fact_ids)
        ):
            raise ValueError("Gemini returned invalid observation values")
        summary = summary.strip()
        if not summary and not fact_ids:
            return None
        if not summary:
            raise ValueError("Observation summary cannot be empty")
        if len(fact_ids) != len(set(fact_ids)) or set(fact_ids) != expected_ids:
            raise ValueError("Observation provenance must match its supporting facts")
        if any(summary == fact.text.strip() or summary == fact.evidence.strip() for fact in facts):
            return None
        return summary

    def _generate(self, prompt):
        for attempt in range(4):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": _summary_schema(),
                    },
                )
            except Exception as error:
                retryable = "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)
                if not retryable or attempt == 3:
                    raise
                time.sleep(3 * (2**attempt))


def _group_key(scope, visibility, entity_id):
    if visibility == Visibility.GLOBAL:
        scope = Scope(scope.user_id)
    elif visibility == Visibility.PROJECT:
        scope = Scope(scope.user_id, project_id=scope.project_id)
    else:
        scope = Scope(scope.user_id, session_id=scope.session_id)
    return scope, visibility, entity_id


def _observation_id(scope, visibility, entity_id):
    domain = (
        scope.project_id if visibility == Visibility.PROJECT else None,
        scope.session_id if visibility == Visibility.SESSION else None,
    )
    return stable_id(scope.user_id, visibility.value, *domain, entity_id)


def _next_observation_id(scope, visibility, entity_id, history, previous, fact_ids):
    base_id = _observation_id(scope, visibility, entity_id)
    if not history:
        return base_id
    predecessor = (
        previous.observation_id
        if previous
        else tuple(sorted(observation.observation_id for observation in history))
    )
    return stable_id(base_id, "version", predecessor, *fact_ids)


def _version_start(facts, previous, now):
    if previous is None:
        return min(fact.valid_from for fact in facts)
    added = [fact.valid_from for fact in facts if fact.fact_id not in previous.fact_ids]
    return max(previous.valid_from, min(added)) if added else max(now, previous.valid_from)


def _close_version(observation, at):
    return Observation(
        observation_id=observation.observation_id,
        scope=observation.scope,
        visibility=observation.visibility,
        text=observation.text,
        fact_ids=observation.fact_ids,
        valid_from=observation.valid_from,
        entity_id=observation.entity_id,
        valid_to=max(at, observation.valid_from),
        embedding=observation.embedding,
    )


def _summary_schema():
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "fact_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "fact_ids"],
        "additionalProperties": False,
    }
