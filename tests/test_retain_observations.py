import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from retain.models import (
    Category,
    ClaimType,
    FactType,
    MemoryFact,
    Scope,
    Speaker,
    Supersession,
    Visibility,
    WriteBatch,
    stable_id,
)
from retain.observations import GeminiObservationBuilder
from retain.store import InMemoryRetainStore

AT = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
USER = "observation-test-user"
ENTITY = stable_id("observation-entity")
PROJECT = "observation-project"
SESSION = "observation-session"


def memory_fact(
    number,
    *,
    scope=Scope(USER, PROJECT, SESSION),
    visibility=Visibility.PROJECT,
    claim_type=ClaimType.ASSERTED,
    entity_ids=(ENTITY,),
    valid_from=AT,
    valid_to=None,
):
    return MemoryFact(
        fact_id=stable_id("observation-fact", number, scope, visibility),
        scope=scope,
        visibility=visibility,
        fact_type=FactType.WORLD,
        category=Category.DECISION,
        speaker=Speaker.USER,
        claim_type=claim_type,
        text=f"The service uses setting {number}.",
        evidence=f"Setting {number} is enabled.",
        source_document_id=stable_id("observation-doc", number),
        source_episode_id=f"episode-{number}",
        recorded_at=AT,
        valid_from=valid_from,
        valid_to=valid_to,
        embedding=(0.25,) * 768,
        entity_ids=entity_ids,
    )


def source_rows(payload):
    return json.loads(payload.rsplit("\n", 1)[-1])


def controlled_client(summary="The service uses a consistent setting.", *, ids=None):
    client = Mock()

    def generate(*, contents, **kwargs):
        facts = source_rows(contents)
        return SimpleNamespace(
            text=json.dumps(
                {
                    "summary": summary,
                    "fact_ids": ids if ids is not None else [f["fact_id"] for f in facts],
                }
            )
        )

    client.models.generate_content.side_effect = generate
    return client


def builder(store, client=None, *, min_facts=2):
    client = client or controlled_client()
    embedder = Mock(side_effect=lambda texts: [[0.1] * 768 for _ in texts])
    return GeminiObservationBuilder(client, store, embedder, min_facts=min_facts), client, embedder


def test_observations_are_grounded_scoped_and_keep_contributing_fact_ids():
    store = InMemoryRetainStore()
    global_facts = [
        memory_fact(i, scope=Scope(USER), visibility=Visibility.GLOBAL) for i in range(2)
    ]
    project_a = [memory_fact(i) for i in range(2, 4)]
    project_b = [
        memory_fact(
            i, scope=Scope(USER, "other-project", "other-session"), visibility=Visibility.PROJECT
        )
        for i in range(4, 6)
    ]
    session_facts = [
        memory_fact(
            i,
            scope=Scope(USER, PROJECT, SESSION),
            visibility=Visibility.SESSION,
        )
        for i in range(6, 8)
    ]
    proposed = memory_fact(8, claim_type=ClaimType.PROPOSED)
    store.write(
        WriteBatch(
            (), facts=tuple((*global_facts, *project_a, *project_b, *session_facts, proposed))
        )
    )
    stage, client, embedder = builder(store)

    written = stage.consolidate(Scope(USER, PROJECT, SESSION))

    assert len(written) == 3
    assert {observation.visibility for observation in written} == {
        Visibility.GLOBAL,
        Visibility.PROJECT,
        Visibility.SESSION,
    }
    assert all(len(observation.fact_ids) == 2 for observation in written)
    assert all(observation.scope.user_id == USER for observation in written)
    project_observation = next(o for o in written if o.visibility == Visibility.PROJECT)
    assert project_observation.scope.project_id == PROJECT
    assert set(project_observation.fact_ids) == {fact.fact_id for fact in project_a}
    assert not {fact.fact_id for fact in project_b} & set(project_observation.fact_ids)
    assert len(client.models.generate_content.call_args_list) == 3
    assert len(embedder.call_args.args[0]) == 3
    assert all(o.observation_id in store.observations for o in written)


def test_subset_citations_retry_once_without_relaxing_provenance():
    store = InMemoryRetainStore()
    facts = [memory_fact(i) for i in range(3)]
    store.write(WriteBatch((), facts=tuple(facts)))
    client = Mock()
    client.models.generate_content.side_effect = [
        SimpleNamespace(
            text=json.dumps({"summary": "A partial summary.", "fact_ids": [facts[0].fact_id]})
        ),
        SimpleNamespace(
            text=json.dumps(
                {
                    "summary": "The service uses all three settings.",
                    "fact_ids": [f.fact_id for f in facts],
                }
            )
        ),
    ]
    stage, _, embedder = builder(store, client)
    (result,) = stage.consolidate(Scope(USER, PROJECT, SESSION))
    assert set(result.fact_ids) == {f.fact_id for f in facts}
    assert client.models.generate_content.call_count == 2
    assert (
        "EVERY input fact_id" in client.models.generate_content.call_args_list[0].kwargs["contents"]
    )
    assert (
        "previous response failed validation"
        in client.models.generate_content.call_args.kwargs["contents"]
    )
    embedder.assert_called_once()


def test_observation_consolidation_is_idempotent_and_closes_stale_groups():
    store = InMemoryRetainStore()
    facts = [memory_fact(i) for i in range(2)]
    store.write(WriteBatch((), facts=tuple(facts)))
    stage, client, embedder = builder(store)

    first = stage.consolidate(Scope(USER, PROJECT, SESSION))
    assert len(first) == 1
    assert stage.consolidate(Scope(USER, PROJECT, SESSION)) == []
    assert client.models.generate_content.call_count == 1
    assert embedder.call_count == 1

    store.facts[facts[0].fact_id] = replace(facts[0], valid_to=AT)
    store.facts[facts[1].fact_id] = replace(facts[1], valid_to=AT)
    closed = stage.consolidate(Scope(USER, PROJECT, SESSION))
    assert len(closed) == 1
    assert closed[0].observation_id == first[0].observation_id
    assert closed[0].valid_to is not None
    assert store.observations[closed[0].observation_id].valid_to == closed[0].valid_to


def test_changed_supporting_facts_create_a_new_version_and_preserve_the_old_one():
    store = InMemoryRetainStore()
    first_facts = [memory_fact(30), memory_fact(31)]
    store.write(WriteBatch((), facts=tuple(first_facts)))
    client = controlled_client()
    client.models.generate_content.side_effect = [
        SimpleNamespace(
            text=json.dumps(
                {
                    "summary": "The service uses the original deployment settings.",
                    "fact_ids": [fact.fact_id for fact in first_facts],
                }
            )
        ),
        SimpleNamespace(
            text=json.dumps(
                {
                    "summary": "The service uses the revised deployment settings.",
                    "fact_ids": [fact.fact_id for fact in (*first_facts, memory_fact(32))],
                }
            )
        ),
    ]
    stage, _, _ = builder(store, client)

    original = stage.consolidate(Scope(USER, PROJECT, SESSION))[0]
    added = memory_fact(32, valid_from=AT.replace(day=26))
    store.write(WriteBatch((), facts=(added,)))
    refreshed = stage.consolidate(Scope(USER, PROJECT, SESSION))

    previous = store.observations[original.observation_id]
    current = next(item for item in refreshed if item.valid_to is None)
    assert current.observation_id != original.observation_id
    assert previous.text == original.text
    assert previous.fact_ids == original.fact_ids
    assert previous.valid_from == original.valid_from
    assert previous.valid_to == added.valid_from
    assert current.text == "The service uses the revised deployment settings."
    assert set(current.fact_ids) == {fact.fact_id for fact in (*first_facts, added)}
    assert len(store.observations) == 2


def test_superseded_supporting_fact_refreshes_only_its_scope():
    store = InMemoryRetainStore()
    project_facts = [memory_fact(40), memory_fact(41)]
    other_scope = Scope(USER, "another-project", "another-session")
    other_facts = [
        memory_fact(42, scope=other_scope),
        memory_fact(43, scope=other_scope),
    ]
    store.write(WriteBatch((), facts=tuple((*project_facts, *other_facts))))
    stage, _, _ = builder(store)
    original = stage.consolidate(Scope(USER, PROJECT, SESSION))[0]
    other_original = stage.consolidate(other_scope)[0]

    replacement = memory_fact(44, valid_from=AT.replace(day=27))
    store.write(
        WriteBatch(
            (),
            facts=(replacement,),
            supersessions=(
                Supersession(project_facts[0].fact_id, replacement.fact_id, replacement.valid_from),
            ),
        )
    )
    refreshed = stage.consolidate(Scope(USER, PROJECT, SESSION))

    current = next(item for item in refreshed if item.valid_to is None)
    assert current.observation_id != original.observation_id
    assert project_facts[0].fact_id not in current.fact_ids
    assert replacement.fact_id in current.fact_ids
    assert store.observations[original.observation_id].valid_to == replacement.valid_from
    assert store.observations[other_original.observation_id].valid_to is None
    assert store.facts[project_facts[0].fact_id].text == "The service uses setting 40."


def test_observations_require_sufficient_asserted_facts_and_skip_exact_fact_copies():
    store = InMemoryRetainStore()
    store.write(
        WriteBatch(
            (),
            facts=(
                memory_fact(1),
                memory_fact(2, claim_type=ClaimType.HYPOTHETICAL),
                memory_fact(3, entity_ids=()),
            ),
        )
    )
    stage, client, _ = builder(store)
    assert stage.consolidate(Scope(USER, PROJECT, SESSION)) == []
    client.models.generate_content.assert_not_called()

    facts = [memory_fact(4), memory_fact(5)]
    store.write(WriteBatch((), facts=tuple(facts)))
    exact_client = controlled_client(summary=facts[0].text)
    exact_stage, _, _ = builder(store, exact_client)
    assert exact_stage.consolidate(Scope(USER, PROJECT, SESSION)) == []
    assert not store.observations


@pytest.mark.parametrize(
    "text,ids,error",
    [
        ("not-json", None, "malformed observation JSON"),
        (json.dumps({"summary": "unsupported", "fact_ids": ["unknown"]}), None, "provenance"),
        (json.dumps({"summary": "summary", "fact_ids": []}), None, "provenance"),
    ],
)
def test_invalid_observation_output_is_rejected_without_writes(text, ids, error):
    facts = [memory_fact(10), memory_fact(11)]
    store = InMemoryRetainStore()
    store.write(WriteBatch((), facts=tuple(facts)))
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text=text)
    stage, _, _ = builder(store, client)

    with pytest.raises(ValueError, match=error):
        stage.consolidate(Scope(USER, PROJECT, SESSION))
    assert not store.observations


def test_empty_observation_can_be_declined_without_persisting_a_placeholder():
    facts = [memory_fact(20), memory_fact(21)]
    store = InMemoryRetainStore()
    store.write(WriteBatch((), facts=tuple(facts)))
    client = controlled_client(summary="", ids=[])
    stage, _, embedder = builder(store, client)

    assert stage.consolidate(Scope(USER, PROJECT, SESSION)) == []
    assert not store.observations
    embedder.assert_not_called()


def test_observations_fail_if_store_cannot_supply_scoped_source_facts():
    stage, _, _ = builder(object())
    with pytest.raises(RuntimeError, match="store read methods"):
        stage.consolidate(Scope(USER, PROJECT, SESSION))
