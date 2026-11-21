"""Speaker names on facts: two people with the same role never merge or replace each other."""

from dataclasses import replace
from unittest.mock import Mock

from retain.alloydb_store import _fact
from retain.linking import GeminiSupersedeClassifier
from retain.models import (
    Category,
    ClaimType,
    EntityResolution,
    ExtractedFact,
    FactType,
    Observation,
    Scope,
    Speaker,
    UpdateAction,
    UpdateDecision,
    Visibility,
    WriteBatch,
    stable_id,
)
from retain.observations import _observation_id
from retain.pipeline import _build_fact, _target_problem
from retain.store import InMemoryRetainStore
from tests.test_retain_alloydb_store import AT as DB_AT
from tests.test_retain_alloydb_store import SCOPE as DB_SCOPE
from tests.test_retain_alloydb_store import database_row, make_document, make_fact, store_for
from tests.test_retain_conversations import MONDAY, document
from tests.test_retain_observations import ENTITY, PROJECT, SESSION, USER, builder, memory_fact
from tests.test_retain_stages import ClassifierStore
from tests.test_retain_stages import fact as stage_fact


def extracted(text="Caroline painted a sunset."):
    return ExtractedFact(
        key=0,
        chunk_id="c",
        text=text,
        evidence="I painted a sunset.",
        fact_type=FactType.WORLD,
        category=Category.EVENT,
        claim_type=ClaimType.ASSERTED,
        visibility=Visibility.SESSION,
    )


def named(number, name):
    return replace(memory_fact(number), speaker_name=name)


def test_a_fact_keeps_the_name_of_the_person_who_said_it():
    named = document("Caroline: I painted a sunset.", speaker=Speaker.USER, name="Caroline")
    agent = document("You painted a sunset.", position=1)

    person, _, _ = _build_fact(extracted(), named, EntityResolution({}), MONDAY)
    unnamed, _, _ = _build_fact(extracted(), agent, EntityResolution({}), MONDAY)

    assert person.speaker_name == "Caroline"
    assert unnamed.speaker_name is None


def test_one_persons_same_day_copy_is_not_a_duplicate_of_another_persons():
    """Both people are users here, so only the name tells their claims apart."""
    caroline = replace(stage_fact("I ran a race.", fact_id="caroline"), speaker_name="Caroline")
    melanie = replace(caroline, fact_id="melanie", speaker_name="Melanie")
    client = Mock()
    classifier = GeminiSupersedeClassifier(client, ClassifierStore(similar=[(caroline, 0.99)]))

    (across,) = classifier.classify([melanie])
    (asked,) = classifier.classify([melanie], updating={melanie.fact_id})
    same = replace(caroline, fact_id="caroline-again")
    (repeat,) = classifier.classify([same])

    assert across.action == UpdateAction.ADD
    assert asked.action == UpdateAction.ADD
    client.models.generate_content.assert_not_called()
    assert repeat.action == UpdateAction.DUPLICATE and repeat.target_ids == (caroline.fact_id,)


def test_retainer_rejects_a_target_said_by_someone_else():
    old = named(1, "Caroline")
    new = replace(old, fact_id="new", speaker_name="Melanie")

    for action in (UpdateAction.DUPLICATE, UpdateAction.SUPERSEDE):
        decision = UpdateDecision(new.fact_id, action, (old.fact_id,))
        assert _target_problem(decision, new, {old.fact_id: old}) is not None
    same = replace(new, speaker_name="Caroline")
    decision = UpdateDecision(same.fact_id, UpdateAction.SUPERSEDE, (old.fact_id,))
    assert _target_problem(decision, same, {old.fact_id: old}) is None


def test_observations_summarize_each_persons_facts_separately():
    store = InMemoryRetainStore()
    facts = (named(1, "Caroline"), named(2, "Caroline"), named(3, "Melanie"), named(4, "Melanie"))
    store.write(WriteBatch((), facts=facts))
    stage, client, _ = builder(store)
    reader = Scope(USER, PROJECT, SESSION)

    written = stage.consolidate(reader)

    by_name = {o.speaker_name: set(o.fact_ids) for o in written}
    assert by_name == {
        "Caroline": {facts[0].fact_id, facts[1].fact_id},
        "Melanie": {facts[2].fact_id, facts[3].fact_id},
    }
    assert all(o.entity_id == ENTITY for o in written)
    prompts = [c.kwargs["contents"] for c in client.models.generate_content.call_args_list]
    assert all('"speaker_name"' in prompt for prompt in prompts)
    # A second pass finds each person's current observation and writes nothing.
    assert stage.consolidate(reader) == []


def test_unnamed_observations_keep_their_ids():
    scope = Scope(USER, PROJECT, SESSION)
    old_id = stable_id(USER, Visibility.PROJECT.value, PROJECT, None, ENTITY)

    assert _observation_id(scope, Visibility.PROJECT, ENTITY) == old_id
    assert _observation_id(scope, Visibility.PROJECT, ENTITY, "Melanie") != old_id


def test_alloydb_writes_and_reads_the_fact_speaker_name():
    doc = make_document()
    fact = make_fact(speaker_name="Melanie")
    source = {
        "document_id": doc.document_id,
        "episode_id": doc.episode_id,
        "user_id": doc.scope.user_id,
        "project_id": doc.scope.project_id,
        "session_id": doc.scope.session_id,
        "speaker": doc.speaker.value,
    }

    def respond(sql, _params):
        if sql.startswith("INSERT INTO documents"):
            return ({"document_id": doc.document_id}, [], 1)
        if sql.startswith("INSERT INTO memory_facts"):
            return ({"fact_id": fact.fact_id}, [], 1)
        if sql.startswith("SELECT document_id::text, episode_id"):
            return (None, [source], 1)
        if "FROM memory_facts f LEFT JOIN memory_entities" in sql:
            return (None, [database_row(fact)], 1)
        return (None, [], 1)

    store, conn = store_for(respond)
    store.write(WriteBatch((doc,), facts=(fact,)))

    insert_sql, params = next(
        (sql, params) for sql, params in conn.calls if sql.startswith("INSERT INTO memory_facts")
    )
    assert "speaker_name)" in insert_sql and params[-1] == "Melanie"
    assert _fact(database_row(fact)).speaker_name == "Melanie"


def test_alloydb_current_observation_check_is_per_person():
    observation = Observation(
        observation_id=stable_id("observation-melanie"),
        scope=DB_SCOPE,
        visibility=Visibility.PROJECT,
        text="Melanie paints.",
        fact_ids=(),
        valid_from=DB_AT,
        embedding=(0.1,) * 768,
        speaker_name="Melanie",
    )
    store, conn = store_for(lambda sql, params: (None, [], 1))

    store.write_observations([observation])

    current_sql, current_params = next(
        (sql, params) for sql, params in conn.calls if "AND valid_to IS NULL FOR UPDATE" in sql
    )
    assert "speaker_name IS NOT DISTINCT FROM" in current_sql
    assert current_params[-1] == "Melanie"
    insert_params = next(
        params for sql, params in conn.calls if sql.startswith("INSERT INTO observations")
    )
    assert insert_params[-1] == "Melanie"
