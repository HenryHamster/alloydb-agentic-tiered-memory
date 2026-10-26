"""Live Retain check against AlloyDB and Gemini. Skipped unless RETAIN_LIVE=1.

    RETAIN_LIVE=1 uv run pytest tests/test_retain_live.py -s

Needs the configuration in docs/RETAIN_HANDOFF.md (PROJECT_ID, DB_HOST, DB_USER, ADC) and writes
to a dedicated test identity: RETAIN_TEST_USER_ID and RETAIN_TEST_PROJECT_ID, defaulting to
retain-pm-test / retain-smoke-project. Each run uses a fresh session and episode. Observation
writes change that user's observation history, so they also need RETAIN_LIVE_CONSOLIDATE=1.
"""

import json
import os
import uuid
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from retain.models import (
    Category,
    ClaimType,
    FactType,
    MemoryFact,
    Message,
    Observation,
    Scope,
    Speaker,
    Visibility,
    WriteBatch,
    stable_id,
)
from retain.store import InMemoryRetainStore

live = pytest.mark.skipif(os.getenv("RETAIN_LIVE") != "1", reason="set RETAIN_LIVE=1")
AT = datetime(2026, 10, 25, tzinfo=timezone.utc)


def verify_observations(store, scope, expected):
    """Every expected version is readable as written, over readable facts; returns current IDs."""
    persisted = {item.observation_id: item for item in store.observations_for(scope)}
    missing = {item.observation_id for item in expected} - persisted.keys()
    assert not missing, f"Observation versions not readable after commit: {sorted(missing)}"
    fact_ids = sorted({fact_id for item in expected for fact_id in item.fact_ids})
    facts = {fact.fact_id: fact for fact in store.get_facts(fact_ids)}
    for item in expected:
        stored = persisted[item.observation_id]
        assert stored.same_version(item) and stored.valid_to == item.valid_to
        for fact_id in item.fact_ids:
            fact = facts.get(fact_id)
            assert fact and fact.visible_to(scope) and fact.visible_to(item.scope), fact_id
    return sorted(item.observation_id for item in persisted.values() if item.valid_to is None)


@pytest.fixture(scope="module")
def live_retainer():
    project = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    missing = [name for name in ("DB_HOST", "DB_USER") if not os.getenv(name)]
    missing += [] if project else ["PROJECT_ID"]
    if missing:
        pytest.fail(f"RETAIN_LIVE=1 but missing configuration: {', '.join(missing)}")

    from google import genai

    from db_clients import get_db_connection
    from retain.pipeline import build_retainer

    generation = genai.Client(
        vertexai=True,
        project=project,
        location=os.getenv("GENAI_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION", "us"),
    )
    embedding = genai.Client(
        vertexai=True, project=project, location=os.getenv("REGION", "us-east1")
    )
    retainer = build_retainer(get_db_connection, generation, embedding_client=embedding)
    scope = Scope(
        os.getenv("RETAIN_TEST_USER_ID", "retain-pm-test"),
        os.getenv("RETAIN_TEST_PROJECT_ID", "retain-smoke-project"),
        f"retain-live-{uuid.uuid4().hex[:12]}",
    )
    try:
        yield retainer, scope, get_db_connection
    finally:
        retainer.store.close()
        generation.close()
        embedding.close()


@live
def test_live_round_is_stored_with_scope_evidence_and_skipped_on_replay(live_retainer):
    retainer, scope, connect = live_retainer
    now = datetime.now(timezone.utc)
    episode_id = f"retain-live-{uuid.uuid4()}"
    source = (
        f"Retain validation marker {uuid.uuid4().hex}: this isolated test project uses "
        "PostgreSQL and a request timeout of 43 seconds."
    )
    messages = [
        Message(Speaker.USER, source, now),
        Message(Speaker.ASSISTANT, "Recorded for the Retain validation.", now),
    ]

    result = retainer.retain(scope, messages, episode_id=episode_id)

    assert result.batch is not None and result.batch.facts, "no facts were persisted"
    fact_ids = [fact.fact_id for fact in result.batch.facts]
    document_ids = [document.document_id for document in result.batch.documents]
    persisted = retainer.store.get_facts(fact_ids)
    assert len(persisted) == len(fact_ids)
    for fact in persisted:
        assert fact.scope == scope and fact.source_document_id in document_ids
        assert fact.embedding is not None and len(fact.embedding) == 768

    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT document_id::text, user_id, project_id, session_id, text FROM documents "
                "WHERE document_id = ANY(%s::uuid[]) AND retained_at IS NOT NULL",
                (document_ids,),
            )
            documents = {row[0]: row[1:] for row in cursor.fetchall()}
            cursor.execute(
                "SELECT count(*) FROM memory_entities WHERE fact_id = ANY(%s::uuid[])",
                (fact_ids,),
            )
            entity_rows = cursor.fetchone()[0]
    finally:
        connection.close()
    assert documents.keys() == set(document_ids), "source documents not committed as retained"
    for fact in persisted:
        user_id, project_id, session_id, text = documents[fact.source_document_id]
        assert (user_id, project_id, session_id) == (
            scope.user_id,
            scope.project_id,
            scope.session_id,
        )
        assert fact.evidence in text

    assert retainer.retain(scope, messages, episode_id=episode_id).skipped
    summary = {"facts": len(persisted), "fact_entity_rows": entity_rows, "episode_id": episode_id}
    print(json.dumps({"status": "ok", "scope": scope.__dict__, **summary}, indent=2))


@live
@pytest.mark.skipif(
    os.getenv("RETAIN_LIVE_CONSOLIDATE") != "1", reason="set RETAIN_LIVE_CONSOLIDATE=1"
)
def test_live_observations_are_persisted_as_written(live_retainer):
    retainer, scope, _ = live_retainer
    written = retainer.consolidate(scope)
    current = verify_observations(retainer.store, scope, written)
    print(json.dumps({"observations_written": len(written), "current": current}, indent=2))


def test_verify_observations_detects_unreadable_versions():
    scope = Scope("live-user", "live-project", "live-session")
    store = InMemoryRetainStore()
    fact = MemoryFact(
        fact_id=stable_id("live-fact"),
        scope=scope,
        visibility=Visibility.PROJECT,
        fact_type=FactType.WORLD,
        category=Category.DECISION,
        speaker=Speaker.USER,
        claim_type=ClaimType.ASSERTED,
        text="The service uses PostgreSQL.",
        evidence="We use PostgreSQL.",
        source_document_id=stable_id("live-document"),
        source_episode_id="live-episode",
        recorded_at=AT,
        valid_from=AT,
    )
    store.write(WriteBatch((), facts=(fact,)))
    observation = Observation(
        observation_id=stable_id("live-observation"),
        scope=scope,
        visibility=Visibility.PROJECT,
        text="The project uses PostgreSQL.",
        fact_ids=(fact.fact_id,),
        valid_from=AT,
    )
    store.write_observations((observation,))

    assert verify_observations(store, scope, [observation]) == [observation.observation_id]
    unreadable = replace(observation, observation_id=stable_id("missing"))
    with pytest.raises(AssertionError, match="not readable after commit"):
        verify_observations(store, scope, [unreadable])
