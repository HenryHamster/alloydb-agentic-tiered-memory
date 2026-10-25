from datetime import datetime, timezone

import pytest

from retain.models import (
    Category,
    ClaimType,
    FactType,
    MemoryFact,
    Observation,
    Scope,
    Speaker,
    Visibility,
    WriteBatch,
    stable_id,
)
from retain.smoke import _verify_observations
from retain.store import InMemoryRetainStore

AT = datetime(2026, 10, 25, tzinfo=timezone.utc)
SCOPE = Scope("smoke-user", "smoke-project", "smoke-session")


def test_smoke_checks_persisted_observation_versions_and_source_fact_scope():
    store = InMemoryRetainStore()
    fact = MemoryFact(
        fact_id=stable_id("smoke-fact"),
        scope=SCOPE,
        visibility=Visibility.PROJECT,
        fact_type=FactType.WORLD,
        category=Category.DECISION,
        speaker=Speaker.USER,
        claim_type=ClaimType.ASSERTED,
        text="The service uses PostgreSQL.",
        evidence="We use PostgreSQL.",
        source_document_id=stable_id("smoke-document"),
        source_episode_id="smoke-episode",
        recorded_at=AT,
        valid_from=AT,
    )
    store.write(WriteBatch((), facts=(fact,)))
    observation = Observation(
        observation_id=stable_id("smoke-observation"),
        scope=SCOPE,
        visibility=Visibility.PROJECT,
        text="The project uses PostgreSQL.",
        fact_ids=(fact.fact_id,),
        valid_from=AT,
    )
    store.write_observations((observation,))

    assert _verify_observations(store, SCOPE, [observation]) == [observation.observation_id]

    missing = Observation(
        observation_id=stable_id("missing-observation"),
        scope=SCOPE,
        visibility=Visibility.PROJECT,
        text=observation.text,
        fact_ids=(fact.fact_id,),
        valid_from=AT,
    )
    with pytest.raises(RuntimeError, match="not readable after commit"):
        _verify_observations(store, SCOPE, [missing])
