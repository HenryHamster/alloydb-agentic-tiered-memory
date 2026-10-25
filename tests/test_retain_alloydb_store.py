from datetime import datetime, timedelta, timezone

import pytest

from retain.alloydb_store import AlloyDBRetainStore, _fact
from retain.models import (
    Category,
    ClaimType,
    Document,
    Entity,
    EntityType,
    FactType,
    MemoryFact,
    MemoryLink,
    Observation,
    Relation,
    Scope,
    Speaker,
    Supersession,
    Visibility,
    WriteBatch,
    stable_id,
)

AT = datetime(2026, 10, 1, tzinfo=timezone.utc)
SCOPE = Scope("user-a", "project-a", "session-a")


def make_document(text="I use Python.", scope=SCOPE, episode="episode-a", position=0):
    return Document(
        stable_id(episode, position),
        episode,
        position,
        scope,
        Speaker.USER,
        text,
        AT,
        AT + timedelta(minutes=1),
    )


def make_fact(text="User uses Python.", scope=SCOPE, visibility=Visibility.PROJECT, **kwargs):
    fields = {
        "fact_id": stable_id(text, scope.user_id, scope.project_id),
        "scope": scope,
        "visibility": visibility,
        "fact_type": FactType.WORLD,
        "category": Category.PREFERENCE,
        "speaker": Speaker.USER,
        "claim_type": ClaimType.ASSERTED,
        "text": text,
        "evidence": "I use Python.",
        "source_document_id": make_document(scope=scope).document_id,
        "source_episode_id": "episode-a",
        "recorded_at": AT + timedelta(minutes=1),
        "valid_from": AT,
        "details": {"who": "user", "what": "Python"},
        "embedding": (0.25,) * 768,
    }
    return MemoryFact(**(fields | kwargs))


def database_row(fact):
    return {
        "fact_id": fact.fact_id,
        "user_id": fact.scope.user_id,
        "project_id": fact.scope.project_id,
        "session_id": fact.scope.session_id,
        "visibility": fact.visibility.value,
        "fact_type": fact.fact_type.value,
        "category": fact.category.value,
        "speaker": fact.speaker.value,
        "claim_type": fact.claim_type.value,
        "text": fact.text,
        "evidence": fact.evidence,
        "source_document_id": fact.source_document_id,
        "source_episode_id": fact.source_episode_id,
        "recorded_at": fact.recorded_at,
        "valid_from": fact.valid_from,
        "details": fact.details,
        "occurred_at": fact.occurred_at,
        "valid_to": fact.valid_to,
        "superseded_by": fact.superseded_by,
        "evidence_count": fact.evidence_count,
        "embedding": "[" + ",".join(map(str, fact.embedding)) + "]",
        "entity_ids": list(fact.entity_ids),
    }


class Cursor:
    def __init__(self, connection):
        self.connection = connection
        self.one = None
        self.many = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.connection.calls.append((normalized, params))
        self.one, self.many, self.rowcount = self.connection.respond(normalized, params)

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


class Connection:
    closed = False

    def __init__(self, respond):
        self.respond = respond
        self.calls = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, **_):
        return Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def store_for(respond):
    connection = Connection(respond)
    return AlloyDBRetainStore(lambda: connection), connection


def test_write_maps_document_scope_source_evidence_and_embedding():
    doc = make_document()
    fact = make_fact()

    def respond(sql, _params):
        if sql.startswith("INSERT INTO documents"):
            return ({"document_id": doc.document_id}, [], 1)
        if sql.startswith("INSERT INTO memory_facts"):
            return ({"fact_id": fact.fact_id}, [], 1)
        if sql.startswith("SELECT document_id::text, episode_id"):
            return (
                None,
                [
                    {
                        "document_id": doc.document_id,
                        "episode_id": doc.episode_id,
                        "user_id": doc.scope.user_id,
                        "project_id": doc.scope.project_id,
                        "session_id": doc.scope.session_id,
                        "speaker": doc.speaker.value,
                    }
                ],
                1,
            )
        if "FROM memory_facts f LEFT JOIN memory_entities" in sql:
            return (None, [database_row(fact)], 1)
        if sql.startswith("SELECT entity_id::text"):
            return (None, [], 0)
        return (None, [], 1)

    store, conn = store_for(respond)
    store.write(WriteBatch((doc,), facts=(fact,)))
    document_params = next(
        params for sql, params in conn.calls if sql.startswith("INSERT INTO documents")
    )
    fact_params = next(
        params for sql, params in conn.calls if sql.startswith("INSERT INTO memory_facts")
    )
    assert document_params[3:9] == (
        SCOPE.user_id,
        SCOPE.project_id,
        SCOPE.session_id,
        "user",
        doc.text,
        AT,
    )
    assert fact_params[1:11] == (
        SCOPE.user_id,
        SCOPE.project_id,
        SCOPE.session_id,
        "project",
        "world",
        "preference",
        "user",
        "asserted",
        fact.text,
        fact.evidence,
    )
    assert fact_params[12:14] == (doc.document_id, "episode-a")
    assert fact_params[14].startswith("[") and len(fact_params[14][1:-1].split(",")) == 768
    assert conn.commits == 1 and conn.rollbacks == 0


def test_document_replay_is_idempotent_and_changed_source_is_rejected():
    original = make_document()

    def respond(sql, _params):
        if sql.startswith("INSERT INTO documents"):
            return (None, [], 0)
        if sql.startswith("SELECT document_id::text, episode_id"):
            return (
                {
                    "document_id": original.document_id,
                    "episode_id": original.episode_id,
                    "position": original.position,
                    "user_id": original.scope.user_id,
                    "project_id": original.scope.project_id,
                    "session_id": original.scope.session_id,
                    "speaker": original.speaker.value,
                    "text": original.text,
                    "occurred_at": original.occurred_at,
                    "recorded_at": original.recorded_at,
                    "expires_at": original.expires_at,
                    "retained_at": AT,
                },
                [],
                1,
            )
        raise AssertionError(sql)

    store, conn = store_for(respond)
    store.write(WriteBatch((original,), facts=(make_fact(),)))
    assert len(conn.calls) == 2
    assert conn.commits == 1

    changed = make_document(text="I use Go.")
    with pytest.raises(ValueError, match="Conflicting replay"):
        store.write(WriteBatch((changed,)))
    assert conn.rollbacks == 1


def test_supersession_updates_historical_row_inside_batch_transaction():
    doc = make_document(text="I now use Go.", episode="episode-b")
    old = make_fact(valid_from=AT + timedelta(days=1))
    new = make_fact(
        text="User now uses Go.",
        fact_id=stable_id("new fact"),
        source_document_id=doc.document_id,
        source_episode_id=doc.episode_id,
        valid_from=AT,
        evidence="I now use Go.",
    )

    def respond(sql, _params):
        if sql.startswith("INSERT INTO documents"):
            return ({"document_id": doc.document_id}, [], 1)
        if sql.startswith("INSERT INTO memory_facts"):
            return ({"fact_id": new.fact_id}, [], 1)
        if sql.startswith("SELECT document_id::text, episode_id"):
            return (
                None,
                [
                    {
                        "document_id": doc.document_id,
                        "episode_id": doc.episode_id,
                        "user_id": doc.scope.user_id,
                        "project_id": doc.scope.project_id,
                        "session_id": doc.scope.session_id,
                        "speaker": doc.speaker.value,
                    }
                ],
                1,
            )
        if "FROM memory_facts f LEFT JOIN memory_entities" in sql:
            return (None, [database_row(old), database_row(new)], 2)
        if sql.startswith("UPDATE memory_facts SET valid_to"):
            return (None, [], 1)
        return (None, [], 1)

    store, conn = store_for(respond)
    store.write(
        WriteBatch(
            (doc,),
            facts=(new,),
            supersessions=(Supersession(old.fact_id, new.fact_id, AT),),
        )
    )
    update = next(
        (sql, params)
        for sql, params in conn.calls
        if sql.startswith("UPDATE memory_facts SET valid_to")
    )
    assert update[1] == (old.valid_from, new.fact_id, old.fact_id)
    assert conn.commits == 1


def test_narrower_fact_cannot_supersede_global_and_cross_project_links_rollback():
    doc = make_document()
    project_fact = make_fact()
    global_fact = make_fact(
        text="User uses Python everywhere.",
        visibility=Visibility.GLOBAL,
        scope=Scope(SCOPE.user_id),
        fact_id=stable_id("global"),
    )

    def rows_respond(sql, _params):
        if sql.startswith("INSERT INTO documents"):
            return ({"document_id": doc.document_id}, [], 1)
        if sql.startswith("INSERT INTO memory_facts"):
            return ({"fact_id": project_fact.fact_id}, [], 1)
        if sql.startswith("SELECT document_id::text, episode_id"):
            return (
                None,
                [
                    {
                        "document_id": doc.document_id,
                        "episode_id": doc.episode_id,
                        "user_id": doc.scope.user_id,
                        "project_id": doc.scope.project_id,
                        "session_id": doc.scope.session_id,
                        "speaker": doc.speaker.value,
                    }
                ],
                1,
            )
        if "FROM memory_facts f LEFT JOIN memory_entities" in sql:
            return (None, [database_row(project_fact), database_row(global_fact)], 2)
        return (None, [], 1)

    store, conn = store_for(rows_respond)
    with pytest.raises(ValueError, match="narrower fact cannot supersede"):
        store.write(
            WriteBatch(
                (doc,),
                facts=(project_fact,),
                supersessions=(Supersession(global_fact.fact_id, project_fact.fact_id, AT),),
            )
        )
    assert conn.commits == 0 and conn.rollbacks == 1

    other_project = make_fact(
        text="Different project uses Rust.",
        scope=Scope(SCOPE.user_id, "project-b", "session-b"),
        fact_id=stable_id("project b"),
    )
    store, conn = store_for(
        lambda sql, _params: (
            ({"document_id": doc.document_id}, [], 1)
            if sql.startswith("INSERT INTO documents")
            else ({"fact_id": project_fact.fact_id}, [], 1)
            if sql.startswith("INSERT INTO memory_facts")
            else (
                None,
                [
                    {
                        "document_id": doc.document_id,
                        "episode_id": doc.episode_id,
                        "user_id": doc.scope.user_id,
                        "project_id": doc.scope.project_id,
                        "session_id": doc.scope.session_id,
                        "speaker": doc.speaker.value,
                    }
                ],
                1,
            )
            if sql.startswith("SELECT document_id::text, episode_id")
            else (None, [database_row(project_fact), database_row(other_project)], 2)
            if "FROM memory_facts f LEFT JOIN memory_entities" in sql
            else (None, [], 1)
        )
    )
    with pytest.raises(ValueError, match="visibility domain"):
        store.write(
            WriteBatch(
                (doc,),
                facts=(project_fact,),
                links=(MemoryLink(project_fact.fact_id, other_project.fact_id, Relation.SIMILAR),),
            )
        )
    assert conn.commits == 0 and conn.rollbacks == 1


def test_database_fact_mapping_keeps_evidence_provenance_and_scope():
    source = make_fact()
    mapped = _fact(database_row(source))
    assert mapped.evidence == source.evidence
    assert mapped.source_document_id == source.source_document_id
    assert mapped.source_episode_id == source.source_episode_id
    assert mapped.scope == source.scope
    assert mapped.visibility == source.visibility
    assert mapped.embedding == source.embedding


def test_database_error_after_partial_batch_rolls_back_transaction():
    doc = make_document()
    fact = make_fact()
    other = make_fact(text="User uses another tool.", fact_id=stable_id("other"))

    def respond(sql, _params):
        if sql.startswith("INSERT INTO documents"):
            return ({"document_id": doc.document_id}, [], 1)
        if sql.startswith("INSERT INTO memory_facts"):
            return ({"fact_id": fact.fact_id}, [], 1)
        if sql.startswith("SELECT document_id::text, episode_id"):
            return (
                None,
                [
                    {
                        "document_id": doc.document_id,
                        "episode_id": doc.episode_id,
                        "user_id": doc.scope.user_id,
                        "project_id": doc.scope.project_id,
                        "session_id": doc.scope.session_id,
                        "speaker": doc.speaker.value,
                    }
                ],
                1,
            )
        if "FROM memory_facts f LEFT JOIN memory_entities" in sql:
            return (None, [database_row(fact), database_row(other)], 2)
        if sql.startswith("INSERT INTO memory_links"):
            raise RuntimeError("simulated database constraint failure")
        return (None, [], 1)

    store, conn = store_for(respond)
    with pytest.raises(RuntimeError, match="constraint failure"):
        store.write(
            WriteBatch(
                (doc,),
                facts=(fact,),
                links=(MemoryLink(fact.fact_id, other.fact_id, Relation.SIMILAR),),
            )
        )
    assert any(sql.startswith("INSERT INTO memory_facts") for sql, _ in conn.calls)
    assert conn.commits == 0 and conn.rollbacks == 1


def test_read_methods_map_rows_and_apply_visibility_filters():
    fact = make_fact()
    entity = Entity(stable_id("entity"), SCOPE.user_id, "Python", "python", EntityType.TECHNOLOGY)
    fact_row = database_row(fact)

    def respond(sql, _params):
        if sql.startswith("SELECT document_id::text FROM documents"):
            return (None, [{"document_id": fact.source_document_id}], 1)
        if sql.startswith("SELECT entity_id::text, user_id, name"):
            return (
                None,
                [
                    {
                        "entity_id": entity.entity_id,
                        "user_id": entity.user_id,
                        "name": entity.name,
                        "canonical_name": entity.canonical_name,
                        "entity_type": entity.entity_type.value,
                        "aliases": [],
                    }
                ],
                1,
            )
        if "FROM memory_facts f" in sql:
            row = fact_row | ({"similarity": 0.9} if "AS similarity" in sql else {})
            return (None, [row], 1)
        return (None, [], 0)

    store, conn = store_for(respond)
    assert store.retained_document_ids([fact.source_document_id]) == {fact.source_document_id}
    assert store.find_entities(SCOPE.user_id, ["python"])[0].entity_id == entity.entity_id
    similar = store.similar_facts(SCOPE, fact.embedding, limit=3)
    assert similar[0][0].fact_id == fact.fact_id and similar[0][1] == 0.9
    assert store.facts_near_time(SCOPE, AT, timedelta(days=1), limit=3)[0].fact_id == fact.fact_id
    assert store.get_facts([fact.fact_id])[0].evidence == fact.evidence
    assert store.facts_with_entities(SCOPE, [entity.entity_id], limit=3)[0].fact_id == fact.fact_id
    scoped_queries = [sql for sql, _ in conn.calls if "f.user_id = %(user_id)s" in sql]
    assert len(scoped_queries) == 3
    assert all("f.user_id = %(user_id)s" in sql for sql in scoped_queries)
    assert all("f.visibility = 'global'" in sql for sql in scoped_queries)


def test_observation_reads_map_scoped_facts_and_existing_observations():
    entity_id = stable_id("observation-entity")
    fact = make_fact(entity_ids=(entity_id,))
    observation = Observation(
        observation_id=stable_id("observation"),
        scope=SCOPE,
        visibility=Visibility.PROJECT,
        text="User uses Python for project work.",
        fact_ids=(fact.fact_id,),
        valid_from=AT,
        entity_id=entity_id,
        embedding=(0.5,) * 768,
    )
    observation_row = {
        "observation_id": observation.observation_id,
        "user_id": SCOPE.user_id,
        "project_id": SCOPE.project_id,
        "session_id": SCOPE.session_id,
        "visibility": "project",
        "entity_id": entity_id,
        "text": observation.text,
        "fact_ids": [fact.fact_id],
        "valid_from": AT,
        "valid_to": None,
        "embedding": "[" + ",".join(map(str, observation.embedding)) + "]",
    }

    def respond(sql, _params):
        if "FROM memory_facts f JOIN memory_entities me" in sql:
            return (None, [database_row(fact)], 1)
        if "FROM observations o" in sql:
            return (None, [observation_row], 1)
        return (None, [], 0)

    store, conn = store_for(respond)
    assert store.facts_for_observations(SCOPE) == [fact]
    assert store.observations_for(SCOPE) == [observation]
    queries = [sql for sql, _ in conn.calls]
    assert "f.user_id = %(user_id)s" in queries[0]
    assert "f.visibility = 'global'" in queries[0]
    assert "o.user_id = %(user_id)s" in queries[1]
    assert "o.visibility = 'project' AND o.project_id = %(project_id)s" in queries[1]


def test_observation_version_close_and_insert_roll_back_as_one_transaction():
    old = Observation(
        observation_id=stable_id("observation-version-old"),
        scope=SCOPE,
        visibility=Visibility.PROJECT,
        text="The project uses the original settings.",
        fact_ids=(),
        valid_from=AT,
        embedding=(0.1,) * 768,
    )
    closed = Observation(
        observation_id=old.observation_id,
        scope=old.scope,
        visibility=old.visibility,
        text=old.text,
        fact_ids=old.fact_ids,
        valid_from=old.valid_from,
        valid_to=AT + timedelta(days=1),
        embedding=old.embedding,
    )
    new = Observation(
        observation_id=stable_id("observation-version-new"),
        scope=SCOPE,
        visibility=Visibility.PROJECT,
        text="The project uses revised settings.",
        fact_ids=(),
        valid_from=AT + timedelta(days=1),
        embedding=(0.2,) * 768,
    )
    row = {
        "observation_id": old.observation_id,
        "user_id": SCOPE.user_id,
        "project_id": SCOPE.project_id,
        "session_id": SCOPE.session_id,
        "visibility": old.visibility.value,
        "entity_id": None,
        "text": old.text,
        "fact_ids": [],
        "valid_from": old.valid_from,
        "valid_to": None,
        "embedding": "[" + ",".join(map(str, old.embedding)) + "]",
    }

    def respond(sql, _params):
        if "WHERE observation_id = %s::uuid FOR UPDATE" in sql:
            return (row, [], 1)
        if "AND valid_to IS NULL FOR UPDATE" in sql:
            return (None, [], 0)
        if sql.startswith("UPDATE observations SET valid_to"):
            return (None, [], 1)
        if sql.startswith("INSERT INTO observations"):
            raise RuntimeError("simulated new-version insert failure")
        return (None, [], 1)

    store, connection = store_for(respond)
    with pytest.raises(RuntimeError, match="new-version insert failure"):
        store.write_observations((new, closed))

    statements = [sql for sql, _ in connection.calls]
    assert any(sql.startswith("UPDATE observations SET valid_to") for sql in statements)
    assert any(sql.startswith("INSERT INTO observations") for sql in statements)
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_stale_observation_refresh_cannot_create_a_second_current_version():
    old = Observation(
        observation_id=stable_id("observation-stale-old"),
        scope=SCOPE,
        visibility=Visibility.PROJECT,
        text="Original project settings.",
        fact_ids=(),
        valid_from=AT,
        valid_to=AT + timedelta(days=1),
        embedding=(0.1,) * 768,
    )
    stale_close = Observation(
        observation_id=old.observation_id,
        scope=old.scope,
        visibility=old.visibility,
        text=old.text,
        fact_ids=old.fact_ids,
        valid_from=old.valid_from,
        valid_to=AT + timedelta(days=2),
        embedding=old.embedding,
    )
    winner_id = stable_id("observation-stale-winner")

    def respond(sql, _params):
        if "WHERE observation_id = %s::uuid FOR UPDATE" in sql:
            row = {
                "observation_id": old.observation_id,
                "user_id": SCOPE.user_id,
                "project_id": SCOPE.project_id,
                "session_id": SCOPE.session_id,
                "visibility": old.visibility.value,
                "entity_id": None,
                "text": old.text,
                "fact_ids": [],
                "valid_from": old.valid_from,
                "valid_to": old.valid_to,
                "embedding": "[" + ",".join(map(str, old.embedding)) + "]",
            }
            return (row, [], 1)
        if "AND valid_to IS NULL FOR UPDATE" in sql:
            return (None, [{"observation_id": winner_id}], 1)
        return (None, [], 1)

    store, connection = store_for(respond)
    stale_new = Observation(
        observation_id=stable_id("observation-stale-loser"),
        scope=SCOPE,
        visibility=Visibility.PROJECT,
        text="Stale project settings.",
        fact_ids=(),
        valid_from=AT + timedelta(days=2),
        embedding=(0.2,) * 768,
    )
    with pytest.raises(ValueError, match="another current version"):
        store.write_observations((stale_new, stale_close))
    assert connection.commits == 0
    assert connection.rollbacks == 1
