"""Live Gemini and AlloyDB Retain validation; run with `python -m retain.smoke`."""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description="Retain a test conversation in AlloyDB")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--session-id", default=f"retain-smoke-{uuid.uuid4().hex[:12]}")
    parser.add_argument("--consolidate", action="store_true")
    return parser.parse_args(argv)


def _verify_observations(store, scope, expected):
    persisted = {item.observation_id: item for item in store.observations_for(scope)}
    missing = {item.observation_id for item in expected} - persisted.keys()
    if missing:
        raise RuntimeError(
            "Observation versions were not readable after commit: " + ", ".join(sorted(missing))
        )

    fact_ids = {fact_id for item in expected for fact_id in item.fact_ids}
    facts = {fact.fact_id: fact for fact in store.get_facts(sorted(fact_ids))}
    for item in expected:
        stored = persisted[item.observation_id]
        if (
            stored.text != item.text
            or stored.scope != item.scope
            or stored.visibility != item.visibility
            or stored.fact_ids != item.fact_ids
            or stored.valid_from != item.valid_from
            or stored.valid_to != item.valid_to
        ):
            raise RuntimeError(
                f"Persisted observation version did not match: {item.observation_id}"
            )
        for fact_id in item.fact_ids:
            fact = facts.get(fact_id)
            if fact is None or not fact.visible_to(scope) or not fact.visible_to(item.scope):
                raise RuntimeError(
                    f"Observation {item.observation_id} references a missing or out-of-scope fact"
                )

    return sorted(item.observation_id for item in persisted.values() if item.valid_to is None)


def main(argv=None):
    args = _arguments(argv)
    project = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    db_required = ("DB_HOST", "DB_USER")
    missing = [name for name in db_required if not os.getenv(name)]
    if not project:
        missing.append("PROJECT_ID")
    if missing:
        raise RuntimeError("Missing required configuration: " + ", ".join(missing))

    from google import genai

    from db_clients import get_db_connection
    from retain.models import Message, Scope, Speaker
    from retain.pipeline import build_retainer

    generation_location = os.getenv("GENAI_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION", "us")
    embedding_location = os.getenv("REGION", "us-east1")
    generation_client = genai.Client(vertexai=True, project=project, location=generation_location)
    embedding_client = genai.Client(vertexai=True, project=project, location=embedding_location)
    scope = Scope(args.user_id, args.project_id, args.session_id)
    timestamp = datetime.now(timezone.utc)
    episode_id = f"retain-smoke-{uuid.uuid4()}"
    marker = uuid.uuid4().hex
    source = (
        f"Retain validation marker {marker}: this isolated test project uses PostgreSQL "
        "and a request timeout of 43 seconds."
    )
    messages = [
        Message(Speaker.USER, source, timestamp),
        Message(Speaker.ASSISTANT, "Recorded for the Retain validation.", timestamp),
    ]
    retainer = build_retainer(
        get_db_connection, generation_client, embedding_client=embedding_client
    )
    try:
        result = retainer.retain(scope, messages, episode_id=episode_id)
        if result.batch is None:
            raise RuntimeError("Fresh Retain smoke episode was unexpectedly skipped")
        if not result.batch.facts:
            raise RuntimeError("Gemini produced no persisted facts for the smoke source")

        fact_ids = [fact.fact_id for fact in result.batch.facts]
        persisted = retainer.store.get_facts(fact_ids)
        if len(persisted) != len(fact_ids):
            raise RuntimeError("Not all Retain facts were readable after the database commit")
        if any(
            fact.scope != scope
            or fact.source_document_id not in {doc.document_id for doc in result.batch.documents}
            or not fact.embedding
            or len(fact.embedding) != 768
            for fact in persisted
        ):
            raise RuntimeError("Persisted fact scope, provenance, or embedding did not verify")

        connection = get_db_connection()
        try:
            with connection.cursor() as cursor:
                document_ids = [doc.document_id for doc in result.batch.documents]
                cursor.execute(
                    "SELECT document_id::text, user_id, project_id, session_id, speaker, text "
                    "FROM documents WHERE document_id = ANY(%s::uuid[]) "
                    "AND retained_at IS NOT NULL",
                    (document_ids,),
                )
                documents = {row[0]: row[1:] for row in cursor.fetchall()}
                cursor.execute(
                    "SELECT count(*) FROM memory_entities WHERE fact_id = ANY(%s::uuid[])",
                    (fact_ids,),
                )
                entity_links = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT count(*) FROM memory_links l "
                    "JOIN memory_facts source ON source.fact_id = l.from_fact_id "
                    "JOIN memory_facts target ON target.fact_id = l.to_fact_id "
                    "WHERE source.source_episode_id = %s OR target.source_episode_id = %s",
                    (episode_id, episode_id),
                )
                stored_links = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT count(*) FROM memory_facts WHERE source_episode_id = %s "
                    "AND user_id = %s AND project_id IS NOT DISTINCT FROM %s "
                    "AND session_id IS NOT DISTINCT FROM %s",
                    (episode_id, scope.user_id, scope.project_id, scope.session_id),
                )
                stored_facts = cursor.fetchone()[0]
        finally:
            connection.close()

        if len(documents) != len(result.batch.documents) or stored_facts < len(persisted):
            raise RuntimeError("Persisted source documents or scoped facts did not verify")
        for fact in persisted:
            doc = documents[fact.source_document_id]
            if doc[0] != scope.user_id or doc[1] != scope.project_id or doc[2] != scope.session_id:
                raise RuntimeError("Stored source document scope did not verify")
            if fact.evidence not in doc[4]:
                raise RuntimeError("Stored fact evidence was not found in its source document")

        replay = retainer.retain(scope, messages, episode_id=episode_id)
        if not replay.skipped:
            raise RuntimeError("Replay did not skip the already committed episode")
        observations = retainer.consolidate(scope) if args.consolidate else []
        current_observation_ids = (
            _verify_observations(retainer.store, scope, observations) if args.consolidate else []
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "episode_id": episode_id,
                    "user_id": scope.user_id,
                    "project_id": scope.project_id,
                    "session_id": scope.session_id,
                    "documents_verified": len(documents),
                    "facts_verified": stored_facts,
                    "entities_written": len(result.batch.entities),
                    "fact_entity_links": entity_links,
                    "links_written": stored_links,
                    "supersessions": len(result.batch.supersessions),
                    "observations_written": len(observations),
                    "observation_versions_verified": len(observations),
                    "current_observation_ids": current_observation_ids,
                    "replay_skipped": replay.skipped,
                    "generation_location": generation_location,
                    "embedding_location": embedding_location,
                },
                indent=2,
            )
        )
        return 0
    finally:
        retainer.store.close()
        generation_client.close()
        embedding_client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Retain smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
