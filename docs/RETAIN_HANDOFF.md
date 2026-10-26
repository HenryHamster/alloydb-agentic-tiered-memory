# Retain cloud test handoff

## What the smoke test exercises

`retain.build_retainer` wires Gemini extraction, regional Gemini embeddings, entity resolution,
supersession classification, semantic and temporal linking, observations, and AlloyDB storage.
The runnable check is `python -m retain.smoke`. It retains one isolated test exchange, verifies
committed source documents and facts, checks fact scope, evidence, and 768-dimensional embeddings,
then replays the same episode and verifies it is skipped. `--consolidate` also runs observation
generation and reads back every changed version, checking its scope, validity fields, and
supporting fact IDs. When contents change, Retain closes the previous version and inserts a row with
a distinct ID; unchanged versions are reused.

The smoke test does not delete data. Each run creates a unique episode and session by default.
Use a dedicated test user and project. A run can add documents, facts, entities, links, history,
and optional observations for that user.

## Configuration

The project reads exported variables; it does not load `.env` automatically. Start from
`.env.example` and set:

| Variable | Required | Use |
| --- | --- | --- |
| `PROJECT_ID` | yes | Vertex AI project; `GOOGLE_CLOUD_PROJECT` is accepted as fallback by the smoke script |
| `DB_HOST` | yes | Reachable AlloyDB instance endpoint |
| `DB_USER` | yes | IAM database user |
| `DB_PORT` | no | Defaults to `5432` |
| `DB_NAME` | no | Defaults to `postgres` |
| `REGION` | no | Regional embedding endpoint; defaults to `us-east1` |
| `GENAI_LOCATION` | no | Gemini generation endpoint; falls back to `GOOGLE_CLOUD_LOCATION`, then `us` |

The host needs the Google Cloud CLI, Application Default Credentials, Vertex AI access for
Gemini and `text-embedding-005`, and private network access to AlloyDB. The IAM database user
must be authorized for the target database. Valkey is not used by this smoke command.

From the repository root, install dependencies and export configuration:

```bash
python -m pip install -e '.[dev]'
gcloud auth application-default login
set -a
source .env
set +a
export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="${GENAI_LOCATION:-us}"
```

On the development VM, use its existing approved ADC setup if already configured; the Python
database client obtains an IAM token by running `gcloud auth application-default print-access-token`.
Do not copy credentials into the repository.

## Schema setup

For the repository's AlloyDB database, apply migrations in this order using AlloyDB Studio or
the project's approved SQL runner:

1. `sql/001_schema.sql` for the existing base schema and extensions.
2. `sql/004_retain_schema.sql` for Retain tables, `vector`, and `pg_trgm` support.

The Retain migration is additive and repeatable. `sql/002_register_model.sql` and
`sql/003_auto_embeddings.sql` are for the legacy in-database AI retrieval path; the Retain smoke
test uses Gemini and embeddings through the Python client and does not require those steps.
Do not run `cleanup_memory_system.py` as part of Retain validation.

## Run

Use an isolated test identity and project ID:

```bash
python -m retain.smoke \
  --user-id retain-pm-test \
  --project-id retain-smoke-project
```

The command prints JSON with `status: ok`, source documents and scoped facts verified, entity and
link counts, optional observation count, endpoint locations, and `replay_skipped: true`. Entity,
link, supersession, and observation counts can be zero for an individual extraction; the smoke
fails if committed source evidence or any expected observation version cannot be verified. Add
`--consolidate` to invoke the real observation generator. The output includes current observation
IDs and the count of persisted versions verified. That flag may update observation history for the
test user, so keep the test identity dedicated.

For asynchronous application wiring, pass the Retain builder into the existing worker:

```python
from async_worker import AsyncMemoryWorker
from db_clients import get_db_connection
from retain import build_retainer

retainer = build_retainer(
    get_db_connection,
    generation_client,
    embedding_client=regional_embedding_client,
)
worker = AsyncMemoryWorker(get_db_connection, generation_client, retainer=retainer)
job = worker.enqueue_extraction(
    user_id, session_id, user_message, assistant_message, project_id=project_id
)
result = job.result(timeout=120)
worker.close(timeout=30)
```

The returned job completes only after the Retain call and its database commit finish. Its
`result()` raises on extraction, embedding, or persistence failure. `worker.wait()` also waits
for queued jobs and surfaces background failures.

## Inspect persisted records

Run these read-only queries in AlloyDB Studio, replacing the test identifiers with the values
used by the smoke output:

```sql
SELECT document_id, episode_id, position, user_id, project_id, session_id,
       speaker, occurred_at, retained_at, text
FROM documents
WHERE user_id = 'retain-pm-test'
  AND project_id = 'retain-smoke-project'
ORDER BY recorded_at DESC;
```

```sql
SELECT fact_id, visibility, fact_type, category, speaker, claim_type, text, evidence,
       source_document_id, source_episode_id, occurred_at, recorded_at, valid_from,
       valid_to, superseded_by, evidence_count, vector_dims(embedding) AS dimensions
FROM memory_facts
WHERE user_id = 'retain-pm-test'
  AND project_id = 'retain-smoke-project'
ORDER BY recorded_at DESC;
```

```sql
SELECT e.entity_id, e.name, e.canonical_name, e.entity_type, e.aliases,
       me.fact_id, me.role
FROM entities e
JOIN memory_entities me USING (entity_id)
WHERE e.user_id = 'retain-pm-test'
ORDER BY e.canonical_name;
```

```sql
SELECT l.from_fact_id, l.to_fact_id, l.link_type, l.relation, l.weight
FROM memory_links l
JOIN memory_facts f ON f.fact_id = l.from_fact_id
WHERE f.user_id = 'retain-pm-test'
ORDER BY l.created_at DESC;
```

```sql
SELECT fact_id, text, valid_from, valid_to, superseded_by, evidence, source_document_id
FROM memory_facts
WHERE user_id = 'retain-pm-test' AND valid_to IS NOT NULL
ORDER BY valid_to DESC;
```

```sql
SELECT observation_id, visibility, entity_id, text, fact_ids, valid_from, valid_to,
       vector_dims(embedding) AS dimensions
FROM observations
WHERE user_id = 'retain-pm-test'
ORDER BY updated_at DESC;
```

## Boundaries

- Existing codelab demos continue using their legacy `agent_entities` writer and retriever by
  default. They are not silently dual-written. Use `build_retainer` (or inject it into the worker)
  for the new Retain tables.
- The legacy `hybrid_retriever.py` reads `agent_entities`; it cannot retrieve Retain facts until
  Recall supplies its adapter for `memory_facts`, `observations`, and source episodes.
- Observations are not built on every message write. The Retainer consolidates a scope's
  observations and purges expired documents every `RetainConfig.maintenance_every` rounds
  (default 20; 0 disables), after the round commits; a maintenance failure is logged and never
  loses the round. `--consolidate` runs it on demand.
- Entities are canonicalized per user as specified by the current model/schema.
- Offline adapter tests do not validate the live AlloyDB SQL dialect, IAM access, Gemini model
  availability, or the complete Retain → Recall → Reflect flow. Those checks remain with the PM
  cloud run.
