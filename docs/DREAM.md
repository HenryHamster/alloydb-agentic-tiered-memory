# Background Dream reflection

Dream is an optional extension to Reflect. The normal Reflect pass consolidates
stored facts and maintains tiers/macros. Dream rereads explicitly selected source
sessions, stages missed source quotes, and synthesizes reusable lessons. A caller
reviews and applies or discards the entire result. No scheduler starts by default.

## Lifecycle

`start → queued → running → ready → apply / discard`

- `DreamRequest(Scope(user, project), session_ids, request_key=None)` requires one
  project and an explicit nonempty session list. A request key makes enqueue
  idempotent within that scope; it cannot be reused with different inputs/config.
- `Dreamer.worker_once()` claims one job. Expired leases can be reclaimed; tokens
  fence old workers. Source-page checkpoints survive transient failures. Invalid
  model output fails the run; transient errors retry up to three attempts.
- `get(run_id, scope)` returns status, staged output, evidence, and freshness.
- `apply(run_id, scope)` embeds the reviewed output, rechecks evidence inside a
  transaction, then publishes recovered facts, insights and applied status
  atomically. Repeat apply is idempotent. Changed/expired/deleted sources require
  a new run. Applying never automatically supersedes or deletes existing facts.
- `discard(run_id, scope)` clears staged output and fences an active worker. It
  cannot undo an already applied run.

The store is either `InMemoryDreamStore(retain_store)` for tests or
`AlloyDBDreamStore(conn_factory)` for durable PostgreSQL jobs. The reference
in-memory store is deliberately not a durable background service.

## Evidence and visibility

Recovered fact text is an exact source quote, with the original document,
episode, speaker and event time. Assistant statements cannot become asserted
user facts. This first version uses extractive quotes with category `other`;
it does not run entity resolution or automatically resolve older states again.
Normal Reflect/Retain maintenance remains a separate operation.

Lessons are proposed inferences, not newly asserted facts. They must cite valid
fact IDs and exact quotes from their source documents. Structural validation
proves that citations exist; **human review must still check whether the evidence
actually supports the inferred lesson**. A quote alone does not prove entailment.

Read access is restricted to one user/project and the selected sessions. Global
facts are readable context, but a persisted lesson needs a selected source.
Session-private supporting facts narrow derived output to that session;
incompatible private sessions cannot be combined. Visibility is chosen by code.
Prompts omit user/project names and episode IDs, including benchmark metadata.

Published lessons live in `dream_insights`, separately from `macro_memories`, so
ordinary macro maintenance cannot close them. Versions remain attached to runs.
The optional Recall adapter returns only approved, fresh, visible insights and
labels them as inferences in the existing `mental_model` channel. Date-filtered
queries use the underlying facts rather than treating a lesson's creation date
as an event date.

Freshness is deliberately conservative: a semantic change anywhere in a run's
selected source/fact snapshot hides its insights until a new run is reviewed.
Embeddings and access/tier changes do not invalidate insights. This trades some
availability for avoiding stale lessons. Read-time checks scan the relevant
snapshot; this is a small-project implementation, not indexed large-bank search.

Expired source messages cannot be reconstructed from their old facts. Retain's
default document retention is 90 days. Explicit fact deletion invalidates related
published lessons, but does not erase the source conversation: remove retained
source documents too when the original material must never be reread. A new
explicit Dream request can otherwise propose recovering a deleted quote again.

## PostgreSQL setup and worker

Run `sql/006_dream_schema.sql` explicitly after migrations 004 and 005. Importing
Dream and constructing the regular memory system do not execute migrations.
No migration has been applied to a live database as part of this change.

```bash
python -m reflect.dream_cli start --user USER --project PROJECT --sessions s1,s2 --request-key review-1
python -m reflect.dream_cli worker --once --adapter my_adapters:dream_models
python -m reflect.dream_cli status --user USER --project PROJECT --run RUN_ID
python -m reflect.dream_cli apply --user USER --project PROJECT --run RUN_ID --adapter my_adapters:dream_models
# Or reject the staged result:
python -m reflect.dream_cli discard --user USER --project PROJECT --run RUN_ID
```

Omit `--once` to run a polling worker. Use a service manager for process restarts;
queued/checkpointed jobs survive process restarts in PostgreSQL. The worker is a
trusted service that can claim jobs for any user; management APIs require trusted
caller scope. This CLI is for operators, not an unauthenticated public endpoint.

`my_adapters.dream_models()` returns `(structured_generator, embedder)`:

```python
from google import genai
from retain.embeddings import GeminiEmbedder
from shared.llm import GeminiStructuredGenerator


def dream_models():
    # Supply your project's configured clients and billing controls here.
    generation = genai.Client(vertexai=True, project="YOUR_PROJECT", location="global")
    embedding = genai.Client(vertexai=True, project="YOUR_PROJECT", location="us-east1")
    return GeminiStructuredGenerator(generation), GeminiEmbedder(embedding)
```

Production embeddings must match `memory_facts`' 768 dimensions. Generation and
embedding retries, provider authentication, model choice and spending limits
belong to the adapters. Dream itself does not create a billing cap or a payment
method. Long calls should finish inside the configured lease; otherwise their
results are rejected and may be recomputed by another worker.

Apply briefly locks source tables against writes for a consistent publish check;
model calls run outside that transaction. Other users' writes can pause during
this short step. The queue uses `FOR UPDATE SKIP LOCKED` for independent workers.

For application composition, pass `dream_config=DreamConfig()` to
`build_memory_system(...)`, then call
`system.start_dream(Scope(user, project), ["s1", "s2"])`. This also wraps Recall
with the insight adapter. Use `system.dreamer` for status/apply/discard. The
separate worker uses the same database. `finish_session()` does not enqueue
Dream automatically.

## Reproducible diagnostics

```bash
python -m bench.reflect_dream PATH_TO_LONGMEMEVAL_JSON --out bench_results/dream-run-001
# Also accepts the local cached LongMemEval-s directory containing queries.json.gz/documents.json.gz.
python -m bench.reflect_dream PATH_TO_CACHE --out bench_results/dream-recovery-001 --drop-every 5
# Compare a code change using the exact previous ingestion snapshot:
python -m bench.reflect_dream PATH_TO_CACHE --out bench_results/dream-recovery-002 --snapshots-from bench_results/dream-recovery-001 --drop-every 5
```

The runner reuses the existing benchmark's ingestion, Recall, and optional QA
judging. It freezes ingestion once per question, forks the same state for
baseline / Reflect / Reflect+Dream, uses benchmark time, and resets counters.
`--drop-every 5` removes every fifth fact by sorted ID in every arm, independent
of the question/gold; it is explicit synthetic fault injection.

Defaults are **offline plumbing smoke**: verbatim Retain, hash embeddings, and a
question-blind extractive fixture generator with no synthesized lessons. This
exercises recovery/approval/Recall and never calls a provider. It is not a claim
about semantic quality, conflict resolution quality, or LongMemEval QA accuracy.

For actual model diagnostics, supply `--adapter module:factory` returning
`(structured_generator, embedder)`, `--retain-adapter module:factory` returning
`stages(store)` for a real Retain pipeline, and optionally
`--qa-adapter module:factory` returning a plain-text generator. These adapters
must enforce the caller's budget. Merely supplying a model adapter while keeping
verbatim ingestion still lacks production entities/links; label that limitation.

All arms share the same Recall token budget. Retrieved derived memories receive
source-session coverage through their transitive supporting facts, rather than
being assigned to an arbitrary single session. Session coverage is a proxy for
retrieval; it does not establish that an answer or lesson is correct. The cached
dataset lacks turn-level `has_answer` labels, so no turn-level labels are invented.

Every run gets a **new** directory containing a manifest, frozen snapshots,
per-question outputs and summary. Existing directories are rejected. The manifest
records commit, dirty-state flag, executable-code hash, dataset hash, question
IDs, adapter names and arguments. Adapter/model versions must be pinned in their
own configuration. Exceptions remain visible as failed cases; a partial run
exits unsuccessfully. Raw histories and generated output stay in ignored
`bench_results/`; commit only a compact experiment record with conclusions.

Keep test IDs/question IDs stable. For each change record its hypothesis,
baseline run, commit/code hash, fixture/dataset version, models/prompts, metric
changes and regressions. Reserve validation histories separately from these 14
development questions, grouping questions that share conversations. Run semantic
judging repeatedly and review changed cases before claiming a quality gain.

Local tests:

```bash
uv run --extra dev pytest tests/test_dream.py tests/test_bench_reflect_dream.py
# Disposable pgvector PostgreSQL only (also runs in CI when DEV_TEST_DSN is set):
uv run --extra dev pytest tests/test_dream_postgres.py
```
