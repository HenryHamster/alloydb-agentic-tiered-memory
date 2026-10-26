# Validation and inherited limitations

## Offline checks

The local test suite imports the nine library modules with the installed
dependencies, checks rolling-summary ordering and recent-turn preservation,
exercises the in-process Valkey fallback, sends a real background queue exchange
through fake model/database boundaries, checks permission outcomes, constructs a
Google ADK agent with the callback and function tool, and checks prompt assembly.
Ruff checks imports, undefined names, and consistent formatting.
`tests/test_retain.py` covers the Retain skeleton: chunking, visibility rules, the pipeline
with fake stages, idempotent replay, duplicates and supersession, classifier target checks,
atomic store writes and worker wiring. Its Gemini and AlloyDB stages are stubs.
`tests/test_reflect.py` covers the Reflect skeleton: config and record validation, link types
and provider-neutral production wiring, including the merged store module. Every Reflect
processing stage is a stub; Recall integration with facts/macros is not implemented.

On October 10, 2026, all 11 offline tests passed on Windows with Python 3.14.7,
Google GenAI 2.29.0, and Google ADK 2.11.0. Ruff checks and formatting passed.
The extracted modules also matched the upstream Python syntax trees after
accounting for import cleanup and the documented project-setting change.
`uv.lock` records resolved dependency versions. CI uses the same lockfile on
Python 3.11 and 3.12.

These checks do not call Google Cloud, connect to live storage, or measure recall
quality or latency. The original `test_memory_system.py` and `test_adk_agent.py`
are live demos; they are excluded from pytest discovery by `testpaths = ["tests"]`.
Run them after provisioning and SQL initialization to validate your deployment.

## Properties inherited from the tutorial

- The buffer is bounded by message count, not tokens. The summary is not subject
  to a hard token budget. Summarization happens synchronously before a turn returns.
- Entity extraction uses a daemon thread and an in-process queue. Writes become
  available later, queued work is not durable, and the demos use fixed waits.
- The entity key is `(user_id, entity_name)`. The same entity name across a user's
  projects can overwrite an earlier row. Project/session metadata does not provide
  versioned memory history or database-level tenant isolation.
- Hybrid search builds SQL filter expressions from IDs. The sample is unsuitable
  as an isolation boundary for arbitrary untrusted IDs. Permissions also use
  regex matches and a non-atomic get/delete for single-use grants; no production
  authorization guarantee is established by the offline checks.
- Several errors are logged and replaced with fallbacks, which can hide a failed
  storage or model call. A successful demonstration must be checked against its
  actual database records and logs.
- Token totals use character-count approximations and do not account for every
  query decomposition, embedding, completion, or retry. They do not establish
  billed token/cost savings or reproduce the separate 45-turn headline benchmark.
- The tutorial uses preview AlloyDB AI functions and specific model identifiers.
  Their availability and cloud behavior must be checked in the target project.

The repository preserves this educational algorithm. Its improvements make the
sample source installable and reviewable; a live cloud run and production
hardening are separate work.
