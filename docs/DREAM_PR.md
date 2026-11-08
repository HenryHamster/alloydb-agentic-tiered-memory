# Add background Dream reflection with review before publishing

## Summary

Reflect previously operated on stored facts only. This change adds an optional
Dream worker that rereads selected source conversations, proposes missed source
quotes, and synthesizes reusable, evidence-backed lessons. Results stay staged
until a caller explicitly reviews and applies them.

## Changes

- Add a PostgreSQL job queue with leases, fenced workers, bounded retries and
  source-page checkpoints; provide an in-memory reference implementation.
- Add scoped start, status, apply and discard controls, plus a worker CLI.
- Validate exact source quotes and supporting fact IDs. Preserve speaker,
  temporal context and session-private visibility; label lessons as inferences.
- Publish recovered facts and separate insight versions atomically. Reject stale
  staged results and hide approved insights when their evidence snapshot changes.
- Add optional application/Recall integration. Normal session-end Reflect does
  not automatically launch Dream.
- Add a paired LongMemEval runner with frozen, reusable snapshots; baseline,
  Reflect and Reflect+Dream arms; immutable manifests and a compact experiment
  record connecting code/data hashes to results.

## Validation

- Full local test suite: **442 passed, 9 skipped**.
- Lint, formatting and whitespace checks pass.
- New tests cover review/apply, rollback, stale/deleted/expired evidence, scope
  isolation, speaker attribution, private insights, retry/checkpoint behavior,
  lease fencing, approved-only Recall and snapshot replay.
- All **14 selected LongMemEval questions × 3 arms = 42 offline smoke runs**
  completed, preserving original fact IDs in every arm. The existing Recall
  benchmark also completed the same 14 questions.
- **No paid model calls; API cost $0.**

## Limits and rollout

The benchmark used verbatim ingestion, hash embeddings and an extractive fixture
generator. It demonstrates pipeline behavior, not improved QA accuracy or the
quality of inferred lessons. No model-backed QA comparison was run.

Four new real-PostgreSQL tests are included and run when `DEV_TEST_DSN` points to
a disposable pgvector database; they were skipped locally because none was
configured. A live AlloyDB migration or deployment has not been performed.

Enable Dream explicitly and run `sql/006_dream_schema.sql` after 004/005 when
deploying. Provider adapters own billing controls. Apply briefly locks source
tables; freshness checks conservatively scan the source/fact snapshot. Human
review remains necessary to validate the meaning of generated lessons.

Usage and design: [DREAM.md](DREAM.md). Experiment metadata:
[RF-DREAM-001](../bench/experiments/reflect_dream.json). Raw local reports remain
under ignored `bench_results/dream-20261107-final/`.
