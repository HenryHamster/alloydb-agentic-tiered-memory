# Retain skeleton

This is an offline, synchronous write-path scaffold. It does not replace the codelab
worker or change Valkey, Recall or Reflect. Inspected `origin/recall` using `git show`;
no branches were switched or merged. No SQL, database tables or cloud calls are added.

## Implemented and deferred

- `shared/chunking.py`: `chunk_conversation` preserves each original message, trusted
  scope, speaker, timestamp, episode/source ID and exact character offsets. Splits at
  paragraph/sentence/whitespace boundaries when practical, with a hard character limit
  even for unbroken text. Default 2,000 characters approximates 500 tokens; no tokenizer,
  overlap, cross-message grouping or semantic topic detection. Chunk IDs are deterministic
  for the same source, content, scope and offsets. Source IDs must be unique per scope.
- `retain_models.py`: frozen dataclasses and value sets, nonempty identifiers/text,
  timezone-aware dates, provenance consistency, finite embeddings and validity checks.
  Facts target 8–15 words through the extraction contract, with no hard length restriction.
- `retain.py`: injected extraction, source validation, optional entity resolver, explicit
  temporal update planning, typed update links, one atomic store call. `retain` defaults
  to ADD for every extracted fact; it does not detect duplicates or contradictions.
- `retain_store.py`: atomic write protocol and an AlloyDB adapter that explicitly raises
  `NotImplementedError`. `GeminiFactExtractor` also raises `NotImplementedError`.
  These are boundaries, not working cloud integrations.
- `tests/test_retain.py`: offline models, chunking, provenance, explicit update planning
  and orchestration using a mock persistence boundary.

A minimal caller provides `extractor.extract(chunks) -> Sequence[AtomicFact]` and
`store.write(batch) -> None`, then calls `retain(messages, extractor=..., store=...)`.
An optional entity resolver maps a memory to canonical entity IDs. An optional
`classify_updates(memories)` returns explicit `MemoryUpdate` decisions, including old
records loaded under trusted scope. No entity alias matching or LLM conflict classifier
is provided. A resolver/classifier is trusted application code, not raw model output.
The returned batch describes the persisted changes; it is not a Recall search result.
Classification may choose an update category and previous record, but must preserve the
validated new memory, including its text, scope and provenance.

Validation binds scope/speaker/episode to source chunks rather than accepting these
fields from the extractor. Multiple sources must belong to the same scope, speaker and
episode; separate facts are required otherwise. Source modality is preserved; assistant
recommendations remain assistant claims. Extraction must label proposals/hypotheticals/
rejections correctly. Validation does not determine semantic truth or infer user adoption.
Unknown occurrence/validity dates stay null; recorded_at is ingestion time, not event time.

## Temporal semantics

All intervals are half-open `[valid_from, valid_to)`, with null end meaning open.
All timestamps must include a timezone. Supersession requires an explicitly dated,
open previous interval and an explicit effective timestamp later than its start.
Historical SUPERSEDE plans closure of the previous row and insertion of a distinct new
version starting at that timestamp, linked `new -> old`. It preserves the previous text
and provenance. Atomic execution is a future store requirement, not implemented storage.

CONFIRM, REFINE, CONTEST and CONTEXT insert a new attributed fact and link it to the old
fact; they do not overwrite evidence, silently resolve a dispute or change validity.
CONFIRM does not yet aggregate evidence_count on the old record. ADD needs no old record.
Corrections are distinct from historical changes: `ChangeKind.CORRECTION` is representable
but reconciliation raises `NotImplementedError` until invalidation and audit semantics
are agreed. Closing validity alone could falsely imply an incorrect fact was once true.

The store must compare `expected_previous` under locking/version checks and atomically
write insertions, validity replacements, complete source provenance and links. Failed
writes must roll back everything and propagate errors. The adapter must enforce scope,
endpoint existence and uniqueness. No production retry, ingestion idempotency, batch
scheduling or concurrency adapter is implemented. Fact IDs are fresh UUIDs per call;
reprocessing the same messages can create duplicates. Episodes with no extracted facts
are not persisted by this pipeline; source ingestion/retention needs a separate decision.
Even when facts are extracted, the batch carries only cited messages. It cannot reconstruct
an entire conversation round if other messages were not cited. Full-episode ingestion is
required before Recall's get_episode contract can be satisfied.

## Provisional schema proposal — no migration

| Logical record | Proposed fields / responsibilities |
| --- | --- |
| memory_facts | id, text, category, claim_type, speaker, user_id, project_id, session_id, scope, occurred_at, recorded_at, valid_from, valid_to, optional embedding, source_episode_id, provenance/chunk references, entity_ids, supporting_ids |
| memory_links | source memory ID, target memory ID, typed relationship, trusted scope; no weight yet |
| source documents / episodes (optional physical design) | stable episode/source IDs, original text, speaker, timestamp, scope, chunk offsets; retrievable original conversation round |
| entities / memory_entities (optional) | canonical scoped entity IDs, aliases and membership; resolver shared with query-time entity search |
| observations (future, Reflect/Dreaming) | stable observation ID, text, dates, scope, supporting fact IDs and optional embedding |

`MemoryFact.fact` nests text/category/claim/dates; an adapter will flatten these to columns
and expose Recall's `type="fact"`. Category is independent of Recall's memory type.
`evidence_count` currently counts distinct source message IDs; supporting_ids are separate
memory references. The future schema must decide how evidence is deduplicated/aggregated.
Full source messages are retained by chunks in Python, not durably stored by this skeleton.

## Recall and Reflect handoff

Recall's `Memory` needs id, type, text, occurred_at, recorded_at, valid_from, valid_to,
speaker, claim_type, source_episode_id, supporting_ids, entity_ids and evidence_count.
Retain preserves these concepts without importing branch-only code. An eventual adapter
must translate `fact.*` into Recall's flat model and map enum values to strings.
Reflect can follow old versions, typed links and supporting fact IDs without losing evidence.
Observations and mental models remain owned by their respective components.

RecallStore requires four scope-filtered read methods, all using bound SQL parameters:

- `search`: vector, keyword, query-entity and temporal channels, ranked lists by channel;
  exclude expired facts before candidate limits unless include_invalid is requested.
- `expand_neighbors`: one hop from seed memory IDs plus shared entities, with the same
  scope, type, as-of and invalidity filters. Endpoint scope must also be checked.
- `get_episode`: episode ID to original conversation round, filtered by caller scope.
- `get_supporting`: observation ID to its supporting facts, with scope enforcement.

Proposed visibility for agreement: always constrain user_id; global memories visible
within that user; project memories only for matching project; session memories only for
matching session AND matching project (including null project). Preserve origin project/
session metadata even for global facts. Scope is never supplied by an LLM. This policy is
provisional and is not yet an implemented read adapter or an authorization claim.
Retain's Scope includes a memory visibility label; Recall's Scope is caller read context
without that label. They are distinct contracts, not interchangeable dataclasses.
Updates currently require exact Scope equality, including origin project/session IDs.
Thus global/project facts from different origin sessions cannot yet reconcile. Separating
visibility identity from origin metadata is deferred; do not drop provenance to bypass it.

Recall treats occurred_at as a point event, so source message timestamps must not be
copied there for ongoing preferences/states. Those use valid_from when known. The future
adapter must agree valid_from/as-of filtering and half-open interval boundaries with Recall;
its current stale check only tests valid_to, and its overlap helper includes an end boundary.
The categories and claim types in retain_models.py are provisional value sets; no separate
world/experience fact-kind field is implemented. Canonicalizing these labels with Recall
remains an integration decision.

## Decisions still needed

1. Approve physical tables, keys, foreign keys, provenance normalization and episode
   persistence. Define expired/deleted episode responses and source retention policy.
2. Approve visibility rules above, project/session null semantics, and claim eligibility
   filters (proposed/rejected facts must not be presented as established user preferences).
3. Choose embedding model/dimensions and write mechanism. Existing codelab uses 768-dim
   text-embedding-005 on agent_entities; its auto-embedding trigger does not cover new facts.
4. Choose generated text tsvector and GIN/RUM keyword index, HNSW embedding index and
   scope/validity indexes (Recall suggests user_id, project_id, valid_to).
5. Define entity identity/alias resolution for writes AND Recall query-entity search.
6. Approve link directions, traversable types and Recall neighbor ordering/weight meaning.
   ENTITY links here connect memory IDs; canonical entity membership uses entity_ids.
   No all-pairs graph search, graph infrastructure or advanced weights are provided.
7. Agree correction/invalidation audit model, unknown valid_from handling, evidence
   aggregation, update lookup/classification, transaction conflict behavior and idempotency.
8. Agree observation/mental-model storage and stale validity with Reflect. Expensive
   semantic/entity/causal linking and observations can run in a future Dreaming stage.

The explicit setuptools configuration includes retain, retain_models, retain_store and
shared so installed imports and the repository's plain pytest command work. No dependency
changes are needed. Run `uv run --extra adk --extra dev pytest` and the Ruff checks listed
in README.md for local validation.

The existing worker, Valkey lifecycle and ADK wiring remain unchanged. A working cloud
Retain → Recall → Reflect pipeline requires the schema, adapters and explicit wiring above.
LongMemEval ingestion/evaluation and a codelab baseline remain future work; this skeleton
makes no benchmark or end-to-end cloud claim.
