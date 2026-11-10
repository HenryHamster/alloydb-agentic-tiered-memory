# Retain: design and status

Retain is the write path. For every conversation round it stores the original messages,
extracts facts, resolves entities and links facts to each other, so Recall and Reflect can
search and reason over them. Gemini extraction, embedding, AlloyDB storage, entity resolution,
linking, update classification, and observation building are implemented.

## Pipeline

```
AsyncMemoryWorker (retainer=...)          one queued turn, trusted Scope(user, project, session)
└─ Retainer.retain_turn
   1 documents        store both messages; episode = one round; skip if already retained
   2 chunk_documents  700-char lossless spans (shared/chunking.py)
   3 FactExtractor    ~8-15 word facts + evidence quote, when, entities, relations,
                      causal refs, proposed visibility; tool calls/results built by code
   4 EntityResolver   mentions -> canonical per-user entities (exact, alias, fuzzy)
   5 Embedder         768-dim text-embedding-005 vectors
   6 SupersedeClassifier  add / duplicate (reinforce) / supersede (close valid_to)
   7 links            supersedes + causal (from extraction) + Linker (semantic, temporal)
   8 RetainStore.write    one transaction; marks documents retained
Retainer maintenance     every 20 rounds per scope (RetainConfig.maintenance_every):
                          ObservationBuilder -> observations, then purge expired documents
```

Short-term memory is `valkey_buffer.py`: the sliding window and rolling summary. Its `round_id`
is passed to Retain as the `episode_id`, so Recall can skip rounds still in the window.

## Files

| File | Contents | Status |
| --- | --- | --- |
| `sql/004_retain_schema.sql` | `documents`, `memory_facts`, `entities`, `memory_entities`, `memory_links`, `entity_relations`, `observations` | Complete |
| `retain/models.py` | Enums, records, `is_visible`, `resolve_visibility`, `stable_id` | Complete |
| `retain/pipeline.py` | `Retainer`, decision handling, `build_retainer` (production wiring) | Complete |
| `retain/store.py` | `RetainStore` protocol, `InMemoryRetainStore` reference | Complete |
| `retain/extraction.py` | `chunk_documents`; `GeminiFactExtractor` | Implemented |
| `retain/entities.py` | `normalize_name`; `FuzzyEntityResolver` | Implemented |
| `retain/embeddings.py` | `Embedder` type; `GeminiEmbedder` | Implemented |
| `retain/linking.py` | `causal_links`; `StoreLinker`, `GeminiSupersedeClassifier` | Implemented |
| `retain/observations.py` | `GeminiObservationBuilder` | Implemented; runs in periodic maintenance or `Retainer.consolidate` |
| `retain/alloydb_store.py` | `AlloyDBRetainStore`, shared visibility SQL | Implemented |
| `async_worker.py` | `retainer=` replaces the `agent_entities` upsert; passes `project_id` and message time | Complete |

## Decisions

- **Scope.** The caller supplies `Scope(user_id, project_id, session_id)`. The model only proposes
  `visibility` (global, project, session); `resolve_visibility` narrows it to what the context
  supports. A project fact with no project falls to session, never to global.
- **Read rule** (shared with Recall): same user, then global, same project, or same session.
  Every store query and both ends of every link must satisfy it.
- **Idempotency.** Documents, facts and entities have deterministic IDs. `documents.retained_at`
  is the ingestion ledger, so a replayed round is skipped without a model call. A failed stage
  writes nothing and the round can be retried.
- **Time.** `occurred_at` is event time (null for timeless states); `valid_from` defaults to the
  message time; `recorded_at` is ingestion time and never stands in for either. Supersession
  sets the old fact's `valid_to` to the new fact's `valid_from`, never earlier than the old start.
  Reduced-precision ISO dates (`2023`, `2024-03`) in either extracted timestamp field preserve
  the fact and date in `details.when`, leaving the exact timestamp unset. The usual source-time
  default for `valid_from` still applies in storage; it does not mean the partial date was
  resolved. No missing month or day is invented. Complete dates and timestamps remain validated;
  a cited relative phrase can supply an exact date only when the source timestamp is known.
- **Trust.** Speaker, scope and episode come from documents, not the model. An assistant cannot
  assert a user preference; it is stored as `proposed`.
- **Classifier output is untrusted.** Each duplicate/supersede target must be stored, current,
  same user and visible from the new fact; a fact never supersedes a broader one.
- **Fail-soft on model output.** A bad extraction row, repeated key or bad reference drops only
  that row; an invalid or missing classifier decision keeps the fact as new; an ambiguous entity
  name gets its own entity. A response that is not valid JSON still fails the round, which then
  stays unretained so it can be retried.
- **Extraction cost.** `build_retainer` runs extraction at `low` thinking with the lean schema
  (no who/what/where/why details, which Recall never reads) and builds tool-call and tool-result
  facts from their JSON, since low thinking tends to skip tool output. The class defaults keep the
  full schema at `medium`. On three Mem2Act histories this cut extraction from $2.63 to $0.35 and
  kept every key fact. Code locates the evidence quote, including quotes of serialized assistant
  turns whose line breaks are stored as `\n`; the words must still appear in order.
- **Requests are facts.** The prompt records what the user asked for, and a short user message
  that no extracted fact cites is kept verbatim (`The user said: ...`), unless the entire message
  is known courtesy/reaction wording (e.g. "Okay, thank you very much!"). The fallback-only
  filter accepts combinations of known phrases, case/whitespace/punctuation variations and
  curly apostrophes. Additional content ("Thanks, but use server psql-001") and ambiguous
  decisions ("Yes, use that one") remain eligible. This conservative rule adds no model calls;
  it does not remove validated extracted facts or source documents. The existing single-chunk,
  user-only, at-least-three-word fallback bounds remain. Unknown wording can still produce noise.
- **Duplicates** reinforce the stored fact (`evidence_count + 1`) instead of inserting a copy.
- **Links.** `link_type` is semantic, temporal or causal as proposed; `relation` gives direction
  (`similar`, `near_in_time`, `supersedes`, `caused_by`, `evidenced_by`). Weight is in [0, 1].
  Entity links live in `memory_entities`, not `memory_links`. Temporal links use `occurred_at`
  only and never join two facts from the same round, so facts are not linked merely for being
  said in the same conversation.
- **Round links.** With `link_rounds` (on in `build_retainer`) code adds `evidenced_by` links that
  similarity cannot find, because requests, tool events and answers fall in different
  categories: a tool call cites the user's request, a tool result the call before it, and an
  assistant fact the tool event whose names it repeats, else the request. `evidenced_by` is not
  a Reflect clustering relation, so these links add no macro summaries.
- **Code-built facts** (tool calls, tool results, `The user said: ...`) get `other` entities from
  naming fields (`*_id`, `name`, `path`, `url`, `make`, ...) and ID-like tokens, never tool
  names. Two code-built facts never get a `similar` link: their template wording inflates cosine.
- **Graph.** Stored relationally (`memory_links`, `entity_relations`); traversals are SQL joins or
  recursive CTEs. Apache Spark can read these tables for offline graph analytics; it is not on
  the per-turn path.
- **Embeddings** are computed by Retain, not `ai.initialize_embeddings`, because linking needs the
  vector before commit.

## Answers to Recall's open questions (docs/RECALL_NOTES.md on `recall`)

1. Store tables: see `sql/004_retain_schema.sql`. The AlloyDB `RecallStore` reads them.
2. `memory_facts` has `user_id`, `project_id`, `speaker`, `claim_type`, `source_episode_id`,
   `evidence_count`, `text_tsv` (GIN), HNSW on `embedding`, index on `(user_id, project_id,
   valid_to)`. `observations` has text, `text_tsv`, dates and `user_id`.
3. Episodes are `documents` rows sharing `episode_id`, ordered by `position`. An expired episode
   returns nothing; facts keep their `evidence` quote.
4. Scope: the read rule above.
5. Values: `claim_type` asserted/proposed/hypothetical/rejected; `fact_type` world/experience.
6. Weight: semantic = cosine, temporal = time proximity (1 for supersedes), causal = 1.

## Validation and integration

### Fallback-filter evaluation

Offline regression checks: `uv run pytest tests/test_retain_extraction.py`. These use synthetic
examples, not the missing raw LoCoMo traces. They cover courtesy-only fallbacks, mixed-content
and uncertain messages, existing extraction coverage, source retention/replay, and tool links.

The follow-up cloud comparison should use `conv-26-q037`, `q138`, `q140`, `q144`, and `q147`.
Junk fallbacks were reported in their recall slots, not as the primary failure cause. The
recorded experiment had 34 whole-turn fallbacks among 329 facts; not all are necessarily junk.
Using the raw exports referenced in `experiments/locomo-001/README.md`, label each candidate
fallback as junk, useful, or uncertain before comparing the base and changed code. Re-ingest
identical history into separate fresh memory snapshots: existing facts and prepared memory
caches do not get cleaned by this change. Hold extraction responses, Recall settings, answer
model and scoring fixed where possible; provider-response replay can isolate the filter.
Report junk fallbacks stored/retrieved, useful evidence preserved/retrieved and answer
correctness for each question. Fewer fallbacks alone do not establish better answer accuracy.

See [Retain cloud test handoff](RETAIN_HANDOFF.md) for configuration, schema setup, the live
test (`tests/test_retain_live.py`), and read-only inspection queries. The codelab continues to use its legacy
`agent_entities` writer and reader unless a `build_retainer(...)` instance is explicitly injected
into the worker. The legacy reader cannot retrieve `memory_facts` or `observations`; Recall still
needs its AlloyDB adapter for the Retain schema before Retain → Recall → Reflect can be verified.
