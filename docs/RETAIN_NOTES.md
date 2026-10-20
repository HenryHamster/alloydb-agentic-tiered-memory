# Retain: design and status

Retain is the write path. For every conversation round it stores the original messages,
extracts short dated facts, resolves entities and links facts to each other, so Recall and
Reflect can search and reason over them. This change is a **skeleton**: the contracts, schema
and pipeline wiring are complete and tested offline; model and database stages are stubs
marked `TODO(retain)`.

## Pipeline

```
AsyncMemoryWorker (retainer=...)          one queued turn, trusted Scope(user, project, session)
└─ Retainer.retain_turn
   1 documents        store both messages; episode = one round; skip if already retained
   2 chunk_documents  700-char lossless spans (shared/chunking.py)
   3 FactExtractor    8-15 word facts + evidence quote, who/what/when/where/why, entities,
                      relations, causal refs, proposed visibility
   4 EntityResolver   mentions -> canonical per-user entities (exact, alias, fuzzy)
   5 Embedder         768-dim text-embedding-005 vectors
   6 SupersedeClassifier  add / duplicate (reinforce) / supersede (close valid_to)
   7 links            supersedes + causal (from extraction) + Linker (semantic, temporal)
   8 RetainStore.write    one transaction; marks documents retained
Retainer.consolidate      periodic: ObservationBuilder -> observations
```

Short-term memory (Valkey rolling summary) is unchanged codelab code in `valkey_buffer.py`.

## Files

| File | Contents | Status |
| --- | --- | --- |
| `sql/004_retain_schema.sql` | `documents`, `memory_facts`, `entities`, `memory_entities`, `memory_links`, `entity_relations`, `observations` | Complete |
| `retain/models.py` | Enums, records, `is_visible`, `resolve_visibility`, `stable_id` | Complete |
| `retain/pipeline.py` | `Retainer`, decision handling, `build_retainer` (production wiring) | Complete |
| `retain/store.py` | `RetainStore` protocol, `InMemoryRetainStore` reference | Complete |
| `retain/extraction.py` | `chunk_documents`; `GeminiFactExtractor` | Extractor stub |
| `retain/entities.py` | `normalize_name`; `FuzzyEntityResolver` | Resolver stub |
| `retain/embeddings.py` | `Embedder` type; `GeminiEmbedder` | Stub |
| `retain/linking.py` | `causal_links`; `StoreLinker`, `GeminiSupersedeClassifier` | Stubs except causal |
| `retain/observations.py` | `GeminiObservationBuilder` | Stub |
| `retain/alloydb_store.py` | `AlloyDBRetainStore`, shared visibility SQL | Stub with SQL plan per method |
| `async_worker.py` | `retainer=` replaces the `agent_entities` upsert; passes `project_id` and message time | Complete |

Each stub's docstring is its specification. Fill one without touching the others.

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
- **Trust.** Speaker, scope and episode come from documents, not the model. An assistant cannot
  assert a user preference; it is stored as `proposed`.
- **Classifier output is untrusted.** Each duplicate/supersede target must be stored, current,
  same user and visible from the new fact; a fact never supersedes a broader one.
- **Duplicates** reinforce the stored fact (`evidence_count + 1`) instead of inserting a copy.
- **Links.** `link_type` is semantic, temporal or causal as proposed; `relation` gives direction
  (`similar`, `near_in_time`, `supersedes`, `caused_by`, `evidenced_by`). Weight is in [0, 1].
  Entity links live in `memory_entities`, not `memory_links`. Temporal links use `occurred_at`
  only, so facts are not linked merely for being said in the same conversation.
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

## Next

Implement the stubs in this order: `AlloyDBRetainStore`, `GeminiEmbedder`, `GeminiFactExtractor`,
`FuzzyEntityResolver`, `StoreLinker`, `GeminiSupersedeClassifier`, `GeminiObservationBuilder`.
Then pass `build_retainer(get_db_connection, client)` to `AsyncMemoryWorker` in the demos.
`hybrid_retriever.py` still reads `agent_entities`, so the codelab demos keep the legacy writer
until Recall's store reads the Retain tables.
