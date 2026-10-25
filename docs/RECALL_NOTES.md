# Recall: design and status

Recall is the read path. It finds stored memories that the caller may read, ranks them, and
returns them dated, typed and fitted to a token budget. It reads Retain's tables
(`sql/004_retain_schema.sql`) and reuses Retain's records, scope and read rule. The pipeline,
ranking, in-memory store and AlloyDB store are implemented. The AlloyDB store is checked
offline (every statement variant parses as PostgreSQL, every parameter is bound, rows decode)
but has not yet run against a live instance. `bench/` runs LongMemEval against Recall
([benchmarks](BENCHMARKS.md)).

## Pipeline

The stage overview lives in the `recall/recall.py` module docstring.

Callers: `turn_context.build_turn_context` (every turn, `depth=low`, with the sliding window's
round IDs excluded) and the agent tool `search_memory` (`depth=mid`, 4,096 tokens).

## Files

| File | Contents | Status |
| --- | --- | --- |
| `recall/recall.py` | `Recaller` (stage overview in the module docstring), `build_recaller` (production wiring) | Complete |
| `recall/config.py` | `RecallDepth`, `MemoryType`, `RecallConfig` (every tuning constant) | Complete |
| `recall/models.py` | `TimeWindow`, `Memory` (one view over facts, observations, episodes), results, trace | Complete |
| `recall/query.py` | `parse_time_phrase`, `parse_time_range`, `analyze_query` | Complete |
| `recall/ranking.py` | `fuse_rrf`, `rerank`, `apply_boosts`, `filter_and_abstain` | Complete |
| `recall/packing.py` | `format_line`, `pack` | Complete |
| `recall/store.py` | `RecallStore` protocol, `InMemoryRecallStore` over `InMemoryRetainStore` | Complete |
| `recall/alloydb_store.py` | `AlloyDBRecallStore`: one-statement multi-channel search, graph expansion, episodes, supporting facts | Implemented; not run on a live instance |
| `recall/reranker.py` | `AlloyDBReranker` (`ai.rank()`), one reused connection | Written, not run on a live instance |
| `bench/` | LongMemEval harness, benchmark embedders, QA judge, AlloyDB benchmark writer | Runs offline; Vertex and AlloyDB paths not run here |
| `recall/tools.py` | `search_memory`, `get_episode`, `get_supporting`; ADK `FunctionTool`s | Complete |
| `turn_context.py` | Sliding window plus Recall under one turn budget | Complete |
| `shared/llm.py`, `shared/tokens.py` | Gemini text generation with 429 retry, token estimate | Complete |

## Reused from Retain

- `retain.models.Scope` is the caller context; tools bind it in code, so no tool schema has
  user, project or session arguments.
- `is_visible` is the read rule in memory. In AlloyDB, `CANDIDATE_SQL` is Retain's `VISIBLE_SQL`
  with the validity check swapped (as of a given time, optionally including superseded rows);
  keep the two in sync.
- `Speaker`, `ClaimType`, `FactType` and `Category` label packed lines and results.
- `retain.embeddings.Embedder` / `GeminiEmbedder` embed queries with the same model as facts.
- `normalize_name` matches query text to entity names and aliases.
- `InMemoryRetainStore` tables and its cosine similarity back `InMemoryRecallStore`.

## Decisions

- **Episodes.** An episode is the `documents` rows sharing an `episode_id`. The sliding window's
  `round_id` is passed as the `episode_id` (`SessionBuffer.append` → `enqueue_extraction(...,
  episode_id=)` → `retain_turn`), so `exclude_episode_ids` matches `source_episode_id`.
- **Episode visibility.** Documents have no visibility column. Recall reads them in the narrowest
  domain they belong to: the project if set, else the session, else every project of the user
  (`episode_visible`, `EPISODE_SQL`). An expired episode returns nothing; facts keep `evidence`.
- **Superseded memories** are excluded inside the store query (so they never take pool slots)
  and checked again before packing. `include_invalid=True` returns them flagged `stale`.
- **Score floor** applies to the raw reranker score, before boosts, so a recency boost cannot
  rescue an irrelevant memory. With no reranker there is no floor: fused ranks are not
  calibrated relevance.
- **Recency boost** is skipped when the question has a window ("what did I do in 2023?").
- **Rerank the fused head only.** 1,000 cross-encoder scores per call at `depth=high` is too slow.
- **Graph expansion** follows `memory_links` in both directions by weight, then facts sharing an
  entity with a seed.
- **Query entities** are matched by name or alias appearing in the query (`find_entities`-style
  lookup), not by running Retain's `EntityResolver`, which would create entities on read. In
  AlloyDB the query's word n-grams (1-4 words, no stopword at either end) are compared with
  `canonical_name`, `name` and `aliases`, plus `pg_trgm` similarity >= 0.6 on canonical names
  for n-grams of 4+ characters (`min_entity_similarity`).
- **Keyword search ORs the query's lexemes** (`plainto_tsquery` with `&` turned into `|`), ranked
  by `ts_rank`. `websearch_to_tsquery` ANDs every word, so whole questions rarely matched.
- **One search statement.** Each channel is a CTE that ranks facts, observations and episodes
  together by one key (vector distance, `ts_rank`, or primary date), so a channel is one list
  across types as in `InMemoryRecallStore`. Rows come back once each as JSON payloads.
- **Filtered HNSW.** The store sets `hnsw.iterative_scan = relaxed_order` (pgvector >= 0.8) on
  connect, so the per-user filter does not empty the vector channel; without it, a warning is
  logged. Rank order is recomputed from the distance, so relaxed order does not leak through.
- **Supporting facts** are returned even when superseded, matching `InMemoryRecallStore`.

## Open questions

For Retain:
1. Is the episode visibility rule above what you intend, or should `documents` get a
   `visibility` column?
2. Should a fact reinforced by a later round (`evidence_count + 1`) also record that round's
   episode? Today only the first round is its `source_episode_id`, so a window showing the
   later round does not exclude it.

For Reflect (`origin/reflect`, not merged here):
1. `types=["mental_model"]` is accepted but returns nothing until a table exists.
2. Recall should call `ReflectStore.record_access` for returned rows and search through
   `tiered_search` (hot → warm → archived); both are stubs on that branch.
3. Reflect owns the thin-evidence threshold; Recall returns `items[*].relevance` and `truncated`.

## Next

1. Run `AlloyDBRecallStore` against a live instance (`python -m bench.longmemeval ...
   --backend alloydb`) and check the plans use the HNSW, GIN and scope indexes (`EXPLAIN`).
2. Verify `ai.rank()` (argument names, model ID, 1-based `index`; needs the Discovery Engine API).
3. Calibrate `score_floor` on real `ai.rank()` scores; tune `recency_half_life_days`. LongMemEval
   runs with `--reranker alloydb` give the labelled sample.
4. `GeminiEmbedder` is still Retain's stub, so production query embedding raises and Recall
   falls back to keyword, entity and temporal search until it is filled. The benchmark uses
   its own `bench.embedders.VertexEmbedder` (same model and dimensions) meanwhile.
5. Replace `retrieve_hybrid_entities` in `adk_memory_provider.py` and `agent_orchestrator.py`
   with `build_turn_context(...)` and register `make_recall_tools(...)` on the agent.
