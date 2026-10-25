# Recall: design and status

Recall is the read path. It finds stored memories that the caller may read, ranks them, and
returns them dated, typed and fitted to a token budget. It reads Retain's tables
(`sql/004_retain_schema.sql`) and reuses Retain's records, scope and read rule. The pipeline,
ranking and an in-memory store are complete and tested offline. The AlloyDB store is a stub
marked `TODO(recall)`, in the same style as Retain's.

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
| `recall/alloydb_store.py` | `AlloyDBRecallStore`, validity and episode SQL | Stub with SQL plan per method |
| `recall/reranker.py` | `AlloyDBReranker` (`ai.rank()`) | Written, not run on a live instance |
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
  lookup), not by running Retain's `EntityResolver`, which would create entities on read.

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

1. Implement `AlloyDBRecallStore` from its SQL plans and run it against a live instance.
2. Verify `ai.rank()` (argument names, model ID, 1-based `index`; needs the Discovery Engine API).
3. Calibrate `score_floor` on real `ai.rank()` scores; tune `recency_half_life_days`.
4. `GeminiEmbedder` is still Retain's stub, so production query embedding raises and Recall
   falls back to keyword, entity and temporal search until it is filled.
5. Replace `retrieve_hybrid_entities` in `adk_memory_provider.py` and `agent_orchestrator.py`
   with `build_turn_context(...)` and register `make_recall_tools(...)` on the agent.
