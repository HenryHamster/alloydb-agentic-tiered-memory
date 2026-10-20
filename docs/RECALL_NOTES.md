# Recall: status and open dependencies

Recall is the read path: it finds stored memories, ranks them, and returns them dated, typed and
fitted to a token budget. This file lists what is built and what still waits on Retain, Reflect,
or a live cloud run. In the code, those gaps are marked `PLACEHOLDER(RETAIN)`,
`PLACEHOLDER(REFLECT)` and `PLACEHOLDER(CLOUD)`; search for `PLACEHOLDER(` to find them.

## Built

| File | Contents |
| --- | --- |
| `recall.py` | `recall()` and its stages: `analyze_query`, `fuse_rrf`, `rerank`, `apply_boosts`, `filter_and_abstain`, `pack`; result types; the `RecallStore` read contract; Gemini embedder, Gemini generator and `ai.rank()` reranker adapters |
| `recall_tools.py` | `search_memory`, `get_episode`, `get_supporting` as plain callables (for Reflect) and as ADK `FunctionTool`s |
| `tests/test_recall.py` | Offline tests with a fake store, covering every stage and the tool schemas |

### Sliding window (short-term tier)

`valkey_buffer.py` keeps each session's recent turns in Valkey. Past `trigger_limit` rounds (10),
the oldest rounds are folded into a rolling summary and then trimmed, leaving the last
`window_size` rounds (4) verbatim. Each round gets a `round_id`. `turn_context.build_turn_context`
builds the per-turn context:

1. Read the window; a new session with no summary starts from the previous session's summary.
2. Charge the summary and recent turns to the turn budget (4,096 tokens) first.
3. Call `recall()` (`depth="low"`) with the remaining budget (at least 512 tokens) and
   `exclude_episode_ids` set to the window's round IDs. Memories drawn from those rounds are
   dropped after fusion, before reranking, so they don't repeat what the window already shows
   or take rerank slots.
4. If Recall fails, the turn continues with the window alone.

Fixed from the codelab buffer: a summary failure appended the turn a second time into the
in-process fallback; the trim kept "the last N" and could drop a turn added during
summarization (it now removes exactly the summarized count from the head); `window_size=0`
summarized nothing; unreadable stored turns failed the whole read. The codelab functions
`append_session_turn_with_rolling_summary` and `get_session_context_buffer` remain as wrappers.

`search_memory` is a preset `recall()` call (`depth="mid"`, `max_tokens=4096`). It is not a
second pipeline. It exists so the agent sees three arguments (`query`, `time_range`,
`include_episodes`) and never sees scope; user, project and session are bound in code.

### Changes from the written plan

| Plan | Built | Reason |
| --- | --- | --- |
| Four parallel searches | Vector, keyword, query-entity and temporal searches in one store call (meant to be one SQL statement), then a one-hop graph expansion from the fused top 10 | "One hop from top hits" needs the other searches' results first; psycopg2 connections cannot be shared across threads |
| Rerank the whole candidate pool | Rerank the top 150 fused results, keep 30 | 1,000 cross-encoder scores per call at `depth="high"` is too slow; 150 matches the tested setting |
| Score floor | Applied to the raw reranker score, before boosts | A recency boost must not rescue an irrelevant memory |
| Hide superseded facts | Excluded inside the search query (store contract), and checked again before packing | Stale rows would otherwise use candidate slots |
| (not specified) | No score floor when no reranker ran | Fused ranks are not calibrated relevance, so the floor would abstain at random |
| (not specified) | Recency boost is skipped when the question has a time window | "What did I do in 2023?" should not be pulled toward recent facts |
| (not specified) | If embedding fails, skip the vector channel | The old retriever used a zero vector, which returns arbitrary neighbours |
| Time phrases via LLM | Rule parser first (yesterday, last week/month/year, past N days, in March, in 2025), LLM only for what rules miss | Cheaper, deterministic, and still works when `decompose=False` |

Window precedence: an explicit `temporal_window`, then the rule parser, then the LLM.

Boosts (multiplied in as `relevance * (1 + boost)`, total clipped to ±0.25): recency ±0.10
(90-day half-life), window fit ±0.10, evidence count up to +0.05.

### Also fixed

`hybrid_retriever.py` formatted `user_id` and `project_id` straight into the
`ai.hybrid_search()` filter string, which allowed SQL injection. The IDs are now quoted with
`cursor.mogrify`.

## Needed from Retain — PLACEHOLDER(RETAIN)

1. **An AlloyDB implementation of `RecallStore`** (`recall.py`). It can't be written until the
   tables exist. It needs four methods: `search`, `expand_neighbors`, `get_episode`,
   `get_supporting`. Every filter must use bound parameters.
2. **Columns missing from the plan's tables:**
   - `memory_facts`: `user_id`, `project_id` (scoping); a generated `tsvector` column with a
     GIN/RUM index (keyword search); `speaker`; `claim_type`; `source_episode_id`;
     `evidence_count` (or a way to derive it). An HNSW index on `embedding`. An index on
     `(user_id, project_id, valid_to)`.
   - `observations`: `observation_id`, `text`, a `tsvector`, dates (`valid_from`, `valid_to`,
     `recorded_at`), `user_id`. The plan only lists `fact_id (list)`, `embedding` and
     `session/project_id`.
   - **Episodes**: no table is defined. Is "episode" the `documents` table (original text)?
     `get_episode` needs `episode_id → original round`. If documents expire after "x" time,
     `get_episode` needs a defined response for an expired episode.
3. **Scope rules.** What does a recall in project P, session S see: only P? P plus the user's
   global memories (the old `agent_entities` table worked that way)? Is session only a ranking
   signal or a filter?
4. **Query-entity resolution.** The entity search channel needs Retain's NER and alias matching
   applied to the query text. Either expose it as a SQL function, or let the store call it.
5. **Value sets** for `claim_type` and fact kinds (the plan's "world or experience") so the
   packed lines use labels Retain actually writes.
6. **Episode IDs from the window.** `SessionBuffer.append` returns the round's `round_id`.
   Retain must store it as the episode ID / `source_episode_id`, or `exclude_episode_ids` never
   matches. `AsyncMemoryWorker.enqueue_extraction` does not take it yet.
7. **`memory_links.weight` meaning** for ordering graph neighbours, and which link types the
   one-hop expansion should follow (all of them, or semantic and causal only).

### Intended `RecallStore.search` SQL (sketch, table names from the plan)

One statement, one round trip. Each search is a subquery that returns `(channel, rank, id)`;
the rows are then joined back to their memory once.

```sql
WITH
vec AS (
  SELECT 'vector:0' AS channel, fact_id AS id,
         ROW_NUMBER() OVER (ORDER BY embedding <=> %(qvec0)s::vector) AS rank
  FROM memory_facts
  WHERE user_id = %(user_id)s AND project_id = %(project_id)s
    AND (%(include_invalid)s OR valid_to IS NULL OR valid_to > %(as_of)s)
  ORDER BY embedding <=> %(qvec0)s::vector
  LIMIT %(limit)s
),
kw AS (
  SELECT 'keyword:0', fact_id,
         ROW_NUMBER() OVER (ORDER BY ts_rank(text_tsv, q) DESC)
  FROM memory_facts, websearch_to_tsquery('english', %(q0)s) q
  WHERE text_tsv @@ q AND user_id = %(user_id)s AND ...same filters...
  LIMIT %(limit)s
),
ent AS ( /* facts linked through memory_entities to entities resolved from the query */ ),
tmp AS ( /* facts whose interval overlaps [%(win_start)s, %(win_end)s), skipped when no window */ )
SELECT c.channel, c.rank, f.*
FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw UNION ALL ...) c
JOIN memory_facts f ON f.fact_id = c.id
ORDER BY c.channel, c.rank;
```

Repeat `vec`/`kw` for each sub-query (at most 5), and union in the same subqueries for
`observations` when `"observation"` is in `types`.

## Needed from Reflect — PLACEHOLDER(REFLECT)

1. **Mental model storage.** `types=["mental_model"]` is accepted, but the store has no table to
   read from. Recall needs the mental-model table's name and its text, embedding, `as_of`
   and validity columns.
2. **Staleness for mental models and observations.** Recall flags a memory as stale only from
   `valid_to`. If Reflect marks models stale through `change_event`, it needs to write a
   `valid_to` (or a `stale` column) that Recall can read.
3. **Thin-evidence threshold.** Reflect escalates when results are under 3, below `tau_thin`, or
   truncated. Recall returns `items[*].relevance` and `truncated`; Reflect owns `tau_thin`.

## Needs a live cloud run — PLACEHOLDER(CLOUD)

1. **`ai.rank()` call** (`make_alloydb_reranker`). Check the argument names, the
   `semantic-ranker-default-003` model ID, and whether `index` is 1-based. It needs the
   Discovery Engine API and the preview AI functions flag.
2. **`SCORE_FLOOR = 0.3`**. Calibrate on real `ai.rank()` scores from a small labelled set
   (questions that should and should not abstain).
3. **`RECENCY_HALF_LIFE_DAYS = 90`**. Tune once real fact ages exist.
4. **Token counting.** `estimate_tokens` uses about four characters per token. Pass
   `count_tokens=` with Gemini's `count_tokens` if exact budgets matter.
5. **Benchmarks** for the plan's "Tested" claims (temporal recall, rerank cut, packing accuracy).
   They are not reproduced here.

## Not yet wired

`adk_memory_provider.py` and `agent_orchestrator.py` still call the old
`retrieve_hybrid_entities`. Once a `RecallStore` implementation exists, replace their per-turn
context with `build_turn_context(...)`, pass the `round_id` from `SessionBuffer.append` to
Retain, and register `make_recall_tools(...)` on the agent.
