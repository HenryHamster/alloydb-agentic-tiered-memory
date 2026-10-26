# Reflect: design and status

Reflect is the background maintenance path. It runs off the request path (new facts, session
end or a schedule) and does three things per user in one transaction:

1. **Consolidate and resolve.** It merges redundant facts and settles state changes. Conflicts
   it cannot settle are flagged.
2. **Synthesize** macro memories from clusters of linked facts.
3. **Re-rank and re-tier** every fact and macro (hot, warm, archived, purged).

Every stage is implemented and covered offline by `tests/test_reflect.py` (in-memory store plus a
recording fake connection for the AlloyDB store). The AlloyDB store has also been run end to end
against PostgreSQL 16 with the 004 + 005 schema (pgvector stubbed); it has not yet run on AlloyDB
itself. Recall does not call it yet (see "Recall integration boundary").

## Pipeline

```
Reflector.reflect(user_id, trigger)            trigger: new_facts | session_end | scheduled
  read      user_facts, fact_states, links_among, entity_rows, user_macros
  1 find_candidates   similar links >= 0.85, same subject entity + category, same domain
    judge(...)        injected callable labels each group: distinct / redundant / state_change / conflict
    resolve           code applies the verdict (the model never picks the winner)
  2 rank_score / next_tier / apply_hot_cap     facts; purged facts leave the view
  3 build_clusters    temporal + causal + strong semantic links, shared subject/object entities
    MacroWriter       LLM title + summary, only when the cluster's evidence changed
    Embedder          embed(title + summary) for every regenerated macro
    rank_score / next_tier / apply_hot_cap     macros, same lifecycle
  write     ReflectStore.write_reflect(ReflectBatch)   one transaction
Reflector.forget(scope, fact_ids)              explicit delete: purge now, drop dependent macros
ReflectStore.record_access(...)                called by Recall; archived hit -> promoted back
Recall retrieval                              existing hybrid search owns search policy
```

## Files

| File | Contents | Status |
| --- | --- | --- |
| `sql/005_reflect_schema.sql` | lifecycle columns on `memory_facts`, link relations, `memory_conflict_reviews`, `macro_memories` | Complete |
| `reflect/models.py` | `Tier`, `Verdict`, `FactState`, `MacroMemory`, `Cluster`, `ReflectBatch` | Complete |
| `reflect/tiers.py` | `TierConfig`; `rank_score`, `next_tier`, `apply_hot_cap`, `promote_on_hit` | Complete |
| `reflect/consolidation.py` | `find_candidates`, `resolve`, `consolidate`, `LLMConflictJudge` | Complete |
| `reflect/macros.py` | `build_clusters`, `causal_chains`, `temporal_span`, `match_existing`, `LLMMacroWriter` | Complete |
| `reflect/graph.py` | union-find `components` shared by candidates and clusters | Complete |
| `reflect/generation.py` | `StructuredGenerator`: injected prompt/schema -> parsed JSON callable | Complete |
| `reflect/pipeline.py` | `Reflector.reflect` / `forget`; `build_reflector` (production wiring) | Complete |
| `reflect/store.py` | `ReflectStore` protocol, `InMemoryReflectStore`, `AlloyDBReflectStore` | Complete |
| `retain/models.py` | `Relation.CONSOLIDATED_INTO`, `Relation.CONFLICTS_WITH` | Added |

## Model adapters

Reflect does not select a model or depend on a provider SDK. `Reflector` accepts a conflict
judge callable, a `MacroWriter`, and an `Embedder`. The AlloyDB factory wraps a caller-provided
`generate(prompt, response_schema)` function with `LLMConflictJudge` and `LLMMacroWriter`.
The function returns a parsed JSON object; its adapter owns SDK calls, model selection,
authentication and retries. Stages still need to validate the output against their schema.

```python
reflector = build_reflector(
    get_db_connection,
    embedder=embed_texts,  # texts -> vectors; must match the schema's 768 dimensions
    generate=generate_json,  # (prompt, response_schema) -> parsed JSON object
)
```

These adapters can wrap any compatible model or offline fake. Both LLM stages validate the
returned JSON before using it: a malformed answer, an unknown verdict or a fact outside its group
raises, and the pass writes nothing.

## Recall integration boundary

Reflect owns background scoring, tier transitions and access metadata. Search execution stays
in the existing Recall path (`hybrid_retriever.py`); there is no separate Reflect search loop.
The tier search policy below is a proposed Recall integration, not implemented behavior.
Currently that retriever queries `agent_entities`, not `memory_facts` or `macro_memories`.
A follow-up must add those tables to the existing retrieval path, enforce their visibility
rules, and record accesses. Removing the duplicate wrapper does not provide that integration.

## Per-row fields (`memory_facts`, and the same columns on `macro_memories`)

| Field | Meaning |
| --- | --- |
| `access_count`, `last_accessed_at` | Bumped by Recall for every row it returns (`record_access`). |
| `rank_score` | `confidence * (log1p(access) * decay(last access) + 0.5 * log1p(evidence - 1) + decay(age))`. Half-lives are 14 days for access and 3 days for freshness. |
| `tier` | `hot` / `warm` / `archived`. A purged row is deleted, not marked. |
| `confidence` | 1.0 by default. It is halved for each side of an unresolved conflict. |
| `consolidated_into` | Set on a redundant original. Points to the merged fact. |
| `conflicts_with` | The other side(s) of an unresolved conflict. |
| `valid_from`, `valid_to`, `superseded_by` | Already on the table from Retain. A state change closes the old fact. |

**"Top of the table" is an index, not physical row order:** `(user_id, tier, rank_score DESC)`.

## Tiers

| Tier | Searched | Enter | Leave |
| --- | --- | --- | --- |
| Hot | every recall | score >= 1.0 (new facts start at 1.0) | score < 0.6, or pushed out by the cap of 200 per user |
| Warm | only when hot is weak | demoted from hot | score >= 1.0 -> hot; score < 0.2 **and** no hit for 30 days -> archived |
| Archived | last fallback | cold warm rows, superseded or merged facts, closed macros | a hit promotes it now; score >= 0.35 -> warm; 365 days dormant -> purged |
| Purged | never | long dormancy or `forget()` | row deleted; links and entity rows cascade |

- **Hysteresis:** a row is promoted at 1.0 but demoted only below 0.6. A score between the two
  keeps the current tier, so rows don't bounce between tiers.
- **Small tables:** `TierConfig(use_warm=False)` keeps only hot and archived. Turn warm on later
  without a migration.
- **History and merged originals never take hot slots.** These are superseded facts, closed
  macros and `consolidated_into` rows. They go straight to archived.

## Redundancy and conflicts

Candidates are groups of facts that meet all of these:

- current
- not consolidated
- in the same visibility domain
- with the same subject entity and the same category (the "attribute")
- joined by Retain's `similar` links at >= 0.85

The LLM judge labels each group. The code decides everything else:

| Verdict | Write |
| --- | --- |
| `redundant` | Writes one merged fact (the LLM provides its text). It sums `access_count` and `evidence_count`, keeps the earliest `valid_from` and copies the entity rows. The originals get `consolidated_into` and are archived, with a `consolidated_into` link. |
| `state_change` | The newer `valid_from` wins. The old fact gets `valid_to = new.valid_from` and `superseded_by = new`, plus a `supersedes` link. The old fact is kept for history. |
| `conflict` | Keeps both facts. Sets `conflicts_with` on each, multiplies confidence by 0.5, adds a `conflicts_with` link and queues a row in `memory_conflict_reviews`. Recall returns both, with timestamps. |
| `distinct` | Nothing is written. |

The judge's output is untrusted. A judgment that names facts outside a candidate group fails
the whole pass. A `state_change` whose facts share one `valid_from` is downgraded to
`conflict`, because there is no time order to pick a winner from. A pair that is already
flagged is not judged again.

## MacroMemories

The macro layer is the proposed asynchronous synthesis layer. Recall should search
`macro_memories` alongside facts (HNSW on `macro_embedding`, plus full-text search on title +
summary). That integration remains to be implemented in the existing retrieval path. A matching
macro can supply precomputed context; generating a tailored answer may still require an LLM.

| Field | Source |
| --- | --- |
| `macro_id` | `stable_id("macro", user, visibility, *fact_ids)` at creation. It is kept across versions. |
| `macro_title`, `macro_summary` | `MacroWriter` (LLM) |
| `macro_embedding` | `embed(title + "\n" + summary)`. It is regenerated on every new version. |
| `supporting_fact_ids` | every fact in the cluster (all from/to endpoints used) |
| `primary_entity_ids` | subject/object entities that joined the cluster |
| `causal_chain_ids` | maximal cause -> effect paths over `caused_by` links |
| `temporal_span` | min/max time of the facts touching a temporal link (all facts if none do) |
| `version` | 1 for a new macro, +1 when the cluster's evidence set changes |
| `access_count`, `last_accessed_at`, `rank_score`, `tier` | same lifecycle as facts |

Rules:

- **Clusters.** A cluster is a connected component of at least 3 facts. Facts join through
  temporal links (`near_in_time`, `supersedes`), `caused_by` links, `similar` links at >= 0.8,
  and shared subject/object entities.
- **Hub entities are ignored.** Any entity on more than 25 facts (for example "user") is left
  out, or it would merge everything into one cluster.
- **Superseded facts stay in a cluster.** The history is the story: "2 hours" ->
  DuckDB -> "10 minutes".
- **A cluster never spans visibility domains.** A macro is therefore readable exactly where all
  of its evidence is readable.
- **Matching.** A cluster maps to the existing macro it overlaps most, with Jaccard >= 0.5.
- **Regeneration.** A macro is regenerated only when its evidence set changed. A macro whose
  cluster dissolved is closed (`valid_to`) and archived.
- **Example.** The DuckDB example, once implemented: S1 "takes 2 hours", S3 "DuckDB replaces
  Pandas", S4 "now under 10 minutes". S4 is `caused_by` S3, and S4 `supersedes` S1. Together
  they produce one macro with causal chain `(S3, S4)` and span S1..S4.

### Relation to `observations`

Retain's `observations` are single-entity summaries. Macros are multi-fact narratives across
sessions, built from link paths. Both stay for now; observations can later become macros with
one entity.

## Intended benefits and trade-offs

This comparison is with Hindsight's published design (retain / recall / reflect over world,
experience, opinion and observation memories). Hindsight's reflect is an agentic reasoning
loop that runs **at query time**. Nothing here has been benchmarked, so these are design
arguments, not measured results.

1. **No LLM loop at query time.**
   - **This design:** synthesis happens in the background. Recall gets precomputed macros
     through one vector/full-text lookup.
   - **Hindsight:** high-level questions cost a multi-step LLM call on every ask.
   - **Effect:** for recurring high-level questions, latency and cost move off the
     conversation's critical path.
2. **Smaller active working set.**
   - **This design:** Hot has a proposed cap. Recall can use tier metadata once its
     integration is implemented; no search-cost bound has been measured yet.
   - **Hindsight:** has no access-driven lifecycle. Every retained fact stays equally
     searchable forever.
3. **Usage feedback.** `access_count` and `last_accessed_at` feed `rank_score`, so what the
   agent actually recalls stays hot. Hindsight ranks by retrieval signals per query, not by
   long-term usage.
4. **Non-destructive, explicit conflict handling.** Each case writes a different marker:
   - redundant -> `consolidated_into`
   - state change -> `superseded_by` / `valid_to`
   - unresolvable -> `conflicts_with`, lower confidence and a review queue

   No case deletes data. An unresolvable conflict is surfaced with both timestamps; it is
   never decided silently by a model. The model labels, and code applies a fixed time rule.
5. **Causal and temporal narratives, not only entity summaries.** Macros follow `caused_by`
   and temporal paths across sessions ("switched to DuckDB, so the runtime fell from 2 h to
   10 min"). Entity-centric observations miss this kind of cause-and-effect story.
6. **One transactional store.**
   - **This design:** facts, links, entities, tiers and macros live in AlloyDB, are written in
     one transaction per pass, and are scoped by the same visibility rule as Recall. A macro
     never mixes project/session/global evidence.
   - **Hindsight:** this guarantee does not come for free from a separate memory service.

**Trade-offs we accept:**

- Macros are only as fresh as the last pass.
- A new question that no cluster covers still needs a reasoning step at query time. Keep a
  runtime reflect as the fallback.
- The thresholds (0.85 similarity, 1.0 / 0.6 hysteresis, 30 / 365 days) need tuning on real
  traffic.

## Implementation notes

Decisions made while filling the stubs, beyond what the specs above say:

- **`ReflectBatch.user_id`.** The batch names the user the pass ran for. The AlloyDB store takes
  its advisory lock on it and rejects any fact, macro or review of another user.
- **Recall's counters win.** `write_reflect` never overwrites `access_count` or
  `last_accessed_at` of existing facts or macros, so hits recorded during a pass are kept.
  (A tier promoted by such a hit can still be overwritten by the pass; the next hit fixes it.)
- **Merged facts stand in for their originals in clusters.** Links and entity rows on a
  consolidated original are redirected to its merged fact, so a merge does not break a story:
  in the DuckDB example the causal chain becomes (merged, S4).
- **Facts are tiered before macros are built,** so no macro is built on a fact purged in the
  same pass.
- **A macro's `updated_at` changes only when it is regenerated.** It is the macro's freshness
  clock, so an unchanged macro keeps cooling. A rewrite without an embedding keeps the stored
  one, so `user_macros` does not read vectors.
- **`user_facts` skips embeddings.** Reflect finds candidates through links, so it never needs
  the 768-float column for a whole user.
- **A judgment that reuses a fact is skipped** (the first wins) instead of failing the pass.
  A fact with several subject entities joins only one candidate group.
- **`forget()` also purges merged facts built from a forgotten fact,** since they carry its
  text.
- **A closed macro whose exact evidence set comes back is reopened** under the same `macro_id`
  without a model call.

## Open questions

- Should Retain's `VISIBLE_SQL` hide `consolidated_into` rows, so that the supersede
  classifier targets only the merged fact?
- Should `forget()` also delete the source `documents` rows behind a fact (right-to-erasure),
  not only the fact?
- Should clusters that cross domains produce a macro at the narrowest visibility, instead of
  being split?

## Next

1. Run `AlloyDBReflectStore` against a live AlloyDB instance (it has run on PostgreSQL 16).
2. Write a `generate(prompt, response_schema)` adapter (for example Gemini structured output)
   and call `build_reflector(get_db_connection, embedder=..., generate=...)` from the
   application composition layer.
3. Call `.reflect(user_id, trigger)` from `AsyncMemoryWorker` at session end, and add a
   scheduled job for the daily pass.
4. Extend the existing Recall retrieval path to search facts/macros by tier and call
   `record_access`; do not add a separate search loop.
5. Tune the thresholds on real traffic.
