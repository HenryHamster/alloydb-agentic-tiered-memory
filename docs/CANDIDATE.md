# Integrated candidate

`tiered_memory.build_memory_system` connects the production Retain stages, AlloyDB
Recall, Reflect and Valkey. Gemini generation uses the global endpoint and
text-embedding-005 uses a regional endpoint. Apply schemas 004 and 005. Enable
Discovery Engine and grant the AlloyDB service agent `roles/discoveryengine.viewer`
for `ai.rank`; the database AI query engine flag must already be enabled.

Retain commits synchronously. Observations and Reflect run once at an explicit
session boundary. A scheduled job can call `maintain`; no scheduler is created.
Recall searches hot rows first, expands through warm/archive until three reranked
hits survive the score floor, then packs context and records access. Without a
reranker it searches all tiers. Consolidated originals are hidden; unresolved
conflicts carry identifiers and confidence. A conflict's counterpart is not
guaranteed to fit the context budget. Macro memories share the existing search,
ranking and packing path and expose their supporting facts. Strict mode propagates
provider/storage failures. The original demos retain their existing behavior.

The candidate benchmark is `python -m bench.mem2act --help` from the repository.
It consumes the same gold-free cases as the original pilot, preserves every source
event in ordered JSON, isolates questions, uses a fresh query session, and retains
the generated response. It emits JSON tool proposals; tools are not executed.
Source dates in this dataset are unknown. Extraction receives null source dates;
processing timestamps are bookkeeping, not asserted historical dates.

The serial ledger reserves before each paid request and includes failed/uncertain
calls, schema input, all thinking tokens and SQL ranking retries. Generation rates
verified 2026-10-26: global Gemini 3.5 Flash USD 1.50/M input and USD 9/M output,
including reasoning. Embeddings conservatively reserve USD .25/M UTF-8 bytes.
Ranking reserves USD .001 per 100 documents or fraction. Infrastructure is reserved
separately and requires an external runtime/shutdown bound. The ledger is an
application spending guard, not a Google Cloud billing cap.

Observation summaries must cover and cite every fact in their input group, or
return an empty synthesis. Validation failures receive one corrected retry before
propagating. This matches the existing all-facts provenance contract.

`--resume-history` is an explicit recovery option after maintenance failure. It
verifies every source message against saved receipts and committed database
documents before reusing a complete history. It rejects input/namespace changes
and changes to ingestion code; an observation-builder repair is allowed. Original
source/status and paid-call receipts are preserved, the ledger is never reset,
and recovery records distinguish previously retained history from new work.

Sources: [generation pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing),
[ranking pricing](https://cloud.google.com/generative-ai-app-builder/pricing),
[AlloyDB prerequisites](https://docs.cloud.google.com/alloydb/docs/ai/rank-rerank-search-results-rag).

The pilot's canonical records live in the adjacent benchmark-kit workspace:
`experiments/candidate-mem2act-001/RECORD.md`. Its experiment registry preserves the
earlier original-codelab evidence and scoring implementation. No accuracy result
is inferred from offline checks.
