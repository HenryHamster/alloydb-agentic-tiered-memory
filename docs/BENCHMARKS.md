# Benchmarks: LongMemEval for Recall

`bench/longmemeval.py` runs [LongMemEval](https://github.com/xiaowu0162/LongMemEval) against
Recall. For each question it ingests that question's chat history, runs Recall on the
question at the question's date, and scores what Recall returned. With `--qa` it also answers
from the packed context and judges the answer.

## Data

Download from the Hugging Face dataset `xiaowu0162/longmemeval-cleaned` into `data/`:

| File | Haystack per question | Use |
| --- | --- | --- |
| `longmemeval_oracle.json` | Evidence sessions only | Quick checks; session-level retrieval is near trivial |
| `longmemeval_s_cleaned.json` | About 50 sessions (~115k tokens) | The main retrieval benchmark |
| `longmemeval_m_cleaned.json` | About 500 sessions | Scale test; slow on the in-memory backend |

## Running

```bash
# Offline smoke run: in-memory stores, hashing embedder, no cloud calls.
python -m bench.longmemeval data/longmemeval_oracle.json --limit 50

# Real retrieval numbers: text-embedding-005 vectors (cached on disk), in-memory stores.
python -m bench.longmemeval data/longmemeval_s_cleaned.json --embedder vertex --workers 4

# Full pipeline: LLM query analysis, ai.rank() reranking, answers judged by Gemini.
python -m bench.longmemeval data/longmemeval_s_cleaned.json --embedder vertex \
    --decompose --reranker alloydb --qa --workers 4

# Against the AlloyDB tables (run sql/004_retain_schema.sql first).
python -m bench.longmemeval data/longmemeval_s_cleaned.json --embedder vertex \
    --backend alloydb --cleanup
```

Cloud options read the same environment as the demos (`PROJECT_ID`, `REGION` for embeddings,
`GENAI_LOCATION` for Gemini, `DB_*` for AlloyDB; see [setup](SETUP.md)). Other options:
`--depth low|mid|high` (default `mid`), `--max-tokens` (4096), `--chunk-chars` (700),
`--question-types`, `--ids`, `--limit`, `--save-context`, `--gemini-model`.

## What is measured

- **Ingestion.** Each round (a user turn and its replies) is one Retain episode, stored
  through Retain's `Retainer` under scope `lme:<question_id>`. Retain's Gemini stages are
  stubs, so the benchmark plugs in `VerbatimExtractor`: every 700-character chunk of every
  message becomes one fact, dated at its session. No entities or links are created, so Recall's
  entity and graph channels are idle. These numbers measure Recall's vector, keyword and temporal
  search, fusion, reranking and packing on raw conversation. Once Retain's extractor, resolver
  and linker exist, pass them to `Harness` instead to measure the full system.
- **Retrieval** is scored on Recall's packed items, which are what the agent sees, so `--max-tokens`
  and `--depth` affect it. Each item counts for the session and round its
  `source_episode_id` came from. The harness reports `recall_any@k`, `recall_all@k` and `ndcg@k`
  at session level (`answer_session_ids`) and round level (rounds with a `has_answer` turn).
  Abstention questions (`*_abs`) are excluded, as in LongMemEval.
- **QA** (`--qa`). Gemini answers from the packed context, then judges the answer with
  LongMemEval's per-type prompts (`bench/qa.py`). The paper's judge is GPT-4o, so to compare with
  published numbers, run LongMemEval's `evaluate_qa.py` on the hypotheses file.

## Outputs

`--out` defaults to `bench_results/<data>-<time>.jsonl` (git-ignored):

- `<run>.jsonl` has one line per question: retrieval metrics, returned episodes, sub-queries, window,
  per-stage timings and, with `--qa`, the hypothesis and verdict. Rerunning with the same
  `--out` skips finished questions, so an interrupted run resumes where it stopped.
- `<run>.summary.json` has means overall and per question type, plus `qa_task_averaged` and
  `abstention_accuracy` with `--qa`.
- `<run>.hypotheses.jsonl` has `{"question_id", "hypothesis"}` lines for LongMemEval's evaluator.

Embeddings are cached in `.bench_cache/embeddings.sqlite`, keyed by model, task type and text.
Reruns and haystack sessions shared between questions are embedded once.

## Caveats

- The hashing embedder exists for smoke tests only. Do not report its numbers.
- `--backend alloydb` writes with `bench/alloydb.py` (documents and facts only) because
  `AlloyDBRetainStore.write` is still Retain's stub. Without `--cleanup`, rows stay and later
  runs skip re-ingesting them (the `retained_at` ledger).
- LongMemEval dates have no time zone and are read as UTC.
