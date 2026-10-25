"""LongMemEval harness for Recall: ingest each question's chat history, recall, score.

    python -m bench.longmemeval data/longmemeval_oracle.json --limit 20
    python -m bench.longmemeval data/longmemeval_s_cleaned.json --embedder vertex --qa

Every question has its own haystack, ingested under its own Scope (user_id "lme:<question_id>")
through Retain's Retainer. Retain's Gemini stages are stubs, so ingestion uses benchmark stages:
VerbatimExtractor stores each 700-char chunk of each message as one fact (dated at its
session), with no entities or links. That measures Recall's search, fusion, reranking and
packing on raw conversation; the entity and graph channels stay empty until Retain's extractor,
resolver and linker are filled and passed in instead.

Retrieval is scored on Recall's packed items (what the agent would see): an item counts for the
session and round its source_episode_id came from. recall_any@k / recall_all@k / ndcg@k at
session level (answer_session_ids) and round level (rounds with a has_answer turn); abstention
questions are left out, as in LongMemEval. With --qa, Gemini answers from the packed context
and judges with LongMemEval's prompts (bench/qa.py).

Results go to a JSONL file (one line per question, resumable), a summary JSON and a
hypotheses JSONL in the format LongMemEval's own evaluate_qa.py reads.
"""

import argparse
import json
import logging
import math
import os
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from recall import InMemoryRecallStore, Recaller, RecallResult
from recall.config import RecallConfig, RecallDepth
from retain import InMemoryRetainStore, RetainConfig, Retainer, Scope
from retain.models import (
    Category,
    ClaimType,
    EntityResolution,
    ExtractedFact,
    FactType,
    Message,
    Speaker,
    Visibility,
)
from shared.chunking import chunk_spans

logger = logging.getLogger(__name__)

USER_PREFIX = "lme:"
_DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})(?:\s*\(\w+\))?(?:\s+(\d{1,2}):(\d{2}))?")


# LongMemEval records ----------------------------------------------------------------------


def parse_date(text: str) -> datetime:
    """LongMemEval dates look like "2023/05/20 (Sat) 02:21"; read as UTC."""
    match = _DATE_RE.search(text)
    if not match:
        raise ValueError(f"Unrecognised LongMemEval date {text!r}")
    year, month, day, hour, minute = (int(g) if g else 0 for g in match.groups())
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Round:
    """One user turn and the replies to it: one Retain episode."""

    episode_id: str
    session_id: str
    index: int
    messages: tuple[Message, ...]
    has_answer: bool


@dataclass(frozen=True)
class Instance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    question_time: datetime
    answer_session_ids: tuple[str, ...]
    rounds: tuple[Round, ...]

    @property
    def abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def scope(self) -> Scope:
        return Scope(USER_PREFIX + self.question_id)

    @property
    def answer_episode_ids(self) -> set[str]:
        return {r.episode_id for r in self.rounds if r.has_answer}

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "Instance":
        question_id = str(record["question_id"])
        rounds: list[Round] = []
        sessions = zip(
            record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"]
        )
        for session_id, date, turns in sessions:
            at = parse_date(date)
            grouped: list[list[dict]] = []
            for turn in turns:
                if turn.get("role") == "user" or not grouped:
                    grouped.append([])
                grouped[-1].append(turn)
            for index, group in enumerate(grouped):
                messages = tuple(
                    Message(
                        Speaker.USER if t.get("role") == "user" else Speaker.ASSISTANT,
                        str(t.get("content") or ""),
                        at,
                    )
                    for t in group
                )
                rounds.append(
                    Round(
                        # Includes the question id: document ids derive from episode ids and
                        # haystack sessions repeat across questions sharing one database.
                        episode_id=f"{question_id}:{session_id}:{index}",
                        session_id=str(session_id),
                        index=index,
                        messages=messages,
                        has_answer=any(t.get("has_answer") for t in group),
                    )
                )
        return cls(
            question_id=question_id,
            question_type=str(record["question_type"]),
            question=str(record["question"]),
            answer=str(record["answer"]),
            question_date=str(record["question_date"]),
            question_time=parse_date(record["question_date"]),
            answer_session_ids=tuple(str(s) for s in record.get("answer_session_ids") or ()),
            rounds=tuple(rounds),
        )


def load_instances(path: str) -> list[Instance]:
    with open(path, encoding="utf-8") as f:
        return [Instance.from_json(record) for record in json.load(f)]


# Benchmark Retain stages ------------------------------------------------------------------


class VerbatimExtractor:
    """FactExtractor baseline: every chunk becomes one fact, quoted verbatim and dated at its
    message, so each is an event at the session time (Memory.interval is then a point)."""

    def extract(self, chunks):
        facts = []
        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue
            speaker = chunk.document.speaker
            facts.append(
                ExtractedFact(
                    key=len(facts),
                    chunk_id=chunk.chunk_id,
                    text=text,
                    evidence=text,
                    fact_type=FactType.WORLD if speaker == Speaker.USER else FactType.EXPERIENCE,
                    category=Category.OTHER,
                    claim_type=ClaimType.ASSERTED,
                    visibility=Visibility.GLOBAL,
                    occurred_at=chunk.document.occurred_at,
                )
            )
        return facts


class NoEntities:
    def resolve(self, user_id, mentions):
        return EntityResolution({})


class NoLinks:
    def link(self, facts):
        return []


class _Clock:
    """Retainer's clock, set to each session's date so recorded_at is the ingestion time a
    live deployment would have seen."""

    def __init__(self):
        self.now = datetime.now(timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def chunk_texts(instance: Instance, max_chars: int) -> list[str]:
    """The fact texts VerbatimExtractor will produce, for warming the embedding cache in bulk."""
    texts = []
    for round_ in instance.rounds:
        for message in round_.messages:
            for start, end in chunk_spans(message.text, max_chars):
                if text := message.text[start:end].strip():
                    texts.append(text)
    return list(dict.fromkeys(texts))


# Metrics ----------------------------------------------------------------------------------


def ranked_unique(keys: Sequence[str | None]) -> list[str]:
    return [k for k in dict.fromkeys(keys) if k is not None]


def retrieval_metrics(ranked: Sequence[str], relevant: set[str], ks: Sequence[int]) -> dict:
    """recall_any@k (any relevant found), recall_all@k (all found), binary ndcg@k."""
    metrics = {}
    for k in ks:
        top = ranked[:k]
        found = relevant & set(top)
        dcg = sum(1 / math.log2(i + 2) for i, key in enumerate(top) if key in relevant)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        metrics[f"recall_any@{k}"] = float(bool(found))
        metrics[f"recall_all@{k}"] = float(found == relevant)
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
    return metrics


def summarize(records: Sequence[dict]) -> dict:
    """Means per question type and overall; QA also as the task-averaged accuracy."""

    def means(rows: Sequence[dict]) -> dict:
        out: dict[str, Any] = {"count": len(rows)}
        for level in ("session", "round"):
            scored = [r["retrieval"][level] for r in rows if r["retrieval"].get(level)]
            if scored:
                out[level] = {m: round(mean(s[m] for s in scored), 4) for m in scored[0]}
                out[level]["count"] = len(scored)
        labels = [r["qa"]["correct"] for r in rows if r.get("qa")]
        if labels:
            out["qa_accuracy"] = round(mean(labels), 4)
        out["recall_ms"] = round(mean(r["timing"]["recall_ms"] for r in rows), 1)
        out["abstained"] = round(mean(r["recall"]["abstained"] for r in rows), 4)
        return out

    by_type = defaultdict(list)
    for record in records:
        by_type[record["question_type"]].append(record)
    summary = {
        "overall": means(records),
        "by_type": {t: means(rows) for t, rows in sorted(by_type.items())},
    }
    abstention = [r for r in records if r["abstention"] and r.get("qa")]
    if abstention:
        summary["abstention_accuracy"] = round(mean(r["qa"]["correct"] for r in abstention), 4)
    per_type = [v["qa_accuracy"] for v in summary["by_type"].values() if "qa_accuracy" in v]
    if per_type:
        summary["qa_task_averaged"] = round(mean(per_type), 4)
    return summary


# Running ----------------------------------------------------------------------------------


@dataclass
class Harness:
    """Everything one run shares; stores are per question (memory) or shared (alloydb)."""

    doc_embedder: Callable
    query_embedder: Callable
    make_stores: Callable[[], tuple[Any, Any]]
    reranker: Callable | None = None
    generate: Callable[[str], str] | None = None  # query decomposition
    qa_generate: Callable[[str], str] | None = None  # answering and judging
    config: RecallConfig = RecallConfig()
    depth: RecallDepth = RecallDepth.MID
    max_tokens: int = 4096
    chunk_chars: int = 700
    decompose: bool = True
    ks: tuple[int, ...] = (1, 3, 5, 10)
    save_context: bool = False
    cleanup: Callable[[Instance], None] | None = None

    def run(self, instance: Instance) -> dict:
        retain_store, recall_store = self.make_stores()
        started = time.perf_counter()
        texts = chunk_texts(instance, self.chunk_chars)
        if texts:
            self.doc_embedder(texts)  # one batched pass; Retainer then hits the cache
        clock = _Clock()
        retainer = Retainer(
            store=retain_store,
            extractor=VerbatimExtractor(),
            entity_resolver=NoEntities(),
            embedder=self.doc_embedder,
            linker=NoLinks(),
            config=RetainConfig(max_chunk_chars=self.chunk_chars, document_ttl=None),
            clock=clock,
        )
        facts = 0
        for round_ in instance.rounds:
            clock.now = round_.messages[0].timestamp
            result = retainer.retain(instance.scope, round_.messages, episode_id=round_.episode_id)
            facts += len(result.batch.facts) if result.batch else 0
        ingest_ms = (time.perf_counter() - started) * 1000

        recaller = Recaller(
            store=recall_store,
            embedder=self.query_embedder,
            reranker=self.reranker,
            generate=self.generate,
            config=self.config,
            clock=lambda: instance.question_time,
        )
        started = time.perf_counter()
        result = recaller.recall(
            instance.scope,
            instance.question,
            query_time=instance.question_time,
            max_tokens=self.max_tokens,
            depth=self.depth,
            decompose=self.decompose,
        )
        recall_ms = (time.perf_counter() - started) * 1000

        record = {
            "question_id": instance.question_id,
            "question_type": instance.question_type,
            "abstention": instance.abstention,
            "rounds": len(instance.rounds),
            "facts": facts,
            "retrieval": self._retrieval(instance, result),
            "recall": self._recall_summary(result),
            "timing": {"ingest_ms": round(ingest_ms, 1), "recall_ms": round(recall_ms, 1)},
        }
        if self.qa_generate is not None:
            record["qa"] = self._qa(instance, result)
        if self.cleanup is not None:
            self.cleanup(instance)
        return record

    def _retrieval(self, instance: Instance, result: RecallResult) -> dict:
        if instance.abstention:
            return {}
        by_episode = {r.episode_id: r for r in instance.rounds}
        episodes = [item.memory.source_episode_id for item in result.items]
        sessions = [by_episode[e].session_id if e in by_episode else None for e in episodes]
        out = {}
        if instance.answer_session_ids:
            out["session"] = retrieval_metrics(
                ranked_unique(sessions), set(instance.answer_session_ids), self.ks
            )
        if instance.answer_episode_ids:
            out["round"] = retrieval_metrics(
                ranked_unique(episodes), instance.answer_episode_ids, self.ks
            )
        return out

    def _recall_summary(self, result: RecallResult) -> dict:
        summary = {
            "items": len(result.items),
            "episodes": ranked_unique([i.memory.source_episode_id for i in result.items]),
            "tokens_used": result.tokens_used,
            "truncated": result.truncated,
            "abstained": result.abstained,
            "sub_queries": result.sub_queries,
            "window": result.window.to_dict() if result.window else None,
            "stages": {s.stage: round(s.ms, 1) for s in result.trace},
        }
        if self.save_context:
            summary["context"] = result.context
        return summary

    def _qa(self, instance: Instance, result: RecallResult) -> dict:
        from bench.qa import answer, judge

        started = time.perf_counter()
        hypothesis = answer(
            self.qa_generate, result.context, instance.question, instance.question_date
        )
        correct = judge(
            self.qa_generate,
            instance.question_type,
            instance.question,
            instance.answer,
            hypothesis,
            instance.abstention,
        )
        return {
            "hypothesis": hypothesis,
            "correct": correct,
            "ms": round((time.perf_counter() - started) * 1000, 1),
        }


def run_all(
    harness: Harness, instances: Sequence[Instance], out_path: Path, workers: int = 1
) -> dict:
    """Runs every instance not already in out_path, appending one JSON line each; then writes
    the summary and hypotheses files next to it and returns the summary."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    done[record["question_id"]] = record
    todo = [i for i in instances if i.question_id not in done]
    logger.info("%d questions to run, %d already in %s", len(todo), len(done), out_path)

    lock = threading.Lock()
    with open(out_path, "a", encoding="utf-8") as out:

        def finish(record: dict) -> None:
            with lock:
                done[record["question_id"]] = record
                out.write(json.dumps(record) + "\n")
                out.flush()
                logger.info(
                    "[%d/%d] %s %s", len(done), len(instances), record["question_id"],
                    _brief(record),
                )  # fmt: skip

        if workers <= 1:
            for instance in todo:
                finish(harness.run(instance))
        else:
            with ThreadPoolExecutor(workers) as pool:
                futures = {pool.submit(harness.run, i): i.question_id for i in todo}
                for future in as_completed(futures):
                    try:
                        finish(future.result())
                    except Exception:
                        logger.exception("Question %s failed; rerun to retry", futures[future])

    wanted = {i.question_id for i in instances}
    records = [r for qid, r in done.items() if qid in wanted]
    summary = summarize(records) if records else {}
    stem = out_path.with_suffix("")
    Path(f"{stem}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if any(r.get("qa") for r in records):
        with open(f"{stem}.hypotheses.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                if r.get("qa"):
                    line = {"question_id": r["question_id"], "hypothesis": r["qa"]["hypothesis"]}
                    f.write(json.dumps(line) + "\n")
    return summary


def _brief(record: dict) -> str:
    parts = []
    session = record["retrieval"].get("session")
    if session:
        parts.append(f"session recall_any@5={session.get('recall_any@5', 0):.0f}")
    if record.get("qa"):
        parts.append(f"qa={'ok' if record['qa']['correct'] else 'wrong'}")
    parts.append(f"recall {record['timing']['recall_ms']:.0f}ms")
    return ", ".join(parts)


# CLI --------------------------------------------------------------------------------------


def _genai_client(location_env: str, default_location: str):
    from google import genai

    project = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv(location_env, default_location)
    return genai.Client(vertexai=True, project=project, location=location)


def build_harness(args) -> Harness:
    from bench.embedders import CachedEmbedder, HashingEmbedder, VertexEmbedder
    from shared.llm import GeminiGenerator, generate_with_retry

    if args.embedder == "vertex":
        client = _genai_client("REGION", "us-east1")  # text-embedding-005 is regional
        Path(args.embed_cache).parent.mkdir(parents=True, exist_ok=True)
        doc_embedder = CachedEmbedder(
            VertexEmbedder(client, task_type="RETRIEVAL_DOCUMENT"),
            args.embed_cache,
            "text-embedding-005:document",
        )
        query_embedder = CachedEmbedder(
            VertexEmbedder(client, task_type="RETRIEVAL_QUERY"),
            args.embed_cache,
            "text-embedding-005:query",
        )
    else:
        doc_embedder = query_embedder = HashingEmbedder()

    generate = qa_generate = None
    if args.decompose or args.qa:
        client = _genai_client("GENAI_LOCATION", "us")
        model = GeminiGenerator(client, model=args.gemini_model)

        def generate(prompt):
            return generate_with_retry(model, prompt)

        qa_generate = generate if args.qa else None

    conn_factory = None
    if args.backend == "alloydb" or args.reranker == "alloydb":
        from db_clients import get_db_connection

        conn_factory = get_db_connection

    cleanup = None
    if args.backend == "alloydb":
        from bench.alloydb import BenchAlloyDBWriter
        from recall.alloydb_store import AlloyDBRecallStore

        writer, reader = BenchAlloyDBWriter(conn_factory), AlloyDBRecallStore(conn_factory)

        def make_stores():
            return writer, reader

        if args.cleanup:

            def cleanup(instance):
                writer.delete_users(instance.scope.user_id)
    else:

        def make_stores():
            retain_store = InMemoryRetainStore()
            return retain_store, InMemoryRecallStore(retain_store)

    reranker = None
    if args.reranker == "alloydb":
        from recall.reranker import AlloyDBReranker

        reranker = AlloyDBReranker(conn_factory)

    return Harness(
        doc_embedder=doc_embedder,
        query_embedder=query_embedder,
        make_stores=make_stores,
        reranker=reranker,
        generate=generate if args.decompose else None,
        qa_generate=qa_generate,
        depth=RecallDepth.parse(args.depth),
        max_tokens=args.max_tokens,
        chunk_chars=args.chunk_chars,
        decompose=args.decompose,
        ks=tuple(int(k) for k in args.ks.split(",")),
        save_context=args.save_context,
        cleanup=cleanup,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m bench.longmemeval", description=__doc__.split("\n")[0]
    )
    parser.add_argument("data", help="LongMemEval JSON (oracle, s_cleaned or m_cleaned)")
    parser.add_argument("--out", help="results JSONL (default bench_results/<data>-<time>.jsonl)")
    parser.add_argument("--backend", choices=["memory", "alloydb"], default="memory")
    parser.add_argument("--embedder", choices=["hash", "vertex"], default="hash")
    parser.add_argument("--embed-cache", default=".bench_cache/embeddings.sqlite")
    parser.add_argument("--reranker", choices=["none", "alloydb"], default="none")
    parser.add_argument("--decompose", action="store_true", help="LLM query analysis (Gemini)")
    parser.add_argument("--qa", action="store_true", help="answer and judge with Gemini")
    parser.add_argument("--gemini-model", default=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    parser.add_argument("--depth", choices=[d.value for d in RecallDepth], default="mid")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--chunk-chars", type=int, default=700)
    parser.add_argument("--ks", default="1,3,5,10")
    parser.add_argument("--limit", type=int, help="first N questions after filtering")
    parser.add_argument("--question-types", help="comma-separated question_type filter")
    parser.add_argument("--ids", help="comma-separated question_id filter")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-context", action="store_true", help="store packed context")
    parser.add_argument(
        "--cleanup", action="store_true", help="alloydb: delete each question's rows after it"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    instances = load_instances(args.data)
    if args.question_types:
        wanted = set(args.question_types.split(","))
        instances = [i for i in instances if i.question_type in wanted]
    if args.ids:
        wanted = set(args.ids.split(","))
        instances = [i for i in instances if i.question_id in wanted]
    if args.limit:
        instances = instances[: args.limit]
    out = Path(
        args.out or f"bench_results/{Path(args.data).stem}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    )
    summary = run_all(build_harness(args), instances, out, args.workers)
    print(json.dumps(summary, indent=2))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
