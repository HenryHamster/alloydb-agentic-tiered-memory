"""Paired Reflect/Dream diagnostics using the existing LongMemEval ingestion and Recall.

Default: offline plumbing smoke, NOT a model-quality benchmark. An optional
module:factory adapter returns (structured_generator, embedder). A separate
retain adapter can return the production stages(store) callable. Provider
adapters own credentials, pricing and budget enforcement; none are selected here.
"""

import argparse
import gzip
import importlib
import json
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from bench.embedders import HashingEmbedder
from bench.longmemeval import Instance, _Clock, load_instances, verbatim_stages
from recall import InMemoryRecallStore, Recaller
from recall.config import MemoryType
from reflect import InMemoryReflectStore, Reflector
from reflect.consolidation import LLMConflictJudge
from reflect.dream import Dreamer, DreamRequest, digest, encoded, fact_from_payload, fact_payload
from reflect.dream_recall import DreamRecallStore
from reflect.dream_store import InMemoryDreamStore
from reflect.macros import LLMMacroWriter
from retain import InMemoryRetainStore, RetainConfig, Retainer
from retain.models import (
    Document,
    Entity,
    EntityRelation,
    EntityRole,
    EntityType,
    FactEntity,
    MemoryLink,
    Observation,
    Relation,
    Scope,
    Speaker,
    Visibility,
)

CASES = (
    "031748ae",
    "07741c44",
    "07741c45",
    "f685340e",
    "6071bd76",
    "gpt4_ab202e7f",
    "gpt4_7fce9456",
    "gpt4_d9af6064",
    "1da05512",
    "195a1a1b",
    "505af2f5",
    "031748ae_abs",
    "f685340e_abs",
    "e47becba",
)


def load_data(path, ids):
    """Official JSON or the existing AMB cached LongMemEval-s directory.

    The cache contains session-level gold labels but not turn-level has_answer.
    Preserve that distinction rather than manufacturing turn-level labels.
    """
    path = Path(path)
    if path.is_file():
        return [i for i in load_instances(str(path)) if i.question_id in ids]
    with gzip.open(path / "queries.json.gz", "rt", encoding="utf-8") as stream:
        queries = [q for q in json.load(stream) if q["id"] in ids]
    users = {q["user_id"] for q in queries}
    with gzip.open(path / "documents.json.gz", "rt", encoding="utf-8") as stream:
        docs = [d for d in json.load(stream) if d["user_id"] in users]
    by_user = defaultdict(list)
    for doc in docs:
        by_user[doc["user_id"]].append(doc)
    out = []
    for q in queries:
        sessions = sorted(by_user[q["user_id"]], key=lambda d: (d["timestamp"], d["id"]))

        def date(text):
            return datetime.fromisoformat(text).strftime("%Y/%m/%d %H:%M")

        out.append(
            Instance.from_json(
                {
                    "question_id": q["id"],
                    "question_type": q["meta"]["question_type"],
                    "question": q["query"],
                    "answer": "\n".join(q["gold_answers"]),
                    "question_date": date(q["meta"]["query_timestamp"]),
                    "answer_session_ids": q["gold_ids"],
                    "haystack_session_ids": [d["id"] for d in sessions],
                    "haystack_dates": [date(d["timestamp"]) for d in sessions],
                    "haystack_sessions": [json.loads(d["content"]) for d in sessions],
                }
            )
        )
    return out


class SmokeGenerator:
    """Extractive, question-blind fixtures. Exercises plumbing, not semantic quality."""

    def __call__(self, prompt, schema):
        data = json.loads(prompt.split("\n", 2)[-1])
        if prompt.startswith("DREAM_RECOVER"):
            return {
                "recoveries": [
                    {
                        "document_id": d["document_id"],
                        "quote": d["text"].split(". ")[0],
                        "claim_type": "asserted",
                    }
                    for d in data["documents"]
                    if d["speaker"] == "user" and d["text"].strip()
                ]
            }
        # This deliberately produces no invented semantic lesson.
        return {"insights": []}


def clone_store(source):
    target = InMemoryRetainStore()
    for name in (
        "documents",
        "retained",
        "facts",
        "entities",
        "fact_entities",
        "entity_relations",
        "links",
        "observations",
    ):
        setattr(target, name, deepcopy(getattr(source, name)))
    return target


def frozen_payload(instance, store, dropped, drop_every):
    return {
        "schema_version": 1,
        "question_id": instance.question_id,
        "history_sha256": digest(instance.sessions),
        "drop_every": drop_every,
        "dropped_fact_ids": dropped,
        **{
            name: list(getattr(store, name).values())
            for name in ("documents", "facts", "entities", "links", "observations")
        },
        "fact_entities": sorted(
            store.fact_entities, key=lambda r: (r.fact_id, r.entity_id, r.role)
        ),
        "entity_relations": sorted(store.entity_relations, key=lambda r: (r.fact_id, r.predicate)),
        "retained": sorted(store.retained),
    }


def load_frozen(path, instance, drop_every):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        data.get("schema_version") != 1
        or data["question_id"] != instance.question_id
        or data["history_sha256"] != digest(instance.sessions)
        or data["drop_every"] != drop_every
    ):
        raise ValueError("Frozen snapshot does not match this history or omission configuration")
    store = InMemoryRetainStore()
    for d in data["documents"]:
        d["scope"], d["speaker"] = Scope(**d["scope"]), Speaker(d["speaker"])
        for field in ("recorded_at", "occurred_at", "expires_at"):
            d[field] = datetime.fromisoformat(d[field]) if d[field] else None
        doc = Document(**d)
        store.documents[doc.document_id] = doc
    store.facts = {f["fact_id"]: fact_from_payload(f) for f in data["facts"]}
    for e in data["entities"]:
        e["entity_type"], e["aliases"] = EntityType(e["entity_type"]), tuple(e["aliases"])
        entity = Entity(**e)
        store.entities[entity.entity_id] = entity
    store.fact_entities = {
        FactEntity(r["fact_id"], r["entity_id"], EntityRole(r["role"]))
        for r in data["fact_entities"]
    }
    store.entity_relations = {EntityRelation(**r) for r in data["entity_relations"]}
    for link in data["links"]:
        link["relation"] = Relation(link["relation"])
        row = MemoryLink(**link)
        store.links[(row.from_fact_id, row.to_fact_id, row.relation)] = row
    for o in data["observations"]:
        o["scope"], o["visibility"] = Scope(**o["scope"]), Visibility(o["visibility"])
        o["fact_ids"] = tuple(o["fact_ids"])
        for field in ("valid_from", "valid_to"):
            o[field] = datetime.fromisoformat(o[field]) if o[field] else None
        if o.get("embedding") is not None:
            o["embedding"] = tuple(o["embedding"])
        obs = Observation(**o)
        store.observations[obs.observation_id] = obs
    store.retained = set(data["retained"])
    return store, data["dropped_fact_ids"]


def prepare(instance, embedder, stages, drop_every):
    store, clock = InMemoryRetainStore(), _Clock()
    retainer = Retainer(
        store=store,
        embedder=embedder,
        clock=clock,
        config=RetainConfig(document_ttl=None, maintenance_every=0),
        **stages(store),
    )
    for round_ in sorted(instance.sessions, key=lambda r: (r.messages[0].timestamp, r.episode_id)):
        if round_.messages[0].timestamp > instance.question_time:
            raise ValueError("History contains a session later than the evaluation question")
        clock.now = round_.messages[0].timestamp
        retainer.retain(
            Scope(instance.scope.user_id, "longmemeval", round_.session_id),
            round_.messages,
            episode_id=round_.episode_id,
        )
    # Explicit fault injection: same omissions in every arm, independent of gold.
    dropped = sorted(store.facts)[::drop_every] if drop_every else []
    for key in dropped:
        del store.facts[key]
    store.fact_entities = {r for r in store.fact_entities if r.fact_id in store.facts}
    store.entity_relations = {r for r in store.entity_relations if r.fact_id in store.facts}
    store.links = {
        k: v
        for k, v in store.links.items()
        if v.from_fact_id in store.facts and v.to_fact_id in store.facts
    }
    return store, dropped


def provenance_coverage(instance, result, store, lifecycle, job=None):
    episodes = {r.episode_id: r.session_id for r in instance.sessions}
    derived = {m.macro_id: m.supporting_fact_ids for m in lifecycle.macros.values()}
    for state in lifecycle.states.values():
        if state.consolidated_into:
            derived.setdefault(state.consolidated_into, ())
            derived[state.consolidated_into] += (state.fact_id,)
    if job:
        derived.update({i["insight_id"]: tuple(i["fact_ids"]) for i in job["insights"]})

    def sources(memory):
        pending = [memory.id, *memory.supporting_ids]
        seen, sessions = set(), set()
        if memory.source_episode_id in episodes:
            sessions.add(episodes[memory.source_episode_id])
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            if key in store.facts:
                episode = store.facts[key].source_episode_id
                if episode in episodes:
                    sessions.add(episodes[episode])
            pending.extend(derived.get(key, ()))
        return sessions

    if instance.abstention or not instance.answer_session_ids:
        return {}
    relevant = set(instance.answer_session_ids)
    out = {}
    for k in (1, 3, 5, 10):
        found = set().union(*(sources(i.memory) for i in result.items[:k])) & relevant
        out[f"source_session_recall@{k}"] = len(found) / len(relevant)
        out[f"source_session_all@{k}"] = float(found == relevant)
    return out


def compare(instance, frozen, generate, embedder, *, offline=True, qa_generate=None):
    records = []
    scope = Scope(instance.scope.user_id, "longmemeval")
    for arm in ("baseline", "reflect", "reflect_dream"):
        started = time.perf_counter()
        store = clone_store(frozen)
        lifecycle = InMemoryReflectStore(store)
        job = None
        if arm != "baseline":
            Reflector(
                store=lifecycle,
                embedder=embedder,
                clock=lambda: instance.question_time,
                judge=None if offline else LLMConflictJudge(generate),
                macro_writer=None if offline else LLMMacroWriter(generate),
            ).reflect(scope.user_id)
        reader = InMemoryRecallStore(store, lifecycle)
        if arm == "reflect_dream":
            dreamer = Dreamer(
                store=InMemoryDreamStore(store),
                generate=generate,
                embedder=embedder,
                clock=lambda: instance.question_time,
            )
            sessions = tuple(sorted({r.session_id for r in instance.sessions}))
            job = dreamer.start(DreamRequest(scope, sessions))
            dreamer.worker_once()
            # Benchmark explicitly approves into this disposable in-memory fork only.
            job = dreamer.apply(job["run_id"], scope)
            reader = DreamRecallStore(reader, dreamer)
        maintenance_ms = (time.perf_counter() - started) * 1000
        recaller = Recaller(
            store=reader,
            reflect_store=lifecycle,
            embedder=embedder,
            clock=lambda: instance.question_time,
        )
        started = time.perf_counter()
        result = recaller.recall(
            scope,
            instance.question,
            query_time=instance.question_time,
            max_tokens=4096,
            decompose=False,
            types=[MemoryType.FACT, MemoryType.OBSERVATION, MemoryType.MENTAL_MODEL],
        )
        record = {
            "question_id": instance.question_id,
            "question_type": instance.question_type,
            "arm": arm,
            "snapshot_hash": digest([fact_payload(f) for f in frozen.facts.values()]),
            "facts_before": len(frozen.facts),
            "facts_after": len(store.facts),
            "original_fact_ids_preserved": set(frozen.facts) <= set(store.facts),
            "recovered_facts": len(job["recoveries"]) if job else 0,
            "insights": len(job["insights"]) if job else 0,
            "tokens_used": result.tokens_used,
            "retrieval": provenance_coverage(instance, result, store, lifecycle, job),
            "context": result.context,
            "dream": job,
            "timing": {
                "maintenance_ms": maintenance_ms,
                "recall_ms": (time.perf_counter() - started) * 1000,
            },
        }
        if qa_generate:
            from bench.qa import answer, judge

            response = answer(
                qa_generate, result.context, instance.question, instance.question_date
            )
            record["qa"] = {
                "answer": response,
                "correct": judge(
                    qa_generate,
                    instance.question_type,
                    instance.question,
                    instance.answer,
                    response,
                    instance.abstention,
                ),
            }
        records.append(record)
    return records


def code_version():
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    paths = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    files = {
        p: (root / p).read_text(encoding="utf-8")
        for p in paths
        if p.endswith((".py", ".sql", ".toml", "uv.lock")) and (root / p).is_file()
    }
    return {
        "commit": git("rev-parse", "HEAD"),
        "code_sha256": digest(files),
        "dirty": bool(git("status", "--porcelain")),
    }


def factory(spec):
    module, name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module), name)()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("data")
    parser.add_argument("--out", required=True, help="New immutable run directory (must not exist)")
    parser.add_argument("--ids", default=",".join(CASES))
    parser.add_argument("--adapter", help="module:factory -> (structured_generator, embedder)")
    parser.add_argument("--retain-adapter", help="module:factory -> stages(store)")
    parser.add_argument(
        "--snapshots-from", help="Reuse frozen inputs from a previous run directory"
    )
    parser.add_argument("--qa-adapter", help="module:factory -> plain text generator")
    parser.add_argument("--drop-every", type=int, default=0, help="Synthetic omissions; 0 disables")
    args = parser.parse_args(argv)
    if args.drop_every < 0:
        parser.error("--drop-every must be nonnegative")
    ids = set(args.ids.split(","))
    instances = load_data(args.data, ids)
    missing = ids - {i.question_id for i in instances}
    if missing:
        parser.error(f"Missing question IDs: {sorted(missing)}")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    generate, embedder = (
        factory(args.adapter) if args.adapter else (SmokeGenerator(), HashingEmbedder())
    )
    stages = factory(args.retain_adapter) if args.retain_adapter else verbatim_stages
    qa_generate = factory(args.qa_adapter) if args.qa_adapter else None
    manifest = {
        **code_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args),
        "question_ids": sorted(ids),
        "dataset_sha256": digest(instances),
        "status": "running",
        "evaluation_kind": "adapter_diagnostic" if args.adapter else "offline_plumbing_smoke",
        "retain_kind": "frozen_snapshot"
        if args.snapshots_from
        else ("adapter" if args.retain_adapter else "verbatim_without_entities_or_links"),
        "quality_claim": False,
        "paid_calls_from_harness": False,
        "note": "Adapters own any paid calls. Offline/hash results do not measure semantic quality.",
    }

    def save(name, value):
        (output / name).write_text(encoded(value) + "\n", encoding="utf-8")

    save("manifest.json", manifest)
    records, errors = [], []
    for instance in instances:
        try:
            if args.snapshots_from:
                frozen, dropped = load_frozen(
                    Path(args.snapshots_from) / f"{instance.question_id}.snapshot.json",
                    instance,
                    args.drop_every,
                )
            else:
                frozen, dropped = prepare(instance, embedder, stages, args.drop_every)
            save(
                f"{instance.question_id}.snapshot.json",
                frozen_payload(instance, frozen, dropped, args.drop_every),
            )
            rows = compare(
                instance,
                frozen,
                generate,
                embedder,
                offline=not args.adapter,
                qa_generate=qa_generate,
            )
            records.extend(rows)
            save(f"{instance.question_id}.results.json", rows)
            print(f"{instance.question_id}: all three arms completed", flush=True)
        except Exception as exc:
            errors.append(
                {"question_id": instance.question_id, "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"{instance.question_id}: {type(exc).__name__}", flush=True)
    summary = {"completed_questions": len(records) // 3, "failed_questions": errors, "arms": {}}
    for arm in ("baseline", "reflect", "reflect_dream"):
        rows = [r for r in records if r["arm"] == arm]
        if rows:
            summary["arms"][arm] = {
                "count": len(rows),
                "original_fact_ids_preserved": all(r["original_fact_ids_preserved"] for r in rows),
                "recovered_facts": sum(r["recovered_facts"] for r in rows),
                "insights": sum(r["insights"] for r in rows),
                "mean_context_tokens": mean(r["tokens_used"] for r in rows),
            }
    save("summary.json", summary)
    manifest["status"] = "failed" if errors else "complete"
    save("manifest.json", manifest)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
