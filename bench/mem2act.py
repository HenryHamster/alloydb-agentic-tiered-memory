"""Run the production tiered-memory pipeline on pinned Mem2Act linked histories.

JSON tool proposals only; no external tool execution and no gold labels at inference.
Use a fresh namespace for every attempt. Completed questions may be resumed with
the same source/input hashes and ledger; partial questions are never silently reused.
"""

import argparse
import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

from bench.metering import BudgetStop, Ledger, Models, RankConnection, digest, save
from retain import Scope
from tiered_memory import build_memory_system


def exchanges(messages):
    """Same lossless role/event grouping as the original-codelab pilot."""
    i = 0
    while i < len(messages):
        start, user = i, ""
        if messages[i]["role"] == "user":
            user = messages[i]["content"]
            if not isinstance(user, str):
                user = json.dumps(user, ensure_ascii=False)
            i += 1
        events = []
        while i < len(messages) and messages[i]["role"] != "user":
            if messages[i]["role"] not in {"assistant", "tool", "tool_calls", "system"}:
                raise ValueError("Unsupported role: " + messages[i]["role"])
            events.append(messages[i])
            i += 1
        yield user, json.dumps(events, ensure_ascii=False), list(range(start, i))


def serial(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items() if k not in {"embedding", "macro_embedding"}}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return str(value)


def run_case(case, system, ledger, output, run_id):
    qid = case["question_id"]
    folder = output / qid
    if folder.exists():
        raise ValueError(f"Partial/existing namespace {qid}; do not replay silently")
    folder.mkdir(parents=True)
    namespace = f"{run_id}:{qid}"
    started = time.perf_counter()
    initial_cost = ledger.state["accounted_usd"]
    mapping = []
    for session_index, messages in enumerate(case["haystack_sessions"]):
        scope = Scope(namespace, namespace, f"{namespace}:history:{session_index}")
        for index, (user, events, indices) in enumerate(exchanges(messages)):
            with ledger.scope(
                question_id=qid, stage="retain_history", session=session_index, round=index
            ):
                result = system.record_turn(
                    scope, user, events, episode_id=f"{scope.session_id}:{index}"
                )
                save(folder / "retention" / f"{session_index}-{index}.json", serial(result))
                mapping.append({"session": session_index, "round": index, "messages": indices})
                save(folder / "message_mapping.json", mapping)
        with ledger.scope(question_id=qid, stage="session_maintenance", session=session_index):
            save(folder / f"reflect-{session_index}.json", serial(system.finish_session(scope)))
    ingested = time.perf_counter()
    query_scope = Scope(namespace, namespace, f"{namespace}:query")
    with ledger.scope(question_id=qid, stage="recall_and_answer"):
        answer = system.answer(query_scope, case["question"], record=False)
    prediction = {
        "question_id": qid,
        "hypothesis": answer.text,
        "timing": {
            "ingestion_seconds": ingested - started,
            "answer_ready_seconds": answer.answer_seconds,
            "turn_seconds_including_post_retention": None,
        },
        "post_answer_retained": False,
    }
    save(folder / "prediction.json", prediction)
    save(folder / "context.json", answer.context.to_dict())
    (folder / "prompt.txt").write_text(answer.prompt, encoding="utf8")
    with ledger.scope(question_id=qid, stage="post_answer_retention"):
        system.record_turn(
            query_scope, case["question"], answer.text, episode_id=f"{namespace}:answer"
        )
    prediction["post_answer_retained"] = True
    prediction["timing"]["turn_seconds_including_post_retention"] = time.perf_counter() - ingested
    prediction["question_accounted_usd"] = ledger.state["accounted_usd"] - initial_cost
    save(folder / "prediction.json", prediction)
    save(
        folder / "memory_snapshot.json",
        serial(
            {
                "facts": system.reflector.store.user_facts(namespace),
                "macros": system.reflector.store.user_macros(namespace),
            }
        ),
    )
    save(
        folder / "completed.json",
        {"question_id": qid, "accounted_usd": prediction["question_accounted_usd"]},
    )
    print(
        json.dumps(
            {
                "completed": qid,
                "budget": ledger.state["accounted_usd"],
                "question_usd": prediction["question_accounted_usd"],
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--limit-usd", type=float, default=10)
    p.add_argument("--infrastructure-reserve-usd", type=float, default=2)
    p.add_argument("--limit-questions", type=int, default=10)
    p.add_argument("--resume-completed", action="store_true")
    a = p.parse_args()
    if not 0 < a.limit_usd <= 10 or not 0 <= a.infrastructure_reserve_usd < a.limit_usd:
        p.error("This pilot has a USD 10 aggregate authorization")
    if os.environ.get("PROJECT_ID") != "alloydb-memory":
        p.error("This pilot is authorized only in alloydb-memory")
    a.output.mkdir(parents=True, exist_ok=True)
    a.ledger.mkdir(parents=True, exist_ok=True)
    import fcntl

    from google import genai

    from db_clients import get_db_connection, get_valkey_client

    with (a.ledger / ".candidate.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = Ledger(a.ledger, a.limit_usd, a.infrastructure_reserve_usd)
        identity = {
            "run_id": a.run_id,
            "cases_sha256": digest(a.cases),
            "source_manifest_sha256": digest(a.source_manifest),
        }
        identity_path = a.output / "identity.json"
        if identity_path.exists():
            if not a.resume_completed or json.loads(identity_path.read_text()) != identity:
                raise ValueError("Resume needs identical source/input identity and explicit flag")
        save(identity_path, identity)
        save(a.output / "source_manifest.json", json.loads(a.source_manifest.read_text()))
        save(a.output / "cases.json", json.loads(a.cases.read_text()))
        logging.basicConfig(filename=a.output / "pipeline.log", level=logging.INFO)
        generation = genai.Client(vertexai=True, project="alloydb-memory", location="global")
        embedding = genai.Client(vertexai=True, project="alloydb-memory", location="us-east1")
        metered = SimpleNamespace(models=Models(generation, embedding, ledger))
        system = build_memory_system(
            lambda: RankConnection(get_db_connection(), ledger),
            get_valkey_client(),
            metered,
            metered,
            strict=True,
        )
        system.retainer.extractor.source_dates_known = False
        status = {"status": "running", **identity, "started_at": time.time()}
        save(a.output / "status.json", status)
        try:
            done = 0
            for case in json.loads(a.cases.read_text(encoding="utf8")):
                if (
                    a.resume_completed
                    and (a.output / case["question_id"] / "completed.json").exists()
                ):
                    continue
                run_case(case, system, ledger, a.output, a.run_id)
                done += 1
                if done >= a.limit_questions:
                    break
            status["status"] = "completed_requested_questions"
        except BaseException as exc:
            status.update(
                status="budget_stopped" if isinstance(exc, BudgetStop) else "failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        finally:
            status.update(finished_at=time.time(), budget=ledger.state)
            save(a.output / "status.json", status)
            system.close()
            generation.close()
            embedding.close()


if __name__ == "__main__":
    main()
