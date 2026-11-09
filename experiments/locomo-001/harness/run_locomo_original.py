"""LoCoMo adapter for the untouched original codelab (commit 6f343d5).

Reuses run_original.configure (source-manifest check, Gemini/embedding routing, namespaced
Valkey, synchronous extraction, ledger) and run_mem2act's metered SQL cursor and lossless
exchange grouping, so each exchange is (user string, JSON list of following events) exactly as
PR13's developer worker builds it from the same message objects.

Differences from run_original.run_cases, all declared in RECORD.md:
- One memory (user_id = <run_id>_<case>) is ingested once; every question is then answered from
  it in its own fresh question session.
- Answers do not write back: the post-answer Valkey append and entity extraction are no-ops
  during the answer phase (PR13 uses record=False). Retrieval and prompt are unchanged.
- Answers run in a thread pool (read-only), each with its own DB connection.

Usage (on the VM, with cloud.env sourced):
  python run_locomo_original.py ingest|answer --baseline ... --case ... --output ... \
      --ledger ... --run-id ... --limit-usd X [--questions-limit N] [--workers W]
"""

import argparse
import fcntl
import json
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # benchmark-kit: run_original.py, run_mem2act.py

import run_mem2act  # noqa: E402
import run_original as base  # noqa: E402


def session_id(user_id, index, date):
    return f"{user_id}_s{index:03d}_" + re.sub(r"[^A-Za-z0-9_]", "_", date)


def ingest(a, env, case):
    user_id = a.run_id + "_" + case["case_id"]
    state_path = a.output / "ingest-progress.json"
    if state_path.exists():
        raise ValueError("Ingestion already started for this run id; use a fresh run id")
    cache = env.db.get_valkey_client()
    assert cache.ping()
    worker = env.worker(env.db.get_db_connection, env.client)
    previous, sessions = None, []
    started = time.time()
    for index, (sid, date, messages) in enumerate(
        zip(case["session_ids"], case["session_dates"], case["sessions"], strict=True)
    ):
        current = session_id(user_id, index, date)
        env.ledger.context = {
            "operation": "ingestion",
            "stage": "ingestion",
            "session_index": index,
        }
        t0 = time.perf_counter()
        mapping = []
        for user, events, indices in run_mem2act.exchanges(messages):
            env.buffer.append_session_turn_with_rolling_summary(
                cache, env.client, current, user, events, 10, 4
            )
            env.errors.check()
            worker.enqueue_extraction(user_id, current, user, events, previous)
            env.errors.check()
            mapping.append(indices)
        if env.buffer.IN_MEMORY_VALKEY_FALLBACK:
            raise RuntimeError("Valkey fallback was used")
        covered = sorted(i for m in mapping for i in m)
        assert covered == list(range(len(messages))), "every message exactly once"
        sessions.append(
            {
                "source_session_id": sid,
                "source_date": date,
                "session_id": current,
                "exchange_source_indices": mapping,
                "elapsed_seconds": time.perf_counter() - t0,
            }
        )
        base.save(state_path, {"completed_sessions": index + 1, "sessions": sessions})
        previous = current
        print(f"ingested session {index + 1}/{len(case['sessions'])}", flush=True)
    conn = env.db.get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT entity_name, summary, scope, project_id, summary_embedding IS NOT NULL "
                "FROM agent_entities WHERE user_id=%s ORDER BY entity_name",
                (user_id,),
            )
            entities = cur.fetchall()
    finally:
        conn.close()
    if not entities or not all(r[4] for r in entities):
        raise RuntimeError("Expected populated entities with transactional embeddings")
    base.save(a.output / "entities.json", entities)
    base.save(
        a.output / "ingest.json",
        {
            "status": "completed",
            "user_id": user_id,
            "last_session_id": previous,
            "entities": len(entities),
            "sessions": len(sessions),
            "messages": sum(len(s) for s in case["sessions"]),
            "exchanges": sum(len(s["exchange_source_indices"]) for s in sessions),
            "ingestion_seconds": sum(s["elapsed_seconds"] for s in sessions),
            "wall_seconds": time.time() - started,
        },
    )


class NoWrite:
    """Answer-phase replacement for the post-answer extraction worker."""

    def enqueue_extraction(self, *values):
        return None


def answer(a, env, case):
    done = json.loads((a.output / "ingest.json").read_text(encoding="utf-8"))
    assert done["status"] == "completed"
    user_id, previous = done["user_id"], done["last_session_id"]
    env.agent.append_session_turn_with_rolling_summary = lambda *x, **k: None
    cache = env.db.get_valkey_client()
    assert cache.ping()
    env.ledger.context = {"operation": "answer", "stage": "answer_generation"}
    questions = case["questions"][: a.questions_limit] if a.questions_limit else case["questions"]
    out = a.output / "answers"
    out.mkdir(exist_ok=True)
    lock = threading.Lock()

    def one(q):
        path = out / f"{q['id']}.json"
        if path.exists():
            return
        conn = env.db.get_db_connection()
        try:
            qsession = user_id + "_question_" + re.sub(r"[^A-Za-z0-9_]", "_", q["id"])
            t0 = time.perf_counter()
            text, prompt, latency = env.agent.run_agent_turn(
                conn,
                cache,
                NoWrite(),
                env.client,
                user_id,
                qsession,
                q["question"],
                prev_session_id=previous,
            )
            elapsed = time.perf_counter() - t0
        finally:
            conn.close()
        base.save(
            path,
            {
                "id": q["id"],
                "answer": text,
                "answer_seconds": elapsed,
                "orchestrator_latency_seconds": latency,
                "prompt": prompt,
                "question": q["question"],
                "question_session": qsession,
            },
        )
        with lock:
            print("answered", q["id"], f"{elapsed:.1f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        for f in [pool.submit(one, q) for q in questions]:
            f.result()
    env.errors.check()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["ingest", "answer"])
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--limit-usd", type=float, required=True)
    p.add_argument("--infrastructure-reserve-usd", type=float, default=0.0)
    p.add_argument("--questions-limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]{1,40}", a.run_id):
        p.error("run-id must be ASCII alphanumerics/underscores")
    a.output.mkdir(parents=True, exist_ok=True)
    a.ledger.mkdir(parents=True, exist_ok=True)
    case = json.loads(a.case.read_text(encoding="utf-8"))
    for q in case["questions"]:
        assert set(q) == {"id", "question"}, "no labels at runtime"
    with (a.ledger / ".pilot.lock").open("w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = base.Ledger(a.ledger, a.limit_usd, a.infrastructure_reserve_usd)
        ledger.state["sql_generate_policy"] = (
            "Maximum reservation settled from opt-in provider usage notices; original output "
            "transform unchanged."
        )
        base.save(ledger.path, ledger.state)
        base.Cursor = run_mem2act.MeteredCursor
        shutil.copyfile(__file__, a.output / f"executed_locomo_adapter_{a.mode}.py")
        base.save(
            a.output / f"adapter-{a.mode}.json",
            {
                "adapter_sha256": base.digest(__file__),
                "run_original_sha256": base.digest(base.__file__),
                "run_mem2act_sha256": base.digest(run_mem2act.__file__),
                "case_sha256": base.digest(a.case),
                "run_id": a.run_id,
                "started_at": time.time(),
                "workers": a.workers,
                "questions_limit": a.questions_limit,
            },
        )
        try:
            env = base.configure(a, ledger)
            original_connection = env.db.get_db_connection

            def metered_connection():
                conn = original_connection()
                with conn.cursor() as cur:
                    cur.execute("SET codelab.usage_meter = 'on'")
                return conn

            env.db.get_db_connection = metered_connection
            (ingest if a.mode == "ingest" else answer)(a, env, case)
            status = {"status": "completed", "mode": a.mode, "budget": ledger.state}
        except BaseException as e:
            status = {
                "status": "budget_stopped" if isinstance(e, base.BudgetStop) else "failed",
                "mode": a.mode,
                "error_type": type(e).__name__,
                "error": str(e)[:2000],
                "budget": ledger.state,
            }
            base.save(a.output / f"status-{a.mode}.json", status)
            raise
        base.save(a.output / f"status-{a.mode}.json", status)


if __name__ == "__main__":
    main()
