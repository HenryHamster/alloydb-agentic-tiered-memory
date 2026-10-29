"""Run selected source code against isolated memory; only the bootstrap owns VM leases."""

import argparse
import dataclasses
import json
import os
import time
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

from devtools.cache import (
    CachedConnection,
    CachedModels,
    CacheMiss,
    RequestCache,
    Usage,
)
from devtools.files import fingerprint, locked, private_dir, read_json, save_json
from devtools.inputs import (
    inference_questions,
    reference_time,
    timestamp,
    validate_history,
    validate_questions,
)


def serial(value):
    if dataclasses.is_dataclass(value):
        return serial(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def preparation_source(source):
    """Conservative invalidation; Recall-only files do not invalidate prepared memory."""
    paths = [
        p
        for folder in ("retain", "reflect", "shared", "sql", "devtools", "bench")
        for p in (source / folder).glob("**/*")
        if p.is_file() and p.suffix in {".py", ".sql"}
    ]
    paths += [source / "tiered_memory.py", source / "valkey_buffer.py"]
    import hashlib

    return {
        p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def ingest(system, history, logical, clock, usage, output):
    from bench.mem2act import HistoryVisibility, exchanges
    from retain import Scope

    # A fixture is one user's cross-session history, exactly as in the Mem2Act harness.
    # Session-only extraction must not make earlier sessions invisible to the question.
    extractor = system.retainer.extractor
    system.retainer.extractor = HistoryVisibility(extractor)
    mapping = []
    ordinal = 0
    for session in history["sessions"]:
        scope = Scope(logical, logical, f"{logical}:{session['id']}")
        for index, (user, events, indices) in enumerate(exchanges(session["messages"])):
            selected = [session["messages"][i] for i in indices]
            known = all(m.get("timestamp") for m in selected)
            extractor.source_dates_known = known
            at = (
                timestamp(selected[0]["timestamp"])
                if known
                else clock
                - timedelta(seconds=sum(len(s["messages"]) for s in history["sessions"]) - ordinal)
            )
            usage.context = {"stage": "retain", "session": session["id"], "round": index}
            receipt = system.record_turn(
                scope, user, events, timestamp=at, episode_id=f"{logical}:{session['id']}:{index}"
            )
            save_json(output / "retention" / f"{ordinal:05d}.json", serial(receipt))
            mapping.append({"session": session["id"], "round": index, "messages": indices})
            ordinal += 1
        usage.context = {"stage": "reflect", "session": session["id"]}
        save_json(output / f"reflect-{session['id']}.json", serial(system.finish_session(scope)))
    save_json(output / "message-mapping.json", mapping)


class CloudBackend:
    def __init__(self, config, cache, models, source, clock, run_id):
        from google.auth import compute_engine

        from db_clients import get_db_connection, get_valkey_client
        from devtools.state import PrefixedValkey

        self.credentials = compute_engine.Credentials()
        self.connect = lambda: get_db_connection(credentials=self.credentials)
        self.valkey = PrefixedValkey(get_valkey_client(), run_id)
        self.valkey.client.ping()
        self.cache, self.models, self.source, self.clock = cache, models, source, clock

    def schema(self):
        from devtools.state import RunSchema

        return RunSchema(self.connect, uuid.uuid4().hex, self.source)

    def system(self, schema):
        from tiered_memory import build_memory_system

        client = SimpleNamespace(models=self.models)
        system = build_memory_system(
            lambda: CachedConnection(schema.connection(), self.cache), self.valkey, client, client
        )
        for component in (system.retainer, system.reflector, system.recaller):
            component.clock = lambda: self.clock
        return system

    def close(self):
        self.valkey.close()


def run_job(
    job, source, cache_folder, output, generation, embedding, *, backend_factory=CloudBackend
):
    """Usage receipts persist on failures and replay misses. Fresh schemas start each question."""
    source, output = Path(source), private_dir(output)
    history = validate_history(job["history"])
    questions = validate_questions({"questions": job["questions"]})
    if questions != inference_questions(questions):
        raise ValueError("Scoring labels must not be sent to the worker")
    save_json(output / "job.json", job)
    usage = Usage(output)
    config = job["config"]
    cache = RequestCache(
        Path(cache_folder) / "requests",
        usage,
        mode=job["mode"],
        provider={
            "project": config["project"],
            "generation_location": config["generation_location"],
            "embedding_location": config["region"],
        },
        epoch=job.get("cache_epoch", "1"),
    )
    models = CachedModels(generation, embedding, cache, max_output_tokens=job["max_output_tokens"])
    clock = reference_time(history) + timedelta(
        seconds=sum(len(s["messages"]) for s in history["sessions"]) + 1
    )
    logical = "fixture:" + fingerprint(history)[:32]
    backend = None
    status = {
        "status": "running",
        "started_at": time.time(),
        "logical_clock": clock.isoformat(),
        "source_dates": "Only supplied source timestamps are treated as known.",
        "answers": [],
    }
    save_json(output / "status.json", status)
    try:
        backend = backend_factory(config, cache, models, source, clock, job["run_id"])
        prep_key = fingerprint(
            {
                "source": preparation_source(source),
                "history": history,
                "provider": cache.provider,
                "epoch": cache.epoch,
                "max_output_tokens": job["max_output_tokens"],
                "format": 3,
            }
        )
        prepared = Path(cache_folder) / "prepared" / f"{prep_key}.json"
        started = time.perf_counter()
        with locked(prepared.with_suffix(".lock")):
            if job["mode"] != "fresh" and prepared.exists():
                entry = read_json(prepared)
                if entry["hash"] != fingerprint(entry["snapshot"]):
                    raise ValueError("Prepared memory cache failed integrity validation")
                snapshot = entry["snapshot"]
                status["prepared_cache_hit"] = True
            else:
                schema = backend.schema()
                system = None
                try:
                    schema.create()
                    system = backend.system(schema)
                    ingest(system, history, logical, clock, usage, output)
                    snapshot = schema.snapshot()
                    if not prepared.exists():
                        save_json(prepared, {"hash": fingerprint(snapshot), "snapshot": snapshot})
                    status["prepared_cache_hit"] = False
                finally:
                    try:
                        if system:
                            system.close()
                        if schema.created:
                            save_json(output / "prepared-state.json", schema.snapshot())
                    finally:
                        schema.close()
        status["preparation_seconds"] = time.perf_counter() - started
        status["prepared_key"] = prep_key
        save_json(output / "prepared-state.json", snapshot)
        from retain import Scope

        for question in questions:
            usage.context = {"stage": "question", "question_id": question["id"]}
            schema = backend.schema()
            system = None
            try:
                schema.create()
                schema.restore(snapshot)  # Includes tiers, access counts and Reflect state.
                system = backend.system(schema)
                prompt = question["question"]
                if question.get("tools"):
                    prompt += "\n\nAvailable tool schemas:\n" + json.dumps(
                        question["tools"], ensure_ascii=False
                    )
                    prompt += (
                        "\nReturn only one JSON object with keys name and arguments describing the tool call. "
                        "Do not execute the tool. Fill arguments from stored history and the schemas."
                    )
                answer = system.answer(
                    Scope(logical, logical, logical + ":query:" + question["id"]),
                    prompt,
                    record=False,
                    query_time=clock,
                )
                result = {
                    "id": question["id"],
                    "answer": answer.text,
                    "answer_seconds": answer.answer_seconds,
                    "context": answer.context.to_dict(),
                    "prompt": answer.prompt,
                }
                save_json(output / "answers" / f"{question['id']}.json", result)
                status["answers"].append(question["id"])
                save_json(output / "status.json", status)
            finally:
                try:
                    if system:
                        system.close()
                finally:
                    schema.close()
        status["status"] = "completed"
    except CacheMiss as error:
        status.update(status="cache_miss", error=str(error))
    except BaseException as error:
        # Error class only: SDK exception strings can contain credentials/request headers.
        import traceback

        status.update(
            status="failed",
            error_type=type(error).__name__,
            error_frames=[
                {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(error.__traceback__)
            ],
        )
    finally:
        if backend:
            try:
                backend.close()
            except Exception as error:
                status.update(status="failed", cleanup_error_type=type(error).__name__)
        status.update(finished_at=time.time())
        save_json(output / "status.json", status)
        usage.write()
    return status


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vm-config", type=Path, default=Path("/etc/tiered-dev/config.json"))
    args = parser.parse_args(argv)
    job, vm = read_json(args.job), read_json(args.vm_config)
    # Network destinations and database identity are trusted VM configuration, never job input.
    if job["config"]["project"] != vm["project"]:
        parser.error("Job project differs from this VM's configured development project")
    job["config"] = {k: vm[k] for k in ("project", "region", "generation_location")}
    for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "VALKEY_HOST", "VALKEY_PORT"):
        os.environ[key] = str(vm[key])
    from google import genai
    from google.auth import compute_engine

    credentials = compute_engine.Credentials()
    # One SDK attempt per recorded call; adapter retries get separate usage receipts.
    options = {"timeout": 90000, "retry_options": {"attempts": 1}}
    generation = genai.Client(
        vertexai=True,
        project=vm["project"],
        location=vm["generation_location"],
        credentials=credentials,
        http_options=options,
    )
    embedding = genai.Client(
        vertexai=True,
        project=vm["project"],
        location=vm["region"],
        credentials=credentials,
        http_options=options,
    )
    try:
        result = run_job(job, Path.cwd(), args.cache, args.output, generation, embedding)
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "completed" else 2
    finally:
        generation.close()
        embedding.close()


if __name__ == "__main__":
    raise SystemExit(main())
