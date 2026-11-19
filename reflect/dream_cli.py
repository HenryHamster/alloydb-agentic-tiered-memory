"""Explicit Dream controls: start, worker, status, apply, discard (PostgreSQL)."""

import argparse
import importlib
import json
import time

from reflect.dream import DreamConfig, Dreamer, DreamRequest
from reflect.dream_store import AlloyDBDreamStore
from retain.models import Scope


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "worker", "status", "apply", "discard"])
    parser.add_argument("--user")
    parser.add_argument("--project")
    parser.add_argument("--sessions", help="Comma-separated explicit source session IDs")
    parser.add_argument("--run")
    parser.add_argument("--request-key")
    parser.add_argument("--once", action="store_true", help="Worker processes at most one job")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument(
        "--adapter", help="module:factory returning (structured_generator, embedder)"
    )
    parser.add_argument("--page-chars", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.command != "worker" and (not args.user or not args.project):
        parser.error("--user and --project are required")
    if args.command in ("status", "apply", "discard") and not args.run:
        parser.error("--run is required")
    if args.command == "start" and not args.sessions:
        parser.error("--sessions is required")

    def unavailable(*_):
        raise RuntimeError("A provider adapter is required for this operation")

    generate = embedder = unavailable
    if args.command in ("worker", "apply"):
        if not args.adapter or ":" not in args.adapter:
            parser.error("--adapter module:factory is required for worker/apply")
        module, factory = args.adapter.rsplit(":", 1)
        generate, embedder = getattr(importlib.import_module(module), factory)()
    from db_clients import get_db_connection

    dreamer = Dreamer(
        store=AlloyDBDreamStore(get_db_connection),
        generate=generate,
        embedder=embedder,
        config=DreamConfig(page_chars=args.page_chars),
    )
    if args.command == "worker":
        while True:
            try:
                job = dreamer.worker_once()
                if job:
                    print(
                        json.dumps({"run_id": job["run_id"], "status": job["status"]}), flush=True
                    )
            except Exception as exc:
                if args.once:
                    raise
                # Job errors are already persisted. Avoid printing provider payloads.
                print(json.dumps({"worker_error": type(exc).__name__}), flush=True)
            if args.once:
                return
            time.sleep(args.poll_seconds)
    else:
        scope = Scope(args.user, args.project)
        if args.command == "start":
            job = dreamer.start(
                DreamRequest(scope, tuple(args.sessions.split(",")), args.request_key)
            )
        else:
            method = dreamer.get if args.command == "status" else getattr(dreamer, args.command)
            job = method(args.run, scope)
        print(json.dumps(job, indent=2))


if __name__ == "__main__":
    main()
