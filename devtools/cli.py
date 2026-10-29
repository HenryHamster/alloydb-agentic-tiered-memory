"""Thin local CLI over gcloud/IAP. No credentials are bundled with developer source."""

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

from devtools.files import private_dir, read_json, safe_name, save_json
from devtools.inputs import inference_questions, score, validate_history, validate_questions

TARGETS = ("project", "zone", "vm", "region", "db_cluster", "db_instance")
DEFAULTS = {"generation_location": "global", "idle_minutes": 30, "max_hours": 4}


def load_config(path):
    raw = {**DEFAULTS, **read_json(path)}
    for key in TARGETS:
        if not isinstance(raw.get(key), str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,99}", raw[key]):
            raise ValueError(f"Invalid or missing config field: {key}")
    if not 0.25 <= raw["max_hours"] <= 8 or not 5 <= raw["idle_minutes"] <= 120:
        raise ValueError("max_hours must be 0.25-8 and idle_minutes 5-120")
    # Only public routing/policy fields go into jobs. Ignore any accidental local secret values.
    return {key: raw[key] for key in (*TARGETS, *DEFAULTS)}


def command(args, *, timeout=120):
    executable = shutil.which(args[0])
    if executable is None and sys.platform == "win32":
        candidates = {
            "gcloud": Path.home() / "google-cloud-sdk/bin/gcloud.cmd",
            "git": Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/cmd/git.exe",
        }
        candidate = candidates.get(args[0])
        if candidate and candidate.is_file():
            executable = str(candidate)
    if executable is None:
        raise RuntimeError(f"Install {args[0]} and add it to PATH")
    env = dict(os.environ)
    if args[0] == "gcloud" and sys.platform == "win32":
        # IAP's PuTTY proxy needs a real Python executable, not a shell alias.
        env.setdefault("CLOUDSDK_PYTHON", sys.executable)
    result = subprocess.run(
        [executable, *args[1:]], text=True, capture_output=True, timeout=timeout, env=env
    )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): " + result.stderr[-1500:])
    return result.stdout


class Cloud:
    def __init__(self, config, *, invoke=command, sleep=time.sleep):
        self.config, self.invoke, self.sleep = config, invoke, sleep

    def gcloud(self, args, **kwargs):
        return self.invoke(
            ["gcloud", *args, "--project=" + self.config["project"], "--quiet"], **kwargs
        )

    def vm_state(self):
        return json.loads(
            self.gcloud(
                [
                    "compute",
                    "instances",
                    "describe",
                    self.config["vm"],
                    "--zone=" + self.config["zone"],
                    "--format=json(status)",
                ]
            )
        )["status"]

    def db_state(self):
        return json.loads(
            self.gcloud(
                [
                    "alloydb",
                    "instances",
                    "describe",
                    self.config["db_instance"],
                    "--cluster=" + self.config["db_cluster"],
                    "--region=" + self.config["region"],
                    "--format=json(state,activationPolicy,reconciling)",
                ]
            )
        )

    def ssh(self, remote, **kwargs):
        return self.gcloud(
            [
                "compute",
                "ssh",
                self.config["vm"],
                "--zone=" + self.config["zone"],
                "--tunnel-through-iap",
                "--command",
                remote,
            ],
            **kwargs,
        )

    def control(self, action):
        return json.loads(self.ssh("python3 /opt/tiered-dev/control.py " + action))

    def status(self):
        vm, database = self.vm_state(), self.db_state()
        result = {"vm": vm, "database": database}
        if vm == "RUNNING":
            result["controller"] = self.control("status")
        return result

    def start(self):
        state = self.vm_state()
        if state not in {"RUNNING", "TERMINATED"}:
            raise RuntimeError(f"VM is {state}; wait for its current operation")
        fresh_after = time.time() if state == "TERMINATED" else None
        if state == "TERMINATED":
            # Root idle controller has an independent DB deadline before this VM backstop.
            seconds = int(self.config["max_hours"] * 3600 + 600)
            self.gcloud(
                [
                    "compute",
                    "instances",
                    "set-scheduling",
                    self.config["vm"],
                    "--zone=" + self.config["zone"],
                    f"--max-run-duration={seconds}s",
                    "--instance-termination-action=STOP",
                    "--no-restart-on-failure",
                ]
            )
            self.gcloud(
                [
                    "compute",
                    "instances",
                    "start",
                    self.config["vm"],
                    "--zone=" + self.config["zone"],
                    "--async",
                ]
            )
        deadline = time.monotonic() + 900
        while True:
            try:
                if self.vm_state() == "RUNNING":
                    controller = self.control("status")
                    if controller["target"] != {k: self.config[k] for k in TARGETS}:
                        raise ValueError("Local target differs from the installed VM controller")
                    if controller.get("policy") != {
                        k: self.config[k] for k in ("idle_minutes", "max_hours")
                    }:
                        raise ValueError(
                            "Local idle/runtime policy differs from installed VM policy"
                        )
                    if fresh_after and controller.get("checked_at", 0) < fresh_after:
                        raise RuntimeError("Waiting for a fresh controller check after startup")
                    if controller.get("booting"):
                        raise RuntimeError("Waiting for this boot's controller state")
                    if controller.get("stopping"):
                        raise ValueError("VM shutdown is already pending; wait until it stops")
                    self.control("touch")
                    break
            except RuntimeError:
                # SSH/boot races are expected, but a missing installer never starts the database.
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "VM/controller not ready. Complete the one-time VM install first."
                )
            print("Waiting for VM/IAP and its idle controller...", flush=True)
            self.sleep(10)
        db = self.db_state()
        while db["state"] == "STOPPING":
            if time.monotonic() >= deadline:
                raise RuntimeError("Previous database shutdown has not finished; use dev status")
            print("Waiting for the previous AlloyDB shutdown...", flush=True)
            self.control("touch")
            self.sleep(15)
            db = self.db_state()
        if db["state"] == "STOPPED":
            self.gcloud(
                [
                    "alloydb",
                    "instances",
                    "update",
                    self.config["db_instance"],
                    "--cluster=" + self.config["db_cluster"],
                    "--region=" + self.config["region"],
                    "--activation-policy=ALWAYS",
                    "--async",
                ]
            )
        elif db["state"] not in {"READY", "STARTING"}:
            raise RuntimeError(f"Database is {db['state']}; wait for its current operation")
        while True:
            db = self.db_state()
            if db["state"] == "READY" and not db.get("reconciling"):
                self.control("touch")
                return self.status()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Database startup timed out; use dev status/stop. Idle shutdown remains armed."
                )
            print("Waiting for AlloyDB (startup can take several minutes)...", flush=True)
            self.sleep(15)

    def stop(self):
        if self.vm_state() == "RUNNING":
            request = self.control("stop")
            deadline = time.monotonic() + 150
            while self.vm_state() == "RUNNING":
                try:
                    state = self.control("status")
                except RuntimeError:
                    # SSH can disappear before Compute reports STOPPING.
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Shutdown status unavailable; use dev status") from None
                    self.sleep(3)
                    continue
                if state.get("last_stop_request", 0) >= request["requested_at"]:
                    if state.get("stop_result") == "busy":
                        raise RuntimeError("Other runs are active; shutdown refused")
                    if state.get("stopping"):
                        return {"status": "shutdown_requested", "controller": state}
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Controller did not acknowledge stop; inspect its systemd service"
                    )
                self.sleep(5)
            return self.status()
        # No jobs can be active on a stopped VM. Reconcile an orphan running database.
        self.gcloud(
            [
                "alloydb",
                "instances",
                "update",
                self.config["db_instance"],
                "--cluster=" + self.config["db_cluster"],
                "--region=" + self.config["region"],
                "--activation-policy=NEVER",
                "--async",
            ]
        )
        return {"status": "database_shutdown_requested", "vm": self.vm_state()}


def source_bundle(source, destination, job):
    tracked = command(
        ["git", "-C", str(source), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    )
    names = sorted(set(tracked.split("\0")) - {""})
    files = {}
    for name in names:
        path = source / name
        if (
            not path.is_file()
            or path.is_symlink()
            or any(p in {".git", ".venv", ".dev"} for p in path.parts)
            or not (path.suffix in {".py", ".sql"} or name in {"pyproject.toml", "uv.lock"})
        ):
            continue
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not {"devtools/worker.py", "uv.lock", "pyproject.toml"} <= files.keys():
        raise ValueError("Run from a checkout containing the developer runner and uv.lock")
    manifest = {
        "commit": command(["git", "-C", str(source), "rev-parse", "HEAD"]).strip(),
        "files": files,
        "working_tree_diff": command(["git", "-C", str(source), "diff", "HEAD", "--", *files]),
    }
    save_json(destination.parent / "source-manifest.json", manifest)
    with tarfile.open(destination, "w:gz") as bundle:
        for name in files:
            bundle.add(source / name, arcname="source/" + name, recursive=False)
        for name, value in (("job.json", job), ("manifest.json", manifest)):
            body = json.dumps(value, ensure_ascii=False).encode()
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(body), 0o600
            bundle.addfile(member, io.BytesIO(body))
    return manifest


def collect_archive(archive, expected_hash, output, run_id):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Remote export checksum mismatch")
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            parts = Path(member.name).parts
            if (
                not parts
                or parts[0] != run_id
                or ".." in parts
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("Unsafe result archive")
        bundle.extractall(output, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def run_remote(cloud, args, source):
    history = validate_history(read_json(args.history))
    if args.question is not None:
        questions = [{"id": "adhoc", "question": args.question}]
    else:
        if args.questions is None:
            raise ValueError("Supply --questions or --question")
        questions = validate_questions(read_json(args.questions))
    if args.question_id:
        questions = [q for q in questions if q["id"] == args.question_id]
        if not questions:
            raise ValueError("Question ID not found")
    questions = validate_questions({"questions": questions})
    status = cloud.status()
    if status["vm"] != "RUNNING" or status["database"]["state"] != "READY":
        raise RuntimeError("Environment is stopped or starting; run dev start first")
    cloud.control("touch")
    run_id = uuid.uuid4().hex
    name = safe_name(args.name or "run")
    output = private_dir(args.output / (name + "-" + run_id))
    save_json(output / "history.json", history)
    save_json(output / "questions.json", {"questions": questions})
    job = {
        "run_id": run_id,
        "history": history,
        "questions": inference_questions(questions),
        "mode": args.mode,
        "max_output_tokens": args.max_output_tokens,
        "cache_epoch": args.cache_epoch,
        "config": cloud.config,
    }
    archive = output / "upload.tar.gz"
    source_bundle(source, archive, job)
    upload = "/tmp/tiered-dev-" + run_id + ".tar.gz"
    cloud.gcloud(
        [
            "compute",
            "scp",
            str(archive),
            cloud.config["vm"] + ":" + upload,
            "--zone=" + cloud.config["zone"],
            "--tunnel-through-iap",
        ],
        timeout=180,
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(f"Run {run_id}: results will be saved in {output}.", flush=True)
    # Bootstrap returns an export even when the worker fails or reaches a replay miss.
    remote = " ".join(
        shlex.quote(s) for s in ("python3", "/opt/tiered-dev/bootstrap.py", upload, digest, run_id)
    )
    run_started = time.time()
    response = cloud.ssh(remote, timeout=int(cloud.config["max_hours"] * 3600 + 600))
    match = re.search(
        r"^DEV_RESULT (/(?:home/[^/]+|root)/\.tiered-dev/runs/([0-9a-f]{32})\.tar\.gz) ([0-9a-f]{64})$",
        response,
        re.MULTILINE,
    )
    if not match or match[2] != run_id:
        raise RuntimeError(
            f"No verified export receipt; inspect ~/.tiered-dev/runs/{run_id} on the VM"
        )
    export = output / "export.tar.gz"
    cloud.gcloud(
        [
            "compute",
            "scp",
            cloud.config["vm"] + ":" + match[1],
            str(export),
            "--zone=" + cloud.config["zone"],
            "--tunnel-through-iap",
        ],
        timeout=180,
    )
    collect_archive(export, match[3], output, run_id)
    remote_output = output / run_id
    rows = []
    for question in questions:
        answer = remote_output / "answers" / (question["id"] + ".json")
        if answer.exists():
            result = read_json(answer)
            rows.append(
                {
                    "id": question["id"],
                    "answer": result["answer"],
                    "answer_seconds": result["answer_seconds"],
                    "score": score(question, result["answer"]),
                }
            )
    report = {
        "run_id": run_id,
        "questions": rows,
        "status": read_json(remote_output / "status.json"),
        "api_usage": read_json(remote_output / "usage.json"),
        "infrastructure": {
            "scope": "Separate shared uptime estimates, before credits.",
            "estimated_vm_plus_db_usd_per_hour": 0.37837142,
            "run_wall_seconds": time.time() - run_started,
            "estimated_run_share_usd": (time.time() - run_started) / 3600 * 0.37837142,
            "excludes": "Startup/idle time, storage, network and concurrent-run overlap; not a billing total.",
            "existing_valkey_usd_per_day": 0.7632,
        },
    }
    save_json(output / "report.json", report)
    calls = report["api_usage"]["calls"]
    hits = report["api_usage"]["cache_hits"]
    print(
        f"{report['status']['status']}: {len(rows)} answers, {len(calls)} new provider calls, {len(hits)} cache hits."
    )
    for row in rows:
        check = (
            ""
            if row["score"] is None
            else ("; exact match" if row["score"]["correct"] else "; exact match failed")
        )
        print(f"{row['id']} ({row['answer_seconds']:.2f}s{check}): {row['answer']}")
    totals = {
        key: sum((c.get("usage_metadata") or {}).get(key) or 0 for c in calls)
        for key in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")
    }
    print(
        f"Reported new tokens: {totals['prompt_token_count']} input, "
        f"{totals['candidates_token_count']} output, {totals['thoughts_token_count']} thinking."
    )
    print(f"Full results: {output / 'report.json'}")
    return 0 if report["status"]["status"] == "completed" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("dev.config.json"))
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "stop", "status"):
        sub.add_parser(action)
    run = sub.add_parser("run")
    run.add_argument("--history", type=Path, required=True)
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--questions", type=Path)
    selection.add_argument("--question")
    run.add_argument("--question-id")
    run.add_argument("--name")
    run.add_argument("--mode", choices=["cached", "replay", "fresh"], default="cached")
    run.add_argument("--max-output-tokens", type=int, default=65536)
    run.add_argument("--cache-epoch", default="1")
    run.add_argument("--output", type=Path, default=Path("dev_runs"))
    args = parser.parse_args(argv)
    try:
        cloud = Cloud(load_config(args.config))
        if args.action == "run":
            if not 1 <= args.max_output_tokens <= 65536:
                raise ValueError("--max-output-tokens must be 1-65536")
            return run_remote(cloud, args, Path(__file__).resolve().parents[1])
        print(json.dumps(getattr(cloud, args.action)(), indent=2))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"dev: {error}\n")
