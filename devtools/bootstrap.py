"""Root-owned entry point, executed as the developer, never sudo. Receives a source bundle."""

import argparse
import hashlib
import os
import subprocess
import tarfile
from pathlib import Path

from devtools.files import locked, private_dir, read_json, save_json
from devtools.lifecycle import Lease


def extract_bundle(archive, destination):
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if (
                not target.is_relative_to(destination.resolve())
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("Unsafe source archive member")
        if sum(m.size for m in bundle.getmembers()) > 512 * 1024 * 1024:
            raise ValueError("Source bundle exceeds 512 MiB")
        bundle.extractall(destination, filter="data")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("run_id")
    args = parser.parse_args(argv)
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != args.sha256:
        parser.error("Upload checksum mismatch")
    root = private_dir(Path.home() / ".tiered-dev")
    import re

    run_id = args.run_id
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        parser.error("Invalid generated run ID")
    workspace = root / "work" / run_id
    workspace.mkdir(parents=True, exist_ok=False, mode=0o700)
    with Lease():  # Includes dependency setup and every child call, even if worker code changes.
        output = private_dir(root / "runs" / run_id)
        return_code = 2
        try:
            execute(args, root, workspace, output, run_id)
            return_code = 0
        except (Exception, SystemExit) as error:
            save_json(output / "bootstrap-error.json", {"error_type": type(error).__name__})
            if not (output / "status.json").exists():
                save_json(
                    output / "status.json", {"status": "failed", "error_type": type(error).__name__}
                )
            if not (output / "usage.json").exists():
                save_json(output / "usage.json", {"calls": [], "cache_hits": []})
        status_path = output / "status.json"
        status = read_json(status_path) if status_path.exists() else {"status": "failed"}
        if status.get("status") == "running":
            status.update(status="failed", error_type="WorkerInterrupted")
        save_json(status_path, status)
        if not (output / "usage.json").exists():
            save_json(output / "usage.json", {"calls": [], "cache_hits": []})
        if not (output / "worker-exit.json").exists():
            save_json(output / "worker-exit.json", {"returncode": return_code})
        archive = root / "runs" / (run_id + ".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(output, arcname=run_id)
        args.archive.unlink(missing_ok=True)
        print(
            "DEV_RESULT " + str(archive) + " " + hashlib.sha256(archive.read_bytes()).hexdigest(),
            flush=True,
        )
        return 0


def execute(args, root, workspace, output, run_id):
    extract_bundle(args.archive, workspace)
    source = workspace / "source"
    manifest = read_json(workspace / "manifest.json")
    for name, expected in manifest["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Unsafe manifest path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source manifest mismatch")
    job = read_json(workspace / "job.json")
    job["run_id"] = run_id
    save_json(workspace / "job.json", job)
    environment = root / "envs" / hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest()
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(environment))
    # No personal ADC file or exported bearer token may override the attached VM identity.
    for name in list(env):
        if name in {
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        } or name.startswith("CLOUDSDK_AUTH_"):
            del env[name]
    with locked(root / "envs" / (environment.name + ".lock")):
        subprocess.run(
            [
                "/usr/local/bin/uv",
                "sync",
                "--locked",
                "--no-dev",
                "--no-install-project",
                "--project",
                str(source),
            ],
            check=True,
            env=env,
            timeout=300,
        )
    output = private_dir(root / "runs" / run_id)
    save_json(output / "source-manifest.json", manifest)
    try:
        result = subprocess.run(
            [
                str(environment / "bin/python"),
                "-m",
                "devtools.worker",
                "--job",
                str(workspace / "job.json"),
                "--cache",
                str(root / "cache"),
                "--output",
                str(output),
            ],
            cwd=source,
            env=env,
            timeout=12000,
        )
        return_code = result.returncode
    except subprocess.TimeoutExpired:
        save_json(output / "bootstrap-error.json", {"error": "worker_timeout"})
        return_code = 2
    save_json(output / "worker-exit.json", {"returncode": return_code})


if __name__ == "__main__":
    raise SystemExit(main())
