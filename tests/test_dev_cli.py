import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from devtools.bootstrap import extract_bundle
from devtools.cli import collect_archive
from devtools.files import read_json

SOURCE = Path(__file__).resolve().parents[1]


def test_archive_traversal_rejected(tmp_path):
    name = "../escape"
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(name)
        member.size = 1
        stream.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        extract_bundle(archive, tmp_path / "destination")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        collect_archive(archive, digest, tmp_path / "results", "a" * 32)


def test_bootstrap_exports_failed_setup_with_exit_receipt(tmp_path, monkeypatch, capsys):
    from contextlib import nullcontext

    from devtools import bootstrap

    archive = tmp_path / "upload.tar.gz"
    archive.write_bytes(b"not-an-archive")
    monkeypatch.setattr(bootstrap.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "Lease", nullcontext)
    run_id = "b" * 32
    assert (
        bootstrap.main([str(archive), hashlib.sha256(archive.read_bytes()).hexdigest(), run_id])
        == 0
    )
    receipt = capsys.readouterr().out
    assert "DEV_RESULT" in receipt
    export = tmp_path / ".tiered-dev/runs" / (run_id + ".tar.gz")
    with tarfile.open(export) as stream:
        assert json.load(stream.extractfile(run_id + "/status.json"))["status"] == "failed"
        assert json.load(stream.extractfile(run_id + "/worker-exit.json"))["returncode"] == 2
        assert json.load(stream.extractfile(run_id + "/usage.json"))["calls"] == []


def test_collect_worker_pipeline_and_local_scoring_protocol(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dev_fixture import FixtureModels, MemoryBackend

    from devtools.cli import run_remote
    from devtools.files import save_json
    from devtools.worker import run_job

    class FakeCloud:
        config = dict(
            project="fixture",
            vm="fixture-vm",
            zone="us-east1-b",
            region="us-east1",
            db_cluster="cluster",
            db_instance="db",
            generation_location="global",
            max_hours=4,
        )

        def status(self):
            return {"vm": "RUNNING", "database": {"state": "READY"}}

        def control(self, action):
            assert action == "touch"

        def gcloud(self, args, **kwargs):
            if args[3].startswith("fixture-vm:"):
                self.upload = Path(args[2])
            else:
                Path(args[3]).write_bytes(self.export.read_bytes())

        def ssh(self, command, **kwargs):
            source = tmp_path / "uploaded"
            extract_bundle(self.upload, source)
            job = read_json(source / "job.json")
            assert "budget" not in job
            assert not any(
                n.endswith(".json") or ".env" in n
                for n in read_json(source / "manifest.json")["files"]
            )
            assert all(
                "expected_answer" not in q and "expected_tool_call" not in q
                for q in job["questions"]
            )
            output = tmp_path / "remote" / job["run_id"]
            client = SimpleNamespace(models=FixtureModels())
            result = run_job(
                job,
                SOURCE,
                tmp_path / "cache",
                output,
                client,
                client,
                backend_factory=MemoryBackend,
            )
            assert result["status"] == "completed", result
            self.export = tmp_path / "remote.tar.gz"
            with tarfile.open(self.export, "w:gz") as stream:
                stream.add(output, arcname=job["run_id"])
            return (
                "DEV_RESULT /home/fixture/.tiered-dev/runs/"
                + job["run_id"]
                + ".tar.gz "
                + hashlib.sha256(self.export.read_bytes()).hexdigest()
            )

    questions = tmp_path / "questions.json"
    save_json(
        questions,
        {
            "questions": [
                {"id": "movie", "question": "Which movie?", "expected_answer": "Inception"}
            ]
        },
    )
    args = SimpleNamespace(
        history=SOURCE / "fixtures/dev/history.json",
        question=None,
        questions=questions,
        question_id="movie",
        name="protocol",
        output=tmp_path / "results",
        mode="cached",
        max_output_tokens=1024,
        cache_epoch="1",
    )
    assert run_remote(FakeCloud(), args, SOURCE) == 0
    report = read_json(next((tmp_path / "results").glob("*/report.json")))
    assert report["questions"][0]["score"]["correct"]
    assert report["api_usage"]["calls"]
    assert any(
        c.get("usage_metadata", {}).get("thoughts_token_count")
        for c in report["api_usage"]["calls"]
        if c["kind"] == "generate"
    )


def test_cloud_start_waits_for_matching_controller_before_database():
    from devtools.cli import DEFAULTS, TARGETS, Cloud

    config = dict(
        project="fixture",
        zone="us-east1-b",
        vm="vm",
        region="us-east1",
        db_cluster="cluster",
        db_instance="db",
        **DEFAULTS,
    )
    commands = []
    running = False
    ready = False

    def invoke(args, **kwargs):
        nonlocal running, ready
        commands.append(args)
        assert "--project=fixture" in args
        if args[1:4] == ["compute", "instances", "describe"]:
            return json.dumps({"status": "RUNNING" if running else "TERMINATED"})
        if args[1:4] == ["compute", "instances", "start"]:
            running = True
        if args[1:4] == ["alloydb", "instances", "describe"]:
            return json.dumps({"state": "READY" if ready else "STOPPED", "reconciling": False})
        if args[1:4] == ["alloydb", "instances", "update"]:
            ready = True
        if args[1:3] == ["compute", "ssh"]:
            command = args[args.index("--command") + 1]
            if command.endswith("status"):
                return json.dumps(
                    {
                        "target": {k: config[k] for k in TARGETS},
                        "policy": {k: config[k] for k in ("idle_minutes", "max_hours")},
                    }
                )
            return "{}"
        return ""

    result = Cloud(config, invoke=invoke, sleep=lambda _: None).start()
    assert result["database"]["state"] == "READY"
    scheduling = next(c for c in commands if "set-scheduling" in c)
    assert "--max-run-duration=15000s" in scheduling
    update = next(i for i, c in enumerate(commands) if c[1:4] == ["alloydb", "instances", "update"])
    assert any("touch" in " ".join(c) for c in commands[:update])
