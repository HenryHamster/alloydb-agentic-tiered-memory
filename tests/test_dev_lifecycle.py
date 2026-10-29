import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from devtools.files import read_json, save_json
from devtools.lifecycle import Lease, cloud_stop, process_alive, process_start, tick


@pytest.fixture
def controller(tmp_path):
    (tmp_path / "activity").mkdir()
    return tmp_path


def step(root, now, *, active=False, stop=None, uptime=10):
    return tick(
        {"max_hours": 4, "idle_minutes": 30},
        root=root,
        now=now,
        boot_id="boot",
        uptime=uptime,
        alive=lambda *_: active,
        stop=stop or (lambda _: None),
    )


def entry(root, name, value, modified):
    owner = root.stat().st_uid
    path = root / "activity" / str(owner) / (name + ".json")
    save_json(path, value)
    os.utime(path, (modified, modified))


def test_idle_deadline_and_live_jobs(controller):
    calls = []
    entry(controller, "previous-boot-stop", {"action": "stop"}, 50)
    assert not step(controller, 100).get("stopping")
    entry(controller, "job", {"action": "run", "active": True}, 110)
    entry(controller, "stop", {"action": "stop"}, 120)
    busy = step(controller, 2000, active=True)
    assert busy["stop_result"] == "busy" and not busy["stopping"]
    entry(controller, "job", {"action": "run", "active": False}, 1999)
    assert not step(controller, 2000).get("stopping")  # Busy requests are not queued.
    assert not process_alive({"pid": -1234567, "process_start": "0"}, 0)
    entry(controller, "job", {"action": "run", "active": False}, 2001)
    assert not step(controller, 2002).get("stopping")
    assert step(controller, 3802, stop=calls.append)["stopping"]
    assert len(calls) == 1


def test_hard_deadline_and_failed_shutdown_remain_closed(controller):
    step(controller, 100)
    entry(controller, "job", {"action": "run", "active": True}, 110)

    def fail(_):
        raise OSError("fixture")

    state = step(controller, 120, active=True, uptime=14401, stop=fail)
    assert state["stopping"] and state["stop_error"] == "OSError"
    if os.name == "posix":
        assert (controller / "state.json").stat().st_mode & 0o777 == 0o644
    assert read_json(controller / "state.json")["stopping"]
    assert state["stop_reason"] == "maximum_runtime"
    calls = []
    step(controller, 121, active=True, uptime=100, stop=calls.append)
    assert len(calls) == 1  # Retry even before idle/deadline/request thresholds recur.


def test_database_shutdown_failure_never_stops_vm():
    calls = []

    def fail(args, **kwargs):
        calls.append(args)
        if "describe" in args:
            return SimpleNamespace(stdout="READY")
        raise subprocess.CalledProcessError(1, args)

    config = dict(
        project="fixture",
        vm="vm",
        zone="zone",
        db_instance="db",
        db_cluster="cluster",
        region="region",
    )
    with pytest.raises(subprocess.CalledProcessError):
        cloud_stop(config, run=fail)
    assert len(calls) == 2 and all(c[1] == "alloydb" for c in calls)
    assert "--project=fixture" in calls[0]
    calls.clear()

    def already_stopped(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="STOPPED")

    cloud_stop(config, run=already_stopped)
    assert [c[1] for c in calls] == ["alloydb", "compute"]


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
    checks = 0

    def invoke(args, **kwargs):
        nonlocal running, ready, checks
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
                checks += 1
                return json.dumps(
                    {
                        "checked_at": time.time(),
                        "booting": checks == 1,
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


@pytest.mark.skipif(sys.platform != "linux", reason="VM process identity requires Linux")
def test_real_process_lease_admission_cleanup_and_stopping(controller):
    config = {"max_hours": 4, "idle_minutes": 30}
    tick(config, root=controller, uptime=10, stop=lambda _: pytest.fail("unexpected shutdown"))
    with Lease(controller, interval=0.01) as lease:
        entry = read_json(lease.path)
        assert process_alive(entry, os.getuid())
        assert entry["process_start"] == process_start(os.getpid())
        state = tick(
            config, root=controller, uptime=10, stop=lambda _: pytest.fail("active shutdown")
        )
        assert state["active_jobs"] == 1
    assert not read_json(lease.path)["active"]
    state = tick(config, root=controller, uptime=10, stop=lambda _: None)
    assert state["active_jobs"] == 0
    state["stopping"] = True
    save_json(controller / "state.json", state)
    with pytest.raises(RuntimeError):
        with Lease(controller):
            pytest.fail("admitted during stop")
    entry["process_start"] = "not-the-same-start"
    assert not process_alive(entry, os.getuid())
