import os
import subprocess

import pytest

from devtools.files import read_json, save_json
from devtools.lifecycle import cloud_stop, process_alive, tick


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
    assert not step(controller, 100).get("stopping")
    entry(controller, "job", {"action": "run", "active": True}, 110)
    assert not step(controller, 2000, active=True).get("stopping")
    entry(controller, "job", {"action": "run", "active": False}, 2001)
    assert not step(controller, 2002).get("stopping")
    assert step(controller, 3802, stop=calls.append)["stopping"]
    assert len(calls) == 1


def test_busy_stop_is_not_queued_and_dead_pid_cannot_hold_vm(controller):
    step(controller, 100)
    entry(controller, "job", {"action": "run", "active": True}, 110)
    entry(controller, "stop", {"action": "stop"}, 120)
    state = step(controller, 121, active=True)
    assert state["stop_result"] == "busy" and not state["stopping"]
    assert not step(controller, 122).get("stopping")
    assert step(controller, 2000)["stopping"]
    assert not process_alive({"pid": -1234567, "process_start": "0"}, 0)


def test_hard_deadline_and_failed_shutdown_remain_closed(controller):
    step(controller, 100)
    entry(controller, "job", {"action": "run", "active": True}, 110)

    def fail(_):
        raise OSError("fixture")

    state = step(controller, 120, active=True, uptime=14401, stop=fail)
    assert state["stopping"] and state["stop_error"] == "OSError"
    assert read_json(controller / "state.json")["stopping"]
    assert state["stop_reason"] == "maximum_runtime"
    calls = []
    step(controller, 121, active=True, uptime=100, stop=calls.append)
    assert len(calls) == 1  # Retry even before idle/deadline/request thresholds recur.


def test_database_shutdown_failure_never_stops_vm():
    calls = []

    def fail(args, **kwargs):
        calls.append(args)
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
    assert len(calls) == 1 and calls[0][1] == "alloydb"
    assert "--project=fixture" in calls[0]
    calls.clear()
    cloud_stop(config, run=lambda args, **_: calls.append(args))
    assert [c[1] for c in calls] == ["alloydb", "compute"]
