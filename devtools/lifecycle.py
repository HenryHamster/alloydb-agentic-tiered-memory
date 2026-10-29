"""Root-owned idle controller and unprivileged process leases (Linux VM only).

No developer code is executed by the controller. Locks serialize run admission with
shutdown, and /proc identity prevents a stale lease from pinning the VM on PID reuse.
"""

import argparse
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from devtools.files import locked, private_dir, read_json, save_json

STATE = Path("/var/lib/tiered-dev")


def process_start(pid):
    return Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()[19]


def process_alive(lease, owner):
    try:
        pid = int(lease["pid"])
        return (
            Path(f"/proc/{pid}").stat().st_uid == owner
            and process_start(pid) == lease["process_start"]
        )
    except (OSError, ValueError, KeyError, IndexError):
        return False


class Lease:
    def __init__(self, root=STATE, *, interval=15):
        self.root, self.interval = Path(root), interval
        self.path = self.root / "activity" / str(os.getuid()) / (uuid.uuid4().hex + ".json")
        self.stop = threading.Event()

    def write(self, active=True):
        save_json(
            self.path,
            {
                "pid": os.getpid(),
                "process_start": process_start(os.getpid()),
                "active": active,
                "updated_at": time.time(),
                "action": "run",
            },
        )

    def __enter__(self):
        if not (self.root / "lifecycle.lock").exists():
            raise RuntimeError("VM idle controller is not installed")
        with locked(self.root / "lifecycle.lock"):
            state = read_json(self.root / "state.json")
            if state.get("stopping") or time.time() - state.get("checked_at", 0) > 150:
                raise RuntimeError("VM idle controller is stopping or unhealthy; run dev status")
            private_dir(self.path.parent)
            self.write()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        return self

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            self.write()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        with locked(self.root / "lifecycle.lock"):
            self.write(active=False)


def cloud_stop(config, run=subprocess.run):
    """Stop database first. A failed database request must not stop its watchdog VM."""
    gcloud = config.get("gcloud", "gcloud")
    common = ["--project=" + config["project"], "--quiet"]
    clean = {
        k: v
        for k, v in os.environ.items()
        if k != "GOOGLE_APPLICATION_CREDENTIALS" and not k.startswith("CLOUDSDK_AUTH_")
    }
    clean["CLOUDSDK_CONFIG"] = str(STATE / "gcloud")
    state = run(
        [
            gcloud,
            "alloydb",
            "instances",
            "describe",
            config["db_instance"],
            "--cluster=" + config["db_cluster"],
            "--region=" + config["region"],
            "--format=value(state)",
            *common,
        ],
        check=True,
        env=clean,
        stdout=subprocess.PIPE,
        text=True,
        timeout=45,
    ).stdout.strip()
    if state not in {"STOPPED", "STOPPING"}:
        run(
            [
                gcloud,
                "alloydb",
                "instances",
                "update",
                config["db_instance"],
                "--cluster=" + config["db_cluster"],
                "--region=" + config["region"],
                "--activation-policy=NEVER",
                "--async",
                *common,
            ],
            check=True,
            env=clean,
            stdout=subprocess.DEVNULL,
            timeout=45,
        )
    run(
        [
            gcloud,
            "compute",
            "instances",
            "stop",
            config["vm"],
            "--zone=" + config["zone"],
            "--async",
            *common,
        ],
        check=True,
        env=clean,
        stdout=subprocess.DEVNULL,
        timeout=45,
    )


def tick(
    config, *, root=STATE, now=None, boot_id=None, uptime=None, alive=process_alive, stop=cloud_stop
):
    root, now = Path(root), time.time() if now is None else now
    boot_id = boot_id or Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    uptime = float(Path("/proc/uptime").read_text().split()[0]) if uptime is None else uptime
    with locked(root / "lifecycle.lock"):
        state_path = root / "state.json"
        old = read_json(state_path) if state_path.exists() else {}
        if old.get("boot_id") != boot_id:
            old = {
                "boot_id": boot_id,
                "last_active": now,
                "stopping": False,
                "last_stop_request": now - uptime,
            }
        active, requests = [], []
        last = old.get("last_active", now)
        for user in (root / "activity").iterdir():
            if user.is_symlink() or not user.is_dir() or not user.name.isdigit():
                continue
            for path in user.glob("*.json"):
                if path.is_symlink() or path.stat().st_uid != int(user.name):
                    continue
                try:
                    entry = read_json(path)
                    modified = min(path.stat().st_mtime, now)
                    if entry.get("action") == "stop" and modified > old.get("last_stop_request", 0):
                        requests.append(modified)
                    if entry.get("action") == "run":
                        last = max(last, modified)
                        if entry.get("active") and alive(entry, int(user.name)):
                            active.append(path.name)
                except (ValueError, OSError, KeyError):
                    continue
        hard_stop = uptime >= config["max_hours"] * 3600
        requested = bool(requests)
        should_stop = (
            old.get("stopping", False)
            or hard_stop
            or (not active and (requested or now - last >= config["idle_minutes"] * 60))
        )
        state = {
            **old,
            "checked_at": now,
            "last_active": last,
            "active_jobs": len(active),
            "hard_deadline_at": now - uptime + config["max_hours"] * 3600,
        }
        if requested:
            state.update(
                last_stop_request=max(requests), stop_result="busy" if active else "stopping"
            )
        if should_stop:
            state.update(
                stopping=True, stop_reason="maximum_runtime" if hard_stop else "requested_or_idle"
            )
            save_json(
                state_path, state
            )  # Run admission sees stopping before either cloud mutation.
            try:
                stop(config)
                state.pop("stop_error", None)
            except (subprocess.SubprocessError, OSError) as error:
                # Retry on the next tick; do not admit runs into a half-stopped environment.
                state["stop_error"] = type(error).__name__
        save_json(state_path, state, mode=0o644)
        # Non-root users only need read access to controller health, not its credentials/configuration.
        return state


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["tick", "status", "touch", "stop"])
    parser.add_argument("--config", type=Path, default=Path("/etc/tiered-dev/config.json"))
    args = parser.parse_args(argv)
    if args.action == "tick":
        print(json.dumps(tick(read_json(args.config))))
    elif args.action == "status":
        state = read_json(STATE / "state.json")
        config = read_json(args.config)
        state["booting"] = (
            state.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        )
        state["target"] = {
            k: config[k] for k in ("project", "vm", "zone", "db_cluster", "db_instance", "region")
        }
        state["policy"] = {k: config[k] for k in ("idle_minutes", "max_hours")}
        print(json.dumps(state))
    elif args.action == "touch":
        with Lease():
            pass
        print(json.dumps({"touched": True}))
    else:
        path = private_dir(STATE / "activity" / str(os.getuid())) / (uuid.uuid4().hex + ".json")
        save_json(path, {"action": "stop"})
        print(json.dumps({"requested_at": path.stat().st_mtime}))


if __name__ == "__main__":
    main()
