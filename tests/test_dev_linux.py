"""Linux /proc and kernel-lock smoke, also runnable with python tests/test_dev_linux.py."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from devtools.files import read_json, save_json
from devtools.lifecycle import Lease, process_alive, process_start, tick


@unittest.skipUnless(sys.platform == "linux", "VM process identity requires Linux")
class LinuxLeases(unittest.TestCase):
    def test_real_process_lease_admission_cleanup_and_stopping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "activity").mkdir()
            config = {"max_hours": 4, "idle_minutes": 30}
            tick(config, root=root, uptime=10, stop=lambda _: self.fail("unexpected shutdown"))
            with Lease(root, interval=0.01) as lease:
                entry = read_json(lease.path)
                self.assertTrue(process_alive(entry, os.getuid()))
                self.assertEqual(entry["process_start"], process_start(os.getpid()))
                state = tick(
                    config, root=root, uptime=10, stop=lambda _: self.fail("active shutdown")
                )
                self.assertEqual(state["active_jobs"], 1)
            self.assertFalse(read_json(lease.path)["active"])
            state = tick(config, root=root, uptime=10, stop=lambda _: None)
            self.assertEqual(state["active_jobs"], 0)
            state["stopping"] = True
            save_json(root / "state.json", state)
            with self.assertRaises(RuntimeError):
                with Lease(root):
                    self.fail("admitted during stop")
            entry["process_start"] = "not-the-same-start"
            self.assertFalse(process_alive(entry, os.getuid()))
            print(
                json.dumps(
                    {
                        "linux_lease": "passed",
                        "active_jobs": state["active_jobs"],
                        "checked_at": time.time(),
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()
