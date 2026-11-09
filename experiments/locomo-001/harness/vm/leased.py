"""Run a command while holding a dev idle-controller lease, so the VM is not idled out."""

import subprocess
import sys

sys.path.insert(0, "/opt/tiered-dev")
from devtools.lifecycle import Lease  # noqa: E402

with Lease():
    sys.exit(subprocess.run(sys.argv[1:]).returncode)
