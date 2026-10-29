#!/bin/bash
# One-time administrator install on a Debian VM. No IAM grants or cloud starts here.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: sudo bash deploy/install-dev-vm.sh /path/to/filled-dev-vm.json' >&2
  exit 2
fi
source_root=$(dirname "$(dirname "$(readlink -f "$0")")")
config_path=$(readlink -f "$1")
python3 - "$config_path" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
for key in ('project','zone','vm','region','db_cluster','db_instance','generation_location',
            'DB_HOST','DB_NAME','DB_USER','VALKEY_HOST','DB_PORT','VALKEY_PORT'):
    assert p.get(key) and not str(p[key]).startswith('FILL_'), 'Fill '+key
assert 5 <= p['idle_minutes'] <= 120
assert .25 <= p['max_hours'] <= 8
assert not any('password' in k.lower() or 'token' in k.lower() or 'credential' in k.lower() for k in p)
PY
command -v gcloud >/dev/null
command -v python3 >/dev/null
if [[ ! -x /usr/local/bin/uv ]]; then
  echo 'Install uv from its official source at /usr/local/bin/uv before running this installer.' >&2
  exit 2
fi
install -d -m 755 /opt/tiered-dev/devtools /etc/tiered-dev /var/lib/tiered-dev
install -d -m 1777 /var/lib/tiered-dev/activity
install -d -m 700 /var/lib/tiered-dev/gcloud
touch /var/lib/tiered-dev/lifecycle.lock
chmod 666 /var/lib/tiered-dev/lifecycle.lock
install -m 644 "$config_path" /etc/tiered-dev/config.json
for module in __init__ files lifecycle bootstrap; do
  install -m 644 "$source_root/devtools/$module.py" "/opt/tiered-dev/devtools/$module.py"
done
cat >/opt/tiered-dev/control.py <<'PY'
from devtools.lifecycle import main
main()
PY
cat >/opt/tiered-dev/bootstrap.py <<'PY'
from devtools.bootstrap import main
raise SystemExit(main())
PY
chmod 644 /opt/tiered-dev/control.py /opt/tiered-dev/bootstrap.py
cat >/etc/systemd/system/tiered-dev-idle.service <<'UNIT'
[Unit]
Description=Stop idle developer VM and AlloyDB together
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /opt/tiered-dev/control.py tick
TimeoutStartSec=110
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/tiered-dev
Environment=PYTHONDONTWRITEBYTECODE=1
UNIT
cat >/etc/systemd/system/tiered-dev-idle.timer <<'UNIT'
[Unit]
Description=Check idle developer jobs and maximum session duration
[Timer]
OnBootSec=20s
OnUnitInactiveSec=30s
Unit=tiered-dev-idle.service
[Install]
WantedBy=timers.target
UNIT
systemctl daemon-reload
systemctl enable --now tiered-dev-idle.timer
systemctl start tiered-dev-idle.service
echo 'Installed. Check systemctl status tiered-dev-idle.timer before granting developer access.'
