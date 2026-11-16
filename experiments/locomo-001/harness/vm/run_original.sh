#!/bin/bash
# usage: run_original.sh ingest|answer   (run as Henry on the VM)
set -euo pipefail
source /opt/codelab/benchmark-kit/cloud.env
export PGCONNECT_TIMEOUT=20
K=/opt/codelab/benchmark-kit/locomo-001
R=/opt/codelab/runs/locomo001_orig
mkdir -p $R
exec python3 $K/leased.py /opt/codelab/venv/bin/python $K/run_locomo_original.py "$1" \
  --baseline /opt/codelab/original-codelab --case $K/case-conv-26.json \
  --output $R --ledger $R/ledger --run-id locomo001_orig --limit-usd 15 --workers 6
