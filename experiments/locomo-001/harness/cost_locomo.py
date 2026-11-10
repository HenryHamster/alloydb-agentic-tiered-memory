"""Experiment-wide accounting from recorded usage (not an invoice).

Generation: USD1.50/M prompt, USD9/M output+thinking. Failed PR13 generate calls reserve USD0.01.
Original: ledger accounted_usd minus its infrastructure reserve (embeddings at USD0.25/M bytes).
PR13: embedding calls are counted but not priced (small; listed per report).
Infrastructure: USD0.6/h over recorded cloud sessions (start..verified stop).

Usage: cost_locomo.py accounting.json  (reads the paths listed in that file, writes totals back)
"""

import json
import sys
from pathlib import Path

IN_RATE, OUT_RATE, FAILED_RESERVE, INFRA_PER_HOUR = 1.5e-6, 9e-6, 0.01, 0.6


def pr13_report(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    tok = {"prompt": 0, "output": 0, "thinking": 0}
    stages, failed, kinds = {}, 0, {}
    for c in report["api_usage"]["calls"]:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        if c["kind"] != "generate":
            continue
        if c["status"] != "completed":
            failed += 1
            continue
        u = c.get("usage_metadata") or {}
        p, o, t = (
            u.get(k) or 0
            for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")
        )
        tok["prompt"] += p
        tok["output"] += o
        tok["thinking"] += t
        stage = c.get("stage")
        stages[stage] = stages.get(stage, 0.0) + p * IN_RATE + (o + t) * OUT_RATE
    usd = tok["prompt"] * IN_RATE + (tok["output"] + tok["thinking"]) * OUT_RATE
    return {
        "path": str(path),
        "tokens": tok,
        "calls_by_kind": kinds,
        "failed_generate": failed,
        "generation_usd": usd + failed * FAILED_RESERVE,
        "usd_by_stage": {k: round(v, 6) for k, v in stages.items()},
        "cache_hits": len(report["api_usage"]["cache_hits"]),
        "status": report["status"]["status"],
        "preparation_seconds": report["status"].get("preparation_seconds"),
        "prepared_cache_hit": report["status"].get("prepared_cache_hit"),
    }


def original_ledger(state_path):
    """The run_original ledger state: accounted_usd = settled usage + retained reservations
    (SQL generate keeps its reservation unless the provider meter settled it; embeddings at
    USD0.25/M bytes). Token totals come from settled events in usage.jsonl beside it."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    tok = {"prompt": 0, "output": 0, "thinking": 0}
    events = Path(state_path).parent / "usage.jsonl"
    phases = {}
    if events.exists():
        for line in events.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            phases[e.get("phase")] = phases.get(e.get("phase"), 0) + 1
            if e.get("phase") == "settled":
                u = e.get("usage_metadata") or {}
                tok["prompt"] += u.get("prompt_token_count") or 0
                tok["output"] += u.get("candidates_token_count") or 0
                tok["thinking"] += u.get("thoughts_token_count") or 0
    reserve = state.get("infrastructure_reserve_usd", 0.0)
    return {
        "path": str(state_path),
        "tokens": tok,
        "phases": phases,
        "calls": state["calls"],
        "generation_usd": state["accounted_usd"] - reserve,
    }


def main(spec_path):
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent
    rows = []
    for item in spec.get("pr13_reports", []):
        rows.append({"system": "pr13", **pr13_report(base / item)})
    for item in spec.get("original_ledgers", []):
        rows.append({"system": "original", **original_ledger(base / item)})
    judge = 0.0
    for item in spec.get("judge_results", []):
        judge += json.loads((base / item).read_text(encoding="utf-8")).get("usd", 0.0)
    hours = sum(s["hours"] for s in spec.get("cloud_sessions", []))
    totals = {
        "pr13_generation_usd": sum(r["generation_usd"] for r in rows if r["system"] == "pr13"),
        "original_generation_usd": sum(
            r["generation_usd"] for r in rows if r["system"] == "original"
        ),
        "judge_usd": judge,
        "infrastructure_hours": hours,
        "infrastructure_usd": hours * INFRA_PER_HOUR,
    }
    totals["total_usd"] = sum(v for k, v in totals.items() if k.endswith("_usd"))
    spec["rows"], spec["totals"] = rows, totals
    spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
