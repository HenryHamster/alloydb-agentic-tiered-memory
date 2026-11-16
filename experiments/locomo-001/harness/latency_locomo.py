"""Answer-latency comparison from recorded per-call receipts (no new run).

Inputs: PR13 report.json (per-call seconds + token usage), PR13 answers/*.json (answer_seconds),
the original's ledger usage.jsonl (per-call elapsed_seconds + usage) and answers/*.json.
Method:
  1. Per-question latency is the sum of that question's sequential calls plus local work; the
     original's 6 workers ran different questions in parallel, which changes throughput but not
     one question's critical path, provided per-call speed is unaffected.
  2. Contention check: fit seconds = a + b * (output + thinking tokens) per call group. Groups
     include the isolated PR13 pilot (nothing else running). Similar a, b => overlap and
     parallelism did not slow calls.
  3. Paired per-question comparison over the 151 questions both systems answered live
     (PR13 q000 was a replay-cache hit, excluded); bootstrap over questions.
Usage: latency_locomo.py  (writes the "per_call_analysis" key of latency.json)
"""

import glob
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PR13 = HERE / "attempts/pr13/locomo-pr13-4f0636ef11594dabb456b068e1204dc9"
PILOT = HERE / "attempts/pr13/locomo-pilot-65ea0ce7bffa43a280e300f5d11043dd"
ORIG = HERE / "attempts/original/locomo001_orig"


def gen_tokens(u):
    if "candidates_token_count" in u:
        return (u.get("candidates_token_count") or 0) + (u.get("thoughts_token_count") or 0)
    return (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)


def fit(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    return {
        "n": len(points),
        "fixed_s": round(my - b * mx, 2),
        "s_per_1k_tokens": round(b * 1000, 2),
        "mean_tokens": round(mx),
        "mean_s": round(my, 2),
    }


def summ(xs):
    xs = sorted(xs)
    return {
        "n": len(xs),
        "median": round(st.median(xs), 2),
        "mean": round(st.mean(xs), 2),
        "p90": round(xs[int(0.9 * len(xs))], 2),
    }


def pr13_calls(report):
    return [
        c
        for c in json.loads(report.read_text())["api_usage"]["calls"]
        if c.get("seconds") is not None
    ]


def main():
    out = {}
    calls = pr13_calls(PR13 / "report.json")
    gen = [c for c in calls if c["kind"] == "generate" and c["status"] == "completed"]
    pilot_gen = [
        c
        for c in pr13_calls(PILOT / "report.json")
        if c["kind"] == "generate" and c["status"] == "completed"
    ]
    ev = [json.loads(line) for line in (ORIG / "ledger/usage.jsonl").read_text().splitlines()]
    kind = {(e["operation"], e["id"]): e["kind"] for e in ev if e.get("phase") == "reserved"}
    orig_groups = defaultdict(list)
    orig_all = defaultdict(list)
    for e in ev:
        if e.get("phase") not in ("settled", "completed_reservation_retained"):
            continue
        k = (e["operation"], kind[(e["operation"], e["id"])])
        orig_all[k].append(e["elapsed_seconds"])
        u = e.get("usage_metadata") or e.get("raw_usage_metadata")
        if e["phase"] == "settled" and u:
            orig_groups[k].append((gen_tokens(u), e["elapsed_seconds"]))

    out["call_speed_fits"] = {
        "pr13_pilot_ingest_isolated": fit(
            [(gen_tokens(c["usage_metadata"]), c["seconds"]) for c in pilot_gen]
        ),
        "pr13_answers_during_original_ingest": fit(
            [(gen_tokens(c["usage_metadata"]), c["seconds"]) for c in gen]
        ),
        "original_ingest_sdk_generate": fit(orig_groups[("ingestion", "sdk_generate")]),
        "original_answers_sql_generate_6_workers": fit(orig_groups[("answer", "sql_generate")]),
        "original_answers_sdk_generate_6_workers": fit(orig_groups[("answer", "sdk_generate")]),
    }

    per_q = defaultdict(lambda: defaultdict(float))
    ntok = defaultdict(int)
    for c in calls:
        per_q[c["question_id"]][c["kind"]] += c["seconds"]
    for c in gen:
        ntok[c["question_id"]] += gen_tokens(c["usage_metadata"])
    pa = {
        json.loads(Path(f).read_text())["id"]: json.loads(Path(f).read_text())["answer_seconds"]
        for f in glob.glob(str(PR13 / "extracted/*/answers/*.json"))
    }
    oa = {
        json.loads(Path(f).read_text())["id"]: json.loads(Path(f).read_text())["answer_seconds"]
        for f in glob.glob(str(ORIG / "answers/*.json"))
    }
    ids = sorted(i for i in pa if i in per_q and i in oa)
    n_o = len(oa)
    out["per_question_breakdown"] = {
        "pr13": {
            "generate_calls_per_question": round(len(gen) / len(ids), 2),
            "generated_tokens_per_question": summ([ntok[i] for i in ids]),
            "generate_seconds": summ([per_q[i]["generate"] for i in ids]),
            "ranking_plus_embedding_seconds": summ(
                [per_q[i]["ranking"] + per_q[i]["embedding"] for i in ids]
            ),
            "local_seconds": summ([pa[i] - sum(per_q[i].values()) for i in ids]),
            "answer_seconds": summ([pa[i] for i in ids]),
        },
        "original": {
            "generate_calls_per_question": round(
                (
                    len(orig_all[("answer", "sql_generate")])
                    + len(orig_all[("answer", "sdk_generate")])
                )
                / n_o,
                2,
            ),
            "generated_tokens_per_question_mean": round(
                sum(
                    t
                    for k in (("answer", "sql_generate"), ("answer", "sdk_generate"))
                    for t, _ in orig_groups[k]
                )
                / n_o
            ),
            "model_seconds_per_question_mean": round(
                sum(sum(v) for (op, _), v in orig_all.items() if op == "answer") / n_o, 2
            ),
            "answer_seconds": summ(list(oa.values())),
        },
    }
    rng = random.Random(20261109)
    red_med, red_mean = [], []
    for _ in range(10000):
        s = [rng.choice(ids) for _ in ids]
        red_med.append(1 - st.median(pa[i] for i in s) / st.median(oa[i] for i in s))
        red_mean.append(1 - sum(pa[i] for i in s) / sum(oa[i] for i in s))
    red_med.sort()
    red_mean.sort()
    out["paired"] = {
        "n_questions": len(ids),
        "excluded": sorted(set(pa) - set(ids)),
        "pr13_faster_count": sum(pa[i] < oa[i] for i in ids),
        "median_paired_difference_s": round(st.median(oa[i] - pa[i] for i in ids), 2),
        "reduction_of_median": round(
            1 - st.median(pa[i] for i in ids) / st.median(oa[i] for i in ids), 3
        ),
        "reduction_of_median_ci95": [round(red_med[250], 3), round(red_med[9750], 3)],
        "reduction_of_mean": round(1 - sum(pa[i] for i in ids) / sum(oa[i] for i in ids), 3),
        "reduction_of_mean_ci95": [round(red_mean[250], 3), round(red_mean[9750], 3)],
        "bootstrap": "10,000 question resamples, seed 20261109",
    }
    path = HERE / "latency.json"
    lat = json.loads(path.read_text())
    lat["per_call_analysis"] = out
    path.write_text(json.dumps(lat, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
