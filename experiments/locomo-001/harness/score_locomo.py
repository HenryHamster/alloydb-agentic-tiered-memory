"""Local LoCoMo scoring: blinded judge items, token F1, per-category accuracy, paired tests.

  score_locomo.py collect-pr13 RUN_DIR... --out predictions/pr13.json
  score_locomo.py collect-original ANSWERS_DIR --out predictions/original.json
  score_locomo.py items REFS PRED_A PRED_B --out judge-items.json --key judge-key.json
  score_locomo.py score REFS JUDGE_RESULTS KEY --out scores.json

Systems are named only in the key; judge items are shuffled (seed 20261108) and anonymous.
Token F1 follows mem0 v0.1.115 metrics/utils.py: lower-case, split on whitespace after
replacing . , ! ? with spaces, set overlap.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

SEED = 20261108
SYSTEMS = ("original", "pr13")


def tokens(text):
    text = str(text)
    for ch in ".,!?":
        text = text.replace(ch, " ")
    return set(text.lower().split())


def f1(prediction, reference):
    p, r = tokens(prediction or ""), tokens(reference or "")
    if not p or not r:
        return 0.0
    common = len(p & r)
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def mcnemar_exact(b, c):
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_diff(pairs, n=10000, seed=SEED):
    """95% percentile CI of accuracy(pr13) - accuracy(original), resampling questions."""
    rng = random.Random(seed)
    diffs = []
    m = len(pairs)
    for _ in range(n):
        s = [pairs[rng.randrange(m)] for _ in range(m)]
        diffs.append(sum(b - a for a, b in s) / m)
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def collect_pr13(run_dirs, out):
    preds = {}
    for d in run_dirs:
        report = read(Path(d) / "report.json")
        for row in report["questions"]:
            if row["id"] in preds:
                raise ValueError("duplicate prediction " + row["id"])
            preds[row["id"]] = {
                "answer": row["answer"],
                "answer_seconds": row["answer_seconds"],
                "run": Path(d).name,
            }
    write(out, preds)
    print(len(preds), "pr13 predictions")


def collect_original(answers_dir, out):
    preds = {}
    for p in sorted(Path(answers_dir).glob("*.json")):
        r = read(p)
        preds[r["id"]] = {"answer": r["answer"], "answer_seconds": r["answer_seconds"]}
    write(out, preds)
    print(len(preds), "original predictions")


def items(refs, pred_original, pred_pr13, out, key_out):
    refs = read(refs)
    preds = {"original": read(pred_original), "pr13": read(pred_pr13)}
    rows = []
    for qid, ref in refs.items():
        for system in SYSTEMS:
            if qid not in preds[system]:
                raise ValueError(f"missing {system} prediction {qid}")
            rows.append((system, qid, preds[system][qid]["answer"]))
    random.Random(SEED).shuffle(rows)
    judge, key = [], {}
    for i, (system, qid, answer) in enumerate(rows):
        item_id = f"j{i:04d}"
        judge.append(
            {
                "item_id": item_id,
                "question": refs[qid]["question"],
                "gold": refs[qid]["answer"],
                "generated": answer,
            }
        )
        key[item_id] = {"system": system, "question_id": qid}
    write(out, judge)
    write(key_out, key)
    print(len(judge), "judge items")


def score(refs, judge_results, key, out, pred_original=None, pred_pr13=None):
    refs, key = read(refs), read(key)
    labels = {r["item_id"]: r["label"] for r in read(judge_results)["results"]}
    preds = {}
    if pred_original and pred_pr13:
        preds = {"original": read(pred_original), "pr13": read(pred_pr13)}
    table = defaultdict(dict)
    for item_id, k in key.items():
        if labels.get(item_id) is None:
            raise ValueError("unjudged or unparsed item " + item_id)
        table[k["question_id"]][k["system"]] = {
            "item_id": item_id,
            "correct": labels[item_id] == "CORRECT",
        }
    per_question = []
    for qid, ref in refs.items():
        row = {
            "question_id": qid,
            "category": ref["category"],
            "category_name": ref["category_name"],
            "question": ref["question"],
            "gold": ref["answer"],
        }
        for system in SYSTEMS:
            row[system] = table[qid][system]["correct"]
            if preds:
                row[system + "_answer"] = preds[system][qid]["answer"]
                row[system + "_f1"] = f1(preds[system][qid]["answer"], ref["answer"])
        per_question.append(row)

    def summarize(rows):
        n = len(rows)
        a = sum(r["original"] for r in rows)
        b = sum(r["pr13"] for r in rows)
        only_o = sum(1 for r in rows if r["original"] and not r["pr13"])
        only_p = sum(1 for r in rows if r["pr13"] and not r["original"])
        s = {
            "n": n,
            "original_correct": a,
            "pr13_correct": b,
            "original_accuracy": a / n,
            "pr13_accuracy": b / n,
            "both_correct": sum(1 for r in rows if r["original"] and r["pr13"]),
            "both_wrong": sum(1 for r in rows if not r["original"] and not r["pr13"]),
            "only_original": only_o,
            "only_pr13": only_p,
            "mcnemar_exact_p": mcnemar_exact(only_o, only_p),
        }
        if preds:
            s["original_f1"] = sum(r["original_f1"] for r in rows) / n
            s["pr13_f1"] = sum(r["pr13_f1"] for r in rows) / n
        return s

    result = {"overall": summarize(per_question), "by_category": {}}
    lo, hi = bootstrap_diff([(int(r["original"]), int(r["pr13"])) for r in per_question])
    result["overall"]["bootstrap95_pr13_minus_original"] = [lo, hi]
    for c in sorted({r["category"] for r in per_question}):
        rows = [r for r in per_question if r["category"] == c]
        result["by_category"][f"{c} {rows[0]['category_name']}"] = summarize(rows)
    result["per_question"] = per_question
    result["note"] = (
        "Questions share one conversation and one memory per system, so they are "
        "not independent; McNemar/bootstrap treat questions as exchangeable units "
        "and overstate certainty about other conversations."
    )
    write(out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_question"}, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect-pr13")
    c.add_argument("runs", nargs="+")
    c.add_argument("--out", required=True)
    c = sub.add_parser("collect-original")
    c.add_argument("answers")
    c.add_argument("--out", required=True)
    c = sub.add_parser("items")
    c.add_argument("refs")
    c.add_argument("original")
    c.add_argument("pr13")
    c.add_argument("--out", required=True)
    c.add_argument("--key", required=True)
    c = sub.add_parser("score")
    c.add_argument("refs")
    c.add_argument("judge")
    c.add_argument("key")
    c.add_argument("--out", required=True)
    c.add_argument("--original")
    c.add_argument("--pr13")
    a = p.parse_args()
    if a.cmd == "collect-pr13":
        collect_pr13(a.runs, a.out)
    elif a.cmd == "collect-original":
        collect_original(a.answers, a.out)
    elif a.cmd == "items":
        items(a.refs, a.original, a.pr13, a.out, a.key)
    else:
        score(a.refs, a.judge, a.key, a.out, a.original, a.pr13)


if __name__ == "__main__":
    main()
