"""First-pass cause of every discordant question, from each system's executed prompt.

Evidence coverage = share of the gold answer's content tokens (mem0 tokenization, stopwords and
months-as-words kept, digits kept) present in the long-term-memory section of that system's prompt (question echo and
session ids excluded).
Rules (checked by hand afterwards, see RECORD.md):
  winner had >=0.5 coverage, loser <0.5            -> loser_memory_missing
  both >=0.5                                       -> loser_answer_stage
  loser abstained and coverage <0.5                -> loser_memory_missing (abstained)
  neither >=0.5                                    -> inference_or_judge (gold needs inference)
Usage: discordant_locomo.py scores.json OUT.json
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PR13 = (
    HERE
    / "attempts/pr13/locomo-pr13-4f0636ef11594dabb456b068e1204dc9/extracted/4f0636ef11594dabb456b068e1204dc9/answers"
)
ORIG = HERE / "attempts/original/locomo001_orig/answers"
STOP = set(
    "a an the and or of to in on at for with by from is was were be her his she he they "
    "it its that this as about their them likely yes no not".split()
)
ABSTAIN = (
    "not mentioned",
    "not specified",
    "no information",
    "not stated",
    "unknown",
    "does not mention",
    "not in the",
    "no mention",
    "cannot",
    "not provided",
)


def toks(text):
    text = str(text).lower()
    for ch in ".,!?;:()'\"":
        text = text.replace(ch, " ")
    return [t for t in text.split() if t not in STOP]


def coverage(gold, memory):
    g = toks(gold)
    if not g:
        return None
    m = set(toks(memory))
    return sum(t in m for t in g) / len(g)


def memory_part(prompt, marker):
    """Long-term memory section only (not the echoed question); session ids and record uuids removed."""
    i = prompt.find(marker)
    j = prompt.find("[SHORT-TERM", i)
    text = prompt[i : j if j > i else None] if i >= 0 else ""
    text = re.sub(r"locomo001_orig_conv_\w+", " ", text)
    return re.sub(r"\[[-0-9a-f]{36}\]", " ", text)


def main(scores_path, out_path):
    scores = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    rows = []
    for r in scores["per_question"]:
        if r["original"] == r["pr13"]:
            continue
        o = json.loads((ORIG / f"{r['question_id']}.json").read_text(encoding="utf-8"))
        p = json.loads((PR13 / f"{r['question_id']}.json").read_text(encoding="utf-8"))
        cov_o = coverage(r["gold"], memory_part(o["prompt"], "[LONG-TERM"))
        cov_p = coverage(r["gold"], memory_part(p["prompt"], "[LONG-TERM"))
        winner = "pr13" if r["pr13"] else "original"
        loser = "original" if winner == "pr13" else "pr13"
        cw, cl = (cov_p, cov_o) if winner == "pr13" else (cov_o, cov_p)
        lose_answer = r[loser + "_answer"]
        abstained = any(a in lose_answer.lower() for a in ABSTAIN)
        cw, cl = cw or 0.0, cl or 0.0
        if cl >= 0.5 and cw >= 0.5:
            cause = "loser_answer_stage"
        elif cw >= 0.5 or (abstained and cl < 0.5):
            cause = "loser_memory_missing"
        else:
            cause = "inference_or_judge"
        rows.append(
            {
                "question_id": r["question_id"],
                "category": r["category_name"],
                "winner": winner,
                "question": r["question"],
                "gold": r["gold"],
                "original_answer": r["original_answer"],
                "pr13_answer": r["pr13_answer"],
                "coverage_original": round(cov_o or 0, 2),
                "coverage_pr13": round(cov_p or 0, 2),
                "loser_abstained": abstained,
                "first_pass_cause": cause,
            }
        )
    summary = {}
    for row in rows:
        k = f"{row['winner']} wins: {row['first_pass_cause']}"
        summary[k] = summary.get(k, 0) + 1
    Path(out_path).write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
