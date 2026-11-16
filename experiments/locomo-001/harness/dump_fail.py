import json
import os
import sys

base = sys.argv[1]
run = os.path.join(
    base,
    "attempts/pr13/locomo-pr13-4f0636ef11594dabb456b068e1204dc9/4f0636ef11594dabb456b068e1204dc9",
)
scores = json.load(open(os.path.join(base, "scores.json")))["per_question"]
refs = json.load(open(os.path.join(base, "inputs/references-conv-26.json")))
hist = json.load(open(os.path.join(base, "inputs/history-conv-26.json")))
st = json.load(open(os.path.join(run, "prepared-state.json")))
docs = st["documents"]
facts = st["memory_facts"]
obs = st["observations"]
sess = {s["id"]: s["messages"] for s in hist["sessions"]}


def ev_episodes(ev):
    out = []
    for e in ev:
        try:
            s, t = e[1:].split(":")
            msg = sess[f"session_{s}"][int(t) - 1]
        except Exception:
            out.append((e, None, "?"))
            continue
        c = msg["content"][:60]
        eps = sorted(
            {d["episode_id"] for d in docs if c in d["text"].replace('\\"', '"') or c in d["text"]}
        )
        out.append((e, eps, msg["timestamp"][:10] + " " + msg["content"][:220]))
    return out


fails = [q for q in scores if not q["pr13"]]
print("PR13 failures:", len(fails))
for q in fails:
    qid = q["question_id"]
    a = json.load(open(os.path.join(run, "answers", qid + ".json")))
    rec = a["context"]["recall"]
    items = rec["items"]
    ids = {i["id"] for i in items}
    eps_recalled = {i.get("source_episode_id") for i in items}
    print("=" * 100)
    print(f"{qid} [{q['category_name']}] winner_orig={q['original']}")
    print("Q:", q["question"])
    print("GOLD:", q["gold"])
    print("PR13:", q["pr13_answer"])
    print(
        "abstained:",
        rec.get("abstained"),
        "n_items:",
        len(items),
        "stages:",
        [t["stage"] for t in rec["trace"]],
    )
    for e, eps, txt in ev_episodes(refs[qid]["evidence"]):
        print(f"  EV {e}: {txt}")
        for ep in eps or []:
            fs = [f for f in facts if f["source_episode_id"] == ep]
            print(
                f"     episode {ep.split(':', 2)[-1]} recalled_episode={ep in eps_recalled} facts={len(fs)}"
            )
            for f in fs:
                print(
                    f"       {'R' if f['fact_id'] in ids else '-'} [{(f.get('occurred_at') or '')[:10]}|{(f.get('valid_from') or '')[:10]}] {f['text'][:170]}"
                )
    print("  RECALLED:")
    for i in items:
        d = i.get("dates") or {}
        print(
            f"    {i['type'][:4]} {i.get('relevance')} [{(d.get('occurred_at') or '')[:10]}|{(d.get('valid_from') or '')[:10]}] {i['text'][:150]}"
        )
