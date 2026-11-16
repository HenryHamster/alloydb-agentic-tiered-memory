"""Record actual LoCoMo statistics (conversations, sessions, turns, QA per category)."""

import collections
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
out = {"conversations": len(data), "per_conversation": []}
tot = collections.Counter()
for d in data:
    c = d["conversation"]
    sess = sorted(
        [
            k
            for k in c
            if k.startswith("session_") and not k.endswith("date_time") and isinstance(c[k], list)
        ],
        key=lambda k: int(k.split("_")[1]),
    )
    turns = sum(len(c[s]) for s in sess)
    imgs = sum(1 for s in sess for t in c[s] if t.get("img_url") or t.get("blip_caption"))
    cats = collections.Counter(q["category"] for q in d["qa"])
    tot.update(cats)
    out["per_conversation"].append(
        {
            "sample_id": d["sample_id"],
            "speakers": [c["speaker_a"], c["speaker_b"]],
            "sessions": len(sess),
            "turns": turns,
            "turns_with_image": imgs,
            "chars": sum(len(t["text"]) for s in sess for t in c[s]),
            "qa": len(d["qa"]),
            "qa_by_category": dict(sorted(cats.items())),
            "first_date": c[sess[0] + "_date_time"],
            "last_date": c[sess[-1] + "_date_time"],
        }
    )
out["qa_by_category_total"] = dict(sorted(tot.items()))
out["qa_total"] = sum(tot.values())
out["sessions_total"] = sum(p["sessions"] for p in out["per_conversation"])
out["turns_total"] = sum(p["turns"] for p in out["per_conversation"])
print(json.dumps(out, indent=1))
