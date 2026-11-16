"""Pin one LoCoMo conversation as identical runtime inputs for PR13 and the original codelab.

Usage: python prepare_locomo.py <locomo10.json> <out_dir> [conversation]
where conversation is a sample_id such as conv-26 or conv-41, or a 0-based index in file
order (default conv-26, the first).

Runtime files contain only the conversation and question text. Gold answers, evidence ids and
category labels go to references.json, which is read only by scoring.

Mapping (declared in RECORD.md):
- speaker_a -> role "user", speaker_b -> role "assistant"; every speaker turn is one message,
  in released order, untruncated.
- content = "<Speaker>: <text>", plus " [shares a photo: <blip_caption>]" when the turn has one.
- timestamp = the session's LoCoMo date_time, parsed as UTC (the source gives no zone).
- questions: categories 1-4 only (category 5 excluded, as in Mem0's J score), in file order.
  "question" is the LoCoMo question alone; "instructions" is one fixed answer instruction,
  identical for every question and both systems. PR13's runner searches memory with the
  question only and appends the instructions to the answer prompt after one blank line; the
  original codelab gets the same text, question + "\n\n" + instructions, as its query.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# LoCoMo's answer format. The last sentence was a shared answer rule in PR #19; it belongs to
# this benchmark only (agents in general should say when memory doesn't cover something).
INSTRUCTIONS = (
    "Answer with a short phrase of at most 6 words, based only on what was said in the "
    "conversation. If the answer is a time, give the specific date, month or year. "
    "When the question asks what would or might likely be true, give the most likely answer "
    "the memory supports instead of saying it is not mentioned."
)
DEFAULT_CONVERSATION = "conv-26"
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def parse_date(text):
    # e.g. "1:56 pm on 8 May, 2023"
    return datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)


def sessions_of(conversation):
    keys = [k for k in conversation if k.startswith("session_") and k.count("_") == 1]
    keys = [k for k in keys if isinstance(conversation[k], list) and conversation[k]]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def content(turn):
    text = f"{turn['speaker']}: {turn['text']}"
    if turn.get("blip_caption"):
        text += f" [shares a photo: {turn['blip_caption']}]"
    return text


def build(sample):
    conv = sample["conversation"]
    roles = {conv["speaker_a"]: "user", conv["speaker_b"]: "assistant"}
    sessions, dia_ids = [], {}
    for key in sessions_of(conv):
        when = parse_date(conv[key + "_date_time"]).isoformat()
        messages = []
        for turn in conv[key]:
            dia_ids[turn["dia_id"]] = (key, len(messages))
            messages.append(
                {
                    "role": roles[turn["speaker"]],
                    "name": turn["speaker"],
                    "content": content(turn),
                    "timestamp": when,
                }
            )
        sessions.append({"id": key, "date_time": conv[key + "_date_time"], "messages": messages})
    questions, references = [], {}
    for index, qa in enumerate(sample["qa"]):
        if qa["category"] == 5:
            continue
        qid = f"{sample['sample_id']}-q{index:03d}"
        questions.append({"id": qid, "question": qa["question"], "instructions": INSTRUCTIONS})
        references[qid] = {
            "question": qa["question"],
            "answer": str(qa["answer"]),
            "category": qa["category"],
            "category_name": CATEGORY_NAMES[qa["category"]],
            "evidence": qa.get("evidence", []),
        }
    return sessions, questions, references, dia_ids


def select(data, conversation=DEFAULT_CONVERSATION):
    """(index, sample) for a sample_id ("conv-41") or a 0-based index in file order ("1")."""
    ids = [sample["sample_id"] for sample in data]
    if conversation in ids:
        index = ids.index(conversation)
    elif str(conversation).isdigit() and int(conversation) < len(data):
        index = int(conversation)
    else:
        raise SystemExit(f"Unknown conversation {conversation!r}; choose one of {ids}")
    return index, data[index]


def main(dataset, out_dir, conversation=DEFAULT_CONVERSATION):
    dataset, out = Path(dataset), Path(out_dir)
    data = json.loads(dataset.read_text(encoding="utf-8"))
    sample_index, sample = select(data, conversation)
    out.mkdir(parents=True, exist_ok=True)
    sid = sample["sample_id"]
    sessions, questions, references, _ = build(sample)
    files = {
        # PR13 developer CLI format (devtools/inputs.py).
        f"history-{sid}.json": {
            "id": sid,
            # Both LoCoMo speakers are people: the assistant role is a second human.
            "assistant_is_human": True,
            "sessions": [{"id": s["id"], "messages": s["messages"]} for s in sessions],
        },
        f"questions-{sid}.json": {"questions": questions},
        # Original-codelab adapter format: the same message objects.
        f"case-{sid}.json": {
            "case_id": sid.replace("-", "_"),
            "session_ids": [s["id"] for s in sessions],
            "session_dates": [s["date_time"] for s in sessions],
            "sessions": [s["messages"] for s in sessions],
            "question_date": sessions[-1]["date_time"],
            "questions": questions,
        },
        f"references-{sid}.json": references,  # scoring only
    }
    manifest = {
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "sample_index": sample_index,
        "sample_id": sid,
        "selection_rule": f"conversation {conversation!r} of locomo10.json, fixed before any run",
        "instructions": INSTRUCTIONS,
        "sessions": len(sessions),
        "messages": sum(len(s["messages"]) for s in sessions),
        "questions": len(questions),
        "questions_by_category": {
            str(c): sum(1 for r in references.values() if r["category"] == c) for c in (1, 2, 3, 4)
        },
        "files": {},
    }
    for name, doc in files.items():
        path = out / name
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
