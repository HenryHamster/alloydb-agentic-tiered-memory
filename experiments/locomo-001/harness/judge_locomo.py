"""Blinded binary LLM judge (Mem0 J-score prompt, gemini-3.5-flash substituted for gpt-4o-mini).

Input: a judge-items.json built locally by score_locomo.py `items` -- a shuffled list of
{item_id, question, gold, generated}. It carries no system name; the key mapping item_id to
(system, question id) stays local. Output: judge-results.json with raw responses, labels and
usage. Runs on the VM with its attached service account (no credential files).

Usage: python judge_locomo.py ITEMS OUT --project P [--limit-usd X] [--workers W]
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Verbatim ACCURACY_PROMPT from mem0 v0.1.115 evaluation/metrics/llm_judge.py, extracted to a
# file because editors strip its trailing spaces.
ACCURACY_PROMPT = (Path(__file__).resolve().parent / "mem0_accuracy_prompt.txt").read_text(
    encoding="utf-8"
)
MODEL = "gemini-3.5-flash"
IN_RATE, OUT_RATE = 1.5e-6, 9e-6


def parse_label(text):
    """mem0 reads json['label']; fall back to the single label word if JSON is malformed."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if match:
        try:
            label = json.loads(match.group(0)).get("label")
            if label in ("CORRECT", "WRONG"):
                return label
        except ValueError:
            pass
    words = set(re.findall(r"\b(CORRECT|WRONG)\b", text or ""))
    return words.pop() if len(words) == 1 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("items", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--project", required=True)
    p.add_argument("--limit-usd", type=float, required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=a.project, location="global")
    items = json.loads(a.items.read_text(encoding="utf-8"))
    done = {}
    if a.out.exists():
        done = {r["item_id"]: r for r in json.loads(a.out.read_text(encoding="utf-8"))["results"]}
    spent = [sum(r.get("usd", 0) for r in done.values())]
    lock = threading.Lock()

    def one(item):
        if item["item_id"] in done:
            return
        with lock:
            if spent[0] + 0.05 > a.limit_usd:
                raise RuntimeError("judge allowance exhausted")
        prompt = ACCURACY_PROMPT.format(
            question=item["question"], gold_answer=item["gold"], generated_answer=item["generated"]
        )
        t0 = time.perf_counter()
        for attempt in range(4):
            try:
                r = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0, response_mime_type="application/json"
                    ),
                )
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(3 * 2**attempt)
                last = type(e).__name__  # noqa: F841
        u = r.usage_metadata
        tok = {
            "prompt": u.prompt_token_count or 0,
            "output": u.candidates_token_count or 0,
            "thinking": u.thoughts_token_count or 0,
        }
        usd = tok["prompt"] * IN_RATE + (tok["output"] + tok["thinking"]) * OUT_RATE
        row = {
            "item_id": item["item_id"],
            "raw": r.text,
            "label": parse_label(r.text),
            "tokens": tok,
            "usd": usd,
            "seconds": time.perf_counter() - t0,
            "model_version": r.model_version,
            "attempts": attempt + 1,
        }
        with lock:
            done[item["item_id"]] = row
            spent[0] += usd
            a.out.write_text(
                json.dumps(
                    {"model": MODEL, "results": list(done.values())}, indent=1, ensure_ascii=False
                ),
                encoding="utf-8",
            )

    with ThreadPoolExecutor(a.workers) as pool:
        for f in [pool.submit(one, i) for i in items]:
            f.result()
    a.out.write_text(
        json.dumps(
            {"model": MODEL, "results": list(done.values()), "usd": spent[0]},
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "judged": len(done),
                "usd": round(spent[0], 4),
                "unparsed": sum(1 for r in done.values() if r["label"] is None),
            }
        )
    )


if __name__ == "__main__":
    main()
