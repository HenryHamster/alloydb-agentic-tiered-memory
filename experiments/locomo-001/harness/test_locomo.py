"""Offline checks: inputs, label isolation, adapter write-back isolation, judge prompt, scoring."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import judge_locomo  # noqa: E402
import prepare_locomo  # noqa: E402
import score_locomo  # noqa: E402

from devtools.inputs import inference_questions, validate_questions  # noqa: E402
from devtools.worker import answer_query, retrieval_query  # noqa: E402

DATA = ROOT / "datasets/locomo-src/data/locomo10.json"
INPUTS = HERE / "inputs"
needs_kit = pytest.mark.skipif(
    not (DATA.exists() and INPUTS.exists()), reason="needs the benchmark kit's dataset and inputs"
)
# The instruction locomo-001 appended to every question (results/inputs-manifest.json).
LOCOMO_001_INSTRUCTION = (
    "\n\nAnswer with a short phrase of at most 6 words, based only on what was said in the "
    "conversation. If the answer is a time, give the specific date, month or year."
)


def sample(sample_id="conv-41"):
    """A two-session LoCoMo-shaped conversation with one question per category."""
    turns = [
        {"speaker": "John", "dia_id": "D1:1", "text": "I got promoted!"},
        {"speaker": "Maria", "dia_id": "D1:2", "text": "Congrats!", "blip_caption": "a cake"},
    ]
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "John",
            "speaker_b": "Maria",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": turns,
            "session_2_date_time": "10:37 am on 27 June, 2023",
            "session_2": [{"speaker": "John", "dia_id": "D2:1", "text": "Back from Rome."}],
        },
        "qa": [
            {"question": f"Q{c}?", "answer": "a", "category": c, "evidence": ["D1:1"]}
            for c in (1, 2, 3, 4, 5)
        ],
    }


def test_questions_carry_the_question_alone_and_the_fixed_instructions():
    _, questions, references, _ = prepare_locomo.build(sample())
    assert [q["id"] for q in questions] == [f"conv-41-q{i:03d}" for i in range(4)]
    for q in questions:
        assert set(q) == {"id", "question", "instructions"}
        assert q["instructions"] == prepare_locomo.INSTRUCTIONS
        assert q["question"] == references[q["id"]]["question"]
    assert prepare_locomo.INSTRUCTIONS == (
        "Answer with a short phrase of at most 6 words, based only on what was said in the "
        "conversation. If the answer is a time, give the specific date, month or year. "
        "When the question asks what would or might likely be true, give the most likely "
        "answer the memory supports instead of saying it is not mentioned."
    )
    # The developer runner accepts them, searches with the question alone and answers with
    # question + one blank line + instructions.
    validate_questions({"questions": questions})
    q = inference_questions(questions)[0]
    assert q == questions[0]
    assert retrieval_query(q) == "Q1?"
    assert answer_query(q) == "Q1?\n\n" + prepare_locomo.INSTRUCTIONS


def test_the_split_keeps_the_locomo_001_question_text_byte_identical():
    # With locomo-001's instruction alone, the runner's prompt is the question text that run
    # used: no extra blank line from moving the instruction into its own field.
    q = {"question": "When did Caroline go?", "instructions": LOCOMO_001_INSTRUCTION.lstrip()}
    assert answer_query(q) == "When did Caroline go?" + LOCOMO_001_INSTRUCTION
    assert prepare_locomo.INSTRUCTIONS.startswith(LOCOMO_001_INSTRUCTION.lstrip() + " ")


def test_any_conversation_can_be_prepared(tmp_path):
    dataset = tmp_path / "locomo10.json"
    dataset.write_text(json.dumps([sample("conv-26"), sample("conv-41")]), encoding="utf-8")
    assert prepare_locomo.select(json.loads(dataset.read_text()))[0] == 0
    assert prepare_locomo.select(json.loads(dataset.read_text()), "1")[1]["sample_id"] == "conv-41"
    with pytest.raises(SystemExit, match="conv-99"):
        prepare_locomo.select(json.loads(dataset.read_text()), "conv-99")

    prepare_locomo.main(dataset, tmp_path / "out", "conv-41")
    out = tmp_path / "out"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_id"] == "conv-41" and manifest["sample_index"] == 1
    assert manifest["questions"] == 4 and manifest["messages"] == 3
    questions = json.loads((out / "questions-conv-41.json").read_text(encoding="utf-8"))
    case = json.loads((out / "case-conv-41.json").read_text(encoding="utf-8"))
    assert questions["questions"] == case["questions"]
    runtime = (out / "questions-conv-41.json").read_text() + (out / "case-conv-41.json").read_text()
    assert '"answer"' not in runtime and "evidence" not in runtime and "category" not in runtime


@needs_kit
def test_inputs_lossless_and_label_free():
    sample = json.loads(DATA.read_text(encoding="utf-8"))[0]
    hist = json.loads((INPUTS / "history-conv-26.json").read_text(encoding="utf-8"))
    case = json.loads((INPUTS / "case-conv-26.json").read_text(encoding="utf-8"))
    conv = sample["conversation"]
    turns = [t for k in prepare_locomo.sessions_of(conv) for t in conv[k]]
    msgs = [m for s in hist["sessions"] for m in s["messages"]]
    assert len(msgs) == len(turns) == 419
    for t, m in zip(turns, msgs):
        assert m["content"].startswith(t["speaker"] + ": " + t["text"])
    # Identical message objects for both systems.
    assert [s["messages"] for s in hist["sessions"]] == case["sessions"]
    qs = json.loads((INPUTS / "questions-conv-26.json").read_text(encoding="utf-8"))["questions"]
    assert qs == case["questions"]
    refs = json.loads((INPUTS / "references-conv-26.json").read_text(encoding="utf-8"))
    assert len(qs) == len(refs) == 152 and all(r["category"] != 5 for r in refs.values())
    runtime = json.dumps([hist, qs, case])
    assert '"answer"' not in runtime and "evidence" not in runtime and "category" not in runtime
    # Gold answers never embedded in question text.
    for q in qs:
        assert set(q) == {"id", "question", "instructions"}
        assert q["instructions"] == prepare_locomo.INSTRUCTIONS


def test_dates():
    assert (
        prepare_locomo.parse_date("1:56 pm on 8 May, 2023").isoformat()
        == "2023-05-08T13:56:00+00:00"
    )
    assert prepare_locomo.parse_date("10:37 am on 27 June, 2023").hour == 10


@pytest.mark.skipif(
    not (ROOT / "datasets/mem0-judge").exists(), reason="needs the benchmark kit's judge source"
)
def test_judge_prompt_verbatim():
    src = (ROOT / "datasets/mem0-judge/llm_judge-v0.1.115.py").read_text(encoding="utf-8")
    original = re.search(r'ACCURACY_PROMPT = """(.*?)"""', src, re.S).group(1)
    assert judge_locomo.ACCURACY_PROMPT == original
    assert judge_locomo.parse_label('{"label": "CORRECT"}') == "CORRECT"
    assert judge_locomo.parse_label("reason... WRONG") == "WRONG"
    assert judge_locomo.parse_label("CORRECT or WRONG") is None


def test_f1_and_mcnemar():
    assert score_locomo.f1("7 May 2023", "7 May 2023") == 1.0
    assert score_locomo.f1("", "x") == 0.0
    assert abs(score_locomo.f1("on 7 May", "7 May 2023") - 2 * (2 / 3) * (2 / 3) / (4 / 3)) < 1e-9
    assert score_locomo.mcnemar_exact(0, 0) == 1.0
    assert abs(score_locomo.mcnemar_exact(0, 6) - 2 / 64) < 1e-12
    assert score_locomo.mcnemar_exact(3, 3) == 1.0


def test_items_blind_and_score(tmp_path):
    refs = {
        f"q{i}": {"question": f"Q{i}", "answer": "a", "category": 1 + i % 2, "category_name": "x"}
        for i in range(4)
    }
    po = {q: {"answer": "a", "answer_seconds": 1} for q in refs}
    pp = {q: {"answer": "b", "answer_seconds": 1} for q in refs}
    for name, v in (("refs", refs), ("po", po), ("pp", pp)):
        (tmp_path / f"{name}.json").write_text(json.dumps(v))
    score_locomo.items(
        tmp_path / "refs.json",
        tmp_path / "po.json",
        tmp_path / "pp.json",
        tmp_path / "items.json",
        tmp_path / "key.json",
    )
    items = json.loads((tmp_path / "items.json").read_text())
    assert all(set(i) == {"item_id", "question", "gold", "generated"} for i in items)
    assert "original" not in json.dumps(items) and "pr13" not in json.dumps(items)
    key = json.loads((tmp_path / "key.json").read_text())
    results = {
        "results": [
            {"item_id": k, "label": "CORRECT" if v["system"] == "original" else "WRONG"}
            for k, v in key.items()
        ]
    }
    (tmp_path / "judge.json").write_text(json.dumps(results))
    score_locomo.score(
        tmp_path / "refs.json",
        tmp_path / "judge.json",
        tmp_path / "key.json",
        tmp_path / "scores.json",
        tmp_path / "po.json",
        tmp_path / "pp.json",
    )
    s = json.loads((tmp_path / "scores.json").read_text())["overall"]
    assert s["original_correct"] == 4 and s["pr13_correct"] == 0 and s["only_original"] == 4
    assert s["original_f1"] == 1.0 and s["pr13_f1"] == 0.0


@needs_kit
def test_original_adapter_answer_phase_writes_nothing(tmp_path, monkeypatch):
    """Answer phase must not call extraction or Valkey appends; ingestion covers all messages."""
    pytest.importorskip("fcntl")
    import run_locomo_original as adapter

    calls = {"append": 0, "extract": 0, "turn": 0}

    class Worker:
        def __init__(self, *a):
            pass

        def enqueue_extraction(self, *a):
            calls["extract"] += 1

    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a):
            pass

        def fetchall(self):
            return [("e", "s", "global", None, True)]

    class Conn:
        def cursor(self):
            return Cur()

        def close(self):
            pass

    def append(*a, **k):
        calls["append"] += 1

    def run_agent_turn(conn, cache, worker, client, user_id, session, query, prev_session_id=None):
        calls["turn"] += 1
        worker.enqueue_extraction(user_id, session, query, "ans", prev_session_id)
        env.agent.append_session_turn_with_rolling_summary(None, None, session, query, "ans", 10, 4)
        return "ans", "prompt", 0.1

    env = SimpleNamespace(
        db=SimpleNamespace(
            get_valkey_client=lambda: SimpleNamespace(ping=lambda: True), get_db_connection=Conn
        ),
        worker=Worker,
        client=None,
        buffer=SimpleNamespace(
            append_session_turn_with_rolling_summary=append, IN_MEMORY_VALKEY_FALLBACK=False
        ),
        agent=SimpleNamespace(
            run_agent_turn=run_agent_turn, append_session_turn_with_rolling_summary=append
        ),
        errors=SimpleNamespace(check=lambda: None),
        ledger=SimpleNamespace(context={}),
    )
    case = json.loads((INPUTS / "case-conv-26.json").read_text(encoding="utf-8"))
    a = SimpleNamespace(run_id="t", output=tmp_path, questions_limit=3, workers=2)
    adapter.ingest(a, env, case)
    ing = json.loads((tmp_path / "ingest.json").read_text())
    assert ing["messages"] == 419 and ing["exchanges"] == 215
    assert calls["append"] == calls["extract"] == 215
    adapter.answer(a, env, case)
    assert calls["turn"] == 3 and calls["append"] == 215 and calls["extract"] == 215
    assert len(list((tmp_path / "answers").glob("*.json"))) == 3
    answer = json.loads(next((tmp_path / "answers").glob("*.json")).read_text())
    assert answer["question"].endswith("\n\n" + prepare_locomo.INSTRUCTIONS)
