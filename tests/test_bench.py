"""Offline checks of the LongMemEval harness on a tiny dataset in LongMemEval's format."""

import json
from datetime import datetime, timezone

import pytest

from bench.embedders import CachedEmbedder, HashingEmbedder
from bench.longmemeval import (
    Harness,
    Instance,
    parse_date,
    retrieval_metrics,
    run_all,
)
from bench.qa import judge_prompt
from recall import InMemoryRecallStore
from retain import InMemoryRetainStore


def session(user, assistant, has_answer=False):
    turn = {"role": "user", "content": user}
    if has_answer:
        turn["has_answer"] = True
    return [turn, {"role": "assistant", "content": assistant}]


def record(question_id="q1", question="What breed is my dog?", answer_sessions=("s2",)):
    return {
        "question_id": question_id,
        "question_type": "single-session-user",
        "question": question,
        "answer": "Golden retriever",
        "question_date": "2023/06/01 (Thu) 10:00",
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_dates": [
            "2023/05/01 (Mon) 09:00",
            "2023/05/10 (Wed) 12:30",
            "2023/05/20 (Sat) 02:21",
        ],
        "haystack_sessions": [
            session("Can you suggest a pasta recipe for dinner?", "Try a carbonara."),
            session(
                "My dog is a golden retriever named Max.", "Max sounds lovely!", has_answer=True
            )
            + session("He loves the park.", "Parks are great for dogs."),
            session("How do I fix my bike chain?", "Use a chain tool."),
        ],
        "answer_session_ids": list(answer_sessions),
    }


def harness(**kwargs):
    def stores():
        retain_store = InMemoryRetainStore()
        return retain_store, InMemoryRecallStore(retain_store)

    embedder = HashingEmbedder()
    return Harness(doc_embedder=embedder, query_embedder=embedder, make_stores=stores, **kwargs)


def test_dates_parse_as_utc():
    assert parse_date("2023/05/20 (Sat) 02:21") == datetime(2023, 5, 20, 2, 21, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        parse_date("yesterday")


def test_sessions_split_into_rounds_at_each_user_turn():
    instance = Instance.from_json(record())
    assert [(r.session_id, r.index) for r in instance.rounds] == [
        ("s1", 0),
        ("s2", 0),
        ("s2", 1),
        ("s3", 0),
    ]
    assert instance.answer_episode_ids == {"q1:s2:0"}
    assert instance.rounds[1].messages[0].timestamp == datetime(
        2023, 5, 10, 12, 30, tzinfo=timezone.utc
    )
    assert instance.scope.user_id == "lme:q1"


def test_retrieval_metrics():
    metrics = retrieval_metrics(["a", "b", "c"], {"b", "z"}, (1, 3))
    assert metrics["recall_any@1"] == 0.0 and metrics["recall_any@3"] == 1.0
    assert metrics["recall_all@3"] == 0.0
    assert 0 < metrics["ndcg@3"] < 1 and metrics["ndcg@1"] == 0.0
    assert retrieval_metrics(["b"], {"b"}, (1,))["ndcg@1"] == 1.0


def test_a_question_finds_its_answer_session():
    result = harness().run(Instance.from_json(record()))
    assert result["facts"] > 0
    assert result["recall"]["episodes"][0] == "q1:s2:0"
    assert result["retrieval"]["session"]["recall_any@1"] == 1.0
    assert result["retrieval"]["round"]["recall_all@1"] == 1.0


def test_abstention_questions_are_not_scored_for_retrieval():
    result = harness().run(Instance.from_json(record("q2_abs", "What breed is my cat?", ())))
    assert result["abstention"] and result["retrieval"] == {}


def test_run_all_writes_results_summary_and_hypotheses_and_resumes(tmp_path):
    calls = []

    def generate(prompt):
        calls.append(prompt)
        return "yes" if "Answer yes or no" in prompt else "A golden retriever."

    instances = [Instance.from_json(record()), Instance.from_json(record("q2_abs"))]
    out = tmp_path / "run.jsonl"
    summary = run_all(harness(qa_generate=generate), instances, out, workers=2)

    assert summary["overall"]["count"] == 2 and summary["overall"]["qa_accuracy"] == 1.0
    assert summary["overall"]["session"]["count"] == 1
    hypotheses = (tmp_path / "run.hypotheses.jsonl").read_text().splitlines()
    assert {json.loads(line)["question_id"] for line in hypotheses} == {"q1", "q2_abs"}
    assert (tmp_path / "run.summary.json").exists()

    made = len(calls)
    run_all(harness(qa_generate=generate), instances, out)
    assert len(calls) == made and len(out.read_text().splitlines()) == 2


def test_judge_prompts_follow_the_question_type():
    args = ("Q", "A", "R")
    assert "unanswerable" in judge_prompt("temporal-reasoning", *args, abstention=True)
    assert "off-by-one" in judge_prompt("temporal-reasoning", *args, abstention=False)
    assert "updated answer" in judge_prompt("knowledge-update", *args, abstention=False)
    assert "Rubric" in judge_prompt("single-session-preference", *args, abstention=False)


def test_cached_embedder_only_embeds_new_texts(tmp_path):
    seen = []

    def inner(texts):
        seen.extend(texts)
        return HashingEmbedder()(texts)

    cached = CachedEmbedder(inner, str(tmp_path / "cache.sqlite"), "test")
    first = cached(["alpha", "beta", "alpha"])
    second = cached(["beta", "gamma"])
    assert seen == ["alpha", "beta", "gamma"]
    assert first[1] == pytest.approx(second[0], abs=1e-6)
