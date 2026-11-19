"""Dream behavior tests. No credentials, network or paid generation required."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from bench.embedders import HashingEmbedder
from recall import InMemoryRecallStore, Recaller
from recall.config import MemoryType
from reflect.dream import DreamConfig, Dreamer, DreamRequest, LostLease, StaleDream
from reflect.dream_recall import DreamRecallStore
from reflect.dream_store import InMemoryDreamStore
from reflect.pipeline import Reflector
from reflect.store import InMemoryReflectStore
from retain.models import (
    Category,
    ClaimType,
    Document,
    FactType,
    MemoryFact,
    Scope,
    Speaker,
    Visibility,
    stable_id,
)
from retain.store import InMemoryRetainStore

NOW = datetime(2026, 11, 7, tzinfo=timezone.utc)
SCOPE = Scope("dream-user", "project")


def document(name="a", *, session="s1", speaker=Speaker.USER, scope=None, **kw):
    fields = dict(
        document_id=stable_id("dream-doc", name),
        episode_id=f"episode-{name}",
        position=0,
        scope=scope or Scope(SCOPE.user_id, SCOPE.project_id, session),
        speaker=speaker,
        text="I use six ounces of water per tablespoon of coffee.",
        occurred_at=NOW - timedelta(days=10),
        recorded_at=NOW - timedelta(days=10),
    )
    return Document(**(fields | kw))


def source_fact(doc, *, visibility=Visibility.PROJECT):
    return MemoryFact(
        fact_id=stable_id("source-fact", doc.document_id),
        scope=doc.scope,
        visibility=visibility,
        fact_type=FactType.WORLD,
        category=Category.ATTRIBUTE,
        speaker=doc.speaker,
        claim_type=ClaimType.ASSERTED,
        text=doc.text,
        evidence=doc.text,
        source_document_id=doc.document_id,
        source_episode_id=doc.episode_id,
        recorded_at=doc.recorded_at,
        valid_from=doc.occurred_at,
    )


class ScriptedGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, schema):
        self.calls.append(prompt)
        data = json.loads(prompt.split("\n", 2)[-1])
        if prompt.startswith("DREAM_RECOVER"):
            return {
                "recoveries": [
                    {"document_id": d["document_id"], "quote": d["text"], "claim_type": "asserted"}
                    for d in data["documents"]
                ]
            }
        return {
            "insights": [
                {
                    "text": "Coffee brewing preferences changed over time; check the latest ratio.",
                    "fact_ids": [f["fact_id"] for f in data],
                    "evidence": [
                        {"document_id": f["document_id"], "quote": f["quote"]} for f in data
                    ],
                }
            ]
        }


@pytest.fixture
def setup():
    retain = InMemoryRetainStore()
    docs = [document(), document("b", session="s2", text="I now use five ounces per tablespoon.")]
    retain.documents = {d.document_id: d for d in docs}
    store = InMemoryDreamStore(retain)
    generator = ScriptedGenerator()
    dreamer = Dreamer(
        store=store, generate=generator, embedder=HashingEmbedder(), clock=lambda: NOW
    )
    return retain, store, generator, dreamer


def start(dreamer, **kwargs):
    return dreamer.start(DreamRequest(SCOPE, ("s1", "s2"), **kwargs))


def test_output_stays_staged_until_apply_and_replay_is_idempotent(setup):
    retain, store, generate, dreamer = setup
    job = start(dreamer, request_key="test")
    assert start(dreamer, request_key="test")["run_id"] == job["run_id"]
    result = dreamer.worker_once()
    assert result["status"] == "ready" and len(result["recoveries"]) == 2
    assert len(result["insights"]) == 1
    assert not retain.facts and not dreamer.insights(SCOPE)
    applied = dreamer.apply(job["run_id"], SCOPE)
    assert applied["status"] == "applied" and len(retain.facts) == 2
    assert len(dreamer.insights(SCOPE)) == 1
    assert dreamer.get(job["run_id"], SCOPE)["fresh"]
    assert dreamer.apply(job["run_id"], SCOPE) == applied
    assert dreamer.worker_once() is None
    assert len(generate.calls) == 2


@pytest.mark.parametrize("change", ["document", "fact", "delete", "expiry"])
def test_changed_or_deleted_evidence_blocks_apply(setup, change):
    retain, _, _, dreamer = setup
    job = start(dreamer)
    dreamer.worker_once()
    doc = next(iter(retain.documents.values()))
    if change == "document":
        retain.documents[doc.document_id] = replace(doc, text="Changed")
    elif change == "fact":
        f = source_fact(doc)
        retain.facts[f.fact_id] = f
    elif change == "delete":
        del retain.documents[doc.document_id]
    else:
        retain.documents[doc.document_id] = replace(doc, expires_at=NOW)
    with pytest.raises(StaleDream):
        dreamer.apply(job["run_id"], SCOPE)
    assert not dreamer.insights(SCOPE)


def test_apply_rolls_back_facts_and_job_if_insight_write_fails(setup, monkeypatch):
    retain, store, _, dreamer = setup
    job = start(dreamer)
    dreamer.worker_once()

    def fail(_):
        raise RuntimeError("Simulated storage failure")

    monkeypatch.setattr(store, "put_insights", fail)
    with pytest.raises(RuntimeError):
        dreamer.apply(job["run_id"], SCOPE)
    assert not retain.facts
    assert dreamer.get(job["run_id"], SCOPE)["status"] == "ready"


def test_stale_approved_insight_disappears_and_cross_scope_reads_are_empty(setup):
    retain, _, _, dreamer = setup
    job = start(dreamer)
    dreamer.worker_once()
    dreamer.apply(job["run_id"], SCOPE)
    assert not dreamer.insights(Scope("another-user", "project"))
    assert not dreamer.insights(Scope(SCOPE.user_id, "another-project"))
    with pytest.raises(KeyError):
        dreamer.get(job["run_id"], Scope("another-user", "project"))
    retain.facts.clear()  # deletion invalidates the evidence version
    assert not dreamer.insights(SCOPE)


def test_scope_filters_sources_and_never_exposes_other_projects(setup):
    retain, _, generate, dreamer = setup
    hidden = document("hidden", scope=Scope(SCOPE.user_id, "secret", "s1"), text="secret-project")
    retain.documents[hidden.document_id] = hidden
    start(dreamer)
    dreamer.worker_once()
    assert all("secret-project" not in p for p in generate.calls)
    with pytest.raises(ValueError, match="no unexpired sources"):
        dreamer.start(DreamRequest(SCOPE, ("missing",)))
    with pytest.raises(ValueError):
        DreamRequest(Scope("u"), ("s1",))


def test_invented_quotes_and_unknown_fact_ids_fail_closed(setup):
    retain, _, _, dreamer = setup
    job = start(dreamer)
    dreamer.generate = lambda *_: {
        "recoveries": [
            {
                "document_id": next(iter(retain.documents)),
                "quote": "Invented",
                "claim_type": "asserted",
            }
        ]
    }
    with pytest.raises(ValueError, match="exact quote"):
        dreamer.worker_once()
    assert dreamer.get(job["run_id"], SCOPE)["status"] == "failed"
    assert not retain.facts


def test_assistant_advice_is_not_promoted_to_asserted_user_fact(setup):
    retain, _, _, dreamer = setup
    for key, doc in list(retain.documents.items()):
        retain.documents[key] = replace(doc, speaker=Speaker.ASSISTANT)
    job = start(dreamer)
    dreamer.worker_once()
    dreamer.apply(job["run_id"], SCOPE)
    assert all(f.claim_type == ClaimType.PROPOSED for f in retain.facts.values())
    assert all(f.fact_type == FactType.EXPERIENCE for f in retain.facts.values())


def test_private_source_fact_prevents_cross_session_insight(setup):
    retain, _, _, dreamer = setup
    doc = next(iter(retain.documents.values()))
    f = source_fact(doc, visibility=Visibility.SESSION)
    retain.facts[f.fact_id] = f
    job = start(dreamer)
    with pytest.raises(ValueError, match="private insight"):
        dreamer.worker_once()
    assert dreamer.get(job["run_id"], SCOPE)["status"] == "failed"


def test_discard_cancels_worker_and_fences_old_lease(setup):
    _, store, _, dreamer = setup
    job = start(dreamer)
    with store.transaction() as tx:
        claimed = tx.claim(NOW)
    dreamer.discard(job["run_id"], SCOPE)
    with pytest.raises(LostLease):
        dreamer._checkpoint(claimed, claimed["lease_token"])
    assert dreamer.worker_once() is None


def test_expired_worker_is_reclaimed_and_old_token_is_rejected(setup):
    _, store, _, dreamer = setup
    start(dreamer)
    with store.transaction() as tx:
        first = tx.claim(NOW)
        assert tx.claim(NOW) is None
        second = tx.claim(NOW + timedelta(seconds=601))
    assert second["attempts"] == 2 and first["lease_token"] != second["lease_token"]
    with pytest.raises(LostLease):
        dreamer._checkpoint(first, first["lease_token"])


def test_checkpoint_resumes_completed_pages_after_transient_failure(setup):
    _, store, generate, dreamer = setup
    dreamer.config = DreamConfig(page_chars=700)
    job = start(dreamer)
    calls = 0

    def flaky(prompt, schema):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Transient provider failure")
        return generate(prompt, schema)

    dreamer.generate = flaky
    with pytest.raises(OSError):
        dreamer.worker_once()
    assert store.jobs[job["run_id"]]["next_page"] == 1
    assert store.jobs[job["run_id"]]["status"] == "queued"
    assert dreamer.worker_once()["status"] == "ready"
    # Three recovery attempts (one failed) + one synthesis, not a first-page rerun.
    assert calls == 4


def test_oversized_source_and_empty_output_are_explicit(setup):
    _, _, _, dreamer = setup
    dreamer.config = DreamConfig(page_chars=10)
    job = start(dreamer)
    with pytest.raises(ValueError, match="exceeds page_chars"):
        dreamer.worker_once()
    assert dreamer.get(job["run_id"], SCOPE)["status"] == "failed"


def test_embeddings_are_validated_before_writing(setup):
    retain, _, _, dreamer = setup
    job = start(dreamer)
    dreamer.worker_once()
    dreamer.embedder = lambda texts: [[float("nan")]] * len(texts)
    with pytest.raises(ValueError, match="finite"):
        dreamer.apply(job["run_id"], SCOPE)
    assert not retain.facts


def test_recall_only_sees_approved_fresh_insights(setup):
    retain, _, _, dreamer = setup
    job = start(dreamer)
    dreamer.worker_once()
    base = InMemoryRecallStore(retain)
    reader = DreamRecallStore(base, dreamer)
    recaller = Recaller(store=reader, embedder=HashingEmbedder(), clock=lambda: NOW)
    kw = dict(types=[MemoryType.MENTAL_MODEL], decompose=False)
    assert recaller.recall(SCOPE, "coffee brewing ratio", **kw).abstained
    dreamer.apply(job["run_id"], SCOPE)
    result = recaller.recall(SCOPE, "coffee brewing ratio", **kw)
    assert result.items and "Dream inference" in result.context
    assert result.items[0].memory.claim_type == ClaimType.PROPOSED
    assert reader.get_supporting(SCOPE, result.items[0].memory.id)
    retain.documents.clear()
    assert recaller.recall(SCOPE, "coffee brewing ratio", **kw).abstained


def test_existing_facts_are_not_recovered_twice_and_ranking_does_not_stale_lessons(setup):
    retain, _, _, dreamer = setup
    for doc in retain.documents.values():
        f = source_fact(doc)
        retain.facts[f.fact_id] = f
    job = start(dreamer)
    computed = dreamer.worker_once()
    assert computed["recoveries"] == []
    dreamer.apply(job["run_id"], SCOPE)
    Reflector(
        store=InMemoryReflectStore(retain), embedder=HashingEmbedder(), clock=lambda: NOW
    ).reflect(SCOPE.user_id)
    assert dreamer.insights(SCOPE)


def test_generation_never_sees_scope_or_benchmark_metadata(setup):
    _, _, generate, dreamer = setup
    start(dreamer)
    dreamer.worker_once()
    assert all(
        SCOPE.user_id not in prompt and SCOPE.project_id not in prompt for prompt in generate.calls
    )
    assert all("episode-a" not in prompt and "episode-b" not in prompt for prompt in generate.calls)


def test_unknown_fact_citation_is_rejected(setup):
    _, _, generate, dreamer = setup
    job = start(dreamer)

    def bad_citation(prompt, schema):
        response = generate(prompt, schema)
        if "insights" in response:
            response["insights"][0]["fact_ids"] = ["unknown"]
        return response

    dreamer.generate = bad_citation
    with pytest.raises(ValueError, match="allowed supporting facts"):
        dreamer.worker_once()
    assert dreamer.get(job["run_id"], SCOPE)["status"] == "failed"


def test_empty_model_output_is_a_valid_reviewable_noop(setup):
    retain, _, _, dreamer = setup
    dreamer.generate = lambda prompt, _: {"recoveries": []}
    job = start(dreamer)
    assert dreamer.worker_once()["status"] == "ready"
    dreamer.apply(job["run_id"], SCOPE)
    assert not retain.facts and not dreamer.insights(SCOPE)


def test_private_insight_is_only_visible_in_its_source_session(setup):
    retain, _, _, dreamer = setup
    retain.documents = {k: d for k, d in retain.documents.items() if d.scope.session_id == "s1"}
    doc = next(iter(retain.documents.values()))
    f = source_fact(doc, visibility=Visibility.SESSION)
    retain.facts[f.fact_id] = f
    job = dreamer.start(DreamRequest(SCOPE, ("s1",)))
    dreamer.worker_once()
    dreamer.apply(job["run_id"], SCOPE)
    assert not dreamer.insights(SCOPE)
    assert not dreamer.insights(Scope(SCOPE.user_id, SCOPE.project_id, "s2"))
    assert len(dreamer.insights(doc.scope)) == 1


def test_long_source_windows_cover_original_text(setup):
    from reflect.dream import _pages

    doc = document(text="Some source text. " * 500)
    pages = _pages([doc], 1000)
    windows = [d.text for page in pages for d in page]
    assert len(windows) > 1
    assert all(w in doc.text for w in windows)
    assert windows[0].startswith("Some source") and windows[-1].endswith("text. ")


def test_transient_failures_stop_after_configured_attempts(setup):
    _, _, _, dreamer = setup
    job = start(dreamer)

    def fail(*_):
        raise OSError("Provider unavailable")

    dreamer.generate = fail
    for _ in range(dreamer.config.max_attempts):
        with pytest.raises(OSError):
            dreamer.worker_once()
    assert dreamer.get(job["run_id"], SCOPE)["status"] == "failed"
    assert dreamer.worker_once() is None
