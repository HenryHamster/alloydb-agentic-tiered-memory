import json

import pytest
from test_bench import record

from bench.embedders import HashingEmbedder
from bench.longmemeval import Instance, verbatim_stages
from bench.reflect_dream import SmokeGenerator, compare, frozen_payload, load_frozen, main, prepare
from reflect.dream import encoded


def test_paired_arms_reuse_frozen_input_and_do_not_mutate_it():
    instance = Instance.from_json(record())
    frozen, _ = prepare(instance, HashingEmbedder(), verbatim_stages, 2)
    before = dict(frozen.facts)
    rows = compare(instance, frozen, SmokeGenerator(), HashingEmbedder())
    assert [r["arm"] for r in rows] == ["baseline", "reflect", "reflect_dream"]
    assert len({r["snapshot_hash"] for r in rows}) == 1
    assert frozen.facts == before
    assert all(r["original_fact_ids_preserved"] for r in rows)


def test_run_manifest_and_reports_are_immutable(tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps([record()]), encoding="utf-8")
    out = tmp_path / "run"
    args = [str(data), "--out", str(out), "--ids", record()["question_id"]]
    main(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["evaluation_kind"] == "offline_plumbing_smoke"
    assert manifest["status"] == "complete" and not manifest["quality_claim"]
    assert manifest["code_sha256"] and manifest["dataset_sha256"]
    with pytest.raises(FileExistsError):
        main(args)


def test_saved_snapshots_replay_exact_inputs_and_reject_mismatched_history(tmp_path):
    instance = Instance.from_json(record())
    original, dropped = prepare(instance, HashingEmbedder(), verbatim_stages, 2)
    path = tmp_path / "snapshot.json"
    path.write_text(encoded(frozen_payload(instance, original, dropped, 2)), encoding="utf-8")
    restored, omissions = load_frozen(path, instance, 2)
    assert restored.facts == original.facts and restored.documents == original.documents
    assert restored.retained == original.retained and omissions == dropped
    assert [f.embedding for f in restored.facts.values()] == [
        f.embedding for f in original.facts.values()
    ]
    with pytest.raises(ValueError, match="does not match"):
        load_frozen(path, instance, 0)
    changed = record()
    changed["haystack_sessions"][0][0]["content"] = "Different source history"
    with pytest.raises(ValueError, match="does not match"):
        load_frozen(path, Instance.from_json(changed), 2)
