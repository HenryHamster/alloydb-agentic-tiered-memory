"""Offline checks of Reflect: tiers, consolidation, macros, the Reflector pass and both stores."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from reflect import Reflector, TierConfig, build_reflector
from reflect.consolidation import (
    LLMConflictJudge,
    consolidate,
    domain_key,
    domain_scope,
    find_candidates,
    resolve,
)
from reflect.macros import (
    LLMMacroWriter,
    MacroConfig,
    build_clusters,
    causal_chains,
    match_existing,
    needs_refresh,
    new_or_updated_macro,
    temporal_span,
)
from reflect.models import (
    Candidate,
    FactState,
    Judgment,
    MacroMemory,
    ReflectBatch,
    Tier,
    Verdict,
)
from reflect.store import AlloyDBReflectStore, InMemoryReflectStore
from reflect.tiers import apply_hot_cap, next_tier, promote_on_hit, rank_score
from retain.models import (
    Category,
    ClaimType,
    EntityRole,
    FactEntity,
    FactType,
    LinkType,
    MemoryFact,
    MemoryLink,
    Relation,
    Scope,
    Speaker,
    Visibility,
    stable_id,
)
from retain.store import InMemoryRetainStore

T0 = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)
USER = "user_dev_42"
SCOPE = Scope(USER, "proj", "sess")
ENTITY = stable_id("entity", "pipeline")
OTHER_ENTITY = stable_id("entity", "duckdb")


def fact(name, text=None, *, days=0, visibility=Visibility.GLOBAL, scope=SCOPE, **kwargs):
    fields = dict(
        fact_id=stable_id("fact", name),
        scope=scope,
        visibility=visibility,
        fact_type=FactType.WORLD,
        category=Category.ATTRIBUTE,
        speaker=Speaker.USER,
        claim_type=ClaimType.ASSERTED,
        text=text or f"User fact {name}.",
        evidence=f"quote {name}",
        source_document_id=stable_id("doc", name),
        source_episode_id=f"episode-{name}",
        recorded_at=T0 + timedelta(days=days),
        valid_from=T0 + timedelta(days=days),
        embedding=(0.1,) * 768,
    )
    return MemoryFact(**(fields | kwargs))


def subject(*facts, entity=ENTITY, role=EntityRole.SUBJECT):
    return [FactEntity(f.fact_id, entity, role) for f in facts]


def similar(left, right, weight=0.9):
    return MemoryLink(left.fact_id, right.fact_id, Relation.SIMILAR, weight)


def by_id(*facts):
    return {f.fact_id: f for f in facts}


# Tiers


def test_tier_thresholds_need_hysteresis():
    TierConfig()
    with pytest.raises(ValueError):
        TierConfig(hot_promote=0.5, hot_demote=0.6)


def test_purged_is_a_transition_not_a_stored_tier():
    assert FactState("f").tier == Tier.HOT
    with pytest.raises(ValueError):
        FactState("f", tier=Tier.PURGED)
    with pytest.raises(ValueError):
        FactState("f", confidence=1.5)


def test_reflect_relations_have_link_types():
    assert Relation.CONSOLIDATED_INTO.link_type == LinkType.SEMANTIC
    assert Relation.CONFLICTS_WITH.link_type == LinkType.SEMANTIC


def score(**kwargs):
    fields = dict(
        access_count=0,
        last_accessed_at=None,
        created_at=T0,
        evidence_count=1,
        confidence=1.0,
        now=T0,
    )
    return rank_score(**(fields | kwargs))


def test_new_fact_scores_one_and_cools_unless_recalled():
    assert score() == pytest.approx(1.0)
    assert score(now=T0 + timedelta(days=3)) == pytest.approx(0.5)
    recalled = score(access_count=3, last_accessed_at=T0 + timedelta(days=30))
    assert recalled == pytest.approx(score(access_count=3, last_accessed_at=T0), rel=0.5)
    assert score(evidence_count=3) > score()
    assert score(confidence=0.5) == pytest.approx(0.5)


def test_usage_decays_with_the_access_half_life():
    now = T0 + timedelta(days=100)
    fresh = score(access_count=4, last_accessed_at=now, now=now)
    stale = score(access_count=4, last_accessed_at=now - timedelta(days=14), now=now)
    assert stale == pytest.approx(fresh / 2, rel=1e-3)


@pytest.mark.parametrize(
    ("current", "value", "dormant_days", "expected"),
    [
        (Tier.HOT, 0.7, 0, Tier.HOT),  # between demote and promote: stays
        (Tier.HOT, 0.5, 0, Tier.WARM),
        (Tier.WARM, 0.9, 0, Tier.WARM),  # needs 1.0 to climb back: hysteresis
        (Tier.WARM, 1.0, 0, Tier.HOT),
        (Tier.WARM, 0.1, 10, Tier.WARM),  # cold but not dormant long enough
        (Tier.WARM, 0.1, 31, Tier.ARCHIVED),
        (Tier.ARCHIVED, 0.4, 31, Tier.WARM),
        (Tier.ARCHIVED, 0.1, 100, Tier.ARCHIVED),
        (Tier.ARCHIVED, 0.9, 366, Tier.PURGED),
    ],
)
def test_next_tier(current, value, dormant_days, expected):
    now = T0 + timedelta(days=dormant_days)
    assert next_tier(current, value, last_seen=T0, now=now) == expected


def test_next_tier_without_warm():
    config = TierConfig(use_warm=False)
    later = T0 + timedelta(days=40)
    assert next_tier(Tier.HOT, 0.5, last_seen=T0, now=T0, config=config) == Tier.HOT
    assert next_tier(Tier.HOT, 0.1, last_seen=T0, now=later, config=config) == Tier.ARCHIVED
    assert next_tier(Tier.ARCHIVED, 0.5, last_seen=T0, now=T0, config=config) == Tier.ARCHIVED
    assert next_tier(Tier.ARCHIVED, 1.2, last_seen=T0, now=T0, config=config) == Tier.HOT
    assert promote_on_hit(Tier.ARCHIVED, config) == Tier.HOT
    assert promote_on_hit(Tier.ARCHIVED) == Tier.WARM
    assert promote_on_hit(Tier.HOT) == Tier.HOT


def test_hot_cap_demotes_lowest_rank_ties_by_id():
    proposed = {"a": Tier.HOT, "b": Tier.HOT, "c": Tier.HOT, "d": Tier.WARM}
    scores = {"a": 2.0, "b": 1.0, "c": 1.0, "d": 3.0}
    capped = apply_hot_cap(proposed, scores, TierConfig(hot_cap=2))
    assert capped == {"a": Tier.HOT, "b": Tier.HOT, "c": Tier.WARM, "d": Tier.WARM}
    no_warm = apply_hot_cap(proposed, scores, TierConfig(hot_cap=1, use_warm=False))
    assert no_warm["b"] == no_warm["c"] == Tier.ARCHIVED


# Consolidation


def test_domain_key_and_scope_follow_visibility():
    project = fact("p", visibility=Visibility.PROJECT)
    session = fact("s", visibility=Visibility.SESSION)
    world = fact("g")
    assert domain_key(project) == (USER, Visibility.PROJECT, "proj")
    assert domain_key(session) == (USER, Visibility.SESSION, "sess")
    assert domain_key(world) == (USER, Visibility.GLOBAL, None)
    assert domain_scope(project) == Scope(USER, project_id="proj")
    assert domain_scope(session) == Scope(USER, session_id="sess")
    assert domain_scope(world) == Scope(USER)


def test_candidates_need_similarity_subject_category_and_domain():
    a, b, c = fact("a"), fact("b"), fact("c")
    weak = fact("weak")
    other_category = fact("cat", category=Category.PREFERENCE)
    other_domain = fact("dom", visibility=Visibility.PROJECT)
    closed = fact("closed", valid_to=T0 + timedelta(days=1))
    facts = by_id(a, b, c, weak, other_category, other_domain, closed)
    links = [
        similar(a, b),
        similar(b, c),
        similar(a, weak, 0.5),
        similar(a, other_category),
        similar(a, other_domain),
        similar(a, closed),
    ]
    rows = subject(*facts.values())
    candidates = find_candidates(facts, {}, links, rows)
    assert candidates == [Candidate(tuple(sorted((a.fact_id, b.fact_id, c.fact_id))), ENTITY)]


def test_candidates_skip_flagged_pairs_and_consolidated_facts():
    a, b, c = fact("a"), fact("b"), fact("c")
    states = {
        a.fact_id: FactState(a.fact_id, conflicts_with=(b.fact_id,)),
        c.fact_id: FactState(c.fact_id, consolidated_into=stable_id("merged")),
    }
    rows = subject(a, b, c)
    links = [similar(a, b), similar(b, c)]
    assert find_candidates(by_id(a, b, c), states, links, rows) == []


def test_a_fact_with_two_subjects_is_judged_once():
    a, b = fact("a"), fact("b")
    rows = subject(a, b) + subject(a, b, entity=OTHER_ENTITY)
    assert len(find_candidates(by_id(a, b), {}, [similar(a, b)], rows)) == 1


def test_consolidate_merges_into_one_stable_fact():
    old = fact("old", days=0, evidence_count=2, entity_ids=(ENTITY,))
    new = fact("new", days=5, entity_ids=(OTHER_ENTITY,), occurred_at=T0 + timedelta(days=4))
    now = T0 + timedelta(days=9)
    merged = consolidate([new, old], "User runs a weekly data pipeline.", now=now)
    assert merged.fact_id == stable_id("consolidated", *sorted((old.fact_id, new.fact_id)))
    assert merged.valid_from == old.valid_from
    assert merged.recorded_at == now
    assert merged.evidence_count == 3
    assert merged.evidence == "quote old\nquote new"
    assert merged.source_episode_id == new.source_episode_id
    assert merged.entity_ids == tuple(sorted((ENTITY, OTHER_ENTITY)))
    assert merged.embedding is None
    with pytest.raises(ValueError):
        consolidate([old], "x", now=now)


def test_resolve_redundant_archives_originals_and_sums_usage():
    a, b = fact("a", days=0), fact("b", days=1)
    states = {
        a.fact_id: FactState(a.fact_id, access_count=2, tier=Tier.WARM, confidence=0.8),
        b.fact_id: FactState(b.fact_id, access_count=3, last_accessed_at=T0),
    }
    candidate = Candidate(tuple(sorted((a.fact_id, b.fact_id))), ENTITY)
    judgment = Judgment(candidate.fact_ids, Verdict.REDUNDANT, "User owns the pipeline.")
    rows = subject(a, b) + subject(b, entity=OTHER_ENTITY, role=EntityRole.OBJECT)
    result = resolve([judgment], [candidate], by_id(a, b), states, rows, now=T0)

    (item,) = result.consolidations
    assert item.original_ids == (a.fact_id, b.fact_id)
    assert item.state.access_count == 5
    assert item.state.tier == Tier.HOT
    assert item.state.confidence == 0.8
    assert {(r.entity_id, r.role) for r in item.fact_entities} == {
        (ENTITY, EntityRole.SUBJECT),
        (OTHER_ENTITY, EntityRole.OBJECT),
    }
    assert {s.fact_id: s.tier for s in result.states} == {
        a.fact_id: Tier.ARCHIVED,
        b.fact_id: Tier.ARCHIVED,
    }
    assert all(s.consolidated_into == item.merged.fact_id for s in result.states)
    assert {link.relation for link in result.links} == {Relation.CONSOLIDATED_INTO}


def test_resolve_state_change_newest_wins_and_keeps_history():
    old, new = fact("works-x", days=0), fact("works-y", days=10)
    candidate = Candidate(tuple(sorted((old.fact_id, new.fact_id))), ENTITY)
    judgment = Judgment(candidate.fact_ids, Verdict.STATE_CHANGE)
    result = resolve([judgment], [candidate], by_id(old, new), {}, [], now=T0)
    (change,) = result.supersessions
    assert (change.old_fact_id, change.new_fact_id) == (old.fact_id, new.fact_id)
    assert change.valid_to == new.valid_from
    assert result.links == (MemoryLink(new.fact_id, old.fact_id, Relation.SUPERSEDES),)


def test_resolve_conflict_flags_both_once():
    a, b = fact("a"), fact("b", recorded_at=T0 + timedelta(hours=1))
    candidate = Candidate(tuple(sorted((a.fact_id, b.fact_id))), ENTITY)
    # Same valid_from: a "state change" has no time order, so it is a conflict.
    judgment = Judgment(candidate.fact_ids, Verdict.STATE_CHANGE, reason="disagree")
    result = resolve([judgment], [candidate], by_id(a, b), {}, [], now=T0)
    assert not result.supersessions
    assert all(s.confidence == 0.5 for s in result.states)
    assert {s.fact_id: s.conflicts_with for s in result.states} == {
        a.fact_id: (b.fact_id,),
        b.fact_id: (a.fact_id,),
    }
    (review,) = result.reviews
    assert review.fact_ids == candidate.fact_ids and review.reason == "disagree"
    low, high = sorted(candidate.fact_ids)
    assert result.links == (MemoryLink(low, high, Relation.CONFLICTS_WITH),)

    flagged = {s.fact_id: s for s in result.states}
    again = resolve([judgment], [candidate], by_id(a, b), flagged, [], now=T0)
    assert again.states == () and again.reviews == ()


def test_resolve_rejects_untrusted_judgments():
    a, b, c = fact("a"), fact("b"), fact("c")
    candidate = Candidate(tuple(sorted((a.fact_id, b.fact_id))), ENTITY)
    facts = by_id(a, b, c)
    bad = [
        Judgment((a.fact_id,), Verdict.REDUNDANT, "x"),
        Judgment((a.fact_id, c.fact_id), Verdict.CONFLICT),
        Judgment(candidate.fact_ids, Verdict.REDUNDANT, None),
    ]
    for judgment in bad:
        with pytest.raises(ValueError):
            resolve([judgment], [candidate], facts, {}, [], now=T0)
    distinct = Judgment(candidate.fact_ids, Verdict.DISTINCT)
    assert resolve([distinct], [candidate], facts, {}, [], now=T0) == resolve(
        [], [candidate], facts, {}, [], now=T0
    )


def test_llm_conflict_judge_batches_and_validates():
    a, b, c, d = fact("a"), fact("b"), fact("c"), fact("d")
    first = Candidate(tuple(sorted((a.fact_id, b.fact_id))), ENTITY)
    second = Candidate(tuple(sorted((c.fact_id, d.fact_id))), ENTITY)
    prompts = []

    def generate(prompt, schema):
        prompts.append(prompt)
        assert "judgments" in schema["properties"]
        ids = first.fact_ids if a.fact_id in prompt else second.fact_ids
        return {
            "judgments": [
                {"group": 0, "verdict": "redundant", "fact_ids": list(ids), "merged_text": " M "}
            ]
        }

    judge = LLMConflictJudge(generate, batch_size=1)
    judgments = judge([first, second], by_id(a, b, c, d))
    assert len(prompts) == 2
    assert [j.fact_ids for j in judgments] == [first.fact_ids, second.fact_ids]
    assert judgments[0].merged_text == "M"

    def outside(prompt, schema):
        return {"judgments": [{"group": 0, "verdict": "conflict", "fact_ids": [c.fact_id]}]}

    with pytest.raises(ValueError):
        LLMConflictJudge(outside)([first], by_id(a, b, c))
    with pytest.raises(ValueError):
        LLMConflictJudge(lambda p, s: {"judgments": [{"group": 0, "verdict": "maybe"}]})(
            [first], by_id(a, b)
        )
    with pytest.raises(ValueError):
        LLMConflictJudge(lambda p, s: ["not", "an", "object"])([first], by_id(a, b))


# Macros


def duckdb_story():
    """The proposal's example: S1 takes 2 h, S3 adds DuckDB, S4 now under 10 min."""
    s1 = fact("s1", "User's weekly Python script takes 2 hours.", days=0)
    s3 = fact("s3", "User added DuckDB to replace Pandas.", days=14)
    s4 = fact("s4", "User's weekly script finishes in under 10 minutes.", days=21)
    s4 = replace(s4, occurred_at=T0 + timedelta(days=21))
    s1 = replace(s1, valid_to=s4.valid_from, superseded_by=s4.fact_id)
    links = [
        MemoryLink(s4.fact_id, s3.fact_id, Relation.CAUSED_BY),
        MemoryLink(s4.fact_id, s1.fact_id, Relation.SUPERSEDES),
    ]
    return (s1, s3, s4), links


def test_duckdb_story_forms_one_cluster_with_its_causal_chain():
    (s1, s3, s4), links = duckdb_story()
    (cluster,) = build_clusters(by_id(s1, s3, s4), {}, links, [])
    assert set(cluster.fact_ids) == {s1.fact_id, s3.fact_id, s4.fact_id}
    assert cluster.causal_chains == ((s3.fact_id, s4.fact_id),)
    assert cluster.temporal_span == (s1.valid_from, s4.occurred_at)
    assert cluster.scope == Scope(USER) and cluster.visibility == Visibility.GLOBAL


def test_clusters_skip_hubs_small_groups_merged_originals_and_domains():
    facts = [fact(f"f{i}") for i in range(4)]
    hub_rows = subject(*facts, entity="hub")
    assert len(build_clusters(by_id(*facts), {}, [], hub_rows)) == 1  # under the hub limit
    assert build_clusters(by_id(*facts), {}, [], hub_rows, MacroConfig(max_entity_facts=3)) == []

    merged_away = {facts[0].fact_id: FactState(facts[0].fact_id, consolidated_into="m")}
    (cluster,) = build_clusters(by_id(*facts), merged_away, [], hub_rows)
    assert facts[0].fact_id not in cluster.fact_ids

    project = [fact(f"p{i}", visibility=Visibility.PROJECT) for i in range(2)]
    mixed = by_id(facts[0], facts[1], *project)
    rows = subject(*mixed.values(), entity="shared")
    assert build_clusters(mixed, {}, [], rows) == []  # 2 + 2 across domains: never joined


def test_causal_chains_are_maximal_and_cut_cycles():
    a, b, c, d = "a", "b", "c", "d"
    links = [
        MemoryLink(b, a, Relation.CAUSED_BY),
        MemoryLink(c, b, Relation.CAUSED_BY),
        MemoryLink(d, b, Relation.CAUSED_BY),
    ]
    assert causal_chains(links) == ((a, b, c), (a, b, d))
    loop = [MemoryLink(a, b, Relation.CAUSED_BY), MemoryLink(b, a, Relation.CAUSED_BY)]
    assert causal_chains(loop) == ()
    assert len(causal_chains(links, limit=1)) == 1


def test_temporal_span_falls_back_to_all_facts():
    a, b = fact("a", days=0), fact("b", days=5)
    assert temporal_span([a, b], []) == (a.valid_from, b.valid_from)
    assert temporal_span([], []) is None


def macro_for(cluster, **kwargs):
    macro = new_or_updated_macro(cluster, None, "Title", "Summary.", now=T0)
    return replace(macro, **kwargs)


def test_matching_refresh_and_versions():
    (s1, s3, s4), links = duckdb_story()
    (cluster,) = build_clusters(by_id(s1, s3, s4), {}, links, [])
    macro = macro_for(cluster, access_count=4, tier=Tier.WARM)
    assert macro.version == 1 and macro.macro_embedding is None
    assert match_existing([cluster], [macro]) == [(cluster, macro)]
    assert not needs_refresh(cluster, macro)

    grown = replace(cluster, fact_ids=(*cluster.fact_ids, fact("s5").fact_id))
    assert needs_refresh(grown, macro)
    later = T0 + timedelta(days=1)
    updated = new_or_updated_macro(grown, macro, "New", "Longer story.", now=later)
    assert updated.macro_id == macro.macro_id and updated.version == 2
    assert updated.access_count == 4 and updated.updated_at == later

    closed = replace(macro, valid_to=T0)
    assert match_existing([cluster], [closed]) == [(cluster, None)]
    elsewhere = replace(macro, visibility=Visibility.PROJECT, scope=Scope(USER, "proj"))
    assert match_existing([cluster], [elsewhere]) == [(cluster, None)]


def test_llm_macro_writer_validates_output():
    (s1, s3, s4), links = duckdb_story()
    (cluster,) = build_clusters(by_id(s1, s3, s4), {}, links, [])
    seen = {}

    def generate(prompt, schema):
        seen["prompt"] = prompt
        return {"title": " one two three four five six seven eight nine ", "summary": " Done. "}

    title, summary = LLMMacroWriter(generate).write(cluster, [s4, s1, s3])
    assert title == "one two three four five six seven eight"
    assert summary == "Done."
    assert seen["prompt"].index(s1.text) < seen["prompt"].index(s4.text)
    with pytest.raises(ValueError):
        LLMMacroWriter(lambda p, s: {"title": "", "summary": "x"}).write(cluster, [s1])


# Reflector with the in-memory stores


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[0.2] * 768 for _ in texts]


class FakeWriter:
    def __init__(self):
        self.calls = 0

    def write(self, cluster, facts, states=None):
        self.calls += 1
        return "Faster pipeline", f"Story of {len(facts)} facts."


def seeded_store(*facts, links=(), rows=()):
    retain = InMemoryRetainStore()
    retain.facts = by_id(*facts)
    retain.links = {(k.from_fact_id, k.to_fact_id, k.relation): k for k in links}
    retain.fact_entities = set(rows)
    return InMemoryReflectStore(retain)


def redundant_judge(candidates, facts):
    return [Judgment(c.fact_ids, Verdict.REDUNDANT, "User runs DuckDB.") for c in candidates]


def test_reflect_pass_consolidates_synthesizes_and_is_idempotent():
    (s1, s3, s4), links = duckdb_story()
    dup = fact("dup", "User switched to DuckDB.", days=15)
    rows = subject(s3, dup, entity=OTHER_ENTITY)
    store = seeded_store(s1, s3, s4, dup, links=[*links, similar(s3, dup)], rows=rows)
    embedder, writer = FakeEmbedder(), FakeWriter()
    clock = [T0 + timedelta(days=22)]
    reflector = Reflector(
        store=store,
        embedder=embedder,
        judge=redundant_judge,
        macro_writer=writer,
        clock=lambda: clock[0],
    )

    result = reflector.reflect(USER)
    (item,) = result.batch.consolidations
    merged_id = item.merged.fact_id
    assert store.retain_store.facts[merged_id].embedding == (0.2,) * 768
    assert store.states[s3.fact_id].consolidated_into == merged_id
    assert store.states[s3.fact_id].tier == Tier.ARCHIVED
    assert store.states[s1.fact_id].tier == Tier.ARCHIVED  # superseded history
    assert store.states[merged_id].tier == Tier.HOT
    (macro,) = store.user_macros(USER)
    assert macro.macro_embedding == (0.2,) * 768
    assert s3.fact_id not in macro.supporting_fact_ids  # replaced by the merged fact
    assert set(macro.supporting_fact_ids) == {s1.fact_id, s4.fact_id, merged_id}
    assert macro.causal_chain_ids == ((merged_id, s4.fact_id),)  # s4 caused_by s3 -> merged
    assert embedder.calls == [["User runs DuckDB."], ["Faster pipeline\nStory of 3 facts."]]

    clock[0] += timedelta(hours=1)
    again = reflector.reflect(USER)
    assert again.batch.consolidations == () and writer.calls == 1 and len(embedder.calls) == 2
    assert store.user_macros(USER)[0].version == 1


def test_record_access_promotes_archived_and_survives_a_pass():
    a = fact("a")
    store = seeded_store(a)
    store.states[a.fact_id] = FactState(a.fact_id, tier=Tier.ARCHIVED)
    store.record_access([a.fact_id, "unknown"], [], T0)
    state = store.states[a.fact_id]
    assert (state.access_count, state.last_accessed_at, state.tier) == (1, T0, Tier.WARM)

    reflector = Reflector(store=store, embedder=FakeEmbedder(), clock=lambda: T0)
    reflector.reflect(USER)
    assert store.states[a.fact_id].access_count == 1


def test_dormant_rows_are_purged_with_their_links():
    old = fact("old", days=-400)
    keep = fact("keep")
    store = seeded_store(old, keep, links=[similar(old, keep)])
    store.states[old.fact_id] = FactState(old.fact_id, tier=Tier.ARCHIVED)
    reflector = Reflector(store=store, embedder=FakeEmbedder(), clock=lambda: T0)
    result = reflector.reflect(USER)
    assert result.batch.purged_fact_ids == (old.fact_id,)
    assert old.fact_id not in store.retain_store.facts
    assert store.retain_store.links == {}


def test_forget_purges_facts_derived_merges_and_macros():
    a, b, c = fact("a"), fact("b"), fact("c")
    merged = fact("merged")
    store = seeded_store(a, b, c, merged)
    store.states[a.fact_id] = FactState(a.fact_id, consolidated_into=merged.fact_id)
    store.macros["m"] = MacroMemory(
        "m", Scope(USER), Visibility.GLOBAL, "t", "s", (b.fact_id, merged.fact_id), T0, T0
    )
    reflector = Reflector(store=store, embedder=FakeEmbedder(), clock=lambda: T0)
    other = Scope("someone-else")
    assert reflector.forget(other, [a.fact_id]).purged_fact_ids == ()
    batch = reflector.forget(Scope(USER), [a.fact_id, "unknown"])
    assert set(batch.purged_fact_ids) == {a.fact_id, merged.fact_id}
    assert batch.purged_macro_ids == ("m",)
    assert set(store.retain_store.facts) == {b.fact_id, c.fact_id}
    assert store.macros == {}


def test_in_memory_store_rejects_other_users_and_missing_facts():
    a = fact("a")
    stranger = fact("x", scope=Scope("other"))
    store = seeded_store(a, stranger)
    with pytest.raises(ValueError):
        store.write_reflect(ReflectBatch(user_id=USER, fact_states=(FactState(stranger.fact_id),)))
    with pytest.raises(ValueError):
        store.write_reflect(ReflectBatch(user_id=USER, fact_states=(FactState("missing"),)))
    assert store.states == {}


# AlloyDB store against a recording fake connection


class Cursor:
    def __init__(self, connection):
        self.connection = connection
        self.one, self.many, self.rowcount = None, [], 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.connection.calls.append((normalized, params))
        self.one, self.many, self.rowcount = self.connection.respond(normalized, params)

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.many


class Connection:
    closed = False

    def __init__(self, respond):
        self.respond, self.calls, self.commits, self.rollbacks = respond, [], 0, 0

    def cursor(self, **_):
        return Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def fact_row(f):
    return {
        "fact_id": f.fact_id,
        "user_id": f.scope.user_id,
        "project_id": f.scope.project_id,
        "session_id": f.scope.session_id,
        "visibility": f.visibility.value,
        "fact_type": f.fact_type.value,
        "category": f.category.value,
        "speaker": f.speaker.value,
        "claim_type": f.claim_type.value,
        "text": f.text,
        "evidence": f.evidence,
        "source_document_id": f.source_document_id,
        "source_episode_id": f.source_episode_id,
        "recorded_at": f.recorded_at,
        "valid_from": f.valid_from,
        "details": {},
        "occurred_at": f.occurred_at,
        "valid_to": f.valid_to,
        "superseded_by": f.superseded_by,
        "evidence_count": f.evidence_count,
        "embedding": None,
        "entity_ids": list(f.entity_ids),
    }


def alloydb(respond=lambda sql, params: (None, [], 1)):
    connection = Connection(respond)
    return AlloyDBReflectStore(lambda: connection), connection


def test_alloydb_reads_skip_embeddings_and_map_states():
    a = fact("a")

    def respond(sql, params):
        if sql.startswith("SELECT f.fact_id::text"):
            return None, [fact_row(a)], 1
        if sql.startswith("SELECT fact_id::text, access_count"):
            row = {
                "fact_id": a.fact_id,
                "access_count": 2,
                "last_accessed_at": T0,
                "rank_score": 0.75,
                "tier": "warm",
                "tier_changed_at": T0,
                "confidence": 0.5,
                "consolidated_into": None,
                "conflicts_with": ["z", "y"],
            }
            return None, [row], 1
        return None, [], 1

    store, connection = alloydb(respond)
    assert [f.fact_id for f in store.user_facts(USER)] == [a.fact_id]
    assert "NULL::text AS embedding" in connection.calls[0][0]
    state = store.fact_states([a.fact_id])[a.fact_id]
    assert (state.tier, state.conflicts_with, state.rank_score) == (Tier.WARM, ("y", "z"), 0.75)
    assert store.fact_states([]) == {} and store.links_among([]) == []


def test_alloydb_record_access_promotes_archived_rows():
    store, connection = alloydb()
    store.record_access(["f1", "f1"], ["m1"], T0)
    (fact_sql, fact_params), (macro_sql, macro_params) = connection.calls
    assert fact_sql.startswith("UPDATE memory_facts") and "access_count + 1" in fact_sql
    assert macro_sql.startswith("UPDATE macro_memories")
    assert fact_params == {"at": T0, "promoted": "warm", "ids": ["f1"]}
    no_warm = AlloyDBReflectStore(lambda: connection, tier_config=TierConfig(use_warm=False))
    no_warm.record_access(["f1"], [], T0)
    assert connection.calls[-1][1]["promoted"] == "hot"


def test_alloydb_write_reflect_locks_and_writes_in_order():
    a, b = fact("a"), fact("b")
    macro = MacroMemory(
        "00000000-0000-0000-0000-000000000001",
        Scope(USER),
        Visibility.GLOBAL,
        "t",
        "s",
        (a.fact_id, b.fact_id),
        T0,
        T0,
        temporal_span=(T0, T0),
    )
    batch = ReflectBatch(
        user_id=USER,
        fact_states=(FactState(a.fact_id, tier=Tier.WARM), FactState(b.fact_id)),
        links=(MemoryLink(a.fact_id, b.fact_id, Relation.CONFLICTS_WITH),),
        macros=(macro,),
        purged_fact_ids=("gone",),
    )

    def respond(sql, params):
        if "FROM memory_facts f LEFT JOIN memory_entities" in sql:
            return None, [fact_row(a), fact_row(b)], 2
        if sql.startswith("UPDATE memory_facts f SET rank_score"):
            return None, [], 2
        if sql.startswith("SELECT count(*)"):
            return {"other_users": 0}, [], 1
        return None, [], 1

    store, connection = alloydb(respond)
    store.write_reflect(batch)
    statements = [sql.split(" (")[0].split(" SET")[0] for sql, _ in connection.calls]
    assert statements[0].startswith("SELECT pg_advisory_xact_lock")
    assert connection.calls[0][1] == (f"reflect:{USER}",)
    order = [
        "UPDATE memory_facts f",
        "INSERT INTO memory_links",
        "INSERT INTO macro_memories",
        "DELETE FROM memory_facts WHERE fact_id = ANY(%s::uuid[]) AND user_id = %s",
    ]
    positions = [next(i for i, s in enumerate(statements) if s.startswith(o)) for o in order]
    assert positions == sorted(positions)
    assert connection.commits == 1 and connection.rollbacks == 0
    macro_params = next(p for sql, p in connection.calls if "INTO macro_memories" in sql)
    assert macro_params["embedding"] is None and macro_params["span_start"] == T0


def test_alloydb_write_reflect_rolls_back_on_missing_or_foreign_facts():
    a = fact("a")
    stranger = fact("x", scope=Scope("other"))
    store, connection = alloydb()
    with pytest.raises(ValueError):
        store.write_reflect(ReflectBatch(fact_states=(FactState(a.fact_id),)))

    def foreign(sql, params):
        if "FROM memory_facts f LEFT JOIN memory_entities" in sql:
            return None, [fact_row(stranger)], 1
        return None, [], 1

    store, connection = alloydb(foreign)
    with pytest.raises(ValueError):
        store.write_reflect(ReflectBatch(user_id=USER, fact_states=(FactState(stranger.fact_id),)))
    assert connection.rollbacks == 1 and connection.commits == 0

    store, connection = alloydb(lambda sql, params: (None, [], 0))
    with pytest.raises(ValueError):
        store.write_reflect(ReflectBatch(user_id=USER, fact_states=(FactState(a.fact_id),)))
    assert connection.rollbacks == 1


# Production wiring


def test_production_wiring_uses_injected_adapters():
    generate = MagicMock()
    embedder = MagicMock()
    production = build_reflector(MagicMock(), embedder=embedder, generate=generate)
    assert isinstance(production, Reflector)
    assert isinstance(production.store, AlloyDBReflectStore)
    assert production.embedder is embedder
    assert production.judge.generate is generate
    assert production.macro_writer.generate is generate
    assert production.store.tier_config == production.config.tiers
    generate.assert_not_called()
    embedder.assert_not_called()


def test_a_pass_reads_each_table_once():
    store = seeded_store(*(fact(f"f{i}") for i in range(5)))
    store.fact_states = MagicMock(wraps=store.fact_states)
    Reflector(store=store, embedder=FakeEmbedder(), clock=lambda: T0).reflect(USER)
    store.fact_states.assert_called_once()
