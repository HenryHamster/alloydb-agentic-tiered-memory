"""Reflect pass for one user: consolidate/resolve -> re-tier facts -> synthesize macros -> re-tier
macros, written as one batch."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from reflect.consolidation import (
    ConflictJudge,
    ConsolidationConfig,
    Resolution,
    find_candidates,
    resolve,
    state_of,
    time_order,
)
from reflect.generation import StructuredGenerator
from reflect.macros import (
    MacroConfig,
    MacroWriter,
    build_clusters,
    macro_id_for,
    match_existing,
    needs_refresh,
    new_or_updated_macro,
)
from reflect.models import (
    FactState,
    MacroMemory,
    ReflectBatch,
    ReflectResult,
    ReflectTrigger,
    Tier,
)
from reflect.store import ReflectStore
from reflect.tiers import TierConfig, apply_hot_cap, next_tier, rank_score
from retain.embeddings import Embedder
from retain.models import FactEntity, MemoryFact, MemoryLink, Scope


@dataclass(frozen=True)
class ReflectConfig:
    consolidation: ConsolidationConfig = ConsolidationConfig()
    macros: MacroConfig = MacroConfig()
    tiers: TierConfig = TierConfig()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Reflector:
    """Runs off the request path: after new facts, at session end, or on a schedule.

    Each pass reads one user's rows, decides everything in memory and writes one ReflectBatch.
    Model or store errors propagate and write nothing, so the pass can simply be rerun.
    """

    def __init__(
        self,
        *,
        store: ReflectStore,
        embedder: Embedder,
        judge: ConflictJudge | None = None,
        macro_writer: MacroWriter | None = None,
        config: ReflectConfig = ReflectConfig(),
        clock: Callable[[], datetime] = _utcnow,
    ):
        self.store = store
        self.embedder = embedder
        self.judge = judge
        self.macro_writer = macro_writer
        self.config = config
        self.clock = clock

    def reflect(
        self, user_id: str, trigger: ReflectTrigger = ReflectTrigger.SCHEDULED
    ) -> ReflectResult:
        """One pass:

        1. Read user_facts, fact_states, links_among, entity_rows, user_macros.
        2. find_candidates -> judge(candidates, facts) -> resolve; embed merged facts.
           Apply the resolution to the in-memory view so later steps see merged facts and
           closed validity.
        3. _tier_facts; purged facts leave the view, so no macro is built on them.
        4. _macros: clusters, refresh changed ones, close dissolved ones; then _tier_macros.
        5. store.write_reflect(ReflectBatch(...)) once.
        """
        if not user_id:
            raise ValueError("reflect requires a user_id")
        now = self.clock()
        facts = {fact.fact_id: fact for fact in self.store.user_facts(user_id)}
        if any(fact.scope.user_id != user_id for fact in facts.values()):
            raise ValueError("Store returned facts of another user")
        ids = sorted(facts)
        stored = self.store.fact_states(ids)
        states = {fact_id: state_of(stored, fact_id) for fact_id in ids}
        links = list(self.store.links_among(ids))
        rows = list(self.store.entity_rows(ids))
        existing_macros = list(self.store.user_macros(user_id))

        resolution = self._resolve(facts, states, links, rows, now)
        facts, states, links, rows = _apply(resolution, facts, states, links, rows)

        fact_states, purged_facts = self._tier_facts(facts, states, now)
        for fact_id in purged_facts:
            facts.pop(fact_id)
        gone = set(purged_facts)
        states = {state.fact_id: state for state in fact_states}
        links = [
            link for link in links if link.from_fact_id not in gone and link.to_fact_id not in gone
        ]
        rows = [row for row in rows if row.fact_id not in gone]

        macros = self._macros(user_id, facts, states, links, rows, now, existing_macros)
        macros, purged_macros = self._tier_macros(macros, facts, states, now)

        batch = ReflectBatch(
            user_id=user_id,
            consolidations=resolution.consolidations,
            supersessions=resolution.supersessions,
            reviews=resolution.reviews,
            links=resolution.links,
            fact_states=tuple(fact_states),
            macros=tuple(macros),
            purged_fact_ids=tuple(purged_facts),
            purged_macro_ids=tuple(purged_macros),
        )
        self.store.write_reflect(batch)
        return ReflectResult(user_id, trigger, batch)

    def forget(self, scope: Scope, fact_ids: Sequence[str]) -> ReflectBatch:
        """Explicit delete request: purge the user's own facts among `fact_ids` now and drop
        macros built on them (the next pass rebuilds from what is left).

        A merged fact built from a forgotten original carries its text, so it is purged too.
        """
        owned = {fact.fact_id: fact for fact in self.store.user_facts(scope.user_id)}
        doomed = {fact_id for fact_id in fact_ids if fact_id in owned}
        if not doomed:
            return ReflectBatch(user_id=scope.user_id)
        states = self.store.fact_states(sorted(owned))
        while True:
            derived = {
                state_of(states, fact_id).consolidated_into
                for fact_id in doomed
                if state_of(states, fact_id).consolidated_into in owned
            } - doomed
            if not derived:
                break
            doomed |= derived
        macros = [
            macro.macro_id
            for macro in self.store.user_macros(scope.user_id)
            if doomed & set(macro.supporting_fact_ids)
        ]
        batch = ReflectBatch(
            user_id=scope.user_id,
            purged_fact_ids=tuple(sorted(doomed)),
            purged_macro_ids=tuple(sorted(macros)),
        )
        self.store.write_reflect(batch)
        return batch

    def _resolve(self, facts, states, links, rows, now) -> Resolution:
        if self.judge is None:
            return Resolution()
        config = self.config.consolidation
        candidates = find_candidates(facts, states, links, rows, config)
        if not candidates:
            return Resolution()
        judgments = self.judge(candidates, facts)
        resolution = resolve(judgments, candidates, facts, states, rows, now=now, config=config)
        if not resolution.consolidations:
            return resolution
        texts = [item.merged.text for item in resolution.consolidations]
        vectors = _embed(self.embedder, texts)
        embedded = tuple(
            replace(item, merged=replace(item.merged, embedding=vector))
            for item, vector in zip(resolution.consolidations, vectors)
        )
        return replace(resolution, consolidations=embedded)

    def _macros(self, user_id, facts, states, links, rows, now, existing=()) -> list[MacroMemory]:
        """Every macro of the user after this pass. build_clusters -> match_existing; for each
        cluster that needs_refresh, macro_writer.write -> new_or_updated_macro; one embedder
        call over "title\\nsummary" of every refreshed macro; unmatched current macros are
        closed (valid_to = now). Without a macro_writer, return the existing macros."""
        existing = list(existing)
        if self.macro_writer is None:
            return existing
        by_id = {macro.macro_id: macro for macro in existing}
        clusters = build_clusters(facts, states, links, rows, self.config.macros)
        refreshed: list[MacroMemory] = []
        kept: dict[str, MacroMemory] = {}
        for cluster, matched in match_existing(clusters, existing, self.config.macros):
            if matched is None and macro_id_for(cluster) in by_id:
                matched = by_id[macro_id_for(cluster)]  # closed earlier, same evidence again
            if matched is not None and not needs_refresh(cluster, matched):
                kept[matched.macro_id] = (
                    matched if matched.is_current else replace(matched, valid_to=None)
                )
                continue
            evidence = sorted((facts[i] for i in cluster.fact_ids), key=time_order)
            title, summary = self.macro_writer.write(cluster, evidence, states=states)
            macro = new_or_updated_macro(cluster, matched, title, summary, now=now)
            refreshed.append(macro)
        if refreshed:
            texts = [f"{macro.macro_title}\n{macro.macro_summary}" for macro in refreshed]
            vectors = _embed(self.embedder, texts)
            refreshed = [
                replace(macro, macro_embedding=tuple(vector))
                for macro, vector in zip(refreshed, vectors)
            ]
        touched = {macro.macro_id for macro in refreshed} | set(kept)
        rest = [
            macro if not macro.is_current else replace(macro, valid_to=now)
            for macro in existing
            if macro.macro_id not in touched
        ]
        return sorted([*refreshed, *kept.values(), *rest], key=lambda macro: macro.macro_id)

    def _tier_facts(self, facts, states, now) -> tuple[list[FactState], list[str]]:
        """rank_score + next_tier per fact, then apply_hot_cap. Superseded facts and
        consolidated originals are capped at archived. Returns (updated states, purged ids)."""
        config = self.config.tiers
        scores, proposed, current = {}, {}, {}
        for fact_id in sorted(facts):
            fact, state = facts[fact_id], state_of(states, fact_id)
            current[fact_id] = state
            scores[fact_id] = rank_score(
                access_count=state.access_count,
                last_accessed_at=state.last_accessed_at,
                created_at=fact.recorded_at,
                evidence_count=fact.evidence_count,
                confidence=state.confidence,
                now=now,
                config=config,
            )
            tier = next_tier(
                state.tier,
                scores[fact_id],
                last_seen=state.last_accessed_at or fact.recorded_at,
                now=now,
                config=config,
            )
            history = not fact.is_current or state.consolidated_into is not None
            proposed[fact_id] = tier if tier == Tier.PURGED or not history else Tier.ARCHIVED
        purged = sorted(i for i, tier in proposed.items() if tier == Tier.PURGED)
        kept = {i: tier for i, tier in proposed.items() if tier != Tier.PURGED}
        tiers = apply_hot_cap(kept, scores, config)
        updated = [
            _retier(current[fact_id], tiers[fact_id], scores[fact_id], now)
            for fact_id in sorted(tiers)
        ]
        return updated, purged

    def _tier_macros(self, macros, facts, states, now) -> tuple[list[MacroMemory], list[str]]:
        """Same lifecycle as facts: evidence = len(supporting_fact_ids), confidence = min of
        the supporting facts, created_at = updated_at. Closed macros are capped at archived.
        Returns (updated macros, purged ids)."""
        config = self.config.tiers
        scores, proposed = {}, {}
        by_id = {macro.macro_id: macro for macro in macros}
        for macro_id, macro in sorted(by_id.items()):
            confidences = [
                state_of(states, fact_id).confidence
                for fact_id in macro.supporting_fact_ids
                if fact_id in facts
            ]
            scores[macro_id] = rank_score(
                access_count=macro.access_count,
                last_accessed_at=macro.last_accessed_at,
                created_at=macro.updated_at,
                evidence_count=len(macro.supporting_fact_ids),
                confidence=min(confidences, default=1.0),
                now=now,
                config=config,
            )
            tier = next_tier(
                macro.tier,
                scores[macro_id],
                last_seen=macro.last_accessed_at or macro.updated_at,
                now=now,
                config=config,
            )
            closed = not macro.is_current
            proposed[macro_id] = tier if tier == Tier.PURGED or not closed else Tier.ARCHIVED
        purged = sorted(i for i, tier in proposed.items() if tier == Tier.PURGED)
        kept = {i: tier for i, tier in proposed.items() if tier != Tier.PURGED}
        tiers = apply_hot_cap(kept, scores, config)
        updated = [
            replace(by_id[macro_id], tier=tiers[macro_id], rank_score=scores[macro_id])
            for macro_id in sorted(tiers)
        ]
        return updated, purged


def _retier(state: FactState, tier: Tier, score: float, now: datetime) -> FactState:
    changed = tier != state.tier or state.tier_changed_at is None
    return replace(
        state,
        tier=tier,
        rank_score=score,
        tier_changed_at=now if changed else state.tier_changed_at,
    )


def _embed(embedder: Embedder, texts: Sequence[str]) -> list[tuple[float, ...]]:
    vectors = embedder(list(texts))
    if len(vectors) != len(texts):
        raise ValueError("Embedder must return one vector per text")
    return [tuple(float(value) for value in vector) for vector in vectors]


def _apply(
    resolution: Resolution,
    facts: Mapping[str, MemoryFact],
    states: Mapping[str, FactState],
    links: Sequence[MemoryLink],
    rows: Sequence[FactEntity],
):
    """The view after the resolution's writes, so tiering and clustering see them."""
    facts, states, links, rows = dict(facts), dict(states), list(links), list(rows)
    for item in resolution.consolidations:
        facts[item.merged.fact_id] = item.merged
        states[item.merged.fact_id] = item.state
        rows.extend(item.fact_entities)
    for change in resolution.supersessions:
        old = facts[change.old_fact_id]
        facts[old.fact_id] = replace(
            old, valid_to=max(change.valid_to, old.valid_from), superseded_by=change.new_fact_id
        )
    states.update({state.fact_id: state for state in resolution.states})
    links.extend(resolution.links)
    return facts, states, links, rows


def build_reflector(
    conn_factory,
    *,
    embedder: Embedder,
    generate: StructuredGenerator,
    config: ReflectConfig = ReflectConfig(),
):
    """Wire AlloyDB to caller-provided embedding and structured-generation adapters.

    Reflect selects no SDK or model. Adapters own provider configuration and retries.
    """
    from reflect.consolidation import LLMConflictJudge
    from reflect.macros import LLMMacroWriter
    from reflect.store import AlloyDBReflectStore

    return Reflector(
        store=AlloyDBReflectStore(conn_factory, tier_config=config.tiers),
        embedder=embedder,
        judge=LLMConflictJudge(generate),
        macro_writer=LLMMacroWriter(generate),
        config=config,
    )
