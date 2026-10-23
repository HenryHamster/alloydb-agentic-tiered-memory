"""Reflect pass for one user: consolidate/resolve -> synthesize macros -> re-rank and re-tier."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from reflect.consolidation import ConflictJudge, ConsolidationConfig
from reflect.macros import MacroConfig, MacroWriter
from reflect.models import FactState, MacroMemory, ReflectBatch, ReflectResult, ReflectTrigger
from reflect.store import ReflectStore
from reflect.tiers import TierConfig
from retain.embeddings import Embedder
from retain.models import Scope


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

        1. Read user_facts, fact_states, links_among, entity_rows.
        2. find_candidates -> judge -> resolve; embed merged facts. Apply the resolution to the
           in-memory view so steps 3-4 see merged facts and closed validity.
        3. _macros: clusters, refresh changed ones, close dissolved ones.
        4. _tier_facts and _tier_macros.
        5. store.write_reflect(ReflectBatch(...)) once.
        """
        # TODO(reflect): wire the stages above.
        raise NotImplementedError("Reflect pass")

    def forget(self, scope: Scope, fact_ids: Sequence[str]) -> ReflectBatch:
        """Explicit delete request: purge the user's own facts among `fact_ids` now and drop
        macros built on them (the next pass rebuilds from what is left)."""
        # TODO(reflect): filter to scope.user_id, collect dependent macros, write_reflect.
        raise NotImplementedError("Forget")

    def _macros(self, user_id, facts, states, links, rows, now) -> list[MacroMemory]:
        """Every macro of the user after this pass. build_clusters -> match_existing; for each
        cluster that needs_refresh, macro_writer.write -> new_or_updated_macro; one embedder
        call over "title\\nsummary" of every refreshed macro; unmatched current macros are
        closed (valid_to = now). Without a macro_writer, return the existing macros."""
        # TODO(reflect): implement.
        raise NotImplementedError("Macro pass")

    def _tier_facts(self, facts, states, now) -> tuple[list[FactState], list[str]]:
        """rank_score + next_tier per fact, then apply_hot_cap. Superseded facts and
        consolidated originals are capped at archived. Returns (updated states, purged ids)."""
        # TODO(reflect): implement.
        raise NotImplementedError("Fact tiers")

    def _tier_macros(self, macros, facts, states, now) -> tuple[list[MacroMemory], list[str]]:
        """Same lifecycle as facts: evidence = len(supporting_fact_ids), confidence = min of
        the supporting facts, created_at = updated_at. Closed macros are capped at archived.
        Returns (updated macros, purged ids)."""
        # TODO(reflect): implement.
        raise NotImplementedError("Macro tiers")


def build_reflector(conn_factory, genai_client, *, config: ReflectConfig = ReflectConfig()):
    """Production wiring: AlloyDB store with Gemini stages. Stages are skeletons for now."""
    from reflect.alloydb_store import AlloyDBReflectStore
    from reflect.consolidation import GeminiConflictJudge
    from reflect.macros import GeminiMacroWriter
    from retain.embeddings import GeminiEmbedder

    return Reflector(
        store=AlloyDBReflectStore(conn_factory),
        embedder=GeminiEmbedder(genai_client),
        judge=GeminiConflictJudge(genai_client),
        macro_writer=GeminiMacroWriter(genai_client),
        config=config,
    )
