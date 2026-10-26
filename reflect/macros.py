"""MacroMemories: cluster linked facts, then synthesize one narrative per cluster.

Clusters are found by traversing memory_links (temporal, causal, strong semantic) and shared
subject/object entities in memory_entities. Superseded facts stay in, because the history is
the story ("2 hours" -> DuckDB -> "10 minutes"); consolidated originals are replaced by their
merged fact. A cluster never spans visibility domains, so a macro is readable exactly where all
of its evidence is.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from reflect.generation import StructuredGenerator
from reflect.models import Cluster, FactState, MacroMemory
from retain.models import FactEntity, MemoryFact, MemoryLink


@dataclass(frozen=True)
class MacroConfig:
    min_facts: int = 3
    semantic_min: float = 0.8  # semantic links only join clusters when this strong
    max_entity_facts: int = 25  # hub entities (e.g. "user") would merge everything: ignore them
    match_overlap: float = 0.5  # Jaccard needed to treat a cluster as an existing macro


def build_clusters(
    facts: Mapping[str, MemoryFact],
    states: Mapping[str, FactState],
    links: Sequence[MemoryLink],
    fact_entities: Sequence[FactEntity],
    config: MacroConfig = MacroConfig(),
) -> list[Cluster]:
    """Connected components of >= min_facts facts within one visibility domain.

    Eligible: every fact not consolidated_into another (current or superseded). Edges:
    near_in_time, supersedes, caused_by, similar >= semantic_min, and shared subject/object
    entities with <= max_entity_facts facts. Each Cluster carries domain_scope, entity_ids,
    causal_chains() and temporal_span() of its own links.
    """
    # TODO(reflect): union-find over the edges above.
    raise NotImplementedError("Fact clusters")


def causal_chains(links: Sequence[MemoryLink], *, limit: int = 20) -> tuple[tuple[str, ...], ...]:
    """Maximal cause -> effect paths over caused_by edges (stored effect -> cause), starting
    at causes that have no cause themselves; cycles are cut; at most `limit` chains."""
    # TODO(reflect): DFS from roots.
    raise NotImplementedError("Causal chains")


def temporal_span(
    facts: Sequence[MemoryFact], links: Sequence[MemoryLink]
) -> tuple[datetime, datetime] | None:
    """Min/max (occurred_at or valid_from) of facts touching a temporal link; all facts if none
    do; None for no facts."""
    # TODO(reflect): implement.
    raise NotImplementedError("Temporal span")


def match_existing(
    clusters: Sequence[Cluster],
    existing: Sequence[MacroMemory],
    config: MacroConfig = MacroConfig(),
) -> list[tuple[Cluster, MacroMemory | None]]:
    """Pair each cluster with the current macro of the same domain it overlaps most
    (Jaccard of fact ids >= match_overlap); each macro is used at most once."""
    # TODO(reflect): greedy best-overlap match.
    raise NotImplementedError("Match macros")


def needs_refresh(cluster: Cluster, existing: MacroMemory | None) -> bool:
    """Regenerate (and re-embed) only when the evidence set changed or the macro is new."""
    # TODO(reflect): compare supporting_fact_ids with cluster.fact_ids.
    raise NotImplementedError("Needs refresh")


def new_or_updated_macro(
    cluster: Cluster, existing: MacroMemory | None, title: str, summary: str, *, now: datetime
) -> MacroMemory:
    """New macro at version 1 (macro_id = stable_id("macro", user, visibility, *fact_ids)), or
    the existing one at version + 1 keeping macro_id and usage counters. Embedding is reset;
    the caller re-embeds title + summary for every returned macro."""
    # TODO(reflect): build or replace the MacroMemory.
    raise NotImplementedError("Macro record")


class MacroWriter(Protocol):
    def write(self, cluster: Cluster, facts: Sequence[MemoryFact]) -> tuple[str, str]:
        """(macro_title, macro_summary) for the cluster."""
        ...


class LLMMacroWriter:
    """LLM synthesis of one cluster, off the request path.

    Prompt input: facts oldest first (fact_id, text, evidence, occurred_at, valid_from,
    valid_to, superseded_by, conflicts_with), the cluster's causal chains and temporal span,
    and the primary entity names. Output JSON: title (<= 8 words), summary (1-3 sentences,
    past tense for closed facts, mentions unresolved conflicts, no claims beyond the facts).
    Example: facts "weekly Python script takes 2 hours", "added DuckDB to replace Pandas",
    "weekly script now finishes in under 10 minutes" -> "User optimized their weekly data
    pipeline by switching from Pandas to DuckDB, cutting runtime from 2 hours to 10 minutes."
    """

    def __init__(self, generate: StructuredGenerator):
        self.generate = generate

    def write(self, cluster, facts):
        # TODO(reflect): build prompt/schema, call self.generate; reject empty title/summary.
        raise NotImplementedError("LLM macro synthesis")
