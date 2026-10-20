"""Proposal step 4: consolidate related facts into observations (refined later by Reflect)."""

from typing import Protocol

from retain.models import Observation, Scope


class ObservationBuilder(Protocol):
    def consolidate(self, reader: Scope) -> list[Observation]:
        """Create, refresh or close observations visible to `reader`; return those written."""
        ...


class GeminiObservationBuilder:
    """Periodic job, not part of the per-turn path.

    Groups current facts by (entity_id, visibility domain) so an observation never mixes
    visibility levels. A group with >= `min_facts` facts gets an LLM summary and embedding;
    observation_id = stable_id(user_id, visibility, domain id, entity_id). Regenerate only when
    the group's fact_ids changed; close (valid_to) observations whose group fell below
    `min_facts` or whose facts were all superseded.
    """

    def __init__(self, client, store, embedder, *, model="gemini-3.5-flash", min_facts=3):
        self.client = client
        self.store = store
        self.embedder = embedder
        self.model = model
        self.min_facts = min_facts

    def consolidate(self, reader: Scope) -> list[Observation]:
        # TODO(retain): implement the class rules; store.write_observations persists them.
        raise NotImplementedError("Observation consolidation")
