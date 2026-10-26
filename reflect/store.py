"""Reflect persistence contract, an in-memory reference store, and the AlloyDB implementation
over sql/004 + sql/005."""

import json
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from threading import RLock
from typing import Protocol

from psycopg2.extras import RealDictCursor

from reflect.models import ConflictReview, FactState, MacroMemory, ReflectBatch, Tier
from reflect.tiers import TierConfig, promote_on_hit
from retain.alloydb_store import (
    _FACT_COLUMNS,
    _apply_supersessions,
    _fact,
    _insert_fact,
    _load_facts,
    _validate_links,
    _vector,
)
from retain.models import (
    EntityRole,
    FactEntity,
    MemoryFact,
    MemoryLink,
    Relation,
    Scope,
    Visibility,
)
from retain.store import InMemoryRetainStore


class ReflectStore(Protocol):
    """Reads are per user (Reflect is a per-user batch job); writes are one transaction."""

    def user_facts(self, user_id: str) -> list[MemoryFact]:
        """Every stored fact of the user, current or closed, in any tier. Embeddings may be
        omitted: Reflect finds candidates through links, not vector search."""
        ...

    def fact_states(self, fact_ids: Sequence[str]) -> dict[str, FactState]:
        """Lifecycle columns per fact; defaults (hot, rank 0, confidence 1) for untouched rows."""
        ...

    def links_among(self, fact_ids: Sequence[str]) -> list[MemoryLink]:
        """memory_links rows whose both endpoints are in `fact_ids`."""
        ...

    def entity_rows(self, fact_ids: Sequence[str]) -> list[FactEntity]: ...

    def user_macros(self, user_id: str) -> list[MacroMemory]:
        """Every macro of the user, current or closed. Embeddings may be omitted; a macro
        written back without one keeps its stored embedding."""
        ...

    def record_access(
        self, fact_ids: Sequence[str], macro_ids: Sequence[str], at: datetime
    ) -> None:
        """Called by Recall for every row it returns: access_count + 1, last_accessed_at = at,
        and an archived row is promoted back immediately (tiers.promote_on_hit)."""
        ...

    def write_reflect(self, batch: ReflectBatch) -> None:
        """Apply atomically. Rejects the whole batch if a referenced fact is missing, belongs to
        another user, or a supersession target is already closed. Purges delete rows (links and
        entity rows cascade; superseded_by / consolidated_into pointing at them become NULL).
        access_count and last_accessed_at are only written for new merged facts, so accesses
        Recall records during a pass are not lost."""
        ...


class InMemoryReflectStore:
    """Reference implementation of ReflectStore over an InMemoryRetainStore. Not durable."""

    def __init__(self, retain_store: InMemoryRetainStore, tier_config: TierConfig = TierConfig()):
        self.retain_store = retain_store
        self.tier_config = tier_config
        self.states: dict[str, FactState] = {}
        self.macros: dict[str, MacroMemory] = {}
        self.reviews: dict[str, ConflictReview] = {}

    @property
    def _lock(self) -> RLock:
        return self.retain_store._lock

    def user_facts(self, user_id):
        with self._lock:
            facts = list(self.retain_store.facts.values())
        return sorted(
            (fact for fact in facts if fact.scope.user_id == user_id),
            key=lambda fact: fact.fact_id,
        )

    def fact_states(self, fact_ids):
        with self._lock:
            return {
                fact_id: self.states.get(fact_id) or FactState(fact_id)
                for fact_id in dict.fromkeys(fact_ids)
                if fact_id in self.retain_store.facts
            }

    def links_among(self, fact_ids):
        wanted = set(fact_ids)
        with self._lock:
            links = list(self.retain_store.links.values())
        return [link for link in links if link.from_fact_id in wanted and link.to_fact_id in wanted]

    def entity_rows(self, fact_ids):
        wanted = set(fact_ids)
        with self._lock:
            rows = list(self.retain_store.fact_entities)
        return sorted(
            (row for row in rows if row.fact_id in wanted),
            key=lambda row: (row.fact_id, row.entity_id, row.role.value),
        )

    def user_macros(self, user_id):
        with self._lock:
            return sorted(
                (m for m in self.macros.values() if m.scope.user_id == user_id),
                key=lambda macro: macro.macro_id,
            )

    def record_access(self, fact_ids, macro_ids, at):
        with self._lock:
            for fact_id in dict.fromkeys(fact_ids):
                if fact_id not in self.retain_store.facts:
                    continue
                state = self.states.get(fact_id) or FactState(fact_id)
                tier = promote_on_hit(state.tier, self.tier_config)
                self.states[fact_id] = replace(
                    state,
                    access_count=state.access_count + 1,
                    last_accessed_at=at,
                    tier=tier,
                    tier_changed_at=at if tier != state.tier else state.tier_changed_at,
                )
            for macro_id in dict.fromkeys(macro_ids):
                macro = self.macros.get(macro_id)
                if macro is not None:
                    self.macros[macro_id] = replace(
                        macro,
                        access_count=macro.access_count + 1,
                        last_accessed_at=at,
                        tier=promote_on_hit(macro.tier, self.tier_config),
                    )

    def write_reflect(self, batch):
        with self._lock:
            retain = self.retain_store
            facts = dict(retain.facts)
            states = dict(self.states)
            links = dict(retain.links)
            fact_entities = set(retain.fact_entities)

            def owned(fact_id):
                fact = facts.get(fact_id)
                if fact is None:
                    raise ValueError(f"Reflect batch references missing fact {fact_id}")
                if batch.user_id is not None and fact.scope.user_id != batch.user_id:
                    raise ValueError(f"Fact {fact_id} belongs to another user")
                return fact

            for item in batch.consolidations:
                for fact_id in item.original_ids:
                    owned(fact_id)
                previous = facts.setdefault(item.merged.fact_id, item.merged)
                if previous != item.merged:
                    raise ValueError(f"Conflicting merged fact {item.merged.fact_id}")
                states[item.merged.fact_id] = item.state
                fact_entities |= set(item.fact_entities)
            for change in batch.supersessions:
                old, new = owned(change.old_fact_id), owned(change.new_fact_id)
                if not old.is_current or not new.is_current:
                    raise ValueError(f"Cannot supersede {change.old_fact_id}: missing or closed")
                facts[old.fact_id] = replace(
                    old,
                    valid_to=max(change.valid_to, old.valid_from),
                    superseded_by=new.fact_id,
                )
            merged_ids = {item.merged.fact_id for item in batch.consolidations}
            for state in batch.fact_states:
                owned(state.fact_id)
                if state.fact_id in merged_ids:
                    states[state.fact_id] = state
                    continue
                previous = states.get(state.fact_id) or FactState(state.fact_id)
                states[state.fact_id] = replace(
                    state,
                    access_count=previous.access_count,
                    last_accessed_at=previous.last_accessed_at,
                )
            for link in batch.links:
                owned(link.from_fact_id), owned(link.to_fact_id)
                key = (link.from_fact_id, link.to_fact_id, link.relation)
                old_link = links.get(key)
                links[key] = old_link if old_link and old_link.weight >= link.weight else link
            macros = dict(self.macros)
            for macro in batch.macros:
                if batch.user_id is not None and macro.scope.user_id != batch.user_id:
                    raise ValueError(f"Macro {macro.macro_id} belongs to another user")
                for fact_id in macro.supporting_fact_ids:
                    if fact_id in facts:  # evidence purged by an earlier pass may be gone
                        owned(fact_id)
                previous = macros.get(macro.macro_id)
                if previous is not None:
                    macro = replace(
                        macro,
                        access_count=previous.access_count,
                        last_accessed_at=previous.last_accessed_at,
                        macro_embedding=macro.macro_embedding or previous.macro_embedding,
                    )
                macros[macro.macro_id] = macro

            purged = set(batch.purged_fact_ids)
            for fact_id in purged:
                if fact_id in facts:
                    owned(fact_id)
            for fact_id in purged:
                facts.pop(fact_id, None)
                states.pop(fact_id, None)
            for fact_id, fact in list(facts.items()):
                if fact.superseded_by in purged:
                    facts[fact_id] = replace(fact, superseded_by=None)
            for fact_id, state in list(states.items()):
                if state.consolidated_into in purged:
                    states[fact_id] = replace(state, consolidated_into=None)
            links = {
                key: link
                for key, link in links.items()
                if link.from_fact_id not in purged and link.to_fact_id not in purged
            }
            fact_entities = {row for row in fact_entities if row.fact_id not in purged}
            for macro_id in batch.purged_macro_ids:
                macro = macros.get(macro_id)
                if macro is not None and batch.user_id not in (None, macro.scope.user_id):
                    raise ValueError(f"Macro {macro_id} belongs to another user")
                macros.pop(macro_id, None)

            # Validation done: commit.
            retain.facts, retain.links, retain.fact_entities = facts, links, fact_entities
            retain.entity_relations = {
                row for row in retain.entity_relations if row.fact_id not in purged
            }
            self.states, self.macros = states, macros
            for review in batch.reviews:
                self.reviews.setdefault(review.review_id, review)


# Reflect reads every fact of a user, so it skips the 768-float embedding column.
_STATE_FACT_COLUMNS = _FACT_COLUMNS.replace("f.embedding::text,", "NULL::text AS embedding,")
assert _STATE_FACT_COLUMNS != _FACT_COLUMNS, "retain.alloydb_store._FACT_COLUMNS changed"

_MACRO_COLUMNS = """
    macro_id::text, user_id, project_id, session_id, visibility, macro_title, macro_summary,
    supporting_fact_ids::text[] AS supporting_fact_ids,
    primary_entity_ids::text[] AS primary_entity_ids, causal_chain_ids,
    lower(temporal_span) AS span_start, upper(temporal_span) AS span_end, version,
    access_count, last_accessed_at, rank_score, tier, valid_to, created_at, updated_at
"""


class AlloyDBReflectStore:
    """Same connection handling as retain.alloydb_store.AlloyDBRetainStore.

    Reflect runs as a per-user batch job; write_reflect takes
    `pg_advisory_xact_lock(hashtextextended('reflect:' || user_id, 0))` so two passes for one
    user never interleave.
    """

    def __init__(self, conn_factory, *, tier_config: TierConfig = TierConfig()):
        self.conn_factory = conn_factory
        self.tier_config = tier_config
        self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def user_facts(self, user_id):
        with self._transaction() as cur:
            cur.execute(
                f"SELECT {_STATE_FACT_COLUMNS} FROM memory_facts f "
                "LEFT JOIN memory_entities me USING (fact_id) "
                "WHERE f.user_id = %s GROUP BY f.fact_id ORDER BY f.fact_id",
                (user_id,),
            )
            return [_fact(row) for row in cur.fetchall()]

    def fact_states(self, fact_ids):
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return {}
        with self._transaction() as cur:
            cur.execute(
                "SELECT fact_id::text, access_count, last_accessed_at, rank_score, tier, "
                "tier_changed_at, confidence, consolidated_into::text, "
                "conflicts_with::text[] AS conflicts_with "
                "FROM memory_facts WHERE fact_id = ANY(%s::uuid[])",
                (ids,),
            )
            return {row["fact_id"]: _state(row) for row in cur.fetchall()}

    def links_among(self, fact_ids):
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return []
        with self._transaction() as cur:
            cur.execute(
                "SELECT from_fact_id::text, to_fact_id::text, relation, weight FROM memory_links "
                "WHERE from_fact_id = ANY(%(ids)s::uuid[]) AND to_fact_id = ANY(%(ids)s::uuid[]) "
                "ORDER BY from_fact_id, to_fact_id, relation",
                {"ids": ids},
            )
            return [
                MemoryLink(
                    row["from_fact_id"],
                    row["to_fact_id"],
                    Relation(row["relation"]),
                    float(row["weight"]),
                )
                for row in cur.fetchall()
            ]

    def entity_rows(self, fact_ids):
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return []
        with self._transaction() as cur:
            cur.execute(
                "SELECT fact_id::text, entity_id::text, role FROM memory_entities "
                "WHERE fact_id = ANY(%s::uuid[]) ORDER BY fact_id, entity_id, role",
                (ids,),
            )
            return [
                FactEntity(row["fact_id"], row["entity_id"], EntityRole(row["role"]))
                for row in cur.fetchall()
            ]

    def user_macros(self, user_id):
        with self._transaction() as cur:
            cur.execute(
                f"SELECT {_MACRO_COLUMNS} FROM macro_memories WHERE user_id = %s ORDER BY macro_id",
                (user_id,),
            )
            return [_macro(row) for row in cur.fetchall()]

    def record_access(self, fact_ids, macro_ids, at):
        """Recall calls this off its hot path (fire-and-forget)."""
        promoted = promote_on_hit(Tier.ARCHIVED, self.tier_config).value
        facts, macros = list(dict.fromkeys(fact_ids)), list(dict.fromkeys(macro_ids))
        if not facts and not macros:
            return
        with self._transaction() as cur:
            for table, key, ids in (
                ("memory_facts", "fact_id", facts),
                ("macro_memories", "macro_id", macros),
            ):
                if not ids:
                    continue
                cur.execute(
                    f"UPDATE {table} SET access_count = access_count + 1, "
                    "last_accessed_at = GREATEST(last_accessed_at, %(at)s), "
                    "tier_changed_at = CASE WHEN tier = 'archived' THEN %(at)s "
                    "ELSE tier_changed_at END, "
                    "tier = CASE WHEN tier = 'archived' THEN %(promoted)s ELSE tier END "
                    f"WHERE {key} = ANY(%(ids)s::uuid[])",
                    {"at": at, "promoted": promoted, "ids": ids},
                )

    def write_reflect(self, batch):
        user_id = batch.user_id
        if not user_id:
            raise ValueError("write_reflect needs the batch's user_id")
        if not any(
            (
                batch.consolidations,
                batch.supersessions,
                batch.reviews,
                batch.links,
                batch.fact_states,
                batch.macros,
                batch.purged_fact_ids,
                batch.purged_macro_ids,
            )
        ):
            return
        with self._transaction() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"reflect:{user_id}",)
            )
            merged = [item.merged for item in batch.consolidations]
            for fact in merged:
                if fact.scope.user_id != user_id:
                    raise ValueError(f"Merged fact {fact.fact_id} belongs to another user")
                if fact.embedding is None:
                    raise ValueError(f"Merged fact {fact.fact_id} has no embedding")
                _insert_fact(cur, fact)

            facts = _load_facts(cur, _referenced_fact_ids(batch))
            missing = _referenced_fact_ids(batch) - set(facts)
            if missing:
                raise ValueError(f"Reflect batch references missing facts {sorted(missing)}")
            if any(fact.scope.user_id != user_id for fact in facts.values()):
                raise ValueError("Reflect batch references another user's facts")

            for item in batch.consolidations:
                for row in item.fact_entities:
                    cur.execute(
                        "INSERT INTO memory_entities (fact_id, entity_id, role) "
                        "VALUES (%s::uuid, %s::uuid, %s) ON CONFLICT DO NOTHING",
                        (row.fact_id, row.entity_id, row.role.value),
                    )
                cur.execute(
                    "UPDATE memory_facts SET access_count = %s, last_accessed_at = %s "
                    "WHERE fact_id = %s::uuid AND access_count = 0",
                    (item.state.access_count, item.state.last_accessed_at, item.merged.fact_id),
                )

            _apply_supersessions(batch.supersessions, facts, cur)

            if batch.fact_states:
                cur.execute(
                    "UPDATE memory_facts f SET rank_score = s.rank_score, tier = s.tier, "
                    "tier_changed_at = COALESCE(s.tier_changed_at, f.tier_changed_at), "
                    "confidence = s.confidence, "
                    "consolidated_into = s.consolidated_into, conflicts_with = s.conflicts_with "
                    "FROM jsonb_to_recordset(%s::jsonb) AS s(fact_id uuid, rank_score real, "
                    "tier text, tier_changed_at timestamptz, confidence real, "
                    "consolidated_into uuid, conflicts_with uuid[]) "
                    "WHERE f.fact_id = s.fact_id AND f.user_id = %s",
                    (json.dumps([_state_json(s) for s in batch.fact_states]), user_id),
                )
                if cur.rowcount != len({state.fact_id for state in batch.fact_states}):
                    raise ValueError("Fact state update touched a missing fact")

            _validate_links(batch.links, facts)
            for link in batch.links:
                cur.execute(
                    "INSERT INTO memory_links (from_fact_id, to_fact_id, link_type, relation, weight) "
                    "VALUES (%s::uuid, %s::uuid, %s, %s, %s) "
                    "ON CONFLICT (from_fact_id, to_fact_id, relation) DO UPDATE "
                    "SET weight = GREATEST(memory_links.weight, EXCLUDED.weight)",
                    (
                        link.from_fact_id,
                        link.to_fact_id,
                        link.link_type.value,
                        link.relation.value,
                        link.weight,
                    ),
                )

            for review in batch.reviews:
                if review.user_id != user_id:
                    raise ValueError(f"Review {review.review_id} belongs to another user")
                cur.execute(
                    "INSERT INTO memory_conflict_reviews (review_id, user_id, fact_ids, reason) "
                    "VALUES (%s::uuid, %s, %s::uuid[], %s) ON CONFLICT (review_id) DO NOTHING",
                    (review.review_id, review.user_id, list(review.fact_ids), review.reason),
                )

            evidence = {i for macro in batch.macros for i in macro.supporting_fact_ids}
            if evidence:
                # Evidence purged by an earlier pass may be gone; what remains must be ours.
                cur.execute(
                    "SELECT count(*) AS other_users FROM memory_facts "
                    "WHERE fact_id = ANY(%s::uuid[]) AND user_id <> %s",
                    (sorted(evidence), user_id),
                )
                if cur.fetchone()["other_users"]:
                    raise ValueError("A macro cites another user's facts")
            for macro in batch.macros:
                if macro.scope.user_id != user_id:
                    raise ValueError(f"Macro {macro.macro_id} belongs to another user")
                _upsert_macro(cur, macro)

            if batch.purged_fact_ids:
                cur.execute(
                    "DELETE FROM memory_facts WHERE fact_id = ANY(%s::uuid[]) AND user_id = %s",
                    (list(batch.purged_fact_ids), user_id),
                )
            if batch.purged_macro_ids:
                cur.execute(
                    "DELETE FROM macro_memories WHERE macro_id = ANY(%s::uuid[]) AND user_id = %s",
                    (list(batch.purged_macro_ids), user_id),
                )

    @contextmanager
    def _transaction(self):
        conn = self._connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            if getattr(conn, "closed", False):
                conn.close()
                self._conn = None
            raise

    def _connection(self):
        if self._conn is None or getattr(self._conn, "closed", False):
            self._conn = self.conn_factory()
        return self._conn


def _referenced_fact_ids(batch: ReflectBatch) -> set[str]:
    return {
        *(i for item in batch.consolidations for i in (item.merged.fact_id, *item.original_ids)),
        *(i for change in batch.supersessions for i in (change.old_fact_id, change.new_fact_id)),
        *(i for link in batch.links for i in (link.from_fact_id, link.to_fact_id)),
        *(state.fact_id for state in batch.fact_states),
        *(i for review in batch.reviews for i in review.fact_ids),
    }


def _state(row) -> FactState:
    return FactState(
        fact_id=row["fact_id"],
        access_count=row["access_count"],
        last_accessed_at=row["last_accessed_at"],
        rank_score=float(row["rank_score"]),
        tier=Tier(row["tier"]),
        tier_changed_at=row["tier_changed_at"],
        confidence=float(row["confidence"]),
        consolidated_into=row["consolidated_into"],
        conflicts_with=tuple(sorted(row["conflicts_with"] or ())),
    )


def _state_json(state: FactState) -> dict:
    return {
        "fact_id": state.fact_id,
        "rank_score": state.rank_score,
        "tier": state.tier.value,
        "tier_changed_at": state.tier_changed_at.isoformat() if state.tier_changed_at else None,
        "confidence": state.confidence,
        "consolidated_into": state.consolidated_into,
        "conflicts_with": list(state.conflicts_with),
    }


def _macro(row) -> MacroMemory:
    span = None
    if row["span_start"] is not None and row["span_end"] is not None:
        span = (row["span_start"], row["span_end"])
    chains = row["causal_chain_ids"]
    if isinstance(chains, str):
        chains = json.loads(chains)
    return MacroMemory(
        macro_id=row["macro_id"],
        scope=Scope(row["user_id"], row["project_id"], row["session_id"]),
        visibility=Visibility(row["visibility"]),
        macro_title=row["macro_title"],
        macro_summary=row["macro_summary"],
        supporting_fact_ids=tuple(row["supporting_fact_ids"] or ()),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        primary_entity_ids=tuple(row["primary_entity_ids"] or ()),
        causal_chain_ids=tuple(tuple(str(i) for i in chain) for chain in chains or ()),
        temporal_span=span,
        version=row["version"],
        access_count=row["access_count"],
        last_accessed_at=row["last_accessed_at"],
        rank_score=float(row["rank_score"]),
        tier=Tier(row["tier"]),
        valid_to=row["valid_to"],
    )


def _upsert_macro(cur, macro: MacroMemory) -> None:
    """Insert, or update every column except usage counters (Recall owns those) and keep the
    stored embedding when this version carries none. updated_at comes from the record, so an
    unchanged macro keeps cooling instead of looking new on every pass."""
    span_start, span_end = macro.temporal_span or (None, None)
    params = {
        "macro_id": macro.macro_id,
        "user_id": macro.scope.user_id,
        "project_id": macro.scope.project_id,
        "session_id": macro.scope.session_id,
        "visibility": macro.visibility.value,
        "title": macro.macro_title,
        "summary": macro.macro_summary,
        "embedding": _vector(macro.macro_embedding) if macro.macro_embedding else None,
        "supporting": list(macro.supporting_fact_ids),
        "entities": list(macro.primary_entity_ids),
        "chains": json.dumps([list(chain) for chain in macro.causal_chain_ids]),
        "span_start": span_start,
        "span_end": span_end,
        "version": macro.version,
        "access_count": macro.access_count,
        "last_accessed_at": macro.last_accessed_at,
        "rank_score": macro.rank_score,
        "tier": macro.tier.value,
        "valid_to": macro.valid_to,
        "created_at": macro.created_at,
        "updated_at": macro.updated_at,
    }
    cur.execute(
        "INSERT INTO macro_memories (macro_id, user_id, project_id, session_id, visibility, "
        "macro_title, macro_summary, macro_embedding, supporting_fact_ids, primary_entity_ids, "
        "causal_chain_ids, temporal_span, version, access_count, last_accessed_at, rank_score, "
        "tier, valid_to, created_at, updated_at) "
        "VALUES (%(macro_id)s::uuid, %(user_id)s, %(project_id)s, %(session_id)s, "
        "%(visibility)s, %(title)s, %(summary)s, %(embedding)s::vector, %(supporting)s::uuid[], "
        "%(entities)s::uuid[], %(chains)s::jsonb, "
        "CASE WHEN %(span_start)s::timestamptz IS NULL THEN NULL "
        "ELSE tstzrange(%(span_start)s::timestamptz, %(span_end)s::timestamptz, '[]') END, "
        "%(version)s, %(access_count)s, %(last_accessed_at)s, %(rank_score)s, %(tier)s, "
        "%(valid_to)s, %(created_at)s, %(updated_at)s) "
        "ON CONFLICT (macro_id) DO UPDATE SET "
        "macro_title = EXCLUDED.macro_title, macro_summary = EXCLUDED.macro_summary, "
        "macro_embedding = COALESCE(EXCLUDED.macro_embedding, macro_memories.macro_embedding), "
        "supporting_fact_ids = EXCLUDED.supporting_fact_ids, "
        "primary_entity_ids = EXCLUDED.primary_entity_ids, "
        "causal_chain_ids = EXCLUDED.causal_chain_ids, temporal_span = EXCLUDED.temporal_span, "
        "version = EXCLUDED.version, rank_score = EXCLUDED.rank_score, "
        "tier_changed_at = CASE WHEN macro_memories.tier <> EXCLUDED.tier THEN NOW() "
        "ELSE macro_memories.tier_changed_at END, "
        "tier = EXCLUDED.tier, valid_to = EXCLUDED.valid_to, updated_at = EXCLUDED.updated_at "
        "WHERE macro_memories.user_id = EXCLUDED.user_id "
        "AND macro_memories.visibility = EXCLUDED.visibility "
        "AND macro_memories.project_id IS NOT DISTINCT FROM EXCLUDED.project_id "
        "AND macro_memories.session_id IS NOT DISTINCT FROM EXCLUDED.session_id",
        params,
    )
    if cur.rowcount == 0:
        raise ValueError(f"Conflicting macro ID {macro.macro_id}")
