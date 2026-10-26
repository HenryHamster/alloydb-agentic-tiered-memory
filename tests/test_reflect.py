"""Offline checks of the Reflect skeleton: records, configs and wiring. Stages are stubs."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from reflect import Reflector, TierConfig, build_reflector
from reflect.models import FactState, ReflectBatch, Tier
from reflect.store import AlloyDBReflectStore
from reflect.tiers import next_tier
from retain.models import LinkType, Relation

T0 = datetime(2026, 10, 19, 12, tzinfo=timezone.utc)


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


def test_production_wiring_builds_and_unfinished_stages_fail_loudly():
    generate = MagicMock()
    embedder = MagicMock()
    production = build_reflector(MagicMock(), embedder=embedder, generate=generate)
    assert isinstance(production, Reflector)
    assert isinstance(production.store, AlloyDBReflectStore)
    assert production.embedder is embedder
    assert production.judge.generate is generate
    assert production.macro_writer.generate is generate
    generate.assert_not_called()
    embedder.assert_not_called()
    with pytest.raises(NotImplementedError):
        production.reflect("user_dev_42")
    with pytest.raises(NotImplementedError):
        production.judge([], {})
    with pytest.raises(NotImplementedError):
        production.macro_writer.write(None, [])
    with pytest.raises(NotImplementedError):
        production.store.write_reflect(ReflectBatch())
    with pytest.raises(NotImplementedError):
        next_tier(Tier.HOT, 1.0, last_seen=T0, now=T0)
