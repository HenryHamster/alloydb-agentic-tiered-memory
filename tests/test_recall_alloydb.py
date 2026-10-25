"""Offline checks of AlloyDBRecallStore: SQL parameters, row decoding and connection handling.

No database runs here; a scripted connection records statements and returns canned rows.
"""

import itertools
import re
from datetime import datetime, timezone

import pytest

from recall.alloydb_store import AlloyDBRecallStore, _query_grams
from recall.config import MemoryType
from recall.models import TimeWindow
from retain.models import Scope

SCOPE = Scope("user_1", "project_a", "session_1")
NOW = datetime(2026, 10, 18, tzinfo=timezone.utc)
FACT_ID = "0b6f1e5a-6f43-4f8e-9a52-2f0f6bfe6b11"


class FakeConnection:
    def __init__(self, rows=(), fail_with=None, drop=False):
        self.rows = list(rows)
        self.fail_with = fail_with
        self.drop = drop  # the failure also closes the connection
        self.statements: list[tuple[str, dict | None]] = []
        self.closed = 0
        self.commits = self.rollbacks = 0

    def set_session(self, **kwargs):
        pass

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = 1


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if params is not None and self.conn.fail_with is not None:
            self.conn.closed = 2 if self.conn.drop else 0
            raise self.conn.fail_with
        self.conn.statements.append((sql, params))

    def fetchall(self):
        return self.conn.rows


def queries(conn):
    return [(sql, params) for sql, params in conn.statements if params is not None]


def fact_row(fact_id=FACT_ID, text="Backups run nightly", **extra):
    return {
        "fact_id": fact_id,
        "user_id": "user_1",
        "project_id": "project_a",
        "session_id": "session_1",
        "visibility": "global",
        "fact_type": "world",
        "category": "event",
        "speaker": "user",
        "claim_type": "asserted",
        "text": text,
        "source_document_id": None,
        "source_episode_id": "round-1",
        "occurred_at": "2026-10-01T09:00:00+00:00",
        "recorded_at": "2026-10-01T09:00:05+00:00",
        "valid_from": "2026-10-01T09:00:00+00:00",
        "valid_to": None,
        "superseded_by": None,
        "evidence_count": 2,
        "entity_ids": ["e1"],
        **extra,
    }


@pytest.mark.parametrize(
    "types,vectors,window,n_queries",
    [
        (types, vectors, window, n)
        for r in range(1, 5)
        for types in itertools.combinations(list(MemoryType), r)
        for vectors, window, n in itertools.product([True, False], [True, False], [1, 3])
    ],
)
def test_every_search_placeholder_is_bound(types, vectors, window, n_queries):
    conn = FakeConnection()
    store = AlloyDBRecallStore(lambda: conn)
    store.search(
        SCOPE,
        ["what did Bank of America say about Project Falcon"] * n_queries,
        [[0.1, 0.2]] * n_queries if vectors else None,
        TimeWindow(datetime(2025, 1, 1), datetime(2025, 2, 1)) if window else None,
        types,
        NOW,
        False,
        50,
    )
    for sql, params in queries(conn):
        assert set(re.findall(r"%\((\w+)\)s", sql)) <= set(params)
        assert not re.findall(r"%(?!\(\w+\)s)", sql), "a bare % breaks psycopg2 formatting"
        assert "Bank of America" not in sql, "query text must be bound, never formatted in"


def test_search_skips_the_database_when_no_channel_applies():
    conn = FakeConnection()
    store = AlloyDBRecallStore(lambda: conn)
    lists = store.search(SCOPE, ["q"], None, None, [MemoryType.MENTAL_MODEL], NOW, False, 10)
    assert lists == {} and queries(conn) == []


def test_search_decodes_rows_into_channel_lists():
    episode = [
        {
            "document_id": "d1",
            "episode_id": "round-7",
            "position": position,
            "user_id": "user_1",
            "project_id": "project_a",
            "session_id": "session_1",
            "speaker": speaker,
            "text": text,
            "occurred_at": "2026-10-02T10:00:00+00:00",
            "recorded_at": "2026-10-02T10:00:01+00:00",
            "expires_at": None,
        }
        for position, speaker, text in [(1, "assistant", "Noted."), (0, "user", "Nightly backups")]
    ]
    observation = {
        "observation_id": "obs-1",
        "user_id": "user_1",
        "project_id": None,
        "session_id": None,
        "visibility": "global",
        "entity_id": "e1",
        "text": "Backups are nightly",
        "fact_ids": [FACT_ID],
        "valid_from": "2026-09-01T00:00:00+00:00",
        "valid_to": None,
    }
    conn = FakeConnection(
        rows=[
            ("keyword:0", "fact", FACT_ID, fact_row()),
            ("keyword:0", "episode", "round-7", episode),
            ("vector:0", "observation", "obs-1", observation),
            ("vector:0", "fact", FACT_ID, fact_row()),
        ]
    )
    store = AlloyDBRecallStore(lambda: conn)
    lists = store.search(SCOPE, ["backups"], [[0.1]], None, list(MemoryType), NOW, False, 10)

    assert [m.id for m in lists["keyword:0"]] == [FACT_ID, "round-7"]
    assert [m.id for m in lists["vector:0"]] == ["obs-1", FACT_ID]
    fact = lists["keyword:0"][0]
    assert fact is lists["vector:0"][1]
    assert fact.occurred_at == datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    assert fact.entity_ids == ("e1",) and fact.evidence_count == 2
    assert lists["keyword:0"][1].text == "user: Nightly backups\nassistant: Noted."
    assert lists["vector:0"][0].supporting_ids == (FACT_ID,)


def test_graph_expansion_only_queries_fact_seeds():
    conn = FakeConnection(rows=[(fact_row(text="Linked"),)])
    store = AlloyDBRecallStore(lambda: conn)

    assert store.expand_neighbors(SCOPE, ["round-1"], list(MemoryType), NOW, False, 5) == []
    assert store.expand_neighbors(SCOPE, [FACT_ID], [MemoryType.EPISODE], NOW, False, 5) == []
    assert queries(conn) == []

    neighbors = store.expand_neighbors(
        SCOPE, [FACT_ID, "round-1"], [MemoryType.FACT], NOW, False, 5
    )
    assert [m.text for m in neighbors] == ["Linked"]
    assert queries(conn)[0][1]["seeds"] == [FACT_ID]


def test_get_episode_and_supporting():
    at = datetime(2026, 10, 2, 10, tzinfo=timezone.utc)
    conn = FakeConnection(rows=[("user", "Nightly backups", at), ("assistant", "Noted.", at)])
    store = AlloyDBRecallStore(lambda: conn)
    episode = store.get_episode(SCOPE, "round-7")
    assert [m["speaker"] for m in episode["messages"]] == ["user", "assistant"]

    conn.rows = []
    assert store.get_episode(SCOPE, "unknown") is None
    assert store.get_supporting(SCOPE, "not-a-uuid") == []

    conn.rows = [(fact_row(),)]
    assert [m.id for m in store.get_supporting(SCOPE, FACT_ID)] == [FACT_ID]


def test_reconnects_once_when_the_connection_drops():
    broken = FakeConnection(fail_with=RuntimeError("server closed the connection"), drop=True)
    healthy = FakeConnection(rows=[])
    connections = iter([broken, healthy])
    store = AlloyDBRecallStore(lambda: next(connections))

    assert store.get_episode(SCOPE, "round-7") is None
    assert len(queries(healthy)) == 1


def test_query_errors_roll_back_and_raise_without_retrying():
    conn = FakeConnection(fail_with=ValueError("syntax error"))
    store = AlloyDBRecallStore(lambda: conn)
    with pytest.raises(ValueError):
        store.get_episode(SCOPE, "round-7")
    assert conn.rollbacks == 1 and not conn.closed


def test_query_grams_cover_multi_word_entity_names():
    grams = _query_grams("What did Bank of America say about Project Falcon?")
    assert {"bank of america", "project falcon", "falcon"} <= set(grams)
    assert "of" not in grams and "of america" not in grams
