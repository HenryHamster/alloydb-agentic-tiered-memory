"""LoCoMo-shaped histories: two named people are ingested as two users, not a user and an agent.

A. devtools ingest pairs named turns and passes both names with ai_is_human
B. end to end, the second speaker's turns are plain text, asserted and kept under their name
C. a fact that gives a speaker's own first-person claim to the other person is rejected
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from test_speaker_names import SCOPE, lean_row, production_extractor, retainer, scripted_client

from devtools.inputs import validate_history
from devtools.worker import conversation_rounds, ingest, is_conversation
from retain.models import Speaker

WHEN = "2023-05-08T13:56:00+00:00"


def turn(name, text, role):
    return {"role": role, "name": name, "content": f"{name}: {text}", "timestamp": WHEN}


SESSION = [
    turn("Caroline", "I adopted a guinea pig named Oscar last month.", "user"),
    turn("Melanie", "I painted an abstract sunset for my art class.", "assistant"),
    turn("Melanie", "We also got a third pet, a dog named Bailey!", "assistant"),
]
HISTORY = {"id": "conv-26", "sessions": [{"id": "session_1", "messages": SESSION}]}


class FakeSystem:
    """record_turn straight into a Retainer, as TieredMemory does without the turn buffer."""

    def __init__(self, stage):
        self.retainer = stage
        self.calls = []

    def record_turn(self, scope, user_text, assistant_text, **kwargs):
        self.calls.append((user_text, assistant_text, kwargs))
        return self.retainer.retain_turn(scope, user_text, assistant_text, **kwargs)

    def finish_session(self, scope):
        return {}


def run_ingest(rows_for, tmp_path):
    client, prompts = scripted_client(rows_for)
    stage = retainer(production_extractor(client))
    system = FakeSystem(stage)
    usage = SimpleNamespace(context=None)
    ingest(system, validate_history(HISTORY), SCOPE.user_id, datetime.now(timezone.utc), usage,
           tmp_path)  # fmt: skip
    return system, stage, prompts


# A. Rounds ------------------------------------------------------------------------------------


def test_named_turns_pair_up_and_lose_their_name_prefix():
    assert is_conversation(SESSION)
    rounds = list(conversation_rounds(SESSION))
    assert rounds == [
        (
            ("Caroline", "I adopted a guinea pig named Oscar last month."),
            ("Melanie", "I painted an abstract sunset for my art class."),
            [0, 1],
        ),
        (("Melanie", "We also got a third pet, a dog named Bailey!"), None, [2]),
    ]


def test_unnamed_or_tool_sessions_keep_the_agent_path():
    unnamed = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert not is_conversation(unnamed)
    assert not is_conversation([*SESSION, {"role": "tool", "name": "x", "content": "{}"}])


def test_blank_message_names_are_rejected():
    bad = {"id": "h", "sessions": [{"id": "s", "messages": [{**SESSION[0], "name": " "}]}]}
    with pytest.raises(ValueError, match="name"):
        validate_history(bad)


# B. End to end --------------------------------------------------------------------------------


def test_second_speaker_is_a_user_whose_turns_are_text_and_kept(tmp_path):
    def rows(chunks):
        if "Caroline" not in chunks:
            return []  # Bailey's round: nothing extracted, so the verbatim fallback keeps it
        return [
            lean_row(chunks["Caroline"]["chunk_id"], 0, "Caroline adopted a guinea pig named Oscar.",
                     "I adopted a guinea pig named Oscar", [("Oscar", "other")]),
            lean_row(chunks["Melanie"]["chunk_id"], 1, "Melanie painted an abstract sunset.",
                     "I painted an abstract sunset"),
        ]  # fmt: skip

    system, stage, prompts = run_ingest(rows, tmp_path)

    assert [kwargs["ai_is_human"] for _, _, kwargs in system.calls] == [True, True]
    assert system.calls[0][1] == "I painted an abstract sunset for my art class."  # not JSON
    assert system.calls[1][2]["ai_name"] is None

    documents = stage.store.documents.values()
    assert {d.speaker for d in documents} == {Speaker.USER}
    assert not any(d.text.startswith(("Caroline:", "Melanie:", "[{")) for d in documents if d.text)

    facts = {f.text: f for f in stage.store.facts.values()}
    assert facts["Melanie painted an abstract sunset."].speaker_name == "Melanie"
    assert facts["Melanie painted an abstract sunset."].claim_type == "asserted"
    bailey = facts["Melanie said: We also got a third pet, a dog named Bailey!"]
    assert bailey.speaker_name == "Melanie"
    assert all('"participants": ["Caroline", "Melanie"]' in p for p in prompts[:1])


# C. Misattribution -----------------------------------------------------------------------------


def test_own_first_person_claim_given_to_the_other_person_is_rejected(tmp_path):
    def rows(chunks):
        if "Caroline" not in chunks:
            return []
        melanie = chunks["Melanie"]["chunk_id"]
        return [
            lean_row(melanie, 0, "Caroline has been trying out abstract painting.",
                     "I painted an abstract sunset"),
            lean_row(melanie, 1, "Melanie is taking an art class.", "my art class"),
        ]  # fmt: skip

    _, stage, _ = run_ingest(rows, tmp_path)

    texts = {f.text for f in stage.store.facts.values()}
    assert "Caroline has been trying out abstract painting." not in texts
    assert "Melanie is taking an art class." in texts


def test_claims_about_the_listener_or_by_the_speaker_are_kept(tmp_path):
    """Melanie's "I love your Oscar" is about Caroline; Caroline's "I adopted" is her own."""

    def rows(chunks):
        if "Caroline" not in chunks:
            return []
        caroline, melanie = chunks["Caroline"]["chunk_id"], chunks["Melanie"]["chunk_id"]
        return [
            lean_row(caroline, 0, "Caroline adopted a guinea pig named Oscar.", "I adopted"),
            lean_row(melanie, 1, "Caroline has a guinea pig named Oscar.", "I love your Oscar"),
        ]

    session = [SESSION[0], {**SESSION[1], "content": "I love your Oscar! In my art class..."}]
    history = {"id": "conv-26", "sessions": [{"id": "s", "messages": session}]}
    client, _ = scripted_client(rows)
    stage = retainer(production_extractor(client))
    usage = SimpleNamespace(context=None)
    ingest(FakeSystem(stage), history, SCOPE.user_id, datetime.now(timezone.utc), usage, tmp_path)

    texts = {f.text for f in stage.store.facts.values()}
    assert "Caroline adopted a guinea pig named Oscar." in texts
    # Melanie's quote says "your", so it may be about Caroline and is not second-guessed.
    assert "Caroline has a guinea pig named Oscar." in texts
