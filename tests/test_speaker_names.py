"""Named speakers: conversations between people (LoCoMo) keep who said what.

A. names reach extraction, facts, entities and packed context; role words are rejected
B. two people are both users, so one person's facts are never linked as the other's evidence
C. observations summarize one speaker's facts, never two people's claims together
"""

import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from test_retain import T0, ExactResolver, NoLinks, embed, stored_fact
from test_retain_observations import AT, PROJECT, SESSION, USER, builder, memory_fact

from recall.config import MemoryType
from recall.models import Memory
from recall.packing import format_line
from recall.store import InMemoryRecallStore
from retain import Retainer, Scope
from retain.extraction import _NAMED_SPEAKERS_NOTE, GeminiFactExtractor, chunk_documents
from retain.linking import GeminiSupersedeClassifier
from retain.models import (
    Document,
    Message,
    Relation,
    Speaker,
    UpdateAction,
    UpdateDecision,
    WriteBatch,
    is_code_built,
    stable_id,
)
from retain.pipeline import _episode_id, _target_problem
from retain.store import InMemoryRetainStore

SCOPE = Scope("locomo-user", "locomo", "session-1")
CAROLINE = "I adopted a guinea pig named Oscar last month."
MELANIE = "That's lovely! I painted an abstract sunset for my art class."


def lean_row(chunk_id, key, text, evidence, entities=()):
    return {
        "key": key,
        "chunk_id": chunk_id,
        "text": text,
        "evidence": evidence,
        "fact_type": "world",
        "category": "event",
        "claim_type": "asserted",
        "visibility": "session",
        "occurred_at": None,
        "valid_from": None,
        "when": None,
        "entities": [{"name": n, "entity_type": t, "role": "subject"} for n, t in entities],
        "relations": [],
        "caused_by": [],
        "evidenced_by": [],
    }


def scripted_client(rows_for):
    """A Gemini fake whose rows can cite the chunk IDs it was sent."""
    client = Mock()
    prompts = []

    def generate(*, model, contents, config):
        prompts.append(contents)
        payload = json.loads(contents.split("SOURCE JSON:\n", 1)[1])
        by_name = {c.get("speaker_name"): c for c in payload["chunks"]}
        return SimpleNamespace(text=json.dumps(rows_for(by_name)))

    client.models.generate_content.side_effect = generate
    return client, prompts


def production_extractor(client):
    return GeminiFactExtractor(
        client, thinking_level="low", lean=True, tool_result_facts=True, link_rounds=True
    )


def retainer(extractor, store=None):
    return Retainer(
        store=store or InMemoryRetainStore(),
        extractor=extractor,
        entity_resolver=ExactResolver(),
        embedder=embed,
        linker=NoLinks(),
        clock=lambda: T0,
    )


def conversation(rows_for, store=None):
    client, prompts = scripted_client(rows_for)
    stage = retainer(production_extractor(client), store)
    result = stage.retain_turn(
        SCOPE,
        CAROLINE,
        MELANIE,
        timestamp=T0,
        episode_id="round-1",
        user_name="Caroline",
        ai_name="Melanie",
        ai_is_human=True,
    )
    return result, prompts


# A. Names end to end -----------------------------------------------------------------------


def test_two_named_people_are_both_users_and_facts_name_them():
    def rows(chunks):
        caroline, melanie = chunks["Caroline"]["chunk_id"], chunks["Melanie"]["chunk_id"]
        return [
            lean_row(
                caroline,
                0,
                "Caroline adopted a guinea pig named Oscar.",
                "I adopted a guinea pig named Oscar",
                [("Caroline", "person"), ("Oscar", "other")],
            ),
            lean_row(
                melanie,
                1,
                "Melanie painted an abstract sunset for her art class.",
                "I painted an abstract sunset",
                [("Melanie", "person")],
            ),
            # The misattribution this change exists to stop: a role word, not a person.
            lean_row(
                melanie,
                2,
                "The user has a guinea pig named Oscar.",
                "That's lovely!",
            ),
        ]

    result, prompts = conversation(rows)
    batch = result.batch

    assert [(d.speaker, d.speaker_name) for d in batch.documents] == [
        (Speaker.USER, "Caroline"),
        (Speaker.USER, "Melanie"),
    ]
    by_text = {fact.text: fact for fact in batch.facts}
    assert set(by_text) == {
        "Caroline adopted a guinea pig named Oscar.",
        "Melanie painted an abstract sunset for her art class.",
    }
    assert by_text["Caroline adopted a guinea pig named Oscar."].speaker_name == "Caroline"
    melanie_fact = by_text["Melanie painted an abstract sunset for her art class."]
    assert melanie_fact.speaker_name == "Melanie"
    assert melanie_fact.claim_type == "asserted"  # a person asserts; only an agent proposes
    # Participants are not entities; the guinea pig still is.
    assert [entity.name.casefold() for entity in batch.entities] == ["oscar"]

    (prompt,) = prompts
    payload = json.loads(prompt.split("SOURCE JSON:\n", 1)[1])
    assert _NAMED_SPEAKERS_NOTE in prompt
    assert payload["participants"] == ["Caroline", "Melanie"]
    melanie_chunk = next(c for c in payload["chunks"] if c.get("speaker_name") == "Melanie")
    assert melanie_chunk["preceding_context"].startswith("Caroline: I adopted")


def test_uncited_named_turn_is_kept_verbatim_under_the_speakers_name():
    def rows(chunks):
        caroline = chunks["Caroline"]["chunk_id"]
        return [lean_row(caroline, 0, "Caroline adopted a guinea pig.", "I adopted a guinea pig")]

    result, _ = conversation(rows)

    kept = next(f for f in result.batch.facts if f.speaker_name == "Melanie")
    assert kept.text == "Melanie said: " + MELANIE
    assert is_code_built(kept.text, kept.speaker_name)
    # Only the fact's own speaker marks it; a model fact quoting someone else is not code-built.
    assert not is_code_built("The doctor said: rest for a week.", "Melanie")
    assert not is_code_built(kept.text)


def test_long_named_turn_keeps_each_uncited_chunk():
    """A turn split across chunks is not lost when the model cites only part of it."""
    long_turn = " ".join(f"Sentence {i} about my pottery class and the kiln." for i in range(40))

    def rows(chunks):  # the long turn may span several requests; only one holds Caroline
        if "Caroline" not in chunks:
            return []
        caroline = chunks["Caroline"]["chunk_id"]
        return [lean_row(caroline, 0, "Caroline adopted a guinea pig.", "I adopted a guinea pig")]

    client, _ = scripted_client(rows)
    stage = retainer(production_extractor(client))
    result = stage.retain_turn(
        SCOPE,
        CAROLINE,
        long_turn,
        timestamp=T0,
        user_name="Caroline",
        ai_name="Melanie",
        ai_is_human=True,
    )

    melanie = [f for f in result.batch.facts if f.speaker_name == "Melanie"]
    chunks = [c for c in chunk_documents(result.batch.documents) if c.document.speaker_name]
    melanie_chunks = [c for c in chunks if c.document.speaker_name == "Melanie"]
    assert len(melanie_chunks) > 1
    assert len(melanie) == len(melanie_chunks)
    assert all(f.text.startswith("Melanie said: Sentence") for f in melanie)


def test_role_words_stay_allowed_for_an_unnamed_agent():
    """A named user talking to an unnamed agent may still mention "the assistant"."""

    def rows(chunks):
        user = chunks["Kate"]["chunk_id"]
        return [lean_row(user, 0, "Kate asked the assistant for a recipe.", "a recipe")]

    client, _ = scripted_client(rows)
    stage = retainer(production_extractor(client))
    result = stage.retain_turn(
        SCOPE, "Can you find me a recipe?", "Here is one.", timestamp=T0, user_name="Kate"
    )

    assert "Kate asked the assistant for a recipe." in {f.text for f in result.batch.facts}
    assert {d.speaker for d in result.batch.documents} == {Speaker.USER, Speaker.ASSISTANT}


def test_unnamed_rounds_keep_their_prompt_and_ids():
    client, prompts = scripted_client(lambda chunks: [])
    document = Document(
        document_id="document-a",
        episode_id="episode-a",
        position=0,
        scope=SCOPE,
        speaker=Speaker.USER,
        text="Check the status of server psql-001.",
        occurred_at=T0,
        recorded_at=T0,
    )
    extractor = production_extractor(client)
    extractor.extract(chunk_documents([document]))

    (prompt,) = prompts
    payload = json.loads(prompt.split("SOURCE JSON:\n", 1)[1])
    assert prompt.startswith(extractor._prompt_prefix())
    assert _NAMED_SPEAKERS_NOTE not in prompt
    assert set(payload) == {"chunks"}
    assert "speaker_name" not in payload["chunks"][0]

    message = Message(Speaker.USER, "hello there friend", T0)
    expected = stable_id(*SCOPE.__dict__.values(), (Speaker.USER, T0.isoformat(), message.text))
    assert _episode_id(SCOPE, [message]) == expected
    assert _episode_id(SCOPE, [replace(message, name="Kate")]) != expected


# B. No links between the two people ---------------------------------------------------------


def test_one_persons_facts_are_never_linked_as_anothers_evidence():
    def rows(chunks):
        caroline, melanie = chunks["Caroline"]["chunk_id"], chunks["Melanie"]["chunk_id"]
        return [
            lean_row(caroline, 0, "Caroline adopted a guinea pig.", "I adopted a guinea pig"),
            lean_row(melanie, 1, "Melanie painted an abstract sunset.", "I painted an abstract"),
        ]

    result, _ = conversation(rows)

    assert not [link for link in result.batch.links if link.relation == Relation.EVIDENCED_BY]


def test_one_persons_fact_never_reinforces_or_supersedes_anothers():
    caroline = stored_fact("Has a guinea pig named Oscar.", speaker_name="Caroline")
    melanie = replace(
        stored_fact("Has a guinea pig named Oliver."), speaker_name="Melanie", fact_id="m"
    )

    assert not GeminiSupersedeClassifier._eligible(melanie, caroline)
    assert GeminiSupersedeClassifier._eligible(melanie, replace(caroline, speaker_name="Melanie"))
    decision = UpdateDecision(melanie.fact_id, UpdateAction.SUPERSEDE, (caroline.fact_id,))
    assert _target_problem(decision, melanie, {caroline.fact_id: caroline}) is not None


# C. Observations per speaker -----------------------------------------------------------------


def test_observations_summarize_one_speakers_facts_about_an_entity():
    store = InMemoryRetainStore()
    facts = [
        replace(memory_fact(i), speaker_name="Caroline" if i < 2 else "Melanie") for i in range(4)
    ]
    store.write(WriteBatch((), facts=tuple(facts)))
    stage, client, _ = builder(store)
    reader = Scope(USER, PROJECT, SESSION)

    written = stage.consolidate(reader)

    assert {o.speaker_name for o in written} == {"Caroline", "Melanie"}
    for observation in written:
        speakers = {f.speaker_name for f in facts if f.fact_id in observation.fact_ids}
        assert speakers == {observation.speaker_name}
    assert written[0].observation_id != written[1].observation_id
    prompts = [call.kwargs["contents"] for call in client.models.generate_content.call_args_list]
    assert any("Every fact was said by Caroline," in prompt for prompt in prompts)
    assert any("Every fact was said by Melanie," in prompt for prompt in prompts)
    # The speaker is not assumed to be the subject (Melanie praising Caroline's painting).
    assert all("Name the person each claim is about" in prompt for prompt in prompts)
    assert not any("Name the speaker" in prompt for prompt in prompts)
    # Each speaker's group finds its own current version, so a rerun writes nothing.
    assert stage.consolidate(reader) == []

    recalled = Memory.from_observation(written[0])
    assert recalled.speaker_name == written[0].speaker_name and AT


# Recall shows the name -------------------------------------------------------------------------


def test_packed_lines_and_episodes_show_the_speakers_name():
    line = format_line(
        Memory(
            id="f1",
            type=MemoryType.FACT,
            text="Melanie painted an abstract sunset.",
            occurred_at=datetime(2023, 5, 8, tzinfo=timezone.utc),
            speaker=Speaker.USER,
            speaker_name="Melanie",
        ),
        stale=False,
    )
    assert line == "[f1] (fact | 2023-05-08 | Melanie) Melanie painted an abstract sunset."

    store = InMemoryRetainStore()
    conversation(lambda chunks: [], store)
    episode = InMemoryRecallStore(store).get_episode(SCOPE, "round-1")
    assert [m["speaker_name"] for m in episode["messages"]] == ["Caroline", "Melanie"]
    documents = [d for d in store.documents.values() if d.episode_id == "round-1"]
    assert Memory.from_episode(documents).text == f"Caroline: {CAROLINE}\nMelanie: {MELANIE}"
