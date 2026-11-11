"""Two-person conversations (LoCoMo-style), relative dates and observation scope."""

import json
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from retain.entities import normalize_name
from retain.extraction import (
    GeminiFactExtractor,
    _is_chit_chat,
    _preceding_contexts,
    _relative_date,
    _resolve_relative_dates,
    _tool_call_facts,
    chunk_documents,
)
from retain.models import (
    Category,
    ClaimType,
    Document,
    Entity,
    EntityResolution,
    EntityType,
    ExtractedFact,
    FactType,
    Message,
    Scope,
    Speaker,
    Visibility,
    WriteBatch,
    is_code_built,
)
from retain.pipeline import Retainer, _build_fact, _split_events
from retain.store import InMemoryRetainStore
from tests.test_retain_observations import ENTITY, builder, memory_fact

SCOPE_OBS = Scope("observation-test-user", "observation-project", "observation-session")

# A Monday, like a LoCoMo session date.
MONDAY = datetime(2023, 5, 22, 13, 0, tzinfo=timezone.utc)
SCOPE = Scope("user-c", "project-c", "session-c")


def document(text, *, speaker=Speaker.ASSISTANT, name=None, position=0, at=MONDAY):
    return Document(
        document_id=f"doc-{position}",
        episode_id="episode-c",
        position=position,
        scope=SCOPE,
        speaker=speaker,
        text=text,
        occurred_at=at,
        recorded_at=at,
        name=name,
    )


def model_row(chunk, text, evidence, **overrides):
    row = {
        "key": 0,
        "chunk_id": chunk.chunk_id,
        "text": text,
        "evidence": evidence,
        "fact_type": "experience",
        "category": "event",
        "claim_type": "proposed",
        "visibility": "session",
        "occurred_at": None,
        "valid_from": None,
        "details": {"who": None, "what": None, "when": None, "where": None, "why": None},
        "entities": [],
        "relations": [],
        "caused_by": [],
        "evidenced_by": [],
    }
    row.update(overrides)
    return row


def extractor_returning(rows, **kwargs):
    client = Mock()
    client.models.generate_content.return_value.text = json.dumps(rows)
    return GeminiFactExtractor(client, **kwargs), client


# --- speakers and per-event documents ---------------------------------------------------------


def test_serialized_replies_become_one_clean_named_message_each():
    events = [
        {"role": "assistant", "content": "Melanie: We adopted Bailey!", "name": "Melanie"},
        {"role": "assistant", "content": "Melanie: She is a gray cat."},
    ]
    message = Message(Speaker.ASSISTANT, json.dumps(events), MONDAY)

    first, second = _split_events(message)

    assert first.text == "Melanie: We adopted Bailey!" and first.name == "Melanie"
    assert second.text == "Melanie: She is a gray cat." and second.name is None


def test_tool_events_stay_json_and_still_give_exact_tool_facts():
    call = {"name": "get_server_status", "arguments": {"server_id": "psql-001"}}
    events = [
        {"role": "assistant", "content": None, "tool_calls": [{"function": call}]},
        {"role": "tool", "content": json.dumps({"status": "running"})},
        {"role": "assistant", "content": "psql-001 is running."},
    ]
    tools, reply = _split_events(Message(Speaker.ASSISTANT, json.dumps(events), MONDAY))

    assert json.loads(tools.text) == events[:2]
    assert reply.text == "psql-001 is running."
    (fact,) = _tool_call_facts(chunk_documents([document(tools.text)]), 0, True)
    assert (
        fact.text == 'The assistant called the get_server_status tool with server_id = "psql-001".'
    )


def test_plain_and_user_messages_are_not_split():
    user = Message(Speaker.USER, json.dumps([{"role": "user", "content": "hi"}]), MONDAY)
    reply = Message(Speaker.ASSISTANT, "Plain text reply.", MONDAY)

    assert _split_events(user) == [user]
    assert _split_events(reply) == [reply]


def test_named_speaker_is_sent_to_the_model_and_recorded_as_a_person():
    (chunk,) = chunk_documents([document("Melanie: I painted a sunset.", name="Melanie")])
    row = model_row(chunk, "Melanie painted a sunset.", "I painted a sunset.")
    extractor, client = extractor_returning([row], keep_unrecorded_user_messages=False)

    (fact,) = extractor.extract([chunk])

    payload = client.models.generate_content.call_args.kwargs["contents"]
    assert '"speaker_name": "Melanie"' in payload
    assert fact.claim_type == ClaimType.ASSERTED and fact.fact_type == FactType.WORLD


def test_an_unnamed_agent_keeps_proposed_claims():
    (chunk,) = chunk_documents([document("You could try Python.")])
    row = model_row(chunk, "The assistant suggested Python.", "You could try Python.")
    extractor, client = extractor_returning([row], keep_unrecorded_user_messages=False)

    (fact,) = extractor.extract([chunk])

    assert '"speaker_name"' not in client.models.generate_content.call_args.kwargs["contents"]
    assert fact.claim_type == ClaimType.PROPOSED and fact.fact_type == FactType.EXPERIENCE


def test_a_named_speakers_preference_is_not_demoted():
    named = document("Melanie: I love pottery.", name="Melanie")
    agent = document("You would love pottery.", position=1)
    preference = ExtractedFact(
        key=0,
        chunk_id="c",
        text="Melanie loves pottery.",
        evidence="I love pottery.",
        fact_type=FactType.WORLD,
        category=Category.PREFERENCE,
        claim_type=ClaimType.ASSERTED,
        visibility=Visibility.SESSION,
    )
    now = MONDAY

    person, _, _ = _build_fact(preference, named, EntityResolution({}), now)
    proposal, _, _ = _build_fact(preference, agent, EntityResolution({}), now)

    assert person.claim_type == ClaimType.ASSERTED
    assert proposal.claim_type == ClaimType.PROPOSED


# --- fallback ------------------------------------------------------------------------------------


def test_uncovered_turns_of_both_people_are_kept_and_chit_chat_is_not():
    docs = [
        document("Caroline: Wow, that's great Mel! So proud of you.", speaker=Speaker.USER,
                 name="Caroline"),
        document("Melanie: The necklace stands for love, faith and strength.", name="Melanie",
                 position=1),
        document("Here is a summary of what I found for you today.", position=2),
        document("please restart it now", speaker=Speaker.USER, position=3),
    ]  # fmt: skip
    extractor, _ = extractor_returning([])

    facts = extractor.extract(chunk_documents(docs))

    assert [fact.text for fact in facts] == [
        "Melanie said: The necklace stands for love, faith and strength.",
        "The user said: please restart it now",
    ]
    assert all(is_code_built(fact.text) for fact in facts)


@pytest.mark.parametrize(
    "text,chit_chat",
    [
        ("Caroline: Wow, that's great Mel! So proud of you.", True),
        ("Melanie: Thanks Caroline, that means a lot!", True),
        ("Caroline: Aww, that is so lovely, Mel!", True),
        ("Melanie: Haha yeah, so true!", True),
        ("Melanie: The necklace stands for love, faith and strength.", False),
        ("Caroline: I was so scared.", False),
        ("Melanie: Bailey is so sweet!", False),
        ("Melanie: We made pots!", False),
        ("yes go ahead", False),
    ],
)
def test_chit_chat_is_only_reactions(text, chit_chat):
    assert _is_chit_chat(text, {"Caroline", "Melanie"}) is chit_chat


def test_named_fallbacks_count_as_code_built():
    assert is_code_built("Melanie said: hello there friend")
    assert not is_code_built("Melanie said she adopted a cat.")


# --- relative dates ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase,kind,value",
    [
        ("last Friday", "day", datetime(2023, 5, 19).date()),
        ("last Fri", "day", datetime(2023, 5, 19).date()),
        ("last Tues", "day", datetime(2023, 5, 16).date()),
        ("this past Monday", "day", datetime(2023, 5, 15).date()),
        ("next Friday", "day", datetime(2023, 5, 26).date()),
        ("last weekend", "day", datetime(2023, 5, 20).date()),
        ("two weekends ago", "day", datetime(2023, 5, 13).date()),
        ("yesterday", "day", datetime(2023, 5, 21).date()),
        ("the day before yesterday", "day", datetime(2023, 5, 20).date()),
        ("this morning", "day", datetime(2023, 5, 22).date()),
        ("3 days ago", "day", datetime(2023, 5, 19).date()),
        ("last week", "week", datetime(2023, 5, 15).date()),
        ("two weeks ago", "week", datetime(2023, 5, 8).date()),
        ("last month", "month", (2023, 4)),
        ("a couple of months ago", "month", (2023, 3)),
        ("last year", "year", 2022),
        ("ten years ago", "year", 2013),
    ],
)
def test_relative_phrases_resolve_from_the_day_they_were_said(phrase, kind, value):
    found = _relative_date(f"We went there {phrase}, it was fun.", MONDAY)

    assert found is not None
    assert found[1:3] == (kind, value)


def test_weekend_said_on_a_sunday_is_the_one_before():
    sunday = datetime(2023, 5, 21, tzinfo=timezone.utc)

    assert _relative_date("last weekend", sunday)[2] == datetime(2023, 5, 13).date()


def resolve(text, evidence, category=Category.EVENT, **fields):
    row = {
        "text": text,
        "evidence": evidence,
        "category": category,
        "fact_type": FactType.WORLD,
        "occurred_at": None,
        "valid_from": None,
        "details": {"who": None, "what": None, "when": None, "where": None, "why": None},
        **fields,
    }
    (chunk,) = chunk_documents([document(evidence)])
    return _resolve_relative_dates(row, chunk)


def test_code_date_overrides_a_model_date_a_week_early_and_rewrites_the_text():
    row = resolve(
        "Melanie went to a pride parade last Friday.",
        "I went to a pride parade last Friday!",
        occurred_at="2023-05-12",
    )

    assert row["occurred_at"] == "2023-05-19T00:00:00+00:00"
    assert row["text"] == "Melanie went to a pride parade on 19 May 2023."


def test_last_week_is_stored_as_the_week_before_not_the_session_day():
    row = resolve("Caroline attended a support group last week.", "I went last week.")

    assert row["occurred_at"] == "2023-05-15T00:00:00+00:00"
    assert row["details"]["when"] == "the week before 22 May 2023"
    assert row["text"] == "Caroline attended a support group the week before 22 May 2023."


def test_last_month_is_written_out_so_it_is_not_counted_back_twice():
    row = resolve("Caroline ran a charity race last month.", "I ran a charity race last month.")

    assert row["occurred_at"] is None
    assert row["details"]["when"] == "April 2023"
    assert row["text"] == "Caroline ran a charity race in April 2023."


def test_years_ago_keeps_the_duration_and_adds_the_year():
    row = resolve(
        "Caroline moved from Sweden ten years ago.",
        "I moved from Sweden ten years ago.",
        category=Category.ATTRIBUTE,
    )

    assert row["details"]["when"] == "2013"
    assert row["text"] == "Caroline moved from Sweden ten years ago (around 2013)."


# --- observations --------------------------------------------------------------------------------


class RecordingBuilder:
    def __init__(self):
        self.skipped = None

    def consolidate(self, reader, skip_entity_ids=frozenset()):
        self.skipped = skip_entity_ids
        return []


def test_the_people_talking_get_no_observation_of_their_own():
    store = InMemoryRetainStore()
    entities = [
        Entity("e-mel", SCOPE.user_id, "Melanie", normalize_name("Melanie"), EntityType.PERSON),
        Entity("e-melo", SCOPE.user_id, "Melody", normalize_name("Melody"), EntityType.PERSON),
        Entity("e-bailey", SCOPE.user_id, "Bailey", normalize_name("Bailey"), EntityType.OTHER),
    ]
    store.write(WriteBatch((), entities=tuple(entities)))
    builder = RecordingBuilder()
    retainer = Retainer(
        store=store,
        extractor=Mock(extract=Mock(return_value=[])),
        entity_resolver=Mock(),
        embedder=Mock(),
        linker=Mock(),
        observation_builder=builder,
    )
    retainer._documents(
        SCOPE, [Message(Speaker.ASSISTANT, "Melanie: hi", MONDAY, "Melanie")], "ep", MONDAY
    )

    retainer.consolidate(SCOPE)

    assert builder.skipped == frozenset({"e-mel"})


def test_skipped_entities_get_no_observation():
    store = InMemoryRetainStore()
    store.write(WriteBatch((), facts=(memory_fact(1), memory_fact(2))))
    stage, client, _ = builder(store)

    assert stage.consolidate(SCOPE_OBS, skip_entity_ids=frozenset({ENTITY})) == []
    client.models.generate_content.assert_not_called()


# --- dates the model wrote itself, quotes that leave the phrase out ----------------------------

SATURDAY = datetime(2023, 7, 15, 13, 51, tzinfo=timezone.utc)
POTTERY = (
    "Melanie: Hey Caroline, it's been super busy here. Last Fri I finally took my kids to a "
    "pottery workshop. We all made our own pots, it was fun and therapeutic!"
)


def resolve_in(source, text, evidence, at, category=Category.EVENT, **fields):
    row = {
        "text": text,
        "evidence": evidence,
        "category": category,
        "fact_type": FactType.WORLD,
        "occurred_at": None,
        "valid_from": None,
        "details": {"who": None, "what": None, "when": None, "where": None, "why": None},
        **fields,
    }
    (chunk,) = chunk_documents([document(source, at=at, name="Melanie")])
    return _resolve_relative_dates(row, chunk)


def test_a_week_early_date_the_model_wrote_into_the_text_is_replaced():
    # The quote leaves out "Last Fri"; the sentence around it has it.
    row = resolve_in(
        POTTERY,
        "Melanie took her kids to a pottery workshop on Friday, 7 July 2023.",
        "took my kids to a pottery workshop",
        SATURDAY,
        occurred_at="2023-07-07",
    )

    assert row["occurred_at"] == "2023-07-14T00:00:00+00:00"
    assert row["text"] == "Melanie took her kids to a pottery workshop on 14 July 2023."


def test_a_model_date_off_the_chunks_only_day_phrase_is_corrected():
    row = resolve_in(
        POTTERY,
        "Melanie and her kids made their own pots at a pottery workshop.",
        "We all made our own pots",
        SATURDAY,
        occurred_at="2023-07-07",
    )

    assert row["occurred_at"] == "2023-07-14T00:00:00+00:00"
    assert row["text"] == (
        "Melanie and her kids made their own pots at a pottery workshop on 14 July 2023."
    )


def test_the_quote_has_the_phrase_and_the_text_a_computed_date():
    row = resolve_in(
        "Caroline: Woohoo Melanie! I passed the adoption agency interviews last Friday!",
        "Caroline passed her adoption agency interviews on Friday, 13 October 2023.",
        "I passed the adoption agency interviews last Friday!",
        datetime(2023, 10, 22, 9, 55, tzinfo=timezone.utc),
    )

    assert row["occurred_at"] == "2023-10-20T00:00:00+00:00"
    assert row["text"] == "Caroline passed her adoption agency interviews on 20 October 2023."


def test_a_date_already_in_the_text_is_not_written_twice():
    row = resolve_in(
        "Melanie: Thanks, Caroline! Yup, we just did it yesterday!",
        "Melanie and her kids walked a trail yesterday on 19 October 2023.",
        "we just did it yesterday!",
        datetime(2023, 10, 20, 18, 55, tzinfo=timezone.utc),
    )

    assert row["text"] == "Melanie and her kids walked a trail on 19 October 2023."


def test_last_week_later_in_the_sentence_dates_the_event_and_its_text():
    row = resolve_in(
        "Melanie: Actually, I just took my fam camping in the mountains last week - it was a "
        "really nice time together!",
        "Melanie took her family camping in the mountains.",
        "I just took my fam camping in the mountains",
        datetime(2023, 6, 27, 10, 37, tzinfo=timezone.utc),
    )

    assert row["occurred_at"] == "2023-06-20T00:00:00+00:00"
    assert row["text"] == (
        "Melanie took her family camping in the mountains the week before 27 June 2023."
    )


def test_an_unrelated_date_far_from_the_phrase_is_left_alone():
    row = resolve_in(
        "Melanie: I signed up for a pottery class yesterday.",
        "Melanie signed up yesterday for a pottery class that runs until 1 December 2023.",
        "I signed up for a pottery class yesterday.",
        datetime(2023, 7, 3, 13, 36, tzinfo=timezone.utc),
    )

    assert row["text"] == (
        "Melanie signed up on 2 July 2023 for a pottery class that runs until 1 December 2023."
    )


# --- questions between people, and the previous turn as context -------------------------------


def test_a_persons_questions_are_dropped_and_their_statements_kept_without_the_name_twice():
    docs = [
        document("Caroline: Wow, what a fun moment! What's the band?", speaker=Speaker.USER,
                 name="Caroline"),
        document("Melanie: How was it? We made pots at the workshop.", name="Melanie",
                 position=1),
        document("Caroline: How have you been lately?", speaker=Speaker.USER, name="Caroline",
                 position=2),
        document("what is the status of db-7?", speaker=Speaker.USER, position=3),
    ]  # fmt: skip
    extractor, _ = extractor_returning([])

    facts = extractor.extract(chunk_documents(docs))

    assert [fact.text for fact in facts] == [
        "Melanie said: We made pots at the workshop.",
        "The user said: what is the status of db-7?",
    ]


def test_the_previous_turn_is_context_for_the_next_round_and_a_retry():
    retainer = Retainer(
        store=Mock(), extractor=Mock(), entity_resolver=Mock(), embedder=Mock(), linker=Mock()
    )
    photo = "Melanie: Having my fam around helps. [shares a photo: a woman and a child on a trail]"
    first = [Message(Speaker.USER, "Caroline: Hi!", MONDAY, "Caroline"),
             Message(Speaker.ASSISTANT, photo, MONDAY, "Melanie")]  # fmt: skip
    second = [Message(Speaker.USER, "Caroline: Is that recent?", MONDAY, "Caroline"),
              Message(Speaker.ASSISTANT, "Melanie: Yup, we just did it yesterday!", MONDAY,
                      "Melanie")]  # fmt: skip

    assert retainer._documents(SCOPE, first, "ep-1", MONDAY)[0].context is None
    docs = retainer._documents(SCOPE, second, "ep-2", MONDAY)
    retry = retainer._documents(SCOPE, second, "ep-2", MONDAY)

    assert [d.context for d in docs] == [photo, None]
    assert [d.context for d in retry] == [photo, None]
    chunks = chunk_documents(docs)
    contexts = _preceding_contexts(chunks, 600)
    assert contexts[chunks[1].chunk_id].startswith(photo)
    assert "trail" in contexts[chunks[0].chunk_id]


def test_the_text_may_word_the_phrase_differently_from_the_quote():
    row = resolve_in(
        POTTERY,
        "Melanie took her kids to a pottery workshop last Friday.",
        "Last Fri I finally took my kids to a pottery workshop.",
        SATURDAY,
    )

    assert row["text"] == "Melanie took her kids to a pottery workshop on 14 July 2023."


def test_a_named_speakers_fact_that_says_the_assistant_gets_their_name():
    doc = document("Melanie: Any more art? [shares a photo: a stained glass window]",
                   name="Melanie")  # fmt: skip
    (chunk,) = chunk_documents([doc])
    rows = [
        {"key": 0, "chunk_id": chunk.chunk_id, "text": text, "evidence": "a stained glass window",
         "fact_type": "experience", "category": "event", "claim_type": "asserted",
         "visibility": "session", "occurred_at": None, "valid_from": None, "when": None,
         "entities": [], "relations": [], "caused_by": [], "evidenced_by": []}
        for text in ("The assistant shared a photo of a stained glass window.",
                     "The assistant's photo shows a stained glass window.")
    ]  # fmt: skip
    rows[1]["key"] = 1
    extractor, _ = extractor_returning(rows, lean=True)

    facts = extractor.extract([chunk])

    assert [fact.text for fact in facts] == [
        "Melanie shared a photo of a stained glass window.",
        "Melanie's photo shows a stained glass window.",
    ]


FRIDAY_9_JUNE = datetime(2023, 6, 9, 19, 55, tzinfo=timezone.utc)


def test_a_far_date_with_no_date_words_in_the_turn_is_dropped():
    # "for 4 years" is a duration; the model once turned it into a 2019 start date.
    row = resolve_in(
        "Caroline: I've known these friends for 4 years, since I moved from my home country.",
        "Caroline has known her supportive friends for 4 years.",
        "I've known these friends for 4 years",
        FRIDAY_9_JUNE,
        category=Category.ATTRIBUTE,
        valid_from="2019-06-09",
    )

    assert row["valid_from"] is None  # storage falls back to the message time
    assert row["text"] == "Caroline has known her supportive friends for 4 years."


@pytest.mark.parametrize(
    "source",
    [
        "Caroline: We went to Sweden in March.",
        "Caroline: I moved here in 2019.",
        "Caroline: We went camping last summer.",
    ],
)
def test_a_far_date_the_turn_gives_words_for_is_kept(source):
    row = resolve_in(source, "Caroline went somewhere.", source.split(": ", 1)[1], FRIDAY_9_JUNE,
                     occurred_at="2019-03-01")  # fmt: skip

    assert row["occurred_at"] == "2019-03-01"


def test_a_near_date_with_no_date_words_is_kept():
    row = resolve_in("Caroline: The workshop was great.", "Caroline enjoyed the workshop.",
                     "The workshop was great.", FRIDAY_9_JUNE, occurred_at="2023-06-02")  # fmt: skip

    assert row["occurred_at"] == "2023-06-02"


def test_a_shared_photo_keeps_the_day_it_was_shared_and_the_text_dates_its_subject():
    row = resolve_in(
        "Melanie: Yeah, Here's one I did last week. It's inspired by the sunsets. "
        "[shares a photo: a photo of a painting of a sunset with a pink sky]",
        "Melanie shared a photo of a painting of a sunset with a pink sky that she did last week.",
        "Here's one I did last week.",
        datetime(2023, 10, 13, 10, 31, tzinfo=timezone.utc),
        occurred_at="2023-10-13T10:31:00+00:00",
    )

    assert row["occurred_at"] == "2023-10-13T10:31:00+00:00"
    assert row["text"] == (
        "Melanie shared a photo of a painting of a sunset with a pink sky that she did the week "
        "before 13 October 2023."
    )
