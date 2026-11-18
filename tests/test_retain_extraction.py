import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from retain.embeddings import EMBEDDING_DIMENSIONS, GeminiEmbedder
from retain.extraction import GeminiFactExtractor, chunk_documents
from retain.models import (
    ClaimType,
    Document,
    EntityRole,
    EntityType,
    FactType,
    Message,
    Scope,
    Speaker,
    Visibility,
)
from retain.pipeline import Retainer
from retain.store import InMemoryRetainStore
from tests.extraction_requests import sent_chunks

STAMP = datetime(2026, 10, 25, 12, 30, tzinfo=timezone.utc)
SOURCE = "I prefer Python only for embedded tasks. The project starts yesterday."


def chunks(text=SOURCE, *, timestamp=STAMP, max_chars=700):
    document = Document(
        document_id="document-a",
        episode_id="episode-a",
        position=0,
        scope=Scope("user-a", "project-a", "session-a"),
        speaker=Speaker.USER,
        text=text,
        occurred_at=timestamp,
        recorded_at=STAMP,
    )
    return chunk_documents([document], max_chars)


def fact_row(chunk=0, **overrides):
    """A model row citing the request's chunk at index `chunk`."""
    row = {
        "chunk": chunk,
        "text": "Python only for embedded tasks.",
        "evidence": "I prefer Python only for embedded tasks.",
        "fact_type": "world",
        "category": "preference",
        "claim_type": "asserted",
        "visibility": "project",
        "occurred_at": None,
        "valid_from": None,
        "when": None,
        "entities": [{"name": "Python", "entity_type": "technology", "role": "object"}],
        "updates": False,
        "caused_by": [],
    }
    row.update(overrides)
    return row


def gemini_returning(text):
    client = Mock()
    client.models.generate_content.return_value.text = text
    return client


def extractor_for(rows):
    """Row-validation tests see only model rows, without the verbatim user-message fallback."""
    client = gemini_returning(json.dumps(rows))
    return GeminiFactExtractor(client, keep_unrecorded_user_messages=False), client


def test_gemini_extractor_preserves_structured_fact_and_resolves_relative_date():
    selected = chunks()
    preference = fact_row()
    event = fact_row(
        text="The project started yesterday.",
        evidence="The project starts yesterday.",
        fact_type="experience",
        category="event",
        claim_type="hypothetical",
        visibility="session",
        when="yesterday",
        entities=[],
        caused_by=[0],
    )
    extractor, client = extractor_for([preference, event])

    facts = extractor.extract(selected)

    assert len(facts) == 2
    assert facts[0].key == 0 and facts[1].key == 1
    assert facts[0].chunk_id == selected[0].chunk_id
    assert facts[0].evidence == preference["evidence"]
    assert facts[0].entities[0].entity_type == EntityType.TECHNOLOGY
    assert facts[0].entities[0].role == EntityRole.OBJECT
    assert facts[0].visibility == Visibility.PROJECT
    assert facts[0].text == "Python only for embedded tasks."
    assert facts[1].fact_type == FactType.EXPERIENCE
    assert facts[1].occurred_at == datetime(2026, 10, 24, tzinfo=timezone.utc)
    assert facts[1].caused_by == (0,)
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-3.5-flash"
    assert call["config"]["response_mime_type"] == "application/json"
    assert SOURCE in call["contents"]
    assert "[0] user: " in call["contents"]
    assert "document-a" not in call["contents"]


@pytest.mark.parametrize(
    "source,evidence",
    [
        (SOURCE, "A fabricated statement."),
        ("Python is fine.", "Python is fine for embedded tasks."),
    ],
    ids=["not-in-source", "longer-than-source"],
)
def test_extractor_discards_evidence_the_chunk_does_not_contain(source, evidence):
    selected = chunks(source)
    row = fact_row(evidence=evidence)
    extractor, _ = extractor_for([row])

    assert extractor.extract(selected) == []


@pytest.mark.parametrize("response", ["not-json", "{}"])
def test_extractor_rejects_malformed_provider_responses(response):
    # The turn stays unretained, so a transient model failure can be retried.
    with pytest.raises(ValueError):
        GeminiFactExtractor(gemini_returning(response)).extract(chunks())


def test_extractor_drops_non_object_rows_and_keeps_causal_indexes():
    selected = chunks()
    first = fact_row()
    second = fact_row(text="A different claim.", caused_by=[1])
    extractor, _ = extractor_for([None, first, second])

    one, two = extractor.extract(selected)

    assert one.text == first["text"] and two.caused_by == (one.key,)


def test_extractor_discards_rows_with_unknown_chunks_or_invalid_fields():
    selected = chunks()
    invalid_chunk = fact_row(5)
    invalid_details = fact_row(when=4)
    valid = fact_row()
    extractor, _ = extractor_for([invalid_chunk, invalid_details, valid])

    facts = extractor.extract(selected)

    assert len(facts) == 1
    assert facts[0].key == 0


def test_relative_dates_come_from_the_quote_and_never_drop_the_fact():
    selected = chunks()
    unsupported = fact_row(
        when="tomorrow",
    )
    conflict = fact_row(
        text="The project started yesterday.",
        evidence="The project starts yesterday.",
        occurred_at="2026-10-26T00:00:00Z",
        category="event",
        when="yesterday",
    )
    extractor, _ = extractor_for([unsupported, conflict])

    kept, dated = extractor.extract(selected)

    # A relative "when" the quote doesn't contain is dropped; the fact stays.
    assert "when" not in kept.details and kept.occurred_at is None
    # The code's date wins over the model's, and the text gets the absolute date.
    assert dated.occurred_at == datetime(2026, 10, 24, tzinfo=timezone.utc)
    assert dated.text == "The project started on 24 October 2026."


@pytest.mark.parametrize("field", ["occurred_at", "valid_from"])
@pytest.mark.parametrize("partial", ["2023", "2024-03"])
@pytest.mark.parametrize("source_dates_known", [True, False])
def test_partial_dates_preserve_facts_without_inventing_exact_dates(
    field, partial, source_dates_known
):
    source = f"Jakarta has 4,237 schools according to {partial} data."
    selected = chunks(source)
    row = fact_row(text=source, evidence=source, entities=[], **{field: partial})
    extractor, _ = extractor_for([row])
    extractor.source_dates_known = source_dates_known

    (fact,) = extractor.extract(selected)

    assert fact.text == source and fact.evidence == source
    assert fact.occurred_at is None and fact.valid_from is None
    assert fact.details["when"] == partial


def test_partial_dates_preserve_existing_when_and_both_date_fields():
    selected = chunks()
    row = fact_row(occurred_at="2023", valid_from="2024-03")
    row["when"] = "during the annual survey"
    extractor, _ = extractor_for([row])

    (fact,) = extractor.extract(selected)

    assert fact.details["when"] == (
        "during the annual survey; occurred_at: 2023; valid_from: 2024-03"
    )
    assert fact.occurred_at is None and fact.valid_from is None


def test_partial_date_already_in_when_is_not_duplicated():
    selected = chunks()
    row = fact_row(occurred_at=" 2023 ")
    row["when"] = "2023"
    extractor, _ = extractor_for([row])

    assert extractor.extract(selected)[0].details["when"] == "2023"


@pytest.mark.parametrize("field", ["occurred_at", "valid_from"])
@pytest.mark.parametrize("value", ["0000", "2024-00", "2024-13", "2023-02-29", "not a date"])
def test_invalid_dates_still_reject_the_row(field, value):
    selected = chunks()
    extractor, _ = extractor_for([fact_row(**{field: value})])

    assert extractor.extract(selected) == []


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2024-02-29", datetime(2024, 2, 29, tzinfo=timezone.utc)),
        ("2024-02-29T12:30:00Z", datetime(2024, 2, 29, 12, 30, tzinfo=timezone.utc)),
    ],
)
def test_complete_dates_keep_their_exact_timestamp(value, expected):
    # The source states the date, so it is not mistaken for one the model made up.
    selected = chunks(SOURCE + " That was on 29 February 2024.")
    extractor, _ = extractor_for([fact_row(occurred_at=value)])

    (fact,) = extractor.extract(selected)

    assert fact.occurred_at == expected
    assert "when" not in fact.details


@pytest.mark.parametrize("field", ["occurred_at", "valid_from"])
@pytest.mark.parametrize("partial", ["2026", "2026-10", "2023", "2026-09"])
def test_relative_dates_override_any_model_partial_date(field, partial):
    selected = chunks()
    row = fact_row(
        evidence="The project starts yesterday.",
        category="event" if field == "occurred_at" else "preference",
        **{field: partial},
    )
    row["when"] = "yesterday"
    extractor, _ = extractor_for([row])

    facts = extractor.extract(selected)

    # The code's date replaces a conflicting model date instead of dropping the fact.
    assert len(facts) == 1
    assert getattr(facts[0], field) == datetime(2026, 10, 24, tzinfo=timezone.utc)
    assert facts[0].details["when"] == "yesterday"


@pytest.mark.parametrize("partial", [None, "2023", "2024-03"])
def test_extracted_fact_keeps_pipeline_document_scope_and_provenance(partial):
    scope = Scope("user-a", "project-a", "session-a")
    message = Message(Speaker.USER, SOURCE, STAMP)

    def generate(*, contents, **kwargs):
        # Cite the first chunk the pipeline sent, so its real chunks are exercised.
        assert sent_chunks(contents)[0]["chunk"] == 0
        row = fact_row(entities=[], occurred_at=partial, valid_from=partial)
        return SimpleNamespace(text=json.dumps([row]))

    client = Mock()
    client.models.generate_content.side_effect = generate
    store = InMemoryRetainStore()
    retainer = Retainer(
        store=store,
        extractor=GeminiFactExtractor(client),
        entity_resolver=None,
        embedder=lambda texts: [[0.25] * EMBEDDING_DIMENSIONS for _ in texts],
        linker=Mock(link=Mock(return_value=[])),
        clock=lambda: STAMP,
    )

    result = retainer.retain(scope, [message], episode_id="episode-a")

    (fact,) = result.batch.facts
    document = result.batch.documents[0]
    assert fact.source_document_id == document.document_id
    assert fact.source_episode_id == "episode-a"
    assert fact.scope == scope
    assert fact.speaker == Speaker.USER
    assert fact.recorded_at == STAMP
    assert fact.evidence == "I prefer Python only for embedded tasks."
    assert fact.occurred_at is None
    assert fact.valid_from == STAMP  # Source-time default, never a fabricated January 1.
    if partial is not None:
        assert fact.details["when"] == partial
    assert store.facts[fact.fact_id] == fact


def test_extractor_returns_empty_without_calling_gemini():
    client = Mock()

    assert GeminiFactExtractor(client).extract([]) == []
    client.models.generate_content.assert_not_called()


def test_extractor_retries_retryable_gemini_failure(monkeypatch):
    selected = chunks()
    client = Mock()
    client.models.generate_content.side_effect = [
        RuntimeError("429 RESOURCE_EXHAUSTED"),
        SimpleNamespace(text=json.dumps([fact_row()])),
    ]
    sleep = Mock()
    monkeypatch.setattr("retain.extraction.time.sleep", sleep)

    assert len(GeminiFactExtractor(client).extract(selected)) == 1
    assert client.models.generate_content.call_count == 2
    sleep.assert_called_once_with(3)


def test_embedder_batches_and_preserves_vector_order():
    client = Mock()
    vector_a = [0.25] * EMBEDDING_DIMENSIONS
    vector_b = [0.5] * EMBEDDING_DIMENSIONS
    client.models.embed_content.side_effect = [
        SimpleNamespace(
            embeddings=[SimpleNamespace(values=vector_a), SimpleNamespace(values=vector_b)]
        ),
        SimpleNamespace(embeddings=[SimpleNamespace(values=vector_a)]),
    ]

    vectors = GeminiEmbedder(client, batch_size=2)(["first", "second", "third"])

    assert vectors == [vector_a, vector_b, vector_a]
    assert client.models.embed_content.call_args_list[0].kwargs == {
        "model": "text-embedding-005",
        "contents": ["first", "second"],
    }
    assert client.models.embed_content.call_args_list[1].kwargs["contents"] == ["third"]


@pytest.mark.parametrize(
    "values",
    [
        [1.0],
        [0.0] * EMBEDDING_DIMENSIONS,
        [float("nan")] * EMBEDDING_DIMENSIONS,
    ],
)
def test_embedder_rejects_invalid_dimensions_or_values(values):
    client = Mock()
    client.models.embed_content.return_value.embeddings = [SimpleNamespace(values=values)]

    with pytest.raises(ValueError):
        GeminiEmbedder(client)(["fact"])


@pytest.mark.parametrize("embeddings", [None, []])
def test_embedder_rejects_missing_or_wrong_embedding_count(embeddings):
    client = Mock()
    client.models.embed_content.return_value.embeddings = embeddings

    with pytest.raises(ValueError, match="one embedding"):
        GeminiEmbedder(client)(["fact"])


def test_embedder_empty_input_does_not_call_client():
    client = Mock()

    assert GeminiEmbedder(client)([]) == []
    client.models.embed_content.assert_not_called()


def test_embedder_propagates_non_retryable_provider_failure():
    client = Mock()
    client.models.embed_content.side_effect = RuntimeError("permission denied")

    with pytest.raises(RuntimeError, match="permission denied"):
        GeminiEmbedder(client)(["fact"])

    client.models.embed_content.assert_called_once()


def assistant_chunks(text):
    document = Document(
        document_id="document-b",
        episode_id="episode-a",
        position=1,
        scope=Scope("user-a", "project-a", "session-a"),
        speaker=Speaker.ASSISTANT,
        text=text,
        occurred_at=STAMP,
        recorded_at=STAMP,
    )
    return chunk_documents([document], 700)


def test_tool_calls_become_exact_facts_without_the_model():
    call = {"name": "get_server_status", "arguments": json.dumps({"server_id": "psql-001"})}
    events = [{"role": "assistant", "content": "Checking.", "tool_calls": [{"function": call}]}]
    selected = assistant_chunks(json.dumps(events, ensure_ascii=False))
    extractor, _ = extractor_for([])

    (fact,) = extractor.extract(selected)

    assert (
        fact.text == 'The assistant called the get_server_status tool with server_id = "psql-001".'
    )
    assert fact.evidence in selected[0].text
    assert fact.fact_type == FactType.EXPERIENCE
    assert [(e.name, e.entity_type) for e in fact.entities] == [("psql-001", EntityType.OTHER)]
    # Plain-text replies, as in production, have no tool calls to read.
    assert extractor.extract(assistant_chunks("Checking the server now.")) == []


def test_user_message_is_kept_verbatim_only_when_extraction_recorded_nothing():
    request = "Can you check the status of the server with ID 'psql-001'?"
    selected = chunks(request)
    client = gemini_returning("[]")

    (fact,) = GeminiFactExtractor(client).extract(selected)
    assert fact.text == "The user said: " + request
    assert fact.evidence == request

    row = fact_row(text="The user asked about psql-001.", evidence=request)
    covered = GeminiFactExtractor(gemini_returning(json.dumps([row]))).extract(selected)
    assert [f.text for f in covered] == ["The user asked about psql-001."]


def test_acknowledgements_skip_the_model_and_speakers_are_not_entities():
    client = gemini_returning("[]")
    assert GeminiFactExtractor(client).extract(chunks("Thanks!")) == []
    client.models.generate_content.assert_not_called()

    people = [
        {"name": "the user", "entity_type": "person", "role": "subject"},
        {"name": "Assistant", "entity_type": "person", "role": "mentioned"},
        {"name": "Python", "entity_type": "technology", "role": "object"},
    ]
    extractor, _ = extractor_for([fact_row(entities=people)])
    (fact,) = extractor.extract(chunks())
    assert [e.name for e in fact.entities] == ["Python"]


def test_retain_calls_request_a_thinking_level_and_none_keeps_the_default():
    extractor, client = extractor_for([fact_row()])
    extractor.extract(chunks())
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config["thinking_config"] == {"thinking_level": "medium"}

    client = gemini_returning("[]")
    GeminiFactExtractor(client, thinking_level=None).extract(chunks())
    assert "thinking_config" not in client.models.generate_content.call_args.kwargs["config"]


def test_extraction_asks_for_one_slim_row_and_finds_requoted_or_repeated_evidence():
    selected = chunks("I prefer Python   only for embedded tasks. More embedded tasks soon.")
    respaced = fact_row(when="2026-03")
    repeated = fact_row(evidence="embedded tasks")
    client = gemini_returning(json.dumps([respaced, repeated]))
    extractor = GeminiFactExtractor(client, keep_unrecorded_user_messages=False)

    first, second = extractor.extract(selected)

    # Quotes are kept in the chunk's own wording, even when re-spaced or repeated.
    assert first.evidence == "I prefer Python   only for embedded tasks."
    assert first.details == {"when": "2026-03"}
    assert second.evidence == "embedded tasks"
    config = client.models.generate_content.call_args.kwargs["config"]
    properties = config["response_json_schema"]["items"]["properties"]
    assert "details" not in properties and "when" in properties
    assert "chunk_id" not in properties and "key" not in properties


def test_fields_left_out_take_their_defaults_but_required_ones_reject_the_row():
    defaults = {"fact_type", "claim_type", "visibility", "occurred_at", "caused_by", "updates"}
    slim = {key: value for key, value in fact_row().items() if key not in defaults}
    uncategorized = {key: value for key, value in fact_row().items() if key != "category"}
    extractor, client = extractor_for([slim, uncategorized])

    (fact,) = extractor.extract(chunks())

    assert fact.fact_type == FactType.WORLD and fact.claim_type == ClaimType.ASSERTED
    assert fact.visibility == Visibility.GLOBAL and fact.occurred_at is None
    assert fact.caused_by == () and fact.updates is False
    schema = client.models.generate_content.call_args.kwargs["config"]["response_json_schema"]
    assert defaults.isdisjoint(schema["items"]["required"])
    assert "category" in schema["items"]["required"]


def test_each_session_is_its_own_request_and_states_its_date_once():
    def session(episode, day):
        at = datetime(2026, 10, day, 9, tzinfo=timezone.utc)
        return [
            Document(f"{episode}-{position}", episode, position, Scope("user-a", "project-a",
                     "session-a"), Speaker.USER, text, at, at)
            for position, text in enumerate(["I adopted a cat.", "She is called Mochi."])
        ]  # fmt: skip

    client = gemini_returning("[]")
    extractor = GeminiFactExtractor(client, keep_unrecorded_user_messages=False)

    extractor.extract(chunk_documents(session("episode-a", 1) + session("episode-b", 2)))

    prompts = [call.kwargs["contents"] for call in client.models.generate_content.call_args_list]
    assert len(prompts) == 2
    assert [prompt.count("Date: ") for prompt in prompts] == [1, 1]
    assert [chunk["text"] for chunk in sent_chunks(prompts[1])] == [
        "I adopted a cat.",
        "She is called Mochi.",
    ]


def test_tool_results_become_exact_facts_only_when_enabled():
    result = {"server_id": "psql-002", "message": "Restart initiated.", "at": "2024-06-15T10:30Z"}
    events = [
        {"role": "assistant", "content": "Restarting it now..."},
        {"role": "tool", "content": json.dumps(result)},
    ]
    selected = assistant_chunks(json.dumps(events, ensure_ascii=False))

    (fact,) = GeminiFactExtractor(gemini_returning("[]"), tool_result_facts=True).extract(selected)

    assert fact.text == (
        'A tool returned: server_id = "psql-002"; message = "Restart initiated."; '
        'at = "2024-06-15T10:30Z".'
    )
    assert fact.evidence in selected[0].text
    assert [e.name for e in fact.entities] == ["psql-002"]
    assert GeminiFactExtractor(gemini_returning("[]")).extract(selected) == []


def test_evidence_matches_quotes_of_serialized_multiline_replies():
    reply = 'Here\'s the summary for store ID 157:\n\n- Last week: $54,841.75 total "net" sales'
    selected = assistant_chunks(json.dumps([{"role": "assistant", "content": reply}]))
    quoted = 'store ID 157:\n\n- Last week: $54,841.75 total "net" sales'
    respaced = "store ID 157: - Last week: $54,841.75"
    rows = [
        fact_row(evidence=quoted),
        fact_row(evidence=respaced),
        fact_row(evidence="Last week: $99"),
    ]
    extractor = GeminiFactExtractor(
        gemini_returning(json.dumps(rows)), keep_unrecorded_user_messages=False
    )

    first, second = extractor.extract(selected)

    assert first.evidence == r"store ID 157:\n\n- Last week: $54,841.75 total \"net\" sales"
    assert second.evidence == r"store ID 157:\n\n- Last week: $54,841.75"
    assert all(fact.evidence in selected[0].text for fact in (first, second))


def test_a_quote_of_a_serialized_reply_keeps_the_serialized_wording():
    reply = "Summary:\n- Last week: $54,841.75\n- Previous week: $48,001.50"
    selected = assistant_chunks(json.dumps([{"role": "assistant", "content": reply}]))
    extractor, _ = extractor_for([fact_row(evidence="- Last week: $54,841.75\n")])

    (fact,) = extractor.extract(selected)

    assert fact.evidence == r"- Last week: $54,841.75\n"
    assert fact.evidence in selected[0].text


def round_chunks(user_text, events):
    scope = Scope("user-a", "project-a", "session-a")
    documents = [
        Document(
            document_id=f"document-{position}",
            episode_id="episode-a",
            position=position,
            scope=scope,
            speaker=speaker,
            text=text,
            occurred_at=STAMP,
            recorded_at=STAMP,
        )
        for position, (speaker, text) in enumerate(
            [(Speaker.USER, user_text), (Speaker.ASSISTANT, json.dumps(events))]
        )
    ]
    return chunk_documents(documents, 700)


def test_round_links_join_request_tool_call_result_and_answer():
    call = {"name": "get_server_status", "arguments": json.dumps({"server_id": "psql-001"})}
    events = [
        {"role": "assistant", "content": "Checking.", "tool_calls": [{"function": call}]},
        {"role": "tool", "content": json.dumps({"server_id": "psql-001", "status": "running"})},
        {"role": "assistant", "content": "Server psql-001 is running."},
    ]
    selected = round_chunks("Check the status of server psql-001.", events)
    user, assistant = selected
    rows = [
        fact_row(
            text="The user asked to check server psql-001.",
            evidence="Check the status of server psql-001.",
            category="goal",
        ),
        fact_row(
            1,
            text="Server psql-001 is running.",
            evidence="Server psql-001 is running.",
            fact_type="experience",
            category="attribute",
        ),
    ]
    extractor = GeminiFactExtractor(
        gemini_returning(json.dumps(rows)), tool_result_facts=True, link_rounds=True
    )

    facts = {f.text[:22]: f for f in extractor.extract(selected)}

    request, reply = facts["The user asked to chec"], facts["Server psql-001 is run"]
    call_fact, result = facts["The assistant called t"], facts["A tool returned: serve"]
    assert call_fact.evidenced_by == (request.key,)
    assert result.evidenced_by == (call_fact.key,)
    assert reply.evidenced_by[0] == result.key
    assert request.evidenced_by == ()
    # Without the setting, nothing is added.
    plain = GeminiFactExtractor(gemini_returning(json.dumps(rows)), tool_result_facts=True)
    assert all(not f.evidenced_by for f in plain.extract(selected))


def test_round_links_tie_an_answer_without_tools_to_the_request():
    selected = round_chunks(
        "How can AI job losses be reduced?",
        [{"role": "assistant", "content": "Retraining programs reduce AI job losses."}],
    )
    rows = [
        fact_row(
            text="The user asked how to reduce AI job losses.",
            evidence="How can AI job losses be reduced?",
            category="goal",
        ),
        fact_row(
            1,
            text="Retraining programs reduce AI job losses.",
            evidence="Retraining programs reduce AI job losses.",
            category="attribute",
        ),
    ]
    extractor = GeminiFactExtractor(gemini_returning(json.dumps(rows)), link_rounds=True)

    request, answer = extractor.extract(selected)

    assert answer.evidenced_by == (request.key,)


def test_code_built_facts_name_entities_by_field_and_skip_empty_results():
    result = {
        "status": "success",
        "device_id": "1213",
        "bike_make": "Harley-Davidson",
        "logFile": "/var/logs/sync_log.txt",
        "message": "Your policy HD2020-98367 has been cancelled.",
        "measured_at": "2024-06-15T10:30Z",
    }
    events = [
        {"role": "tool", "content": json.dumps(result)},
        {"role": "tool", "content": json.dumps({"ok": True})},
    ]
    selected = assistant_chunks(json.dumps(events))

    (fact,) = GeminiFactExtractor(gemini_returning("[]"), tool_result_facts=True).extract(selected)

    assert [e.name for e in fact.entities] == [
        "1213",
        "Harley-Davidson",
        "/var/logs/sync_log.txt",
        "HD2020-98367",
    ]
    assert {e.entity_type for e in fact.entities} == {EntityType.OTHER}
