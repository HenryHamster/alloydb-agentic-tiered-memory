import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from retain.embeddings import EMBEDDING_DIMENSIONS, GeminiEmbedder
from retain.extraction import GeminiFactExtractor, chunk_documents
from retain.models import EntityRole, EntityType, FactType, Message, Scope, Speaker, Visibility
from retain.pipeline import Retainer
from retain.store import InMemoryRetainStore

STAMP = datetime(2026, 10, 25, 12, 30, tzinfo=timezone.utc)
SOURCE = "I prefer Python only for embedded tasks. The project starts yesterday."


def chunks(text=SOURCE, *, timestamp=STAMP, max_chars=700):
    from retain.models import Document

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


def fact_row(chunk, **overrides):
    row = {
        "key": 7,
        "chunk_id": chunk.chunk_id,
        "text": "Python only for embedded tasks.",
        "evidence": "I prefer Python only for embedded tasks.",
        "fact_type": "world",
        "category": "preference",
        "claim_type": "asserted",
        "visibility": "project",
        "occurred_at": None,
        "valid_from": None,
        "details": {
            "who": "user",
            "what": "Python preference",
            "when": None,
            "where": None,
            "why": None,
        },
        "entities": [{"name": "Python", "entity_type": "technology", "role": "object"}],
        "relations": [],
        "caused_by": [],
        "evidenced_by": [],
    }
    row.update(overrides)
    return row


def extractor_for(rows):
    client = Mock()
    client.models.generate_content.return_value.text = json.dumps(rows)
    return GeminiFactExtractor(client), client


def test_gemini_extractor_preserves_structured_fact_and_resolves_relative_date():
    selected = chunks()
    preference = fact_row(selected[0])
    event = fact_row(
        selected[0],
        key=8,
        text="The project started yesterday.",
        evidence="The project starts yesterday.",
        fact_type="experience",
        category="event",
        claim_type="hypothetical",
        visibility="session",
        details={
            "who": "project",
            "what": "start",
            "when": "yesterday",
            "where": None,
            "why": None,
        },
        entities=[],
        caused_by=[7],
    )
    extractor, client = extractor_for([preference, event])

    facts = extractor.extract(selected)

    assert len(facts) == 2
    assert facts[0].key == 0 and facts[1].key == 1
    assert facts[0].chunk_id == selected[0].chunk_id
    assert facts[0].evidence == preference["evidence"]
    assert facts[0].details["what"] == "Python preference"
    assert facts[0].entities[0].entity_type == EntityType.TECHNOLOGY
    assert facts[0].entities[0].role == EntityRole.OBJECT
    assert facts[0].visibility == Visibility.PROJECT
    assert facts[0].text == "Python only for embedded tasks."
    assert len(facts[0].text.split()) < 8
    assert facts[1].fact_type == FactType.EXPERIENCE
    assert facts[1].occurred_at == datetime(2026, 10, 24, tzinfo=timezone.utc)
    assert facts[1].caused_by == (0,)
    call = client.models.generate_content.call_args.kwargs
    assert call["model"] == "gemini-3.5-flash"
    assert call["config"]["response_mime_type"] == "application/json"
    assert SOURCE in call["contents"]
    assert selected[0].chunk_id in call["contents"]
    assert "document-a" not in call["contents"]


@pytest.mark.parametrize(
    "evidence",
    ["A fabricated statement.", "repeated quote"],
)
def test_extractor_discards_unsupported_or_ambiguous_evidence(evidence):
    source = SOURCE if evidence != "repeated quote" else "repeated quote; repeated quote"
    selected = chunks(source)
    row = fact_row(selected[0], evidence=evidence)
    extractor, _ = extractor_for([row])

    assert extractor.extract(selected) == []


@pytest.mark.parametrize("response", ["not-json", "{}"])
def test_extractor_rejects_malformed_provider_responses(response):
    # The turn stays unretained, so a transient model failure can be retried.
    client = Mock()
    client.models.generate_content.return_value.text = response

    with pytest.raises(ValueError):
        GeminiFactExtractor(client).extract(chunks())


def test_extractor_drops_non_object_and_repeated_key_rows():
    selected = chunks()
    first = fact_row(selected[0], key=4)
    repeated = fact_row(selected[0], key=4, text="A different claim.")
    extractor, _ = extractor_for([None, first, repeated])

    (fact,) = extractor.extract(selected)

    assert fact.text == first["text"]


def test_extractor_discards_rows_with_unknown_chunks_or_invalid_fields():
    selected = chunks()
    invalid_chunk = fact_row(selected[0], chunk_id="unknown")
    invalid_details = fact_row(selected[0], key=8, details={"who": 4})
    valid = fact_row(selected[0], key=9)
    extractor, _ = extractor_for([invalid_chunk, invalid_details, valid])

    facts = extractor.extract(selected)

    assert len(facts) == 1
    assert facts[0].key == 0


def test_relative_dates_must_be_in_the_cited_evidence_and_match_the_source_time():
    selected = chunks()
    unsupported = fact_row(
        selected[0],
        details={"who": None, "what": None, "when": "tomorrow", "where": None, "why": None},
    )
    conflict = fact_row(
        selected[0],
        evidence="The project starts yesterday.",
        occurred_at="2026-10-26T00:00:00Z",
        category="event",
        details={"who": None, "what": None, "when": "yesterday", "where": None, "why": None},
    )
    extractor, _ = extractor_for([unsupported, conflict])

    assert extractor.extract(selected) == []


def test_extracted_fact_keeps_pipeline_document_scope_and_provenance():
    scope = Scope("user-a", "project-a", "session-a")
    message = Message(Speaker.USER, SOURCE, STAMP)

    def generate(*, contents, **kwargs):
        payload = json.loads(contents.split("SOURCE JSON:\n", 1)[1])
        return SimpleNamespace(text=json.dumps([fact_row_id(payload["chunks"][0]["chunk_id"])]))

    def fact_row_id(chunk_id):
        return fact_row(chunks()[0], chunk_id=chunk_id, entities=[])

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


def test_extractor_returns_empty_without_calling_gemini():
    client = Mock()

    assert GeminiFactExtractor(client).extract([]) == []
    client.models.generate_content.assert_not_called()


def test_extractor_retries_retryable_gemini_failure(monkeypatch):
    selected = chunks()
    client = Mock()
    client.models.generate_content.side_effect = [
        RuntimeError("429 RESOURCE_EXHAUSTED"),
        SimpleNamespace(text=json.dumps([fact_row(selected[0])])),
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
