"""Offline generation contracts. The SDK clients below never open connections."""

import json
from types import SimpleNamespace as Response
from unittest.mock import Mock

import pytest

from shared.llm import (
    DEFAULT_MODEL,
    RETRY_ATTEMPTS,
    GeminiGenerator,
    GeminiStructuredGenerator,
    IncompleteResponse,
    InvalidResponse,
    ProviderError,
    RefusedResponse,
    generate_with_retry,
)
from shared.providers import AnthropicGenerator, OpenAIGenerator, provider_from_env

OBJECT = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}
ARRAY = {"type": "array", "items": OBJECT}


def test_gemini_keeps_the_existing_text_and_structured_request_shapes():
    client = Mock()
    call = client.models.generate_content
    call.side_effect = [
        Response(text="answer"),
        Response(text='[{"text":"fact"}]'),
        Response(text='{"text":"macro"}'),
    ]
    provider = GeminiGenerator(client, model="gemini-example")

    assert provider("question") == "answer"
    assert call.call_args.kwargs == {"model": "gemini-example", "contents": "question"}
    assert provider.generate_json("extract", ARRAY, thinking_level="low") == [{"text": "fact"}]
    assert call.call_args.kwargs == {
        "model": "gemini-example",
        "contents": "extract",
        "config": {
            "response_mime_type": "application/json",
            "response_json_schema": ARRAY,
            "thinking_config": {"thinking_level": "low"},
        },
    }
    assert GeminiStructuredGenerator(client, model=provider.model)("write", OBJECT) == {
        "text": "macro"
    }
    assert "thinking_config" not in call.call_args.kwargs["config"]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (Response(text="[]", prompt_feedback=Response(block_reason="SAFETY")), RefusedResponse),
        (
            Response(text=None, prompt_feedback=Response(block_reason="MODEL_ARMOR")),
            RefusedResponse,
        ),
        (Response(text="not JSON"), InvalidResponse),
        (Response(text="{}"), InvalidResponse),
    ],
)
def test_gemini_distinguishes_blocks_and_invalid_output(response, error):
    client = Mock()
    call = client.models.generate_content
    call.return_value = response
    provider = GeminiGenerator(client)
    with pytest.raises(error):
        generate_with_retry(lambda prompt: provider.generate_json(prompt, ARRAY), "extract")
    call.assert_called_once()


def test_gemini_passes_capped_text_and_valid_json_to_existing_callers():
    client = Mock()
    candidates = [Response(finish_reason="MAX_TOKENS")]
    client.models.generate_content.side_effect = [
        Response(text="partial answer", candidates=candidates),
        Response(text=" ", candidates=candidates),
        Response(text=None, candidates=candidates),
        Response(text="[]", candidates=candidates),
    ]
    provider = GeminiGenerator(client)
    assert provider("question") == "partial answer"
    assert provider.generate_text("question") == " "
    assert provider.generate_text("question") is None
    assert provider.generate_json("extract", ARRAY) == []


def claude_reply(text, stop_reason="end_turn"):
    return Response(
        stop_reason=stop_reason,
        content=[Response(type="thinking", text="internal"), Response(type="text", text=text)],
    )


def openai_reply(text, status="completed", *, refusal=False, reason=None):
    return Response(
        status=status,
        output=[Response(type="message", content=[Response(type="refusal")] if refusal else [])],
        output_text=text,
        incomplete_details=Response(reason=reason),
    )


def test_anthropic_messages_text_and_json():
    client = Mock()
    call = client.messages.create
    call.side_effect = [claude_reply("hello"), claude_reply('[{"text":"fact"}]')]
    provider = AnthropicGenerator(client, model="claude-example")

    assert provider.generate_text("question") == "hello"
    assert call.call_args.kwargs["messages"] == [{"role": "user", "content": "question"}]
    assert "output_config" not in call.call_args.kwargs
    assert provider.generate_json("extract", ARRAY, thinking_level="low") == [{"text": "fact"}]
    schema = call.call_args.kwargs["output_config"]["format"]["schema"]
    assert schema["type"] == "array"
    assert schema["items"]["additionalProperties"] is False
    assert "additionalProperties" not in OBJECT


def test_claude_retain_normalizes_known_enums_and_preserves_source_text():
    from test_retain_extraction import chunks, fact_row

    from retain.extraction import GeminiFactExtractor, _schema
    from retain.models import Category, ClaimType, EntityRole, EntityType, FactType, Visibility

    source = "Ticket AbC-42 requires PyThOn."
    row = fact_row(
        text=source,
        evidence=source,
        category="Preference",
        fact_type="WoRlD",
        claim_type="Asserted",
        visibility="Project",
        entities=[{"name": "PyThOn", "type": "Technology", "role": "OBJECT"}],
    )
    rows = [
        row,
        {**row, "category": "UnknownCategory"},
        {**row, "evidence": source.upper()},
    ]
    client = Mock()
    client.messages.create.return_value = claude_reply(json.dumps(rows))
    provider = AnthropicGenerator(client, model="claude-example")

    decoded = provider.generate_json("extract", _schema())
    assert decoded[1]["category"] == "UnknownCategory"
    assert decoded[2]["evidence"] == source.upper()
    extractor = GeminiFactExtractor(None, provider=provider, keep_unrecorded_user_messages=False)
    (fact,) = extractor.extract(chunks(source))

    assert fact.text == fact.evidence == source
    assert fact.category is Category.PREFERENCE
    assert fact.fact_type is FactType.WORLD
    assert fact.claim_type is ClaimType.ASSERTED
    assert fact.visibility is Visibility.PROJECT
    (entity,) = fact.entities
    assert entity.name == "PyThOn"
    assert entity.entity_type is EntityType.TECHNOLOGY
    assert entity.role is EntityRole.OBJECT


def test_claude_enum_normalization_preserves_supersession_ids():
    from retain.linking import _decision_schema

    client = Mock()
    client.messages.create.return_value = claude_reply(
        '[{"id":"N0","action":"Duplicate","target_ids":["S1"]}]'
    )
    provider = AnthropicGenerator(client, model="claude-example")

    assert provider.generate_json("classify", _decision_schema()) == [
        {"id": "N0", "action": "duplicate", "target_ids": ["S1"]}
    ]


@pytest.mark.parametrize("always_truncated", [False, True])
def test_claude_dense_retention_has_room_for_all_facts_and_rejects_truncation(always_truncated):
    from test_retain_extraction import STAMP, fact_row

    from retain.extraction import GeminiFactExtractor
    from retain.models import Message, Scope, Speaker
    from retain.pipeline import RetainConfig, Retainer
    from retain.store import InMemoryRetainStore
    from tests.extraction_requests import sent_chunks

    def respond(*, messages, max_tokens, **_):
        rows = [
            fact_row(chunk["chunk"], text=chunk["text"], evidence=chunk["text"], entities=[])
            for chunk in sent_chunks(messages[0]["content"])
        ]
        assert len(rows) == len(notes)
        # Even a syntactically complete prefix is unsafe if Claude reports truncation.
        if always_truncated or max_tokens < 10000:
            return claude_reply(json.dumps(rows[:3]), "max_tokens")
        return claude_reply(json.dumps(rows))

    notes = [f"Project {index:03} requires Python for embedded tasks." for index in range(180)]
    messages = [Message(Speaker.USER, note, STAMP) for note in notes]
    client = Mock()
    call = client.messages.create
    call.side_effect = respond
    store = InMemoryRetainStore()
    retainer = Retainer(
        store=store,
        extractor=GeminiFactExtractor(
            None,
            provider=AnthropicGenerator(client, model="claude-example"),
        ),
        entity_resolver=None,
        embedder=lambda texts: [(1.0,)] * len(texts),
        linker=Mock(link=Mock(return_value=[])),
        config=RetainConfig(maintenance_every=0),
        clock=lambda: STAMP,
    )
    scope = Scope("user", session_id="session")
    if always_truncated:
        with pytest.raises(IncompleteResponse, match="output token limit"):
            retainer.retain(scope, messages, episode_id="dense")
        assert not store.facts and not store.documents and not store.retained
    else:
        result = retainer.retain(scope, messages, episode_id="dense")
        assert {fact.evidence for fact in result.batch.facts} == set(notes)
        assert len(store.facts) == len(notes)
    call.assert_called_once()


def test_openai_responses_restore_array_root_and_omitted_defaults():
    schema = {
        "type": "array",
        "items": {
            **OBJECT,
            "properties": {
                **OBJECT["properties"],
                "updates": {"type": "boolean"},
                "occurred_at": {"type": "string"},
                "category": {"type": "string", "enum": ["world"]},
            },
        },
    }
    client = Mock()
    call = client.responses.create
    call.side_effect = [
        openai_reply("hello"),
        openai_reply(
            '{"items":[{"text":"fact","updates":false,"occurred_at":null,"category":null}]}'
        ),
    ]
    provider = OpenAIGenerator(client, model="gpt-example")
    assert provider.generate_text("question") == "hello"
    assert call.call_args.kwargs == {"model": "gpt-example", "input": "question", "store": False}
    assert provider.generate_json("extract", schema, thinking_level="low") == [
        {"text": "fact", "updates": False}
    ]
    output = call.call_args.kwargs["text"]["format"]
    assert output["strict"] is True
    assert output["schema"]["required"] == ["items"]
    item = output["schema"]["properties"]["items"]["items"]
    assert item["required"] == ["text", "updates", "occurred_at", "category"]
    assert item["properties"]["category"]["anyOf"] == [
        {"type": "string", "enum": ["world"]},
        {"type": "null"},
    ]
    assert schema["items"]["required"] == ["text"]


def test_openai_retain_uses_the_wire_schema_and_validates_evidence_and_causal_indexes():
    from test_retain_extraction import chunks, fact_row

    from retain.extraction import GeminiFactExtractor, _schema
    from retain.models import ClaimType, FactType, Visibility

    schema = _schema()
    fact_schema = schema["items"]
    optional = set(fact_schema["properties"]) - set(fact_schema["required"])
    first = {**fact_row(), **dict.fromkeys(optional)}
    last = {
        **first,
        "text": "The project starts yesterday.",
        "evidence": "The project starts yesterday.",
        "visibility": "project",
        "updates": True,
        "caused_by": [0, 1, 2, 3, 4],
    }
    missing_evidence = {name: value for name, value in first.items() if name != "evidence"}
    rows = [
        first,
        {**first, "evidence": "Unquoted evidence."},
        {**first, "category": None},
        missing_evidence,
        last,
    ]
    client = Mock()
    call = client.responses.create
    call.return_value = openai_reply(json.dumps({"items": rows}))
    selected = chunks()
    extractor = GeminiFactExtractor(
        None,
        provider=OpenAIGenerator(client, model="test"),
        keep_unrecorded_user_messages=False,
        source_dates_known=False,
    )

    preference, event = extractor.extract(selected)
    assert preference.fact_type is FactType.WORLD
    assert preference.claim_type is ClaimType.ASSERTED
    assert preference.visibility is Visibility.GLOBAL
    assert preference.occurred_at is None and preference.valid_from is None
    assert not preference.updates
    assert event.visibility is Visibility.PROJECT
    assert event.updates
    assert event.key == 1 and event.caused_by == (0,)

    gemini = Mock()
    gemini.models.generate_content.return_value = Response(text="[]")
    GeminiFactExtractor(gemini, source_dates_known=False).extract(selected)
    request = call.call_args.kwargs
    assert request["input"] == [
        {"role": "user", "content": gemini.models.generate_content.call_args.kwargs["contents"]}
    ]
    assert '{"items":[]}' in request["instructions"]
    assert "null" in request["instructions"]
    wire = request["text"]["format"]["schema"]
    assert wire["required"] == ["items"]
    item = wire["properties"]["items"]["items"]
    assert set(item["required"]) == set(fact_schema["properties"])
    for name in fact_schema["required"]:
        assert item["properties"][name] == fact_schema["properties"][name]
    for name in optional:
        assert item["properties"][name]["anyOf"][-1] == {"type": "null"}


@pytest.mark.parametrize("adapter", [AnthropicGenerator, OpenAIGenerator])
def test_adapters_translate_reflect_nullable_without_mutating_schema(adapter):
    client = Mock()
    client.messages.create.return_value = claude_reply('{"text":"fact","merged":null}')
    client.responses.create.return_value = openai_reply('{"text":"fact","merged":null}')
    schema = {
        **OBJECT,
        "properties": {**OBJECT["properties"], "merged": {"type": "string", "nullable": True}},
    }

    assert adapter(client, model="test").generate_json("prompt", schema) == {
        "text": "fact",
        "merged": None,
    }
    if adapter is AnthropicGenerator:
        wire = client.messages.create.call_args.kwargs["output_config"]["format"]["schema"]
        assert wire["required"] == ["text"]
    else:
        wire = client.responses.create.call_args.kwargs["text"]["format"]["schema"]
        assert wire["required"] == ["text", "merged"]
    assert wire["properties"]["merged"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert schema["properties"]["merged"]["nullable"] is True


@pytest.mark.parametrize(
    ("adapter", "endpoint", "response", "error"),
    [
        (AnthropicGenerator, "messages", claude_reply("{}", "refusal"), RefusedResponse),
        (AnthropicGenerator, "messages", claude_reply("{}", "max_tokens"), IncompleteResponse),
        (AnthropicGenerator, "messages", claude_reply("not JSON"), InvalidResponse),
        (OpenAIGenerator, "responses", openai_reply("{}", refusal=True), RefusedResponse),
        (OpenAIGenerator, "responses", openai_reply("{}", "incomplete"), IncompleteResponse),
        (
            OpenAIGenerator,
            "responses",
            openai_reply("{}", "incomplete", reason="content_filter"),
            RefusedResponse,
        ),
        (OpenAIGenerator, "responses", openai_reply("not JSON"), InvalidResponse),
    ],
)
def test_new_provider_output_failures_are_distinct_and_do_not_retry(
    adapter, endpoint, response, error
):
    client = Mock()
    call = getattr(client, endpoint).create
    call.return_value = response
    provider = adapter(client, model="test")
    with pytest.raises(error):
        generate_with_retry(lambda prompt: provider.generate_json(prompt, OBJECT), "prompt")
    call.assert_called_once()


class RequestError(Exception):
    def __init__(self, status_code, *, code=None, body=None, headers=None):
        super().__init__("private API key and prompt")
        self.status_code = status_code
        self.code = code
        self.body = body
        self.response = Response(headers=headers or {})


@pytest.mark.parametrize(
    ("adapter", "endpoint", "failure", "success"),
    [
        (AnthropicGenerator, "messages", RequestError(529), claude_reply("ok")),
        (OpenAIGenerator, "responses", RequestError(429), openai_reply("ok")),
        (OpenAIGenerator, "responses", RequestError(502), openai_reply("ok")),
        (OpenAIGenerator, "responses", TimeoutError(), openai_reply("ok")),
    ],
)
def test_retries_are_owned_by_the_shared_helper(monkeypatch, adapter, endpoint, failure, success):
    client = Mock()
    call = getattr(client, endpoint).create
    call.side_effect = [failure, success]
    sleep = Mock()
    monkeypatch.setattr("shared.llm.time.sleep", sleep)
    assert generate_with_retry(adapter(client, model="test").generate_text, "prompt", 2) == "ok"
    sleep.assert_called_once_with(3)


def test_auth_and_model_errors_are_safe_and_do_not_retry():
    client = Mock()
    call = client.responses.create
    provider = OpenAIGenerator(client, model="test")
    for code, guidance in [(401, "OPENAI_API_KEY"), (400, "LLM_MODEL"), (404, "LLM_MODEL")]:
        call.reset_mock()
        call.side_effect = RequestError(code)
        with pytest.raises(ProviderError, match=guidance) as caught:
            generate_with_retry(provider.generate_text, "prompt")
        assert "private API key" not in str(caught.value)
        assert caught.value.__suppress_context__
        call.assert_called_once()


@pytest.mark.parametrize(
    ("adapter", "endpoint", "failure"),
    [
        (
            OpenAIGenerator,
            "responses",
            RequestError(429, code="credit_balance_exhausted", headers={"Retry-After": "3"}),
        ),
        (
            OpenAIGenerator,
            "responses",
            RequestError(429, body={"error": {"type": "insufficient_quota"}}),
        ),
        (AnthropicGenerator, "messages", RequestError(402)),
    ],
)
def test_permanent_provider_quota_fails_immediately(monkeypatch, adapter, endpoint, failure):
    client = Mock()
    call = getattr(client, endpoint).create
    call.side_effect = failure
    sleep = Mock()
    monkeypatch.setattr("shared.llm.time.sleep", sleep)

    with pytest.raises(ProviderError, match="billing or quota") as caught:
        generate_with_retry(adapter(client, model="test").generate_text, "prompt", RETRY_ATTEMPTS)
    assert "private API key" not in str(caught.value)
    call.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("adapter", "endpoint", "headers", "expected"),
    [
        (OpenAIGenerator, "responses", {"Retry-After": "45"}, 45),
        (OpenAIGenerator, "responses", {"retry-after-ms": "4500"}, 4.5),
        (
            OpenAIGenerator,
            "responses",
            {"Retry-After": "5", "retry-after-ms": "4500"},
            5,
        ),
        (OpenAIGenerator, "responses", {"Retry-After": "Thu, 01 Jan 1970 00:00:10 GMT"}, 10),
        (
            OpenAIGenerator,
            "responses",
            {
                "Retry-After": "invalid",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "5s",
                "x-ratelimit-remaining-tokens": "200000",
                "x-ratelimit-reset-tokens": "2m",
                "x-ratelimit-remaining-project-tokens": "0",
                "x-ratelimit-reset-project-tokens": "7s500ms",
            },
            7.5,
        ),
        (
            AnthropicGenerator,
            "messages",
            {
                "anthropic-ratelimit-input-tokens-remaining": "0",
                "anthropic-ratelimit-input-tokens-reset": "1970-01-01T00:00:12Z",
            },
            12,
        ),
        (
            AnthropicGenerator,
            "messages",
            {
                "retry-after": "5",
                "anthropic-ratelimit-tokens-remaining": "0",
                "anthropic-ratelimit-tokens-reset": "1970-01-01T00:02:00Z",
            },
            5,
        ),
    ],
)
def test_provider_retry_delays(monkeypatch, adapter, endpoint, headers, expected):
    client = Mock()
    call = getattr(client, endpoint).create
    success = openai_reply("ok") if adapter is OpenAIGenerator else claude_reply("ok")
    call.side_effect = [RequestError(429, headers=headers), success]
    sleep = Mock()
    monkeypatch.setattr("shared.llm.time.sleep", sleep)
    monkeypatch.setattr("shared.providers.time", lambda: 0)

    result = generate_with_retry(adapter(client, model="test").generate_text, "prompt", 6)

    assert result == "ok"
    assert call.call_count == 2
    sleep.assert_called_once_with(expected)


@pytest.mark.parametrize(("delays", "waits"), [([45], []), ([15, 10], [15])])
def test_provider_retry_delays_cannot_exceed_the_total_wait_budget(monkeypatch, delays, waits):
    client = Mock()
    call = client.responses.create
    call.side_effect = [RequestError(429, headers={"Retry-After": str(delay)}) for delay in delays]
    sleep = Mock()
    monkeypatch.setattr("shared.llm.time.sleep", sleep)
    with pytest.raises(ProviderError, match="429"):
        generate_with_retry(OpenAIGenerator(client, model="test").generate_text, "prompt")
    assert call.call_count == len(delays)
    assert [entry.args[0] for entry in sleep.call_args_list] == waits


def test_provider_can_mark_a_transient_http_status_as_non_retryable(monkeypatch):
    client = Mock()
    call = client.messages.create
    call.side_effect = RequestError(529, headers={"x-should-retry": "false"})
    sleep = Mock()
    monkeypatch.setattr("shared.llm.time.sleep", sleep)
    with pytest.raises(ProviderError, match="529"):
        generate_with_retry(AnthropicGenerator(client, model="test").generate_text, "prompt")
    call.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"LLM_PROVIDER": "unknown"}, "LLM_PROVIDER"),
        ({"LLM_PROVIDER": "openai"}, "LLM_MODEL"),
        ({"LLM_PROVIDER": "gemini", "LLM_MODEL": "bad id"}, "LLM_MODEL"),
        ({"LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-test"}, "ANTHROPIC_API_KEY"),
        (
            {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-test", "OPENAI_API_KEY": " "},
            "OPENAI_API_KEY",
        ),
    ],
)
def test_invalid_configuration_fails_before_loading_an_sdk(monkeypatch, env, message):
    load = Mock()
    monkeypatch.setattr("shared.providers.import_module", load)
    with pytest.raises(ValueError, match=message):
        provider_from_env("project", env=env)
    load.assert_not_called()


def test_missing_provider_extra_has_an_install_hint(monkeypatch):
    monkeypatch.setattr("shared.providers.import_module", Mock(side_effect=ModuleNotFoundError))
    with pytest.raises(RuntimeError, match="Install the anthropic extra"):
        provider_from_env(
            "project",
            env={"LLM_PROVIDER": "anthropic", "LLM_MODEL": "test", "ANTHROPIC_API_KEY": "dummy"},
        )


def test_gemini_default_and_mcp_legacy_precedence(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("google.genai.Client", factory)
    env = {"GEMINI_MODEL": "legacy"}
    default, _ = provider_from_env("project", env=env)
    assert default.model == DEFAULT_MODEL
    legacy, _ = provider_from_env("project", env=env, use_legacy_gemini_model=True)
    assert legacy.model == "legacy"
    explicit, _ = provider_from_env(
        "project", env={**env, "LLM_MODEL": "explicit"}, use_legacy_gemini_model=True
    )
    assert explicit.model == "explicit"
    factory.assert_called_with(vertexai=True, project="project", location="global")


@pytest.mark.parametrize(
    ("name", "client_class", "generator_class"),
    [("anthropic", "Anthropic", AnthropicGenerator), ("openai", "OpenAI", OpenAIGenerator)],
)
def test_provider_factory_constructs_only_the_selected_client(
    monkeypatch, name, client_class, generator_class
):
    sdk = Response(APIConnectionError=ConnectionError, **{client_class: Mock()})
    load = Mock(return_value=sdk)
    monkeypatch.setattr("shared.providers.import_module", load)
    google = Mock()
    monkeypatch.setattr("google.genai.Client", google)

    provider, client = provider_from_env(
        "project",
        env={
            "LLM_PROVIDER": name.upper(),
            "LLM_MODEL": "chosen-model",
            name.upper() + "_API_KEY": "dummy",
            "GEMINI_MODEL": "legacy",
        },
    )
    load.assert_called_once_with(name)
    create = getattr(sdk, client_class)
    assert client is create.return_value
    create.assert_called_once_with(api_key="dummy", max_retries=0, timeout=90.0)
    assert isinstance(provider, generator_class)
    assert provider.model == "chosen-model"
    google.assert_not_called()


def test_mcp_wires_selected_generation_with_the_existing_google_retrieval(monkeypatch):
    pytest.importorskip("mcp")
    from test_offline import MemoryValkey

    from mcp_server import MemoryService
    from retain import Scope

    monkeypatch.setenv("PROJECT_ID", "gcp-project")
    monkeypatch.setenv("REGION", "us-east1")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    google = Mock()
    monkeypatch.setattr("google.genai.Client", google)
    sdk = Response(APIConnectionError=ConnectionError, OpenAI=Mock())
    monkeypatch.setattr("shared.providers.import_module", Mock(return_value=sdk))
    monkeypatch.setattr("db_clients.get_valkey_client", MemoryValkey)

    service = MemoryService(Scope("reader", session_id="session"))
    system = service.system()
    client = sdk.OpenAI.return_value
    call = client.responses.create
    call.return_value = openai_reply("answer")
    assert system.generate("prompt") == "answer"
    for embedder in (system.retainer.embedder, system.recaller.embedder, system.reflector.embedder):
        assert embedder.client is google.return_value
        assert embedder.model == "text-embedding-005"
    assert system.recaller.reranker.model_id == "semantic-ranker-default-003"
    google.assert_called_once_with(vertexai=True, project="gcp-project", location="us-east1")
    service.close()
    client.close.assert_called_once()
    google.return_value.close.assert_called_once()


def test_memory_system_routes_generation_to_injected_provider():
    from test_offline import MemoryValkey
    from test_retain_extraction import chunks, fact_row

    from recall import InMemoryRecallStore
    from retain import Scope
    from retain.store import InMemoryRetainStore
    from tiered_memory import build_memory_system

    provider = Mock()
    provider.generate_text.side_effect = [
        '{"sub_queries":["query"],"time_window":null,"refers_back":false,"list":false}',
        "answer",
        "summary",
    ]
    provider.generate_json.side_effect = [[fact_row()], [], {"summary": "", "fact_ids": []}]
    embedding = Mock()
    embedding.models.embed_content.return_value = Response(
        embeddings=[Response(values=[1.0] * 768)]
    )
    connect = Mock(side_effect=AssertionError("an offline test must not connect to AlloyDB"))
    system = build_memory_system(
        connect, MemoryValkey(), provider=provider, embedding_client=embedding
    )
    system.recaller.store = InMemoryRecallStore(InMemoryRetainStore())
    scope = Scope("reader", session_id="session")
    assert system.answer(scope, "query", record=False).text == "answer"

    system.buffer.trigger_limit = system.buffer.window_size = 1
    system.buffer.append("session", "first", "answer")
    system.buffer.append("session", "second", "answer")
    assert system.buffer.read("session").summary == "summary"

    assert len(system.retainer.extractor.extract(chunks())) == 1
    assert system.retainer.classifier._generate("supersession") == []
    assert system.retainer.observation_builder._summarize([]) is None
    assert [call.kwargs["thinking_level"] for call in provider.generate_json.call_args_list] == [
        "low",
        "low",
        "low",
    ]
    provider.generate_json.side_effect = None
    provider.generate_json.return_value = {"title": "Topic", "summary": "Macro"}
    assert system.reflector.macro_writer.generate("reflect", OBJECT) == {
        "title": "Topic",
        "summary": "Macro",
    }
    embedding.models.embed_content.assert_called_once_with(
        model="text-embedding-005", contents=["query"]
    )
    connect.assert_not_called()


def test_anthropic_sdk_serializes_the_structured_request_offline():
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    def reply(request):
        body = json.loads(request.content)
        assert body["output_config"]["format"]["schema"]["type"] == "array"
        assert body["output_config"]["format"]["type"] == "json_schema"
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "content": [{"type": "text", "text": '[{"text":"fact"}]'}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    with anthropic.Anthropic(
        api_key="mock",
        base_url="https://example.test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
    ) as client:
        assert AnthropicGenerator(client, model="claude-test").generate_json("extract", ARRAY) == [
            {"text": "fact"}
        ]


def test_anthropic_sdk_monthly_spend_cap_is_not_retried(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")

    def reply(request):
        return httpx2.Response(
            429,
            json={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Monthly spend cap reached",
                    "details": {"error_code": "enforced_spend_limit_reached"},
                },
            },
        )

    send = Mock(side_effect=reply)
    sleep = Mock()
    monkeypatch.setattr("shared.llm.time.sleep", sleep)
    with anthropic.Anthropic(
        api_key="mock",
        base_url="https://example.test",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(send)),
    ) as client:
        provider = AnthropicGenerator(client, model="claude-test")
        with pytest.raises(ProviderError, match="billing or quota"):
            generate_with_retry(provider.generate_text, "prompt", RETRY_ATTEMPTS)

    send.assert_called_once()
    sleep.assert_not_called()


def test_openai_sdk_serializes_the_structured_request_offline():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    def reply(request):
        body = json.loads(request.content)
        assert body["store"] is False
        assert body["input"] == [{"role": "user", "content": "extract"}]
        assert '{"items":[]}' in body["instructions"]
        assert body["text"]["format"]["schema"]["type"] == "object"
        assert body["text"]["format"]["schema"]["properties"]["items"]["type"] == "array"
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"items":[{"text":"fact"}]}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    with openai.OpenAI(
        api_key="mock",
        base_url="https://example.test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(reply)),
    ) as client:
        assert OpenAIGenerator(client, model="gpt-test").generate_json("extract", ARRAY) == [
            {"text": "fact"}
        ]


def test_google_generation_is_closed_if_embedding_initialization_fails(monkeypatch):
    from tiered_memory import build_memory_system_from_env

    monkeypatch.setenv("PROJECT_ID", "gcp-project")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    generation = Mock()
    factory = Mock(side_effect=[generation, RuntimeError("embedding configuration failed")])
    monkeypatch.setattr("google.genai.Client", factory)
    with pytest.raises(RuntimeError, match="embedding configuration failed"):
        build_memory_system_from_env()
    generation.close.assert_called_once()


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (Response(text=" "), "malformed observation JSON"),
        (
            Response(text=None, prompt_feedback=Response(block_reason="SAFETY")),
            "no observation",
        ),
    ],
)
def test_gemini_observation_correction_keeps_the_legacy_prompt(response, reason):
    from retain.observations import GeminiObservationBuilder

    client = Mock()
    call = client.models.generate_content
    call.side_effect = [
        response,
        Response(text='{"summary":"","fact_ids":[]}'),
    ]
    provider = GeminiGenerator(client)
    builder = GeminiObservationBuilder(None, Mock(), Mock(), provider=provider)
    assert builder._summarize([]) is None
    first, second = [args.kwargs["contents"] for args in call.call_args_list]
    assert second == (
        f"The previous response failed validation: Gemini returned {reason}. "
        "Return a corrected response; do not drop any input IDs for a nonempty summary.\n" + first
    )
