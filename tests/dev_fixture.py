"""Deterministic SDK-shaped provider fixture; all memory stages below are real code."""

import base64
import json
import pickle
from types import SimpleNamespace

from google.genai import types
from test_offline import MemoryValkey

from recall import InMemoryRecallStore, Recaller
from reflect import Reflector
from reflect.consolidation import LLMConflictJudge
from reflect.macros import LLMMacroWriter
from reflect.store import InMemoryReflectStore
from retain import RetainConfig
from retain.embeddings import GeminiEmbedder
from retain.entities import FuzzyEntityResolver
from retain.extraction import GeminiFactExtractor
from retain.linking import GeminiSupersedeClassifier, StoreLinker
from retain.observations import GeminiObservationBuilder
from retain.pipeline import Retainer
from retain.store import InMemoryRetainStore
from shared.llm import GeminiGenerator, GeminiStructuredGenerator
from tiered_memory import TieredMemory
from valkey_buffer import SessionBuffer


class FixtureModels:
    def __init__(self):
        self.requests = []

    def generate_content(self, **request):
        self.requests.append(request)
        prompt = request["contents"]
        schema = request.get("config", {}).get("response_json_schema", {})
        if prompt.startswith("Extract useful"):
            payload = json.loads(prompt.split("SOURCE JSON:\n", 1)[1])
            chunk = payload["chunks"][0]
            rows = []
            for key, text in enumerate(
                (
                    "The club picked Inception.",
                    "Inception's movie ID is tt1375666.",
                    "The club will watch Inception on Saturday.",
                )
            ):
                rows.append(
                    {
                        "key": key,
                        "chunk_id": chunk["chunk_id"],
                        "text": text,
                        "evidence": chunk["text"],
                        "fact_type": "world",
                        "category": "attribute",
                        "claim_type": "asserted",
                        "visibility": "session",
                        "occurred_at": None,
                        "valid_from": None,
                        "details": {k: None for k in ("who", "what", "when", "where", "why")},
                        "entities": [
                            {"name": "Inception", "entity_type": "other", "role": "subject"}
                        ],
                        "relations": [],
                        "caused_by": [],
                        "evidenced_by": [],
                    }
                )
            text = json.dumps(rows)
        elif "judgments" in schema.get("properties", {}):
            text = '{"judgments": []}'
        elif "title" in schema.get("properties", {}):
            text = '{"title": "Film club", "summary": "The club picked Inception, movie tt1375666, for Saturday."}'
        elif "fact_ids" in schema.get("properties", {}):
            text = '{"summary": "", "fact_ids": []}'
        elif prompt.startswith("You prepare questions"):
            text = '{"sub_queries": [], "time_window": null}'
        elif "one JSON object with keys name and arguments" in prompt:
            text = '{"name": "Get Movie Reviews", "arguments": {"movie_id": 1375666}}'
        else:
            text = "Inception"
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=text)]))],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=80, candidates_token_count=20, thoughts_token_count=10
            ),
            model_version="fixture-not-a-live-model",
        )

    def embed_content(self, **request):
        self.requests.append(request)
        contents = request["contents"]
        if isinstance(contents, str):
            contents = [contents]
        return types.EmbedContentResponse(
            embeddings=[types.ContentEmbedding(values=[0.1] * 768) for _ in contents]
        )


RETAIN_ATTRS = (
    "documents",
    "retained",
    "facts",
    "entities",
    "fact_entities",
    "entity_relations",
    "links",
    "observations",
)


class MemorySchema:
    def __init__(self):
        self.created = False
        self.retain = InMemoryRetainStore()
        self.reflect = InMemoryReflectStore(self.retain)

    def create(self):
        self.created = True

    def snapshot(self):
        # Test double only: production snapshots are typed PostgreSQL JSON rows, never pickle.
        value = (
            [getattr(self.retain, k) for k in RETAIN_ATTRS],
            self.reflect.states,
            self.reflect.macros,
            self.reflect.reviews,
        )
        return {"fixture": base64.b64encode(pickle.dumps(value)).decode()}

    def restore(self, snapshot):
        values, states, macros, reviews = pickle.loads(base64.b64decode(snapshot["fixture"]))
        for key, value in zip(RETAIN_ATTRS, values):
            setattr(self.retain, key, value)
        self.reflect.states, self.reflect.macros, self.reflect.reviews = states, macros, reviews

    def close(self):
        self.created = False


class MemoryBackend:
    def __init__(self, config, cache, models, source, clock, run_id):
        self.cache, self.models, self.clock = cache, models, clock
        self.schemas = []

    def schema(self):
        schema = MemorySchema()
        self.schemas.append(schema)
        return schema

    def system(self, schema):
        client = SimpleNamespace(models=self.models)
        embedder = GeminiEmbedder(client)
        structured = GeminiStructuredGenerator(client)
        generate = GeminiGenerator(client)
        retainer = Retainer(
            store=schema.retain,
            extractor=GeminiFactExtractor(client),
            entity_resolver=FuzzyEntityResolver(schema.retain),
            embedder=embedder,
            linker=StoreLinker(schema.retain),
            classifier=GeminiSupersedeClassifier(client, schema.retain),
            observation_builder=GeminiObservationBuilder(client, schema.retain, embedder),
            config=RetainConfig(maintenance_every=0),
            clock=lambda: self.clock,
        )
        reflector = Reflector(
            store=schema.reflect,
            embedder=embedder,
            judge=LLMConflictJudge(structured),
            macro_writer=LLMMacroWriter(structured),
            clock=lambda: self.clock,
        )

        def ranking(query, texts):
            result = self.cache.call(
                "ranking",
                {"query": query, "documents": list(texts)},
                lambda key: {"scores": [0.8] * len(texts)},
            )
            return result["scores"]

        recaller = Recaller(
            store=InMemoryRecallStore(schema.retain, schema.reflect),
            reflect_store=schema.reflect,
            embedder=embedder,
            generate=generate,
            reranker=ranking,
            clock=lambda: self.clock,
            strict=True,
        )
        return TieredMemory(
            retainer=retainer,
            reflector=reflector,
            recaller=recaller,
            buffer=SessionBuffer(MemoryValkey(), generate, strict=True),
            generate=generate,
        )

    def close(self):
        assert not any(s.created for s in self.schemas)
