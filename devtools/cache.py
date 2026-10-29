"""Cache provider receipts, with a fresh per-run API allowance and explicit replay misses."""

import math
import threading
import time

from devtools.files import fingerprint, locked, private_dir, read_json, save_json


class BudgetStop(BaseException):
    """Must escape production retry/degraded-service fallbacks."""


class CacheMiss(BaseException):
    """Replay never silently makes a paid call."""


class Budget:
    def __init__(self, folder, limit):
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("API budget must be finite and positive")
        self.folder = private_dir(folder)
        self.limit, self.accounted = limit, 0.0
        self.calls, self.hits, self.context = [], [], {}
        self._lock = threading.RLock()
        if (self.folder / "usage.json").exists():
            previous = read_json(self.folder / "usage.json")
            if previous["limit_usd"] != limit:
                raise ValueError("Existing run allowance differs; never reset it on retry")
            self.accounted = previous["accounted_api_usd"]
            self.calls, self.hits = previous["calls"], previous["cache_hits"]
        self.write()

    def write(self):
        save_json(self.folder / "usage.json", self.report())

    def report(self):
        return {
            "limit_usd": self.limit,
            "accounted_api_usd": self.accounted,
            "calls": self.calls,
            "cache_hits": self.hits,
            "scope": "New API usage estimates/reserves only. Shared infrastructure billed separately.",
            "rates": {
                "generation_input_per_million": 1.5,
                "generation_output_and_thinking_per_million": 9,
                "embedding_per_million_utf8_bytes_reserved": 0.25,
                "ranking_per_100_documents_reserved": 0.001,
                "generation_model": "gemini-3.5-flash",
                "embedding_model": "text-embedding-005",
                "verified_date": "2026-10-28",
            },
        }

    def reserve(self, kind, amount, key):
        with self._lock:
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("Invalid reservation")
            if self.accounted + amount > self.limit + 1e-12:
                raise BudgetStop(
                    f"API cap: ${self.accounted:.4f} + ${amount:.4f} exceeds ${self.limit:.2f}"
                )
            call = {
                "id": len(self.calls) + 1,
                "kind": kind,
                "reserved_usd": amount,
                "accounted_usd": amount,
                "cache_key": key,
                "status": "reserved",
                **self.context,
            }
            self.calls.append(call)
            self.accounted += amount
            self.write()  # Persist before attempting the provider call.
            return call

    def settle(self, call, amount, **receipt):
        with self._lock:
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("Invalid provider cost")
            self.accounted += amount - call["accounted_usd"]
            call.update(accounted_usd=amount, status="settled", **receipt)
            self.write()
            if self.accounted > self.limit + 1e-9:
                raise BudgetStop(
                    "Provider usage exceeded the reserved bound; stopped. Inspect usage.json."
                )

    def hit(self, key, kind, receipt):
        self.hits.append(
            {
                "kind": kind,
                "cache_key": key,
                "original_usage": receipt.get("usage_metadata"),
                **self.context,
            }
        )
        self.write()


class RequestCache:
    def __init__(self, folder, budget, *, mode="cached", provider=None, epoch="1"):
        if mode not in {"cached", "replay", "fresh"}:
            raise ValueError("Unknown cache mode")
        self.folder, self.budget, self.mode = private_dir(folder), budget, mode
        self.provider, self.epoch = provider or {}, epoch

    def call(self, kind, request, invoke):
        key = fingerprint(
            {
                "format": 1,
                "epoch": self.epoch,
                "provider": self.provider,
                "kind": kind,
                "request": request,
            }
        )
        path = self.folder / kind / f"{key}.json"
        with locked(path.with_suffix(".lock")):
            if self.mode != "fresh" and path.exists():
                entry = read_json(path)
                if entry["request"] != request or entry["response_hash"] != fingerprint(
                    entry["response"]
                ):
                    raise ValueError("Cache receipt failed integrity validation")
                self.budget.hit(key, kind, entry["response"])
                return entry["response"]
            if self.mode == "replay":
                raise CacheMiss(f"No recorded {kind} response: {key}. Replay made no paid call.")
            response = invoke(key)
            # Fresh calls don't overwrite a previous experimental receipt.
            if not path.exists():
                save_json(
                    path,
                    {
                        "request": request,
                        "response": response,
                        "response_hash": fingerprint(response),
                    },
                )
            return response


def _config(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return dict(value or {})


class CachedModels:
    """The existing Gemini adapters can use this as client.models, unchanged."""

    def __init__(self, generation, embedding, cache, *, max_output_tokens=65536):
        if not isinstance(max_output_tokens, int) or not 1 <= max_output_tokens <= 65536:
            raise ValueError("max_output_tokens must be between 1 and 65536")
        self.generation, self.embedding, self.cache = generation, embedding, cache
        self.maximum = max_output_tokens

    def generate_content(self, **kwargs):
        from google.genai.types import GenerateContentResponse

        if kwargs.get("model") != "gemini-3.5-flash" or not isinstance(kwargs.get("contents"), str):
            raise ValueError("This rate card supports gemini-3.5-flash text requests only")
        config = _config(kwargs.get("config"))
        config["max_output_tokens"] = min(
            config.get("max_output_tokens") or self.maximum, self.maximum
        )
        if not isinstance(config["max_output_tokens"], int) or config["max_output_tokens"] < 1:
            raise ValueError("Invalid maximum output tokens")
        if config.get("candidate_count", 1) != 1:
            raise ValueError("Budgeted generation supports one candidate per call")
        request = {**kwargs, "config": config}

        def invoke(key):
            bound = len(request["contents"].encode()) + len(str(config).encode()) + 1024
            reserve = bound * 1.5e-6 + config["max_output_tokens"] * 9e-6
            budget = self.cache.budget
            call = budget.reserve("generate", reserve, key)
            started = time.perf_counter()
            try:
                response = self.generation.models.generate_content(**request)
            except BaseException as error:
                status = getattr(error, "code", None)
                if isinstance(status, int) and 400 <= status < 500:
                    budget.settle(call, 0, http_status=status)
                # Unknown provider completion keeps its full reservation across failure.
                raise
            raw = response.model_dump(mode="json", exclude={"sdk_http_response"})
            save_json(
                budget.folder / "calls" / f"{call['id']:05d}.json",
                {"request": request, "response": raw},
            )
            usage = response.usage_metadata
            if usage is None or usage.prompt_token_count is None:
                raise BudgetStop("Missing provider usage; reservation retained")
            tokens = {
                name: getattr(usage, name) or 0
                for name in ("prompt_token_count", "candidates_token_count", "thoughts_token_count")
            }
            actual = (
                tokens["prompt_token_count"] * 1.5e-6
                + (tokens["candidates_token_count"] + tokens["thoughts_token_count"]) * 9e-6
            )
            budget.settle(
                call,
                actual,
                tokens=tokens,
                seconds=time.perf_counter() - started,
                model_version=response.model_version,
            )
            return raw

        return GenerateContentResponse.model_validate(self.cache.call("generate", request, invoke))

    def embed_content(self, **kwargs):
        from google.genai.types import EmbedContentResponse

        if kwargs.get("model") != "text-embedding-005":
            raise ValueError("This rate card supports text-embedding-005 only")
        request = dict(kwargs)
        if "config" in request:
            request["config"] = _config(request["config"])

        def invoke(key):
            self.cache.budget.reserve("embedding", len(str(request).encode()) * 0.25e-6, key)
            response = self.embedding.models.embed_content(**request)
            if not response.embeddings or any(
                len(e.values or []) != 768
                or not all(math.isfinite(x) for x in e.values)
                or not any(e.values)
                for e in response.embeddings
            ):
                raise ValueError("Invalid embedding response")
            return response.model_dump(mode="json", exclude={"sdk_http_response"})

        return EmbedContentResponse.model_validate(self.cache.call("embedding", request, invoke))


class CachedConnection:
    """Intercept each ai.rank SQL attempt, including reconnect retries, before billing."""

    def __init__(self, inner, cache):
        self.inner, self.cache = inner, cache

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def cursor(self, *args, **kwargs):
        return CachedCursor(self.inner.cursor(*args, **kwargs), self.cache)


class CachedCursor:
    def __init__(self, inner, cache):
        self.inner, self.cache, self.rows = inner, cache, None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __enter__(self):
        self.inner.__enter__()
        return self

    def __exit__(self, *args):
        return self.inner.__exit__(*args)

    def execute(self, sql, params=None):
        self.rows = None
        if "ai.rank(" not in str(sql).lower():
            return self.inner.execute(sql, params)
        if params[0] != "semantic-ranker-default-003":
            raise ValueError("This ranking rate card supports semantic-ranker-default-003 only")
        request = {"model": params[0], "query": params[1], "documents": list(params[2])}

        def invoke(key):
            self.cache.budget.reserve("ranking", math.ceil(len(params[2]) / 100) * 0.001, key)
            self.inner.execute(sql, params)
            return {"rows": [[int(i), float(s)] for i, s in self.inner.fetchall()]}

        result = self.cache.call("ranking", request, invoke)
        self.rows = [tuple(row) for row in result["rows"]]

    def fetchall(self):
        return self.rows if self.rows is not None else self.inner.fetchall()
