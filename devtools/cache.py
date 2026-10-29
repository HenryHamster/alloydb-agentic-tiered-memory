"""Exact provider-response caching and usage receipts; no spending limits or rate cards."""

import time

from devtools.files import fingerprint, locked, private_dir, read_json, save_json


class CacheMiss(BaseException):
    """Replay misses must escape production retry/degraded-service fallbacks."""


class Usage:
    def __init__(self, folder):
        self.folder = private_dir(folder)
        self.calls, self.hits, self.context = [], [], {}
        self.write()

    def report(self):
        return {
            "calls": self.calls,
            "cache_hits": self.hits,
            "note": "Reported provider usage only; failed calls may have unreported usage.",
        }

    def write(self):
        save_json(self.folder / "usage.json", self.report())

    def hit(self, key, kind, response):
        self.hits.append(
            {
                "kind": kind,
                "cache_key": key,
                "original_usage": response.get("usage_metadata"),
                **self.context,
            }
        )
        self.write()


class RequestCache:
    def __init__(self, folder, usage, *, mode="cached", provider=None, epoch="1"):
        if mode not in {"cached", "replay", "fresh"}:
            raise ValueError("Unknown cache mode")
        self.folder, self.usage, self.mode = private_dir(folder), usage, mode
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
                self.usage.hit(key, kind, entry["response"])
                return entry["response"]
            if self.mode == "replay":
                raise CacheMiss(f"No recorded {kind} response: {key}. Replay made no paid call.")
            call = {
                "id": len(self.usage.calls) + 1,
                "kind": kind,
                "cache_key": key,
                "status": "started",
                **self.usage.context,
            }
            self.usage.calls.append(call)
            self.usage.write()
            started = time.perf_counter()
            try:
                response = invoke(key)
            except BaseException as error:
                call.update(status="failed", error_type=type(error).__name__)
                raise
            else:
                call.update(
                    status="completed",
                    usage_metadata=response.get("usage_metadata"),
                    model_version=response.get("model_version"),
                )
                save_json(
                    self.usage.folder / "calls" / f"{call['id']:05d}.json",
                    {"request": request, "response": response},
                )
            finally:
                call["seconds"] = time.perf_counter() - started
                self.usage.write()
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
    """The production SDK adapters use this as client.models, unchanged."""

    def __init__(self, generation, embedding, cache, *, max_output_tokens=65536):
        self.generation, self.embedding, self.cache = generation, embedding, cache
        self.maximum = max_output_tokens

    def generate_content(self, **kwargs):
        from google.genai.types import GenerateContentResponse

        config = _config(kwargs.get("config"))
        config["max_output_tokens"] = min(
            config.get("max_output_tokens") or self.maximum, self.maximum
        )
        request = {**kwargs, "config": config}
        raw = self.cache.call(
            "generate",
            request,
            lambda _: self.generation.models.generate_content(**request).model_dump(
                mode="json", exclude={"sdk_http_response"}, warnings=False
            ),
        )
        # The SDK assigns arrays to parsed despite its narrower Pydantic annotation.
        # Validate wire fields, then restore this derived value as the SDK does.
        response = GenerateContentResponse.model_validate(
            {k: v for k, v in raw.items() if k != "parsed"}
        )
        response.parsed = raw.get("parsed")
        return response

    def embed_content(self, **kwargs):
        from google.genai.types import EmbedContentResponse

        request = dict(kwargs)
        if "config" in request:
            request["config"] = _config(request["config"])
        return EmbedContentResponse.model_validate(
            self.cache.call(
                "embedding",
                request,
                lambda _: self.embedding.models.embed_content(**request).model_dump(
                    mode="json", exclude={"sdk_http_response"}
                ),
            )
        )


class CachedConnection:
    """Cache ai.rank responses without changing the production SQL adapter."""

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
        request = {"model": params[0], "query": params[1], "documents": list(params[2])}

        def invoke(key):
            self.inner.execute(sql, params)
            return {"rows": [[int(i), float(s)] for i, s in self.inner.fetchall()]}

        result = self.cache.call("ranking", request, invoke)
        self.rows = [tuple(row) for row in result["rows"]]

    def fetchall(self):
        return self.rows if self.rows is not None else self.inner.fetchall()
