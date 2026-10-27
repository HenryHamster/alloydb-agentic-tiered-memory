"""Serial paid-call accounting for the bounded candidate pilot.

Rates verified 2026-10-26; reserves persist across failed attempts. This is an
application guard, not a Google Cloud billing cap.
"""

import contextlib
import hashlib
import json
import math
import threading
import time
from pathlib import Path

INPUT_RATE = 1.5 / 1_000_000
OUTPUT_RATE = 9.0 / 1_000_000
MAX_OUTPUT = 65536


class BudgetStop(BaseException):
    """Never swallowed by a model retry or degraded-service fallback."""


class UsageFailure(BaseException):
    """Missing provider receipt invalidates this metered attempt."""


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Ledger:
    def __init__(self, folder, limit, infrastructure=4.0):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "budget.json"
        self.lock = threading.RLock()
        self.context = {}
        self.answer_ready_perf = None
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if self.state["limit_usd"] != limit:
                raise ValueError("Existing allowance differs; do not silently reset the ledger")
        else:
            self.state = {
                "limit_usd": limit,
                "accounted_usd": infrastructure,
                "infrastructure_reserve_usd": infrastructure,
                "calls": 0,
                "pricing_url": "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing",
                "pricing_date": "2026-10-26",
                "sdk_input_usd_per_million": 1.5,
                "sdk_output_including_thinking_usd_per_million": 9.0,
                "ranking_policy": "Reserve USD 0.001 per 100 documents for every SQL attempt",
                "embedding_policy": "Conservative 0.25 USD per million UTF-8 input bytes",
                "billing_hard_cap": False,
            }
            save(self.path, self.state)

    def event(self, event):
        with (self.folder / "usage.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"time": time.time(), **self.context, **event}, ensure_ascii=False)
                + "\n"
            )

    @contextlib.contextmanager
    def scope(self, **labels):
        previous = self.context
        self.context = {**previous, **labels}
        start = time.perf_counter()
        try:
            yield
        finally:
            self.event({"event": "span", "elapsed_seconds": time.perf_counter() - start})
            self.context = previous

    def reserve(self, kind, amount, **details):
        with self.lock:
            if self.state["accounted_usd"] + amount > self.state["limit_usd"]:
                raise BudgetStop(
                    f"Allowance insufficient before {kind}: {self.state['accounted_usd']:.4f} + {amount:.4f}"
                )
            self.state["accounted_usd"] += amount
            self.state["calls"] += 1
            call_id = self.state["calls"]
            save(self.path, self.state)
            self.event(
                {
                    "id": call_id,
                    "phase": "reserved",
                    "kind": kind,
                    "usd": amount,
                    **details,
                }
            )
            return call_id

    def settle(self, call_id, reserved, actual, **details):
        with self.lock:
            self.state["accounted_usd"] += actual - reserved
            save(self.path, self.state)
            self.event({"id": call_id, "phase": "settled", "usd": actual, **details})


class Models:
    def __init__(self, generation, embedding, ledger):
        self.generation, self.embedding, self.ledger = generation, embedding, ledger

    def generate_content(self, **kwargs):
        if kwargs["model"] != "gemini-3.5-flash":
            raise ValueError("Unexpected baseline model")
        contents = kwargs["contents"]
        assert isinstance(contents, str)
        config = kwargs.get("config", {})
        maximum = (
            config.get("max_output_tokens", MAX_OUTPUT) if isinstance(config, dict) else MAX_OUTPUT
        )
        # Include schema and system configuration plus an overhead allowance. Byte
        # count conservatively bounds text tokens; output includes thinking.
        encoded_config = (
            config.model_dump(mode="json", exclude_none=True)
            if hasattr(config, "model_dump")
            else config
        )
        input_bound = (
            len(contents.encode("utf-8")) + len(json.dumps(encoded_config).encode("utf-8")) + 1024
        )
        reserve = input_bound * INPUT_RATE + maximum * OUTPUT_RATE
        call = self.ledger.reserve(
            "sdk_generate", reserve, model=kwargs["model"], input_chars=len(contents)
        )
        start = time.perf_counter()
        try:
            response = self.generation.models.generate_content(**kwargs)
        except BaseException as exc:
            status = getattr(exc, "code", None)
            if isinstance(status, int) and 400 <= status < 600:
                # Google's published pricing excludes non-200 HTTP responses.
                self.ledger.settle(
                    call,
                    reserve,
                    0.0,
                    http_status=status,
                    nonbillable_rejection=True,
                    error=str(exc),
                    elapsed_seconds=time.perf_counter() - start,
                )
            else:
                # Transport failures/cancellations have uncertain provider completion.
                self.ledger.event(
                    {
                        "id": call,
                        "phase": "failed_reservation_retained",
                        "error": str(exc),
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                )
            raise
        usage = response.usage_metadata
        details = usage.model_dump(mode="json") if usage is not None else None
        if usage is not None and usage.prompt_token_count is not None:
            actual = (
                usage.prompt_token_count * INPUT_RATE
                + ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
                * OUTPUT_RATE
            )
        else:
            raise UsageFailure("Missing generation usage; full reservation retained")
        save(
            self.ledger.folder / "calls" / f"{call:05d}.json",
            {
                "request": {
                    "model": kwargs["model"],
                    "contents": contents,
                    "config": encoded_config,
                },
                "response": response.model_dump(mode="json", exclude={"sdk_http_response"}),
            },
        )
        self.ledger.settle(
            call,
            reserve,
            actual,
            usage_metadata=details,
            model_version=response.model_version,
            response_chars=len(response.text or ""),
            elapsed_seconds=time.perf_counter() - start,
        )
        if self.ledger.context.get("stage") == "answer_generation":
            self.ledger.answer_ready_perf = time.perf_counter()
            self.ledger.event({"event": "answer_ready"})
        return response

    def embed_content(self, **kwargs):
        if kwargs["model"] != "text-embedding-005":
            raise ValueError("Unexpected embedding model")
        reserve = len(str(kwargs["contents"]).encode("utf-8")) * 0.25 / 1_000_000
        call = self.ledger.reserve("sdk_embedding", reserve)
        start = time.perf_counter()
        response = self.embedding.models.embed_content(**kwargs)
        if len(response.embeddings[0].values) != 768 or not any(response.embeddings[0].values):
            raise RuntimeError("Invalid embedding")
        self.ledger.event(
            {
                "id": call,
                "phase": "completed_reservation_retained",
                "dimensions": 768,
                "elapsed_seconds": time.perf_counter() - start,
                "embedding_statistics": [
                    value.statistics.model_dump(mode="json")
                    if getattr(value, "statistics", None) is not None
                    else None
                    for value in response.embeddings
                ],
                "metadata": response.model_dump(
                    mode="json", exclude={"embeddings", "sdk_http_response"}
                )
                if hasattr(response, "model_dump")
                else None,
            }
        )
        return response


class RankCursor:
    def __init__(self, inner, ledger):
        self.inner, self.ledger = inner, ledger

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __enter__(self):
        self.inner.__enter__()
        return self

    def __exit__(self, *args):
        return self.inner.__exit__(*args)

    def execute(self, sql, params=None):
        if "ai.rank(" not in str(sql).lower():
            return self.inner.execute(sql, params)
        amount = math.ceil(len(params[2]) / 100) * 0.001
        call = self.ledger.reserve("sql_rank", amount, documents=len(params[2]), model=params[0])
        start = time.perf_counter()
        try:
            return self.inner.execute(sql, params)
        finally:
            self.ledger.event(
                {
                    "id": call,
                    "phase": "rank_reservation_retained",
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )


class RankConnection:
    def __init__(self, inner, ledger):
        self.inner, self.ledger = inner, ledger

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def cursor(self, *args, **kwargs):
        return RankCursor(self.inner.cursor(*args, **kwargs), self.ledger)
