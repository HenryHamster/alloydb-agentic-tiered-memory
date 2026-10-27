"""Query analysis: resolve time phrases to a date window and split multi-part questions."""

import json
import logging
import re
from datetime import datetime, timedelta

from recall.models import QueryPlan, TimeWindow, as_aware
from shared.llm import Generate

logger = logging.getLogger(__name__)

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]  # fmt: skip
_LAST_N_RE = re.compile(r"\b(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?\b")
_MONTH_RE = re.compile(r"\b(?:in|during)\s+(" + "|".join(_MONTHS) + r")(?:\s+((?:19|20)\d{2}))?\b")
_YEAR_RE = re.compile(r"\b(?:in|during)\s+((?:19|20)\d{2})\b")


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _month_window(year: int, month: int, tz) -> TimeWindow:
    next_year, next_month = _shift_month(year, month, 1)
    return TimeWindow(
        datetime(year, month, 1, tzinfo=tz), datetime(next_year, next_month, 1, tzinfo=tz)
    )


def _year_window(year: int, tz) -> TimeWindow:
    return TimeWindow(datetime(year, 1, 1, tzinfo=tz), datetime(year + 1, 1, 1, tzinfo=tz))


def _months_ago(now: datetime, months: int) -> datetime:
    year, month = _shift_month(now.year, now.month, -months)
    next_year, next_month = _shift_month(year, month, 1)
    last_day = (datetime(next_year, next_month, 1) - timedelta(days=1)).day
    return now.replace(year=year, month=month, day=min(now.day, last_day))


def parse_time_phrase(text: str, now: datetime) -> TimeWindow | None:
    """Resolves common relative time phrases without an LLM; returns None when none match."""
    now = as_aware(now)
    tz = now.tzinfo
    lowered = text.lower()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = today - timedelta(days=today.weekday())

    match = _LAST_N_RE.search(lowered)
    if match and int(match.group(1)) > 0:
        count, unit = int(match.group(1)), match.group(2)
        if unit == "day":
            start = now - timedelta(days=count)
        elif unit == "week":
            start = now - timedelta(weeks=count)
        else:
            start = _months_ago(now, count * (12 if unit == "year" else 1))
        return TimeWindow(start, now)

    if re.search(r"\byesterday\b", lowered):
        return TimeWindow(today - timedelta(days=1), today)
    if re.search(r"\btoday\b", lowered):
        return TimeWindow(today, today + timedelta(days=1))
    if re.search(r"\b(?:last|previous) week\b", lowered):
        return TimeWindow(monday - timedelta(weeks=1), monday)
    if re.search(r"\bthis week\b", lowered):
        return TimeWindow(monday, monday + timedelta(weeks=1))
    if re.search(r"\b(?:last|previous) month\b", lowered):
        return _month_window(*_shift_month(now.year, now.month, -1), tz)
    if re.search(r"\bthis month\b", lowered):
        return _month_window(now.year, now.month, tz)
    if re.search(r"\b(?:last|previous) year\b", lowered):
        return _year_window(now.year - 1, tz)
    if re.search(r"\bthis year\b", lowered):
        return _year_window(now.year, tz)

    match = _MONTH_RE.search(lowered)
    if match:
        month = _MONTHS.index(match.group(1)) + 1
        year = int(match.group(2)) if match.group(2) else now.year - (month > now.month)
        return _month_window(year, month, tz)
    match = _YEAR_RE.search(lowered)
    if match:
        return _year_window(int(match.group(1)), tz)
    return None


def parse_time_range(text: str | None, now: datetime) -> TimeWindow | None:
    """Accepts "START/END" ISO dates (end date inclusive) or a phrase such as "last month"."""
    if not text or not text.strip():
        return None
    if "/" not in text:
        return parse_time_phrase(text, now)
    start_text, end_text = (part.strip() for part in text.split("/", 1))
    tz = as_aware(now).tzinfo
    start = as_aware(datetime.fromisoformat(start_text), tz)
    end = as_aware(datetime.fromisoformat(end_text), tz)
    if len(end_text) == 10:  # a bare date includes that whole day
        end += timedelta(days=1)
    return TimeWindow(start, end)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return text


def _decomposition_prompt(query: str, query_time: datetime, max_sub_queries: int) -> str:
    return f"""You prepare questions for an AI agent's memory search.

1. If the question asks about more than one distinct thing, split it into at most {max_sub_queries} self-contained search queries; otherwise return it as the only query.
2. If it refers to a time ("last week", "in March", "before the launch"), resolve it to a date window. The current time is {query_time.isoformat()}. The end date is exclusive.
3. Return ONLY JSON: {{"sub_queries": ["..."], "time_window": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null}}

Question: {query}
JSON:"""


def analyze_query(
    query: str,
    query_time: datetime,
    temporal_window: TimeWindow | None = None,
    decompose: bool = True,
    generate: Generate | None = None,
    max_sub_queries: int = 4,
    *,
    strict: bool = False,
) -> QueryPlan:
    """Turns time phrases into a date window and splits multi-part questions.

    Window precedence: explicit argument, then the rule parser, then the LLM. The original
    query is always the first sub-query. decompose=False (used by Reflect) skips the LLM.
    """
    plan = QueryPlan(sub_queries=[query], window=temporal_window)
    if temporal_window:
        plan.window_source = "explicit"
    else:
        plan.window = parse_time_phrase(query, query_time)
        plan.window_source = "rules" if plan.window else None

    if not decompose or generate is None:
        return plan

    try:
        prompt = _decomposition_prompt(query, query_time, max_sub_queries)
        data = json.loads(_strip_fences(generate(prompt)))
        plan.llm_used = True
        seen = {query.lower()}
        for sub_query in [str(q).strip() for q in data.get("sub_queries") or []][:max_sub_queries]:
            if sub_query and sub_query.lower() not in seen:
                seen.add(sub_query.lower())
                plan.sub_queries.append(sub_query)
        window = data.get("time_window")
        has_window = isinstance(window, dict) and window.get("start") and window.get("end")
        if plan.window is None and has_window:
            tz = query_time.tzinfo
            plan.window = TimeWindow(
                as_aware(datetime.fromisoformat(window["start"].strip()), tz),
                as_aware(datetime.fromisoformat(window["end"].strip()), tz),
            )
            plan.window_source = "llm"
    except Exception as e:
        if strict:
            raise
        logger.warning("Query analysis fell back to the original query: %s", e)
    return plan
