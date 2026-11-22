"""Offline checks of the evidence check on hard questions (LME-002 c4a1ceb8, PR 30)."""

import json
import re

import pytest
from recall_fixtures import NOW, SCOPE, Memories

from recall import RecallConfig
from recall.judge import judge_evidence, judge_prompt
from recall.models import Memory
from recall.query import _decomposition_prompt, analyze_query

QUESTION = "How many different types of citrus fruits have I used in my cocktail recipes?"

# Henry's c4a1ceb8 trace, reduced: reranker relevance by fact. Daiquiri, Gimlet, bitters and
# Sangria answer it (lime, orange, lemon); the requests and plans only share the topic.
C4A1CEB8 = {
    "learned a classic Daiquiri cocktail with fresh lime juice": 0.1746,
    "is planning a summer party and wants refreshing cocktail ideas": 0.1557,
    "asked for a cocktail that pairs with Paella": 0.1520,
    "made a Cucumber Gimlet cocktail with lime juice": 0.1500,
    "likes the sound of an Orange You Glad whiskey sour cocktail recipe": 0.1465,
    "asked for cocktail ideas that use lime juice": 0.1429,
    "asked for unusual cocktail mixers": 0.1378,
    "asked for cocktail drinks for a Spanish menu": 0.1216,
    "made orange bitters for cocktail recipes from orange peels": 0.1195,
    "plans to try making a Cava Spritz cocktail for the next dinner party": 0.0972,
    "served Sangria cocktail in a pitcher with slices of orange and lemon": 0.0943,
    "has never used honey syrup in a cocktail": 0.0925,
}
ANSWERS = ("Daiquiri", "Gimlet", "orange bitters", "Sangria")


def scores(query, texts):
    return [next((s for t, s in C4A1CEB8.items() if t in text), 0.0) for text in texts]


def router(analysis, judge_reply=None, calls=None):
    """One Generate for query analysis and the evidence check, told apart by their prompts."""

    def generate(prompt):
        if prompt.startswith("You check which memories"):
            if calls is not None:
                calls.append(prompt)
            if judge_reply is not None:
                return judge_reply(prompt)
            numbered = re.findall(r"^(\d+)\. (.*)$", prompt, re.MULTILINE)
            picked = [int(n) for n, text in numbered if any(a in text for a in ANSWERS)]
            return json.dumps({"evidence": picked})
        return json.dumps({"sub_queries": [], "time_window": None, **analysis})

    return generate


@pytest.fixture
def cocktails():
    memories = Memories()
    facts = {text: memories.fact(f"The user {text}") for text in C4A1CEB8}
    return memories, facts


def facts_of(result):
    return {item.memory.id for item in result.items if item.memory.type == "fact"}


def kept_texts(result):
    return [item.memory.text for item in result.items]


def test_the_check_keeps_the_fact_the_score_floor_dropped(cocktails):
    memories, facts = cocktails
    sangria = facts["served Sangria cocktail in a pitcher with slices of orange and lemon"]
    done = {"refers_back": False, "list": False, "done": True}

    old = memories.recaller(
        reranker=scores, generate=router(done), config=RecallConfig(judge=False)
    )
    assert sangria.fact_id not in [i.memory.id for i in old.recall(SCOPE, QUESTION).items]

    result = memories.recaller(reranker=scores, generate=router(done)).recall(SCOPE, QUESTION)
    texts = kept_texts(result)
    assert sangria.fact_id in [i.memory.id for i in result.items]
    # The four answering facts lead the context, ahead of requests that outscored them.
    assert all(any(a in t for a in ANSWERS) for t in texts[:4])
    assert texts.index(sangria.text) < texts.index(f"The user {list(C4A1CEB8)[2]}")
    stage = next(s for s in result.trace if s.stage == "judge").to_dict()
    assert set(stage["reasons"]) == {"weak", "done"} and len(stage["evidence"]) == 4
    assert stage["error"] is None


def test_easy_questions_skip_the_check_and_keep_the_same_result(cocktails):
    memories, _ = cocktails

    def strong(query, texts):
        return [0.9 if "Daiquiri" in t else 0.2 for t in texts]

    calls = []
    plain = {"refers_back": False, "list": False, "done": False}
    on = memories.recaller(reranker=strong, generate=router(plain, calls=calls))
    off = memories.recaller(
        reranker=strong, generate=router(plain), config=RecallConfig(judge=False)
    )
    question = "Which cocktail did I learn with lime?"
    with_check, without = on.recall(SCOPE, question), off.recall(SCOPE, question)
    assert calls == [] and all(s.stage != "judge" for s in with_check.trace)
    assert [i.memory.id for i in with_check.items] == [i.memory.id for i in without.items]
    assert with_check.context == without.context


@pytest.mark.parametrize("reply", ["not json", '{"other": 1}', '["1"]'])
def test_a_failed_check_fills_by_score_and_loses_nothing(cocktails, reply):
    memories, facts = cocktails
    done = {"refers_back": False, "list": False, "done": True}
    off = memories.recaller(
        reranker=scores, generate=router(done), config=RecallConfig(judge=False)
    )
    before = {i.memory.id for i in off.recall(SCOPE, QUESTION).items}

    broken = router(done, judge_reply=lambda prompt: reply)
    result = memories.recaller(reranker=scores, generate=broken).recall(SCOPE, QUESTION)
    after = {i.memory.id for i in result.items}
    assert before <= after
    # 0.0943 is over 0.35 * the top score 0.1746, so the relative floor still keeps Sangria.
    assert (
        facts["served Sangria cocktail in a pitcher with slices of orange and lemon"].fact_id
        in after
    )
    stage = next(s for s in result.trace if s.stage == "judge").to_dict()
    assert stage["error"] and stage["evidence"] == []


def test_the_check_never_drops_a_kept_fact_or_abstains(cocktails):
    memories, _ = cocktails
    done = {"refers_back": False, "list": False, "done": True}
    off = memories.recaller(
        reranker=scores, generate=router(done), config=RecallConfig(judge=False)
    )
    before = {i.memory.id for i in off.recall(SCOPE, QUESTION).items}

    none = router(done, judge_reply=lambda prompt: '{"evidence": []}')
    result = memories.recaller(reranker=scores, generate=none).recall(SCOPE, QUESTION)
    assert before <= {i.memory.id for i in result.items} and not result.abstained


def test_a_small_candidate_limit_still_keeps_every_kept_fact(cocktails):
    # Source rounds follow the top kept facts, which the check reorders; facts must all stay.
    memories, _ = cocktails
    done = {"refers_back": False, "list": False, "done": True}
    off = memories.recaller(
        reranker=scores, generate=router(done), config=RecallConfig(judge=False)
    )
    before = facts_of(off.recall(SCOPE, QUESTION))
    calls = []
    tiny = RecallConfig(judge_candidates=2)
    result = memories.recaller(
        reranker=scores, generate=router(done, calls=calls), config=tiny
    ).recall(SCOPE, QUESTION)
    assert len(calls) == 1 and before <= facts_of(result)


def test_list_questions_use_the_check_with_a_middling_top_score(cocktails):
    memories, _ = cocktails

    def middling(query, texts):
        return [0.6 if "Daiquiri" in t else 0.1 for t in texts]

    calls = []
    listing = {"refers_back": False, "list": True, "done": False}
    result = memories.recaller(reranker=middling, generate=router(listing, calls=calls)).recall(
        SCOPE, "Which cocktail recipes have I learned?"
    )
    assert result.items[0].relevance == 0.6
    assert len(calls) == 1
    stage = next(s for s in result.trace if s.stage == "judge").to_dict()
    assert stage["reasons"] == ["list"]
    assert len(result.items) <= RecallConfig().judge_list_max_items


def test_strong_list_and_done_questions_skip_the_check(cocktails):
    memories, _ = cocktails

    def strong(query, texts):
        return [0.9 if "Daiquiri" in t else 0.1 for t in texts]

    calls = []
    both = {"refers_back": False, "list": True, "done": True}
    result = memories.recaller(reranker=strong, generate=router(both, calls=calls)).recall(
        SCOPE, "Which cocktail recipes have I made?"
    )
    assert result.items[0].relevance == 0.9
    assert calls == []
    assert not any(s.stage == "judge" for s in result.trace)


def test_nothing_relevant_still_abstains_without_a_check():
    memories = Memories()
    memories.fact("The team stores orders in a Postgres database")
    calls = []
    analysis = {"refers_back": False, "list": True, "done": True}
    recaller = memories.recaller(
        reranker=lambda q, texts: [0.0] * len(texts), generate=router(analysis, calls=calls)
    )
    result = recaller.recall(SCOPE, "favourite holiday destination")
    assert result.abstained and calls == []


def test_without_decompose_or_an_llm_there_is_no_check(cocktails):
    memories, _ = cocktails
    calls = []
    done = {"refers_back": False, "list": False, "done": True}
    with_llm = memories.recaller(reranker=scores, generate=router(done, calls=calls))
    with_llm.recall(SCOPE, QUESTION, decompose=False)
    memories.recaller(reranker=scores).recall(SCOPE, QUESTION)
    assert calls == []


def test_strict_mode_raises_when_the_check_fails(cocktails):
    memories, _ = cocktails
    done = {"refers_back": False, "list": False, "done": True}
    broken = router(done, judge_reply=lambda prompt: "not json")
    with pytest.raises(ValueError):
        memories.recaller(reranker=scores, generate=broken, strict=True).recall(SCOPE, QUESTION)


# The check itself ---------------------------------------------------------------------------


def memory(n, text="Some fact"):
    return Memory(id=f"m{n}", type="fact", text=text, occurred_at=NOW)


def test_judge_maps_numbers_to_ids_and_ignores_bad_ones():
    memories = [memory(1), memory(2), memory(3)]
    reply = '```json\n{"evidence": [1, 3, 0, 4, true, "2"]}\n```'
    assert judge_evidence(lambda p: reply, "q", memories) == {"m1", "m3"}
    assert judge_evidence(lambda p: '{"evidence": []}', "q", memories) == set()
    assert judge_evidence(lambda p: 1 / 0, "q", []) == set()  # no call without candidates


@pytest.mark.parametrize("reply", ["nope", "[1, 2]", '{"evidence": "1"}'])
def test_judge_rejects_unusable_replies(reply):
    with pytest.raises(ValueError):
        judge_evidence(lambda p: reply, "q", [memory(1)])


def test_judge_prompt_numbers_dated_memories_and_states_the_rules():
    prompt = judge_prompt("Did I use lemon?", [memory(1, "Made  lemonade\nat home")])
    assert f"1. ({NOW.date().isoformat()}) Made lemonade at home" in prompt
    assert "a request, question, plan, wish or idea is not evidence" in prompt
    assert "Question: Did I use lemon?" in prompt


# Query analysis --------------------------------------------------------------------------------


def test_query_analysis_reads_the_done_flag():
    reply = '{"sub_queries": [], "time_window": null, "done": true}'
    assert analyze_query(QUESTION, NOW, generate=lambda p: reply).asks_done
    assert not analyze_query("Where is Caroline from?", NOW, generate=lambda p: "{}").asks_done


def test_query_analysis_asks_for_the_done_flag():
    # The kind-of-thing example once made the model copy it and drop the paraphrase query, and
    # every relevance score fell from 0.45 to 0.17; rule 4's concrete-words query covers it.
    prompt = _decomposition_prompt(QUESTION, NOW, 4)
    assert '"done": true or false' in prompt and "lime lemon orange grapefruit" not in prompt
