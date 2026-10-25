"""Answer LongMemEval questions from recalled context, and judge answers with an LLM.

The judge prompts follow LongMemEval's evaluate_qa.py (one per question type, abstention
separate). The paper judges with GPT-4o; here the judge is whatever `generate` is, so scores are
comparable across runs of this harness, not directly with published numbers. For those, feed
the hypotheses file to the official evaluator.
"""

from shared.llm import Generate, generate_with_retry

ANSWER_PROMPT = """I will give you memories retrieved from several past chats between you and a user. Each memory starts with its id and (type | date | speaker ...) labels. Answer the question based on the relevant memories. If the memories do not contain the information needed, say that you do not know.

Memories:
{context}

Current Date: {question_date}
Question: {question}
Answer:"""

_JUDGE_DEFAULT = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.{extra}

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only."""

_TEMPORAL_EXTRA = " In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct."

_UPDATE_EXTRA = " If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer."

_JUDGE_PREFERENCE = """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {question}

Rubric: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only."""

_JUDGE_ABSTENTION = """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only."""


def judge_prompt(
    question_type: str, question: str, answer: str, response: str, abstention: bool
) -> str:
    fields = {"question": question, "answer": answer, "response": response}
    if abstention:
        return _JUDGE_ABSTENTION.format(**fields)
    if question_type == "single-session-preference":
        return _JUDGE_PREFERENCE.format(**fields)
    extra = {"temporal-reasoning": _TEMPORAL_EXTRA, "knowledge-update": _UPDATE_EXTRA}
    return _JUDGE_DEFAULT.format(extra=extra.get(question_type, ""), **fields)


def answer(generate: Generate, context: str, question: str, question_date: str) -> str:
    prompt = ANSWER_PROMPT.format(
        context=context or "(no memories found)", question=question, question_date=question_date
    )
    return generate_with_retry(generate, prompt).strip()


def judge(
    generate: Generate,
    question_type: str,
    question: str,
    answer_text: str,
    response: str,
    abstention: bool,
) -> bool:
    prompt = judge_prompt(question_type, question, answer_text, response, abstention)
    return "yes" in generate_with_retry(generate, prompt).strip().lower()[:10]
