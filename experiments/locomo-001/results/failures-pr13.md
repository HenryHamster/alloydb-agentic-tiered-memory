# PR13 candidate: causes of all 49 LoCoMo failures (conv-26)

Input: attempt `locomo-pr13-4f0636ef...` (answers/*.json recall traces, prepared-state.json),
`scores.json`, `inputs/references-conv-26.json`. Script: `dump_fail.py` (output `pr13_fail_dump.txt`).
For each failure, the gold evidence turn was mapped to its episode, the facts Retain stored for that
episode were listed, and it was checked whether they were recalled. One primary cause per question,
assigned by hand.

| Cause | n | Questions |
|---|---:|---|
| Retain did not store the needed detail | 23 | q010 q012 q020 q026 q040 q052 q072 q076 q080 q088 q091 q092 q098 q109 q110 q114 q119 q124 q140 q144 q147 q148 q151 |
| Relative date resolved wrongly or not at all | 7 | q008 q029 q031 q035 q041 q073 q079 |
| Open-domain inference refused ("not specified") | 6 | q014 q027 q030 q059 q069 q081 |
| Answer stage chose wrong item with evidence recalled | 6 | q037 q043 q066 q071 q122 q135 |
| Stored fact not recalled (list/breadth or paraphrase) | 4 | q032 q034 q085 q099 |
| Gold or question defect | 3 | q005 q094 q138 |

Mechanisms, with evidence:

1. Melanie's (assistant-role) turns are under-extracted. Turns: Caroline 211, Melanie 208. Facts:
   223 about Caroline, 103 about Melanie. 15 of the 23 Retain misses come from Melanie's turns.
   The extraction prompt (`retain/extraction.py:51-90`) is written for a user/agent dialogue
   ("assistant recommendations are proposed", "experience for what the agent did"), assistant turns
   are stored as serialized JSON event lists, and the uncited-turn fallback (`extraction.py:615-631`)
   keeps only `Speaker.USER` turns. An uncited Melanie turn is lost entirely.
2. Lossy compression at Retain. Facts are 8-15 words; the asked detail is dropped (necklace "love,
   faith and strength", grandma's country tied to Sweden, "ten years ago", "known friends 4 years",
   "made pots", "scared"). Photo captions (`[shares a photo: ...]`) are almost never extracted
   (q110, q140).
3. Relative dates. The code resolver handles only yesterday/today/tomorrow/N days ago
   (`extraction.py:46-50, 824-855`). "last Fri/Tues/Friday", "two weekends ago" are left to the
   model, which placed them one week too early in 4/4 cases (q029, q035, q041, q079). "Last week"
   gets the session date (q008, q031). q073 stored 2023-09-13 for "last month" but kept the phrase
   in the text, so the answer model subtracted another month.
4. Junk facts. 34 of 329 facts are "The user said: <verbatim turn>" fallbacks, often
   acknowledgements, which occupy top-10 slots.
5. Speaker attribution. Two humans mapped to user/assistant produce misattributed facts ("The user
   has a guinea pig named Oscar"; "Caroline has been trying out abstract painting" for Melanie's;
   an observation giving Caroline a guinea pig named Oliver).
6. Stale observations. The "pup and kitty named Luna and Oliver" observation outranks later facts;
   Bailey (third pet) was never stored (q052, q124).
7. Answer prompt. The short-answer instruction ("based only on what was said") makes the model
   refuse "likely"-style inference; 21 of 49 failures are abstentions.
8. Breadth. List questions get ten near-duplicates (five Pride facts for q032) instead of distinct
   events; fixed top-k with no diversity.

Gold defects: q005 gold "Sunday before 25 May" but the turn says "last Saturday" (PR13's May 20 is
right); q094 and q138 attribute Caroline's bowl / Melanie's painting to the other speaker.
