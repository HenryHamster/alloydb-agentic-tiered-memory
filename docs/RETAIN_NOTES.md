# Retain: design and status

Retain is the write path. For every conversation round it stores the original messages,
extracts facts, resolves entities and links facts to each other, so Recall and Reflect can
search and reason over them. Gemini extraction, embedding, AlloyDB storage, entity resolution,
linking, update classification, and observation building are implemented.

## Pipeline

```
AsyncMemoryWorker (retainer=...)          one queued turn, trusted Scope(user, project, session)
└─ Retainer.retain_turn
   1 documents        store the round's messages, one per reply or tool-event run;
                      episode = one round; skip if already retained
   2 chunk_documents  700-char lossless spans (shared/chunking.py)
   3 FactExtractor    detailed facts + evidence quote, when, entities, relations,
                      causal refs, proposed visibility; tool calls/results built by code
   4 EntityResolver   mentions -> canonical per-user entities (exact, alias, fuzzy)
   5 Embedder         768-dim text-embedding-005 vectors
   6 SupersedeClassifier  add / duplicate (reinforce) / supersede (close valid_to)
   7 links            supersedes + causal (from extraction) + Linker (semantic, temporal)
   8 RetainStore.write    one transaction; marks documents retained
Retainer maintenance     every 20 rounds per scope (RetainConfig.maintenance_every):
                          ObservationBuilder -> observations, then purge expired documents
```

Short-term memory is `valkey_buffer.py`: the sliding window and rolling summary. Its `round_id`
is passed to Retain as the `episode_id`, so Recall can skip rounds still in the window.

## Files

| File | Contents | Status |
| --- | --- | --- |
| `sql/004_retain_schema.sql` | `documents`, `memory_facts`, `entities`, `memory_entities`, `memory_links`, `entity_relations`, `observations` | Complete |
| `retain/models.py` | Enums, records, `is_visible`, `resolve_visibility`, `stable_id` | Complete |
| `retain/pipeline.py` | `Retainer`, decision handling, `build_retainer` (production wiring) | Complete |
| `retain/store.py` | `RetainStore` protocol, `InMemoryRetainStore` reference | Complete |
| `retain/extraction.py` | `chunk_documents`; `GeminiFactExtractor` | Implemented |
| `retain/entities.py` | `normalize_name`; `FuzzyEntityResolver` | Implemented |
| `retain/embeddings.py` | `Embedder` type; `GeminiEmbedder` | Implemented |
| `retain/linking.py` | `causal_links`; `StoreLinker`, `GeminiSupersedeClassifier` | Implemented |
| `retain/observations.py` | `GeminiObservationBuilder` | Implemented; runs in periodic maintenance or `Retainer.consolidate` |
| `retain/alloydb_store.py` | `AlloyDBRetainStore`, shared visibility SQL | Implemented |
| `async_worker.py` | `retainer=` replaces the `agent_entities` upsert; passes `project_id` and message time | Complete |

## Decisions

- **Scope.** The caller supplies `Scope(user_id, project_id, session_id)`. The model only proposes
  `visibility` (global, project, session); `resolve_visibility` narrows it to what the context
  supports. A project fact with no project falls to session, never to global.
- **Read rule** (shared with Recall): same user, then global, same project, or same session.
  Every store query and both ends of every link must satisfy it.
- **Idempotency.** Documents, facts and entities have deterministic IDs. `documents.retained_at`
  is the ingestion ledger, so a replayed round is skipped without a model call. A failed stage
  writes nothing and the round can be retried.
- **Time.** `occurred_at` is event time (null for timeless states); `valid_from` defaults to the
  message time; `recorded_at` is ingestion time and never stands in for either. Supersession
  sets the old fact's `valid_to` to the new fact's `valid_from`, never earlier than the old start.
  Reduced-precision ISO dates (`2023`, `2024-03`) in either extracted timestamp field preserve
  the fact and date in `details.when`, leaving the exact timestamp unset. The usual source-time
  default for `valid_from` still applies in storage; it does not mean the partial date was
  resolved. No missing month or day is invented. Complete dates and timestamps remain validated.
- **Relative dates are resolved by code** from the phrase in the evidence quote and the time the
  message was sent (only when the source timestamp is known): days ("yesterday", "3 days ago"),
  weekdays ("last Friday" = the most recent Friday strictly before), weekends, weeks ("last week"
  = `occurred_at` 7 days earlier, `when` "the week before 22 May 2023"), months and years
  (`when` only). The model put "last Friday" a week early in all four LoCoMo cases, so the code's
  date wins over the model's, and a disagreement never drops the fact. The phrase in the fact text
  is replaced with the date ("last month" becomes "in April 2023") so an answer cannot count back
  from it twice; "ten years ago" keeps the duration and adds "(around 2013)". The model is told to
  copy relative words as said rather than compute dates. The phrase is looked for in the quote,
  then the sentence around it (short quotes drop "last week" said a few words later), then the
  fact text. A day the model wrote into the text anyway is replaced when it is the only date
  there and within two weeks of the code's date; a date already present is never written twice;
  an event whose text has no date gets one. Failing all of that, a model date within two weeks
  of the chunk's only day phrase is corrected to it. A full date more than a month from the day said, in a turn
  with no date words at all (no phrase, month, year, season or holiday), is dropped: it can only
  be the model's arithmetic ("known them for 4 years" once came back dated 2019). A "shared a
  photo" fact keeps the day it was shared: in "a painting I did last week" the phrase dates the
  painting, so only the text gets that date (moving the fact a week back lost "what painting did
  Melanie show on 13 October").
- **Trust.** Speaker, scope and episode come from documents, not the model. An assistant cannot
  assert a user preference; it is stored as `proposed`.
- **People, not only agents.** `retain_turn` / `record_turn` take optional `user_name` and
  `assistant_name`, and a serialized assistant event may carry a `name`. A named speaker is a
  person (two-person chats such as LoCoMo): the name goes to the model as `speaker_name`, facts
  name who said what instead of "the user"/"the assistant", their statements are `asserted`
  `world` facts, and their preferences are not demoted. Without a name nothing changes.
  The prompt asks for every statement in a turn (opinions, meanings, reactions, feelings, a view
  phrased as a question) and for "it/that/we did it" to be named from the context. That context
  includes the session's previous turn (`Document.context`, read but never stored or cited), so
  "we just did it yesterday" can name the walk shown in the photo one round earlier.
  A named speaker's fact that still opens with "The assistant" gets their name.
- **One document per reply.** A serialized event list from the assistant side is split: each
  plain reply becomes its own document with clean text (quotable, with its own speaker name),
  and consecutive tool calls and results stay together as JSON for the tool-fact builders.
- **Detail over brevity.** Facts are complete statements that keep names, numbers, durations,
  places, what an object means, what was made, and feelings, with each detail in the same fact as
  the thing it describes. Photo captions (`[shares a photo: ...]`) are recorded as what the
  speaker showed. The earlier 8-15 word target dropped exactly the details LoCoMo asked about.
- **Classifier output is untrusted.** Each duplicate/supersede target must be stored, current,
  same user and visible from the new fact; a fact never supersedes a broader one.
- **Fail-soft on model output.** A bad extraction row, repeated key or bad reference drops only
  that row; an invalid or missing classifier decision keeps the fact as new, and so does a classifier
  call that still fails after its retries (Gemini answered one small LoCoMo request with 429
  on every attempt); an ambiguous entity
  name gets its own entity. A response that is not valid JSON still fails the round, which then
  stays unretained so it can be retried.
- **Extraction cost.** `build_retainer` runs extraction at `low` thinking with the lean schema
  (no who/what/where/why details, which Recall never reads) and builds tool-call and tool-result
  facts from their JSON, since low thinking tends to skip tool output. The class defaults keep the
  full schema at `medium`. On three Mem2Act histories this cut extraction from $2.63 to $0.35 and
  kept every key fact. Code locates the evidence quote, including quotes of serialized assistant
  turns whose line breaks are stored as `\n`; the words must still appear in order.
- **Requests are facts.** The prompt records what the user asked for, and a message chunk that
  no extracted fact cites is kept verbatim: `The user said: ...`, or `Melanie said: ...` for a
  named speaker. An unnamed agent's own reply is not. Chit-chat is not kept: a turn of at most 12
  words made only of function words and reactions ("Wow, so proud of you!"); any number,
  question, request, name or other content word keeps it. Between named people, questions are
  small talk and are dropped from the turn ("Wow! How was it?" keeps nothing), and the fallback
  text drops the "Melanie:" prefix so the name is not repeated. Rejected model rows are counted by
  reason in one log line per extraction.
- **Duplicates** reinforce the stored fact (`evidence_count + 1`) instead of inserting a copy.
- **Links.** `link_type` is semantic, temporal or causal as proposed; `relation` gives direction
  (`similar`, `near_in_time`, `supersedes`, `caused_by`, `evidenced_by`). Weight is in [0, 1].
  Entity links live in `memory_entities`, not `memory_links`. Temporal links use `occurred_at`
  only and never join two facts from the same round, so facts are not linked merely for being
  said in the same conversation.
- **Round links.** With `link_rounds` (on in `build_retainer`) code adds `evidenced_by` links that
  similarity cannot find, because requests, tool events and answers fall in different
  categories: a tool call cites the user's request, a tool result the call before it, and an
  assistant fact the tool event whose names it repeats, else the request. `evidenced_by` is not
  a Reflect clustering relation, so these links add no macro summaries.
- **Code-built facts** (tool calls, tool results, `The user said: ...`, `<Name> said: ...`) get `other` entities from
  naming fields (`*_id`, `name`, `path`, `url`, `make`, ...) and ID-like tokens, never tool
  names. Two code-built facts never get a `similar` link: their template wording inflates cosine.
- **Observations.** One per entity with at least three current asserted facts. The people
  talking get none: theirs would summarize nearly every fact, lose detail and be rebuilt every
  session (observations were 42% of LoCoMo ingest cost). When a group only gains facts, the
  model updates the previous summary from the new facts alone. The prompt says newer facts win
  (with their date), every person, pet and item is named, and each fact belongs to the person it
  is about. A version is dated from its facts, never the wall clock, so a replayed history
  doesn't make a summary look newer than the facts it covers.
- **Graph.** Stored relationally (`memory_links`, `entity_relations`); traversals are SQL joins or
  recursive CTEs. Apache Spark can read these tables for offline graph analytics; it is not on
  the per-turn path.
- **Embeddings** are computed by Retain, not `ai.initialize_embeddings`, because linking needs the
  vector before commit.

## Answers to Recall's open questions (docs/RECALL_NOTES.md on `recall`)

1. Store tables: see `sql/004_retain_schema.sql`. The AlloyDB `RecallStore` reads them.
2. `memory_facts` has `user_id`, `project_id`, `speaker`, `claim_type`, `source_episode_id`,
   `evidence_count`, `text_tsv` (GIN), HNSW on `embedding`, index on `(user_id, project_id,
   valid_to)`. `observations` has text, `text_tsv`, dates and `user_id`.
3. Episodes are `documents` rows sharing `episode_id`, ordered by `position`. An expired episode
   returns nothing; facts keep their `evidence` quote.
4. Scope: the read rule above.
5. Values: `claim_type` asserted/proposed/hypothetical/rejected; `fact_type` world/experience.
6. Weight: semantic = cosine, temporal = time proximity (1 for supersedes), causal = 1.

## Validation and integration

See [Retain cloud test handoff](RETAIN_HANDOFF.md) for configuration, schema setup, the live
test (`tests/test_retain_live.py`), and read-only inspection queries. The codelab continues to use its legacy
`agent_entities` writer and reader unless a `build_retainer(...)` instance is explicitly injected
into the worker. The legacy reader cannot retrieve `memory_facts` or `observations`; Recall still
needs its AlloyDB adapter for the Retain schema before Retain → Recall → Reflect can be verified.

### Follow-up: preserve useful questions in named-speaker fallbacks

Missing extraction is not evidence that a question is disposable. Named speakers can
ask for actions, confirm dates, or express facts as questions. The fallback path now
removes only complete generic social questions (for example, "How was it?") and
preserves other questions, including "We agreed on Friday, right?" and the LoCoMo
wording "Sunflowers mean warmth and happiness, right?". Remaining question marks
prevent the filler-word filter from dropping the message. Source evidence is unchanged;
validated extracted facts are unaffected and no model calls are added.

This is a follow-up to Retain v3, not a replacement for its small-talk filtering.
The earlier courtesy-only branch matched zero user messages in conv-26. Inspecting
v3's named-speaker rules on that same history excluded 13 user messages from fallback
eligibility, but these are potential fallbacks, not observed stored/retrieved facts.
The follow-up deliberately keeps substantive questions that v3 removed. Offline
regression tests cover requests, negation, confirmation, both speaker roles, and
mixed social/useful text. No new cloud run or answer-accuracy gain is claimed.
