# Retain implementation and integration notes

Retain is a synchronous, injectable fact extraction and linking pipeline. It does not
replace the codelab worker, change Recall/Reflect, or create database tables. Changes
are confined to Retain, its tests, and shared chunking. No migration has been applied.

## Review findings and reuse

The starting implementation provided frozen records, source/scope validation, a chunker
with a 2,000-character limit, explicit update classification hooks, and validity-interval
planning. Gemini and AlloyDB adapters raised NotImplementedError. Entity resolution was
only a caller callback. There was no actual entity/semantic discovery, persistence,
retry deduplication, or automatic conflict detection.

The repository's extracted codelab code was reviewed directly:

- `async_worker.py` already calls an injected `llm_client.models.generate_content` with
  JSON output. Retain adapts that client pattern, not its entity-summary prompt or upsert.
  The worker combines user/assistant text and lets the model choose scope; Retain keeps
  speaker and scope under application control. Its temporal helper substitutes session
  identifiers, which would lose event dates, so Retain uses source timestamps instead.
- `db_clients.py:get_db_connection` is reusable as a future connection factory. Importing
  it does not fetch credentials, but calling it invokes gcloud. Offline Retain never does.
- The existing embedding path uses `models.embed_content`, `text-embedding-005`, and
  768-dimensional database vectors. `GeminiEmbedder` adapts that call with an injected
  client and configurable model; it propagates failures instead of inserting zero vectors.
- Existing HNSW/RUM indexes and auto-embedding initialization are examples for later
  fact storage. The current auto-embedding configuration covers only `agent_entities`.

## Chunking algorithm and comparison

Default `max_chars=700`, configurable in both `retain` and `chunk_conversation`.
Messages stay separate. In each full window, search its second half for the last
paragraph boundary, otherwise sentence punctuation followed by whitespace, otherwise
newline, otherwise space. Include the separator in the preceding chunk. If none exists,
cut at the character limit. Sentence detection excludes fenced code, inline code and
URLs, and skips common title abbreviations and single-letter initials. Decimal points
without following whitespace do not count as sentence ends.

Offsets are exact half-open Python Unicode code-point offsets, not bytes or graphemes.
Chunks are contiguous, unstripped and nonoverlapping. Concatenation exactly reconstructs
the source, including Markdown, unusual whitespace, emoji and combining characters.
IDs are deterministic from source, scope, content and offsets. Metadata remains on the
original message: speaker, timestamp (possibly unknown), source/episode IDs and scope.

This is punctuation/length chunking, **not semantic topic detection**. Paragraphs provide
an explicit topic hint; unrelated facts within a paragraph are separated by Gemini.
Fallback cuts can split a long sentence, name, URL, code block, date or grapheme. The
original message remains available for extraction; chunks alone must not be interpreted
as independent complete claims. Protecting every long term while also enforcing an
arbitrary hard character limit is impossible.

An offline fixture contains 341 words / 2,116 characters, covering project architecture,
gardening, travel, conditional preferences and music. Actual chunk counts:

| Limit | Chunk count |
| --- | --- |
| 300 | 9 |
| 350 | 7 |
| 400 | 7 |
| 600 | 5 |
| 700 | 4 |
| 800 | 3 |
| 2,000 | 2 |

700 is a provisional middle ground: fewer fragmented qualifications than 300–400, finer
citation spans than 2,000. This is a deterministic boundary comparison, **not evidence
of improved extraction accuracy**. Compare live extraction precision/recall and cost
before tuning the default. The final short chunk is allowed; no artificial overlap.

## Gemini extraction and validation

`GeminiFactExtractor(client, model=...)` uses the existing google-genai dependency and
client interface. It requests JSON with a response JSON schema and validates the result
locally. See Google's [structured-output documentation](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).
No clients or credentials are constructed implicitly. The default model follows the
codelab; availability must be checked in the target Vertex AI project.

The adapter groups chunks per message, then packs contiguous core windows up to
`max_context_chars=4200`. Each request adds at most `neighbor_chars=350` source characters
on each side, so supplied source text is at most 4,900 characters by default. Limits are
configurable on GeminiFactExtractor; they count code points, not tokens or prompt overhead.
Short messages are sent once. Long messages produce multiple bounded requests, with only
neighboring text repeated. Chunks larger than the core budget fail before any model call;
configure the chunker and extractor limits together.

Payloads include absolute source_start/core_start/core_end and intersecting chunk IDs with
original offsets. Facts may cite adjacent chunks, but quoted evidence must fit both the
request text and contiguous cited spans. Only facts whose evidence starts inside the core
are retained; neighboring context cannot create duplicate facts in another window. Local
validation also rejects invented/hidden evidence. A very long fact or distant reference
can exceed neighboring context and be missed; increase the limits for those workloads.
No semantic chunking or additional dependencies are introduced. 700 controls citation
spans; the separate extraction limits control source context. Prompt/response cost and
live extraction quality still need measurement.

Unknown fields stay null. Concision targets 8–15 words without enforcing a length limit.
Relative dates are resolved by Gemini using source timestamp when unambiguous; precision
and correctness are not guaranteed by schema validation. Month-only or ambiguous dates
should remain text in `when`, with no invented day in occurred_at. No deterministic
calendar parser or semantic entailment verifier is implemented. Missing source timestamps
stay missing; recording time is never substituted for occurrence time.

Local checks reject malformed JSON, wrong fields/enums/types, naive dates, empty text,
unknown chunk IDs and evidence outside contiguous cited spans. Scope/speaker/episode
come from sources, never model output. Assistant preference assertions are conservatively
stored as proposals; they cannot establish user adoption. User assertions, hypotheses,
rejections and tool claims retain their modality and source speaker. Tool output is not
independently verified as successful or true. Exact quotes prove provenance, not that the
generated sentence follows logically; mocked tests cannot verify Gemini's reasoning.

Exact normalized same-message facts have deterministic IDs independent of chunk size.
Duplicates in one extraction merge citation membership. Retrying unchanged extracted
facts against the in-memory store returns no new insertions. Distinct source messages
remain separate evidence, and differing text/conditions/negation remain distinct facts.
Paraphrase deduplication and model-output drift across retries remain unresolved. There
is no processed-message ingestion ledger, so retries still call extraction. A store with
only `write` cannot provide candidate lookup or pipeline replay suppression.

## Persistence and database proposal — approval required before migration

`InMemoryRetainStore` is an executable reference: scoped candidate lookup, atomic batch
writes under a lock, stale-previous comparisons, endpoint/scope checks, duplicate-ID
checks and preservation of closed historical versions. Failed writes leave state
unchanged. It is neither durable storage nor a vector index. `AlloyDBRetainStore` remains
an explicit stub pending an agreed physical schema. No existing fields were altered.

| Existing field/table | Reuse / limitation |
| --- | --- |
| agent_entities user_id/project_id/session_id/scope | Reuse names and visibility concepts. Global is user-wide. Existing project_id='global' sentinel needs explicit mapping to a missing project, not a new visibility level. |
| entity_id/entity_name/summary | Existing entity summaries remain owned by worker/retriever; not atomic fact identities. UNIQUE(user_id, entity_name) cannot safely represent versions or differing project preferences. |
| summary_embedding VECTOR(768), summary_tsv | Reuse model/dimension and index patterns, not the current entity-specific trigger for new facts. |
| updated_at | Last mutation time; cannot substitute for occurred_at, recorded_at or validity intervals. |
| episodic_memory_embeddings uuid/document/cmetadata/custom_id/created_at | Could hold source documents and metadata; currently no normalized source-message/episode contract or fact-link foreign keys. created_at is ingestion time, not an event date. |
| episodic_memory_collections | Could group episodes after defining identity and retention semantics. |
| user_permissions | Execution control; Retain has no reason to change it. |

Minimal proposed additive storage is `memory_facts` plus `memory_links` and durable source
messages (possibly mapped onto episodic documents). Keep existing tables unchanged:

- Facts: UUID id; text; category; claim_type; speaker; user_id/project_id/session_id/scope;
  source_episode_id; occurred_at/recorded_at/valid_from/valid_to; optional embedding;
  source citations with message ID and exact offsets; canonical entity membership.
- Links: source_id, target_id, relationship and trusted scope. Foreign keys and a unique
  endpoint/type key; no graph extension or mandatory weight.
- Sources: original text, speaker, timestamp and full scope, stable source/episode IDs.
  Persist uncited messages too if Recall must retrieve a complete episode.

Who/what/when/where/why are extraction audit fields; JSON metadata can avoid duplicating
search columns. Exact evidence can be reconstructed from source spans if exact fact-level
spans are normalized later; the current quote and chunk spans are intentionally distinct.
`supporting_ids` and evidence_count can be derived from provenance/link membership, rather
than separately mutable counters. Entity aliases can remain caller configuration initially;
a durable shared alias registry is a later decision. Scope origin and visibility differ:
global/project facts may retain origin session metadata without narrowing visibility.
Global memories permit unknown project/session origin; scoped memories must supply IDs
required by their visibility domain. No fabricated origin IDs are added.

Existing worker writes agent_entities; hybrid_retriever reads those summaries and indexes;
ADK tools and consolidation read/write episodic documents; cleanup truncates those tables.
These dependencies make repurposing or deleting their columns unsafe. The additive proposal
must define migrations, indexes and transaction behavior before a database adapter is built.

## Linking status

| Type | Actual behavior |
| --- | --- |
| Entity | Working offline. Gemini supplies name/kind/context. Unicode normalization, case folding and whitespace normalization precede exact matching. Caller-reviewed typed/contextual alias mapping supports canonical variants. Explicit memory-to-memory entity links are capped by top-k. |
| Semantic | Working offline with injected/mock embeddings or GeminiEmbedder. Cosine top-k defaults to 3, threshold 0.92 configurable. Explicit links are persisted separately from embedding generation. No database vector indexing/search is implemented for facts. Threshold is provisional, not calibrated. |
| Temporal | Explicit historical supersession works through injected classification: new → old; close old interval, keep old text/provenance, open new interval. No automatic before/after discovery. Equal/near timestamps alone produce no edges. |
| Causal/evidence | Deferred discovery. caused_by/supported_by and generic causal/evidence relationship values are representable, but no extraction or creation path infers them. A why string is retained, not automatically converted to a causal edge. |

Entity matching intentionally avoids fuzzy merging and inferred aliases. Berkeley the
city and Berkeley the university need different kind/context. The same entity can still
fragment if Gemini varies its context label; a shared reviewed registry/query resolver
is needed before production. No additional NER dependency is introduced.

Discovery considers only the same reconciliation/visibility domain, same speaker and same
claim type, excludes closed candidates, and validates both endpoints on write. This is
stricter than read visibility: visible global facts are not automatically linked or
modified by project/session claims. Similarity can connect opposing or conditional claims;
semantic links mean relatedness only, never equivalence, truth, supersession or causation.
Do not use them as preference updates. No automatic contradiction classifier exists.
No weights are stored; cosine scores select neighbors, not confidence in a claim.

Supersession requires explicit change kind, effective timestamp and an older open interval
with a known start. Intervals are half-open. Corrections still raise NotImplementedError
because closing an interval would misleadingly imply a false claim was once valid.
A replay of an already-applied explicit update batch is rejected as stale; normal retain
message replay is suppressed through existing fact IDs. Concurrent retries with changed
extraction need a durable ingestion/transaction strategy.

## Recall contract and remaining integration

Inspected local `origin/recall` with git show, without switching or merging branches.
Recall's Memory needs id, type, text, occurred_at, recorded_at, valid_from, valid_to,
speaker, claim_type, source_episode_id, supporting_ids, entity_ids and evidence_count.
Retain preserves those concepts; the adapter must flatten fact fields and set type='fact'.
Category is independent of Recall memory type. No Recall/Reflect files were changed.

RecallStore requires scope-filtered `search`, `expand_neighbors`, `get_episode` and
`get_supporting`, including endpoint checks at every graph hop. Its current scope fields
are caller read context; Retain additionally records visibility and source origin. Agree
SQL policy before wiring them together. Recall's stale/interval handling also needs
agreement on valid_from filtering and half-open endpoints. Proposed/rejected/assistant
claims must remain distinguishable in downstream answers. All SQL must bind parameters.

Production integration needs schema approval, transactional AlloyDB persistence, durable
full-episode ingestion, scoped vector/keyword indexes and Recall adapters. Query-time
entity normalization must use the same registry. Worker integration should replace its
write path deliberately, avoiding two independent writers silently diverging. Reflect
and observation/mental-model creation remain outside this change. LongMemEval ingestion,
quality evaluation and a codelab baseline are future work, not claimed results.

## Offline checks and live validation

Run `.venv/bin/pytest`, `.venv/bin/ruff check .` and
`.venv/bin/ruff format --check .` (or equivalent uv commands with dev dependencies).
Tests use fake extraction/embedding clients; no automated test contacts cloud services.
Fixtures cover short/no-fact messages, multi-topic text, cross-boundary evidence, long
unbroken text/sentences, code/URLs/lists, Unicode/whitespace, conflicting/conditional
preferences, scope isolation, alias disambiguation, unrelated equal-time events,
historical event dates, assistant suggestions, malformed responses, duplicate chunks,
replays, stale updates, invalid embeddings and atomic rollback. These demonstrate local
behavior, not live model quality.

For an optional live extraction smoke test, configure ADC and a Vertex-enabled project,
construct the same google.genai.Client used by the codelab, inject it into
GeminiFactExtractor and GeminiEmbedder, and use InMemoryRetainStore first. Select models
available in that project; inspect evidence, modality, month/date precision and entities.
No database is needed for this smoke test. AlloyDB integration additionally requires the
approved additive schema, working store adapter, IAM connection factory, compatible vector
dimensions/indexes and Recall mapping. Do not run the repository's live demo modules as
part of offline pytest. No migration, commit, push or PR update is performed here.

## PR cleanup decisions and shared-model boundary

The review's comment cleanup, fixture defaults, behavior-focused testing, entity-helper
placement, enum overlap and shared-model concerns were reviewed together. Long algorithm
and integration explanations live here; code docstrings describe only local contracts.

`resolve_entities` attaches canonical IDs after source validation and before link discovery.
It now owns both the injected resolver path and default typed/contextual alias matching,
removing inline canonicalization from orchestration. Without entities or a resolver it
preserves existing IDs. This short Retain-specific step stays in retain.py; a new module
would only move a small function without providing an independently reused service.

No enums have duplicate values within their own domain. UpdateCategory is a write decision;
RelationshipType is a persisted edge label. ADD has no edge, while ENTITY/SEMANTIC have no
update decision. CONFIRM becomes a new-to-old CONFIRMS edge, for example. They remain
separate to prevent an entity/semantic edge from being accepted as a write operation.
ChangeKind describes historical change versus correction, not an edge or decision.
Generic TEMPORAL/CAUSAL/EVIDENCE labels overlap more specific reserved labels, but are
existing exported values: removal or aliasing would change serialized strings. Keep them
for compatibility and coordinate a single directed vocabulary before durable persistence.
No unused placeholder vocabulary was added in this cleanup.

Candidate shared records: visibility/read context, memory identity and dates, immutable
source references, entity identity and relationship vocabulary. Retain-only records:
AtomicFact extraction output, ConversationChunk extraction spans, MemoryUpdate/ChangeKind
write decisions, and WriteBatch transactional plans. Message/source references may become
shared after full-episode ingestion is specified. Do not move models to a generic models.py
yet: Recall's Scope is a three-field read context, while Retain's Scope additionally carries
visibility/origin policy; Recall's flat mutable Memory supports multiple types, while
Retain's frozen MemoryFact nests an AtomicFact. Preserve both interfaces and coordinate
an explicit adapter/shared neutral contract with Recall and Reflect. A direct class move
would not resolve these differences and could break imports or validity semantics.

Tests now use concrete account/project/session/message/episode IDs. The visibility test
uses explicit allowed domains rather than reproducing the visibility predicate. Removed
per-enum construction repetitions and arbitrary separator-offset examples; retained tests
for lossless offsets, sentence qualifications and exact update edge direction/type.
Negative cases cover duplicate classifier decisions, nonadjacent/cross-message evidence,
window-hidden evidence, oversized input chunks and atomic cross-user link rejection.
Bounded-context tests check request slices, absolute offsets and single ownership of
neighbor facts. Extraction/embedding responses remain mocked: these tests verify local
contracts, not live Gemini fidelity. Historical updates remain explicit; automatic
temporal/causal discovery, observation creation, AlloyDB writes and end-to-end Recall
integration are deferred.

## Final correctness pass

Evidence attribution now requires exactly one occurrence of the quotation within the
contiguous cited source spans. Repeated text elsewhere in the message does not override
those citations. If the cited spans themselves contain multiple matches, extraction fails
before storage: narrow the chunk citations or include distinguishing context in the quote.
Ownership uses this unique occurrence, not the first occurrence in a broad citation range.
This also applies to injected extraction outputs when they supply evidence. No new evidence
offset fields or changes to shared models are introduced. Chunk citations plus the unique
quote identify the source span; entailment is still not verified by string matching.

A regression fixture repeats the same quotation under prototype and production contexts:
broad ambiguous citations fail, while a specific citation or extended quote succeeds.
Another fixture starts a qualified preference at a core-window boundary and places its
production exclusion in the next core. Multiple overlapping requests return it, but only
the owner retains it, with both source chunks preserved. This tests local ownership and
validation with mock responses, not whether Gemini reliably notices the qualification.
If required context lies beyond the supplied neighbors, a fact can still be missed or
misinterpreted. Increase configured limits or reject the workload; no completeness or
semantic-entailment guarantee is provided. One ambiguous row fails the entire extraction
call rather than silently persisting other rows from that call.

Visibility does not replace origin: a global memory is readable across the same user's
projects while retaining known source project/session IDs unchanged. Existing scope,
source-ID, timestamp and episode checks remain in place. Nullable project/session origins
are useful for external imports that genuinely lack those identifiers, but no importer in
this repository currently requires them. Keep the existing nullable contract; never erase
known origin metadata or invent session identifiers. Whether production ingestion should
require complete origin for native conversations, and which external import metadata is
mandatory, remain policy decisions for the shared contract. Unknown origins do not permit
cross-user access or cross-visibility reconciliation.

Candidate lookup now checks the optional bound method once, without inspecting the store's
class or expanding RetainStore. Write-only stores remain supported without discovery;
lookup is still required when entity/semantic discovery is enabled. Test stores declare
only their actual write capability so mocks do not manufacture candidate methods.
