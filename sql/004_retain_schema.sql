-- Retain storage: source documents, atomic facts, entities, links and observations.
-- Run after 001_schema.sql. Additive: codelab tables are untouched.
-- Embeddings are written by Retain (text-embedding-005, 768 dims), not by ai.initialize_embeddings,
-- because linking needs the vector before the row is committed.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Original messages. An episode is one conversation round; get_episode returns its rows in order.
CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY,
    episode_id TEXT NOT NULL,
    position INT NOT NULL,
    user_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    speaker TEXT NOT NULL CHECK (speaker IN ('user', 'assistant', 'tool')),
    text TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,          -- retention window ("valid for x time"); NULL keeps forever
    retained_at TIMESTAMPTZ,         -- set when Retain finished; ingestion ledger for retries
    UNIQUE (episode_id, position)
);

CREATE INDEX IF NOT EXISTS documents_episode_idx ON documents (user_id, episode_id, position);
CREATE INDEX IF NOT EXISTS documents_expiry_idx ON documents (expires_at) WHERE expires_at IS NOT NULL;

-- One 8-15 word claim per row. Visibility decides who can read it; project_id/session_id record origin.
CREATE TABLE IF NOT EXISTS memory_facts (
    fact_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    visibility TEXT NOT NULL CHECK (visibility IN ('global', 'project', 'session')),
    fact_type TEXT NOT NULL CHECK (fact_type IN ('world', 'experience')),
    category TEXT NOT NULL,
    speaker TEXT NOT NULL CHECK (speaker IN ('user', 'assistant', 'tool')),
    claim_type TEXT NOT NULL CHECK (claim_type IN ('asserted', 'proposed', 'hypothetical', 'rejected')),
    text TEXT NOT NULL,
    evidence TEXT NOT NULL,                       -- exact quote; survives document expiry
    details JSONB NOT NULL DEFAULT '{}'::jsonb,   -- who / what / when / where / why
    source_document_id UUID REFERENCES documents (document_id) ON DELETE SET NULL,
    source_episode_id TEXT NOT NULL,
    embedding VECTOR(768),
    text_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    occurred_at TIMESTAMPTZ,                      -- event time; NULL for timeless states
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,                         -- NULL = current; set on supersession
    superseded_by UUID REFERENCES memory_facts (fact_id) ON DELETE SET NULL,
    evidence_count INT NOT NULL DEFAULT 1 CHECK (evidence_count >= 1),
    CHECK (valid_to IS NULL OR valid_to >= valid_from),
    CHECK (visibility <> 'project' OR project_id IS NOT NULL),
    CHECK (visibility <> 'session' OR session_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS memory_facts_embedding_idx
ON memory_facts USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS memory_facts_tsv_idx ON memory_facts USING gin (text_tsv);
CREATE INDEX IF NOT EXISTS memory_facts_scope_idx ON memory_facts (user_id, project_id, valid_to);
CREATE INDEX IF NOT EXISTS memory_facts_session_idx ON memory_facts (user_id, session_id);
CREATE INDEX IF NOT EXISTS memory_facts_occurred_idx ON memory_facts (user_id, occurred_at);
CREATE INDEX IF NOT EXISTS memory_facts_episode_idx ON memory_facts (source_episode_id);

-- Canonical entities per user. canonical_name is normalized; aliases hold surface forms.
CREATE TABLE IF NOT EXISTS entities (
    entity_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, entity_type, canonical_name)
);

CREATE INDEX IF NOT EXISTS entities_trgm_idx ON entities USING gin (canonical_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS entities_aliases_idx ON entities USING gin (aliases);

CREATE TABLE IF NOT EXISTS memory_entities (
    fact_id UUID NOT NULL REFERENCES memory_facts (fact_id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities (entity_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('subject', 'object', 'mentioned')),
    PRIMARY KEY (fact_id, entity_id, role)
);

CREATE INDEX IF NOT EXISTS memory_entities_entity_idx ON memory_entities (entity_id);

-- Directed fact-to-fact edges. relation gives direction; weight is in [0, 1]:
-- semantic = cosine similarity, temporal = time proximity (1 for supersedes), causal = 1.
CREATE TABLE IF NOT EXISTS memory_links (
    from_fact_id UUID NOT NULL REFERENCES memory_facts (fact_id) ON DELETE CASCADE,
    to_fact_id UUID NOT NULL REFERENCES memory_facts (fact_id) ON DELETE CASCADE,
    link_type TEXT NOT NULL CHECK (link_type IN ('semantic', 'temporal', 'causal')),
    relation TEXT NOT NULL CHECK (
        relation IN ('similar', 'near_in_time', 'supersedes', 'caused_by', 'evidenced_by')
    ),
    weight REAL NOT NULL CHECK (weight >= 0 AND weight <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (from_fact_id, to_fact_id, relation),
    CHECK (from_fact_id <> to_fact_id)
);

CREATE INDEX IF NOT EXISTS memory_links_to_idx ON memory_links (to_fact_id);

-- Entity graph extension: "Alice" -works_at-> "Google", each edge backed by a fact.
CREATE TABLE IF NOT EXISTS entity_relations (
    subject_entity_id UUID NOT NULL REFERENCES entities (entity_id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_entity_id UUID NOT NULL REFERENCES entities (entity_id) ON DELETE CASCADE,
    fact_id UUID NOT NULL REFERENCES memory_facts (fact_id) ON DELETE CASCADE,
    PRIMARY KEY (subject_entity_id, predicate, object_entity_id, fact_id)
);

CREATE INDEX IF NOT EXISTS entity_relations_object_idx ON entity_relations (object_entity_id);

-- Consolidated summaries over facts sharing an entity within one visibility domain.
-- Reflect may refine them; Recall reads them as type "observation".
CREATE TABLE IF NOT EXISTS observations (
    observation_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    visibility TEXT NOT NULL CHECK (visibility IN ('global', 'project', 'session')),
    entity_id UUID REFERENCES entities (entity_id) ON DELETE SET NULL,
    text TEXT NOT NULL,
    embedding VECTOR(768),
    text_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    fact_ids UUID[] NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS observations_embedding_idx
ON observations USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS observations_tsv_idx ON observations USING gin (text_tsv);
CREATE INDEX IF NOT EXISTS observations_facts_idx ON observations USING gin (fact_ids);
CREATE INDEX IF NOT EXISTS observations_scope_idx ON observations (user_id, project_id, valid_to);
