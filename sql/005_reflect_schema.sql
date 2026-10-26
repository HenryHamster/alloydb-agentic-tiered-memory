-- Reflect storage: fact lifecycle (tiers, access, rank), redundancy/conflict markers,
-- conflict review queue and macro memories. Run after 004_retain_schema.sql. Additive.
-- valid_from / valid_to / superseded_by already exist on memory_facts (Retain).

-- Per-row lifecycle. New facts start hot with rank 0; Reflect recomputes rank_score and tier.
-- Recall bumps access_count / last_accessed_at for every fact it returns.
ALTER TABLE memory_facts
    ADD COLUMN IF NOT EXISTS access_count INT NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS rank_score REAL NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'hot'
        CHECK (tier IN ('hot', 'warm', 'archived')),          -- purged rows are deleted
    ADD COLUMN IF NOT EXISTS tier_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS confidence REAL NOT NULL DEFAULT 1
        CHECK (confidence >= 0 AND confidence <= 1),          -- lowered by unresolved conflicts
    ADD COLUMN IF NOT EXISTS consolidated_into UUID
        REFERENCES memory_facts (fact_id) ON DELETE SET NULL, -- redundant original -> merged
    ADD COLUMN IF NOT EXISTS conflicts_with UUID[] NOT NULL DEFAULT '{}';

-- "Top of the table" is this index, not physical row order: Recall scans hot first by rank.
CREATE INDEX IF NOT EXISTS memory_facts_tier_rank_idx
ON memory_facts (user_id, tier, rank_score DESC);
CREATE INDEX IF NOT EXISTS memory_facts_dormant_idx
ON memory_facts (tier, last_accessed_at) WHERE tier = 'archived';

-- Graph edges for Reflect's markers, so traversals see them like supersedes.
ALTER TABLE memory_links DROP CONSTRAINT IF EXISTS memory_links_relation_check;
ALTER TABLE memory_links ADD CONSTRAINT memory_links_relation_check CHECK (
    relation IN (
        'similar', 'near_in_time', 'supersedes', 'caused_by', 'evidenced_by',
        'consolidated_into', 'conflicts_with'
    )
);

-- Unresolvable conflicts waiting for an LLM or human judgment call (optional consumer).
CREATE TABLE IF NOT EXISTS memory_conflict_reviews (
    review_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact_ids UUID[] NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved', 'dismissed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS memory_conflict_reviews_open_idx
ON memory_conflict_reviews (user_id) WHERE status = 'open';

-- Asynchronous synthesis layer: multi-session narratives over linked fact clusters.
-- Recall searches these alongside facts, so high-level questions need no runtime LLM loop.
CREATE TABLE IF NOT EXISTS macro_memories (
    macro_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT,
    session_id TEXT,
    visibility TEXT NOT NULL CHECK (visibility IN ('global', 'project', 'session')),
    macro_title TEXT NOT NULL,
    macro_summary TEXT NOT NULL,
    macro_embedding VECTOR(768),                       -- embed(title + summary); redone per version
    summary_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', macro_title || ' ' || macro_summary)
    ) STORED,
    supporting_fact_ids UUID[] NOT NULL,               -- every from/to fact used as evidence
    primary_entity_ids UUID[] NOT NULL DEFAULT '{}',   -- subject/object entities of the cluster
    causal_chain_ids JSONB NOT NULL DEFAULT '[]'::jsonb, -- [[cause_id, ..., effect_id], ...]
    temporal_span TSTZRANGE,                           -- min/max time of temporally linked facts
    version INT NOT NULL DEFAULT 1 CHECK (version >= 1),
    access_count INT NOT NULL DEFAULT 0 CHECK (access_count >= 0),
    last_accessed_at TIMESTAMPTZ,
    rank_score REAL NOT NULL DEFAULT 0,
    tier TEXT NOT NULL DEFAULT 'hot' CHECK (tier IN ('hot', 'warm', 'archived')),
    tier_changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to TIMESTAMPTZ,                              -- closed when its cluster dissolves
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (visibility <> 'project' OR project_id IS NOT NULL),
    CHECK (visibility <> 'session' OR session_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS macro_memories_embedding_idx
ON macro_memories USING hnsw (macro_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS macro_memories_tsv_idx ON macro_memories USING gin (summary_tsv);
CREATE INDEX IF NOT EXISTS macro_memories_facts_idx
ON macro_memories USING gin (supporting_fact_ids);
CREATE INDEX IF NOT EXISTS macro_memories_tier_rank_idx
ON macro_memories (user_id, tier, rank_score DESC);
