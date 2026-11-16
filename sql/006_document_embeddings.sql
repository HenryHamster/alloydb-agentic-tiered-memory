-- Retain v3: each stored message keeps its own vector, so Recall can quote exact words.
-- Deploy this before code that writes documents.embedding. Additive; existing rows stay NULL.
-- Run it as its own execution block, outside a transaction: CREATE INDEX CONCURRENTLY builds
-- the index without blocking Retain's writes to documents, and cannot run in a transaction.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding VECTOR(768);

CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_embedding_idx
ON documents USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
