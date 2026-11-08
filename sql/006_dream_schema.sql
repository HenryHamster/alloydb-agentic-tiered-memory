-- Optional Dream extension. Run explicitly after 004 and 005.
-- Staged output is kept in dream_runs, outside active memory tables.
CREATE TABLE IF NOT EXISTS dream_runs (
    run_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN
        ('queued', 'running', 'ready', 'failed', 'applied', 'discarded')),
    created_at TIMESTAMPTZ NOT NULL,
    lease_until TIMESTAMPTZ,
    payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS dream_runs_queue_idx
    ON dream_runs (status, created_at) WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS dream_runs_scope_idx ON dream_runs (user_id, project_id);

-- Versions remain attached to their run. The normal macro pass cannot close them.
-- Only published, fresh versions are surfaced by the Dream Recall adapter.
CREATE TABLE IF NOT EXISTS dream_insights (
    run_id UUID NOT NULL REFERENCES dream_runs(run_id) ON DELETE CASCADE,
    insight_id UUID NOT NULL,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (run_id, insight_id)
);
CREATE INDEX IF NOT EXISTS dream_insights_scope_idx ON dream_insights(user_id, project_id);
