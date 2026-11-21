-- Speaker names: the person who said a fact ("Melanie"), so one person's claim never repeats or
-- replaces another's, and an observation summarizes one person's facts. Deploy before code that
-- writes speaker_name. Additive; existing rows stay NULL and keep the role-only behaviour.
ALTER TABLE memory_facts
    ADD COLUMN IF NOT EXISTS speaker_name TEXT
        CHECK (speaker_name IS NULL OR btrim(speaker_name) <> '');

ALTER TABLE observations
    ADD COLUMN IF NOT EXISTS speaker_name TEXT
        CHECK (speaker_name IS NULL OR btrim(speaker_name) <> '');
