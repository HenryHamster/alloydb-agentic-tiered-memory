-- Speaker names: who said a message when a conversation has named people (LoCoMo's Caroline
-- and Melanie). Run after 004_retain_schema.sql. Additive; NULL keeps the role-only behaviour.
-- documents.speaker_name is the source; memory_facts copies it from the source document, and an
-- observation summarizes the facts of one speaker_name, so two people's claims are not merged.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS speaker_name TEXT
        CHECK (speaker_name IS NULL OR btrim(speaker_name) <> '');

ALTER TABLE memory_facts
    ADD COLUMN IF NOT EXISTS speaker_name TEXT
        CHECK (speaker_name IS NULL OR btrim(speaker_name) <> '');

ALTER TABLE observations
    ADD COLUMN IF NOT EXISTS speaker_name TEXT
        CHECK (speaker_name IS NULL OR btrim(speaker_name) <> '');
