-- 0006_extraction_run_merge_mode.sql
--
-- Record which merge rule an extraction run applied, and what it did to
-- every table it wrote.
--
-- Owner decision D12 (2026-09-23): corpus merges only add. A run from any
-- database other than this machine's own live chat.db (a "seed": a
-- `--snapshot` file, another Mac's database, a recovery candidate) may
-- insert rows and fill empty values, never replace a non-empty one. Only
-- the live run may replace a value, and a message body changes only for a
-- strictly newer edit. `imsg.stages.extract.MergeMode` is the rule;
-- `merge_mode` records which one a run applied.
--
-- `upsert_counts` is the per-table report `imsg extract` prints: for each
-- table the run wrote, how many rows were inserted, filled (only empty
-- values changed), given a newer edit's body, replaced (a non-empty value
-- overwritten; only a live run may do this) or left unchanged. Before this
-- migration only the message counts were recorded (0005), and the
-- 2026-09-22 dry run showed why that is not enough: it reported 526 message
-- updates and nothing about the chat and attachment rows it would have
-- rewritten. With this column a real seed run can be checked against
-- "inserts and fills only" after the terminal output is gone.
--
-- Both columns are nullable and unbackfilled on purpose: runs recorded
-- before this migration did not know their mode or their per-table split,
-- and NULL says "not recorded", which is the truth.
--
-- No trigger interaction: `extraction_run` carries no `updated_at` column,
-- so 0003's catalog loop never attached one to it.
--
-- NO explicit BEGIN/COMMIT, matching 0001-0005: the runner wraps each file
-- in its own transaction.

ALTER TABLE extraction_run
  ADD COLUMN IF NOT EXISTS merge_mode text
    CONSTRAINT extraction_run_merge_mode_check CHECK (merge_mode IN ('live', 'seed')),
  ADD COLUMN IF NOT EXISTS upsert_counts jsonb;

COMMENT ON COLUMN extraction_run.merge_mode IS
  'D12 merge rule the run applied: live (this machine''s own chat.db; may '
  'replace a non-empty value) or seed (inserts and fills only). NULL for runs '
  'recorded before migration 0006.';

COMMENT ON COLUMN extraction_run.upsert_counts IS
  'Per table written: {"<table>": {"inserted", "filled", "newer_edit", '
  '"replaced", "unchanged"}}. A seed run must show replaced = 0 everywhere. '
  'NULL for runs recorded before migration 0006.';
