-- 0005_extraction_run_upsert_breakdown.sql
--
-- Split what an extraction run actually did into rows inserted, rows
-- genuinely updated, and rows left untouched.
--
-- `extraction_run.messages_upserted` has always meant "message rows this
-- run processed", and it keeps that meaning — nothing reading it changes.
-- What it never distinguished is the thing that matters most after a
-- re-extraction: whether those rows were *written*. Before S2's upserts
-- were made state-idempotent (2026-09-17) every in-scope row was rewritten
-- with its own values, migration 0003's trigger moved `updated_at` on all
-- of them, and S4's `find_dirty_chats` consequently proposed rebuilding
-- the segmentation and embeddings for effectively the whole corpus. The
-- run row recorded that as "messages_upserted: 662683" — a number
-- indistinguishable from a run that corrected 662,683 messages.
--
-- Three nullable counters make the distinction durable rather than
-- something an operator has to have caught in a terminal:
--
--     messages_upserted = inserted + updated + unchanged
--
-- Nullable and unbackfilled on purpose. Runs recorded before this
-- migration genuinely do not know their own split, and writing zeros
-- would assert "nothing was written" about exactly the runs that wrote
-- the most. NULL says "not recorded", which is the truth.
--
-- No trigger interaction: `extraction_run` carries no `updated_at`
-- column, so 0003's catalog loop never attached one to it, and adding
-- columns here does not change that.
--
-- NO explicit BEGIN/COMMIT, matching 0001-0004: the runner wraps each
-- file in its own transaction.

ALTER TABLE extraction_run
  ADD COLUMN IF NOT EXISTS messages_inserted  bigint,
  ADD COLUMN IF NOT EXISTS messages_updated   bigint,
  ADD COLUMN IF NOT EXISTS messages_unchanged bigint;

COMMENT ON COLUMN extraction_run.messages_upserted IS
  'Message rows this run processed: inserted + updated + unchanged. '
  'Unchanged rows were not written at all, so their updated_at did not '
  'move and they drag no re-segmentation with them.';

COMMENT ON COLUMN extraction_run.messages_inserted IS
  'Rows that did not exist before this run. NULL for runs recorded before '
  'migration 0005, which did not track the split.';

COMMENT ON COLUMN extraction_run.messages_updated IS
  'Rows whose content genuinely differed and were rewritten — the only '
  'ones whose updated_at moved, and therefore the only ones that can make '
  'a chat dirty for S4. NULL for runs predating migration 0005.';

COMMENT ON COLUMN extraction_run.messages_unchanged IS
  'Rows already holding exactly the extracted values, left untouched. A '
  're-extraction of an unchanged corpus reports its whole corpus here. '
  'NULL for runs predating migration 0005.';
