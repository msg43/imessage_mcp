-- 0013_saved_searches.sql
--
-- Saved searches outside any evidence case (search page, D14). Migration
-- 0012 stored a saved search only inside a case (`search_case_search`, with
-- the conversations marked reviewed in `search_case_review`). The owner can
-- now also keep a search on its own, list every saved search on one page,
-- give it a name and run it again.
--
-- The same table holds both, so the reviewed-conversation marks work the
-- same way for either:
--
-- - `case_id` may be NULL: a saved search on its own. Deleting a case still
--   deletes that case's searches (the foreign key's ON DELETE CASCADE) and
--   leaves these alone.
-- - `name`: an optional label the owner types ('' = none; the page then
--   shows the search text and filters).
-- - A search may have no words when it has filters (a search by filters
--   alone lists every message they keep): the text check becomes "at most
--   1,000 characters, and words or at least one filter".
-- - One search on its own per text and filters, as a case already has one
--   per text and filters (a partial unique index: the table's UNIQUE
--   constraint treats NULL case_ids as distinct).
--
-- Additive in effect: no row changes, every existing row satisfies the new
-- checks (0012 required 1 to 1,000 characters). NO explicit BEGIN/COMMIT,
-- matching 0001-0012: the runner wraps each file in its own transaction.

ALTER TABLE search_case_search ALTER COLUMN case_id DROP NOT NULL;

ALTER TABLE search_case_search
  ADD COLUMN IF NOT EXISTS name text NOT NULL DEFAULT '' CHECK (length(name) <= 200);

ALTER TABLE search_case_search DROP CONSTRAINT IF EXISTS search_case_search_query_text_check;
ALTER TABLE search_case_search
  ADD CONSTRAINT search_case_search_words_or_filters
  CHECK (length(query_text) <= 1000 AND (length(query_text) >= 1 OR params <> '{}'::jsonb));

CREATE UNIQUE INDEX IF NOT EXISTS search_case_search_standalone
  ON search_case_search (query_text, params) WHERE case_id IS NULL;

COMMENT ON TABLE search_case_search IS
  'A search the owner saved: to a case (case_id set) or on its own (case_id NULL), with its text '
  'and filters (people, from, to, att, sender, dir, kind, in) and an optional name.';
