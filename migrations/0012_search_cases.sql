-- 0012_search_cases.sql
--
-- Evidence cases on the search page (D14; design review of 2026-09-24,
-- change 4): the owner collects messages and files into a named case, adds
-- notes, keeps the searches that found them with which conversations were
-- reviewed, and downloads the case with exact citations.
--
-- Items are keyed by `message_key` and `attachment_key`, the opaque keys
-- derived from the Messages database's own GUIDs (`imsg.keys`), never by
-- `message_id` or a segment: those keys survive re-segmentation and
-- re-extraction. There is deliberately no foreign key to `message`: the
-- corpus only grows (owner rule D12), and a case must never block or be
-- broken by work on the core tables; the page shows an item whose message
-- is gone as missing.
--
-- `search_case.is_active` marks the case that "Add to case" adds to; at
-- most one case is active (a partial unique index). Nothing here is written
-- automatically: every row comes from an owner's click.
--
-- Additive only: four new tables and a trigger on the first, which carries
-- `updated_at` and gets the `set_updated_at` trigger 0003 gives every such
-- table. NO explicit BEGIN/COMMIT, matching 0001-0011: the runner wraps each
-- file in its own transaction.

CREATE TABLE IF NOT EXISTS search_case (
  case_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name       text NOT NULL CHECK (length(name) BETWEEN 1 AND 200),
  notes      text NOT NULL DEFAULT '' CHECK (length(notes) <= 20000),
  is_active  boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS search_case_one_active ON search_case ((true)) WHERE is_active;

DROP TRIGGER IF EXISTS set_updated_at_search_case ON search_case;
CREATE TRIGGER set_updated_at_search_case BEFORE UPDATE ON search_case
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE IF NOT EXISTS search_case_item (
  item_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  case_id        bigint NOT NULL REFERENCES search_case(case_id) ON DELETE CASCADE,
  message_key    text NOT NULL,
  attachment_key text,
  note           text NOT NULL DEFAULT '' CHECK (length(note) <= 5000),
  added_at       timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS search_case_item_message
  ON search_case_item (case_id, message_key) WHERE attachment_key IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS search_case_item_attachment
  ON search_case_item (case_id, message_key, attachment_key) WHERE attachment_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS search_case_search (
  search_id  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  case_id    bigint NOT NULL REFERENCES search_case(case_id) ON DELETE CASCADE,
  query_text text NOT NULL CHECK (length(query_text) BETWEEN 1 AND 1000),
  params     jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (case_id, query_text, params)
);

CREATE TABLE IF NOT EXISTS search_case_review (
  search_id   bigint NOT NULL REFERENCES search_case_search(search_id) ON DELETE CASCADE,
  thread_key  text NOT NULL,
  reviewed_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (search_id, thread_key)
);

COMMENT ON TABLE search_case IS
  'An evidence case on the search page: a name, notes, and whether "Add to case" adds to it.';
COMMENT ON TABLE search_case_item IS
  'A message (attachment_key NULL) or one attachment of it in a case, keyed by opaque keys that '
  'survive re-segmentation; no foreign key to message on purpose.';
COMMENT ON TABLE search_case_search IS
  'A search the owner saved to a case: its text and filters (people, from, to, att).';
COMMENT ON TABLE search_case_review IS
  'A conversation the owner marked reviewed for a saved search.';
