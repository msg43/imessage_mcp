-- 0008_attachment_location.sql
--
-- Every place a copy of an attachment might be read from, one row per
-- attachment and candidate location.
--
-- Owner decisions D12 (merges only add) and D13 (recall over purity:
-- "build whatever [fetcher] is needed to grab whatever additional messages
-- or attachments can possibly be added"). Before this table the index knew
-- one path per attachment (`attachment.source_path`), and S5a tried only
-- that path, on this machine. `attachment_source` records which source row
-- is which attachment but not the path each source recorded, so when the
-- live chat.db had no path and another Mac's did, that other path was
-- lost. The copies that exist elsewhere -- another Mac's Messages folder,
-- a backup drive, a NAS share -- had nowhere to be written down.
--
-- A row says "a copy of this attachment may be at `path` in `location`":
--
--   location       a host or drive code. A chat.db-recorded path is filed
--                  under the source that recorded it (a Mac's live source is
--                  named after the Mac: `mini`, `studio`). A listing or
--                  catalog match is filed under the code of the host or
--                  drive it describes (drive catalog codes such as
--                  `D-XXXXX`, `V-XXXXX`, `N-XXXXX`).
--   path           as that location names the file: a Messages path as
--                  chat.db records it (`~/Library/Messages/Attachments/...`),
--                  or a path relative to a drive's root.
--   match_quality  why this file is believed to be this attachment:
--                    recorded_path  a chat.db recorded this path for it
--                    guid_folder    the file sits in a folder named after
--                                   the attachment's GUID, under its name
--                    name_size      same file name and byte size, found
--                                   elsewhere. Flagged: 99.5% right when
--                                   measured, not certain.
--                  A name alone is never stored. The type has no value for
--                  it, so no row can ever make a name-only copy fetchable.
--   reported_by    every source, listing or catalog that reported the row.
--   byte_size,     what the location's own listing says about the file,
--   sha256         when known. chat.db's `total_bytes` is not stored here:
--                  it is not an observation of the file.
--   last_tried_at, the last fetch attempt and how it ended. `fetched_at`
--   last_outcome,  marks the copy that was materialized.
--   last_error,
--   fetched_at
--
-- Additive only: no existing table or column changes, and nothing reads
-- this table until the fetcher does. Extraction inserts rows and never
-- rewrites them; `imsg locate-attachments` inserts and refreshes the
-- evidence columns; the backfill writes only the attempt columns.
--
-- No `updated_at` column, so 0003's trigger does not apply.
--
-- NO explicit BEGIN/COMMIT, matching 0001-0007: the runner wraps each file
-- in its own transaction.

CREATE TYPE attachment_match_quality AS ENUM ('recorded_path', 'guid_folder', 'name_size');

CREATE TABLE attachment_location (
  location_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  attachment_id  bigint NOT NULL REFERENCES attachment(attachment_id) ON DELETE CASCADE,
  location       text NOT NULL CHECK (location <> ''),
  path           text NOT NULL CHECK (path <> ''),
  match_quality  attachment_match_quality NOT NULL,
  reported_by    text[] NOT NULL CHECK (cardinality(reported_by) >= 1),
  byte_size      bigint CHECK (byte_size IS NULL OR byte_size >= 0),
  sha256         text CHECK (sha256 IS NULL OR sha256 ~ '^[0-9a-f]{64}$'),
  first_seen_at  timestamptz NOT NULL DEFAULT now(),
  last_seen_at   timestamptz NOT NULL DEFAULT now(),
  last_tried_at  timestamptz,
  last_outcome   text CHECK (last_outcome IN ('fetched', 'staged', 'rejected', 'absent', 'refused', 'unreachable')),
  last_error     text,
  fetched_at     timestamptz,
  UNIQUE (attachment_id, location, path)
);
CREATE INDEX attachment_location_location_idx ON attachment_location (location, match_quality);

COMMENT ON TABLE attachment_location IS
  'Candidate copies of an attachment, one row per attachment, location and path '
  '(D12, D13). A name-only match is never stored.';
COMMENT ON COLUMN attachment_location.location IS
  'Host or drive code. A chat.db-recorded path is filed under the source that recorded it.';
COMMENT ON COLUMN attachment_location.match_quality IS
  'recorded_path | guid_folder | name_size. name_size is flagged: measured 99.5% right, not certain.';
COMMENT ON COLUMN attachment_location.last_outcome IS
  'fetched (materialized from this copy), staged (another host copied it into staging), '
  'rejected (size or hash differed), absent, refused (outside every allowed root, or not a '
  'regular file), unreachable (host or drive could not be read; retried on the next run).';
