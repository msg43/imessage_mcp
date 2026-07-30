-- 0003_updated_at_enforcement.sql
--
-- Two corrections found while building against the schema, neither of
-- which changes any existing row.
--
-- (1) `updated_at` is enforced by the database, not by convention.
--
--     S4's dirty-chat detection re-segments a chat when its messages'
--     `updated_at` moves. S2's upsert does bump it today, so the
--     behaviour is currently correct -- but nothing in the schema
--     required it. That made a whole class of future silent failure
--     available: any code path that updates a row without remembering
--     to set `updated_at` would strand that chat out of
--     re-segmentation, with no error, no failing test, and stale
--     search results as the only symptom. The corpus would simply be
--     quietly wrong about recently edited conversations.
--
--     Enforcing it in a trigger makes the invariant true by
--     construction. This is the same reasoning as the mount gate and
--     the fail-closed auth boundary: prefer structure over discipline
--     wherever an omission would be silent.
--
--     Applied to every table carrying an `updated_at` column, via a
--     catalog loop rather than an enumerated list, so a table added by
--     a later migration inherits the guarantee automatically instead
--     of depending on someone noticing this file.
--
-- (2) `materialization_state` gains 'unsupported'.
--
--     AT-3's reconciliation report has an `unsupported` exception
--     category with no corresponding enum value, so that bucket could
--     only ever report zero -- which reads as "we checked and found
--     none" when it actually means "we can never find any". Adding the
--     value aligns the schema with AT-3's vocabulary.
--
--     NOTE: nothing populates it yet. S5a must set it when it meets an
--     attachment type it cannot materialize (Phase 2). Until then the
--     bucket is honestly zero rather than structurally zero, and that
--     distinction is the point of this change.

-- NO explicit BEGIN/COMMIT here, matching 0001 and 0002: the migration
-- runner wraps each file in its own transaction (and uses a SAVEPOINT
-- when already nested). A COMMIT inside the file would end the runner's
-- transaction out from under it and invalidate that savepoint.
--
-- `ALTER TYPE ... ADD VALUE` is permitted inside a transaction on
-- PostgreSQL 12+; the only restriction is that the new value cannot be
-- USED until the transaction commits. Nothing below uses it.

-- (1) ------------------------------------------------------------------

CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  -- Assigned unconditionally: a caller that supplies its own
  -- `updated_at` is exactly the case this trigger exists to override.
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;

DO $$
DECLARE
  target record;
BEGIN
  FOR target IN
    SELECT c.table_name
    FROM information_schema.columns c
    JOIN information_schema.tables t
      ON t.table_schema = c.table_schema
     AND t.table_name = c.table_name
    WHERE c.table_schema = current_schema()
      AND c.column_name = 'updated_at'
      AND t.table_type = 'BASE TABLE'
    ORDER BY c.table_name
  LOOP
    EXECUTE format(
      'DROP TRIGGER IF EXISTS %I ON %I',
      'set_updated_at_' || target.table_name, target.table_name
    );
    EXECUTE format(
      'CREATE TRIGGER %I BEFORE UPDATE ON %I '
      'FOR EACH ROW EXECUTE FUNCTION set_updated_at()',
      'set_updated_at_' || target.table_name, target.table_name
    );
  END LOOP;
END;
$$;

-- (2) ------------------------------------------------------------------
-- Postgres 12+ permits ADD VALUE inside a transaction; the new value
-- simply cannot be USED until this transaction commits. Nothing below
-- references it, so this is safe here.

ALTER TYPE materialization_state ADD VALUE IF NOT EXISTS 'unsupported';

-- (3) ------------------------------------------------------------------
-- Correct a misleading column comment from 0001 (D8.2).
--
-- 0001 documents `tapback.kind` as including `emoji:<char>`. The
-- upstream crate never emits that: it returns a BARE 'emoji' kind plus
-- a separate `emoji` field, which extraction combines at upsert. The
-- wrong comment is the kind of thing that sends a future reader
-- looking for a format that does not exist, or worse, writing a parser
-- for it.
--
-- Corrected HERE rather than by editing 0001, deliberately: applied
-- migrations are immutable, and the runner enforces that with a hash
-- check. Rewriting a shipped migration is precisely what that check
-- exists to catch, so the fix goes forward as schema metadata --
-- which is also more discoverable than a SQL comment, since it shows
-- up in `\d+ tapback`.

COMMENT ON COLUMN tapback.kind IS
  'One of: loved, liked, disliked, laughed, emphasized, questioned, emoji, sticker. '
  'NOTE (D8.2): the upstream crate emits a bare ''emoji'' kind alongside a separate '
  'emoji character field, which extraction combines at upsert -- it never returns the '
  '''emoji:<char>'' composite that 0001''s inline comment described.';
