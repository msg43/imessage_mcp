-- 0010_mcp_audit_rollup.sql
--
-- Aggregate audit rows for the MCP surfaces. `mcp_audit` keeps one row per
-- request whose token the gate judged, and per tool call; this table keeps
-- counts: how many requests ended one way (the same surface / subject_ok /
-- tool / error columns) over one period. Two writers, named by `source`:
--
-- 'unauthenticated'  The public server counts, in memory, the requests it
--                    turns away before judging any token -- no Authorization
--                    header, a malformed one, a bad Host or Origin, a client
--                    already throttled, the failure budget spent, the
--                    tokeninfo breaker open -- and writes one row per error
--                    code per interval (`mcp.public.rejection_write_interval
--                    _seconds`, 60 s by default). Internet scanners send
--                    exactly these requests; a row each meant a new Postgres
--                    connection and a write on the event loop per request,
--                    with no limit (QA review 2026-09-24). None of them
--                    carries a subject, so nothing AT-1 reads is lost.
--
-- 'retention'        `imsg mcp audit-prune` rolls `mcp_audit` rows older than
--                    `mcp.audit_retention_days` into one row per UTC day and
--                    outcome, then deletes them. Accepted public rows are
--                    never pruned: AT-1's standing check reads the whole
--                    history of `mcp_audit`. The same run merges
--                    'unauthenticated' interval rows older than the window
--                    into one row per day.
--
-- No subject column, on purpose: the unauthenticated counts have none, and
-- retention drops rejected subjects (other people's Google ids) once the
-- detailed window has passed. Rows are never deleted.
--
-- NO explicit BEGIN/COMMIT, matching 0001-0009: the runner wraps each file
-- in its own transaction.

CREATE TABLE mcp_audit_rollup (
  rollup_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  period_start  timestamptz NOT NULL,
  period_end    timestamptz NOT NULL,
  source        text NOT NULL CHECK (source IN ('unauthenticated', 'retention')),
  surface       text NOT NULL CHECK (surface IN ('local', 'public')),
  subject_ok    boolean NOT NULL,
  tool          text,
  error         text,
  request_count bigint NOT NULL CHECK (request_count > 0),
  CHECK (period_end >= period_start)
);

CREATE INDEX mcp_audit_rollup_period_idx ON mcp_audit_rollup (period_start);
