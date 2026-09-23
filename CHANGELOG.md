# CHANGELOG

Dated milestone history, newest-first. Terse bullets explaining the
**why**, not just the what — link commit hashes where useful. Add the
entry **in the same commit(s) as the work** whenever something notable
lands: a schema/migration change, a significant feature, a fix batch, a
gate transition. Skip the genuinely trivial (typo/format/lockfile bump);
when in doubt, add the line.

This is a running document, not a one-time artifact — status must never
live only in a chat transcript or an assistant's session memory.

## 2026-09-23 — Every message is filed: rows with no chat link go into the chat their evidence names, or a holding chat

Owner decision D13: recall over purity. "I'd rather keep all the messages
in the active corpus." `chat.db` keeps some messages with no
`chat_message_join` row, and S2 logged each one
(`extract.message_without_chat`) and dropped it. A read-only survey of the
four sources the production index reads found 2,543 distinct such messages
in no chat of the index: 168 in Apple's "Recently Deleted", 11 with a 1:1
chat id that agrees with the sender, 1,967 carrying the id of a group chat
that no longer exists as a linked chat, and 397 with no chat evidence.

- **Filing rules, in order** (`imsg.stages.unlinked_filing`): the chat
  "Recently Deleted" names (`recoverable_join`, with `message.deleted_at`);
  the 1:1 chat a `ck_chat_id` names when its handle is the sender or the
  owner sent it (`ck_1to1`; created under Apple's own GUID when the index
  lacks it); the one indexed group a group-style id names
  (`ck_group_match`, filed automatically, accepting the measured misfiling);
  otherwise a holding chat per lost group id (`holding_lost_group`); and
  with no evidence at all, a holding chat per sender, one for the owner's
  own messages (`holding_sender`). `message.chat_evidence` records which.
  A 1:1 id that names someone other than the sender counts as no evidence.
- **Group ids are matched across sources.** Every run records each chat's
  `group_id` and `original_group_id` in `chat_group_id`, and a lost group is
  matched against every indexed chat. On the Studio's own snapshot none of
  its 21 lost group ids is carried by any of its own chats, so its share of
  the 688 rows that do match a group can only be matched through the ids
  other sources record.
- **A message moves only toward stronger evidence, never out of a real
  chat.** A message in a holding chat moves when a later run or source
  names a chat on stronger evidence, including a chat link. A message in a
  real chat never changes chat; stronger evidence for the same chat only
  raises `chat_evidence`. A move is filing, not a merge: it does not count
  as `replaced`, so a seed still shows `replaced=0`. S4 follows a moved
  message: `find_dirty_chats` reports both chats, and the destination drops
  the old segment before placing the message, whichever chat runs first.
- **Rescan without a flag.** These rows sit below every ROWID watermark, so
  each run also reads the snapshot's unlinked rows at or below it and
  selects those that still need work: missing from the index, movable out
  of a holding chat, or deleted without a delete date in the index. The
  selection runs before `imsg-dump`, which then decodes back only to the
  lowest selected row, and the next run selects nothing. Measured on the
  Studio: the candidate query takes 0.49–0.52 s warm (2.2 s cold) on a
  699,673-message snapshot with 2,392 candidates; re-recording 21,437
  group-id pairs takes 48–53 ms; the weakly-filed lookup 1.3–1.8 ms
  through its partial index. A run that finds work decodes the file once:
  8.2–8.5 s for 700,000 rows.
- **Serving.** Under `full` scope, on the local surface and the public one,
  holding chats and deleted messages are searchable; a deleted message
  renders as `[deleted]`, and a holding chat's header says it is unfiled.
  Under `allowlist` scope holding chats are denied by a new rule in the
  shared eligibility module (`holding-chat`), and deleted messages never
  show. Export leaves deleted messages out as it leaves out unsent ones:
  `exportable_message_sql` states both exclusions once, and export, the
  allowlist re-render, the conversation window, `list_people` and the
  attachment gate all use it. The unclassified-threads report skips holding
  chats, which no allowlist can make eligible. The renderer version is not
  bumped: every existing row renders byte for byte as before.
- **AT-2** counts a dated message with no chat link in a reference built
  from a `chat.db` file, since extraction now lands it. An undated one
  cannot be filed (`sent_at` is NOT NULL); extraction counts it as
  `skipped_without_date` instead of failing the run.
- **The chat link is read deterministically**: the lowest chat ROWID that
  exists in the snapshot, as `imsg-dump` already did. A link to a chat row
  the file lacks counts as no link.
- **Migration 0007** (additive): `chat.unfiled_key` (unique, set only on
  holding chats), `message.chat_evidence` (NOT NULL, default
  `chat_message_join`, which is how every existing message was filed; no
  table rewrite), `message.deleted_at`, the `chat_group_id` table, and a
  partial index for the weakly-filed lookup. Run `imsg migrate` before this
  build extracts anything.
- **Tests:** 50 new (28 extraction and 4 serving integration tests, 17
  unit tests, 1 AT-2 test), each run first against core 14ee55c, where all
  50 failed or could not import. Four existing test files changed with the
  schema and the rule: the table count, the per-table report, the AT-2
  reference fixture, and the CLI report test, which now also checks the
  unlinked line; the shared `chat.db` fixture gained the columns and table
  the rules read. Full suite 2,002 passed against a scratch Postgres 17,
  rebased on the twin merge (the next entry down; 1,919 on 14ee55c in the
  same environment); ruff, mypy strict and the DDL lint clean.

## 2026-09-23 — One person per sender again: `imsg identity merge-filtered-twins`

Owner decision D13, item 4. iOS tags a filtered sender's handle `(filtered)`
or `(smsft…)`. `normalize_handle` has stripped the tag since 2026-08-15
(`4122c51`), but S3 resolves only source handles that have no resolution
row, so every source handle resolved before that fix kept a tag-carrying
canonical handle and a stub person of its own. Measured read-only on the
production index: 3,448 such handles on 3,448 persons, every person created
by the first identity import on 2026-08-15, about four hours before the fix;
3,357 persons reached only through `(filtered)` handles. A person-scoped
search found only part of what those senders sent.

- **`imsg identity merge-filtered-twins [--dry-run] [--show-names]`**
  (`imsg.stages.identity_filtered_twins`). For each tagged source handle
  whose canonical handle is not what `normalize_handle` gives it today, it
  repoints the source handle to the clean twin handle and merges the two
  persons with `merge_persons`. Where no twin exists, it rewrites the tagged
  handle to its clean value and renames a stub still named after the tagged
  value. The emptied tagged handle is then removed. Left on the kept person
  it would stop `rematch-stubs` naming that stub, and `export-overrides`
  would take it for a hand merge. The twin's person is kept unless only the
  tagged side carries a curated name. Two different curated names, and the
  owner, are refused and listed, never merged. No message, tapback, chat or
  source handle is deleted. Every merge and rename marks its chats for
  re-segmentation. It runs as one transaction, and `--dry-run` rolls it back.
- **`merge_persons` no longer copies the absorbed person's allowlist row**
  onto a kept person that has none. That allowlisted everything the kept
  person ever sent. The kept row now stands, narrowed by the absorbed row
  when both exist.
- **The tag pattern also accepts whitespace inside and between tags and any
  `smsft_` suffix, and never strips a value down to nothing.** None of these
  forms is in the production index today: its 3,414 `(filtered)` and 113
  `(smsft…)` source handles all match the old pattern.
- **Forecast for the production dry run** (read-only SQL that imitates the
  phone parser, so approximate): 2,944 persons merged, 494 handles
  rewritten, 3 pairs refused for different curated names, none involving the
  owner.
- **Tests:** 33 new. Of those, 20 failed on the old code: 13 because the
  command or a helper did not exist, 7 on assertions (4 tag forms left
  unstripped, a bare tag emptied, and the 2 allowlist rules). The other 13
  pin behavior that was already right. The repair tests also caught six
  deliberate breaks of the new code. Full suite 1,952 passed. A synthetic run
  at production size (627,500 messages, 3,000 merges) took 35 s.

## 2026-09-23 — Corpus merges only add: a seed inserts and fills, and never replaces a value

Owner decision D12: "Always want to merge and maintain the fullest corpus
of chat.db and attachments, not overwrite and lose rows." Each column's
rule (`Merge` in `imsg.stages.extract`) decided only whether an incoming
value was evidence, and under `PRESENT` any non-NULL value was — `''` and
0 included — so whichever source ran last won. A dry run on the production
host of a recovery candidate built on an older merged corpus would have
added 608 messages, and would also have put the older values back over 526
messages, 32,971 attachment rows and 365 chats, one group name blanked to
`''`. It printed `inserted=608 updated=526 unchanged=664178` and nothing
about the chats and attachments. Nothing was written; dry runs roll back.

- **Seeds only add.** `run_extract` takes `merge_mode`: `MergeMode.LIVE` or
  `SEED`, default `SEED`. Only S1's copy of this machine's own
  `paths.live_chat_db` is live (`merge_mode_for_source`: same resolved path
  or same inode). `imsg sync` decides from the database S1 copied, so a
  configured source that points at another Mac's copy is a seed.
  `imsg extract --snapshot` is always a seed. `imsg extract` without
  `--snapshot` is live only when every configured source is the live
  database, because all of them share `snapshots/snapshot.db`. A seed fills
  empty values and never replaces a non-empty one, `sent_at` and
  `is_from_me` included; a `POSITIVE` flag's `true` over a stored `false`
  counts as a fill.
- **`''` and 0 are never evidence**, from any source. Each column declares
  which of its values say nothing (`Empty`): NULL or `''` for names, paths,
  types and bodies; NULL, 0 or less for `byte_size`; NULL or `unknown` for a
  chat's service; `false` for the five `POSITIVE` flags; only NULL for
  timestamps, ids and the two asserted columns.
- **The live run still applies genuine changes**, such as a renamed chat or
  an attachment the Messages app moved.
- **Bodies follow edit recency, and edit history only grows.** A body is
  filled when empty and replaced only by a strictly newer `date_edited`,
  from any source; a same-or-older edit never replaces it, the live run
  included. `date_edited` moves only with its body, so an edit whose text
  did not decode cannot lock that text out. The body a newer edit replaces
  is appended to `message_version` when the history does not already hold
  it. A stored version is never replaced (a different text at the same
  position is a different decode); an undecodable `''` version is filled.
- **Every table is reported.** `imsg extract` and `imsg sync` print one line
  per table with `inserted`, `filled`, `newer_edit`, `replaced` and
  `unchanged`, for all eleven tables S2 writes; the first line is
  unchanged. A seed must show `replaced=0` on every line. The split is
  worked out per column inside the same statement, from the row before the
  write and the row after it, using each column's `Empty` kind.
- **Migration 0006** (additive): `extraction_run.merge_mode` and
  `extraction_run.upsert_counts` (jsonb, the per-table split), so a real
  seed run can be checked after the terminal output is gone. Run
  `imsg migrate` before this build extracts anything; without the columns
  the first extraction fails.
- **Changes S4 could not see now mark the chat.** S4 re-segments a chat only
  when one of its messages' `updated_at` moves. A chat's name or kind (in
  every segment header), an attachment's name or type, a new attachment
  link on an existing message, a tapback and an edit-history version are
  all rendered through a message but stored elsewhere, and changing them
  moved nothing, so segments kept the old text. Extract now marks those
  messages at the end of each run (`messages_marked_for_resegmentation`).
  This applies to the live run too: a tapback on an older message, or a
  group rename, now re-segments — a rename re-segments the whole chat, as
  an identity rename already does.
- **Cost:** about 1.3× the database time per extracted row, measured as
  9.1–9.5 s against 7.0–7.3 s per 20,000 messages on a synthetic corpus on
  the development machine.
- **Tests:** 38 new, each run first against the previous code, where all
  failed — on overwrites, blanked values and unmarked chats, or on the
  counts and mode not existing. Two existing tests asserted the old rule
  and were rewritten: a body change with no newer edit time, and a second
  source's rename, service and attachment changes landing from a seed.
  1,919 passed, 0 skipped, against a scratch Postgres 17.

## 2026-09-23 — Public `allowlist` scope serves exactly what export would ship

SPEC §10.3a says the public MCP surface under `allowlist` scope applies
"the same eligibility predicate" as export. It had its own rule instead:
`imsg.retrieval.access` checked only a chat's current `chat_participant`
rows, and counted a chat with no participants as eligible. Export's rule
(`imsg.export.eligibility`) also denies a chat when a former member, a
message sender, a raw participant handle or a tapback sender is
unresolved or not allowlisted. Nothing was exposed, because the
production allowlist had no rows; the first row would have let the public
surface serve threads export refuses.

- **One rule, one implementation.** `imsg.retrieval.access` calls
  `imsg.export.eligibility.eligible_chat_ids` instead of keeping a copy.
  The deny rules are declared once (`_DENY_RULES`), and both
  `compute_chat_eligibility` (with reasons, for the review report) and
  `eligible_chat_ids` build their SQL from them. Both take `among=` to
  evaluate only some chats.
- **Evaluated once per request.** The rule is not written inline in the
  candidate queries: a search runs five of them, and one evaluation of
  the message rules for the largest chat (88,540 messages) takes 74 ms on
  the production host. Each request evaluates the rule once and filters
  candidates on `chat_id = ANY(<eligible ids>)`; with few eligible chats
  the planner switches to an exact scan by chat (1 ms for 53 chats).
  `eligible_chat_ids` runs the participant rules over every chat, then
  the message rules only over the chats left: 17 ms with today's empty
  allowlist, 12-31 ms for allowlists leaving 2 or 53 eligible chats,
  99-192 ms for 1,061 chats or with every person allowlisted (production
  host, read-only, allowlists simulated, 2026-09-22). There is no cache:
  an identity merge that repoints `message.sender_person_id` changes
  eligibility, and no cheap cache key would see it.
- **Attachment text passes export's separate gate.** Under `allowlist`,
  `get_attachment_text` returns text only through a parent segment where
  every non-unsent message linking the attachment has a sender with
  `attachments_allowed` (SPEC §11.2, `compute_attachment_eligibility`).
  The same gate now covers the snippets inside `get_conversation` lines
  and `search_messages` text, which carried every attachment's caption
  and OCR text; a gated attachment renders as `[attachment withheld]`.
  Search results under `allowlist` are re-rendered from their rows
  instead of returned from `segment.rendered_text`, and the reranker
  scores the re-rendered text. The placeholder is a new
  `AttachmentSnippet.withheld` field that segmentation never sets, so no
  stored rendering changes and `RENDERER_VERSION` stays at 1.
- **No unsent messages or edit history under `allowlist`**, whatever
  `policy.*` says. These are the exclusions §11.2 imposes on export
  unconditionally (D1).
- **People.** `list_people` under `allowlist` lists only people who
  appear in an eligible chat, counting only their messages there. Before,
  it listed every text-allowed person with counts over all their
  messages. Person filters resolve within the same set, so an exact name
  outside it is `PERSON_NOT_FOUND` instead of an empty success that
  confirms the person exists. `include_handles` is refused off the local
  surface.
- **Same `NOT_FOUND` text for unknown and denied threads.**
  `get_conversation` gave different messages for the two, and the public
  server returns error text verbatim.
- **An empty allowlist answers at once.** With no eligible chat,
  `search_messages` returns an empty result without embedding the query
  or scanning an index.
- **Tests.** `tests/test_retrieval_allowlist_parity_integration.py`, 20
  cases through the service's public methods. Against the code before
  this change 18 fail and the 2 controls pass; after it, all pass. The
  parity case builds every deny reason in one database and checks that
  the chats `get_conversation` and `search_messages` serve equal
  `eligible_chat_ids`, and that the per-chat and whole-corpus evaluations
  agree. Full suite: 1,881 passed against a scratch Postgres 17 with
  pgvector 0.8.6.
- **Not yet measured:** whole-query p95 under `allowlist` on the
  production host, which needs this build deployed and allowlist rows to
  exist. `scripts/bench_retrieval_latency.py --scope allowlist` times it,
  with the new `scope` and `render_scoped` stages.

## 2026-09-18 — the last two command surfaces: `imsg backup`, and the AT-1 probe's CLI

Both were found the same way — by reading what the deployment guide and
the rendered LaunchAgents *invoke* and checking it against what the CLI
*provides*. `install-agents` rendered a `com.imsgindex.backup` job
running `imsg backup --config <path>` at 04:00 daily, and the command
did not exist: installing the agents would have created a job that
failed every night, silently, forever. The deployment guide makes
`imsg mcp public --probe` the gate that must pass before any corpus is
reachable from the internet, and `run_auth_probe` was written and tested
with no way to run it. Every command an installed agent invokes now
exists.

- **`imsg backup` — scope decided explicitly, not by default.** In:
  the Postgres dump (everything the pipeline derived lives only there,
  and it is the one component that can suffer *logical* corruption,
  which is the only thing same-device copies defend against) and the
  FTS5 sidecar (SPEC §5.5 names it; §14 allows copying it only after a
  checkpoint + integrity check, so those are enforced). Out: the
  **attachment cache (~147 GB)** — 14 nightly copies is over two
  terabytes on the same volume as the original, it is a content-addressed
  cache `imsg backfill-attachments` rebuilds, and a blob store does not
  rot the way a schema does; the **model directory**, whose contents are
  public weights already pinned by repo + immutable revision in
  `models/manifest.lock.yaml`; and `ops/`, which is small and
  irreplaceable but outside what §5.5 scopes to this job — flagged in
  the command's output rather than silently added. The honest caveat
  ships with the decision and is printed on every run: anything iCloud
  has purged exists **only** in `attachments/`, and excluding it is not
  a claim that it is safe.
- **A dump, not a copy of a running data directory.** `pg_dump -Fc`
  against the live instance; a running cluster's files are a process
  mid-flight, not a backup.
- **Verification reads the whole archive, because the obvious check
  cannot fail.** Measured 2026-09-18: a custom-format dump **cut in
  half** still lists every one of its TOC entries under `pg_restore
  --list` and exits 0 — the table of contents precedes the data blocks.
  `pg_restore -f /dev/null` decompresses every block and caught a half
  truncation, a **one-byte** truncation, and a flipped byte 2 kB from
  the end. Both run; the TOC listing is kept only for what it is good
  for, naming the expected tables. An integration test performs a real
  dump, truncates it, and asserts the refusal — and a second test
  restores a real dump into a fresh database and counts the tables that
  arrived, rather than trusting an exit status.
- **A refusal the build machine already needed.** `pg_dump` aborts
  against a newer server *after* creating its output file, leaving zero
  bytes that look like a backup. Homebrew puts `postgresql@16`'s
  `pg_dump` first on `PATH` while SPEC §5.3 pins the instance at
  `pg17/` — measured on this machine, 16.13 against 17.9 does exactly
  that. Major versions are now compared before the subprocess starts and
  the error names both versions and the fix.
- **Retention deletes only what it can prove, and the rule is written
  down.** A directory under `backups/` is a candidate only if it is a
  real (non-symlink) direct child, its name matches the exact set
  pattern, it holds a `MANIFEST.json` that parses, declares the current
  format and says `"complete": true` — written **last**, so its presence
  *is* the completeness evidence — and it is not among the newest 14.
  `keep` is clamped to ≥ 1, so the newest set can never be a candidate
  even with `--keep 0`. Everything else (a half-written set, a
  `.incomplete-*` staging directory, an operator's own file, a symlink)
  is counted, reported and left alone. Twenty-seven tests pin the
  boundaries, including exactly-14, 14+1, and a symlink pointed at a
  directory elsewhere on the volume.
- **Nothing partial is ever promoted.** Each run writes into
  `backups/.incomplete-<uuid4>/` and promotes it with one `rename` after
  the manifest lands. The cleanup catches `BaseException`, not
  `Exception`: launchd killing the 04:00 job is the likeliest failure
  there is, and a bare `except Exception` would leak a directory on
  precisely that path. Two runs in one day produce two sets and neither
  can write into the other's directory.
- **Fixed while testing it:** a sidecar that is not a SQLite database at
  all escaped as a raw `apsw.NotADBError` traceback rather than one
  `imsg: …` line — `apsw` is lazy and does not touch the file until the
  first statement, so the failure surfaced at the integrity check, not
  at open.

- **`imsg mcp public --probe` — token handling first, because that is
  where this goes wrong.** AT-1 step 5 says the tokens are never written
  to disk. `--owner-token <value>` would have broken that twice over:
  shell history, and `argv` readable by every process on the host via
  `ps -ww` — on a machine that by design has a public tunnel attached.
  The command therefore takes `--owner-token-ref` / `--foreign-token-ref`
  holding `keychain:<item>` or `env:<VAR>`, parsed by the existing
  `imsg.config.secrets.SecretRef` rather than a new convention, so a
  literal is refused *structurally*. The refusal prints the
  `security add-generic-password … -w` form that prompts instead of
  taking the value on the command line. No refusal message ever contains
  a token, and `ProbeTokens.__repr__` is overridden so a traceback
  cannot leak one either; a test asserts both.
- **Wired, not run.** The probe needs live OAuth tokens and a live gate,
  so nothing here has been executed against Google, and nothing in the
  test suite can be: every refusal is raised by
  `check_probe_preconditions`, a pure function over config and two
  reference strings that runs to completion *before* the audit sink,
  the gate or `GoogleTokeninfoIntrospector` is constructed. The CLI tests
  drive each refusal with `build_public_gate`, `PostgresAuditSink`,
  `run_auth_probe` and `connect` replaced by landmines, and one test
  replaces `socket.socket` itself.
- **The verdict is unambiguous, because of what it gates.** Distinct
  exit codes: `0` pass, `1` fail (a breach), `2` invalid, `78`
  (`EX_CONFIG`) the probe never ran. `INVALID` is deliberately not
  folded into either neighbour — "proved nothing" and "is safe" are the
  confusion AT-1 exists to prevent, and the rendered output says *NOT a
  pass* in those words. Each verdict prints the D6 next action: pass
  permits `scope: full` with an ops record, anything else pins
  `allowlist` and requires a fresh owner decision. `--probe` does not
  require `mcp.public.enabled`, since AT-1 step 0 runs while the real
  server stays disabled.

- **`imsg status` now reports the unclassified-thread count (SPEC
  §11.5).** A previous pass declined this because `status` "never opens
  a query connection". That premise was simply false — `check_postgres`,
  `check_buffer_pool` and `check_enrichment_yield` each already open one
  and each already reports failure as a reason string rather than
  raising. The real hazard was different: unlike those three, this query
  aggregates over `message`, so a large corpus under contention could
  make a health check slow rather than wrong. It runs under a 5-second
  `statement_timeout` on its own connection, gated on reachability, with
  every failure mode returning `None` plus a reason. Proved with a real
  `ACCESS EXCLUSIVE` lock held from a second session: `status` still
  exits 0, still answers every other field, and reports the count as
  unreadable. The stale note claiming the field was unwired is gone.

- **Tests: 111 new (1850 total, 0 skipped against a scratch
  PostgreSQL).** Weighted toward refusals by design — 27 retention
  boundary tests, 26 backup refusals, 12 backup integration tests, 40
  probe tests, 6 status tests. A session-scoped tripwire confirmed none
  of them opens a connection to port 5433: a real `imsgindex` instance
  listens there on a deployment host, and one draft test reached it
  before the tripwire caught it.

## 2026-09-18 — `imsg export` is a real command surface, and cannot reach Google without a credential you named

The export gate has been complete as a library since Phase 7 groundwork
— eligibility, planner, review, push, purge, the unclassified report,
all integration-tested — behind a CLI that raised
`StageNotImplementedError` and exited 1. So the one stage whose failure
mode is "private content enters a corporate index" was the one stage no
operator could run, the deployment guide documented commands that did
not exist, and `install-agents` already rendered a Monday-08:00 job
invoking `imsg export unclassified-report`, which would have failed
every week from the day the agents were installed.

- **Five commands, each shaped to refuse.** `export plan`, `approve
  <run-id>`, `push <run-id>`, `purge-person <id-or-name>`,
  `unclassified-report`. Refusals are first-class and tested, not
  incidental: an unknown run id, an unapproved push, a drifted push, a
  missing credential, an allowlist with zero rows, a purge naming
  nobody. Each prints one `imsg: …` line and exits 1 — never a
  traceback, because from this surface a traceback leaves the operator
  unable to tell whether anything was uploaded.
- **`export.gcp_credentials`, with no default, ever.** A `keychain:` /
  `env:` reference like every other secret (SPEC §6). `push` checks it
  **before opening the database**, and imports the Google client
  libraries only after that check passes — so a stock checkout has no
  code path to a network call, and `import imsg.cli` pulls in no Google
  client at all. A test asserts exactly that, in a subprocess, so a
  future refactor that hoists the import to module scope fails loudly
  instead of quietly putting a network-capable client in every `imsg`
  invocation.
- **`--dry-run` where the stage has writes worth rehearsing.** `plan`
  reconciles and reports without staging a byte; `push` runs every
  verification the real push runs and stops — it builds no transport, so
  the rehearsal is provably network-free and needs no credential;
  `purge-person` applies the real revocation inside a transaction it
  then rolls back, so the numbers come from the code that would run
  rather than a parallel estimate; `unclassified-report` counts without
  writing. `plan`/`push` share their implementations with the real
  paths (`preview_plan`, `verify_push_preconditions`) — a rehearsal that
  checked less than the performance would be worse than none.
- **Fixed: a purge plan told the operator to do something the push then
  waived.** `plan_export` computed the review banner by calling
  `compute_approval_requirements` without its `mode`, so every purge
  plan printed "OWNER APPROVAL REQUIRED" while `push_export` — which
  does pass the stored mode — correctly exempted the same run under
  D9.3. Conservative, and wrong in a way that matters: §11.4 calls that
  report *the actual control*, and a report demanding a ceremony the
  push skips teaches the operator to stop believing the report. Purge
  plans now say they are exempt, and say that every drift check still
  applies.
- **Push is deliberately not wrapped in a transaction.** Its connection
  is autocommit, so each item's outcome is recorded as it happens.
  Wrapping would mean a crash after a successful upload rolls back the
  `export_document` row recording it — leaving a document in the
  corporate store that this system's reconciler cannot see and
  `purge-person` therefore cannot delete. Redundant re-uploads on retry
  are cheap and idempotent; an invisible document is not.
- **Still unverified, and the reason this entry does not claim more.**
  Every one of these commands was exercised against a live scratch
  PostgreSQL and `FakeTransport`. **No part of the real GCS / Discovery
  Engine transport has ever run against a live API** — not one upload,
  import, delete, or absence check — and nothing in the test suite is
  permitted to make it. That remains a Phase 7 deliverable gated behind
  AT-5.
- **Survey, so the next gap is not a surprise.** No
  `StageNotImplementedError` stub remains anywhere in the CLI, and a
  test now walks the whole command tree asserting none reappears. Two
  documented commands still do not exist: **`imsg backup`**, which the
  daily 04:00 LaunchAgent invokes (SPEC §5.3/§14), and **`imsg mcp
  public --probe`**, the AT-1 "second-account test" the deployment guide makes a
  precondition of exposing any corpus — its library half
  (`imsg.mcp.probe.run_auth_probe`) is built and tested, but a real run
  needs two live OAuth tokens, so it was left unwired rather than
  half-wired.

## 2026-09-18 — The public MCP surface warms its models at start, instead of loading them inside the first request

`imsg mcp local` has warmed in the background since 2026-09-17;
`imsg mcp public` built the same providers and then served with them
cold, so the first request after every restart paid the whole model
load. Restarts are the normal case, not the exception — the launchd
agent is `KeepAlive`, so every crash is followed by another cold load —
and the point of this surface is that a hosted assistant asks it a
question and integrates the answer, which a first response measured in
half-minutes does not support. Measured here on the Studio, against the
live config with the surface on loopback and no tunnel: an unwarmed
first query took **29.1 s** (then 0.93 s, 0.75 s); with the warm-up the
transport accepted connections **1.17 s** after process start, reported
ready at **37.3 s**, and the first query then took **1.87 s** (then
1.18 s, 0.96 s). The load did not get faster — it moved off the request.

- **Warmed at process start, on the model thread.** `mcp public` now
  builds a `ModelThread` and hands it to both `RetrievalService` and a
  `BackgroundWarmUp` it starts just before `uvicorn.run`. The shared
  thread is not optional: MLX gives each OS thread its own default GPU
  stream, so a model prepared on the warm-up's thread cannot be
  evaluated on the event loop's — warming in the background without it
  would have made queries fail rather than fast.
- **A request that arrives first waits, then gets a retryable error.**
  The four retrieval tools wait for the warm-up with `anyio.sleep` (the
  event loop keeps serving meanwhile) for at most 20 s, then answer
  `WARMING_UP` with an estimate; a failed warm-up answers
  `WARM_UP_FAILED` on every call, so a process that came back up with
  half its models loaded says so instead of serving. Both keep their own
  code in `mcp_audit` rather than collapsing to `INTERNAL`. Unknown
  tools and schema violations are still answered without waiting.
- **20 s, not the local surface's 90 s.** A public call is an HTTP
  request held open across a Cloudflare tunnel: the edge's Proxy Read
  Timeout is a documented 125 s and is configurable only on Enterprise
  zones, and no `cloudflared` `originRequest` setting is a response-read
  timeout. Gemini Enterprise's own per-tool-call timeout is not
  documented anywhere, so the bound has to be small enough that no
  plausible one binds first. Past the edge's limit the client gets an
  HTML error page instead of JSON-RPC — a transport failure rather than
  a tool error it can act on.
- **Readiness without an unauthenticated endpoint.** The transport
  requires a bearer token on every request including `initialize` and
  `tools/list`, and a health route would be the one hole in the only
  access control this project has — so none was added. Instead the
  warm-up's phase goes to stderr (launchd captures it to
  `logs/imsgindex-mcp-public.err.log`) and to
  `run/mcp-public-warm-up.json`, which `imsg status` reports as
  `mcp_public_warm_up`. The file records the publishing pid and the
  reader checks it is still alive, so a `ready` left behind by a process
  that died reads as `not_running`, never as a warm server.
- **`BackgroundWarmUp` gained an optional `on_status` callback** (the
  readiness publisher), invoked at every phase and step change and — at
  the two terminal transitions — before `notify_all`, so anything that
  can observe `ready` finds the file already saying so. Like the log
  callback it can never change the outcome.

## 2026-09-18 — A re-segmentation run now has an end, and stops rewriting segments that did not change

Measured on the live instance: 1,582 genuinely changed messages caused
236,300 messages to be re-segmented and 32,473 segments to be deleted
and re-inserted. Every re-inserted segment is a *new* `segment_id`, so
its `segment_embedding` row went with the old one by FK cascade and S6
re-embedded all 32,473 — a multi-hour stage for a few thousand edits.
Two independent causes, both fixed here.

- **The rebuild had no end.** `find_dirty_chats` reported only the
  *earliest* changed point, so one edit rebuilt every session from there
  to the end of the chat. It now reports a `DirtyChatSpan` (earliest and
  latest), and `compute_recompute_end` stops the run at the first
  persisted session starting more than one `session_gap_hours` after the
  last change. That session and every later one are provably unaffected:
  edits and identity changes never move `sent_at`, deletions only widen
  gaps, and every added message is at or before the latest change, so
  nothing can reach across a gap wider than the threshold. The function
  carries the full argument. On the live corpus the currently-dirty chat
  goes from 10 sessions/10 segments rebuilt to 1 and 1.
- **The span's end deliberately overshoots the specified fix.** The
  `changed` arm takes `GREATEST(m.sent_at, session.ended_at)` rather
  than the message's own `sent_at`, so the range always reaches the end
  of the session *currently* holding a changed message. `message.sent_at`
  is `Merge.ASSERTED` in extract — a second source can rewrite it — and a
  timestamp that moved backwards would otherwise leave its old session
  beyond the bound and stale forever.
- **Identical segments are left alone.** `SegmentationRunReport.
  skipped_unchanged` existed as a field and was never assigned. A
  recomputed segment that reproduces the stored row — `stable_key`,
  `seg_config_hash`, `rendered_sha256`, span, `message_count`,
  `token_count`, `topic_label`, `seq_in_session`, and the exact ordered
  `segment_message` membership — is now skipped entirely: no DELETE, no
  INSERT, no `search_index_event`, and the `segment_id` survives, so its
  embedding and any `export_document` pointing at it survive too. A
  session row whose `started_at` still matches is reused (its `ended_at`
  UPDATEd if the tail grew), so per-segment reuse works inside a session
  that changed.
- **Reuse cannot defeat D4's freeze.** `seg_config_hash` is an input to
  every `stable_key` *and* is compared on its own, so a config change
  matches nothing and rewrites every segment — and because a session
  with no surviving segment is deleted and re-inserted whole,
  `session.gap_hours` can never be left describing a threshold its
  segments no longer use.
- **Reuse cannot stall the run, either.** A segment holding a message
  with `updated_at > segment.created_at` is never reused even when it
  re-renders identically. Reuse keeps the old `created_at`, which is
  exactly what `find_dirty_chats` compares against, so skipping such a
  segment would leave the chat dirty forever and re-run the recompute
  every night without ever clearing it.
- **Consequence worth knowing:** `imsg segment --rebuild --chat <id>`
  under an unchanged config is now a no-op rather than a full rewrite.
  It still repairs any row that disagrees with what the config says it
  should be; it no longer repairs a `rendered_text` corrupted out-of-band
  in a way that left `rendered_sha256` matching.
- **Tested against the pre-fix revision first.** Seven of the ten new
  database-gated tests fail on `main` for the stated reason (segments
  rewritten, the boundary provider shown sessions past the bound,
  `skipped_unchanged == 0`); the other two assert behaviour that must
  *not* change. On a 2,400-message / 50-session / 400-segment fixture
  with one early edit: 2,304 -> 48 messages recomputed, 384 -> 1 segments
  rebuilt, 768 -> 2 index events.

## 2026-09-18 — The incremental frontier could jump over new messages, making them permanently unsegmentable

`compute_recompute_start` returns the timestamp a re-segmentation run
re-fetches from (`_fetch_messages_from` uses `sent_at >= T`), and two of
its three return paths could return a `T` *later* than the earliest
changed message. When that happens the changed rows are not merely
skipped on that run — they are unreachable forever: they never enter a
segment, so `find_dirty_chats` keeps reporting the chat, and the next run
computes the same overshooting frontier. No retry, no `--rebuild` of
another chat, and no amount of waiting recovers them.

- **The hole between two sessions.** After a sealed session and before
  the next persisted one, the function returned
  `existing_sessions[i + 1].started_at` unconditionally. Messages that
  arrive in that hole — a backfill or a recovered-history import lands
  them routinely — sit before that timestamp and behind the fetch.
  Observed on the production instance: one chat, 8 eligible messages with
  real text and resolved senders, `earliest_changed_at` 43m46s *before*
  the frontier the function returned, the chat dirty and staying dirty.
- **A change predating all history** hit the same shape through the
  fall-through `return existing_sessions[0].started_at`.
- **The fix is a clamp, and the function now has a single exit** so a
  future return path cannot miss it: `min(frontier, earliest_changed_at)`.
  The candidate the gap scan produces is only ever the relaxation "you
  may safely skip everything before this persisted session" — it says
  nothing about where the changed rows are. Re-fetching earlier than
  strictly necessary costs work; re-fetching too late loses messages.
  Behaviour for a change inside or after the last session is unchanged
  (the candidate is already <= the change there), so today's incremental
  scoping does not regress — the five pre-existing frontier tests pass
  untouched.
- **Tested at both levels.** A Postgres integration test builds the hole
  from fixtures (sealed session, gap, later session, two messages landing
  between) and asserts they end up in exactly one segment while the
  sealed session's segment is left alone; against the unfixed function it
  failed with the mechanism named — the messages were never fetched, and
  `segment_message` held nothing for them. Unit tests cover the hole, the
  predates-everything path, a change inside a middle session, and a
  property sweep asserting the frontier is never later than the change
  across every arrangement of 0-4 sessions and three gap settings.

**Still open (not in this change):** the rebuild's *end* is the end of
the chat, not the last affected session — `_fetch_messages_from` takes no
upper bound — so a change early in a long chat re-segments everything
after it. Tracked separately; this commit only stops messages being lost.

## 2026-09-17 — The nightly enrichment window: one copy of the shared 35B, one PE-Core tower per process, bounded MLX caches, and enrichment that yields to searches

Enrichment runs 01:00–07:00 while the MCP server is `KeepAlive`, and the
overlap was measured on the production host and does not fit: 80.4 GiB of
demand on a 64 GB box, critical memory pressure, 37.6 GiB of swap, 218 GiB
paged in eleven minutes, and search p95 at **8.73 s** against a 2.0 s
budget. Four changes, in the order the measurements said they mattered.

- **One copy of the 35B, not two.** `segmentation.boundary_model` and
  `enrichment.caption_model` are the same pinned checkpoint, but
  captioning went through `mlx-vlm` and boundary detection through
  `mlx-lm`, and each built its own model object: MLX active memory
  18.99 → 37.15 GiB across the second load, in one process, reproduced on
  the development host. `imsg.shared_vlm_runtime.SharedVlmRuntime` now
  loads at most one model per `(repo, revision)` and hands it to both
  providers; boundary detection runs text-only through the
  vision-language model (`mlx_vlm.stream_generate(..., image=None)`).
  **That is the same computation, checked rather than argued:** the
  rendered chat prompt is byte-identical between the two paths (same
  SHA-256), the last-position logit row for a fixed probe is
  bit-identical across all 248,320 float32 values (same SHA-256 over the
  raw bytes), and both produce the same boundaries for the same window.
  The sharing is an object a caller passes in, not a module-level cache,
  so it is visible at the call site and a test can assert on
  `loaded_model_ids`; `models.share_boundary_and_caption_weights` turns
  it off. **The shipped agents are still two processes** —
  `…enrich` builds the captioner and `…sync`'s segment step builds the
  boundary provider — so this removes the duplicate within a process, not
  across them; both load lazily and `…sync` exits between runs, which is
  what keeps that bounded. Recorded where the LaunchAgents are rendered
  (`imsg.agents.plists`), not only here.
- **Only the PE-Core tower a process uses.** `CustomTextCLIP` is
  1,882,033,920 vision parameters (7.011 GiB at fp32) plus 537,233,920
  text parameters (2.001 GiB), and open_clip builds both whichever one
  you intend to run — 9.07 GiB on the query side, which calls only
  `embed_text`, and again on any process that embeds images and calls
  only `embed_images`. The provider now decides on first use and releases
  the other tower. **Numerically free, and proved rather than asserted:**
  `scripts/verify_pe_core_tower_selection.py` embeds the same fixed
  synthetic inputs through the whole model and through the selected tower
  in separate processes and compares the vectors as exact IEEE-754 bit
  patterns — identical, not close — while torch's MPS allocation goes
  9.071 → 2.010 GiB (text) and 9.071 → 7.061 GiB (vision). A process that
  uses both rebuilds once and keeps both, so nothing that worked stops
  working. Precision is untouched; fp16 is a separate question with its
  own evidence to gather.
- **The enrichment MLX buffer cache is bounded.** MLX pools freed GPU
  buffers and its default limit is its memory limit — 60.8 GiB on the
  production host, where one enrichment process was seen holding 37.25
  GiB of freed buffers, which is indistinguishable from a leak. The
  query-side providers had called `bound_buffer_cache` since 2026-09-15
  and held their 8 GiB exactly; the captioner, the boundary provider and
  Whisper called nothing. All three now do
  (`models.enrichment_cache_limit_bytes`, 4 GiB — one prompt at a time
  against one resident model, not a 64-row padded embedding batch), and
  the query side's own bound became `models.query_cache_limit_bytes`
  rather than a constructor default. Whisper gets its own because the
  limit is process-wide and a batch that happens to be all audio loads
  neither of the other two.
- **Enrichment yields to in-flight queries, never the reverse**
  (`imsg.db.enrichment_yield_locks`, `enrichment.yield_to_queries`,
  default on). With a single 35B copy resident — no swap, pressure never
  critical — query p95 still trebled while enrichment ran, so what
  remained was GPU contention on 20 cores. The MCP server now holds a
  **Postgres session-level advisory lock** for the span of each search
  and each warm-up step, and the worker probes it between units of work.
  The mechanism is chosen for its failure mode: a session lock is
  released by the server when the session ends, **however it ends**, so a
  killed or jetsammed MCP server cannot leave enrichment paused — no
  timeout to tune, no stale marker to reap, which is exactly what a lock
  file or a status row would get wrong. The asymmetry is total: the
  server's own acquisition is a `try` that proceeds unmarked rather than
  wait, and an unreachable database degrades the marker to a no-op rather
  than fail a query. The check is between tasks, never during one, so no
  claimed task is abandoned or corrupted; when nobody is searching it is
  one round trip to a local socket and no sleep. `imsg status` reports
  `enrichment_yield_enabled`, `query_in_flight` and
  `enrichment_yielding_now`, read from `pg_locks` without taking a lock.
  Tests cover the crash case and the no-query case explicitly, the crash
  one against a real Postgres because it is a property of the server, not
  of our code.

**Re-measured on the production host, the same concurrent case that
failed** (`scripts/bench_nightly_window.py`, new here: the query side
serving a search every 2 s while one enrichment worker captions
back-to-back, three phases in one run so recovery is observed rather than
inferred; synthetic queries and generated images throughout, nothing read
from the corpus).

| | before (D10, 2026-09-17) | after |
|---|---|---|
| query process, footprint | 27.5 GiB | **16.1 GiB** (17.2 peak-sampled) |
| query process, MLX active / peak / cache | — / — / 8.0 | 7.84 / 8.70 / **4.55** |
| query process, torch MPS | 9.07 | **2.01** |
| enrichment process, footprint | 53.8 GiB (measured alone) | **28.2 GiB** |
| enrichment, MLX active / peak / cache | 37.15 / — / **37.25** | **19.00** / 20.88 / **0.20** |
| enrichment, torch MPS | 9.07–9.11 | **7.06** |
| both, total demand | **80.4 GiB** | **44.3 GiB** |
| memory pressure during the overlap | **critical** | **normal** (131 of 133 samples; `warn` twice, never critical) |
| swap used, peak during the overlap | 37.6 GiB | **8.0 GiB** (1.2 GiB before it, 1.1 GiB after) |
| paged in | **218 GiB / 11 min** | **39.6 GiB / 19 min** |
| swapped out, whole run | — | 9.2 GiB |
| query model stages p50 / p95, baseline | — / 1.21 s | 1.075 / 1.087 s |
| query model stages p50 / p95, **overlap** | — / **8.73 s** | 2.194 / **3.003 s** |
| query model stages p50 / p95, recovery | baseline within one sample | 1.088 / 1.102 s |
| enrichment throughput | — | 36 captions across the overlap (3.75/min, p50 14.2 s each) |

**The overlap fits now; search does not stay inside its 2.0 s budget
during the window.** 44.3 GiB of 64 GB, pressure normal, swap a tenth of
what it was, and recovery immediate — the memory problem is gone. Search
p95 during the window is 3.00 s against a 2.0 s budget: down from 4.4x
over it to 1.5x over it, and still over it.

**The yielding did not cause that improvement, and this is the honest
negative result.** Running the identical case with `--no-yield` gives
overlap p50/p95 of 2.255/3.063 s against 2.194/3.003 s with it on — a
2% difference with the maxima ordered the other way, which is noise —
while the worker recorded 29 genuine pauses totalling 29.4 s and gave up
9.6% of its throughput (3.75 vs 3.84 captions/min). The mechanism works
exactly as designed and the design cannot help here: one caption takes
14 s on this host and a search arrives every 2 s, so a worker that can
only pause *between* units of work resumes into a fresh 14-second caption
that the next seven searches contend with. It is kept — it costs one
round trip when nobody is searching, it is what the decision ratified,
and it will matter at a lower query rate or a shorter unit of work — but
what closed the memory gap was the three residency fixes, not this.
Getting search inside 2.0 s during the window needs something that acts
*within* a unit of work, and that is a new question rather than a
tightening of this one.

**The harness reproduces the "before" configuration, which is why the
"after" column can be trusted.** Run on the development host with every
fix turned off — `--no-share-weights --pe-core-both-towers
--enrich-cache-limit-gib 0 --no-yield` — it measured a 25.06 GiB query
process (torch 9.07) beside a 55.35 GiB enrichment process (MLX active
37.15, torch 9.07): **80.4 GiB of total demand**, and search p95 during
the overlap of **8.92 s**. Those are D10's 80.4 GiB and 8.73 s, on
different hardware, from a harness that knew nothing of them. A benchmark
that could not reproduce the failure would be no evidence that the fixes
removed it.

**Not measured, and honest about it:** the 80.4 GiB configuration was not
re-run *on the production host* — the "before" column is D10's, and
putting that box back into critical pressure and 200+ GiB of paging would
have told us nothing the flags above did not.

The batch is captioning plus image embedding plus a boundary call every
fourth task; Whisper and Apple Vision are not in it, and
neither is the Photos intake session. Database time is excluded from
every latency figure, as it was before. The images are generated rather
than real attachments, so decode cost is representative of size but not
of format variety.

**On lowering the query-side cache bound during the window** (D10.2's
open question): measured, and the answer is not to. 8 GiB versus 2 GiB on
the development host moved the process footprint 20.65 -> 14.66 GiB —
exactly the 6 GiB of cache — with MLX active and peak identical (7.84 and
9.56 GiB) and the three query stages' p50s within run-to-run noise
(0.644 s versus 0.616 s in total). But under the real window workload the
cache only reached 4.55 GiB against its 8 GiB bound, so the bound is not
binding there, and the overlap now has ~20 GiB of headroom it would be
buying nothing with. `models.query_cache_limit_bytes` is the lever if
that changes; `scripts/bench_query_stages.py --query-cache-limit-gib`
re-measures it.
## 2026-09-17 — A source that has nothing to say no longer overwrites what another source knew

**Invariant:** a source that carries no evidence for a column must never
overwrite a value another source supplied — extraction writes a column
only where the incoming snapshot positively asserts it, so absence of
evidence is never recorded as evidence of absence.

Several witnesses of the same conversations feed this index: each Mac's
own `chat.db`, plus a recovered seed. They do not carry the same
information. Every `ON CONFLICT DO UPDATE` in S2 assigned each column
from the incoming row unconditionally, so the *last* source to run won
every column — including the ones it simply could not see. Found as
`has_attachments` flipping to false on messages that still had their
`message_attachment` rows; the flip matters because rendering reads the
real attachment rows while the segmentation boundary prompt reads this
column, so segmenting would have fed degraded input into new embeddings.

Measured against a pre-run dump of the same index, one host's
re-extraction of its own `chat.db` had cost:

| table.column | rows | what the silent source reported |
| --- | --- | --- |
| `message.has_attachments` | 14,282 true→false | no `message_attachment_join` rows |
| `message.is_edited` | 2,203 true→false | empty `edit_history`, NULL `date_edited` |
| `message.date_edited` | 2,203 value→NULL | NULL `date_edited` |
| `message.is_unsent` | 263 true→false | `is_unsent=false` from a blob it lacked |
| `chat.display_name` | 3,125 value→NULL | NULL `chat.display_name` |
| `attachment.source_path` | 4,775 value→NULL | NULL `attachment.filename` |

Two candidate columns turned out to have lost nothing:
`text_original`/`text_normalized` (0 rows) and
`sender_source_handle_id` (0 rows). Both were still fixed — they are the
same defect, and the second instance in the report that opened this work
was the body-blanking path — but the live index needed no repair for
either.

- **The policy is declared per column, once, in `imsg.stages.extract`**,
  and the SQL is generated from it (`_build_upsert_sql`) rather than
  spelled out per column inside each statement. Each column answers one
  question — *can this source's "empty" be told apart from this source
  having nothing to say?* — and gets one of four values:
  - `ASSERTED` — every source that has the row carries this column, so
    NULL and false are positive assertions. Only `message.is_from_me`
    and `message.sent_at` qualify: both are plain `chat.db` columns
    present on every row.
  - `PRESENT` — NULL means "nothing here" and no source can distinguish
    that from "empty". A NULL leaves the stored value alone; any
    non-NULL value still overwrites.
  - `POSITIVE` — a boolean naming the presence of something
    (`has_attachments`, `is_edited`, `is_unsent`, `attachment.is_sticker`,
    `tapback.removed`). Only `true` is evidence: `false` is what a source
    reports both when the thing is absent and when it cannot see it. All
    five also name facts that do not un-happen, so `true → false` was
    never a correction being blocked.
  - `INSERT_ONLY` — recorded on insert as provenance, never updated
    (keys, `chat_id`, `message.service`, `reply_to_guid`).
- **A genuine correction still flows**, which is the half a "keep the
  old value" rule can silently break. A rewritten body, a message that
  really was unsent since, a renamed chat, an attachment whose path
  finally arrived, a tapback that was taken back — all still overwrite,
  and the `missing → dataless` re-open in `_upsert_attachment` still
  fires, because a path is evidence. Asserted directly, not assumed.
- **The 2026-09-17 state-idempotence guard is unchanged and composes
  with this one**: the evidence flag is ANDed into each `IS DISTINCT
  FROM` disjunct, so a column a source cannot see is not merely left
  unwritten, it also cannot drag the row into an UPDATE that would move
  `updated_at` and re-segment the chat for nothing. One behaviour
  changed as a consequence: a body going value → NULL is no longer a
  change, so
  `test_extract_unchanged_rows_stay_untouched_integration.py` proves the
  NULL-sided `IS DISTINCT FROM` comparison in the NULL → value
  direction instead (with NULL → NULL as the control).
- **`_normalize_service` gained `_service_evidence`**, which returns
  NULL where the former returns `"unknown"`. `unknown` is not a service;
  it is the marker for "the snapshot did not say", and written as a
  value it overwrote a service another source knew. It survives as the
  value a first INSERT records.
- **`chat.kind` no longer overwrites on a guess.** An unrecognized
  `chat.style` falls back to a participant-count heuristic; that is good
  enough to insert a new row with and not good enough to retype an
  existing one, so a witness with an incomplete `chat_handle_join` can
  no longer turn a group into a DM.
- **Live repair, three transactions, each with controls that could
  fail.** The `BEFORE UPDATE` trigger from migration 0003 stamps
  `updated_at` unconditionally by design, so it was disabled and
  re-enabled inside each transaction (a ROLLBACK restores it too) and
  `tgenabled` verified after each commit. `updated_at` was restored from
  the pre-run witness only for rows whose content had returned to
  exactly the pre-run content, and only for rows carrying the degrading
  run's own stamp — 61 rows bumped deliberately by `imsg identity` to
  mark chats for re-segmentation kept their request. The strongest
  control: the in-index repair predicate (`has_attachments = false` with
  a `message_attachment` row) and the independent pre-run dump selected
  **the same 14,282 rows, with zero on either side alone**, and 656,202
  rows the run never touched matched the witness on every content column.
  **Dirty chats fell from 1,517 to 317** (14,282 + 2,203 + 263 message
  rows, 3,125 chat names and 4,775 attachment paths restored); what
  remains is 972 genuinely new messages awaiting first segmentation and
  610 that genuinely changed.

## 2026-09-17 — Extraction is idempotent in state, not just identity: an unchanged re-extraction now writes nothing

Re-extracting a corpus whose content had not changed moved
`message.updated_at` on **662,683 of 673,113 rows** for **972** genuinely
new messages. S4's `find_dirty_chats` keys on that column, so **8,463 of
8,569 chats** read as dirty and **8,331** had their incremental frontier
dragged back to the chat's first message — proposing to discard and
rebuild about **17 hours** of correct segmentation and embedding for no
content change at all. Found while cutting the index over to its
production host.

The cause was not the `updated_at` trigger (migration 0003), which is
right and unchanged: it exists precisely so no UPDATE can forget to move
the column. The cause was S2 performing UPDATEs it had no reason to
perform. Every `ON CONFLICT DO UPDATE` fired unconditionally, rewriting
each in-scope row with its own values.

- **Every upsert in `imsg.stages.extract` now carries an explicit
  `WHERE <target>.col IS DISTINCT FROM excluded.col OR ...` guard**, so a
  row whose content is unchanged is not written at all and the trigger
  never fires. `IS DISTINCT FROM`, not `<>`: half these columns are
  nullable, and `<>` against NULL yields NULL, so a message that gained
  or lost a body would read as unchanged for good (proven in both
  directions, plus NULL-to-NULL as the control, rather than assumed).
- **The compared columns are exactly the assigned columns**, on both
  sides of the rule. Compare a column the SET does not write and the row
  updates on every run forever — the original defect made permanent.
  Write one the WHERE does not compare and a genuine change to it alone
  is dropped. For `message` that set is `text_original`,
  `text_normalized`, `is_unsent`, `is_edited`, `date_edited`,
  `has_attachments`, `sent_at`, `is_from_me` and
  `sender_source_handle_id` — read off what `imsg.segment.pipeline`
  actually selects, plus what S3 joins on to resolve the rendered
  sender. `sender_source_handle_id`, `sent_at` and `is_from_me` were not
  previously corrected on re-extraction at all; they are now.
- **`chat_id` is deliberately still first-write-wins**, and not because
  it is unimportant — it matters more than most of the list. The
  incoming value is not trustworthy: `fetch_target_messages` derives it
  from `chat_message_join ... LIMIT 1` with no `ORDER BY`, so for a
  message in more than one chat SQLite may return either, and
  reassigning per run would flap the row between chats forever and
  orphan the `segment_message` rows placing it. Correcting a genuinely
  wrong chat association needs a deterministic choice first — a separate
  change, not a side effect of this one. `service` and `reply_to_guid`
  are excluded for the ordinary reason: nothing downstream reads either
  column, so a stale value cannot affect rendering or retrieval.
- **`message_source` is the one upsert left unguarded**, on purpose:
  `extraction_run_id` means "the run that last observed this row", so a
  new run id is a real change by construction and a guard could never
  skip anything. It carries no `updated_at` and nothing watches it, so
  it drags no re-segmentation behind it — the cost is table churn, not
  17 hours of recomputed embeddings. Every other upsert (`chat`,
  `source_handle`, `attachment`, `attachment_source`, `message_version`,
  `tapback`, `link_preview`) is guarded; the join tables were already
  `ON CONFLICT DO NOTHING`. `source_handle`'s `DO UPDATE SET raw_value =
  EXCLUDED.raw_value` — a no-op write performed purely so `RETURNING`
  would fire — is gone; a `UNION ALL` branch supplies the id at no write
  cost, and the same shape gives every guarded upsert its row id back.
- **Counts now distinguish inserted / genuinely updated / unchanged**
  (`ExtractResult.message_upserts` and friends, printed by `imsg extract`
  and `imsg sync`, and persisted by **migration 0005** as
  `extraction_run.messages_inserted/_updated/_unchanged`).
  `messages_upserted` keeps its original meaning — rows processed — so
  nothing reading it changes; the split is what makes it legible. "662,683
  upserted" reads identically whether a run corrected the corpus or
  rewrote it with its own values, and that ambiguity is what let this
  defect sit unnoticed. The new columns are nullable and unbackfilled:
  runs recorded before 0005 genuinely do not know their split, and zeros
  would assert "nothing was written" about exactly the runs that wrote
  the most.
- **Verified against rows, not reasoning**
  (`tests/test_extract_unchanged_rows_stay_untouched_integration.py`,
  seven tests, all seven failing before the fix). The suite segments the
  corpus and resolves its identities before taking the baseline, so
  `find_dirty_chats` has real segments to compare against and could fail
  for the right reason; it then compares every row's `xmin` — which moves
  on any rewrite, including on tables with no `updated_at` at all — across
  a re-extraction. After the fix: zero rows rewritten outside
  `message_source`, `find_dirty_chats` empty, and on a real run's
  `extraction_run` row `(upserted 3, inserted 0, updated 0, unchanged 3)`
  with every `updated_at` still equal to its `created_at`. Changing one
  message's body moves exactly that row and dirties exactly its chat; a
  new message in an existing chat dirties only that chat; and S3's
  deliberate bump (`rename_person` -> `_mark_chats_dirty_for_persons`)
  still marks every message of the affected chat, asserted immediately
  after a no-op re-extraction so a regression would show as an empty
  dirty set rather than be masked by S2's churn.

## 2026-09-17 — Qwen3-Reranker-0.6B pinned, buffer pool sized and prewarmed, HNSW recall raised: a whole query is p95 1.14 s

A search cost p95 36 s when it was first measured and 2.1-2.3 s a day
later. It now costs **p50 0.87 s / p95 1.14 s / max 1.28 s** end to end
through the real MCP surface on the M2 Ultra (20 fictional queries, three
cold starts, another job using the same disk array at 216-2,177 MB/s
throughout). Estimated for the production host from the measured
per-stage ratios (reranker 0.58, query embedding 0.64, PE-Core text 0.82,
database ~1:1): **p50 1.36 s / p95 1.61 s**, inside the p95 <= 2.0 s
budget.

- **The reranker is Qwen3-Reranker-0.6B** (owner decision), pinned as a
  local conversion like the 8B before it: `rerank_top` 20,
  `rerank_doc_max_tokens` 256. Reproducibility was checked before
  re-pinning — re-running the recorded recipe into a fresh directory
  produced a byte-identical artifact (same `artifact_sha256`, all seven
  hashed files equal). Smoke run through the factory: P(yes) 0.9981 for
  the relevant document against 1.7e-5 for the irrelevant one, load 2.7 s,
  inference 0.08 s, peak 0.87 GiB — against 8.09 GiB for the 8B. The
  conversion's `tie_word_embeddings` is true, so its yes/no logits come
  from the tied embedding matrix rather than an `lm_head`
  (`imsg.mlx_runtime.lm_head_logits` already handled both).
- **The lock gained `status: retained`**, so the 8B conversion stays
  pinned, digest-checked and reproducible without claiming the `reranker`
  role — it is kept for the Phase 4 quality comparison. `entries_by_role`
  ignores retained entries, `imsg models verify` prints the active entry
  per role and lists the retained ones, and the smoke harness now builds
  *the entry it is testing* rather than whatever is active for that role
  (without that, smoke-testing the retained 8B would have loaded the 0.6B
  and filed its numbers under the 8B's name).
- **`shared_buffers` 128 MB -> 3 GB, and something to fill it.** The
  search path's working set was measured, not guessed: `pg_statio_*`
  deltas over the 20 queries name 11 tables and 31 indexes, 2,296 MiB in
  all — the three HNSW indexes (1,193 MiB), the two embedding tables'
  heaps + TOAST + TOAST indexes (857 MiB, most of it `segment_embedding`'s
  642 MiB of TOAST, which every text search detoasts) and the
  summary-fetch and filter relations (246 MiB). `message`'s 629 MiB heap
  is deliberately excluded: 20 queries read 17 MiB of it. 3 GB holds that
  with 30 % headroom and leaves the production host's 64 GB for ~34 GiB of
  query-side models and a later ~20 GiB enrichment model. Migration 0004
  adds `pg_prewarm`; `RetrievalService.warm_up()` prewarms the whole set
  as its last step (2,296 MiB in 14.3 s cold, 0.1 s when already
  resident) and the instance config turns on `autoprewarm` for reboots.
  `imsg status` now prints `shared_buffers` against the total HNSW index
  size and warns when the pool is smaller.
- **`retrieval.hnsw_ef_search`, default 1000** (pgvector's maximum),
  applied with `SET LOCAL` in each vector channel's transaction. Measured
  recall@100 against exact search at 40 / 100 / 200 / 400 / 1000: 0.944,
  0.969, 0.989, 0.997, 1.000 (text) and 0.792, 0.887, 0.945, 0.985, 0.998
  (image); the worst single query 0.79 and 0.38 at the old default of 40
  against 1.00 and 0.98 at 1000. It costs ~52 ms of the two channels' p95
  together. Filtered searches, where the iterative scan does the work,
  were no slower at 1000 than at 40.
- **Found by measuring: the planner abandons the HNSW index in a band of
  `ef_search` values.** At 170-284 the text channel ran an exact
  `Sort + Seq Scan` — 538,197 buffer hits and 245 ms against 4,032 and
  9-18 ms — because pgvector's cost estimate for the index is not
  monotonic in `ef_search` (3,252 at 40, 13,287 at 280, 3,719 at 285)
  while the sequential alternative sits at a flat 8,827 and is undercosted
  anyway: the planner "doesn't consider out-of-line storage in cost
  estimates" (pgvector's FAQ) and the vectors it sorts are TOAST. Every
  vector channel now sets `enable_seqscan = off` for its own transaction —
  pgvector's documented remedy — so search latency no longer depends on
  where a tuned `ef_search` falls against a cost curve.
- **`check_permissions` answers during warm-up** instead of waiting for
  it, and reports the warm-up's own state: which step is loading, how many
  are done, the estimate, and the failure when there is one. It is how an
  operator finds out why searches are returning `WARMING_UP`. Measured:
  0.42-0.58 s while the models were loading.
- **The earlier unexplained ~70 s first query did not recur** in three
  cold starts: the first query after warm-up took 0.94-4.08 s. What is
  left of it is localized — the query embedder's first forward pass at a
  real shape, 0.13-3.07 s against 0.055 s afterwards; every other stage is
  already at its steady value. The likeliest explanation for the old
  number is the one this change set removed (cold index pages: the same
  day's stage log shows a single query spending 16.2 s in the segment
  vector channel and 12.9 s in the multimodal one), but it cannot be
  re-measured, so it is not asserted.
- **Deleted, reproducible:** the two rejected Qwen3-Reranker-4B conversions
  and the upstream 4B snapshot in the Hugging Face cache. Both recipes were
  re-run into fresh directories first and reproduced byte for byte, so
  either can be rebuilt with the `models/manifest.lock.yaml` recipe form
  (`python` = the project venv, `mlx-lm==0.31.3`):
  - `models/qwen3-reranker-4b-mxfp8-22e68366`: `src = snapshot_download('Qwen/Qwen3-Reranker-4B',
    revision='22e683669bc0f0bd69640a1354a6d0aebcfeede5'); convert(hf_path=src,
    mlx_path='$DATA_ROOT/models/qwen3-reranker-4b-mxfp8-22e68366', quantize=True,
    q_mode='mxfp8', q_bits=8, q_group_size=32)` → `artifact_sha256`
    `93458e04b142083f925c52c133e00a7b7872c2fb7c5095d688d66fe3446a4e77` (7 files, 4,159,200,136 bytes).
  - `models/qwen3-reranker-4b-4bit-22e68366`: the same with `q_mode='affine', q_bits=4,
    q_group_size=64` → `f6a8b9d4be75443a8036d3326bed980cda32f48b7d0f10e14c75c60eed677650`
    (7 files, 2,274,126,904 bytes).

Suite: 1,583 passed with a scratch database (30 new tests). The instance
config now carries `retrieval.hnsw_ef_search`, so it needs this build:
an older checkout rejects the key (`extra inputs are not permitted`).

## 2026-09-17 — `imsg mcp local` answers the handshake immediately and warms models in the background

The startup warm-up added earlier the same day ran before the server
answered `initialize`, so the handshake took 121.9 s. Claude Code bounds
MCP server startup with `MCP_TIMEOUT`, 30,000 ms by default (per its
environment-variable documentation and the installed client code), so the
server would have been dropped before it could answer anything.

- `initialize` and `tools/list` never wait for models; measured 0.77–1.06 s
  after process start on the Studio with the live config (four runs).
  Warm-up starts once the server is serving and logs per-model and total
  time to stderr.
- A tool call during warm-up waits up to 90 s without blocking the server
  (under the client's 2-minute point where a call moves to a background
  task; its hard tool timeouts are far longer), then returns a structured
  `WARMING_UP` error with an estimate of seconds remaining. A failed warm-up
  is logged once and every later call returns `WARM_UP_FAILED` naming the
  cause, instead of the process exiting at startup.
- **All model work runs on one dedicated thread.** A lock was not enough:
  MLX refused to run work prepared on another thread (`There is no
  Stream(gpu, 1) in current thread`). Serializing on one thread cost nothing
  measurable.
- **Found and fixed on the way: PE-Core wrote log lines to stdout**, which is
  the MCP protocol channel — two stray lines reached the stream before the
  fix, zero after, because warm-up now starts after the MCP library has
  redirected stdout. A real stdio child-process test holds warm-up and
  asserts no stray bytes reach the stream; it fails if warm-up runs before
  serving or starts too early.
- `WARMING_UP` and `WARM_UP_FAILED` are stored as themselves in `mcp_audit`
  (they were collapsing to `INTERNAL`).
- Measured: warm-up 77–185 s, dominated by loading the 8B reranker's 7.9 GiB
  (51–155 s); a warm search 2.1–2.3 s with the current 8B default. One first
  query after warm-up took ~70 s with the cause not established.

Suite: 1,553 passed with a scratch database.

## 2026-09-17 — search latency: reranker batching, early-stopping vector collapse, startup warm-up

First measured sweep against a real index (20 generic queries, numbers only).

- **Reranker batching.** Length-sorted batches under a padded-token budget
  (1,024, measured best) replaced fixed unsorted batches of 8. At 50
  candidates, rerank time fell from p50 58.4 s / p95 83.1 s to 27.0 / 35.9 s
  with scores unchanged. A new `retrieval.rerank_doc_max_tokens` caps the
  document body only, so the position the yes/no logits are read from never
  moves. The rerank pool is never smaller than the request's `limit`, so a
  small `rerank_top` cannot silently truncate results.
- **The multimodal channel read 500 rows to find 100 distinct segments,**
  which were reached by row 115 (median) / 157 (max). Collapsing channels now
  stream and stop once no unread row can change the result; identical output
  on the live index (40/40, 12/12), 128–192 rows read. The cursor is planned
  like the plain query — default cursor planning chose a different index and
  made an integration test flaky.
- **`RetrievalService.warm_up()`**, called by `imsg mcp local` before
  serving. The 1.27 s / 1.40 s embedding stages seen on a "warm" query were
  first-call loading and compilation: tokenizing 0.15 ms and a 51.6 ms forward
  pass for the query embedding, 17.6 ms for the PE-Core text tower. Without
  warm-up the first query took 113.6 s.
- **Not changed, measured:** vector searches are fast on warm pages and slow
  on cold ones (segment 304 → 7 ms, multimodal 898 → 24 ms via
  `EXPLAIN (ANALYZE, BUFFERS)`) — `shared_buffers` is the 128 MB default
  against 1.19 GB of HNSW indexes. HNSW recall@100 against exact search at
  the default `ef_search` 40 is 0.944 (text) and 0.792 (multimodal, worst
  query 0.38); at 400 it is 0.997 / 0.985. Shared-prefix caching for the
  reranker saved 18–28 % but moves scores by up to 1e-3 and reorders near-ties.
- Defaults `rerank_top` 10 / `rerank_doc_max_tokens` 64 were set as the
  fastest configuration within the pinned 8B reranker. They do not meet a
  2 s budget on the production host, and their ranking agreement with the
  full 50-candidate rerank equals the no-reranker order; see the private
  design record for the model decision this forces.
- `scripts/bench_retrieval_latency.py`: read-only, prints numbers only;
  rerank-only mode matched the full service in 24/24 comparisons.

Suite: 1,524 passed with a scratch database.

## 2026-09-15 — identity changes now mark their chats for re-segmentation

Rendered segments carry people's names — the chat header lists
participants and every line is `[HH:MM] <short_name>: …` — but
`find_dirty_chats` keys on `message.updated_at`, and `rename_person` /
`merge_persons` / `assign_handle` bumped only `person.updated_at`. So a
rename after segmentation never re-rendered anything and the index kept
the old name silently. Non-negotiable #3 makes that a correctness bug.

- `_mark_chats_dirty_for_persons` runs inside each mutation's existing
  transaction, so every caller (interactive commands, `apply-overrides`,
  `rematch-stubs`) inherits it and a dry-run rollback undoes it. It
  collects chats by the three paths the renderer actually reads —
  non-owner participants, non-`is_from_me` message senders, non-`is_from_me`
  tapback senders — then does one set-based `UPDATE message SET updated_at
  = now() … WHERE updated_at < now()`. The guard deduplicates within a
  transaction (probed: a second rename in the same transaction rewrites 0
  rows), which at full `rematch-stubs` scale is 95,516 rewrites instead of
  135,821. A merge marks only the kept person, which by then holds both
  sides' rows; an assign marks the old and the new person.
- End-to-end DB-gated test: segment a chat, embed it, rename a
  participant, assert the chat is dirty from its first message, re-segment,
  assert the rendered text carries the new name and `rendered_sha256`
  changed — which is exactly what makes `imsg embed` re-embed it.
- Owner guard: renaming the owner now needs `--yes-owner`. Belt-and-braces
  by measurement — an owner rename marks 0 chats and re-rendering
  reproduces the stored text byte for byte, because nothing rendered names
  the owner.
- Every curation command prints `chats marked for re-segmentation: N` and,
  when non-zero, says to run `imsg segment` then `imsg embed`.
- Known follow-up: `tapback.sender_person_id` is unindexed, so that branch
  sequentially scans (~5–18 ms per call); irrelevant for a handful of
  renames, roughly 1–3 minutes inside a full-scale rematch.

Suite: 1,442 passed with a scratch database.

## 2026-09-15 — embedding was 3.9× slower than it had to be: padding and an unbounded MLX cache

Measured on the first full-corpus `imsg embed` run (benchmark
`scripts/bench_text_embedding.py`, synthetic texts drawn from the real
segment-length distribution): the model's padded throughput is flat at
~920 tokens/s regardless of batch shape, so fixed batches of 32 in
`segment_id` order — padded to the longest row — wasted 3.9× of it
(17.3M real → 68.3M padded tokens). Independently, MLX's buffer cache
defaults to the memory limit; it grew ~1.3 GiB per batch to a 103 GB GPU
footprint with swap full, and the GPU stalled on paging.

- `imsg.embed.batching.plan_batches`: longest-first, greedy, bounded by
  rows and by rows × longest row (`embedding.max_batch_tokens`, default
  2,048 — the measured optimum); an oversize row goes alone. Results are
  keyed by id, one transaction per batch, so order changes nothing.
- The MLX text provider packs every call the same way on real token
  counts and sets an explicit 8 GiB cache limit at load (D10.2's
  "bound the cache" recommendation, now measured: same throughput,
  footprint 17 GB instead of 103).
- `max_length` was not the problem: the provider already padded to the
  batch's longest row (2048 vs 8192 byte-identical).
- Affine 8-bit local conversion of the embedder was benchmarked against
  the pinned mxfp8: ~10 % slower and not the same vectors (cosine mean
  0.978), so the pin stands. `scripts/convert_qwen3_embedding_mlx.py`
  documents the upstream-layout rename the converter needs.
- Real tokens/s: 174 before → 687 after on the longest rows, 845 on the
  typical distribution. Suite: 1,414 passed with a scratch database.

## 2026-09-15 — backfill names its residue: `unsupported` populated, NULL-path rows are `missing`, long-name bug fixed

The first AT-3 run against a real corpus reported `unsupported: 0`,
`dataless_retrying: 1,362` and `error: 211` — three misdescriptions of
the same residue. Read from the live rows rather than the report:

- **Out-of-root source paths are `unsupported`, not `error`.** The
  containment check is unchanged and still refuses; the outcome is now a
  terminal `unsupported[<class>]: …` with no retry schedule, classed by
  path shape only (`temp-directory-path`, `sticker-cache-path`,
  `out-of-root-path`) — the deterministic OSErrors `EISDIR`/`ENAMETOOLONG`
  join it (`is-a-directory`, `file-name-too-long`); everything else stays
  transient with backoff. A pre-pass reclassifies existing rows without
  reading anything.
- **A row with no source path is `missing` from the start** (extract), and
  a pre-pass heals existing `dataless` NULL-path rows; there is no
  placeholder to read, so "retrying" was false. Re-extraction leaves the
  state alone except `missing` → `dataless` when a path arrives.
- **`--retry-failed`** on `backfill-attachments` mirrors enrich's flag:
  `error` and `missing` rows with a path become eligible now;
  `unsupported` and NULL-path rows are never reset.
- **Two "file name too long" failures were ours:** the `.partial` temp
  name was built from the full basename (251 and 238 chars) and crossed
  NAME_MAX. Bounded to 64 chars; both files then materialized.
- **The printed report now equals `GROUP BY state`** for every terminal
  outcome (the containment branch used to count rows that became
  `missing` as `errored`); a test runs one pass producing every outcome
  and asserts report == database.
- AT-3 buckets `unsupported` and sub-counts it by reason class in the text
  report and the CSV; the rendered reason never carries the offending path.

Measured on the real index after one pass: materialized 99,121 / missing
1,362 / unsupported 443 (362 temp-directory, 73 sticker-cache, 8
directory) / error 0 of 100,926 rows; AT-3 PASS at 98.21 % present.
Suite: 1,385 passed with a scratch database.

## 2026-09-15 — the reranker is pinned as a reproducible local conversion (`source: local_conversion`)

- Owner decision: the reranker stays local. `models/manifest.lock.yaml` gains a
  second entry kind, `source: local_conversion` — `upstream_repo`,
  `upstream_revision` (sha), `upstream_license`, `tool` (`<package>==<exact
  version>`), `command` (the exact, reproducible invocation; `$DATA_ROOT` stands
  for `paths.data_root`), `output_dir` (data-root-relative) — with
  `repo`/`revision`/`license` null: a local directory has no Hub revision, its
  provenance is the upstream pin plus `artifact_sha256`, now computed over the
  OUTPUT directory by the same definition the smoke harness uses (the digest
  moved to `imsg.providers.manifest.artifact_digest`; `model_smoke` re-exports
  it). `qwen3-reranker-8b` is re-pinned this way: mlx-lm 0.31.3's mxfp8 8-bit
  (group 32) conversion of `Qwen/Qwen3-Reranker-8B` @ `77d193c7`, converted
  from the pinned upstream snapshot's directory (not the repo id — mlx-lm's
  `save()` copies `generation_config.json` from the cache's `main` ref, which
  the recorded command therefore does not depend on). Output: 653 tensors with
  `lm_head.weight` + `lm_head.scales`, 7.9 GiB on disk, 16 s to convert on an
  M2 Ultra; the reasons the Hub pin could not work stay in the entry's notes.
- Smoke-run through the factory path (`scripts/smoke_test_models.py --only
  qwen3-reranker-8b --data-root … --write`, M2 Ultra 128 GB): P(yes) relevant
  0.9707 vs irrelevant 0.0000, load 3.71 s, inference 1.19 s, peak 8.09 GiB;
  `artifact_sha256` `29e4a1ae…`. Every pinned model now has a passed smoke
  record. The harness locates a local conversion under `--data-root` (default:
  the schema's data root; passed through to the child process) instead of
  downloading, and reports artifact bytes separately from downloaded bytes.
- `imsg models verify` / `scripts/verify_model_manifest.py` now (a)
  re-resolves a local conversion's upstream repo for drift — reported, never
  written: `--write` never advances an upstream pin, because the directory on
  disk was converted from it; (b) checks each local conversion's directory
  exists under `--data-root` (symlink-resolved containment) and that its
  recomputed digest equals the lock (`--skip-artifacts` when the volume is not
  mounted); (c) notes whether the recorded conversion tool version is what is
  installed (informational — only re-running the command depends on it).
  Verified clean against the live HF API, the real data root and the installed
  runtimes on 2026-09-15.
- Config: `retrieval.reranker_model` may be a directory relative to
  `paths.data_root` as well as a Hub repo id; absolute, `~` and `..` forms are
  rejected with a message naming both forms, and a root validator resolves
  symlinks so the directory cannot escape the data root. The factory reads the
  value as a local directory when `<data_root>/<value>` exists — building
  `MlxRerankerProvider` with `revision=None` and `model_id`
  `<dir>@<upstream sha>` (new `model_id` keyword) — and as a repo id otherwise;
  a missing `models/…` directory is one clear error naming the recorded command,
  not a doomed Hub download. `imsg.constants.RERANKER_MODEL` (renamed from
  `RERANKER_MODEL_REPO`) and `RERANKER_MODEL_REVISION` now default to the
  conversion's `output_dir` and the upstream sha; `config.example.yaml` and the
  README describe the two forms. The Studio instance config (private, outside
  every repo) points at the conversion; `imsg status` validates it and prints
  `models: backend=real`.
- Tests: the new entry kind's parsing and every rejection, upstream drift
  versus `--write`, the artifact check (match, mismatch, unrecorded, missing
  directory, unmounted data root, symlink escape), the tool note, the local-dir
  config forms, the factory's revision/`model_id` handling, and the smoke
  harness's locate-not-download path and `--data-root` plumbing.

## 2026-09-14 — first real execution of every pinned model (`scripts/smoke_test_models.py`)

- `scripts/smoke_test_models.py` / `imsg.providers.model_smoke`: per lock entry,
  download the pinned sha into the Hugging Face cache, compute `artifact_sha256`
  (definition in the lock header; `artifact_digest` is the reference), build the
  real provider through `imsg.providers.factory` from a config carrying exactly
  the pins, run one fictional input per role (two documents + a query; a
  relevant/irrelevant pair; a 12-message two-topic window; a rendered PNG; a
  `say`-synthesised sentence via ffmpeg; a drawn shape + two texts), and record
  load time, inference time and peak memory (`mlx.core.get_peak_memory`, or max
  RSS for torch/Vision). Every (entry, role) runs in its own child process so the
  numbers are per model and two large models are never resident together;
  `--write` rewrites only `smoke_test`/`artifact_sha256`, byte-preserving the
  rest. Stubbed tests in `tests/test_smoke_test_models.py`.
- Ran on an Apple M2 Ultra with 128 GB (not the deployment mini). Passed:
  Qwen3-Embedding-8B mxfp8 (peak 7.41 GiB), whisper-large-v3 (3.66 GiB), PE-Core
  G14-448 on MPS (9.43 GiB RSS, 107 s to build), Apple Vision, and
  Qwen3.5-35B-A3B 4-bit for both roles — boundary via mlx-lm (19.07 GiB) and
  caption via mlx-vlm (19.90 GiB), which settles the open question: mlx-lm 0.31.3
  loads the mlx-vlm-converted checkpoint (`qwen3_5_moe` reads `text_config`).
- Found by running, not reading: neither mxfp8 Qwen3 conversion ships an
  `lm_head` tensor (their `model.safetensors.index.json` files do not describe
  the shards), so mlx-lm's strict load failed on both with `Missing 1
  parameters: lm_head.weight`. The embedder never uses the head — upstream
  `Qwen/Qwen3-Embedding-8B` has none either — so `embed.mlx_text` now declares
  `tie_word_embeddings: true` through a new `model_config` pass-through in
  `mlx_runtime.load_model_and_tokenizer`, and loads. The reranker *needs* the
  head (its P(yes) comes from `lm_head` logits) and upstream
  `Qwen/Qwen3-Reranker-8B` ships a distinct one that the mlx-embeddings
  conversion dropped: the pinned `mlx-community/Qwen3-Reranker-8B-mxfp8` is
  recorded `failed` with the exact error and cannot be made to work from our
  side. The one fix evaluated — a local `mlx_lm.convert` of the upstream repo at
  its pinned sha, 8-bit mxfp8, which keeps the head — loads through the same
  factory path and scores P(yes) 0.97 vs 0.00 at 8.1 GiB peak; command and
  numbers in the entry's notes. Re-pinning is the owner's call (the lock pins
  Hub revisions; a local directory has none).
- `tests/test_verify_model_manifest.py` now checks the shape of a recorded
  smoke run instead of asserting nothing has run.
## 2026-09-14 — two identity integration tests still encoded the pre-August Contacts rules

Found by running the suite against a scratch Postgres for the
`finished_at` fix. `tests/test_identity.py` last changed on 2026-07-30;
`identity.py` changed three times on 2026-08-15; and the two end-to-end
tests covering those changes only run with a database, none of which was
reachable when the rules landed — so they failed silently for a month.
Test-only: the code was right both times.

- The "multiple contact matches fall back to a stub" fixture was
  "Alice" / "Also Alice", which the 2026-08-15 subset rule treats as one
  person at two levels of completeness. It is now a genuine conflict (two
  different people on one household number), and a sibling test pins the
  subset rule end to end through the real S2→S3 handoff: the most
  complete name wins and the person stays off the review worklist.
- `assign_handle` has checked the target person before the handle since
  9fa85ce; the test asserted the handle error against an empty `person`
  table and got the person error instead. It now asserts both, in order.

## 2026-09-14 — read-only means the directory is untouched: no sidecars beside chat.db-shaped files

- **`SQLITE_OPEN_READONLY` was leaving `-wal`/`-shm` files next to databases
  the pipeline only reads.** Every chat.db-shaped file here carries the WAL
  header byte (the live database, S1's backup output, the seed corpora), and
  SQLite's read path opens the `-shm` wal-index read-write even on a
  read-only connection. Evidence: `imsg extract --snapshot <seed>` moved the
  seed's `-shm` mtime to the run's start, and `snapshots/` held orphaned
  `.tmp-snapshot-*.db-shm`/`-wal` files dated 2026-08-14, left by the
  post-backup verify open before the temp file was renamed. New
  `imsg.sqlite_readonly` opens `file:<path>?mode=ro&immutable=1` (path
  percent-encoded); S2's snapshot open and S1's verify open use it, and the
  tests now assert the containing directory is byte-identical before and
  after a read — the flags-only test could not see this class.
- **S1's live source deliberately stays a plain read-only open.**
  `immutable=1` makes SQLite ignore the write-ahead log: measured against the
  live `chat.db`, the immutable open was four messages behind, and it would
  race Messages.app's checkpointer. `readonly_shm=1` and
  `locking_mode=EXCLUSIVE` were rejected too (the first fails hard when the
  wal-index it may not write is missing or unusable; the second needs an
  exclusive lock a read-only descriptor cannot take — measured: disk I/O
  error). The reader-protocol writes to `chat.db-shm` are accepted and
  documented in the module; `chat.db` and `chat.db-wal` are never modified
  by a read-only connection (end-to-end S1 run against the live database:
  both files' sizes and mtimes unchanged).
- **Fail closed where immutable would hide data.** S2 refuses a `--snapshot`
  whose `-wal` holds frames and names the one-line fix
  (`PRAGMA wal_checkpoint(TRUNCATE)` on the copy). Without this, a seed
  copied together with its log would silently lose its newest messages, and
  AT-2 would agree, because `--reference-db` reads the reference the same
  immutable way.
- **The `--snapshot` live-database refusal (landed earlier today, below) now
  compares by inode as well as by resolved path.** Resolved paths catch
  symlinks and `..`, but a hard link, and the macOS firmlink alias
  `/System/Volumes/Data/Users/…` of `~`, resolve to a different string for
  the same file — verified on the host: `Path.resolve()` says different,
  `os.path.samefile` says same. `imsg.paths.is_same_file` does both, and a
  hard-link test pins it.

## 2026-09-14 — real model providers behind `models.backend`, and the QA pass that read them together

Four agents built the real providers in parallel against a contract and the
branches were merged unread; this entry covers everything since `7c71097`,
including the review that followed. The `extraction_run.finished_at` fix from
the same day has its own entry below.

**Landed (the build):**

- `ffmpeg`: `-fps_mode` replaces the removed `-vsync` option (keyframe sampling
  failed on current ffmpeg).
- `imsg verify-seed --reference-db <chat.db>` builds the AT-2 reference straight
  from a chat.db-shaped file, applying the extractor's three inclusion filters
  in SQL — a recovered/merged corpus has no host to run `--export` on, which
  had left AT-2 unrunnable for exactly the seed it exists to check.
- Real providers, each importing its runtime lazily so `imsg` imports without
  the `models` extra: `imsg.mlx_runtime` (shared mlx-lm loading with revision
  pinning, right-padded batching, last-token gather); `embed.mlx_text`
  (Qwen3-Embedding: EOS suffix from the tokenizer's own post-processor,
  last-token pooling, Matryoshka truncation + L2); `retrieval.mlx_reranker`
  (Qwen3-Reranker: the model card's yes/no logit recipe);
  `segment.mlx_boundaries` (Qwen3.5 boundary LLM: chat template with thinking
  off, greedy decoding, between-token deadline, tolerant JSON parse);
  `enrich.vision_ocr` (VNRecognizeTextRequest, accurate, reading-order sort);
  `enrich.mlx_whisper_transcription`; `enrich.mlx_vlm_caption` (fixed prompt,
  `prompt_sha256`); `embed.pe_core_multimodal` (PE-Core via open_clip on MPS,
  per-item image failures, revision-pinned snapshot).
- Wiring: `imsg.providers.factory` is the only place providers are built;
  `models.backend` (`real` default, `fake` opt-in) with a `models: backend=…`
  line on every model-backed command; `models/manifest.lock.yaml` +
  `imsg models verify` / `scripts/verify_model_manifest.py` (drift against the
  HF API, runtime floors, `--write` only on request); config fields for every
  repo/revision and the enrichment knobs; the `models` extra in
  `pyproject.toml`; README rewritten around the above.

**Found by reading the merged result (each fixed, with a test):**

- The PE-Core pin could never load: manifest/constants/config pinned
  `facebook/PE-Core-G14-448 @ a604668…`, but the provider fetches from the
  `timm/PE-Core-bigG-14-448` mirror, which has no such commit. Pinned the
  mirror at its own sha (`17aa0c25…`); a test asserts the pin names the repo
  `resolve_weights_repo()` would fetch from.
- A missing MLX runtime silently degraded segmentation: `MlxBoundaryProvider`
  wrapped load failures as `BoundaryDetectionError`, which the caller answers
  with the session-as-segment fallback — every session, run reports success.
  Environment failures now propagate as `MlxRuntimeError` (abort), consistent
  with `ModelRuntimeUnavailableError` on the enrichment side.
- One unembeddable image aborted the whole multimodal pass: the provider's
  per-item `ImageEmbeddingError` was never caught. The two classes moved to
  `imsg.errors`; the pass logs, counts (`EmbedRunReport.attachments_failed`)
  and continues; every other `EmbeddingError` still aborts.
- `enrichment.ocr_languages: null` promised automatic language detection but
  the provider never enabled it (Vision's default is a fixed en-US list with
  detection off, read back from a live request). Switched on for the null case.
- HEIC — the iPhone default — could not be decoded: `pillow_heif`'s opener is
  registered on load when the package is importable.
- Caption provenance: `prompt_sha256` was computed and never stored; it now
  lands in `enrichment.detail`. The mlx-vlm call passes
  `enable_thinking=False` explicitly (rendered against the pinned Qwen3.5 chat
  template) instead of relying on the library's per-model default.
- `segmentation.boundary_revision` was not part of `seg_config_hash` (D4). It
  is now (payload `v2`); **existing installs re-segment every chat on their
  next run.**
- No `prompts/segment_boundaries.txt` was ever shipped although config
  defaulted to it. It is now, stating the parser's exact contract; both prompt
  lookups fall back from `data_root` to the repo copy and print which file was
  used; a non-UTF-8 prompt is one clear error instead of a traceback.
- `--snapshot` never compared its path with `paths.live_chat_db`: a seed could
  open the live database directly. Refused by resolved path — the live db, any
  configured source db, anything inside the live Messages directory
  (non-negotiable #1). Found by today's real seed run.

**Verified against the installed runtimes, not from memory** (mlx 0.32.2,
mlx-lm 0.31.3, mlx-whisper 0.4.3, mlx-vlm 0.7.1, open_clip_torch 3.3.0,
huggingface_hub 1.31.0, pyobjc Vision 12.2.2): every API name the providers
call exists with the assumed signature — `mlx_lm.load(revision=)`,
`stream_generate(sampler=)`, `make_sampler(temp=)`, `mx.take_along_axis`
broadcasting, `mlx_whisper.transcribe(path_or_hf_repo=, temperature=)`,
`mlx_vlm.load`/`generate`/`prompt_utils.apply_chat_template` (kwargs reach the
template), open_clip's `local-dir:` schema and `require_pretrained`
forwarding, the Vision selectors including `setAutomaticallyDetectsLanguage_`.
mlx-lm ships `models/qwen3_5_moe.py` and its `sanitize` drops the vision tower;
the pinned Qwen3.5 `chat_template.jinja` honours `enable_thinking`; the
Qwen3-Embedding tokenizer's post-processor appends exactly `<|endoftext|>`
(151643). Absent in mlx-lm 0.31.3: `mlx_lm.utils.get_model_path` — harmless,
`load()` takes `revision` and the fallback chain handles its absence.

**Executed, on stand-ins where the pinned weights were not downloaded:** Apple
Vision OCR read a generated image back verbatim on macOS 26.6.2 (recorded in
the manifest — the one entry not `not_run`); the transcription chain (speech
synthesis → ffmpeg → `MlxWhisperTranscriptionProvider`) returned an exact
transcript on the 50 MB `whisper-tiny-mlx-q4`; the three mlx-lm providers
loaded the 79 MB `SmolLM-135M-Instruct-4bit` and produced unit-norm
embeddings, yes/no probabilities and a parsed-or-rejected boundary answer,
bit-identical across pad-token choices (right padding cannot leak). **No
pinned model has been executed end to end; retrieval quality with the 8B / 35B
weights is unknown.**

Suite: 990 passed, 198 skipped (no database here); ruff and mypy strict clean.

## 2026-09-14 — `extraction_run.finished_at` recorded when the write transaction opened, not when the run finished

Postgres `now()` is frozen at transaction start, and S2 stamped
`finished_at = now()` from inside the single transaction that wraps the
whole upsert loop. So the column recorded the moment the loop *began*
writing, and every duration computed from the table excluded the write
phase entirely — always in the flattering direction. Probed on the live
instance: `now()` did not move across a `pg_sleep(2)` inside one
transaction; the seed run of the merged corpus (run 7) was still logging
work at 19:08:22 but its row says it finished at 19:06:01, 11.9s after it
started, for roughly 2m35s of real wall time; the earlier 655,494-message
studio run (run 5) shows 7.3s. The MCP freshness check reads
`max(finished_at)`, so it was early by the same amount.

- **`finished_at = clock_timestamp()`** in `_do_extract` and
  `_fail_extraction_run`, and at the `export_run` ledger's two stamps
  (`plan_export`, `push_export`). The export sites were statement-scoped
  under autocommit and so correct by accident; they change so the stamp's
  meaning no longer depends on whatever transaction a caller wraps them
  in. `started_at` is untouched — its DEFAULT fires in the run row's own
  short transaction.
- **`updated_at = now()` deliberately left alone** (`sync_state` and every
  trigger-backed table): migration 0003's trigger overwrites it with
  `now()` regardless of the statement, and "the transaction that set
  this" is the intended meaning there.
- **Regression test** injects a `pg_sleep` inside the write transaction
  through the `_backfill_tapback_targets` seam and asserts the recorded
  window covers it. It fails on the old code by exactly the injected
  delay; a slow `imsg-dump` would *not* have reproduced the bug, because
  work done before the transaction opens was already inside the window.
  A no-database scan (`tests/test_finished_at_uses_clock_timestamp.py`)
  fails on any completion column written with `now()` under `src/imsg`,
  so the class cannot return quietly.
- Historical rows (runs 5 and 7) keep their wrong `finished_at`; the real
  finish times survive only in the run logs.

## 2026-08-17 — first real corpus run: the write-loss class, and identity curation

Recorded late (2026-09-03). This work landed across 2026-08-14→17 with no
CHANGELOG entry, so for three weeks the only record of it was eleven commit
messages — exactly the failure the living-documents rule exists to prevent.
The entry is written now rather than skipped, because the findings below are
the reason several of the current rules exist.

**The write-loss class — stages reported success and persisted nothing.**
Found only by running against a real corpus, because no test could see it.
A stage that opened with a bare read left the connection `INTRANS`, and
`transaction()` could not rescue it; segment, embed, enrich, and sync's S4–S6
half all open that way. Each printed a success line, exited 0, and wrote
nothing. The first fix closed only the fingerprint check's transaction and
was found by adversarial review not to fix the class it claimed to; the real
fix is `connect()` defaulting to `autocommit=True`, with per-stage rollbacks
replaced by an idle assertion that fails loudly. Verified both directions:
with autocommit off a read leaves the connection `INTRANS`, with it on reads
stay `IDLE` and `transaction()` commits.

**Other defects that only a real corpus surfaces**, from the same window:
SQL variable chunking and NUL stripping (both fatal on any real corpus);
chunking double-upserted attachments shared across messages; `identity assign`
updated only `handle.person_id` and left existing messages attributed to the
old person, splitting one sender across two `person_id`s in violation of
non-negotiable #3; options before a subcommand were parsed then dropped, so
`identity --config other.yaml merge` hit the *default* database;
`review-report` inflated message counts 2x through handle fan-out, distorting
the ranking it exists to produce; `rename` never cleared `needs_review`, so
the worklist could not shrink; the mount gate accepted `/` when the volume was
unmounted, leaving a stale sentinel as the only protection.

**Identity curation, built against real address books.** `imsg identity
duplicate-candidates` ranks possible same-person pairs and shows both sides'
identifiers; `review-report --sample` shows one message per person, which is
very often decisive because automated senders announce themselves in the text.
Contacts matching gained three rules, each measured rather than guessed:
duplicate accounts naming the same person are not a conflict (1,338 of 1,375
identifiers had every card agreeing; the old rule cost the top correspondents
disproportionately, since the people you message most are the ones saved in
more than one account); a name whose words are a subset of another's is the
same person at two levels of completeness; emoji decoration is not a different
person. Nickname equivalence is deliberately **not** inferred — the rule that
merges Joe/Joseph also merges Chris/Christina.

**Handle normalization: 2,803 persons were fragmented by iOS filter tags.**
iOS writes a filtered sender as `<id>(filtered)` / `(smsft_*)`; untagged the
handle parses as a phone and tagged it does not, so one sender resolved to two
persons. Stripping the suffix is anchored and allowlisted rather than "cut at
the first paren", because real handles contain parens. This was the single
largest source of person fragmentation in the index — larger than every
Contacts ambiguity combined.

## 2026-09-03 — `--snapshot`: the one-shot seed path, and a guard against it

SPEC §8 S7 specifies a seed as `imsg sync --source <name> --snapshot <path>`,
and `run_sync(snapshot_override=...)` has implemented it since the build — but
**neither option existed on the CLI**, so the only ways to run a seed were to
overwrite `snapshots/snapshot.db` or to call the library by hand. The first is
destructive: S1 atomically replaces that file from the live chat.db every
`sync.interval_seconds`, so a file staged there is gone within the interval.

- **`imsg sync --source <name> [--snapshot <path>]`** — `--source` alone syncs
  that one source instead of fanning out; with `--snapshot` it is the
  studio-seed one-shot path, skipping S1 and feeding the prepared file to
  S2→S3→S4→S6.
- **`imsg extract --snapshot <path> --source <name>`** — the S2-only
  equivalent.

**The guard is the point of the change, more than the option is.** A seed
advances the ROWID watermark of whatever source it is ingested under, and a
ROWID means something only inside one database file. Ingest a prepared corpus
under a live source whose file has fewer rows, and the watermark jumps past
real messages that were never read — they sit below it forever, and nothing
reports rows it never looked at. There is no error, no count discrepancy at
ingest time, and no way to notice except by missing data much later.

So `--snapshot` fails closed on all three of: no `--source`; a `--source` that
names a configured `sync.sources` entry (the live-source collision); and a
missing file. Six tests cover the refusals and both positive paths, and the
positive extract test asserts the seed path was used **by value** against the
pipeline snapshot path — a test that only checked "it ran" would pass while S2
read the wrong file.

## 2026-09-03 — real contact data had reached the fixtures; scrubbed

The repo's single governing rule is that it is public-safe by construction:
no real names, numbers, or addresses in code, comments, tests, or fixtures —
fictional personas only. The identity work of 2026-08-14→17 was written while
debugging against the real corpus, and **the corpus leaked into the fixtures**:
nine real personal names across `tests/test_contacts_index.py` and the
explanatory comments in `src/imsg/stages/identity.py`, one real mobile number,
and two real vendor brands quoted as automated-sender examples.

All of it is replaced with fictional personas that preserve the exact
structural property each case was written to test — a name-subset pair stays a
subset pair, a shared-surname conflict stays a non-subset conflict, an
emoji-decoration pair stays a decoration pair. 655 tests pass unchanged, so
the fixtures were carrying the *shape* of the real data, not depending on its
identity.

**Why it happened, and the cheap check that finds it:** every one of these
entered as an illustration in a commit that was otherwise correct — the real
example is the most convincing one to reach for while the debugging session is
still in your head, and no test can fail for it. A grep of the diff for
capitalized word-pairs and for phone/email patterns finds the whole class in
seconds, and is worth running before any branch built against the real corpus
is pushed.

Aggregate statistics measured on the corpus (2,803 fragmented persons;
1,338 of 1,375 identifiers agreeing) are deliberately **kept** — they identify
no one and they are the entire rationale for the rules they justify.

Fictional numbers now use the reserved 555-01xx range inside a real area code
(`+1 202 555 0123`), which `phonenumbers.is_valid_number` accepts — the older
`+1555…` fixtures parse as invalid, which is fine where the value is only a
dict key but not where a test asserts `kind == "phone"`.

## 2026-08-12 — first run against real infrastructure; the mount gate never worked

The code had never been run against a real Postgres instance or a real
encrypted volume (836 tests, zero real deployments). Standing it up on
the Mac Studio for the Track C build surfaced two defects immediately —
both invisible to the test suite, both in the code↔system boundary.

- **`real_diskutil_info` passed `data_root` straight to
  `diskutil info`**, which accepts a device node or a *mount point* and
  exits 1 for any path inside a volume. `data_root` is *always* inside
  one (`/Volumes/Data-Encrypted/imsgindex`), so `guard_mount` failed on
  every valid deployment — including the mini's. Fixed by walking up to
  the containing mount point first (`containing_mount_point`), which is
  what the docstring already claimed the function did.
  **Why 836 tests missed it:** every existing mount test injects a fake
  `diskutil_info`; the one line that actually shells out had *no test at
  all*. Three regression tests added.
- **`verify_data_directory` does `Path(str(row[0]))`, which corrupts
  `bytes`.** Under a `SQL_ASCII` cluster psycopg3 returns `bytes`, so
  `str()` yields the literal `"b'/Volumes/…'"` — a *relative* path,
  silently resolved against the process CWD. The fingerprint check then
  refused the correct cluster. Root cause is upstream and is a
  documentation defect, not a code one: the implementation guide's
  mandatory `export LC_ALL=C` (its rule #1, which exists to stop the
  "postmaster became multithreaded" crash) makes `initdb` create a
  **SQL_ASCII** cluster. Fixed at the root — the cluster is now
  `initdb --encoding=UTF8 --locale=C` (C locale keeps the crash fix;
  UTF8 fixes the encoding).
  ⚠️ **The bytes hazard is a class, not one site:** 11 call sites do
  `str(row[...])` over fetched text, including `verify/seed.py:82`
  (the AT-2 GUID sets — the seed-completeness gate) and
  `export/review.py:258` (the manifest SHA on the export gate). Under
  SQL_ASCII those corrupt *silently*. Left unpatched deliberately: the
  UTF8 cluster makes the whole class unrepresentable, which is the
  structural fix. A connect-time assertion that `server_encoding` is
  UTF8 is the belt-and-braces follow-up and is **not yet written**.

## 2026-07-30 — Published: this repo is now public at `msg43/imessage_mcp`

- **Made public.** The repo was built public-safe by construction from
  its first commit, so publishing required no history rewrite and no
  cleaned-snapshot re-initialization — which is the whole reason it was
  built as a separate repo rather than carved out of the design record
  later.
- **README rewritten.** The previous one was written at foundation time
  and had gone stale in the way most likely to mislead: it claimed the
  pipeline stages "exit with a not-implemented-yet error", which
  stopped being true five commits earlier. The rewrite leads with what
  a stranger would otherwise discover the hard way — this has never run
  against a real corpus, and it ships deterministic fake model
  providers, so running the pipeline reports success at every stage
  while producing meaningless results.
- Test counts stated as 639 passing plus 197 integration tests that
  skip without a live database. An earlier draft said "836 passing",
  true only with PostgreSQL running — a cloner would have seen 197
  skips and reasonably concluded the setup was broken.
- The README also carries the traps this build hit, since they cost
  real time and are not discoverable from the code: pgvector's index
  caps sitting below its type limits, why audience and subject
  validation are not redundant, migration immutability,
  trigger-enforced `updated_at`, overfetch-not-post-filter, and
  normalization parity.
- **`LICENSE` (MIT) and `NOTICE` added, deliberately as two files.**
  GitHub's detector reported `NOASSERTION` while the GPL note was
  appended to `LICENSE` — licensee matches the whole file against known
  texts, so trailing prose defeats detection and costs the repo its
  machine-readable license metadata. `LICENSE` is now the canonical MIT
  text alone; `NOTICE` carries why the GPL parser is not optional, why
  the subprocess boundary is shaped as it is, and what changes if
  someone vendors it differently.
- **Adopted the doc-lifecycle protocol** (this file, `GAMEPLAN.md`, a
  `CLAUDE.md`, and a warn-only pre-commit nudge). The repo had been
  `git init`'d as a build target without it.

## 2026-07-30 — Consolidation: migration 0003

- **`updated_at` is now enforced by the database, not by convention.**
  Segmentation re-processes a chat when its messages' `updated_at`
  moves, and extraction does bump it — but nothing in the schema
  required that. Any future code path updating a row without
  remembering `updated_at` would have stranded that chat out of
  re-segmentation with no error, no failing test, and stale search
  results as the only symptom. Attached via a catalog loop, so tables
  added by later migrations inherit it automatically.
- `materialization_state` gains `unsupported`, an exception category the
  reconciliation report could name but the enum could not express — the
  bucket could only ever read zero. Nothing populates it yet; the
  backfill stage must set it at Phase 2.
- Corrected a misleading `tapback.kind` comment forward as a
  `COMMENT ON COLUMN` rather than by editing migration 0001. Applied
  migrations are immutable and the runner enforces that by hash.

## 2026-07-30 — Public MCP surface, export transport, eval harness

- Public StreamableHTTP surface with a transport guard in front of
  everything: a valid bearer token is required on **every** request,
  including `initialize` and `tools/list`, so the tool surface is not
  readable unauthenticated. Scope resolves centrally into one access
  context per call rather than per-tool, so a future tool cannot forget
  it.
- Export transport implemented against GCS + Discovery Engine behind
  the existing Protocol, credentials from env/Keychain only, and
  deliberately never CLI-wired so it cannot fire accidentally.
  Unverified against the live API until Phase 7.
- Eval harness: nDCG@k, pooled recall@k, MRR, coverage, runner and diff
  table. Metrics checked against hand-computed fixtures whose arithmetic
  is written out independently in the test rather than calling the
  module under test.
- Seed-completeness verification homed in `imsg verify-seed`, using an
  exported counts snapshot rather than a live second database — the two
  hosts are never reachable at once.
- `--dry-run` across all stages. Extraction and identity use savepoint
  and forced rollback for genuinely accurate previews; enrichment is
  deliberately partial and says so, since model output cannot be known
  without running the model.

## 2026-07-30 — Export gate

- Eligibility computed as **evidence-of-deny**: every query collects
  reasons to deny, and a thread is eligible only with zero deny-evidence
  and at least one participant — so a bug can only ever *shrink* the
  eligible set.
- Document ids are RFC-1034-safe (`d` + 62 hex = 63 chars). The 63-char
  limit is real and documented; truncation alone would have been an
  incomplete fix, since RFC-1034 also requires a leading letter and a
  hex digest can begin with a digit.
- Purges are exempt from the approval gate: retraction only narrows
  scope and is recoverable by re-export, so gating it would slow the
  safe operation to the speed of the dangerous one.

## 2026-07-30 — Retrieval service and local MCP surface

- Hybrid flow: FTS/BM25, primary text vector, and secondary multimodal
  vector fused by RRF, then reranked. Query-side normalization matches
  ingest-time normalization exactly — a mismatch silently breaks
  exact-phrase search, so it is tested directly.
- Filtered retrieval overfetches rather than post-filtering a fixed
  top-K, which silently starves results when filters are selective. The
  starvation case is reproduced in a test, then proven fixed.
- Tool surface is **closed**: no `run_sql` or raw-query escape hatch on
  either surface. Adding one is a spec change.

## 2026-07-30 — Pipeline stages and the security boundary

- Ingest (snapshot, extract, identity, sync) plus the GPL `imsg-dump`
  shim, invoked strictly as a subprocess. Building against the real
  crate corrected three spec assumptions, including that unsent state
  comes from the typedstream rather than a SQL column.
- Indexing (segmentation, attachment backfill, enrichment, embed) with
  local-only model providers behind interfaces, shipped as deterministic
  fakes until real loaders land.
- Auth boundary authored against the threat model rather than the happy
  path: strict audience equality (the pinned-subject check alone does
  not stop a token minted for another application), fail-closed on every
  validation failure, and no configuration path that disables it.

## 2026-07-30 — Foundation

- Repo scaffolded public-safe by construction: no real names,
  hostnames, or identifiers anywhere, with instance values supplied by a
  private overlay at runtime.
- Config validation is the enforcement mechanism for the project's
  non-negotiables, not documentation of them.
- Migrations 0001/0002 with a hash-checked runner, and a DDL lint that
  asserts pgvector's real index caps — a column can be legal DDL whose
  ANN index can never be created, which is exactly the defect that
  prompted the check.
