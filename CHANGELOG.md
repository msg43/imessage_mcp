# CHANGELOG

Dated milestone history, newest-first. Terse bullets explaining the
**why**, not just the what — link commit hashes where useful. Add the
entry **in the same commit(s) as the work** whenever something notable
lands: a schema/migration change, a significant feature, a fix batch, a
gate transition. Skip the genuinely trivial (typo/format/lockfile bump);
when in doubt, add the line.

This is a running document, not a one-time artifact — status must never
live only in a chat transcript or an assistant's session memory.

## 2026-09-14 — read-only means the directory is untouched: no sidecars beside chat.db-shaped files, and `--snapshot` can no longer name the live database

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
- **`--snapshot` may no longer name the live chat.db.** `_validate_seed_or_die`
  only checked that `--source` was not a configured source, so
  `imsg extract --snapshot ~/Library/Messages/chat.db --source anything` was
  accepted. It now refuses any `--snapshot` that is `paths.live_chat_db` or
  a `sync.sources[].chat_db`, by resolved path *and* by inode
  (`imsg.paths.is_same_file`): the macOS firmlink alias
  `/System/Volumes/Data/Users/…` and a hard link both resolve to a different
  string for the same file.

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
