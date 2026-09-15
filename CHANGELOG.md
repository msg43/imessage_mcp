# CHANGELOG

Dated milestone history, newest-first. Terse bullets explaining the
**why**, not just the what — link commit hashes where useful. Add the
entry **in the same commit(s) as the work** whenever something notable
lands: a schema/migration change, a significant feature, a fix batch, a
gate transition. Skip the genuinely trivial (typo/format/lockfile bump);
when in doubt, add the line.

This is a running document, not a one-time artifact — status must never
live only in a chat transcript or an assistant's session memory.

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
