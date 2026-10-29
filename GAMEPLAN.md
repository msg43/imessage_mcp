# GAMEPLAN

The ONLY status document for this repo. Edited **in place** — never fork
a dated copy (`GAMEPLAN-2026-01.md`, `GAMEPLAN-old.md`, etc.). Flip status
in the same commit that changes it: a gate passes, an owner item resolves,
a deviation gets ratified. The moment any other document starts answering
"what is the current status," that content belongs here instead.

Read this file first, before anything else in the repo — cold start
order is: this file, then `CLAUDE.md`, then the module you're touching.

## Current status

**October 16 extraction repair is deployed and verified:** release
`bb1b368515bf946ab99d1c572e031a938ea6b751` runs from the deployed `adcf4e2` base.
It reuses the exact `0c0e798`/`f765e98` provenance repair plus existing continuous
integration dependency/help-test fixes; runtime files are unchanged from
`54cda878`. Migration 0014 is applied with no pending migrations and clean hashes.
A verified backup preceded migration; provenance counts and foreign keys survived.
A manual wrapper run and a restored scheduled run both persisted new messages,
passed exact-content/source sample checks and retained the conflicting message's
old and new provenance observations. Independent Astra postdeployment review
accepts live database, sample, job and process evidence with no rollout blocker.
The extraction uniqueness bug is closed.
Both wrappers exited 76 because the intentional heavy-work pause remains active.
Original sync/backup jobs and configuration are restored; the sync interval is
unchanged. The search page responds and the public server process was preserved.

**Main is deployed on the production host (2026-10-28); background work runs whenever the public server has been idle for 10 minutes.**
The owner confirmed the photo import finished on 2026-10-26. On 2026-10-28 the host
was timed, then moved from `a482d60` to `92c0f4c` (branch `agent/item-4.5-main-merge`,
a merge of main into the release lineage whose tree is identical to main `5cd086d`)
by fast-forward. That commit is not yet published. This installs the admission
priority (`698e3a8`), sync waiting for searches (`057b81b`), the sandboxed MIME
sniffer and caption scaler (`16dc5cf`) and the doctor credential fix (`c056cb5`).
No migration or dependency changed. Checks after the restart: doctor 9/9 pass; a
sandbox probe refused a write outside the work directory; the public endpoint
answers metadata 200 and refuses a no-token request with 401. Search through
`imsg mcp local` (60 warm searches): 1.054 s p50 / 1.142 s p95 before, 1.045 s /
1.096 s after; reranker 787 ms p50. Sync yielding and server-first admission are
not yet exercised on the host, because both need background work.
Background work was resumed the same evening (22:11) and `VACUUM (ANALYZE)` ran.
Every background load was then refused memory: the enrichment worker (OCR kinds),
the scheduled sync's segmentation and `imsg embed` each waited 600 s and exited 75
(`deferred: memory`). Each needs its `memory.footprints` figure plus the 8.0 GiB
reserve (24.8, 21.0 and 24.9 GiB), and the host had 24.7-26.5 GiB available, less
1.5 GiB still promised to the public server, which keeps its models loaded because
`mcp.public.idle_unload_seconds` defaults to 0. Search was unaffected. The operator
then set `mcp.public.idle_unload_seconds: 600` and made the runner scripts retry
on exit 75. The public server unloaded after 10 idle minutes, and 15 s later the
enrichment worker and the waiting sync were admitted (38.5 GiB available); the
worker handed the heavy-model lock to the sync, which began segmenting the
backlog. Live-server-first then held on the host: a local MCP search was refused
memory and answered `WARMING_UP`, the sync gave way the same second (exit 75),
the server loaded in 8.5 s and answered 30 s after the first try, and the next 20
searches ran at 1.062 s p50 / 1.140 s p95. A refused server answers `WARMING_UP`
immediately rather than waiting, so a burst of calls all fail. OCR then failed on
every image on the host's macOS 27.2: `AppleVisionOcrProvider` passed a Python `{}`
as Vision's handler options, which pyobjc 12.2.2 on macOS 27 turns into
"NSInvalidArgumentException - key does not exist"; the OCR runner was stopped
after one chunk (no task failed permanently). The fix, `e6ab43b` on
`agent/ocr-vision-options-fix-20261028`, was deployed on 2026-10-29 by fast-forward
(no server restart); 1,172 OCR tasks then completed in under 3 minutes with no
failures, and the index caught up (8,274 of 8,281 messages since the freeze
segmented, every segment embedded). Core follow-ups: the
enrichment estimate does not depend on `--kinds`, so a run without captions is
charged for the captioner (what such a run really loads is not measured); and
runner scripts that stop on any non-zero exit should treat 75 as "retry later".
The new messages are extracted; fresh full-text or semantic search coverage is
not established. Caption policy is already decided: run continuously,
skipping link-preview captions. Current queue aggregation is complete; throughput,
search impact and completion estimates remain unmeasured. Command access is
restored without access-setting changes; the interruption's cause is unestablished.

**Authorized work, October 15:** pending availability changes are committed and
pushed as `85ed3dd` on `codex/october-availability`; the configured-credential
doctor fix is `c8f57a5` on `codex/doctor-config-credentials`. Their reviewed deltas
are now integrated on main in `c056cb5`, preserving its sandbox, search-priority
and secret-file support. The original branches remain preserved. This source
publication does not establish deployment.

**October 14 deployment evidence:** the deployed schema contains
`message.deleted_at`, migrations match deployed source, and memory-protection
source is present. The timer's last exit was `1`; its cause remains undiagnosed.
Historical local-schema and undeployed-status paragraphs below retain their
original dates and must not override the newer component-specific evidence.

**October 15 follow-up:** the implementation agent inspected current code and
selectively reconciled the availability and doctor changes with current main.
Explicit keyword fallback and basic/recent reads preserve existing authorization,
pooled parallel retrieval, model admission and search priority. Default hybrid
search still reports model unavailability rather than silently changing meaning.
The doctor uses the configured secret resolver and bounds connection/Keychain
waits without displaying partial secret output. Publication is not deployment.

### Extraction is repaired while heavy indexing remains paused

- [x] **Extraction uniqueness failure:** minimal release and migration are live.
  Persisted manual and restored scheduled runs verify new canonical messages,
  exact-content/source samples and both conflicting source observations. The
  schema/writer pairing, migration hashes and preserved relationships are checked.
  Incremental selection is unchanged; rows below the watermark are not forcibly
  rescanned. Exit 76 records the intentional pause rather than extraction failure.
- [ ] **Caption measurement:** the owner decided September 23 on continuous
  captions with link-preview captions skipped. That policy is settled. The active
  photo import gates heavy work. Current queue composition has been
  measured privately without model calls. Throughput, search latency and combined
  memory still need measurement after the photo-import gate clears. Historical
  measurements do not establish today's completion time. Do not lift
  the pause or start bulk enrichment during the sync rollout.

### Authorized work status

- [x] **Runtime extraction acceptance:** backup, coordinated minimal deployment,
  persisted-output checks and restored scheduled extraction are verified. Heavy
  indexing and broader model/admission rollout remain gated by the photo import.
- [x] **Automatic naming of new review stubs:** uniquely matched new stubs are
  named through existing matching/rename logic and verified from a fresh read-only
  connection. Unmatched new stubs and historical identities remain unchanged
  under the existing human identity review. Scheduled Contacts access is verified;
  no permission grant is needed. Affected chats await the permitted indexing run.
- [ ] **Caption policy measurement:** evaluate the decided continuous-caption
  policy with link-preview captions skipped when the photo-import gate permits
  heavy measurement. The scheduling and scope decision is settled.
- [x] **Reconcile pending source:** port only verified deltas from the published
  availability and doctor branches onto current main. Preserve authorization,
  privacy, sandboxing, model admission and search-priority behavior. Run focused
  tests and the public-data check before publishing integrated code.
- [x] **Independent QA:** `gpt-6.1-sol` implementation and `gpt-6-astra` ultra
  review are complete. Astra passes 359 focused tests and 57 real synthetic
  PostgreSQL/SQLite retrieval, allowlist and concurrency tests. The public-data
  scan has zero hard findings and 34 reviewed warnings for changed source; Ruff
  and diff checks pass. QA added the bounded Keychain probe after reproducing its
  missing timeout. Keychain and connection waits are separately bounded; this
  does not promise a total timeout for all diagnostic queries. Runtime extraction
  acceptance is complete. Broader availability/runtime acceptance and heavy
  measurements remain gated as recorded above.

**Full-suite follow-up:** the first remote CI run exposed an outdated tool count,
colored-help assertions and missing media executables. CI now installs ffmpeg and
Poppler, help tests compare visible text, and the concurrency check includes the
existing recent-message tool. The local no-PostgreSQL suite passes 2,428 tests with
659 skipped under Python 3.14; the CI runner uses Python 3.12. These environment
counts are distinct from the 57 real database tests above. No product behavior or
test coverage was removed to accommodate the runner.

**Remote acceptance:** [CI run 38819507200](https://github.com/msg43/imessage_mcp/actions/runs/38819507200)
passed on `f765e98`, including the extraction repair, Ruff, the public-data check
and the Python 3.12 suite (2,359 passed, 720 skipped). The database-dependent
regressions also passed the independent local PostgreSQL checks above. The remote
optional type check reports 215 errors in
56 files; the green workflow does not establish a clean type check. These source
checks close the CI repair, not the outstanding host/runtime gates.

**Type-check triage:** review of that actual CI output assigns 176 errors to
tests, 29 to scripts and ten to two unrelated optional Google integration modules.
No error names a changed production module or `scripts/doctor.py`. The 34 errors
in changed test files concern fixture/annotation inference and typing interfaces;
they must not all be labelled preexisting. Inspection of the plausible runtime
cases found the expected attributes and populated values. No reproducible product
defect was established, so this remains nonblocking typing debt rather than a
diagnosis of the live sync failure.

Grounded bug/security fixes found along this path are authorized. Larger features
belong in the existing roadmap as ownership questions with supporting evidence,
scope and dependencies. Evaluation and external-exposure gates remain in force.
The preserved alternative segmentation experiment is an ownership question:
should it receive a separate quality/throughput evaluation before any algorithm
change? It is excluded from this availability integration; no benchmark result
has established that it improves the production corpus.

**2026-09-29 the MIME sniffer and the caption scaler run sandboxed
(built, not yet deployed):** closes the QA review's last enrichment
finding. `file` and the caption provider's downscaled copy now run under
the same sandbox and per-task budget as every other decoder: no network,
writes only in the task's work directory under `data_root`, and the
task's time, temp-space and memory ceilings (`file` keeps its 10 s / 120 s
bound). A hit fails that attachment's task. No migration and no config
change; deploying needs only a restart of the enrichment workers. See
`CHANGELOG.md`.

**2026-09-26 the sync waits while a search is running (built, not yet
deployed):** `imsg sync`'s segmentation waits before each boundary-model
call and its embedding before each batch while any search is in flight
(a public or local MCP search, or the search page's model API computing an
embedding or rerank), at most 300 s per wait, as the enrich worker already
did; `imsg segment` and `imsg embed` too. On the M2 Ultra development host
a burst of reranks beside a sync-shaped embedding load went from 2.6x
slower to 1.0x; a lone search that starts during a batch is still 2.4-2.5x
slower. No migration and no config change; background commands pick it up
at their next start, and nothing needs a restart for it. `imsg status` adds
`segment_yielding_now` and `embed_yielding_now`. See `CHANGELOG.md`.

**2026-09-25 MCP servers load before background work (built, not yet
deployed):** an MCP server whose model load is refused posts a notice
(`<data_root>/run/live-servers-waiting/<pid>.json`); while it is up, no
background model load starts and running background work stops after its
current unit of work (exit 75). What background jobs were promised counts
against the server only for the first 60 s of its wait
(`memory.background_yield_seconds`). A refused load is retried every 15 s
and logged when its cause changes and at least once a minute; `imsg status`
shows which servers wait and on what. Deploying needs a restart of
`imsg mcp public`; no migration and no required config key. Background
commands and `imsg mcp local` pick up the change at their next start. See
`CHANGELOG.md`.

**2026-09-25 search page filters, saved searches and result downloads
(built, not yet deployed):** "Sent by" a person or me, sent / received,
one-to-one or group conversations, and one conversation; a search with no
words lists every message the filters keep; searches can be saved outside
any case and are listed on `/saved`; "Download results" writes every result
as Markdown, CSV or JSON with citations. Deploying needs `imsg migrate`
(0013) and a restart of the search page only. See `CHANGELOG.md`.

**2026-09-25 search page changes for forensic review (built, not yet
deployed):** a segment hit is kept only when the words occur in what people
wrote, so a contact's name, a year, an hour or a time-zone word no longer
matches every segment's header lines; the date filter tests each message's
own time; each message has "Copy citation"; decision codes gave way to plain
words and the evaluation progress to a Labels page; "show all hits" loads 50
at a time; Back from a conversation keeps the loaded results and the place;
a Details panel on every message shows its exact times, sender, edits,
deletion, filing and sources (earlier text and raw handles each behind a
`search_page.details` switch, off by default); a grading mode grades every
one of a search's top 20 in random order for the reranker evaluation and
stores the whole fused list (migration 0011); a Timeline shows every
message across conversations for a day or range, and a Media grid shows the
files, both opening the conversation at the chosen message; evidence cases
collect messages and files with notes, track reviewed conversations of saved
searches, and download as Markdown, CSV or JSON with exact citations,
optionally with the original files and a SHA-256 list (migration 0012).
Deploying needs `imsg migrate` (0011, 0012), the two `search_page.details`
keys the owner chose, and a restart of the search page only. See
`CHANGELOG.md`.

**2026-09-24 public endpoint fixes from the QA review (not yet deployed):**
requests refused before any token was judged are counted in memory and
written as one `mcp_audit_rollup` row per code per minute instead of a
row each; an address with 10 failures a minute is throttled; the global
failure budget is 600, so one source can no longer lock the owner's next
token out; a tool call costs one rate-limit event; token checks, audit
writes and searches run off the event loop, and a search's five channels
run side by side on pooled connections (same results; p50 154 → 70 ms
for the database stages on the Studio's synthetic corpus); the metadata
document and uvicorn no longer name the software; keep-alive 60 s; a
naive `get_conversation` anchor is read in `render.timezone`. New
command `imsg mcp audit-prune`. Deploying needs `imsg migrate` (0010)
and a restart of `imsg mcp public` on the index host; see `CHANGELOG.md`.

**2026-09-24 private local search page (D14), built, not deployed:**
`imsg search-page serve` returns every viable hit grouped by conversation,
full text first and semantic above a similarity floor after, with threads
that scroll both ways, attachments inline and relevance toggles that write
eval labels. Local network only, one owner login; it loads no model and
uses `imsg mcp public`'s models through a loopback-only internal API.
Measured on a synthetic production-sized corpus on the Studio: 14-74 ms
full-text search, 26-85 ms for the whole first page, ~70 ms of semantic
database work; a near-stopword query takes ~0.4 s (`CHANGELOG.md`). Owner steps to deploy: add `search_page:`
to config, set the password, create the model-API secret, restart `imsg mcp
public`, then render and install the agent (`imsg install-agents --only
search-page`). 2,734 tests pass against
a scratch Postgres (2026-09-24).

**2026-09-24 enrichment decoder limits and lock hand-over (not yet
deployed):** every enrichment decoder runs sandboxed (no network, writes
only in the task's work directory, now under `data_root`) inside one
budget per task (wall clock, temp bytes, decoder memory, output size);
PDFs are counted and sized before they render; Vision refuses images over
`max_image_pixels`; and the enrichment worker hands the heavy-model lock
to a waiting sync between tasks. See `CHANGELOG.md`.

**2026-09-24 operations fixes (built, not yet deployed):** the public
server's PE-Core query load reads a 2.0 GiB text-tower checkpoint
instead of building the whole model (3.6-3.8 s against 25.4-25.9 s on
the development host, bit-identical vectors); every LaunchAgent runs a
mount-gated supervisor that reads secrets from 0600 files, so Postgres,
the public server and the nightly backup can run under launchd with
`KeepAlive`; the database role works without superuser (prewarm fixed
for TOAST relations); logs rotate at 50 MB from the nightly backup;
`imsg status` reports every SPEC §14 field. Deployment is the owner
to-do below. See `CHANGELOG.md`.

**2026-09-24 public readiness + host memory fixes:** git history
rewritten to remove real contact data used as fixtures; install guide,
examples, setup doctor, Postgres bootstrap, CI with a public-safety
check, and plain-language CLI help landed. `data_root` may now sit on a
FileVault-encrypted startup disk. The index host ran out of memory and
hung on 2026-09-24 (idle MCP servers each holding a full model set, plus
`sync` and `embed` running together): model-heavy commands now take a
host-wide lock, and idle `mcp local` servers unload their models after
10 minutes. Test suite: 1,847 passed / 477 skipped with every extra
(no database); ruff clean; mypy reports 96 pre-existing errors.

**2026-09-24 memory protections (not yet deployed):** every model load now
asks whether the host has room first (both MCP servers, segment, embed,
enrich, sync's heavy steps, eval); `imsg mcp local` loads nothing until
its first retrieval call; heavy background work can be paused (`imsg
background pause`, or the host pause file another project creates) and
stops between units under memory pressure; MCP servers unload at critical
pressure. Follows the 2026-09-24 out-of-memory hang; see `CHANGELOG.md`.
2,380 tests pass against a scratch Postgres (2026-09-24).

**2026-09-23 MCP transport fix:** validated integral JSON numbers are
normalized before retrieval, so `limit: 3.0` cannot cause a slice TypeError.
All three integer tool arguments covered; invalid fractions/booleans remain
rejected before the handler. Focused public/dispatch/schema tests pass.

**Running against a real corpus; Phase 1 in progress.** Public at
[`msg43/imessage_mcp`](https://github.com/msg43/imessage_mcp) (MIT).
Every buildable component of the governing spec is implemented: 8
pipeline stages, hybrid retrieval, both MCP surfaces, the export gate,
the eval harness, 38 CLI commands (counting subcommands), migrations
0001–0009. 2,203 tests, all passing against a scratch Postgres (measured
2026-09-23); ruff clean (mypy: 96 pre-existing errors as of 2026-09-24); DDL lint clean. Since 2026-09-23
extraction merges only add (D12): a seed inserts rows and fills empty
values, and only this machine's own live `chat.db` may replace a value.
`imsg identity merge-filtered-twins` (2026-09-23) merges the persons that
the pre-fix import split on iOS filter tags; it has not yet run on the
production index. Also since 2026-09-23 every message is filed (D13),
including rows with no chat link: into the chat their evidence names, or
a holding chat that `allowlist` scope and export deny; each seed source
picks its rows up on its next extraction, and none has run yet. The
attachment fetcher (D13, 2026-09-23) is built but **not yet run against
the corpus**: `locate-attachments` records every candidate copy of an
attachment (another Mac, its drives, a NAS share, drive catalogs), the
backfill tries them best first and verifies each, and `push-attachments`
copies from a host the index host cannot reach. **No attachment text has
been extracted yet** (production `enrichment` table empty, 2026-09-23):
the queue-filling path is built as of 2026-09-23 — `imsg enrich --plan`,
S5a queueing on materialization, cheap kinds claimed before captions —
and not yet run; see the owner to-do below.

**This status previously read "code complete, unrun" and stayed that way
for three weeks after it stopped being true** — see `CHANGELOG.md`
2026-08-12 and 2026-08-17. S1→S3 have executed against a real Postgres
cluster and a real corpus, and doing so immediately surfaced a class of
defect no test could see: stages that printed success, exited 0, and
persisted nothing. Identity resolution and Contacts curation have run
against real address books.

The real model providers exist as of 2026-09-14 — `imsg.providers.factory`
builds them behind `models.backend: real` (the default; `fake` is explicit
opt-in and every command prints which), pinned in `models/manifest.lock.yaml`
— and every pinned model has now been downloaded (or, for the reranker,
converted locally), checksummed and run once on a synthetic input through
the factory (`scripts/smoke_test_models.py`, 2026-09-14/15; results in the
lock). The reranker is pinned as a reproducible local `mlx_lm.convert` of
the upstream repo since 2026-09-15, because the Hub conversion ships no
LM head; since 2026-09-17 it is the 0.6B rather than the 8B (owner
decision, for the p95 <= 2.0 s search budget — the 8B conversion is kept
in the lock as `status: retained` for the Phase 4 quality comparison), and
since 2026-09-25 the 0.6B's 16-bit (bf16) build, which reranks in about
40 % less time with the same ranking quality on a public test (the mxfp8
build is retained for rollback; not yet deployed on the index host).
**Search now meets its latency budget end to end on the Studio**: p50
0.87 s / p95 1.14 s through the real MCP surface, p95 1.61 s estimated
for the production host from measured per-stage ratios (`CHANGELOG.md`
2026-09-17). **No model has run on the pipeline end to end**: segmentation,
embedding and retrieval quality with the real 8B / 35B weights is unknown,
batched throughput and memory are unmeasured, and a `fake` run still
reports success with meaningless results. Phase 1's exit criteria — seed
completeness (AT-2) and a hand-verified `person` table — are **not** met.
(2026-09-15)

## Gate ladder

Phases 6 and 7 **must not begin** until Phase 4 produces baseline
numbers. That dependency is the whole reason the eval harness exists.

1. [ ] Phase 0 — Host + encrypted volume — volume auto-mounts at login; `guard-mount` passes; risk acceptance recorded
2. [ ] Phase 1 — Snapshot + extract + identity — seed completeness passes; person table hand-verified
3. [ ] Phase 2 — Attachment backfill — reconciliation report produced; gaps enumerated
4. [ ] Phase 3 — Segment + embed + local MCP — **real model providers replace the fakes**; answers real questions
5. [ ] Phase 4 — Eval harness — ≥30 real queries / ≥100 pooled judgments; baseline recorded ⚠️ **gates 6 and 7**
6. [ ] Phase 5 — Enrichment — attachment text searchable; OCR bake-off pins the model
7. [ ] Phase 6 — Tunnel + OAuth + subject validation — isolation test passes; scope set per the probe result
8. [ ] Phase 7 — Allowlist + export + ingestion — pre-push review completed and signed off
9. [ ] Phase 8 — Side-by-side eval — local vs. hosted scored on the real query set

## Owner to-dos

Deployment is a human-gated sequence; the step-by-step guide lives in
the private design-record repo, not here (it names real hosts and
accounts).

- **Deploy the 2026-09-24 operations fixes on the index host** (CHANGELOG
  2026-09-24), in this order: produce the text-tower directory
  (`scripts/convert_pe_core_text_tower.py`, the lock's recorded command,
  or copy it and check `imsg models verify --data-root`); render
  `imsg install-agents --only pg --only mcp-public --only backup` to a
  scratch directory and review it; hand Postgres over to launchd (stop
  the old postmaster with `pg_ctl stop -m fast` and confirm it is gone
  before loading `com.imsgindex.pg`; Postgres's own `postmaster.pid`
  interlock refuses a second one regardless); load `mcp-public` and
  `backup`; make the login script ask launchd instead of starting
  services itself; then `ALTER ROLE <role> NOSUPERUSER` with
  `GRANT pg_read_all_settings`, and `VACUUM (ANALYZE)` once after heavy
  work resumes. Optional but recommended: password (SCRAM) auth for TCP
  and `peer` for the owner-only socket.

- **Deploy the bf16 reranker on the index host** (CHANGELOG 2026-09-25):
  copy `models/qwen3-reranker-0.6b-bf16-e61197ed` under the host's
  `data_root` (or run the lock's recorded command there) and check it with
  `imsg models verify --data-root`; if the instance config names
  `retrieval.reranker_model`, point it at the new directory; restart the
  query servers. Keep the mxfp8 directory: setting `reranker_model` back to
  it is the rollback. Then confirm the speed with
  `scripts/bench_query_stages.py` once heavy work there resumes, and
  re-measure the query process's memory, whose two configured figures
  (`memory.model_footprints.public_server_bytes`,
  `memory.mlx_memory_limits.public_server_bytes`) were measured with the
  mxfp8 build.

- **Run the real model providers end to end** before trusting Phase 3.
  Done as of 2026-09-14 for the single-input smoke level
  (`scripts/smoke_test_models.py`, results in `models/manifest.lock.yaml`):
  every pinned model was downloaded, checksummed and run through the
  factory on an M2 Ultra / 128 GB. **Reranker re-pin: resolved 2026-09-15**
  — owner decision: the reranker stays local, pinned as a reproducible
  local conversion (`source: local_conversion`) of upstream
  Qwen/Qwen3-Reranker-8B @ 77d193c7 with mlx-lm 0.31.3 (mxfp8, 8-bit,
  group 32), because the Hub conversion dropped the LM head; the converted
  directory lives under `paths.data_root` (`models/…`), passes the same
  smoke check (P(yes) 0.9707 vs 0.0000, peak 8.09 GiB), and `imsg models
  verify --data-root` checks its digest. The Studio instance config points
  at it. Still open: `segment` / `embed` / `enrich` on one real chat with
  the real backend. Note the `seg_config_hash` v2 bump: the first real
  `segment` run re-segments every chat. Vector dimensions are load-bearing:
  pgvector's HNSW index caps below what the models natively emit, so
  changing them requires a migration and a full re-embed.
- **Fill and drain the enrichment queue** (Phase 5; CHANGELOG
  2026-09-23). On the production host, in order: `imsg migrate` (0009);
  `imsg enrich --plan --dry-run` (expect about 202,000 tasks: `doc_text`
  976, `pdf_text` 1,039, `ocr` 93,165, `transcript` 4,825, `frame_ocr`
  4,561, `caption` 97,726; 41 unroutable); `imsg enrich --plan`; then the
  cheap kinds, `imsg enrich --kinds doc_text,pdf_text,transcript,ocr,frame_ocr
  --limit 200000` (estimated 14-27 h, run when searches can take the
  contention); captions are left to the nightly agent (estimated 340-400
  GPU-hours, about 75 nights at its `--limit 100` every 30 minutes).
  Owner decisions open: whether captions may also run in the daytime
  (D10.3 measured search p95 3.0 s while captioning), and whether
  link-preview images (28,603 of the 93,165 images, 2,534 of them
  favicons) need captions at all. Estimates are Studio measurements scaled, not
  measured on the production host; the first night's `imsg enrich`
  output gives the real per-kind rate.
- Confirm the host's unified memory before Phase 0 — it selects the
  model ladder. First real inputs (2026-09-14, M2 Ultra 128 GB, one model
  resident at a time, peak per `models/manifest.lock.yaml`): Qwen3.5-35B-A3B
  4-bit 19.9 GiB (mlx-vlm) / 19.1 GiB (mlx-lm); PE-Core G14-448 9.4 GiB RSS;
  Qwen3-Embedding-8B mxfp8 7.4 GiB; whisper-large-v3 3.7 GiB. These are
  single-input peaks — batched embedding and long windows will sit above
  them — and they were not measured on the mini.
- ~~Populate the `unsupported` materialization state in the backfill~~ — **DONE 2026-09-15:** the backfill sets `unsupported` by reason class and AT-3 sub-counts it (CHANGELOG 2026-09-15).
  stage at Phase 2; the reconciliation bucket reads zero until then.
- Arrange the second account needed for the Phase 6 isolation test.

## Document ledger

| Document | Purpose |
|---|---|
| `GAMEPLAN.md` (this file) | Current status — always edited in place |
| `CHANGELOG.md` | Dated history — append-only, newest first |
| `CLAUDE.md` | Standing instructions for any agent working here |
| `migrations/` | Schema, applied in order by a hash-checked runner. Applied migrations are immutable — correct forward, never edit |
| `scripts/lint_ddl.py` | Asserts pgvector's real index caps; a column can be legal DDL whose ANN index can never be created |

**Governing spec and decision records live in the private design-record
repo.** This repo is public-safe by construction and must never carry
real names, hostnames, credentials, or instance configuration.
