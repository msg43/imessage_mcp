# CHANGELOG

Dated milestone history, newest-first. Terse bullets explaining the
**why**, not just the what — link commit hashes where useful. Add the
entry **in the same commit(s) as the work** whenever something notable
lands: a schema/migration change, a significant feature, a fix batch, a
gate transition. Skip the genuinely trivial (typo/format/lockfile bump);
when in doubt, add the line.

This is a running document, not a one-time artifact — status must never
live only in a chat transcript or an assistant's session memory.

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
