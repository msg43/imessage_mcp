# GAMEPLAN

The ONLY status document for this repo. Edited **in place** — never fork
a dated copy (`GAMEPLAN-2026-01.md`, `GAMEPLAN-old.md`, etc.). Flip status
in the same commit that changes it: a gate passes, an owner item resolves,
a deviation gets ratified. The moment any other document starts answering
"what is the current status," that content belongs here instead.

Read this file first, before anything else in the repo — cold start
order is: this file, then `CLAUDE.md`, then the module you're touching.

## Current status

**Running against a real corpus; Phase 1 in progress.** Public at
[`msg43/imessage_mcp`](https://github.com/msg43/imessage_mcp) (MIT).
Every buildable component of the governing spec is implemented: 8
pipeline stages, hybrid retrieval, both MCP surfaces, the export gate,
the eval harness, 34 CLI commands (counting subcommands), migrations
0001–0006. 1,952 tests, all passing against a scratch Postgres (measured
2026-09-23); ruff and mypy strict clean; DDL lint clean. Since 2026-09-23
extraction merges only add (D12): a seed inserts rows and fills empty
values, and only this machine's own live `chat.db` may replace a value.
`imsg identity merge-filtered-twins` (2026-09-23) merges the persons that
the pre-fix import split on iOS filter tags; it has not yet run on the
production index.

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
in the lock as `status: retained` for the Phase 4 quality comparison).
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
