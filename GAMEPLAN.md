# GAMEPLAN

The ONLY status document for this repo. Edited **in place** — never fork
a dated copy (`GAMEPLAN-2026-01.md`, `GAMEPLAN-old.md`, etc.). Flip status
in the same commit that changes it: a gate passes, an owner item resolves,
a deviation gets ratified. The moment any other document starts answering
"what is the current status," that content belongs here instead.

Read this file first, before anything else in the repo — cold start
order is: this file, then `CLAUDE.md`, then the module you're touching.

## Current status

**Code complete, unrun.** Every buildable component of the governing
spec is implemented: 8 pipeline stages, hybrid retrieval, both MCP
surfaces, the export gate, the eval harness, 18 CLI commands, migrations
0001–0003. 836 tests; ruff and mypy strict clean; DDL lint clean.

Nothing has executed against a real corpus. The pipeline runs end to end
on **deterministic fake model providers** (marked `PLACEHOLDER`), which
means it completes successfully and the results are meaningless until
real loaders replace them. Deployment is blocked on hardware,
permissions, and credentials — not on code. (2026-07-30)

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

- **Replace the fake model providers** before trusting Phase 3. The
  interfaces exist and are correctly dimensioned — the loaders do not.
  Vector dimensions are load-bearing: pgvector's HNSW index caps below
  what the models natively emit, so changing them requires a migration
  and a full re-embed.
- Confirm the host's unified memory before Phase 0 — it selects the
  model ladder.
- Populate the `unsupported` materialization state in the backfill
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
