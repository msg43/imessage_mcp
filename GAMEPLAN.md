# GAMEPLAN

The ONLY status document for this repo. Edited **in place**: never fork a
dated copy (`GAMEPLAN-2026-01.md`, `GAMEPLAN-old.md`, etc.). Flip status in
the same commit that changes it. The moment any other document starts
answering "what is the current status," that content belongs here instead.

Cold start order: this file, then `CLAUDE.md`, then the module you're touching.
Detail lives elsewhere: dated history in `CHANGELOG.md`; the retired status text
and old owner to-dos, as written through 2026-11-11, in
[`docs/archive/gameplan-detail-2026-11-23.md`](docs/archive/gameplan-detail-2026-11-23.md)
(read a line range, never the whole file); the governing spec, decision records
and the deployment guide in the private design-record repo (they name real hosts
and accounts, so they are not here).

## Goal

A local-first iMessage retrieval system. The full corpus is indexed locally on an
encrypted volume; a filtered subset can be exported to an external search index.
Answers come from hybrid retrieval (full text, semantic, rerank) through two MCP
surfaces, a local one and a public one behind a tunnel.

## Current status (2026-11-23)

- **Deployed.** The production host runs `main` at `40dd39f`, the same commit as
  GitHub `main`, with a clean checkout (checked 2026-11-23). CI on that commit
  passed on 2026-11-11. Full suite with no database, on the branch tip,
  2026-11-23 (Python 3.14): 2,433 passed, 730 skipped, 0 failed; Ruff clean.
- **Background work is running.** The photo-import gate closed on 2026-10-26 (the
  owner confirmed; the 2026-10-28 deploy record says so). The last "heavy work
  paused" sync exit (76) was 2026-10-28. Since 2026-10-29 every scheduled sync has
  exited 0, or 75 when yielding to memory, except one exit 1 (2026-11-06) and one
  killed run (2026-11-11); neither cause was checked.
- **Search models stay loaded.** `mcp.public.idle_unload_seconds` is 0 in the host
  config (checked 2026-11-23; in place since 2026-10-29). Captions run only in the
  01:00-09:00 window (`com.imsgindex.night-captions`, `scripts/run_night_captions.sh`);
  during it the public server's models are held unloaded and a search answers
  `WARMING_UP`. The host config names the bf16 reranker directory (checked 2026-11-23).
- **Off-box backup since 2026-10-27.** A weekly LaunchAgent (Sunday 06:30) encrypts
  the newest complete nightly set plus `private/` with `age`, copies it to the NAS,
  reads it back to verify and keeps 8 weeks; a restore test from the NAS copy
  passed 2026-10-27 (`CHANGELOG.md`). It covers the database and config, not
  attachments. The owner stored the decryption key on 2026-10-31 (recorded in the
  private design-record repo). `imsg status` does not read the off-box job's result
  (no reference to it in `diagnostics.py` or `cli.py`, checked 2026-11-23).
- **Built, not installed.** `imsg backup-check [--notify]` and the optional agent
  `backup-check` (`imsg install-agents --only backup-check`, daily 10:00) turn
  `imsg status`'s `backup_stale` into an exit status and a notification
  (`CHANGELOG.md` 2026-11-23). It also alerts when the encrypted volume is not
  mounted: that agent is rendered with `--no-mount-gate`, because the supervisor's
  mount wait would exit before the check ran. Installing the agent on the host needs
  the owner's go-ahead: render the plist to a scratch directory and review it first.
- **Eval and gates.** Per the private design-record repo (not re-measured here): a
  retrieval baseline exists, the Phase 0, 2, 3, 4 and 5 gates are closed (Gate
  ladder), and captions, measured on 2026-11-11, bought nothing detectable at that
  eval's resolution, so no further caption effort is planned.

## Strategy and constraints

**Approach.** Stages run in order: snapshot, extract, identity, segment, embed, with
attachment enrichment alongside. They fill Postgres and a full-text index under
`data_root`. Retrieval fuses full text and semantic hits, then reranks. The export
gate is default-deny. Phases 6 to 8 (public tunnel, export, side-by-side eval) wait
on the owner (Gate ladder).

**Rules that must not be broken.** `CLAUDE.md` holds them: public-safe by
construction (no real names, hostnames, credentials or instance config in this
repo), the six non-negotiables (never write the live `chat.db`; derived state on
encrypted storage; identity before segmentation; subject validation fails closed;
default-deny export; dedicated Postgres), and applied migrations are immutable.
Also standing:

- **The off-box backup's decryption key is never copied onto the index host or the
  NAS.** The owner keeps it in a password manager; without it the NAS copies cannot
  be decrypted.
- **Deployment is a human-gated sequence.** An agent renders and reviews plists in a
  scratch directory; loading them on the host is the owner's step.
- **Grounded bug and security fixes are authorized.** Larger features go on the
  roadmap as ownership questions with evidence, scope and dependencies. Evaluation
  and external-exposure gates stay in force.
- **Model loads ask first.** Every model load asks the host whether it has room, model-heavy
  commands take a host-wide lock, MCP servers load before background work, and sync
  waits while a search runs (`CHANGELOG.md` 2026-09-24 to 2026-09-26). Do not bypass
  admission to load a model.
- **Search budget:** whole-query p95 of 2.0 s or less (`config.example.yaml`, retrieval
  section; measured with `scripts/bench_retrieval_latency.py`). Read the README's
  measurements before changing `rerank_top` or `rerank_doc_max_tokens`.

**Decisions taken, with reasons.**

- 2026-09-17: the reranker is the 0.6B model, not the 8B, to meet that budget; the 8B and the
  mxfp8 build stay in `models/manifest.lock.yaml` as `retained`. Since 2026-09-25 the
  bf16 build runs, reranking in about 40 % less time at the same ranking quality. Rollback: set
  `retrieval.reranker_model` to `models/qwen3-reranker-0.6b-mxfp8-e61197ed`.
- 2026-09-23 (D12): extraction merges only add. A seed inserts rows and fills empty values;
  only this machine's own live `chat.db` may replace one.
- 2026-09-23 (D13): every message is filed, into the chat its evidence names or a
  holding chat that `allowlist` scope and export deny.
- 2026-09-23: link-preview images get OCR but no caption. 2026-10-29: captions run only in
  the 01:00-09:00 window and the round-the-clock runner is retired.
- 2026-10-27: a weekly encrypted off-box copy was added to the nightly local set. Neither
  covers attachments, models or `ops/` (`OUT_OF_SCOPE_NOTE`, `src/imsg/backup/pipeline.py`).
- 2026-10-31: the public GitHub repository was deleted and recreated (done 2026-11-01),
  because the 2026-09-24 history rewrite left the old commits fetchable by exact id.
  Reproduced 2026-11-23: 81 of 81 rewritten commit ids are gone and the 19 unchanged ids
  still resolve. Details: `CHANGELOG.md` 2026-10-31.

**Pitfalls that could recur.**

- A stage can print success, exit 0 and persist nothing (`CHANGELOG.md` 2026-08-12 and
  2026-08-17). Check the rows, not the exit code. This file once read "code complete,
  unrun" for three weeks after that stopped being true: keep it current.
- `models.backend: fake` reports success with meaningless results. Every command prints
  which backend it ran.
- Whoever changes what a segment renders must bump `message.updated_at`, or the chat is
  never re-segmented (the 2026-11-11 enrichment fix; `find_dirty_chats`).
- Vector dimensions are load-bearing: pgvector's HNSW index caps below what the models
  emit, so a change needs a migration and a full re-embed (`scripts/lint_ddl.py`).
- The retrieval baseline's labels are an agent's judgments, not the owner's, and its success
  rate hides misses of the known answer (per the design-record repo, 2026-11-10).
- Typing debt: the optional mypy run in CI reports errors, mostly in tests and scripts, and is
  non-blocking (`CHANGELOG.md` 2026-10-15; the triage is in the archive file). It is not a
  diagnosis of any runtime fault.

## In progress and next

1. Install the `backup-check` agent on the host (owner's go-ahead; see Current status).
2. Re-measure the public server's memory with the bf16 reranker, and confirm its speed with
   `scripts/bench_query_stages.py`. `memory.model_footprints.public_server_bytes` (17.2 GiB)
   and `memory.mlx_memory_limits.public_server_bytes` (12 GiB) were measured with the mxfp8
   build (`CHANGELOG.md` 2026-09-25); no later measurement is recorded there (searched 2026-11-23).
3. Owner-gated: Phase 1 exit (seed completeness, AT-2, and an owner-checked `person` table;
   the `imsg identity merge-filtered-twins` run on the production index is tracked in the
   private design-record repo); Phase 6 acceptance (a pilot with one outside user, tracked
   in the private design-record repo); Phase 7 needs a cloud destination the owner chooses.

**Open decisions (owner unless noted).**

- The older unencrypted index copy on the NAS: the owner decided on 2026-10-31 it goes
  on the roadmap, not urgent. Nothing is touched until it is scheduled (tracked in the
  private design-record repo).
- Whether the preserved alternative segmentation experiment gets its own quality and
  throughput evaluation before any algorithm change. It is excluded from the availability
  integration; no benchmark shows it improves the production corpus.
- The superseded remote branches `codex/october-availability` and
  `codex/doctor-config-credentials` (their content was ported to `main` in `c056cb5`) and the
  deploy-record branch `agent/item-4.5-deploy-record-20261028` (its changelog entries are now
  in `CHANGELOG.md`) are still on GitHub. Deleting them is the owner's call.
- Optional host hardening, not checked as done: password (SCRAM) authentication for TCP and
  `peer` authentication for the owner-only socket. (The one-off `VACUUM (ANALYZE)` that
  was on this list completed on 2026-10-28; `CHANGELOG.md`.)

## Gate ladder

Phases 6 and 7 **must not begin** until Phase 4 produces baseline numbers. That
dependency is the whole reason the eval harness exists.

1. [x] Phase 0: Host + encrypted volume. Volume auto-mounts at login; `guard-mount` passes; risk acceptance recorded (per the design-record repo, 2026-11-11; not re-measured)
2. [ ] Phase 1: Snapshot + extract + identity. Seed completeness passes; person table hand-verified
3. [x] Phase 2: Attachment backfill. Reconciliation report produced; gaps enumerated (per the design-record repo, 2026-11-11; not re-measured)
4. [x] Phase 3: Segment + embed + local MCP. **Real model providers replace the fakes**; answers real questions (per the design-record repo, 2026-11-11; not re-measured)
5. [x] Phase 4: Eval harness. At least 30 real queries and 100 pooled judgments; baseline recorded ⚠️ **gates 6 and 7** (per the design-record repo, 2026-11-11; not re-measured)
6. [x] Phase 5: Enrichment. Attachment text searchable; OCR bake-off pins the model (per the design-record repo, 2026-11-11; not re-measured)
7. [ ] Phase 6: Tunnel + OAuth + subject validation. Isolation test passes; scope set per the probe result
8. [ ] Phase 7: Allowlist + export + ingestion. Pre-push review completed and signed off
9. [ ] Phase 8: Side-by-side eval. Local vs. hosted scored on the real query set

## Document ledger

| Document | Purpose |
|---|---|
| `GAMEPLAN.md` (this file) | Current status, always edited in place |
| `CHANGELOG.md` | Dated history, append-only, newest first |
| `CLAUDE.md` | Standing instructions for any agent working here |
| `docs/archive/gameplan-detail-2026-11-23.md` | Retired status text and owner to-dos, as written through 2026-11-11 |
| `migrations/` | Schema, applied in order by a hash-checked runner. Applied migrations are immutable: correct forward, never edit |
| `scripts/lint_ddl.py` | Asserts pgvector's real index caps; a column can be legal DDL whose ANN index can never be created |
| `scripts/check_public_safety.py` | Scans files and commit messages for real contact data before it reaches the public repo |

**Governing spec and decision records live in the private design-record repo.** This
repo is public-safe by construction and must never carry real names, hostnames,
credentials, or instance configuration.
