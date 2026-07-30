# imessage-index

A local-first iMessage retrieval system: full corpus indexed locally,
with a filtered subset exportable to an enterprise search index.

## This repo is public-safe by construction

**The single rule that governs everything here.** This repo is intended
to be published. It must never contain:

- Real names, phone numbers, email addresses, hostnames, project ids,
  bucket names, or business names — in code, comments, tests, fixtures,
  docstrings, or example config. Use fictional personas (Alice, Bob,
  Acme Construction) exclusively.
- Credentials of any kind. Secrets come from the OS keychain or the
  environment, never from a file in either repo.
- Instance configuration. Real values live in a **separate private
  overlay** that the core reads at runtime; `config.example.yaml` ships
  here with placeholders only.

If you are reading from a private design document to implement
something, carry the *design* across and not the *examples* — those
documents contain real names by design. Sweep before committing.

## Non-negotiables

1. **Never write to the live `chat.db`.** Snapshot via SQLite's
   online-backup API; never `cp` (WAL correctness). All derived state
   lives elsewhere.
2. **All derived state under `data_root`**, on the encrypted volume.
   Startup gates on the mount.
3. **Identity resolution precedes segmentation.** Nothing downstream
   keys on a raw handle; everything keys on `person_id`.
4. **Subject validation on the public surface is the only access
   control.** Fail closed. No configuration path disables it. Audience
   validation and subject validation are *not* redundant — subject
   answers "is this the owner?", audience answers "was this token minted
   for this system?", and only both together stop cross-application
   replay.
5. **Default deny on export.** Group threads require every participant
   allowlisted; attachments are gated separately from text bodies.
6. **The Postgres instance is dedicated.** Verified by a two-sided
   cluster fingerprint with no bypass.

## Conventions

- Python 3.12+, `uv` for dependencies.
- Extraction shells out to `imsg-dump` (GPL — process boundary only,
  never linked).
- Load config once via `imsg.config.loader.load_config()`; pass the
  object down. All path containment goes through `imsg.paths`, never
  string-prefix comparison.
- DB code takes an already-open `psycopg.Connection` and never owns its
  lifecycle. Raise `imsg.errors.ImsgError` subclasses; the CLI boundary
  catches them.
- **Applied migrations are immutable** — the runner enforces this by
  hash. Correct a mistake in a *later* migration, never by editing a
  shipped one.
- Vector dimensions are load-bearing. pgvector's HNSW index caps at
  2,000 dimensions for `vector` and 4,000 for `halfvec`, while the
  types themselves allow 16,000 — so a column can be perfectly legal
  DDL whose index can never be created, failing at `CREATE INDEX` and
  silently degrading to sequential scan. `scripts/lint_ddl.py` exists to
  prevent exactly that.

## Before making retrieval changes

Run the eval harness first and record the baseline. Retrieval quality
claims need numbers, not intuition — segmentation thresholds are frozen
until a baseline exists, and the eval diff is the only sanctioned
justification for changing them.

## Testing

Unit tests must pass with no database. Integration tests skip cleanly
when Postgres is unreachable and run for real when it is. On macOS,
Postgres needs `export LC_ALL=C` or the postmaster dies at startup with
an error that looks like a corrupt installation.

## Maintaining the living documents

`CHANGELOG.md` and `GAMEPLAN.md` are **running documents, not one-time
artifacts**, and status must never live only in a chat transcript or an
assistant's session memory.

- **`CHANGELOG.md`** — dated milestone history, newest-first, terse
  bullets explaining the *why*, not just the *what*. Add the entry **in
  the same commit(s) as the work** whenever something notable lands: a
  schema/migration change, a significant feature, a fix batch, a gate
  transition. Skip the genuinely trivial (typo/format/lockfile); when in
  doubt, add the line.
- **`GAMEPLAN.md`** — the ONLY status document: current gate, gate
  ladder, owner to-dos, document ledger. Edited **in place** — never
  fork a dated copy. Flip status in the same commit that changes it.
  The moment any other document starts answering "what is the current
  status," that content belongs here instead.
- **Close-out checklist** (run it when a unit of work lands, not
  "later"): (1) CHANGELOG entry; (2) GAMEPLAN status flip if the status
  changed.

This is **nudged mechanically** (it can't be fully verified by a hook —
notability and "is it done" are judgment calls): `scripts/
check_doc_lifecycle.sh` prints a pre-commit WARNING when a migration or
new non-test source file is staged without `CHANGELOG.md`, and when a
report-shaped doc lands outside an archive location. The warning never
blocks — don't train yourself to ignore it; satisfy it on the landing
commit.
