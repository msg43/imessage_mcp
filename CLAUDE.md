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
2. **All derived state under `data_root`**, on encrypted storage: either
   a FileVault-encrypted startup (boot) disk, or a separate encrypted
   volume. Startup gates on it.
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

# GAMEPLAN.md is the cold-start file

A new session should get the project's direction from `GAMEPLAN.md` for a few thousand tokens, instead of carrying the history in a long conversation. That only works if the file is short, keeps everything that still shapes future work, and is updated on a schedule.

**What goes in it** (about 200 lines in all, roughly 6K tokens)
- The goal, and the current state in a few lines.
- **Strategy and constraints:** the approach and why it was chosen, the rules that must not be broken, and the known pitfalls. This section is required.
- What is in progress, the next three steps, and decisions still open (each with its owner).
- Decisions taken, each with its reason and date, so a restarted session does not re-argue them.
- Where to look for detail (code map, changelog, archive sections by heading and line).

**Keep or archive test:** if something still shapes future work (a strategy, a constraint, a decision's reason, an unresolved item, a pitfall that could recur), it stays in `GAMEPLAN.md`. Only finished work, evidence and superseded material move to `CHANGELOG.md` or `docs/archive/`, and the opening section links to it. When in doubt, keep the one-line version and link the detail.

**When to update it**
- When a task finishes, a decision is made, or a blocker appears, in the same change.
- Before a session ends, and before a restart or compaction. Write the handoff first: what is done, what is not, and what waits on Matthew.
- When the context advisory fires (400K tokens), as a checkpoint.
- Resolved items leave in the same update, so the file does not grow.

**Size check:** if the file is past about 200 lines or 30K tokens, archive per the test above before doing anything else. Each time you archive, read what you removed once and confirm nothing that still shapes future work went with it.

# Keep sessions cheap to start and to run

Every turn re-reads the whole conversation, so a session that starts small and stays small costs far less.

- **Start from `GAMEPLAN.md`.** Read it first; its opening section is written to be enough to begin work. Read only the top of `CHANGELOG.md` (newest entries come first, so about the first 100 lines). Use grep to find older entries. Never read a whole changelog or an archive file; read the line range you need.
- **Changelog entries are short.** About five lines each: what changed, why, and where to look. Detail belongs in the commit message or the pull request. The changelog is append-only history and is never trimmed or archived: its size costs nothing as long as sessions read only the top and grep for the rest. Use `grep -m` or a few lines of context so a search returns a short result.
- **Grep before reading.** Search for the name, then read only the lines around it. Send a large file, log or batch of sessions to a small helper agent that returns a short answer, as the helper-agent rule describes.
- **Restart before the context gets heavy.** At about 400K tokens, update `GAMEPLAN.md` with the handoff (done, not done, waiting on Matthew) and continue in a fresh session, as your global instructions describe.

# Recovering something deleted from GAMEPLAN.md

Git keeps every version of `GAMEPLAN.md`, so do not save dated copies. To find a strategy, constraint, decision or to-do that went missing:

- Find when a phrase left the file: `git log -S"some phrase" --oneline -- GAMEPLAN.md`. The newest commit listed is usually the one that removed it.
- Read the file as it was then: `git show <commit>:GAMEPLAN.md`, or see exactly what a commit removed: `git show <commit> -- GAMEPLAN.md`.
- Sweep for anything lost: list the lines each commit removed with `git log -p --format="== %h %ad %s" --date=short -- GAMEPLAN.md`, then check each removed line against the keep-or-archive test in this file and restore what is still live. Sweep by time window (for example one month per helper agent) and write findings to a file, as the helper-agent rule describes.

