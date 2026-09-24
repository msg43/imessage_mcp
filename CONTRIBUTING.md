# Contributing

Thanks for looking at this project. This guide covers dev setup and the
rules that keep the repo safe to publish.

## Dev setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependencies.

```bash
uv sync --extra dev
```

That installs the project plus test tooling (`pytest`, `ruff`, `mypy`).
See `pyproject.toml`'s `[project.optional-dependencies]` for the other
extras (for example `models`), which you only need if you're working on
those areas.

### Running tests without a database

The unit test suite never needs a database and should pass right after
`uv sync --extra dev`:

```bash
uv run ruff check . && uv run mypy . && uv run pytest
```

Integration tests talk to a real PostgreSQL instance when one is
reachable and skip cleanly when it isn't, so you don't need Postgres
running just to contribute a fix.

### Postgres on macOS

If you do run Postgres locally on macOS, set `LC_ALL=C` first:

```bash
export LC_ALL=C
```

Without it, the postmaster can die at startup with an error that looks
like a corrupt installation — it isn't; it's a locale mismatch.

See [`docs/install-macos.md`](docs/install-macos.md) for the full local
setup walkthrough, and `scripts/doctor.py` for a script that checks your
environment for common setup problems.

## The public-safe rule

This repository is public. Every change to it must stay public-safe:

- **Use fictional personas only** — Alice, Bob, Acme Construction, and
  similar made-up names, numbers, and companies. Never a real name,
  phone number, email address, hostname, project id, bucket name, or
  business name.
- This applies everywhere: code, comments, tests, fixtures, docstrings,
  example config, **and commit messages**. A commit message that quotes
  a real name or number is just as much a leak as a line of code.
- `scripts/check_public_safety.py` scans changes for likely real data
  before they land, but it's a backstop, not a guarantee — it can't
  catch everything (a single common first name, for example). Review
  your own diff with this rule in mind before you open a pull request.

If you're working from a private design document or a real bug report,
carry over the underlying idea, not the real-world example that
illustrated it.

## Migrations are immutable once applied

Never edit a migration that has already been applied (including ones
already merged to `main`). If you need to fix a mistake, write a new,
later migration that corrects it. The migration runner enforces this by
hash, so an edited migration will fail rather than silently apply.

## Before changing retrieval behavior

If your change touches retrieval or segmentation (ranking, thresholds,
fusion, reranking), run the eval harness first and record the baseline
numbers, then run it again after your change. Retrieval quality claims
need numbers, not intuition — an eval diff is the expected justification
for a change like this, not just "it looks better."

## CHANGELOG entries

Add an entry to `CHANGELOG.md` for anything notable: a schema or
migration change, a real feature, a meaningful fix, or a gate
transition. Add it in the same commit as the work, dated, newest first,
and explain briefly *why* the change was made, not just what changed.
Trivial changes (typos, formatting, lockfile bumps) don't need one.

## Pull requests

Use the pull request template's checklist — it covers the public-safety
review and a couple of other easy-to-forget items. Keep PRs focused on
one change so the review (and the public-safety check) stays meaningful.
