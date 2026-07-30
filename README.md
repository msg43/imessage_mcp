# imessage-index

A local-first iMessage retrieval system: extract, resolve identities,
segment conversations, embed and index, and serve hybrid (full-text +
vector) search over your own message history via a scoped MCP server.

Designed to run unattended on a headless Mac, with everything derived
kept on an encrypted volume and nothing leaving the box unless you
explicitly allowlist it.

## Status

This repository is the **core** codebase: pipeline code, migrations,
the MCP server, and the CLI. It is designed to be public-safe by
construction — no real names, hosts, or secrets ever belong here (see
`config.example.yaml` and `CONTRIBUTING` notes below). Instance-specific
configuration (real config values, contact seed data, eval queries)
lives in a separate, private overlay that you supply and point the CLI
at via `IMSG_CONFIG`.

## Requirements

- macOS on Apple Silicon (the pipeline relies on macOS-only APIs:
  Full Disk Access to Messages, the Contacts framework, Apple Vision
  OCR).
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).
- PostgreSQL 17 with the `pgvector` extension, running as a dedicated
  instance (see `docs/` in your deployment for setup — not part of
  this repo).

## Getting started

```bash
uv sync --all-extras
cp config.example.yaml config.yaml   # then edit with your real values
export IMSG_CONFIG=$(pwd)/config.yaml
uv run imsg check-permissions
uv run imsg status
```

`config.yaml` (or any file with real values) must never be committed —
see `.gitignore`. All secrets are resolved from the macOS Keychain
(`keychain:<item-name>`) or environment variables (`env:<VAR>`); the
config file itself never holds a literal secret.

## Layout

```
src/imsg/          application code (config, mount gate, db/migrations, CLI)
migrations/        ordered, immutable SQL migrations (Postgres + pgvector)
scripts/           standalone tooling (DDL lint, etc.)
tests/             pytest suite — no network, no live database required
config.example.yaml  placeholder config showing the full schema
```

## CLI

`imsg --help` lists every subcommand. Pipeline stages
(`snapshot`, `extract`, `identity`, `segment`, `embed`, `sync`,
`enrich`, `backfill-attachments`, `export`, `install-agents`) are
under active development and currently exit with a clear
"not implemented yet" error naming the stage. `migrate`,
`check-permissions`, and `status` are functional today.

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run mypy .
uv run pytest
```

## License

MIT, except any GPL-licensed subprocess shims that may be added under
their own subdirectory with their own `LICENSE` file (kept at
arm's length as a subprocess dependency, never linked into this
codebase).
