#!/usr/bin/env python3
"""CLI entry point for the DDL lint. See `imsg.ddl_lint` for the implementation.

Usage: `uv run python scripts/lint_ddl.py [migrations_dir]` (defaults to
the repo's own `migrations/`).
"""

from __future__ import annotations

from imsg.ddl_lint import main

if __name__ == "__main__":
    raise SystemExit(main())
