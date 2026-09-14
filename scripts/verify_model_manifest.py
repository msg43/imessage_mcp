#!/usr/bin/env python3
"""CLI entry point for the model-manifest verifier. See
`imsg.providers.manifest` for the implementation (also reachable as
`imsg models verify`).

Usage: `uv run python scripts/verify_model_manifest.py [--lock PATH]
[--write] [--skip-remote] [--skip-runtime]` — re-resolves every pinned
Hugging Face repo's current revision and license and reports drift
against `models/manifest.lock.yaml`, and checks the installed runtime
packages against each entry's `min_runtime`. Never modifies the lock
unless `--write` is given.
"""

from __future__ import annotations

from imsg.providers.manifest import main

if __name__ == "__main__":
    raise SystemExit(main())
