#!/usr/bin/env python3
"""CLI entry point for the model smoke test. See
`imsg.providers.model_smoke` for the implementation.

Usage: `uv run python scripts/smoke_test_models.py [--lock PATH]
[--only ENTRY]... [--role ROLE]... [--skip-download] [--write]
[--work-dir DIR] [--report-json PATH] [--in-process]` — for every entry
of `models/manifest.lock.yaml`: download the pinned snapshot, compute
its `artifact_sha256`, build the real provider through
`imsg.providers.factory` and run one synthetic input, and record load
time, inference time and peak memory. Each (entry, role) runs in its
own child process. Never modifies the lock unless `--write` is given.
"""

from __future__ import annotations

from imsg.providers.model_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
