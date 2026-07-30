"""Pipeline-stage implementations (SPEC §8).

Each stage is a plain, importable, independently testable module — a
CLI subcommand wires it up later (see `imsg.cli`'s stub docstring: "the
subcommand names and signatures here are the contract those builds
implement against"). None of the stage modules import `typer` or touch
`imsg.cli`; they take already-loaded `Config` objects and, where they
need Postgres, an already-open `psycopg.Connection` — never their own
config load or connection lifecycle (see each module's docstring).
"""

from __future__ import annotations
