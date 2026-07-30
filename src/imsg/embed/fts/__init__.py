"""The SQLite FTS5 sidecar (SPEC §7.3, D2) — rebuildable at any time
from Postgres, never the source of truth. `schema` owns the DDL and
version fingerprint; `sync` consumes `search_index_event` in
`event_id` order (the outbox, D6); `rebuild` regenerates the whole
sidecar from a Postgres snapshot when the schema/tokenizer config
changes.
"""

from __future__ import annotations
