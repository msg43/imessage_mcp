"""S4 — sessionization + segmentation (SPEC §8 S4, D4).

Pure logic lives in `sessionize` (pass 1 + the incremental-frontier
fix), `boundaries` (pass 2: LLM boundary detection, windowing, hard
caps), `render` (SPEC §9.1 rendered text), and `hashing`
(`seg_config_hash` / `stable_key`) — all unit-testable without a
database. `pipeline` is the only module that talks to Postgres.
"""

from __future__ import annotations
