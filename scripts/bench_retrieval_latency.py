#!/usr/bin/env python3
"""Benchmark `search_messages` latency on the real index, stage by stage,
and sweep the reranker's pool size and document cap.

The queries are FICTIONAL and generic (`QUERIES` below); the index they
run against is real. Results — segment keys and text — are held in
memory only, to compare rankings between configurations, and nothing but
timings, token counts and aggregate numbers is printed or written. Every
Postgres transaction is read-only (`default_transaction_read_only` is set
on the session before anything else runs) and the FTS sidecar is opened
immutable (`imsg.sqlite_readonly`), so a run writes nothing anywhere.

Usage (the project venv with the `models` extra; `IMSG_CONFIG` names the
instance config):

    python scripts/bench_retrieval_latency.py
    python scripts/bench_retrieval_latency.py --configs new:10:256,new:15:512 --passes 2

What a run does:

1. Builds every real provider through `imsg.providers.factory` — the
   path `imsg mcp local` takes — and wraps each in a timer; opens the
   dedicated Postgres instance (data-directory check) and the sidecar.
2. Loads all weights and runs `--warmup` queries that are not in the
   query set through the whole service (kernel compilation, first-call
   allocations).
3. First-touch pass (unless `--no-first-touch`): every query once, with
   the cheapest configuration, to measure the non-rerank stages on
   queries this process has not seen — and to leave every configuration
   of the sweep the same warm pages to run against.
4. The sweep: each configuration runs every query `--passes` times
   through `RetrievalService.search_messages(limit=10)`; per query it
   records each stage's time (the service's own module functions and
   providers, wrapped) and the whole call.

Configurations are `<scorer>:<rerank_top>:<doc cap | none>[:nomm]`:

- `new` — the pinned reranker (`retrieval.reranker_model`) with its
  planned batching;
- `old` — the pinned reranker in fixed groups of 8 pairs in pool order,
  the behaviour before 2026-09-16, rebuilt from the provider's own
  `token_rows` / `score_token_rows`, so it scores exactly the same rows
  and differs only in how they are grouped;
- `new@<label>` — another reranker checkpoint, registered with
  `--reranker <label>=<directory>` (a directory relative to
  `paths.data_root`, or absolute). For measurement only: nothing about
  the pinned model changes;
- `none` — no reranker: the fused (RRF) order, scored by the eval
  harness's pass-through;
- a trailing `:nomm` skips channel C (the multimodal text tower and its
  vector search) for that configuration.

The service never reranks fewer candidates than the request's `limit`
(10 here), so a `rerank_top` below 10 behaves as 10.

`--scope allowlist` runs the sweep as the public MCP server does under
`mcp.public.scope: allowlist` (`AccessContext(surface="public",
scope="allowlist")`): export's eligibility rule evaluated per request
(the `scope` stage), candidates limited to eligible chats, and the
reranked pool re-rendered through the attachment gate (the
`render_scoped` stage). With no eligible chat every search returns at
once, empty, so it measures something only once `allowlist_person` has
rows. It cannot be combined with `--rerank-only`, whose captured pools
are the stored segment texts.

`--pooled` runs the sweep the way `imsg mcp public` runs searches since
2026-09-24: every call borrows its own Postgres connection and a search's
five candidate channels run side by side on connections of their own
(`imsg.retrieval.connections`). Without it, the service runs on one shared
connection pair, channels one after another, as before. Stage times then
overlap, so under `--pooled` they add up to more than the total and
"other" goes negative; the total is the number to compare.

`--fake-model-delays EMBED,MM,RERANK` (milliseconds) is for measuring on a
host without the models (`models.backend: fake` in the config): each fake
provider call sleeps that long first — for example the production host's
measured p50s — so the database stages overlap with model time the way
they would with real models. The fake backend's search results are
meaningless; use it for timing only.

`--rerank-only` isolates the reranker from the database: one pass of the
real service per query (a pass-through reranker, a 50-candidate pool, and
channel C on or off as the configurations need) records each query's
fused pool and its texts in memory, then every configuration reranks
those same pools — the service's own selection (`max(rerank_top,
limit)` candidates in fused order) and ordering (a stable sort by score)
— so its rankings, and therefore the proxy, are exactly the service's,
while its timings are the rerank stage alone. Use it when disk or
database latency on the host would otherwise swamp the comparison, and
add the non-rerank stages measured separately.

Percentiles are nearest-rank over every recorded query (p95 of 20
queries is the 19th slowest).

QUALITY PROXY — a proxy, not an evaluation. The reference is
`new:50:none`, which scores exactly what the service scored before this
change. For each configuration: `ovl@10`, the mean overlap of its top 10
with the reference's top 10; `top3@10`, the mean fraction of the
reference's top 3 that appear in its top 10; `top3@3`, the same within
its own top 3 — the only one of the three that sees ordering when the
pool is no larger than the result list (a 10-candidate pool returns
exactly those 10, reranked or not). They say how far a configuration
moves the ranking from today's, not whether either ranking is right —
the Phase 4 eval decides that.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QUERIES: tuple[str, ...] = (
    "dentist appointment next week",
    "flight confirmation and boarding time",
    "receipt for the new couch",
    "dinner plans saturday night",
    "happy birthday wishes",
    "what is the address for the party",
    "running late be there in ten minutes",
    "weekend plans with the kids",
    "hotel reservation confirmation number",
    "doctor appointment reschedule",
    "rental car pickup",
    "wedding gift ideas",
    "school pickup time change",
    "restaurant recommendation downtown",
    "movie tickets for friday",
    "package tracking number",
    "vacation photos from the beach",
    "car repair estimate",
    "lunch meeting tomorrow",
    "house cleaning schedule",
)
"""Twenty generic, fictional topics — nothing here names a person, place
or event from any real corpus."""

WARMUP_QUERIES: tuple[str, ...] = (
    "grocery list for the week",
    "concert tickets in june",
    "haircut appointment",
)

STAGES: tuple[str, ...] = (
    "scope",
    "filters",
    "analyze",
    "segment_fts",
    "attachment_fts",
    "embed_query",
    "segment_vector",
    "attachment_vector",
    "mm_embed_text",
    "multimodal_vector",
    "fuse",
    "fetch_summaries",
    "render_scoped",
    "rerank",
)
NON_RERANK_STAGES: tuple[str, ...] = tuple(s for s in STAGES if s != "rerank")

DEFAULT_CONFIGS: tuple[str, ...] = (
    "new:50:none",
    "new:50:1024",
    "new:50:512",
    "new:50:256",
    "new:20:none",
    "new:20:1024",
    "new:20:512",
    "new:20:256",
    "new:15:none",
    "new:15:1024",
    "new:15:512",
    "new:15:256",
    "new:10:none",
    "new:10:1024",
    "new:10:512",
    "new:10:256",
    "old:50:none",
)
"""The sweep, in run order: the reference first, the (slow) old batching
last."""
REFERENCE_CONFIG = "new:50:none"


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepConfig:
    name: str
    batching: str  # "new" | "old" | "none"
    rerank_top: int
    doc_cap: int | None
    reranker: str | None = None
    """Label of an alternate reranker (`--reranker`); `None` is the pinned one."""
    multimodal: bool = True


def parse_config(name: str) -> SweepConfig:
    usage = "expected <new|new@label|old|none>:<top>:<cap|none>[:nomm]"
    parts = name.split(":")
    if len(parts) not in (3, 4) or (len(parts) == 4 and parts[3] != "nomm"):
        raise SystemExit(f"bad configuration {name!r}: {usage}")
    scorer, at, label = parts[0].partition("@")
    if at and not label:
        raise SystemExit(f"bad configuration {name!r}: {usage}")
    try:
        parsed = SweepConfig(
            name=name,
            batching=scorer,
            rerank_top=int(parts[1]),
            doc_cap=None if parts[2] == "none" else int(parts[2]),
            reranker=label or None,
            multimodal=len(parts) == 3,
        )
    except ValueError as exc:
        raise SystemExit(f"bad configuration {name!r}: {usage}") from exc
    if parsed.batching not in ("new", "old", "none") or parsed.rerank_top < 1:
        raise SystemExit(f"bad configuration {name!r}: {usage}")
    if parsed.reranker is not None and parsed.batching != "new":
        raise SystemExit(f"{name!r}: an alternate reranker takes the planned batching (new@label)")
    if parsed.batching in ("old", "none") and parsed.doc_cap is not None:
        raise SystemExit(f"{name!r}: only the planned batching has a document cap")
    if parsed.doc_cap is not None and parsed.doc_cap < 1:
        raise SystemExit(f"bad configuration {name!r}: the cap must be >= 1")
    return parsed


# --------------------------------------------------------------------------
# instrumentation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StageClock:
    """Seconds per stage for the query in flight."""

    current: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.current = {}

    def add(self, stage: str, seconds: float) -> None:
        self.current[stage] = self.current.get(stage, 0.0) + seconds


def timed[**P, R](clock: StageClock, stage: str, fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            clock.add(stage, time.perf_counter() - started)

    return wrapper


def instrument_service_modules(clock: StageClock) -> None:
    """Wrap the module-level functions `RetrievalService.search_messages`
    calls (it resolves them through their modules at call time)."""
    from imsg.retrieval import fts_search, segments, service, vector_search

    patches: list[tuple[Any, str, str]] = [
        (service, "resolve_request_scope", "scope"),
        (service, "resolve_filters", "filters"),
        (service, "compile_predicate", "filters"),
        (service, "analyze_query", "analyze"),
        (fts_search, "search_segment_fts", "segment_fts"),
        (fts_search, "search_attachment_chunk_fts", "attachment_fts"),
        (vector_search, "search_segment_vector", "segment_vector"),
        (vector_search, "search_attachment_chunk_vector", "attachment_vector"),
        (vector_search, "search_multimodal_vector", "multimodal_vector"),
        (service, "reciprocal_rank_fusion", "fuse"),
        (segments, "fetch_segment_summaries", "fetch_summaries"),
        (segments, "render_segment_texts", "render_scoped"),
    ]
    for module, name, stage in patches:
        setattr(module, name, timed(clock, stage, getattr(module, name)))


class TimedTextProvider:
    def __init__(self, inner: Any, clock: StageClock) -> None:
        self._inner = inner
        self._clock = clock
        self.model_id = inner.model_id
        self.dim = inner.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return list(self._inner.embed_documents(texts))

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        started = time.perf_counter()
        try:
            return list(self._inner.embed_query(text, instruction=instruction))
        finally:
            self._clock.add("embed_query", time.perf_counter() - started)


class TimedMultimodalProvider:
    def __init__(self, inner: Any, clock: StageClock) -> None:
        self._inner = inner
        self._clock = clock
        self.model_id = inner.model_id
        self.dim = inner.dim

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return list(self._inner.embed_images(image_paths))

    def embed_text(self, text: str) -> list[float]:
        started = time.perf_counter()
        try:
            return list(self._inner.embed_text(text))
        finally:
            self._clock.add("mm_embed_text", time.perf_counter() - started)


class FixedBatchReranker:
    """The reranker before 2026-09-16: the same rows, scored in fixed
    groups of `batch_size` in pool order."""

    def __init__(self, provider: Any, batch_size: int = 8) -> None:
        self._provider = provider
        self._batch_size = batch_size
        self.model_id = provider.model_id

    def score(self, query: str, documents: list[str]) -> list[float]:
        rows = self._provider.token_rows(query, documents)
        out: list[float] = []
        for start in range(0, len(rows), self._batch_size):
            out.extend(self._provider.score_token_rows(rows[start : start + self._batch_size]))
        return out


@dataclass(slots=True)
class RerankTally:
    pairs: int = 0
    real_tokens: int = 0
    padded_tokens: int = 0
    batches: int = 0


class TimedReranker:
    def __init__(
        self,
        scorer: Any,
        provider: Any | None,
        clock: StageClock,
        batching: str,
        tally: RerankTally,
    ) -> None:
        self._scorer = scorer
        self._provider = provider
        self._clock = clock
        self._batching = batching
        self._tally = tally
        self.model_id = scorer.model_id

    def score(self, query: str, documents: list[str]) -> list[float]:
        started = time.perf_counter()
        try:
            return list(self._scorer.score(query, documents))
        finally:
            self._clock.add("rerank", time.perf_counter() - started)
            self._count(query, documents)

    def _count(self, query: str, documents: list[str]) -> None:
        """Token accounting, outside the timed region. For the provider's
        own scoring, the passes its `plan` makes: the shared prompt prefix
        once when it reuses it, then the batches."""
        from imsg.embed.batching import padded_tokens

        if self._provider is None or not hasattr(self._provider, "token_rows"):
            return  # nothing reranked, or the fake backend (no tokenizer)
        rows = self._provider.token_rows(query, documents)
        lengths = [len(row) for row in rows]
        if not lengths:
            return
        self._tally.pairs += len(lengths)
        self._tally.real_tokens += sum(lengths)
        if self._batching == "old":
            plan = [list(range(i, min(i + 8, len(lengths)))) for i in range(0, len(lengths), 8)]
            self._tally.padded_tokens += padded_tokens(lengths, plan)
            self._tally.batches += len(plan)
        else:
            work = self._provider.plan(rows)
            self._tally.padded_tokens += work.padded_tokens
            self._tally.batches += work.passes


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def nearest_rank(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def overlap_at_10(ranking: Sequence[str], reference: Sequence[str]) -> float:
    ref = set(reference[:10])
    if not ref:
        return 1.0
    return len(set(ranking[:10]) & ref) / len(ref)


def top3_in_top10(ranking: Sequence[str], reference: Sequence[str]) -> float:
    ref3 = list(reference[:3])
    if not ref3:
        return 1.0
    return len(set(ref3) & set(ranking[:10])) / len(ref3)


def top3_in_top3(ranking: Sequence[str], reference: Sequence[str]) -> float:
    ref3 = list(reference[:3])
    if not ref3:
        return 1.0
    return len(set(ref3) & set(ranking[:3])) / len(ref3)


@dataclass(slots=True)
class QueryRecord:
    query_index: int
    stages: dict[str, float]
    total: float
    ranking: list[str]


# --------------------------------------------------------------------------
# machine load (the numbers are only comparable on an otherwise idle host)
# --------------------------------------------------------------------------

BIG_PROCESS_KIB = 4 * 2**20


def machine_load() -> str:
    """Other processes resident above 4 GiB (sizes only, no names), the
    kernel's "system-wide memory free percentage" (`memory_pressure -Q`,
    which counts reclaimable file cache as free), memory held by the
    compressor, and swap-outs since boot."""
    try:
        ps = subprocess.run(["ps", "-axo", "pid=,rss="], capture_output=True, text=True, check=True)
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        pressure = subprocess.run(
            ["memory_pressure", "-Q"], capture_output=True, text=True, check=True
        ).stdout
        mem = int(
            subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "machine load: unavailable"
    me = os.getpid()
    big = [
        int(rss)
        for pid, rss in (line.split() for line in ps.stdout.splitlines() if line.strip())
        if int(pid) != me and int(rss) > BIG_PROCESS_KIB
    ]
    page = int(m.group(1)) if (m := re.search(r"page size of (\d+) bytes", vm)) else 16384

    def pages(label: str) -> int:
        found = re.search(rf"{label}:\s+(\d+)", vm)
        return int(found.group(1)) if found else 0

    free = m.group(1) if (m := re.search(r"free percentage: (\d+)%", pressure)) else "?"
    return (
        f"machine load: {len(big)} other process(es) above 4 GiB resident "
        f"({sum(big) / 2**20:.1f} GiB); memory free {free} %, compressor "
        f"{100 * pages('Pages occupied by compressor') * page / mem:.0f} %, "
        f"swap-outs since boot {pages('Swapouts')}"
    )


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def open_connections(cfg: Any) -> tuple[Any, Any]:
    from imsg.db.connection import connect
    from imsg.db.fingerprint import verify_data_directory
    from imsg.embed.fts.schema import assert_schema_current
    from imsg.sqlite_readonly import open_readonly_immutable, wal_frame_bytes

    pg = connect(cfg.database)
    with pg.cursor() as cur:
        cur.execute("SET SESSION default_transaction_read_only = on")
    verify_data_directory(pg, cfg.paths.data_root)
    fts_path = cfg.paths.data_root / "fts" / "fts.db"
    if wal_frame_bytes(fts_path):
        raise SystemExit(
            "the FTS sidecar has un-checkpointed WAL frames; refusing an immutable open"
        )
    fts = open_readonly_immutable(fts_path)
    assert_schema_current(fts)
    return pg, fts


def load_providers(cfg: Any) -> tuple[Any, Any, Any]:
    from imsg.providers.factory import (
        build_multimodal_provider,
        build_reranker,
        build_text_provider,
    )

    text: Any = build_text_provider(cfg)
    reranker: Any = build_reranker(cfg)
    multimodal: Any = build_multimodal_provider(cfg)
    for name, provider in (("text", text), ("reranker", reranker)):
        load = getattr(provider, "load", None)
        if not callable(load):
            continue  # the fake backend holds no weights
        started = time.perf_counter()
        load()
        print(f"loaded {name} in {time.perf_counter() - started:.1f} s", flush=True)
    if multimodal is not None:
        started = time.perf_counter()
        multimodal.embed_text("warm-up")
        print(f"loaded multimodal in {time.perf_counter() - started:.1f} s", flush=True)
    return text, reranker, multimodal


def load_alternate_rerankers(cfg: Any, specs: Sequence[str]) -> dict[str, Any]:
    """`LABEL=DIR` checkpoints for `new@LABEL` rows, loaded now so the
    sweep never times a weight load."""
    from imsg.retrieval.mlx_reranker import MlxRerankerProvider

    out: dict[str, Any] = {}
    for spec in specs:
        label, sep, directory = spec.partition("=")
        if not sep or not label or not directory:
            raise SystemExit(f"--reranker expects LABEL=DIR, got {spec!r}")
        path = Path(directory)
        if not path.is_absolute():
            path = cfg.paths.data_root / path
        if not path.is_dir():
            raise SystemExit(f"--reranker {label}: {path} is not a directory")
        provider = MlxRerankerProvider(str(path), None, model_id=f"{Path(directory).name}@local")
        started = time.perf_counter()
        provider.load()
        print(f"loaded alternate reranker {label} in {time.perf_counter() - started:.1f} s")
        out[label] = provider
    return out


class DelayedProvider:
    """A provider whose every call sleeps `seconds` first
    (`--fake-model-delays`); everything else passes through."""

    def __init__(self, inner: Any, seconds: float) -> None:
        self._inner = inner
        self._seconds = seconds
        self.model_id = getattr(inner, "model_id", "delayed")
        self.dim = getattr(inner, "dim", None)

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        time.sleep(self._seconds)
        return list(self._inner.embed_query(text, instruction=instruction))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return list(self._inner.embed_documents(texts))

    def embed_text(self, text: str) -> list[float]:
        time.sleep(self._seconds)
        return list(self._inner.embed_text(text))

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return list(self._inner.embed_images(image_paths))

    def score(self, query: str, documents: list[str]) -> list[float]:
        time.sleep(self._seconds)
        return list(self._inner.score(query, documents))


def delay_fake_providers(
    cfg: Any, text: Any, reranker: Any, multimodal: Any, spec: str
) -> tuple[Any, Any, Any]:
    if cfg.models.backend != "fake":
        raise SystemExit("--fake-model-delays is for the fake backend only (models.backend: fake)")
    try:
        embed_ms, mm_ms, rerank_ms = (float(part) for part in spec.split(","))
    except ValueError as exc:
        raise SystemExit("--fake-model-delays expects EMBED,MM,RERANK in milliseconds") from exc
    return (
        DelayedProvider(text, embed_ms / 1000),
        DelayedProvider(reranker, rerank_ms / 1000),
        DelayedProvider(multimodal, mm_ms / 1000) if multimodal is not None else None,
    )


BENCH_POOL_PG_CONNECTIONS = 10
BENCH_POOL_FTS_CONNECTIONS = 4


def open_pooled_connections(cfg: Any) -> Any:
    """`--pooled`: connections opened exactly like `open_connections`'s —
    read-only sessions after the data-directory check, the sidecar
    immutable — sized like `imsg mcp public`'s pools."""
    from imsg.db.connection import connect
    from imsg.db.fingerprint import verify_data_directory
    from imsg.db.pool import postgres_pool
    from imsg.retrieval.connections import RetrievalConnections, fts_pool
    from imsg.sqlite_readonly import open_readonly_immutable

    def open_pg() -> Any:
        conn = connect(cfg.database)
        with conn.cursor() as cur:
            cur.execute("SET SESSION default_transaction_read_only = on")
        verify_data_directory(conn, cfg.paths.data_root)
        return conn

    fts_path = cfg.paths.data_root / "fts" / "fts.db"
    return RetrievalConnections(
        pg=postgres_pool(open_pg, max_size=BENCH_POOL_PG_CONNECTIONS, name="bench"),
        fts=fts_pool(
            lambda: open_readonly_immutable(fts_path), max_size=BENCH_POOL_FTS_CONNECTIONS
        ),
    )


def with_retrieval(cfg: Any, **updates: Any) -> Any:
    return cfg.model_copy(update={"retrieval": cfg.retrieval.model_copy(update=updates)})


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Bench:
    cfg: Any
    pg: Any
    fts: Any
    text: Any
    reranker: Any
    multimodal: Any
    clock: StageClock
    alternates: dict[str, Any] = field(default_factory=dict)
    scope: str = "full"
    connections: Any = None
    """`--pooled`: an `imsg.retrieval.connections.RetrievalConnections`."""

    def pooled(self) -> dict[str, Any]:
        """The keyword that makes a service pooled — none at all without
        `--pooled`, so the script also runs against code from before it."""
        return {} if self.connections is None else {"connections": self.connections}

    def context(self) -> Any:
        """The `AccessContext` every sweep search runs under (`--scope`)."""
        from imsg.retrieval.access import LOCAL_FULL_ACCESS, AccessContext

        if self.scope == "allowlist":
            return AccessContext(surface="public", scope="allowlist", subject=None)
        return LOCAL_FULL_ACCESS

    def scorer_for(self, config: SweepConfig) -> tuple[Any, Any | None]:
        """`(scorer, provider)` for a configuration; the provider (for token
        accounting) is `None` when nothing is reranked."""
        from imsg.eval.backend import PassthroughReranker

        if config.batching == "none":
            return PassthroughReranker(), None
        provider = self.alternates[config.reranker] if config.reranker else self.reranker
        provider.doc_max_tokens = config.doc_cap
        if config.batching == "old":
            return FixedBatchReranker(provider), provider
        return provider, provider

    def service_for(self, config: SweepConfig, tally: RerankTally) -> Any:
        from imsg.retrieval.service import RetrievalService

        scorer, provider = self.scorer_for(config)
        return RetrievalService(
            pg_conn=self.pg,
            fts_conn=self.fts,
            config=with_retrieval(
                self.cfg, rerank_top=config.rerank_top, rerank_doc_max_tokens=config.doc_cap
            ),
            text_provider=TimedTextProvider(self.text, self.clock),
            reranker=TimedReranker(scorer, provider, self.clock, config.batching, tally),
            multimodal_provider=(
                TimedMultimodalProvider(self.multimodal, self.clock)
                if self.multimodal is not None and config.multimodal
                else None
            ),
            **self.pooled(),
        )

    def run(
        self, config: SweepConfig, queries: Sequence[str], passes: int, label: str
    ) -> tuple[list[QueryRecord], RerankTally]:
        context = self.context()
        tally = RerankTally()
        service = self.service_for(config, tally)
        records: list[QueryRecord] = []
        started = time.perf_counter()
        for _ in range(passes):
            for index, query in enumerate(queries):
                self.clock.reset()
                t0 = time.perf_counter()
                result = service.search_messages(context, query=query, limit=10)
                total = time.perf_counter() - t0
                records.append(
                    QueryRecord(
                        query_index=index,
                        stages=dict(self.clock.current),
                        total=total,
                        ranking=[str(r["segment_key"]) for r in result.results],
                    )
                )
        rerank = [r.stages.get("rerank", 0.0) for r in records]
        print(
            f"{label:22s} {len(records):3d} queries in {time.perf_counter() - started:7.1f} s  "
            f"rerank p50 {nearest_rank(rerank, 0.5):6.2f} s  "
            f"total p95 {nearest_rank([r.total for r in records], 0.95):6.2f} s  |  "
            f"{machine_load()}",
            flush=True,
        )
        slowest = {
            stage: nearest_rank([r.stages.get(stage, 0.0) for r in records], 0.95)
            for stage in (*NON_RERANK_STAGES, "rerank")
        }
        print(
            f"{'':22s} p95 ms: "
            + " ".join(f"{stage}={1000 * value:.0f}" for stage, value in slowest.items() if value)
            + f" other={1000 * nearest_rank([r.total - sum(r.stages.values()) for r in records], 0.95):.0f}",
            flush=True,
        )
        return records, tally

    def capture_pools(
        self, queries: Sequence[str], *, multimodal: bool, depth: int = 50
    ) -> list[CapturedPool]:
        """One pass of the real service per query with a pass-through
        reranker and a pool of `depth`, recording the pool it fetched —
        held in memory only."""
        from imsg.eval.backend import PassthroughReranker
        from imsg.retrieval import segments
        from imsg.retrieval.access import LOCAL_FULL_ACCESS
        from imsg.retrieval.query import analyze_query
        from imsg.retrieval.service import RetrievalService

        fetched: list[tuple[list[int], dict[int, Any]]] = []
        real_fetch = segments.fetch_segment_summaries

        def capturing(conn: Any, segment_ids: list[int]) -> dict[int, Any]:
            summaries = real_fetch(conn, segment_ids)
            fetched.append((list(segment_ids), summaries))
            return summaries

        service = RetrievalService(
            pg_conn=self.pg,
            fts_conn=self.fts,
            config=with_retrieval(self.cfg, rerank_top=depth),
            text_provider=TimedTextProvider(self.text, self.clock),
            reranker=PassthroughReranker(),
            multimodal_provider=(
                TimedMultimodalProvider(self.multimodal, self.clock)
                if self.multimodal is not None and multimodal
                else None
            ),
            **self.pooled(),
        )
        pools: list[CapturedPool] = []
        records: list[QueryRecord] = []
        setattr(segments, "fetch_segment_summaries", capturing)  # noqa: B010 — restored below
        try:
            for index, query in enumerate(queries):
                fetched.clear()
                self.clock.reset()
                started = time.perf_counter()
                service.search_messages(LOCAL_FULL_ACCESS, query=query, limit=10)
                records.append(
                    QueryRecord(index, dict(self.clock.current), time.perf_counter() - started, [])
                )
                ids, summaries = fetched[-1]
                kept = [i for i in ids if i in summaries]
                pools.append(
                    CapturedPool(
                        phrase=analyze_query(query).phrase,
                        segment_ids=kept,
                        texts={i: summaries[i].text for i in kept},
                        keys={i: str(summaries[i].segment_key) for i in kept},
                    )
                )
        finally:
            setattr(segments, "fetch_segment_summaries", real_fetch)  # noqa: B010
        label = "capture" if multimodal else "capture nomm"
        slowest = " ".join(
            f"{stage}={1000 * nearest_rank([r.stages.get(stage, 0.0) for r in records], 0.95):.0f}"
            for stage in NON_RERANK_STAGES
        )
        print(f"{label:22s} {len(records):3d} queries; non-rerank p95 ms: {slowest}", flush=True)
        return pools

    def run_rerank_only(
        self, config: SweepConfig, pools: Sequence[CapturedPool], label: str, limit: int = 10
    ) -> tuple[list[QueryRecord], RerankTally]:
        """Rerank captured pools exactly as `search_messages` would, timing
        the rerank stage alone."""
        tally = RerankTally()
        scorer, provider = self.scorer_for(config)
        timed_scorer = TimedReranker(scorer, provider, self.clock, config.batching, tally)
        records: list[QueryRecord] = []
        started = time.perf_counter()
        for index, pool in enumerate(pools):
            ids = pool.segment_ids[: max(config.rerank_top, limit)]
            self.clock.reset()
            t0 = time.perf_counter()
            scores = timed_scorer.score(pool.phrase, [pool.texts[i] for i in ids]) if ids else []
            elapsed = time.perf_counter() - t0
            ranked = sorted(zip(ids, scores, strict=True), key=lambda t: -t[1])[:limit]
            records.append(
                QueryRecord(
                    index, dict(self.clock.current), elapsed, [pool.keys[i] for i, _ in ranked]
                )
            )
        rerank = [r.stages.get("rerank", 0.0) for r in records]
        print(
            f"{label:22s} {len(records):3d} pools in {time.perf_counter() - started:7.1f} s  "
            f"rerank p50 {nearest_rank(rerank, 0.5):6.2f} s  p95 {nearest_rank(rerank, 0.95):6.2f} s  "
            f"|  {machine_load()}",
            flush=True,
        )
        return records, tally


@dataclass(slots=True)
class CapturedPool:
    """One query's fused pool as the service fetched it — in memory only."""

    phrase: str
    segment_ids: list[int]
    texts: dict[int, str]
    keys: dict[int, str]


# --------------------------------------------------------------------------
# reporting (numbers only)
# --------------------------------------------------------------------------


def stage_table(
    title: str, records: Sequence[QueryRecord], *, include_rerank: bool = True, quiet: bool = False
) -> dict[str, dict[str, float]]:
    if not quiet:
        print(f"\n{title}")
        print(f"  {'stage':18s} {'p50 ms':>9s} {'p95 ms':>9s} {'max ms':>9s}")
    out: dict[str, dict[str, float]] = {}
    stages = STAGES if include_rerank else NON_RERANK_STAGES
    rows = [*stages, "non_rerank_total"]
    if include_rerank:
        rows += ["other", "total"]
    for stage in rows:
        if stage == "total":
            values = [r.total for r in records]
        elif stage == "non_rerank_total":
            values = [sum(r.stages.get(s, 0.0) for s in NON_RERANK_STAGES) for r in records]
        elif stage == "other":
            values = [r.total - sum(r.stages.values()) for r in records]
        else:
            values = [r.stages.get(stage, 0.0) for r in records]
        stats = {
            "p50_ms": 1000 * nearest_rank(values, 0.5),
            "p95_ms": 1000 * nearest_rank(values, 0.95),
            "max_ms": 1000 * max(values),
        }
        out[stage] = stats
        if not quiet:
            print(
                f"  {stage:18s} {stats['p50_ms']:9.1f} {stats['p95_ms']:9.1f} "
                f"{stats['max_ms']:9.1f}"
            )
    return out


def sweep_table(
    results: dict[str, tuple[list[QueryRecord], RerankTally]],
) -> list[dict[str, Any]]:
    reference = results.get(REFERENCE_CONFIG)
    ref_rankings: dict[int, list[str]] = {}
    if reference is not None:
        for record in reference[0]:
            ref_rankings.setdefault(record.query_index, record.ranking)
    print(
        "\nsweep (seconds; tokens per query; ovl@10 and top3@10 are a PROXY against "
        f"{REFERENCE_CONFIG}, not a quality measurement)"
    )
    header = (
        f"  {'config':14s} {'rerank p50':>10s} {'rerank p95':>10s} {'total p50':>9s} "
        f"{'total p95':>9s} {'non-rr p95':>10s} {'tokens':>7s} {'padded':>7s} {'passes':>6s} "
        f"{'ovl@10':>6s} {'top3@10':>7s} {'top3@3':>6s}"
    )
    print(header)
    rows: list[dict[str, Any]] = []
    for name, (records, tally) in results.items():
        n = len(records)
        rerank = [r.stages.get("rerank", 0.0) for r in records]
        totals = [r.total for r in records]
        non_rerank = [sum(r.stages.get(s, 0.0) for s in NON_RERANK_STAGES) for r in records]
        overlaps = [
            overlap_at_10(r.ranking, ref_rankings[r.query_index])
            for r in records
            if r.query_index in ref_rankings
        ]
        top3 = [
            top3_in_top10(r.ranking, ref_rankings[r.query_index])
            for r in records
            if r.query_index in ref_rankings
        ]
        top3_3 = [
            top3_in_top3(r.ranking, ref_rankings[r.query_index])
            for r in records
            if r.query_index in ref_rankings
        ]
        row = {
            "config": name,
            "queries": n,
            "rerank_p50_s": nearest_rank(rerank, 0.5),
            "rerank_p95_s": nearest_rank(rerank, 0.95),
            "total_p50_s": nearest_rank(totals, 0.5),
            "total_p95_s": nearest_rank(totals, 0.95),
            "non_rerank_p95_s": nearest_rank(non_rerank, 0.95),
            "real_tokens_per_query": tally.real_tokens / n if n else math.nan,
            "padded_tokens_per_query": tally.padded_tokens / n if n else math.nan,
            "batches_per_query": tally.batches / n if n else math.nan,
            "overlap_at_10": statistics.fmean(overlaps) if overlaps else math.nan,
            "top3_in_top10": statistics.fmean(top3) if top3 else math.nan,
            "top3_in_top3": statistics.fmean(top3_3) if top3_3 else math.nan,
        }
        rows.append(row)
        print(
            f"  {name:14s} {row['rerank_p50_s']:10.2f} {row['rerank_p95_s']:10.2f} "
            f"{row['total_p50_s']:9.2f} {row['total_p95_s']:9.2f} {row['non_rerank_p95_s']:10.2f} "
            f"{row['real_tokens_per_query']:7.0f} {row['padded_tokens_per_query']:7.0f} "
            f"{row['batches_per_query']:6.1f} {row['overlap_at_10']:6.2f} "
            f"{row['top3_in_top10']:7.2f} {row['top3_in_top3']:6.2f}"
        )
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, help="config.yaml (default: $IMSG_CONFIG)")
    parser.add_argument(
        "--configs",
        default="all",
        help="comma-separated <new|new@label|old|none>:<top>:<cap|none>[:nomm], or 'all'",
    )
    parser.add_argument(
        "--reranker",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="an alternate reranker checkpoint for new@LABEL rows (a directory relative "
        "to paths.data_root, or absolute); measurement only",
    )
    parser.add_argument("--queries", type=int, default=len(QUERIES), help="use the first N queries")
    parser.add_argument("--passes", type=int, default=1, help="passes over the queries per config")
    parser.add_argument("--warmup", type=int, default=len(WARMUP_QUERIES))
    parser.add_argument("--no-first-touch", action="store_true", help="skip the first-touch pass")
    parser.add_argument(
        "--rerank-only",
        action="store_true",
        help="capture each query's pool once, then time only the rerank stage per configuration",
    )
    parser.add_argument(
        "--scope",
        choices=("full", "allowlist"),
        default="full",
        help="run the sweep under the local surface's full scope (default) or the public "
        "surface's allowlist scope",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="each call borrows its own connections and a search's candidate channels run "
        "side by side, as `imsg mcp public` does (see the module docstring)",
    )
    parser.add_argument(
        "--fake-model-delays",
        metavar="EMBED,MM,RERANK",
        help="with models.backend: fake, delay each fake model call by these milliseconds",
    )
    parser.add_argument("--report-json", type=Path, help="write the aggregate numbers here")
    args = parser.parse_args(argv)
    if args.rerank_only and args.scope != "full":
        parser.error("--rerank-only measures stored segment texts; it cannot run with --scope allowlist")

    from imsg.config.loader import load_config
    from imsg.retrieval import fts_search

    names = list(DEFAULT_CONFIGS) if args.configs == "all" else args.configs.split(",")
    configs = [parse_config(name) for name in names]
    queries = QUERIES[: max(1, min(args.queries, len(QUERIES)))]

    cfg = load_config(args.config)
    pg, fts = open_connections(cfg)
    text, reranker, multimodal = load_providers(cfg)
    if args.fake_model_delays:
        text, reranker, multimodal = delay_fake_providers(
            cfg, text, reranker, multimodal, args.fake_model_delays
        )
    connections = open_pooled_connections(cfg) if args.pooled else None
    alternates = load_alternate_rerankers(cfg, args.reranker)
    missing = {c.reranker for c in configs if c.reranker is not None} - set(alternates)
    if missing:
        raise SystemExit(f"no --reranker registered for {sorted(missing)}")
    clock = StageClock()
    instrument_service_modules(clock)
    if not hasattr(fts_search.search_segment_fts, "__wrapped__"):
        raise SystemExit("stage instrumentation did not reach the service's modules")
    bench = Bench(
        cfg, pg, fts, text, reranker, multimodal, clock, alternates, args.scope, connections
    )

    if args.rerank_only:
        return rerank_only_main(bench, configs, queries, args.warmup, args.report_json)

    warm = WARMUP_QUERIES[: args.warmup]
    if warm:
        bench.run(parse_config("new:50:none"), warm, 1, "warm-up new:50:none")
        bench.run(parse_config("new:10:256"), warm, 1, "warm-up new:10:256")
        for label in alternates:
            bench.run(parse_config(f"new@{label}:50:none"), warm[:1], 1, f"warm-up {label}")

    report: dict[str, Any] = {
        "queries": len(queries),
        "passes": args.passes,
        "scope": args.scope,
        "pooled": bool(args.pooled),
        "fake_model_delays_ms": args.fake_model_delays,
    }
    if not args.no_first_touch:
        pinned = [c for c in configs if c.batching == "new" and c.reranker is None] or [
            parse_config("new:10:256")
        ]
        cheapest = min(pinned, key=lambda c: (c.rerank_top, c.doc_cap or 1 << 30))
        first, _ = bench.run(
            SweepConfig("first-touch", "new", cheapest.rerank_top, cheapest.doc_cap),
            queries,
            1,
            "first-touch",
        )
        report["first_touch_stages"] = stage_table(
            "non-rerank stages, first touch (each query's first run in this process)",
            first,
            include_rerank=False,
        )

    results: dict[str, tuple[list[QueryRecord], RerankTally]] = {}
    for config in configs:
        results[config.name] = bench.run(config, queries, args.passes, config.name)

    if REFERENCE_CONFIG in results:
        report["reference_stages"] = stage_table(
            f"all stages, {REFERENCE_CONFIG} (after the first-touch pass)",
            results[REFERENCE_CONFIG][0],
        )
    for config in configs:
        report.setdefault("stages", {})[config.name] = stage_table(
            config.name, results[config.name][0], quiet=True
        )
    report["sweep"] = sweep_table(results)
    if args.report_json:
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report written to {args.report_json}")
    return 0


def rerank_only_main(
    bench: Bench,
    configs: Sequence[SweepConfig],
    queries: Sequence[str],
    warmup: int,
    report_json: Path | None,
) -> int:
    pools = {True: bench.capture_pools(queries, multimodal=True)}
    if any(not c.multimodal for c in configs):
        pools[False] = bench.capture_pools(queries, multimodal=False)
    if warmup:
        for config in {(c.batching, c.reranker): c for c in configs}.values():
            warm = SweepConfig(
                "warm-up",
                config.batching,
                10,
                256 if config.batching == "new" else None,
                config.reranker,
            )
            bench.run_rerank_only(warm, pools[True][:warmup], f"warm-up {config.name}")
    results: dict[str, tuple[list[QueryRecord], RerankTally]] = {}
    for config in configs:
        results[config.name] = bench.run_rerank_only(config, pools[config.multimodal], config.name)
    report: dict[str, Any] = {"queries": len(queries), "mode": "rerank-only"}
    report["sweep"] = sweep_table(results)
    if report_json:
        report_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report written to {report_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
