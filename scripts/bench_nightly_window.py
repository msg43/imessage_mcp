#!/usr/bin/env python3
"""Measure the nightly enrichment window: the query side serving a steady
trickle of searches while enrichment runs a batch, on SYNTHETIC inputs.

This is the harness for the case D10.3 measured and failed — 80.4 GiB of
demand on a 64 GB host, critical memory pressure, 37.6 GiB of swap, search
p95 at 8.73 s against a 2.0 s budget — and for re-measuring it after the
four fixes. Each fix has a flag, so the same harness produces both sides of
the comparison on one host in one sitting:

    --no-share-weights      load two independent copies of the 35B, the way
                            mlx-vlm and mlx-lm each used to
    --pe-core-both-towers   keep both PE-Core towers resident everywhere,
                            the way the provider used to
    --enrich-cache-limit-gib 0   leave the enrichment MLX buffer cache at the
                            runtime default (its "no bound" behaviour)
    --no-yield              do not let enrichment stand aside for queries

Three phases, in one run, so recovery is visible and not inferred:

    baseline   query side alone
    overlap    query side + one enrichment worker
    recovery   query side alone again, after the worker exits

and per phase it reports what D10 reported: each process's physical
footprint, MLX peak/active/cache, torch's MPS allocation, the system's
memory-pressure level, swap in and out, the query model stages' p50/p95,
and enrichment throughput.

Nothing here reads a message, an attachment, or a database row. Queries are
drawn from a fixed synthetic vocabulary; the "attachments" enrichment
captions are generated images. The only database use is the advisory locks
of `imsg.db.enrichment_yield_locks` (`--dsn`), which write nothing — a
read-only role can take them.

Usage (the project venv with the `models` extra):

    HF_HUB_OFFLINE=1 python scripts/bench_nightly_window.py \\
        --reranker "$DATA_ROOT/models/qwen3-reranker-0.6b-..." \\
        --dsn "postgresql://user@/db?host=/path&port=5433" \\
        --baseline-seconds 120 --overlap-seconds 420 --recovery-seconds 120 \\
        --json > after.json

    python scripts/bench_nightly_window.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import structlog

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bench_query_stages import (  # noqa: E402
    GIB,
    gpu_utilization_percent,
    host_info,
    local_dir,
    memory_state,
    mlx_memory,
    nearest_rank,
    process_footprint,
    system_memory,
    torch_mps_memory,
)
from imsg import constants  # noqa: E402

QUERY_VOCABULARY = (
    "ferry schedule", "bake sale receipt", "kite festival photo", "harbour parking",
    "rehearsal time", "borrowed enclosure", "lease renewal", "bench clearance",
    "long cable", "escalation clause", "scanned copy", "saturday swap",
)
"""Fixed synthetic phrases. A benchmark must never carry a real query."""

CAPTION_IMAGE_SIZE = (1440, 1920)
"""Roughly a phone photo, so the vision tower does realistic work."""

DEFAULT_QUERY_INSTRUCTION = (
    "Given a search query, retrieve the conversation passages that answer it"
)


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def percentiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": round(nearest_rank(ordered, 50), 3),
        "p95": round(nearest_rank(ordered, 95), 3),
        "max": round(ordered[-1], 3),
        "mean": round(statistics.fmean(ordered), 3),
    }


SWAP_COUNTER_KEYS = ("swapins", "swapouts", "pageins", "pageouts")
"""`memory_pressure`'s own counter names (checked against its output on
macOS 25, 2026-09-17), which is where D10's "swap in / swap out" and
"218 GiB paged in 11 minutes" come from. All are cumulative since boot,
so only deltas across a run mean anything."""


def swap_counters() -> dict[str, int]:
    state = system_memory() or {}
    counters = state.get("counters") or {}
    return {key: counters[key] for key in SWAP_COUNTER_KEYS if key in counters}


def write_event(stream: Any, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event) + "\n")
    stream.flush()


# --------------------------------------------------------------------------
# the query child: a steady trickle of searches
# --------------------------------------------------------------------------


def run_query_role(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from imsg.db.enrichment_yield_locks import QueryInFlightMarker
    from imsg.embed.mlx_text import MlxTextEmbeddingProvider
    from imsg.embed.pe_core_multimodal import PeCoreMultimodalEmbeddingProvider
    from imsg.retrieval.mlx_reranker import MlxRerankerProvider

    out = open(args.events, "w", encoding="utf-8")  # noqa: SIM115
    cache_limit = int(args.query_cache_limit_gib * GIB)
    embedder_dir = local_dir(args.embedder)
    reranker_dir = local_dir(args.reranker)

    embedder = MlxTextEmbeddingProvider(
        str(embedder_dir) if embedder_dir else args.embedder,
        None if embedder_dir else args.embedder_revision,
        constants.PRIMARY_EMBEDDING_DIM,
        model_id=f"{embedder_dir.name}@{args.embedder_revision}" if embedder_dir else None,
        cache_limit_bytes=cache_limit,
    )
    reranker = MlxRerankerProvider(
        str(reranker_dir) if reranker_dir else args.reranker,
        None if reranker_dir else args.reranker_revision,
        doc_max_tokens=args.rerank_doc_max_tokens,
        model_id=f"{reranker_dir.name}@{args.reranker_revision}" if reranker_dir else None,
        cache_limit_bytes=cache_limit,
    )
    pe_core = PeCoreMultimodalEmbeddingProvider(
        args.pe_core, args.pe_core_revision, constants.MULTIMODAL_EMBEDDING_DIM
    )

    started = time.monotonic()
    embedder.load()
    reranker.load()
    pe_core.embed_text("warm up")
    if args.pe_core_both_towers:
        # The pre-fix residency: both towers resident in the query process.
        _force_both_towers(pe_core)
    import torch

    write_event(
        out,
        {
            "kind": "loaded",
            "seconds": round(time.monotonic() - started, 2),
            "memory": memory_state(mx, torch),
            **mlx_memory(mx, include_peak=True),
        },
    )

    marker = QueryInFlightMarker(_connect_factory(args.dsn), enabled=bool(args.dsn) and args.yield_)
    rng = random.Random(args.seed)
    documents = [_synthetic_document(rng, i) for i in range(args.rerank_pool)]
    deadline = time.monotonic() + args.duration
    stop_file = Path(args.stop_file) if args.stop_file else None
    instruction = args.query_instruction

    try:
        # `--duration` is only a safety net: the parent decides when the
        # phases are over (its enrichment child's load time is not known in
        # advance) and says so by touching the stop file.
        while time.monotonic() < deadline and not (stop_file and stop_file.exists()):
            query = rng.choice(QUERY_VOCABULARY)
            with marker:
                t0 = time.perf_counter()
                embedder.embed_query(query, instruction=instruction)
                t1 = time.perf_counter()
                pe_core.embed_text(query)
                t2 = time.perf_counter()
                reranker.score(query, documents)
                t3 = time.perf_counter()
            write_event(
                out,
                {
                    "kind": "search",
                    "at": round(time.monotonic() - started, 2),
                    "embed_query_s": round(t1 - t0, 4),
                    "pe_core_text_s": round(t2 - t1, 4),
                    "rerank_s": round(t3 - t2, 4),
                    "model_stages_s": round(t3 - t0, 4),
                },
            )
            _sample(out, mx, torch, started)
            time.sleep(max(0.0, args.query_interval - (time.perf_counter() - t0)))
    finally:
        marker.close()
        write_event(
            out,
            {"kind": "final", "memory": memory_state(mx, torch), **mlx_memory(mx, include_peak=True)},
        )
        out.close()


def _force_both_towers(pe_core: Any) -> None:
    """Reproduce the pre-fix residency for an A/B run: ask for the tower the
    process does not use, which rebuilds with both."""
    pe_core._load("image")


def _synthetic_document(rng: random.Random, index: int) -> str:
    words = rng.choices(
        ["ferry", "harbour", "kite", "festival", "bake", "sale", "receipt", "rehearsal", "enclosure", "lease", "bench", "cable", "clause", "scan", "saturday", "swap", "parking", "schedule"],
        k=180,
    )
    return f"Chat: group {index % 5}\nTime: 2021-04-0{index % 9 + 1}\n" + " ".join(words)


# --------------------------------------------------------------------------
# the enrichment child: a realistic captioning batch
# --------------------------------------------------------------------------


def run_enrich_role(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from imsg.db.enrichment_yield_locks import EnrichmentYieldGate
    from imsg.embed.pe_core_multimodal import PeCoreMultimodalEmbeddingProvider
    from imsg.enrich.mlx_vlm_caption import MlxVlmCaptionProvider
    from imsg.segment.mlx_boundaries import MlxBoundaryProvider
    from imsg.shared_vlm_runtime import SharedVlmRuntime

    out = open(args.events, "w", encoding="utf-8")  # noqa: SIM115
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    images = [_write_synthetic_image(work / f"probe-{i}.png", i) for i in range(args.image_count)]

    cache_limit = int(args.enrich_cache_limit_gib * GIB) or None
    shared = SharedVlmRuntime(cache_limit_bytes=cache_limit) if args.share_weights else None
    caption = MlxVlmCaptionProvider(
        args.caption_model,
        args.caption_revision,
        (REPO_ROOT / "prompts" / "caption.txt").read_text(encoding="utf-8"),
        max_tokens=args.caption_max_tokens,
        shared_runtime=shared,
        cache_limit_bytes=cache_limit,
    )
    boundary = None
    if args.boundary:
        boundary = MlxBoundaryProvider(
            args.caption_model,
            args.caption_revision,
            (REPO_ROOT / "prompts" / "segment_boundaries.txt").read_text(encoding="utf-8"),
            shared_runtime=shared,
            cache_limit_bytes=cache_limit,
        )
    pe_core = (
        PeCoreMultimodalEmbeddingProvider(
            args.pe_core, args.pe_core_revision, constants.MULTIMODAL_EMBEDDING_DIM
        )
        if args.enrich_pe_core
        else None
    )

    started = time.monotonic()
    caption.caption(images[0])  # loads the captioner
    if boundary is not None:
        boundary.load()
    if pe_core is not None:
        pe_core.embed_images(images[:1])
        if args.pe_core_both_towers:
            pe_core.embed_text("warm up")
    import torch

    write_event(
        out,
        {
            "kind": "loaded",
            "seconds": round(time.monotonic() - started, 2),
            "shared_model_ids": list(shared.loaded_model_ids) if shared else [],
            "memory": memory_state(mx, torch),
            **mlx_memory(mx, include_peak=True),
        },
    )

    gate = EnrichmentYieldGate(
        _connect_factory(args.dsn)() if args.dsn and args.yield_ else _NullConn(),  # type: ignore[arg-type]
        enabled=bool(args.dsn) and args.yield_,
        poll_interval_seconds=args.yield_poll_seconds,
        max_pause_seconds=args.yield_max_pause_seconds,
    )

    deadline = time.monotonic() + args.duration
    index = 0
    try:
        while time.monotonic() < deadline:
            report = gate.wait_until_clear()
            image = images[index % len(images)]
            t0 = time.perf_counter()
            text = caption.caption(image)
            caption_s = time.perf_counter() - t0
            embed_s = None
            if pe_core is not None:
                t1 = time.perf_counter()
                pe_core.embed_images([image])
                embed_s = round(time.perf_counter() - t1, 4)
            boundary_s = None
            if boundary is not None and index % args.boundary_every == 0:
                t2 = time.perf_counter()
                boundary.detect_boundaries(_synthetic_window())
                boundary_s = round(time.perf_counter() - t2, 4)
            write_event(
                out,
                {
                    "kind": "task",
                    "at": round(time.monotonic() - started, 2),
                    "caption_s": round(caption_s, 4),
                    "embed_image_s": embed_s,
                    "boundary_s": boundary_s,
                    "caption_chars": len(text),
                    "yield_paused": report.paused,
                    "yield_waited_s": round(report.waited_seconds, 3),
                    "yield_gave_up": report.gave_up,
                },
            )
            _sample(out, mx, torch, started)
            index += 1
    finally:
        write_event(
            out,
            {"kind": "final", "memory": memory_state(mx, torch), **mlx_memory(mx, include_peak=True)},
        )
        out.close()


class _NullConn:
    """Stands in for a database connection when `--dsn` is not given: the
    gate is disabled, so it is never asked anything."""

    def cursor(self) -> Any:  # pragma: no cover - never reached when disabled
        raise AssertionError("the yield gate was used without a --dsn")


def _connect_factory(dsn: str | None) -> Any:
    def _connect() -> Any:
        import psycopg

        return psycopg.connect(dsn, autocommit=True)

    return _connect


def _write_synthetic_image(path: Path, index: int) -> Path:
    """A deterministic phone-sized RGB image. Generated, never a real
    attachment — a benchmark must not read the corpus."""
    from PIL import Image

    width, height = CAPTION_IMAGE_SIZE
    pixels = bytearray(width * height * 3)
    for y in range(height):
        base = y * width * 3
        green = (y * (5 + index)) % 256
        for x in range(width):
            offset = base + x * 3
            pixels[offset] = (x * (3 + index)) % 256
            pixels[offset + 1] = green
            pixels[offset + 2] = ((x + y) * (7 + index)) % 256
    Image.frombytes("RGB", (width, height), bytes(pixels)).save(path)
    return path


def _synthetic_window() -> list[Any]:
    from datetime import UTC, datetime, timedelta

    from imsg.segment.models import MessageForSegmentation

    base = datetime(2021, 4, 3, 10, 0, tzinfo=UTC)
    lines = [
        ("Ada", "are we still on for the hardware swap saturday"),
        ("Bo", "yes, i can bring the spare enclosure"),
        ("Ada", "great, i will clear the bench before you get here"),
        ("Bo", "do we need the long cable or the short one"),
        ("Ada", "long one, the rack is on the far wall"),
        ("Bo", "ok"),
        ("Ada", "different subject: the lease renewal form came back"),
        ("Bo", "what did they change on it"),
        ("Ada", "just the escalation clause, everything else is the same"),
        ("Bo", "i can sign it tomorrow morning"),
        ("Ada", "perfect, i will scan a copy for the file"),
        ("Bo", "thanks"),
    ]
    return [
        MessageForSegmentation(
            message_id=i + 1,
            source_guid=f"guid-{i + 1}",
            chat_id=1,
            sent_at=base + timedelta(minutes=i),
            is_from_me=(who == "Ada"),
            sender_short_name=who,
            text=text,
            is_unsent=False,
            is_edited=False,
            has_attachments=False,
        )
        for i, (who, text) in enumerate(lines)
    ]


_LAST_SAMPLE = [0.0]


def _sample(stream: Any, mx: Any, torch: Any, started: float) -> None:
    now = time.monotonic()
    if now - _LAST_SAMPLE[0] < 5.0:
        return
    _LAST_SAMPLE[0] = now
    write_event(
        stream,
        {
            "kind": "sample",
            "at": round(now - started, 2),
            **(process_footprint() or {}),
            **mlx_memory(mx, include_peak=True),
            **(torch_mps_memory(torch) or {}),
        },
    )


# --------------------------------------------------------------------------
# the parent: phases, system sampling, the report
# --------------------------------------------------------------------------


def _child_argv(role: str, args: argparse.Namespace, events: Path, duration: float) -> list[str]:
    argv = [
        sys.executable, str(Path(__file__).resolve()),
        "--role", role,
        "--events", str(events),
        "--duration", str(duration),
        "--embedder", args.embedder,
        "--embedder-revision", args.embedder_revision,
        "--reranker", args.reranker or "",
        "--reranker-revision", args.reranker_revision,
        "--pe-core", args.pe_core,
        "--pe-core-revision", args.pe_core_revision,
        "--caption-model", args.caption_model,
        "--caption-revision", args.caption_revision,
        "--query-interval", str(args.query_interval),
        "--query-cache-limit-gib", str(args.query_cache_limit_gib),
        "--enrich-cache-limit-gib", str(args.enrich_cache_limit_gib),
        "--caption-max-tokens", str(args.caption_max_tokens),
        "--image-count", str(args.image_count),
        "--rerank-pool", str(args.rerank_pool),
        "--boundary-every", str(args.boundary_every),
        "--work-dir", str(args.work_dir),
        "--stop-file", str(Path(args.work_dir) / "stop"),
        "--seed", str(args.seed),
        "--yield-poll-seconds", str(args.yield_poll_seconds),
        "--yield-max-pause-seconds", str(args.yield_max_pause_seconds),
    ]
    if args.dsn:
        argv += ["--dsn", args.dsn]
    if args.rerank_doc_max_tokens is not None:
        argv += ["--rerank-doc-max-tokens", str(args.rerank_doc_max_tokens)]
    if not args.share_weights:
        argv.append("--no-share-weights")
    if args.pe_core_both_towers:
        argv.append("--pe-core-both-towers")
    if not args.yield_:
        argv.append("--no-yield")
    if not args.boundary:
        argv.append("--no-boundary")
    if not args.enrich_pe_core:
        argv.append("--no-enrich-pe-core")
    return argv


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    query_events = work / "query-events.jsonl"
    enrich_events = work / "enrich-events.jsonl"

    # A safety net only — the stop file is what actually ends the run —
    # so it allows for the enrichment child's load as well as the phases.
    total_query_seconds = (
        args.warmup_seconds
        + args.baseline_seconds
        + args.overlap_seconds
        + args.recovery_seconds
        + args.load_timeout_seconds
    )
    report: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": host_info(mx),
        "settings": {
            "share_weights": args.share_weights,
            "pe_core_both_towers": args.pe_core_both_towers,
            "enrich_cache_limit_gib": args.enrich_cache_limit_gib,
            "query_cache_limit_gib": args.query_cache_limit_gib,
            "yield_to_queries": args.yield_ and bool(args.dsn),
            "boundary_in_enrichment": args.boundary,
            "pe_core_in_enrichment": args.enrich_pe_core,
            "query_interval_s": args.query_interval,
            "phase_seconds": {
                "warmup": args.warmup_seconds,
                "baseline": args.baseline_seconds,
                "overlap": args.overlap_seconds,
                "recovery": args.recovery_seconds,
            },
        },
        "system_samples": [],
    }

    counters_start = swap_counters()
    log(f"starting the query side for {total_query_seconds:.0f}s of searching after it loads")
    query_events.unlink(missing_ok=True)
    enrich_events.unlink(missing_ok=True)
    stop_file = work / "stop"
    stop_file.unlink(missing_ok=True)
    query_proc = subprocess.Popen(_child_argv("query", args, query_events, total_query_seconds))
    loaded = _wait_for_loaded(query_events, query_proc, timeout=args.load_timeout_seconds)
    log(f"query side loaded in {loaded['seconds']}s; phases start now")
    phase_marks: dict[str, float] = {}
    wall_start = time.monotonic()
    # The child timestamps its events from its own start, which was
    # `loaded["seconds"]` before this moment.
    query_offset = -float(loaded["seconds"])

    def _sample_system(phase: str) -> None:
        report["system_samples"].append(
            {
                "at": round(time.monotonic() - wall_start, 1),
                "phase": phase,
                "gpu_percent": gpu_utilization_percent(),
                **(system_memory() or {}),
            }
        )

    def _hold(phase: str, seconds: float) -> None:
        end = time.monotonic() + seconds
        _hold_while(phase, lambda: time.monotonic() < end, limit=seconds)

    def _hold_while(phase: str, keep_going: Any, *, limit: float) -> None:
        phase_marks[f"{phase}_start"] = time.monotonic() - wall_start
        hard_stop = time.monotonic() + limit
        while keep_going() and time.monotonic() < hard_stop:
            _sample_system(phase)
            time.sleep(args.sample_interval)
        phase_marks[f"{phase}_end"] = time.monotonic() - wall_start

    _hold("warmup", args.warmup_seconds)
    _hold("baseline", args.baseline_seconds)

    log(f"starting enrichment for {args.overlap_seconds:.0f}s of work after it loads")
    enrich_proc = subprocess.Popen(_child_argv("enrich", args, enrich_events, args.overlap_seconds))
    # The overlap phase starts when the process starts, not when it
    # finishes loading: the load is when residency spikes, and a
    # measurement that excluded it would miss what D10 measured. It ends
    # when that process exits, which is `--overlap-seconds` of work AFTER
    # a load whose length is not known in advance.
    _hold_while("overlap", lambda: enrich_proc.poll() is None, limit=args.overlap_seconds + 1800)
    log("enrichment exited; measuring recovery")
    _hold("recovery", args.recovery_seconds)
    stop_file.touch()
    query_proc.wait(timeout=300)

    counters_end = swap_counters()
    page_size = (system_memory() or {}).get("page_size") or 16384
    report["phase_marks"] = {k: round(v, 1) for k, v in phase_marks.items()}
    report["swap"] = {
        f"{key}_delta": counters_end[key] - counters_start[key]
        for key in SWAP_COUNTER_KEYS
        if key in counters_start and key in counters_end
    }
    report["swap"]["page_size"] = page_size
    report["swap"]["gib"] = {
        key.removesuffix("_delta"): round(value * page_size / GIB, 2)
        for key, value in report["swap"].items()
        if key.endswith("_delta")
    }
    report["query"] = _summarize_query(_read_events(query_events), phase_marks, query_offset)
    report["enrichment"] = _summarize_enrichment(_read_events(enrich_events))
    report["pressure_by_phase"] = _pressure_by_phase(report["system_samples"])
    return report


def _wait_for_loaded(path: Path, proc: subprocess.Popen[bytes], *, timeout: float) -> dict[str, Any]:
    """Block until the child reports its models are loaded.

    Without this the phase clock would start while the child was still
    loading 20-40 GiB of weights, and the baseline phase — the number
    every comparison is against — would contain no searches at all.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the child exited with {proc.returncode} before loading")
        for event in _read_events(path):
            if event["kind"] == "loaded":
                return event
        time.sleep(0.5)
    raise RuntimeError(f"the child did not load within {timeout}s")


def _phase_of(at: float, marks: dict[str, float], offset: float) -> str:
    absolute = at + offset
    for phase in ("recovery", "overlap", "baseline", "warmup"):
        start = marks.get(f"{phase}_start")
        end = marks.get(f"{phase}_end")
        if start is not None and end is not None and start <= absolute < end:
            return phase
    return "unknown"


def _summarize_query(
    events: list[dict[str, Any]], marks: dict[str, float], offset: float
) -> dict[str, Any]:
    loaded = next((e for e in events if e["kind"] == "loaded"), None)
    final = next((e for e in events if e["kind"] == "final"), None)
    searches = [e for e in events if e["kind"] == "search"]
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for event in searches:
        by_phase.setdefault(_phase_of(event["at"], marks, offset), []).append(event)
    stages = ("embed_query_s", "pe_core_text_s", "rerank_s", "model_stages_s")
    return {
        "load": loaded,
        "final": final,
        "peak_footprint_gib": max(
            (e.get("phys_footprint_gib", 0.0) for e in events if e["kind"] == "sample"),
            default=None,
        ),
        "by_phase": {
            phase: {stage: percentiles([e[stage] for e in rows]) for stage in stages}
            for phase, rows in sorted(by_phase.items())
            if phase != "unknown"
        },
    }


def _summarize_enrichment(events: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = [e for e in events if e["kind"] == "task"]
    loaded = next((e for e in events if e["kind"] == "loaded"), None)
    final = next((e for e in events if e["kind"] == "final"), None)
    if not tasks:
        return {"load": loaded, "final": final, "tasks": 0}
    span = tasks[-1]["at"] - tasks[0]["at"] if len(tasks) > 1 else tasks[0]["caption_s"]
    paused = [t for t in tasks if t["yield_paused"]]
    return {
        "load": loaded,
        "final": final,
        "tasks": len(tasks),
        "caption_seconds": percentiles([t["caption_s"] for t in tasks]),
        "embed_image_seconds": percentiles(
            [t["embed_image_s"] for t in tasks if t["embed_image_s"] is not None]
        ),
        "boundary_seconds": percentiles(
            [t["boundary_s"] for t in tasks if t["boundary_s"] is not None]
        ),
        "captions_per_minute": round(len(tasks) / span * 60, 2) if span > 0 else None,
        "peak_footprint_gib": max(
            (e.get("phys_footprint_gib", 0.0) for e in events if e["kind"] == "sample"),
            default=None,
        ),
        "yield": {
            "pauses": len(paused),
            "seconds_total": round(sum(t["yield_waited_s"] for t in tasks), 1),
            "gave_up": sum(1 for t in tasks if t["yield_gave_up"]),
        },
    }


def _pressure_by_phase(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sample in samples:
        phase = sample["phase"]
        bucket = out.setdefault(phase, {"levels": {}, "swap_used_gib_max": 0.0, "free_percent_min": 100})
        level = sample.get("pressure_level")
        if level is not None:
            bucket["levels"][str(level)] = bucket["levels"].get(str(level), 0) + 1
        swap = sample.get("swap_used_gib")
        if swap is not None:
            bucket["swap_used_gib_max"] = max(bucket["swap_used_gib_max"], swap)
        free = sample.get("free_percent")
        if free is not None:
            bucket["free_percent_min"] = min(bucket["free_percent_min"], free)
    return out


def print_report(report: dict[str, Any], out: Any) -> None:
    settings = report["settings"]
    print("settings:", json.dumps(settings), file=out)
    print(file=out)
    query = report["query"]
    print("query side, model stages (seconds)", file=out)
    print(f"{'phase':<12}{'n':>5}{'p50':>9}{'p95':>9}{'max':>9}", file=out)
    for phase, stages in query["by_phase"].items():
        totals = stages["model_stages_s"]
        print(
            f"{phase:<12}{totals.get('count', 0):>5}{totals.get('p50', 0):>9.3f}"
            f"{totals.get('p95', 0):>9.3f}{totals.get('max', 0):>9.3f}",
            file=out,
        )
    print(file=out)
    for label, side in (("query", query), ("enrichment", report["enrichment"])):
        load = side.get("load") or {}
        final = side.get("final") or {}
        memory = final.get("memory") or {}
        print(
            f"{label:<12} load {load.get('seconds', '—')}s  "
            f"footprint {memory.get('phys_footprint_gib', '—')} GiB  "
            f"peak-sampled {side.get('peak_footprint_gib', '—')} GiB  "
            f"mx active {final.get('mx_active_gib', '—')} / peak "
            f"{final.get('mx_peak_gib', '—')} / cache {final.get('mx_cache_gib', '—')} GiB  "
            f"torch {memory.get('torch_mps_allocated_gib', '—')} GiB",
            file=out,
        )
    enrichment = report["enrichment"]
    print(file=out)
    print(
        f"enrichment: tasks={enrichment.get('tasks')} "
        f"captions/min={enrichment.get('captions_per_minute')} "
        f"caption p50={enrichment.get('caption_seconds', {}).get('p50')}s "
        f"yield pauses={enrichment.get('yield', {}).get('pauses')} "
        f"({enrichment.get('yield', {}).get('seconds_total')}s)",
        file=out,
    )
    print(f"swap: {json.dumps(report['swap'])}", file=out)
    print(f"pressure: {json.dumps(report['pressure_by_phase'])}", file=out)


def compare_reports(before: Path, after: Path, out: Any) -> None:
    a, b = json.loads(before.read_text()), json.loads(after.read_text())
    print(f"{'':<28}{'before':>14}{'after':>14}", file=out)

    def row(label: str, x: Any, y: Any) -> None:
        print(f"{label:<28}{x!s:>14}{y!s:>14}", file=out)

    for phase in ("baseline", "overlap", "recovery"):
        for stat in ("p50", "p95"):
            row(
                f"query {phase} {stat} (s)",
                a["query"]["by_phase"].get(phase, {}).get("model_stages_s", {}).get(stat),
                b["query"]["by_phase"].get(phase, {}).get("model_stages_s", {}).get(stat),
            )
    row("query footprint (GiB)", a["query"]["peak_footprint_gib"], b["query"]["peak_footprint_gib"])
    row(
        "enrich footprint (GiB)",
        a["enrichment"].get("peak_footprint_gib"),
        b["enrichment"].get("peak_footprint_gib"),
    )
    row("captions/min", a["enrichment"].get("captions_per_minute"), b["enrichment"].get("captions_per_minute"))
    row("pages swapped out", a["swap"]["pages_swapped_out_delta"], b["swap"]["pages_swapped_out_delta"])
    row("pages swapped in", a["swap"]["pages_swapped_in_delta"], b["swap"]["pages_swapped_in_delta"])


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--role", choices=("query", "enrich"), help=argparse.SUPPRESS)
    p.add_argument("--events", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--stop-file", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--duration", type=float, default=60.0, help=argparse.SUPPRESS)
    p.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))

    p.add_argument("--embedder", default=constants.TEXT_EMBEDDING_MODEL_REPO)
    p.add_argument("--embedder-revision", default=constants.TEXT_EMBEDDING_MODEL_REVISION)
    p.add_argument("--reranker", default="")
    p.add_argument("--reranker-revision", default=constants.RERANKER_MODEL_REVISION)
    p.add_argument("--rerank-doc-max-tokens", type=int, default=256)
    p.add_argument("--rerank-pool", type=int, default=20)
    p.add_argument("--pe-core", default=constants.MULTIMODAL_EMBEDDING_MODEL_REPO)
    p.add_argument("--pe-core-revision", default=constants.MULTIMODAL_EMBEDDING_MODEL_REVISION)
    p.add_argument("--caption-model", default=constants.CAPTION_MODEL_REPO)
    p.add_argument("--caption-revision", default=constants.CAPTION_MODEL_REVISION)
    p.add_argument("--caption-max-tokens", type=int, default=96)
    p.add_argument("--image-count", type=int, default=8)
    p.add_argument("--boundary-every", type=int, default=4)
    p.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    p.add_argument("--query-interval", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20260917)

    p.add_argument("--warmup-seconds", type=float, default=30.0)
    p.add_argument("--baseline-seconds", type=float, default=120.0)
    p.add_argument("--overlap-seconds", type=float, default=420.0)
    p.add_argument("--recovery-seconds", type=float, default=120.0)
    p.add_argument("--sample-interval", type=float, default=5.0)
    p.add_argument(
        "--load-timeout-seconds",
        type=float,
        default=900.0,
        help="how long to wait for the query side's weights before giving up",
    )
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/imsg-nightly-window"))
    p.add_argument("--dsn", default=None, help="Postgres DSN for the yield advisory locks")
    p.add_argument(
        "--yield-poll-seconds",
        type=float,
        default=constants.ENRICHMENT_YIELD_POLL_INTERVAL_SECONDS,
        help="how often a paused enrichment worker re-checks (config "
        "enrichment.yield_poll_interval_seconds)",
    )
    p.add_argument(
        "--yield-max-pause-seconds",
        type=float,
        default=constants.ENRICHMENT_YIELD_MAX_PAUSE_SECONDS,
        help="how long it waits for one unit of work before proceeding anyway "
        "(config enrichment.yield_max_pause_seconds)",
    )

    p.add_argument("--query-cache-limit-gib", type=float, default=8.0)
    p.add_argument(
        "--enrich-cache-limit-gib",
        type=float,
        default=4.0,
        help="0 leaves MLX's buffer cache at the runtime default (the pre-fix behaviour)",
    )
    p.add_argument("--share-weights", action="store_true", default=True)
    p.add_argument("--no-share-weights", dest="share_weights", action="store_false")
    p.add_argument("--pe-core-both-towers", action="store_true", default=False)
    p.add_argument("--yield", dest="yield_", action="store_true", default=True)
    p.add_argument("--no-yield", dest="yield_", action="store_false")
    p.add_argument("--boundary", action="store_true", default=True)
    p.add_argument("--no-boundary", dest="boundary", action="store_false")
    p.add_argument("--enrich-pe-core", action="store_true", default=True)
    p.add_argument("--no-enrich-pe-core", dest="enrich_pe_core", action="store_false")
    p.add_argument("--json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # The providers log through structlog, whose default printer writes to
    # stdout — which is where `--json` puts the report. Send it to stderr
    # in every role, parent and child alike, or the report is unparseable.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    args = parse_args(argv)
    if args.compare:
        compare_reports(args.compare[0], args.compare[1], sys.stdout)
        return 0
    if args.role == "query":
        run_query_role(args)
        return 0
    if args.role == "enrich":
        run_enrich_role(args)
        return 0
    if not args.reranker:
        print("--reranker is required (a local conversion directory or a Hub repo id)", file=sys.stderr)
        return 2
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    report = run_parent(args)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
