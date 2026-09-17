#!/usr/bin/env python3
"""Time the query-time model stages on SYNTHETIC text, one host per run.

A search request runs three models (SPEC §9.4): the Qwen3-Embedding query
vector (`MlxTextEmbeddingProvider.embed_query`), PE-Core's text vector for
the multimodal channel (`PeCoreMultimodalEmbeddingProvider.embed_text`) and
the Qwen3-Reranker pass over the fused pool (`MlxRerankerProvider.score`).
This script builds the three providers directly by class — no config file,
no database — loads all three into one process, the way the retrieval
service holds them, and measures after warm-up:

- `embedder.embed_query`: latency of short synthetic queries (the query
  instruction template included, as at query time);
- `pe_core.embed_text`: latency of the same queries through PE-Core's text
  tower (a fixed-length token context);
- `reranker.forward` / `embedder.forward` at fixed shapes `BxT` (B rows of
  exactly T tokens, no padding): one forward pass through the same
  `imsg.mlx_runtime` calls each provider makes after tokenizing (base
  transformer, last-token gather, the reranker's yes/no head, values read
  back to Python), reported in seconds and tokens per second;
- `reranker.score` (`--pool-size`, 0 to skip): the provider's own `score()`
  over a pool of synthetic documents whose token lengths are drawn from
  `DOC_TOKEN_LENGTHS` — tokenization and the provider's own batching
  included;
- memory with all three resident: `mx.get_peak_memory()` / active / cache,
  torch MPS allocations, the process's max RSS and its physical footprint
  (`proc_pid_rusage`; GPU buffers count toward the footprint but mostly not
  toward RSS), plus `memory_pressure`, swap use and the kernel's pressure
  level before loading and after the stages;
- `--idle-gaps`: one call of each short stage after the process has sat idle
  for each gap, to show whether an idle process pays a cold-call penalty.

Every stage records its first call (made before warm-up) separately, every
timed repetition, nearest-rank p50/p95, and the GPU time this process and all
other processes used while it ran (per-process `accumulatedGPUTime` from
`ioreg`), so a GPU shared with another workload shows up in the output
instead of hiding in the numbers. Before loading, `--idle-baseline-seconds`
of idle measure the other processes' background GPU use.

All text comes from a seeded vocabulary of fictional chat about festivals,
ferries and bake sales; nothing is read from or written to any message
store. Hosts are recorded as a chip/memory class, never a hostname; model
directories are recorded by name plus `artifact_sha256`
(`imsg.providers.manifest.artifact_digest`, `--no-digest` to skip), so two
reports can be checked to have run the same bytes.

Usage (the project venv with the `models` extra; `HF_HUB_OFFLINE=1` keeps the
providers from contacting the Hub once the pinned snapshots are cached):

    HF_HUB_OFFLINE=1 python scripts/bench_query_stages.py \\
        --reranker "$DATA_ROOT/models/qwen3-reranker-8b-mxfp8-77d193c7" \\
        --json > host-a.json
    python scripts/bench_query_stages.py --compare host-a.json host-b.json

With `--json` the report goes to stdout as JSON and the table to stderr;
otherwise the table goes to stdout. Progress is always on stderr.
`--compare A B` prints, per stage, both hosts' p50/p95 and the B/A ratios.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import gc
import importlib
import json
import math
import os
import platform
import random
import re
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TextIO

import structlog

from imsg import constants
from imsg.embed.mlx_text import MlxTextEmbeddingProvider, format_query_text
from imsg.embed.pe_core_multimodal import PeCoreMultimodalEmbeddingProvider
from imsg.mlx_runtime import (
    base_transformer_hidden_states,
    float_rows,
    gather_last_token_states,
    import_mlx_core,
    lm_head_logits,
    right_pad,
)
from imsg.providers.manifest import ARTIFACT_FILE_PATTERNS, artifact_digest
from imsg.retrieval.mlx_reranker import MlxRerankerProvider, format_reranker_pair

REPORT_SCHEMA = "bench_query_stages/1"
GIB = float(2**30)

DEFAULT_QUERY_INSTRUCTION = (
    "Given a personal message search query, retrieve relevant conversation segments"
)
"""`embedding.query_instruction` as `config.example.yaml` ships it."""

DEFAULT_SHAPES = "1x256,1x512,4x256,8x256,8x512"

DOC_TOKEN_LENGTHS: tuple[int, ...] = (40, 60, 90, 120, 160, 220, 300, 400, 520, 700)
"""Token lengths the rerank-pool documents are drawn from — a spread like
chat segments, from a few exchanged lines to a long back-and-forth."""

SHORT_STAGES = frozenset({"embedder.embed_query", "pe_core.embed_text"})
PACKAGES = (
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "torch",
    "open_clip_torch",
    "timm",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "numpy",
)

# --------------------------------------------------------------------------
# synthetic, fictional text
# --------------------------------------------------------------------------


def _choices(joined: str) -> tuple[str, ...]:
    return tuple(joined.split("|"))


SPEAKERS = _choices("A|B|C|D")
OPENERS = _choices(
    "hey|ok so|quick question|sounds good|haha yes|fyi|update|all set|on my way|no worries|"
    "wait|good news"
)
SUBJECTS = _choices(
    "the kite festival|the harbor walk|the lantern parade|the corner bakery|the ferry schedule|"
    "the orchard trip|the bike repair|the pottery class|the trailhead lot|the rooftop garden|"
    "board game night|the lemon cake|the canoe rental|the library book sale|choir rehearsal|"
    "the chili cook-off"
)
ACTIONS = _choices(
    "moved to|starts at|is booked for|got pushed to|opens at|is sold out until|"
    "needs a headcount by|wraps up around|is back on for"
)
WHENS = _choices(
    "nine|noon|half past six|saturday morning|sunday|friday night|next week|tomorrow|"
    "after lunch|the first weekend of the month"
)
TAILS = _choices(
    "can you bring the blue cooler|I'll save seats near the gate|text me when you park|"
    "we still need napkins|bring a jacket, it gets windy by the water|the map is on the fridge|"
    "I can drive if that helps|let's split the tickets|don't forget sunscreen|"
    "the spare key is under the flower pot|grab extra lemons on the way|"
    "the forecast says clear skies"
)
THINGS = _choices(
    "the cooler|sunscreen|the tickets|the spare key|napkins|the map|folding chairs|snacks|"
    "the kite string"
)
QUERY_TEMPLATES = _choices(
    "when does {subject} start|{subject} plans|who is bringing {thing}|"
    "what time did we pick for {subject}|{thing} for {subject}|photos from {subject}|"
    "did anyone book {subject}|where did we leave {thing}"
)


def encode(tokenizer: Any, text: str) -> list[int]:
    return [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]


class SyntheticChat:
    """Seeded generator of fictional chat lines, documents and queries."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def line(self) -> str:
        r = self.rng
        return (
            f"{r.choice(SPEAKERS)}: {r.choice(OPENERS)}, {r.choice(SUBJECTS)} "
            f"{r.choice(ACTIONS)} {r.choice(WHENS)} - {r.choice(TAILS)}"
        )

    def query(self) -> str:
        r = self.rng
        return r.choice(QUERY_TEMPLATES).format(subject=r.choice(SUBJECTS), thing=r.choice(THINGS))

    def document(self, tokenizer: Any, tokens: int) -> str:
        """Chat lines cut to exactly `tokens` token ids, decoded back to text
        (re-tokenizing it can differ by a token or two)."""
        lines: list[str] = []
        ids: list[int] = []
        while len(ids) < tokens:
            lines.append(self.line())
            ids = encode(tokenizer, "\n".join(lines))
        return str(tokenizer.decode(ids[:tokens]))


# --------------------------------------------------------------------------
# fixed-shape forward passes
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Shape:
    batch: int
    tokens: int

    @property
    def label(self) -> str:
        return f"{self.batch}x{self.tokens}"


def parse_shapes(spec: str) -> list[Shape]:
    shapes: list[Shape] = []
    for part in (p.strip() for p in spec.split(",")):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)x(\d+)", part)
        if m is None or int(m.group(1)) < 1 or int(m.group(2)) < 1:
            raise SystemExit(f"bad shape {part!r}: expected BATCHxTOKENS, e.g. 8x512")
        shapes.append(Shape(int(m.group(1)), int(m.group(2))))
    return shapes


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """What a provider's post-tokenization forward pass uses. The providers
    expose no public handle on their loaded weights, so these are read from
    the same attributes `_embed_rows` / `_score_batch` use."""

    model: Any
    tokenizer: Any
    pad_id: int
    row_prefix: tuple[int, ...]
    row_suffix: tuple[int, ...]
    yes_no_ids: tuple[int, int] | None

    @property
    def fixed_tokens(self) -> int:
        """Tokens every row carries besides its text (chat template, EOS)."""
        return len(self.row_prefix) + len(self.row_suffix)

    def fits(self, shape: Shape) -> bool:
        return shape.tokens > self.fixed_tokens


def embedder_internals(provider: MlxTextEmbeddingProvider) -> LoadedModel:
    provider.load()
    return LoadedModel(
        model=provider._model,
        tokenizer=provider._tokenizer,
        pad_id=provider._pad_id,
        row_prefix=(),
        row_suffix=tuple(provider._eos_suffix),
        yes_no_ids=None,
    )


def reranker_internals(provider: MlxRerankerProvider) -> LoadedModel:
    provider.load()
    return LoadedModel(
        model=provider._model,
        tokenizer=provider._tokenizer,
        pad_id=provider._pad_id,
        row_prefix=tuple(provider._prefix_ids),
        row_suffix=tuple(provider._suffix_ids),
        yes_no_ids=(provider._no_id, provider._yes_id),
    )


def build_rows(
    loaded: LoadedModel,
    shape: Shape,
    chat: SyntheticChat,
    *,
    reranker_instruction: str | None,
) -> list[list[int]]:
    """`shape.batch` rows of exactly `shape.tokens` ids laid out as the
    provider lays out a real row: the reranker's chat prefix + pair body +
    chat suffix, or the embedder's text + EOS suffix."""
    if not loaded.fits(shape):
        raise ValueError(
            f"shape {shape.label} leaves no room for text next to the model's "
            f"{loaded.fixed_tokens} fixed tokens"
        )
    body_len = shape.tokens - loaded.fixed_tokens
    rows: list[list[int]] = []
    for _ in range(shape.batch):
        document = chat.document(loaded.tokenizer, shape.tokens)
        if reranker_instruction is not None:
            text = format_reranker_pair(reranker_instruction, chat.query(), document)
        else:
            text = document
        body = encode(loaded.tokenizer, text)
        while len(body) < body_len:  # decode/re-encode can shrink by a token or two
            body += encode(loaded.tokenizer, "\n" + chat.line())
        row = [*loaded.row_prefix, *body[:body_len], *loaded.row_suffix]
        if len(row) != shape.tokens:
            raise RuntimeError(f"built a {len(row)}-token row for shape {shape.label}")
        rows.append(row)
    return rows


def forward(mx: Any, loaded: LoadedModel, rows: list[list[int]]) -> list[list[float]]:
    """The provider's forward pass after tokenization, read back to Python
    (which forces MLX's lazy graph to run, so the wall time is the work)."""
    padded, lengths = right_pad(rows, loaded.pad_id)
    hidden = base_transformer_hidden_states(loaded.model, mx.array(padded))
    last = gather_last_token_states(mx, hidden, lengths)
    if loaded.yes_no_ids is None:
        return float_rows(mx, last)
    logits = lm_head_logits(loaded.model, last)
    return float_rows(mx, mx.take(logits, mx.array(list(loaded.yes_no_ids)), axis=-1))


# --------------------------------------------------------------------------
# host, memory and GPU probes (macOS; each degrades to None elsewhere)
# --------------------------------------------------------------------------


def run_text(argv: Sequence[str], timeout: float = 30.0) -> str | None:
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def sysctl(name: str) -> str | None:
    out = run_text(["sysctl", "-n", name])
    return out.strip() if out else None


def gpu_utilization_percent() -> int | None:
    out = run_text(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"])
    m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out or "")
    return int(m.group(1)) if m else None


def gpu_time_by_process() -> dict[int, tuple[str, int]] | None:
    """Accumulated GPU time in nanoseconds per process (`pid -> (name,
    ns)`), summed over the process's GPU clients — the `accumulatedGPUTime`
    of each `AGXDeviceUserClient`'s `AppUsage` in `ioreg`. Readable without
    elevated privileges; unlike `Device Utilization %` it separates this
    process's GPU work from everyone else's."""
    out = run_text(["ioreg", "-r", "-c", "AGXDeviceUserClient", "-w0", "-d", "1"])
    if out is None:
        return None
    totals: dict[int, tuple[str, int]] = {}
    for block in re.split(r"\n(?=\+-o )", out):
        creator = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"', block)
        if creator is None:
            continue
        pid = int(creator.group(1))
        ns = sum(int(v) for v in re.findall(r'"accumulatedGPUTime"=(\d+)', block))
        name, previous = totals.get(pid, (creator.group(2), 0))
        totals[pid] = (name, previous + ns)
    return totals


def gpu_time_delta(
    before: dict[int, tuple[str, int]] | None, after: dict[int, tuple[str, int]] | None
) -> dict[str, Any] | None:
    """GPU seconds used between two `gpu_time_by_process` readings: by this
    process, by all others, and the three busiest others by process name.
    A process that exited in between is not counted."""
    if before is None or after is None:
        return None
    me = os.getpid()
    own = 0
    others: dict[str, int] = {}
    for pid, (name, ns) in after.items():
        delta = ns - before.get(pid, (name, 0))[1]
        if delta <= 0:
            continue
        if pid == me:
            own += delta
        else:
            others[name] = others.get(name, 0) + delta
    busiest = sorted(others.items(), key=lambda item: -item[1])[:3]
    return {
        "gpu_s_self": round(own / 1e9, 3),
        "gpu_s_other": round(sum(others.values()) / 1e9, 3),
        "gpu_s_other_busiest": {name: round(ns / 1e9, 3) for name, ns in busiest},
    }


RUSAGE_INFO_V4_U64_FIELDS = (
    "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size "
    "ri_resident_size ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime "
    "ri_child_user_time ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups "
    "ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread ri_diskio_byteswritten "
    "ri_cpu_time_qos_default ri_cpu_time_qos_maintenance ri_cpu_time_qos_background "
    "ri_cpu_time_qos_utility ri_cpu_time_qos_legacy ri_cpu_time_qos_user_initiated "
    "ri_cpu_time_qos_user_interactive ri_billed_system_time ri_serviced_system_time "
    "ri_logical_writes ri_lifetime_max_phys_footprint ri_instructions ri_cycles ri_billed_energy "
    "ri_serviced_energy ri_interval_max_phys_footprint ri_runnable_time"
)
"""The 35 `uint64_t` members of `struct rusage_info_v4` after its 16-byte
`ri_uuid`, in declaration order, as the macOS SDK's `<sys/resource.h>`
declares them. The kernel fills the whole struct, so every member must be
present even though only a few are read."""


class RusageInfoV4(ctypes.Structure):
    """`struct rusage_info_v4` (see `RUSAGE_INFO_V4_U64_FIELDS`)."""

    _fields_ = (
        ("ri_uuid", ctypes.c_uint8 * 16),
        *((name, ctypes.c_uint64) for name in RUSAGE_INFO_V4_U64_FIELDS.split()),
    )


RUSAGE_INFO_V4 = 4


def process_footprint() -> dict[str, float] | None:
    """This process's physical footprint (what Activity Monitor calls
    Memory; includes GPU buffers) now and at its lifetime maximum."""
    if sys.platform != "darwin":
        return None
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        info = RusageInfoV4()
        if libsystem.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
            return None
    except (OSError, AttributeError):
        return None
    return {
        "phys_footprint_gib": round(info.ri_phys_footprint / GIB, 2),
        "lifetime_max_phys_footprint_gib": round(info.ri_lifetime_max_phys_footprint / GIB, 2),
        "resident_size_gib": round(info.ri_resident_size / GIB, 2),
        "wired_size_gib": round(info.ri_wired_size / GIB, 2),
    }


def max_rss_gib() -> float:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((maxrss if sys.platform == "darwin" else maxrss * 1024) / GIB, 2)


PRESSURE_LEVELS = {1: "normal", 2: "warn", 4: "critical"}


def system_memory() -> dict[str, Any] | None:
    """`memory_pressure` (report only — no simulation flags), swap use, and
    the kernel's memory pressure level."""
    if sys.platform != "darwin":
        return None
    out = run_text(["memory_pressure"]) or ""
    counters: dict[str, int] = {}
    for line in out.splitlines():
        m = re.fullmatch(r"\s*([A-Za-z][A-Za-z ]*?):\s*(\d+)\s*", line)
        if m:
            counters[m.group(1).strip().lower().replace(" ", "_")] = int(m.group(2))
    free = re.search(r"System-wide memory free percentage:\s*(\d+)%", out)
    page = re.search(r"page size of (\d+)", out)
    swap = re.search(r"used = ([\d.]+)M", sysctl("vm.swapusage") or "")
    level_raw = sysctl("kern.memorystatus_vm_pressure_level")
    level = int(level_raw) if level_raw and level_raw.isdigit() else None
    return {
        "free_percent": int(free.group(1)) if free else None,
        "pressure_level": PRESSURE_LEVELS.get(level, level) if level is not None else None,
        "swap_used_gib": round(float(swap.group(1)) / 1024, 2) if swap else None,
        "page_size": int(page.group(1)) if page else None,
        "counters": counters,
    }


def mlx_memory(mx: Any, *, include_peak: bool) -> dict[str, float]:
    """MLX's active and cached buffers, and — when asked — its peak since
    the last `mx.reset_peak_memory()` (each stage resets it; see `measure`)."""
    out = {
        "mx_active_gib": round(mx.get_active_memory() / GIB, 2),
        "mx_cache_gib": round(mx.get_cache_memory() / GIB, 2),
    }
    if include_peak:
        out["mx_peak_gib"] = round(mx.get_peak_memory() / GIB, 2)
    return out


def torch_mps_memory(torch: Any) -> dict[str, float] | None:
    mps = getattr(torch, "mps", None)
    if mps is None or not torch.backends.mps.is_available():
        return None
    return {
        "torch_mps_allocated_gib": round(mps.current_allocated_memory() / GIB, 2),
        # driver_allocated_memory() is every Metal allocation the driver holds for this
        # process — MLX's buffers included, not only torch's.
        "metal_driver_allocated_gib": round(mps.driver_allocated_memory() / GIB, 2),
    }


def memory_state(mx: Any | None, torch: Any | None) -> dict[str, Any]:
    state: dict[str, Any] = {"max_rss_gib": max_rss_gib()}
    state.update(process_footprint() or {})
    if mx is not None:
        state.update(mlx_memory(mx, include_peak=False))
    if torch is not None:
        state.update(torch_mps_memory(torch) or {})
    state["system"] = system_memory()
    return state


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def host_info(mx: Any) -> dict[str, Any]:
    """Chip, memory and software — never a hostname."""
    chip = sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    memsize = sysctl("hw.memsize")
    memory_bytes = int(memsize) if memsize and memsize.isdigit() else 0
    cores = re.search(
        r'"gpu-core-count"\s*=\s*(\d+)',
        run_text(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"]) or "",
    )
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    device_info = getattr(mx, "device_info", None) or getattr(
        getattr(mx, "metal", None), "device_info", None
    )
    return {
        "host_class": f"{chip}, {round(memory_bytes / GIB)} GB unified memory",
        "chip": chip,
        "memory_bytes": memory_bytes,
        "gpu_cores": int(cores.group(1)) if cores else None,
        "macos": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
        "packages": versions,
        "mlx_device_info": json_safe(device_info()) if device_info is not None else None,
    }


# --------------------------------------------------------------------------
# model locations and provenance
# --------------------------------------------------------------------------


def local_dir(location: str) -> Path | None:
    path = Path(location).expanduser()
    return path if path.is_dir() else None


def model_provenance(
    location: str, revision: str | None, *, cache_dir: Path | None, digest: bool
) -> dict[str, Any]:
    """Name the weights without an absolute path, and digest them when asked."""
    directory = local_dir(location)
    if directory is not None:
        info: dict[str, Any] = {"model": directory.name, "kind": "local_dir"}
    else:
        info = {"model": f"{location}@{revision or 'main'}", "kind": "hub_snapshot"}
    if not digest:
        return info
    if directory is None:
        hub = importlib.import_module("huggingface_hub")
        # Restricted to the digest's own file set: offline, huggingface_hub
        # refuses a snapshot missing any repo file (e.g. a skipped .bin
        # duplicate of the safetensors weights) unless the call is narrowed.
        directory = Path(
            str(
                hub.snapshot_download(
                    location,
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    local_files_only=True,
                    allow_patterns=list(ARTIFACT_FILE_PATTERNS),
                )
            )
        )
    t0 = time.perf_counter()
    computed = artifact_digest(directory)
    info.update(
        artifact_sha256=computed.sha256,
        artifact_gib=round(computed.total_bytes / GIB, 2),
        digest_seconds=round(time.perf_counter() - t0, 1),
    )
    return info


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def time_call(call: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    call()
    return time.perf_counter() - t0


def nearest_rank(ordered: Sequence[float], percent: float) -> float:
    return ordered[max(1, math.ceil(percent / 100 * len(ordered))) - 1]


def measure(
    stage: str,
    shape: str,
    calls: Sequence[Callable[[], object]],
    *,
    warmup: int,
    mx: Any,
    tokens_per_call: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Time `calls[0]` on its own as the stage's first call (it pays any
    kernel compilation for new shapes), run the next `warmup` calls untimed,
    and time each remaining call as one repetition. MLX's peak-memory
    counter is reset as the stage starts, so the stage's `mx_peak_gib` is
    the resident weights plus that stage's own working memory. GPU time is
    read per process before and after the stage (`gpu_time_by_process`), so
    `gpu_s_other` is the GPU time other processes used while it ran."""
    if len(calls) < warmup + 2:
        raise ValueError("need a first call, the warm-up calls and at least one repetition")
    gc.collect()
    gpu_before = gpu_time_by_process()
    # After a reset MLX reports a peak of 0 until it next allocates, so a stage
    # that allocates nothing through MLX (PE-Core runs on torch) would read 0;
    # floor the stage's peak at what MLX already holds.
    active_at_start = mx.get_active_memory()
    mx.reset_peak_memory()
    started = time.perf_counter()
    first = time_call(calls[0])
    for call in calls[1 : 1 + warmup]:
        call()
    samples = [time_call(call) for call in calls[1 + warmup :]]
    wall = time.perf_counter() - started
    gpu = gpu_time_delta(gpu_before, gpu_time_by_process())
    ordered = sorted(samples)
    p50 = nearest_rank(ordered, 50)
    result: dict[str, Any] = {
        "stage": stage,
        "shape": shape,
        "warmup": warmup,
        "repetitions": len(samples),
        "wall_s": round(wall, 3),
        **(gpu or {}),
        "first_call_s": round(first, 5),
        "min_s": round(ordered[0], 5),
        "p50_s": round(p50, 5),
        "p95_s": round(nearest_rank(ordered, 95), 5),
        "max_s": round(ordered[-1], 5),
        "mean_s": round(sum(samples) / len(samples), 5),
        "mx_peak_gib": round(max(mx.get_peak_memory(), active_at_start) / GIB, 2),
        "seconds": [round(s, 5) for s in samples],
    }
    if tokens_per_call is not None:
        result["tokens_per_call"] = tokens_per_call
        result["tokens_per_s_p50"] = round(tokens_per_call / p50, 1)
    result.update(extra or {})
    rate = f"  {result['tokens_per_s_p50']:9.1f} tok/s" if tokens_per_call is not None else ""
    log(
        f"  {stage:22s} {shape:10s} first {first:8.4f}s  p50 {p50:8.4f}s  "
        f"p95 {result['p95_s']:8.4f}s  max {result['max_s']:8.4f}s{rate}  "
        f"other-GPU {result.get('gpu_s_other')}s of {wall:.1f}s"
    )
    return result


def fixed_batches_padded(lengths: Sequence[int], batch_size: int) -> int:
    """Padded tokens (`rows x longest row`, summed) when rows are batched
    `batch_size` at a time in input order — how `MlxRerankerProvider.score`
    groups pairs as of this script."""
    return sum(
        len(lengths[i : i + batch_size]) * max(lengths[i : i + batch_size])
        for i in range(0, len(lengths), batch_size)
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    mx = import_mlx_core()
    started = datetime.now(UTC)
    shapes = parse_shapes(args.shapes)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "started_at": started.isoformat(timespec="seconds"),
        "host": host_info(mx),
        "settings": {
            "seed": args.seed,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "shapes": [s.label for s in shapes],
            "pool_size": args.pool_size,
            "pool_repetitions": args.pool_repetitions,
            "reranker_batch_size": args.reranker_batch_size,
            "idle_gaps_s": args.idle_gaps,
            "idle_baseline_seconds": args.idle_baseline_seconds,
            "query_instruction": args.query_instruction,
            "doc_token_lengths": list(DOC_TOKEN_LENGTHS),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        },
    }
    log(f"host: {report['host']['host_class']}, {report['host']['gpu_cores']}-core GPU")

    embedder_dir = local_dir(args.embedder)
    reranker_dir = local_dir(args.reranker)
    log("digesting model directories" if args.digest else "skipping digests (--no-digest)")
    report["models"] = {
        "embedder": model_provenance(
            args.embedder, args.embedder_revision, cache_dir=None, digest=args.digest
        ),
        "reranker": model_provenance(
            args.reranker, args.reranker_revision, cache_dir=None, digest=args.digest
        ),
        "pe_core": model_provenance(
            args.pe_core,
            args.pe_core_revision,
            cache_dir=args.pe_core_cache_dir,
            digest=args.digest,
        ),
    }

    log(f"idle GPU baseline: {args.idle_baseline_seconds:g}s")
    baseline_start = gpu_time_by_process()
    utilization: list[int | None] = []
    for _ in range(max(1, math.ceil(args.idle_baseline_seconds))):
        time.sleep(1.0)
        utilization.append(gpu_utilization_percent())
    report["gpu_idle_baseline"] = {
        "seconds": len(utilization),
        "utilization_pct_samples": utilization,
        **(gpu_time_delta(baseline_start, gpu_time_by_process()) or {}),
    }
    log(f"  {report['gpu_idle_baseline']}")
    memory: dict[str, Any] = {"before_load": memory_state(None, None)}
    report["memory"] = memory

    embedder = MlxTextEmbeddingProvider(
        str(embedder_dir) if embedder_dir else args.embedder,
        None if embedder_dir else args.embedder_revision,
        constants.PRIMARY_EMBEDDING_DIM,
        model_id=f"{embedder_dir.name}@{args.embedder_revision}" if embedder_dir else None,
    )
    reranker = MlxRerankerProvider(
        str(reranker_dir) if reranker_dir else args.reranker,
        None if reranker_dir else args.reranker_revision,
        batch_size=args.reranker_batch_size,
        model_id=f"{reranker_dir.name}@{args.reranker_revision}" if reranker_dir else None,
    )
    pe_core = PeCoreMultimodalEmbeddingProvider(
        args.pe_core,
        args.pe_core_revision,
        constants.MULTIMODAL_EMBEDDING_DIM,
        cache_dir=args.pe_core_cache_dir,
    )

    loads: dict[str, Any] = {}
    for name, load in (
        ("embedder", embedder.load),
        ("reranker", reranker.load),
        # PE-Core's loader runs one text-tower forward to prove the width.
        ("pe_core", pe_core._load),
    ):
        seconds = time_call(load)
        loads[name] = {"seconds": round(seconds, 2), **mlx_memory(mx, include_peak=True)}
        log(f"loaded {name} in {seconds:.1f}s ({loads[name]})")
    torch = importlib.import_module("torch")
    loads["pe_core"].update(torch_mps_memory(torch) or {})
    report["load"] = loads
    memory["all_loaded"] = memory_state(mx, torch)

    chat = SyntheticChat(args.seed)
    instruction = args.query_instruction
    calls_per_stage = 1 + args.warmup + args.repetitions
    queries = [chat.query() for _ in range(calls_per_stage)]
    query_tokens = [embedder.token_length(format_query_text(instruction, q)) for q in queries]
    stages: list[dict[str, Any]] = []
    report["stages"] = stages

    log("stages:")
    stages.append(
        measure(
            "embedder.embed_query",
            f"1x{min(query_tokens)}-{max(query_tokens)}",
            [functools.partial(embedder.embed_query, q, instruction=instruction) for q in queries],
            warmup=args.warmup,
            mx=mx,
            extra={"query_tokens": query_tokens[1 + args.warmup :]},
        )
    )
    pe_runtime = pe_core._load()
    pe_context = getattr(pe_runtime.tokenizer, "context_length", None)
    stages.append(
        measure(
            "pe_core.embed_text",
            f"1x{pe_context}",
            [functools.partial(pe_core.embed_text, q) for q in queries],
            warmup=args.warmup,
            mx=mx,
        )
    )

    rerank_loaded = reranker_internals(reranker)
    embed_loaded = embedder_internals(embedder)
    skipped: list[str] = []
    report["skipped_shapes"] = skipped
    for stage, loaded, pair_instruction in (
        ("reranker.forward", rerank_loaded, reranker.instruction),
        ("embedder.forward", embed_loaded, None),
    ):
        for shape in shapes:
            if not loaded.fits(shape):
                skipped.append(
                    f"{stage} {shape.label}: every row already carries "
                    f"{loaded.fixed_tokens} fixed tokens"
                )
                log(f"  skipped {skipped[-1]}")
                continue
            rows = build_rows(loaded, shape, chat, reranker_instruction=pair_instruction)
            call = functools.partial(forward, mx, loaded, rows)
            stages.append(
                measure(
                    stage,
                    shape.label,
                    [call] * calls_per_stage,
                    warmup=args.warmup,
                    mx=mx,
                    tokens_per_call=shape.batch * shape.tokens,
                )
            )

    if args.pool_size > 0:
        lengths = [chat.rng.choice(DOC_TOKEN_LENGTHS) for _ in range(args.pool_size)]
        documents = [chat.document(rerank_loaded.tokenizer, n) for n in lengths]
        pool_query = chat.query()
        budget = reranker.body_token_budget
        fixed = len(rerank_loaded.row_prefix) + len(rerank_loaded.row_suffix)
        pair_tokens = [
            fixed
            + min(
                len(
                    encode(
                        rerank_loaded.tokenizer,
                        format_reranker_pair(reranker.instruction, pool_query, document),
                    )
                ),
                budget,
            )
            for document in documents
        ]
        score_call = functools.partial(reranker.score, pool_query, documents)
        stages.append(
            measure(
                "reranker.score",
                f"{args.pool_size} pairs",
                [score_call] * (1 + args.warmup + args.pool_repetitions),
                warmup=args.warmup,
                mx=mx,
                tokens_per_call=sum(pair_tokens),
                extra={
                    "doc_target_tokens": lengths,
                    "pair_tokens": pair_tokens,
                    "padded_tokens_fixed_batches": fixed_batches_padded(
                        pair_tokens, args.reranker_batch_size
                    ),
                },
            )
        )

    memory["after_stages"] = memory_state(mx, torch)
    memory["mx_peak_gib_whole_run"] = max(
        [load["mx_peak_gib"] for load in loads.values()]
        + [stage["mx_peak_gib"] for stage in stages]
    )

    probes: list[dict[str, Any]] = []
    report["idle_probes"] = probes
    fitting = [s for s in shapes if rerank_loaded.fits(s)]
    smallest = min(fitting, key=lambda s: s.batch * s.tokens) if fitting else None
    probe_rows = (
        build_rows(rerank_loaded, smallest, chat, reranker_instruction=reranker.instruction)
        if smallest
        else None
    )
    for gap in args.idle_gaps:
        # What a search request runs, in its order, right after the idle gap;
        # then the two short calls once more to show how fast they recover.
        query_a, query_b = chat.query(), chat.query()
        sequence: list[tuple[str, Callable[[], object]]] = [
            (
                "embedder.embed_query",
                functools.partial(embedder.embed_query, query_a, instruction=instruction),
            ),
            ("pe_core.embed_text", functools.partial(pe_core.embed_text, query_a)),
        ]
        if smallest is not None and probe_rows is not None:
            sequence.append(
                (
                    f"reranker.forward {smallest.label}",
                    functools.partial(forward, mx, rerank_loaded, probe_rows),
                )
            )
        sequence += [
            (
                "embedder.embed_query again",
                functools.partial(embedder.embed_query, query_b, instruction=instruction),
            ),
            ("pe_core.embed_text again", functools.partial(pe_core.embed_text, query_b)),
        ]
        log(f"idle probe: sleeping {gap:g}s")
        gap_start = gpu_time_by_process()
        time.sleep(gap)
        probe: dict[str, Any] = {
            "idle_seconds": gap,
            **(gpu_time_delta(gap_start, gpu_time_by_process()) or {}),
            "calls_s": {name: round(time_call(call), 5) for name, call in sequence},
        }
        log(f"  {probe}")
        probes.append(probe)

    memory["final"] = memory_state(mx, torch)
    report["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    report["anomalies"] = find_anomalies(report, args.anomaly_seconds)
    return report


def find_anomalies(report: dict[str, Any], threshold_s: float) -> list[str]:
    notes: list[str] = []
    for st in report.get("stages", []):
        label = f"{st['stage']} {st['shape']}"
        if st["stage"] in SHORT_STAGES and st["max_s"] > threshold_s:
            slow = sum(1 for s in st["seconds"] if s > threshold_s)
            notes.append(
                f"{label}: {slow} of {st['repetitions']} timed calls over {threshold_s:g}s "
                f"(max {st['max_s']:.3f}s, p50 {st['p50_s']:.3f}s)"
            )
        if st["first_call_s"] > max(5 * st["p50_s"], st["p50_s"] + 0.5):
            notes.append(
                f"{label}: first call {st['first_call_s']:.3f}s vs p50 {st['p50_s']:.3f}s "
                f"(cold-call cost)"
            )
        if st["p95_s"] > 1.5 * st["p50_s"] and st["p95_s"] - st["p50_s"] > 0.05:
            notes.append(f"{label}: p95 {st['p95_s']:.3f}s is {st['p95_s'] / st['p50_s']:.1f}x p50")
        other = st.get("gpu_s_other")
        if other is not None and other > 0.02 * st["wall_s"]:
            notes.append(
                f"{label}: other processes used {other:.2f}s of GPU time during the stage's "
                f"{st['wall_s']:.1f}s ({st.get('gpu_s_other_busiest')})"
            )
    warm_p50 = {
        (st["stage"] if st["stage"] in SHORT_STAGES else f"{st['stage']} {st['shape']}"): st[
            "p50_s"
        ]
        for st in report.get("stages", [])
    }
    for probe in report.get("idle_probes", []):
        for name, seconds in probe.get("calls_s", {}).items():
            p50 = warm_p50.get(name.removesuffix(" again"))
            if (p50 is not None and seconds > 1.5 * p50 and seconds - p50 > 0.05) or (
                seconds > threshold_s and name.split(" ")[0] in SHORT_STAGES
            ):
                notes.append(
                    f"after {probe['idle_seconds']:g}s idle: {name} took {seconds:.3f}s "
                    f"(warm p50 {p50 if p50 is None else round(p50, 3)}s)"
                )
    before = (report.get("memory", {}).get("before_load") or {}).get("system") or {}
    after = (report.get("memory", {}).get("after_stages") or {}).get("system") or {}
    if before and after:
        swapouts = after.get("counters", {}).get("swapouts", 0) - before.get("counters", {}).get(
            "swapouts", 0
        )
        if swapouts > 0:
            notes.append(f"system swapped out {swapouts} pages during the run")
        if after.get("pressure_level") not in (None, "normal"):
            notes.append(f"memory pressure level after the stages: {after['pressure_level']}")
    return notes


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def print_report(report: dict[str, Any], out: TextIO) -> None:
    host = report["host"]
    pk = host["packages"]
    print(
        f"{host['host_class']}, {host['gpu_cores']}-core GPU, macOS {host['macos']}, "
        f"Python {host['python']}, mlx {pk.get('mlx')}, mlx-lm {pk.get('mlx-lm')}, "
        f"torch {pk.get('torch')}",
        file=out,
    )
    loads = ", ".join(f"{k} {v['seconds']:.1f}s" for k, v in report["load"].items())
    print(f"load: {loads}", file=out)
    print(
        f"\n{'stage':22s} {'shape':10s} {'n':>3s} {'first_s':>8s} {'min_s':>8s} {'p50_s':>8s} "
        f"{'p95_s':>8s} {'max_s':>8s} {'tok/s@p50':>10s} {'wall_s':>7s} {'otherGPU_s':>10s}",
        file=out,
    )
    for st in report["stages"]:
        rate = f"{st['tokens_per_s_p50']:10.1f}" if "tokens_per_s_p50" in st else f"{'-':>10s}"
        other = st.get("gpu_s_other")
        print(
            f"{st['stage']:22s} {st['shape']:10s} {st['repetitions']:3d} {st['first_call_s']:8.4f} "
            f"{st['min_s']:8.4f} {st['p50_s']:8.4f} {st['p95_s']:8.4f} {st['max_s']:8.4f} "
            f"{rate} {st['wall_s']:7.1f} {'-' if other is None else other:>10}",
            file=out,
        )
    memory = report["memory"]
    after = memory["after_stages"]
    print(
        f"\nall three resident (after stages): mx peak {memory.get('mx_peak_gib_whole_run')} GiB "
        f"(whole run), "
        f"mx active {after.get('mx_active_gib')} GiB, mx cache {after.get('mx_cache_gib')} GiB, "
        f"torch mps {after.get('torch_mps_allocated_gib')} GiB "
        f"(all Metal allocations {after.get('metal_driver_allocated_gib')} GiB), "
        f"max RSS {after.get('max_rss_gib')} GiB, "
        f"phys footprint {after.get('phys_footprint_gib')} GiB "
        f"(lifetime max {after.get('lifetime_max_phys_footprint_gib')} GiB)",
        file=out,
    )
    before_sys = memory["before_load"].get("system") or {}
    after_sys = after.get("system") or {}
    if before_sys and after_sys:
        print(
            f"memory_pressure free {before_sys.get('free_percent')}% -> "
            f"{after_sys.get('free_percent')}%, level {before_sys.get('pressure_level')} -> "
            f"{after_sys.get('pressure_level')}, swap used {before_sys.get('swap_used_gib')} -> "
            f"{after_sys.get('swap_used_gib')} GiB",
            file=out,
        )
    for probe in report.get("idle_probes", []):
        calls = ", ".join(f"{name} {s:.3f}s" for name, s in probe.get("calls_s", {}).items())
        print(
            f"after {probe['idle_seconds']:g}s idle (other processes' GPU "
            f"{probe.get('gpu_s_other')}s): {calls}",
            file=out,
        )
    anomalies = report.get("anomalies") or []
    print("anomalies: " + ("none" if not anomalies else ""), file=out)
    for note in anomalies:
        print(f"  - {note}", file=out)


def compare_reports(base_path: Path, other_path: Path, out: TextIO) -> None:
    """Both reports' stages side by side. Ratios are B/A: for seconds, above 1
    means B is slower; for tokens per second, above 1 means B is faster.
    `oGPU` is the GPU time other processes used during the stage on that
    host — a large value means that host's numbers were measured on a
    shared GPU."""
    base = json.loads(base_path.read_text())
    other = json.loads(other_path.read_text())
    for label, report in (("A", base), ("B", other)):
        host = report["host"]
        print(
            f"{label} = {host['host_class']}, {host['gpu_cores']}-core GPU, macOS {host['macos']}, "
            f"started {report['started_at']}",
            file=out,
        )
    for key in ("embedder", "reranker", "pe_core"):
        a = base.get("models", {}).get(key, {}).get("artifact_sha256")
        b = other.get("models", {}).get(key, {}).get("artifact_sha256")
        print(f"{key}: {'same bytes' if a and a == b else 'NOT VERIFIED SAME'}", file=out)
    print(
        f"\n{'stage':22s} {'shape':10s} {'A p50_s':>8s} {'B p50_s':>8s} {'B/A p50':>8s} "
        f"{'A p95_s':>8s} {'B p95_s':>8s} {'B/A p95':>8s} {'A tok/s':>8s} {'B tok/s':>8s} "
        f"{'A oGPU':>7s} {'B oGPU':>7s}",
        file=out,
    )
    other_stages = {(st["stage"], st["shape"]): st for st in other["stages"]}
    for st in base["stages"]:
        match = other_stages.get((st["stage"], st["shape"]))
        if match is None:
            continue
        rates = (
            f"{st['tokens_per_s_p50']:8.1f} {match['tokens_per_s_p50']:8.1f}"
            if "tokens_per_s_p50" in st
            else f"{'-':>8s} {'-':>8s}"
        )
        print(
            f"{st['stage']:22s} {st['shape']:10s} {st['p50_s']:8.4f} {match['p50_s']:8.4f} "
            f"{match['p50_s'] / st['p50_s']:8.3f} {st['p95_s']:8.4f} {match['p95_s']:8.4f} "
            f"{match['p95_s'] / st['p95_s']:8.3f} {rates} "
            f"{st.get('gpu_s_other', '-')!s:>7} {match.get('gpu_s_other', '-')!s:>7}",
            file=out,
        )
    other_probes = {p["idle_seconds"]: p for p in other.get("idle_probes", [])}
    for probe in base.get("idle_probes", []):
        match = other_probes.get(probe["idle_seconds"])
        if match is None:
            continue
        calls = ", ".join(
            f"{name} A {s:.3f}s B {match['calls_s'][name]:.3f}s"
            for name, s in probe.get("calls_s", {}).items()
            if name in match.get("calls_s", {})
        )
        print(f"after {probe['idle_seconds']:g}s idle: {calls}", file=out)
    for key in ("embedder", "reranker", "pe_core"):
        a_load, b_load = base["load"][key]["seconds"], other["load"][key]["seconds"]
        print(f"load {key}: A {a_load:.1f}s  B {b_load:.1f}s  B/A {b_load / a_load:.2f}", file=out)
    for label, report in (("A", base), ("B", other)):
        after = report["memory"]["after_stages"]
        print(
            f"memory {label}: phys footprint {after.get('phys_footprint_gib')} GiB, mx peak "
            f"{report['memory'].get('mx_peak_gib_whole_run')} GiB, mx cache "
            f"{after.get('mx_cache_gib')} GiB, torch mps {after.get('torch_mps_allocated_gib')} GiB",
            file=out,
        )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("A_JSON", "B_JSON"),
        help="print both reports' stages side by side with B/A ratios, then exit",
    )
    parser.add_argument(
        "--embedder",
        default=constants.TEXT_EMBEDDING_MODEL_REPO,
        help="Qwen3-Embedding: Hub repo id or a local MLX-layout directory",
    )
    parser.add_argument(
        "--embedder-revision",
        default=constants.TEXT_EMBEDDING_MODEL_REVISION,
        help="commit sha for a Hub repo; for a local directory, recorded in model_id only",
    )
    parser.add_argument(
        "--reranker",
        help="Qwen3-Reranker: the local conversion's directory (models/manifest.lock.yaml "
        "output_dir under the data root) or a Hub repo id",
    )
    parser.add_argument(
        "--reranker-revision",
        default=constants.RERANKER_MODEL_REVISION,
        help="commit sha for a Hub repo; for a local directory the upstream sha, recorded in "
        "model_id only (as the provider factory does)",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument(
        "--pe-core", default=constants.MULTIMODAL_EMBEDDING_MODEL_REPO, help="PE-Core Hub repo id"
    )
    parser.add_argument("--pe-core-revision", default=constants.MULTIMODAL_EMBEDDING_MODEL_REVISION)
    parser.add_argument(
        "--pe-core-cache-dir", type=Path, help="huggingface_hub cache (default: its own)"
    )
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma-separated BATCHxTOKENS")
    parser.add_argument(
        "--pool-size",
        type=int,
        default=10,
        help="rerank-pool documents for reranker.score (0 skips)",
    )
    parser.add_argument("--pool-repetitions", type=int, default=None)
    parser.add_argument(
        "--idle-gaps",
        default="3,30,300",
        help="comma-separated seconds of idle before each probe ('' for none)",
    )
    parser.add_argument(
        "--idle-baseline-seconds",
        type=float,
        default=10.0,
        help="idle seconds before loading, to measure other processes' GPU use",
    )
    parser.add_argument(
        "--anomaly-seconds", type=float, default=1.0, help="flag short-stage calls slower than this"
    )
    parser.add_argument("--no-digest", dest="digest", action="store_false")
    parser.add_argument("--json", action="store_true", help="report as JSON on stdout")
    args = parser.parse_args(argv)
    if args.compare is None and not args.reranker:
        parser.error("--reranker is required (the local conversion's directory or a repo id)")
    if args.warmup < 0 or args.repetitions < 1 or args.pool_size < 0:
        parser.error("--warmup must be >= 0, --repetitions >= 1, --pool-size >= 0")
    if args.pool_repetitions is None:
        args.pool_repetitions = args.repetitions
    try:
        args.idle_gaps = [float(g) for g in args.idle_gaps.split(",") if g.strip()]
    except ValueError:
        parser.error("--idle-gaps must be comma-separated numbers of seconds")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare is not None:
        compare_reports(args.compare[0], args.compare[1], sys.stdout)
        return 0
    # Provider logs (structlog's default printer writes to stdout) must not
    # interleave with the JSON report.
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))
    report = run_benchmark(args)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
        print_report(report, sys.stderr)
    else:
        print_report(report, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
