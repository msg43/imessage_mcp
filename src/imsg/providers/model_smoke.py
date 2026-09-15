"""Run every pinned model for real, once — `scripts/smoke_test_models.py`.

`imsg models verify` (`imsg.providers.manifest`) proves the lock still
points at the commits it names; it never downloads or executes anything.
This module is the other half of the SPEC's model-manifest requirement —
the "short smoke-test result" per model. For each entry of
`models/manifest.lock.yaml`, in this order:

1. **Download** the pinned snapshot:
   `huggingface_hub.snapshot_download(repo, revision=<pinned sha>,
   ignore_patterns=DOWNLOAD_IGNORE_PATTERNS)` into the default Hugging
   Face cache (public artifacts, not corpus-derived state — the
   encrypted-volume rule does not apply). Everything but the known
   duplicate weight formats is fetched, so the snapshot is complete for
   the providers' own full `snapshot_download` calls. `--skip-download`
   uses the snapshot directory already in the cache instead
   (`cached_snapshot_dir`; `huggingface_hub` would refuse a
   `local_files_only` download for any snapshot missing a file, and
   never contacts the Hub).
2. **Checksum** the snapshot as `artifact_sha256` — exact definition:
   take every regular file in the snapshot directory whose *name* matches
   one of `ARTIFACT_FILE_PATTERNS` (weights: `*.safetensors`, `*.npz`;
   config/tokenizer: `*.json`, `*.txt`, `*.model`, `*.tiktoken`,
   `*.jinja`, `*.py`, `*.jsonl`), skipping hidden files and directories;
   for each, one line `"<relative POSIX path>  <hex sha256 of the file
   bytes>\\n"` (two spaces, like `sha256sum`); sort the lines by relative
   path (plain string order); `artifact_sha256` is the hex SHA-256 of the
   UTF-8 bytes of those lines concatenated. `README.md`,
   `.gitattributes` and the `*.bin`/`*.pt` duplicates of safetensors
   weights (PE-Core's mirror ships both) are deliberately outside the
   definition, so the digest is the same whether or not a full snapshot
   was fetched. `artifact_digest()` is the reference implementation.
3. **Build the real provider through `imsg.providers.factory`** — the
   same builders every CLI command uses — from a `Config` whose model
   fields are exactly the manifest pins (`config_from_manifest`), and run
   one minimal, fictional input per role (`check_*` below). The provider's
   `model_id` (`<repo>@<revision>`) is recorded so the report can prove
   the pinned revision is what actually ran.
4. **Measure** load time, inference time and peak memory:
   `mlx.core.get_peak_memory()` for the MLX runtimes (reset before the
   load), `resource.getrusage` max RSS for torch (PE-Core) and Apple
   Vision. Every (entry, role) runs in its own child process, so max RSS
   is per model and two large models are never resident at once; the
   parent reports `vm_stat`'s free memory before each child. All `*_gb`
   values are GiB (2^30 bytes).
5. With **`--write`**, replace that entry's `smoke_test` record and
   `artifact_sha256` in the lock — a text-level edit that leaves every
   other byte of the file as it was (`apply_lock_updates`), verified by
   re-parsing before the file is replaced.

Every synthetic input is fictional (`EMBED_DOCUMENTS`, the two-topic
`synthetic_boundary_window()`, a rendered PNG, a `say`-synthesised
sentence, a drawn shape); nothing here reads the corpus, and no path
under `~/Library/Messages` is touched — which is why the `Config` is
assembled with `model_construct` (its root validators stat the live
`chat.db` path to prove containment, a check that has no meaning for a
tool that never opens any of those paths). Text that lands in the lock
or the report passes through `scrub_private` (home directory and
hostname removed) because the lock is public.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TextIO

import yaml

from imsg.config.schema import (
    Config,
    DatabaseConfig,
    EmbeddingConfig,
    EnrichmentConfig,
    ExportConfig,
    McpConfig,
    McpPublicConfig,
    ModelsConfig,
    MultimodalEmbeddingConfig,
    PathsConfig,
    RetrievalConfig,
    SegmentationConfig,
    SyncConfig,
    SyncSourceConfig,
)
from imsg.enrich.audio import convert_to_whisper_wav
from imsg.errors import ImsgError
from imsg.hashing import sha256_file, sha256_text
from imsg.providers.factory import (
    build_boundary_provider,
    build_enrichment_providers,
    build_multimodal_provider,
    build_reranker,
    build_text_provider,
    read_prompt_text,
    resolve_prompt_path,
)
from imsg.providers.manifest import (
    STATUS_SYSTEM,
    ManifestEntry,
    ManifestLock,
    default_manifest_path,
    entries_by_role,
    load_manifest,
)
from imsg.segment.models import MessageForSegmentation

GIB = float(2**30)

ARTIFACT_FILE_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.npz",
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.jinja",
    "*.py",
    "*.jsonl",
)
"""The files `artifact_sha256` covers (module docstring, step 2): a
superset of what `mlx_lm.load` fetches for a repo and of what
`imsg.embed.pe_core_multimodal` fetches."""

DOWNLOAD_IGNORE_PATTERNS: tuple[str, ...] = ("*.bin", "*.pt", "*.pth")
"""What step 1 leaves out: duplicate weight formats next to safetensors
(PE-Core's mirror ships a 9.7 GB `open_clip_pytorch_model.bin` twin of
its safetensors, which the provider never reads). Everything else in the
revision is fetched, so the hashed set is always complete."""

SMOKE_RUN_ORDER: tuple[str, ...] = (
    "qwen3-embedding-8b",
    "qwen3-reranker-8b",
    "whisper-large-v3",
    "pe-core-g14-448",
    "apple-vision-ocr",
    "qwen3.5-35b-a3b",
)
"""Default order: largest model last, so an early failure costs the least
download. Entries the lock adds later run after these, in lock order."""

MLX_ROLES: frozenset[str] = frozenset(
    {"text_embedding", "reranker", "segment_boundaries", "image_caption", "transcription"}
)
"""Roles whose runtime is MLX — peak memory comes from `mlx.core`; the
rest (torch, Apple Vision) are measured as max RSS."""

SMOKE_QUERY_INSTRUCTION = (
    "Given a personal message search query, retrieve relevant conversation segments"
)
"""`embedding.query_instruction` as `config.example.yaml` ships it."""

CHILD_MODULE = "imsg.providers.model_smoke"
RESULT_JSON_SUFFIX = ".result.json"

# --------------------------------------------------------------------------
# synthetic inputs — all fictional
# --------------------------------------------------------------------------

EMBED_DOCUMENTS: tuple[str, str] = (
    "The bakery on the corner opens at seven and sells sourdough loaves.",
    "The rocket's second stage separated on schedule after the launch.",
)
EMBED_QUERY = "what time does the bakery open"
EMBED_RELEVANT_INDEX = 0

RERANK_QUERY = "how do I put a slipped bicycle chain back on"
RERANK_DOCUMENTS: tuple[str, str] = (
    "Lift the rear wheel, shift to the smallest cog, and guide the chain back onto the "
    "teeth while turning the pedal.",
    "The museum's new exhibit on deep-sea sponges opens next spring.",
)

TRANSCRIPTION_SENTENCE = "The blue kite drifted over the quiet harbor at dawn."

OCR_LINES: tuple[str, str] = ("Harbor Kite Festival", "Gate opens at nine")
CAPTION_KEYWORDS: tuple[str, ...] = (
    "kite",
    "festival",
    "harbor",
    "gate",
    "nine",
    "text",
    "red",
    "circle",
    "white",
    "black",
    "sign",
    "poster",
)
"""Things visible in the rendered image (`render_text_image`): its words,
its colours, its shapes. A caption must mention at least one."""

MULTIMODAL_MATCHING_TEXT = "a red circle on a white background"
MULTIMODAL_UNRELATED_TEXT = "a photograph of a snowy mountain at night"

_BOUNDARY_BASE_TIME = datetime(2026, 1, 10, 10, 0, tzinfo=UTC)
_BOUNDARY_SENDERS = ("Alpha", "Bravo")
_BOUNDARY_TEXTS: tuple[str, ...] = (
    "Are we still doing the picnic on Saturday?",
    "Yes, noon at the big oak by the pond.",
    "I will bring the lemonade and the folding chairs.",
    "Great, I have the sandwiches and a blanket covered.",
    "Should we invite the neighbours from the corner house?",
    "Sure, the more the merrier. I will message them.",
    "Different question: my bike chain keeps slipping off the big gear.",
    "Is it stretched? A worn chain skips under load.",
    "It is the original chain, about three years old.",
    "Then replace it, and check the cassette teeth for wear too.",
    "Any shop you recommend for the parts?",
    "The one by the river had a good selection last time.",
)
BOUNDARY_DESIGNED_SPLIT = 6
"""Index of the first message of the second topic in
`synthetic_boundary_window()` — reported, never asserted."""


def synthetic_boundary_window() -> list[MessageForSegmentation]:
    """Twelve fictional messages between two invented senders: a picnic
    plan, then a bicycle-chain question starting at index 6."""
    return [
        MessageForSegmentation(
            message_id=index + 1,
            source_guid=f"smoke-{index + 1}",
            chat_id=1,
            sent_at=_BOUNDARY_BASE_TIME + timedelta(minutes=index),
            is_from_me=(index % 2 == 0),
            sender_short_name=_BOUNDARY_SENDERS[index % 2],
            text=text,
            is_unsent=False,
            is_edited=False,
            has_attachments=False,
        )
        for index, text in enumerate(_BOUNDARY_TEXTS)
    ]


def _pil(name: str) -> Any:
    try:
        return importlib.import_module(f"PIL.{name}")
    except ImportError as exc:
        raise SmokeError(
            f"Pillow is needed to render the synthetic images ({exc}) — install the `models` extra"
        ) from exc


def _font(size: int) -> Any:
    image_font = _pil("ImageFont")
    for candidate in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Geneva.ttf"):
        try:
            return image_font.truetype(candidate, size)
        except OSError:
            continue
    return image_font.load_default(size=size)


def render_text_image(path: Path, lines: Sequence[str] = OCR_LINES) -> Path:
    """A 900x340 white PNG with `lines` in 56px black text (one per row)
    and a red disc in the lower right — the OCR and caption input."""
    image_module, draw_module = _pil("Image"), _pil("ImageDraw")
    image = image_module.new("RGB", (900, 340), "white")
    draw = draw_module.Draw(image)
    font = _font(56)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill="black", font=font)
        y += 110
    draw.ellipse((730, 200, 860, 330), fill="red")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def render_shape_image(path: Path) -> Path:
    """A 512x512 white PNG with one large red disc — the PE-Core input
    that `MULTIMODAL_MATCHING_TEXT` describes."""
    image_module, draw_module = _pil("Image"), _pil("ImageDraw")
    image = image_module.new("RGB", (512, 512), "white")
    draw_module.Draw(image).ellipse((96, 96, 416, 416), fill="red")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def synthesize_speech_wav(
    work_dir: Path,
    sentence: str = TRANSCRIPTION_SENTENCE,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> Path:
    """`say` renders `sentence` to AIFF; ffmpeg converts it to the 16 kHz
    mono WAV the transcription provider expects (the pipeline's own
    `convert_to_whisper_wav`)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    aiff = work_dir / "smoke_speech.aiff"
    try:
        run(
            ["say", "-o", str(aiff), sentence],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SmokeError(f"could not synthesise speech with `say`: {exc}") from exc
    return convert_to_whisper_wav(aiff, work_dir / "smoke_speech_16k_mono.wav", timeout_seconds=120)


# --------------------------------------------------------------------------
# errors + scrubbing
# --------------------------------------------------------------------------


class SmokeError(ImsgError):
    """The smoke run itself could not proceed (lock problem, unknown
    entry/role, a synthetic input that could not be produced)."""


class SmokeCheckFailed(ImsgError):
    """A provider ran but its output failed the role's assertion."""


def scrub_private(text: str, *, home: Path | None = None, hostname: str | None = None) -> str:
    """One line, with the home directory replaced by `~` and the hostname
    by `<host>` — the lock and the report are public artifacts."""
    out = text
    home_str = str(home if home is not None else Path.home())
    if home_str and home_str != "/":
        out = out.replace(home_str, "~")
    node = platform.node() if hostname is None else hostname
    if node:
        out = out.replace(node, "<host>")
        short = node.split(".", 1)[0]
        if len(short) > 3:
            out = out.replace(short, "<host>")
    return " ".join(out.split())


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------
# artifact checksum
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    sha256: str
    files: tuple[ArtifactFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


def iter_artifact_files(
    snapshot: Path, patterns: Sequence[str] = ARTIFACT_FILE_PATTERNS
) -> list[Path]:
    """The regular files under `snapshot` (symlinks into the Hub cache's
    blob store followed) whose name matches `patterns`, hidden entries
    skipped, in relative-path order."""
    matched: list[tuple[str, Path]] = []
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            continue
        matched.append((relative.as_posix(), path))
    matched.sort(key=lambda item: item[0])
    return [path for _, path in matched]


def artifact_digest(
    snapshot: Path, patterns: Sequence[str] = ARTIFACT_FILE_PATTERNS
) -> ArtifactDigest:
    """`artifact_sha256` exactly as the module docstring defines it, plus
    the per-file lines it was computed from."""
    if not snapshot.is_dir():
        raise SmokeError(f"snapshot directory does not exist: '{snapshot}'")
    files: list[ArtifactFile] = []
    for path in iter_artifact_files(snapshot, patterns):
        relative = path.relative_to(snapshot).as_posix()
        files.append(ArtifactFile(relative, sha256_file(path), path.stat().st_size))
    if not files:
        raise SmokeError(
            f"snapshot '{snapshot}' holds no weight or config files matching {list(patterns)}"
        )
    files.sort(key=lambda item: item.relative_path)
    listing = "".join(f"{item.relative_path}  {item.sha256}\n" for item in files)
    return ArtifactDigest(sha256=sha256_text(listing), files=tuple(files))


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

Downloader = Callable[..., str]
"""`(repo_id=, revision=, ignore_patterns=) -> snapshot directory`. The
default is `huggingface_hub.snapshot_download`; tests inject a stand-in
so nothing touches the network."""


def hf_snapshot_download(**kwargs: Any) -> str:
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise SmokeError(
            f"huggingface_hub is not importable ({exc}) — install the `models` extra"
        ) from exc
    return str(hub.snapshot_download(**kwargs))


def default_hub_cache_dir() -> Path:
    """Where `huggingface_hub` keeps snapshots (`HF_HUB_CACHE` / `HF_HOME`
    honoured, as the library itself resolves them)."""
    try:
        constants = importlib.import_module("huggingface_hub.constants")
    except ImportError as exc:
        raise SmokeError(
            f"huggingface_hub is not importable ({exc}) — install the `models` extra"
        ) from exc
    return Path(str(constants.HF_HUB_CACHE))


def cached_snapshot_dir(repo: str, revision: str, cache_dir: Path | None = None) -> Path:
    """The cache's snapshot directory for `repo` at commit `revision` —
    `<cache>/models--<owner>--<name>/snapshots/<revision>` — which must
    already exist; nothing is fetched."""
    root = cache_dir if cache_dir is not None else default_hub_cache_dir()
    snapshot = root / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if not snapshot.is_dir():
        raise SmokeError(
            f"{repo}@{revision} is not in the Hugging Face cache at {scrub_private(str(root))} "
            f"(expected {scrub_private(str(snapshot))}); run without --skip-download"
        )
    return snapshot


def fetch_snapshot(
    entry: ManifestEntry,
    downloader: Downloader,
    *,
    skip_download: bool,
    cache_dir: Path | None = None,
) -> Path:
    """Step 1: the pinned snapshot directory — downloaded (everything but
    `DOWNLOAD_IGNORE_PATTERNS`) or, with `skip_download`, located in the
    cache."""
    if not entry.repo or not entry.revision:
        raise SmokeError(f"entry '{entry.name}' has no repo/revision to download")
    if skip_download:
        return cached_snapshot_dir(entry.repo, entry.revision, cache_dir)
    return Path(
        downloader(
            repo_id=entry.repo,
            revision=entry.revision,
            ignore_patterns=list(DOWNLOAD_IGNORE_PATTERNS),
        )
    )


# --------------------------------------------------------------------------
# memory + host probes
# --------------------------------------------------------------------------


class PeakMemoryProbe(Protocol):
    source: str

    def reset(self) -> None: ...

    def read_gb(self) -> float: ...


def max_rss_gb() -> float:
    """`ru_maxrss` — bytes on macOS, kibibytes elsewhere — as GiB."""
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024.0
    return rss / GIB


class MlxPeakMemoryProbe:
    """`mlx.core.reset_peak_memory()` before the load, `get_peak_memory()`
    after the check: the high-water mark of everything MLX allocated
    (weights, KV cache, activations) in this process."""

    source = "mlx.core.get_peak_memory"

    def __init__(self) -> None:
        try:
            self._mx = importlib.import_module("mlx.core")
        except ImportError as exc:
            raise SmokeError(
                f"mlx.core is not importable ({exc}) — install the `models` extra"
            ) from exc

    def reset(self) -> None:
        self._mx.reset_peak_memory()

    def read_gb(self) -> float:
        return float(self._mx.get_peak_memory()) / GIB

    def active_gb(self) -> float:
        return float(self._mx.get_active_memory()) / GIB


class RssPeakMemoryProbe:
    """Process max RSS — monotonic for the process lifetime, which is why
    every (entry, role) gets its own child process."""

    source = "resource.getrusage(RUSAGE_SELF).ru_maxrss"

    def reset(self) -> None:
        return None

    def read_gb(self) -> float:
        return max_rss_gb()


def default_probe_factory(memory_kind: str) -> PeakMemoryProbe:
    if memory_kind == "mlx":
        return MlxPeakMemoryProbe()
    return RssPeakMemoryProbe()


_VM_STAT_LINE_RE = re.compile(r"^(Pages (?:free|inactive|speculative)):\s+(\d+)\.")
_VM_STAT_PAGE_RE = re.compile(r"page size of (\d+) bytes")


def parse_vm_stat(text: str) -> float | None:
    """`vm_stat` output -> (free + inactive + speculative) pages as GiB —
    what macOS can hand out without compressing or swapping; `None` when
    the text does not look like `vm_stat`."""
    page_match = _VM_STAT_PAGE_RE.search(text)
    if page_match is None:
        return None
    page_size = int(page_match.group(1))
    pages = 0
    seen = 0
    for line in text.splitlines():
        match = _VM_STAT_LINE_RE.match(line.strip())
        if match:
            pages += int(match.group(2))
            seen += 1
    if seen == 0:
        return None
    return pages * page_size / GIB


def vm_stat_free_gb(run: Callable[..., Any] = subprocess.run) -> float | None:
    try:
        proc = run(["vm_stat"], capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return parse_vm_stat(str(proc.stdout))


def _sysctl(name: str) -> str | None:
    try:
        proc = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def detect_host_class() -> str:
    """Chip family and memory size, never a hostname — e.g. `Apple M2
    Ultra, 128 GB unified memory`."""
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    memsize = _sysctl("hw.memsize")
    total_gb: float | None = None
    if memsize and memsize.isdigit():
        total_gb = int(memsize) / GIB
    else:
        try:
            total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GIB
        except (ValueError, OSError, AttributeError):
            total_gb = None
    memory_kind = "unified memory" if "Apple" in chip else "RAM"
    if total_gb is None:
        return f"{chip}, memory size unknown"
    return f"{chip}, {total_gb:.0f} GB {memory_kind}"


def macos_version() -> str | None:
    release = platform.mac_ver()[0]
    return release or None


# --------------------------------------------------------------------------
# the role checks — one minimal synthetic input each
# --------------------------------------------------------------------------


def _finite_floats(vector: Sequence[float], label: str) -> list[float]:
    values = [float(v) for v in vector]
    if any(not math.isfinite(v) for v in values):
        raise SmokeCheckFailed(f"{label} embedding contains a non-finite component")
    return values


def _require_unit_vector(vector: Sequence[float], dim: int, label: str) -> list[float]:
    values = _finite_floats(vector, label)
    if len(values) != dim:
        raise SmokeCheckFailed(
            f"{label} embedding has {len(values)} components, expected dim {dim}"
        )
    norm = math.sqrt(math.fsum(v * v for v in values))
    if abs(norm - 1.0) > 1e-3:
        raise SmokeCheckFailed(f"{label} embedding is not unit-norm (|v| = {norm:.6f})")
    return values


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def check_text_embedding(provider: Any, *, dim: int, instruction: str) -> str:
    """Two documents and one query: `dim` components, unit norm, and the
    query closer to the relevant document than to the unrelated one."""
    documents = provider.embed_documents(list(EMBED_DOCUMENTS))
    if len(documents) != len(EMBED_DOCUMENTS):
        raise SmokeCheckFailed(
            f"embed_documents returned {len(documents)} vectors for {len(EMBED_DOCUMENTS)} texts"
        )
    vectors = [
        _require_unit_vector(vector, dim, f"document {index}")
        for index, vector in enumerate(documents)
    ]
    query = _require_unit_vector(
        provider.embed_query(EMBED_QUERY, instruction=instruction), dim, "query"
    )
    relevant = _cosine(query, vectors[EMBED_RELEVANT_INDEX])
    unrelated = _cosine(query, vectors[1 - EMBED_RELEVANT_INDEX])
    if not relevant > unrelated:
        raise SmokeCheckFailed(
            f"query is not closer to the relevant document: cos(relevant)={relevant:.4f} "
            f"<= cos(unrelated)={unrelated:.4f}"
        )
    return (
        f"dim={dim}, unit-norm; cos(query, relevant)={relevant:.4f} > "
        f"cos(query, unrelated)={unrelated:.4f}"
    )


def check_reranker(provider: Any) -> str:
    """One query, one relevant and one irrelevant document: both scores in
    [0, 1], the relevant one higher."""
    scores = [float(s) for s in provider.score(RERANK_QUERY, list(RERANK_DOCUMENTS))]
    if len(scores) != len(RERANK_DOCUMENTS):
        raise SmokeCheckFailed(
            f"score returned {len(scores)} scores for {len(RERANK_DOCUMENTS)} documents"
        )
    for index, score in enumerate(scores):
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise SmokeCheckFailed(f"score for document {index} is outside [0, 1]: {score!r}")
    if not scores[0] > scores[1]:
        raise SmokeCheckFailed(
            f"relevant document did not outscore the irrelevant one: {scores[0]:.4f} <= {scores[1]:.4f}"
        )
    return f"P(yes): relevant={scores[0]:.4f} > irrelevant={scores[1]:.4f}, both in [0, 1]"


def check_boundaries(provider: Any) -> str:
    """The 12-message two-topic window: a sorted, unique, in-range list
    comes back. The exact split is reported, not asserted."""
    window = synthetic_boundary_window()
    found = provider.detect_boundaries(window)
    if not isinstance(found, list):
        raise SmokeCheckFailed(
            f"detect_boundaries returned {type(found).__name__}, expected a list"
        )
    indices: list[int] = []
    for item in found:
        if isinstance(item, bool) or not isinstance(item, int):
            raise SmokeCheckFailed(f"boundary list holds a non-integer: {item!r}")
        indices.append(item)
    if indices != sorted(set(indices)):
        raise SmokeCheckFailed(f"boundaries are not sorted and unique: {indices}")
    out_of_range = [i for i in indices if not 0 < i < len(window)]
    if out_of_range:
        raise SmokeCheckFailed(
            f"boundaries outside 1..{len(window) - 1}: {out_of_range} (got {indices})"
        )
    note = (
        "matches the designed split"
        if indices == [BOUNDARY_DESIGNED_SPLIT]
        else f"designed split was at index {BOUNDARY_DESIGNED_SPLIT} (not asserted)"
    )
    return f"boundaries={indices} for a {len(window)}-message two-topic window; {note}"


def check_ocr(provider: Any, image_path: Path) -> str:
    """Both rendered lines come back (case-insensitive, whitespace-normalised)."""
    text = str(provider.recognize_text(image_path))
    normalised = " ".join(text.split()).lower()
    missing = [line for line in OCR_LINES if line.lower() not in normalised]
    if missing:
        raise SmokeCheckFailed(f"OCR did not recover {missing!r}; recognised text was {text!r}")
    return f"recovered both rendered lines verbatim: {' / '.join(OCR_LINES)}"


def check_caption(provider: Any, image_path: Path) -> str:
    """A non-empty caption that mentions something actually visible."""
    caption = str(provider.caption(image_path))
    if not caption.strip():
        raise SmokeCheckFailed("caption is empty")
    lowered = caption.lower()
    hits = [keyword for keyword in CAPTION_KEYWORDS if keyword in lowered]
    if not hits:
        raise SmokeCheckFailed(
            f"caption mentions nothing visible (none of {list(CAPTION_KEYWORDS)}): {caption!r}"
        )
    return f"caption mentions {hits[:3]}: {_shorten(' '.join(caption.split()), 160)!r}"


def check_transcription(provider: Any, wav_path: Path) -> str:
    """The transcript's words equal the synthesised sentence's words."""
    transcript = str(provider.transcribe(wav_path))
    got, want = _words(transcript), _words(TRANSCRIPTION_SENTENCE)
    if got != want:
        raise SmokeCheckFailed(f"transcript words differ: got {got}, expected {want}")
    return f"transcript matched all {len(want)} words of the synthesised sentence: {transcript.strip()!r}"


def check_multimodal(provider: Any, image_path: Path, *, dim: int) -> str:
    """One drawn image and two texts: `dim` components, unit norm, and the
    matching text scores above the unrelated one."""
    images = provider.embed_images([image_path])
    if len(images) != 1:
        raise SmokeCheckFailed(f"embed_images returned {len(images)} vectors for 1 image")
    image = _require_unit_vector(images[0], dim, "image")
    matching = _require_unit_vector(
        provider.embed_text(MULTIMODAL_MATCHING_TEXT), dim, "matching text"
    )
    unrelated = _require_unit_vector(
        provider.embed_text(MULTIMODAL_UNRELATED_TEXT), dim, "unrelated text"
    )
    cos_match, cos_unrelated = _cosine(image, matching), _cosine(image, unrelated)
    if not cos_match > cos_unrelated:
        raise SmokeCheckFailed(
            f"matching text did not outscore the unrelated one: cos={cos_match:.4f} <= {cos_unrelated:.4f}"
        )
    return (
        f"dim={dim}, unit-norm; cos(image, matching text)={cos_match:.4f} > "
        f"cos(image, unrelated text)={cos_unrelated:.4f}"
    )


# --------------------------------------------------------------------------
# config from the lock, providers from the factory
# --------------------------------------------------------------------------


def _pin(by_role: Mapping[str, ManifestEntry], role: str) -> tuple[str, str]:
    entry = by_role.get(role)
    if entry is None:
        raise SmokeError(f"the lock has no entry for role '{role}'")
    if not entry.repo or not entry.revision:
        raise SmokeError(f"lock entry '{entry.name}' (role '{role}') has no repo/revision pin")
    return entry.repo, entry.revision


def config_from_manifest(lock: ManifestLock, *, data_root: Path) -> Config:
    """A `Config` whose every model field equals the lock's pins and whose
    backend is `real`, for `imsg.providers.factory`. The sections are
    validated individually; the root is assembled with `model_construct`
    so the root path-containment validators — which resolve the live
    `chat.db` path — never run (module docstring). `data_root` only
    decides where prompt overrides would be looked for."""
    by_role = entries_by_role(lock)
    text_repo, text_revision = _pin(by_role, "text_embedding")
    mm_repo, mm_revision = _pin(by_role, "multimodal_embedding")
    reranker_repo, reranker_revision = _pin(by_role, "reranker")
    boundary_repo, boundary_revision = _pin(by_role, "segment_boundaries")
    caption_repo, caption_revision = _pin(by_role, "image_caption")
    whisper_repo, whisper_revision = _pin(by_role, "transcription")
    return Config.model_construct(
        paths=PathsConfig(data_root=data_root),
        database=DatabaseConfig.model_validate({"password": "env:IMSG_SMOKE_TEST_UNUSED"}),
        sync=SyncConfig(
            sources=[
                SyncSourceConfig(name="smoke-test-unused", chat_db=Path("/nonexistent/chat.db"))
            ]
        ),
        segmentation=SegmentationConfig(
            boundary_model=boundary_repo, boundary_revision=boundary_revision
        ),
        enrichment=EnrichmentConfig(
            transcription_model=whisper_repo,
            transcription_revision=whisper_revision,
            caption_model=caption_repo,
            caption_revision=caption_revision,
        ),
        embedding=EmbeddingConfig(
            model=text_repo,
            revision=text_revision,
            query_instruction=SMOKE_QUERY_INSTRUCTION,
            multimodal=MultimodalEmbeddingConfig(model=mm_repo, revision=mm_revision),
        ),
        retrieval=RetrievalConfig(
            reranker_model=reranker_repo, reranker_revision=reranker_revision
        ),
        models=ModelsConfig(backend="real"),
        mcp=McpConfig(public=McpPublicConfig(scope="allowlist")),
        export=ExportConfig(
            gcp_project="smoke-test-unused",
            gcs_bucket="smoke-test-unused",
            data_store_id="smoke-test-unused",
        ),
    )


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """How one manifest role is exercised: `memory` selects the peak
    probe; `prepare` produces the synthetic input (untimed); `build`
    constructs the provider through the factory; `check` runs it."""

    role: str
    memory: str
    prepare: Callable[[Path], Any]
    build: Callable[[Config, Path], Any]
    check: Callable[[Any, Config, Any], str]


def _no_input(work_dir: Path) -> None:
    return None


def _build_boundary(cfg: Config, work_dir: Path) -> Any:
    resolved = resolve_prompt_path(cfg.paths.data_root, cfg.segmentation.boundary_prompt)
    if resolved is None:
        raise SmokeError(
            f"boundary prompt '{cfg.segmentation.boundary_prompt}' not found under data_root "
            f"or in the repository"
        )
    template = read_prompt_text(resolved.path, field_name="segmentation.boundary_prompt")
    return build_boundary_provider(cfg, template)


def _build_multimodal(cfg: Config, work_dir: Path) -> Any:
    provider = build_multimodal_provider(cfg)
    if provider is None:
        raise SmokeError("embedding.multimodal is disabled in the smoke config")
    return provider


ROLE_SPECS: dict[str, RoleSpec] = {
    "text_embedding": RoleSpec(
        "text_embedding",
        "mlx",
        _no_input,
        lambda cfg, work_dir: build_text_provider(cfg),
        lambda provider, cfg, prepared: check_text_embedding(
            provider, dim=cfg.embedding.dim, instruction=cfg.embedding.query_instruction
        ),
    ),
    "reranker": RoleSpec(
        "reranker",
        "mlx",
        _no_input,
        lambda cfg, work_dir: build_reranker(cfg),
        lambda provider, cfg, prepared: check_reranker(provider),
    ),
    "segment_boundaries": RoleSpec(
        "segment_boundaries",
        "mlx",
        _no_input,
        _build_boundary,
        lambda provider, cfg, prepared: check_boundaries(provider),
    ),
    "ocr": RoleSpec(
        "ocr",
        "rss",
        lambda work_dir: render_text_image(work_dir / "smoke_text.png"),
        lambda cfg, work_dir: build_enrichment_providers(cfg).ocr,
        lambda provider, cfg, prepared: check_ocr(provider, prepared),
    ),
    "image_caption": RoleSpec(
        "image_caption",
        "mlx",
        lambda work_dir: render_text_image(work_dir / "smoke_text.png"),
        lambda cfg, work_dir: build_enrichment_providers(cfg).caption,
        lambda provider, cfg, prepared: check_caption(provider, prepared),
    ),
    "transcription": RoleSpec(
        "transcription",
        "mlx",
        synthesize_speech_wav,
        lambda cfg, work_dir: build_enrichment_providers(cfg).transcription,
        lambda provider, cfg, prepared: check_transcription(provider, prepared),
    ),
    "multimodal_embedding": RoleSpec(
        "multimodal_embedding",
        "rss",
        lambda work_dir: render_shape_image(work_dir / "smoke_shape.png"),
        _build_multimodal,
        lambda provider, cfg, prepared: check_multimodal(
            provider, prepared, dim=cfg.embedding.multimodal.dim
        ),
    ),
}


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RoleResult:
    entry: str
    role: str
    status: str
    """`passed` | `failed`."""
    result: str | None = None
    error: str | None = None
    model_id: str | None = None
    load_seconds: float | None = None
    inference_seconds: float | None = None
    peak_memory_gb: float | None = None
    peak_memory_source: str | None = None
    max_rss_gb: float | None = None
    timing_method: str | None = None
    free_memory_before_gb: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "role": self.role,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "model_id": self.model_id,
            "load_seconds": self.load_seconds,
            "inference_seconds": self.inference_seconds,
            "peak_memory_gb": self.peak_memory_gb,
            "peak_memory_source": self.peak_memory_source,
            "max_rss_gb": self.max_rss_gb,
            "timing_method": self.timing_method,
            "free_memory_before_gb": self.free_memory_before_gb,
            "details": dict(self.details),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RoleResult:
        details = data.get("details")
        return cls(
            entry=str(data["entry"]),
            role=str(data["role"]),
            status=str(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            model_id=data.get("model_id"),
            load_seconds=data.get("load_seconds"),
            inference_seconds=data.get("inference_seconds"),
            peak_memory_gb=data.get("peak_memory_gb"),
            peak_memory_source=data.get("peak_memory_source"),
            max_rss_gb=data.get("max_rss_gb"),
            timing_method=data.get("timing_method"),
            free_memory_before_gb=data.get("free_memory_before_gb"),
            details=dict(details) if isinstance(details, Mapping) else {},
        )


@dataclass(slots=True)
class EntryResult:
    name: str
    status: str
    """`passed` | `failed` | `skipped`."""
    roles: list[RoleResult] = field(default_factory=list)
    artifact_sha256: str | None = None
    download_seconds: float | None = None
    download_bytes: int | None = None
    snapshot_path: str | None = None
    error: str | None = None
    """A failure before any role ran (download, checksum)."""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "artifact_sha256": self.artifact_sha256,
            "download_seconds": self.download_seconds,
            "download_bytes": self.download_bytes,
            "snapshot_path": self.snapshot_path,
            "error": self.error,
            "roles": [role.to_json() for role in self.roles],
        }


@dataclass(frozen=True, slots=True)
class SmokeDeps:
    """Everything `run_role` reaches outside itself, injectable for tests."""

    role_specs: Mapping[str, RoleSpec]
    probe_factory: Callable[[str], PeakMemoryProbe] = default_probe_factory
    host_class: str = ""
    free_memory_gb: Callable[[], float | None] = vm_stat_free_gb
    clock: Callable[[], float] = time.perf_counter
    today: str = ""


def default_deps() -> SmokeDeps:
    return SmokeDeps(
        role_specs=ROLE_SPECS,
        host_class=detect_host_class(),
        today=datetime.now(UTC).astimezone().date().isoformat(),
    )


def run_role(
    entry: ManifestEntry, role: str, cfg: Config, work_dir: Path, deps: SmokeDeps
) -> RoleResult:
    """Steps 3 and 4 for one (entry, role): build the real provider,
    run the check, time it, read the peak. Never raises — every failure
    is a `failed` result with a scrubbed error."""
    result = RoleResult(entry=entry.name, role=role, status="failed")
    spec = deps.role_specs.get(role)
    if spec is None:
        result.error = f"no smoke check is defined for role '{role}'"
        return result
    result.free_memory_before_gb = deps.free_memory_gb()
    probe: PeakMemoryProbe | None = None
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        prepared = spec.prepare(work_dir)
        probe = deps.probe_factory(spec.memory)
        result.peak_memory_source = probe.source
        probe.reset()
        provider = spec.build(cfg, work_dir)
        model_id = getattr(provider, "model_id", None)
        result.model_id = str(model_id) if model_id is not None else None
        load = getattr(provider, "load", None)
        if callable(load):
            started = deps.clock()
            load()
            result.load_seconds = deps.clock() - started
            active = getattr(probe, "active_gb", None)
            if callable(active):
                result.details["active_memory_after_load_gb"] = float(active())
            started = deps.clock()
            line = spec.check(provider, cfg, prepared)
            result.inference_seconds = deps.clock() - started
            result.timing_method = "explicit load(); check timed separately"
        else:
            started = deps.clock()
            spec.check(provider, cfg, prepared)
            first = deps.clock() - started
            started = deps.clock()
            line = spec.check(provider, cfg, prepared)
            second = deps.clock() - started
            result.load_seconds = max(first - second, 0.0)
            result.inference_seconds = second
            result.timing_method = (
                "provider loads lazily on first use: load = first call minus second call"
            )
            result.details["first_call_seconds"] = first
        result.result = scrub_private(line)
        result.status = "passed"
    except SmokeCheckFailed as exc:
        result.error = scrub_private(str(exc))
    except Exception as exc:
        result.error = scrub_private(f"{type(exc).__name__}: {exc}")
    finally:
        if probe is not None:
            try:
                result.peak_memory_gb = probe.read_gb()
            except Exception as exc:
                result.details["peak_memory_error"] = scrub_private(f"{type(exc).__name__}: {exc}")
        result.max_rss_gb = max_rss_gb()
    return result


RoleRunner = Callable[[ManifestEntry, str], RoleResult]
"""Runs one (entry, role) and returns its result — in this process
(`in_process_runner`) or in a child (`spawn_role`)."""


def run_entry(
    entry: ManifestEntry,
    roles: Sequence[str],
    *,
    downloader: Downloader,
    skip_download: bool,
    runner: RoleRunner,
    clock: Callable[[], float] = time.perf_counter,
    out: TextIO | None = None,
    cache_dir: Path | None = None,
) -> EntryResult:
    """Steps 1 and 2 (download + checksum) for a hosted entry, then every
    requested role through `runner`. A download or checksum failure ends
    the entry: nothing that did not download can be smoke-tested."""
    stream = out or sys.stdout
    result = EntryResult(name=entry.name, status="failed")
    if entry.hosted:
        try:
            started = clock()
            snapshot = fetch_snapshot(
                entry, downloader, skip_download=skip_download, cache_dir=cache_dir
            )
            result.download_seconds = clock() - started
            digest = artifact_digest(snapshot)
            result.artifact_sha256 = digest.sha256
            result.download_bytes = digest.total_bytes
            result.snapshot_path = scrub_private(str(snapshot))
            print(
                f"  {entry.name}: snapshot {result.snapshot_path} — {len(digest.files)} files, "
                f"{digest.total_bytes / GIB:.2f} GiB, {result.download_seconds:.1f}s; "
                f"artifact_sha256 {digest.sha256}",
                file=stream,
            )
        except Exception as exc:
            result.error = scrub_private(f"{type(exc).__name__}: {exc}")
            print(f"  {entry.name}: FAILED before any model ran — {result.error}", file=stream)
            return result
    elif entry.status != STATUS_SYSTEM:
        result.status = "skipped"
        result.error = "status: unresolved — no repo/revision to run"
        print(f"  {entry.name}: skipped ({result.error})", file=stream)
        return result
    if not roles:
        result.status = "skipped"
        result.error = "no roles selected"
        return result
    for role in roles:
        role_result = runner(entry, role)
        result.roles.append(role_result)
        _print_role_result(role_result, stream)
    result.status = "passed" if all(r.status == "passed" for r in result.roles) else "failed"
    return result


def _print_role_result(role_result: RoleResult, stream: TextIO) -> None:
    label = role_result.status.upper()
    timing = ""
    if role_result.load_seconds is not None and role_result.inference_seconds is not None:
        timing = f" load {role_result.load_seconds:.1f}s, inference {role_result.inference_seconds:.2f}s;"
    memory = ""
    if role_result.peak_memory_gb is not None:
        memory = f" peak {role_result.peak_memory_gb:.2f} GiB ({role_result.peak_memory_source});"
    if role_result.max_rss_gb is not None:
        memory += f" max RSS {role_result.max_rss_gb:.2f} GiB;"
    text = role_result.result if role_result.status == "passed" else role_result.error
    print(
        f"  {role_result.entry}/{role_result.role}: {label} —{timing}{memory} {text}",
        file=stream,
    )


# --------------------------------------------------------------------------
# child processes
# --------------------------------------------------------------------------


def in_process_runner(lock: ManifestLock, work_dir: Path, deps: SmokeDeps) -> RoleRunner:
    cfg = config_from_manifest(lock, data_root=work_dir)

    def _run(entry: ManifestEntry, role: str) -> RoleResult:
        return run_role(entry, role, cfg, work_dir / entry.name, deps)

    return _run


def run_child(
    lock_path: Path, entry_name: str, role: str, work_dir: Path, result_json: Path, deps: SmokeDeps
) -> int:
    """`--child`: one (entry, role) in this process, result written as JSON
    for the parent. Exit 0 when the check passed, 1 when it failed, 2
    when the lock or the entry could not be resolved."""
    try:
        lock = load_manifest(lock_path)
        entry = next((e for e in lock.entries if e.name == entry_name), None)
        if entry is None:
            raise SmokeError(f"no entry '{entry_name}' in {lock_path}")
        cfg = config_from_manifest(lock, data_root=work_dir)
    except ImsgError as exc:
        print(f"smoke child: {exc}", file=sys.stderr)
        return 2
    result = run_role(entry, role, cfg, work_dir / entry.name, deps)
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
    return 0 if result.status == "passed" else 1


def spawn_role(
    lock_path: Path,
    entry: ManifestEntry,
    role: str,
    work_dir: Path,
    *,
    python: str = sys.executable,
    out: TextIO | None = None,
    tail_lines: int = 40,
) -> RoleResult:
    """Run one (entry, role) in a fresh interpreter (`--child`), forwarding
    its output live, and read back its JSON result. A child that dies
    without writing one — a crash, an OOM kill — is a `failed` result
    carrying its exit code and the tail of its output."""
    stream = out or sys.stdout
    work_dir.mkdir(parents=True, exist_ok=True)
    result_json = work_dir / f"{entry.name}.{role}{RESULT_JSON_SUFFIX}"
    if result_json.exists():
        result_json.unlink()
    command = [
        python,
        "-m",
        CHILD_MODULE,
        "--child",
        "--lock",
        str(lock_path),
        "--only",
        entry.name,
        "--role",
        role,
        "--work-dir",
        str(work_dir),
        "--result-json",
        str(result_json),
    ]
    tail: deque[str] = deque(maxlen=tail_lines)
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n")
            tail.append(stripped)
            print(f"    | {stripped}", file=stream)
        returncode = proc.wait()
    if result_json.is_file():
        try:
            data = json.loads(result_json.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                return RoleResult.from_json(data)
        except (OSError, ValueError):
            pass
    return RoleResult(
        entry=entry.name,
        role=role,
        status="failed",
        error=scrub_private(
            f"child process exited with code {returncode} before writing a result; "
            f"last output: {' | '.join(tail)}"
        ),
    )


def subprocess_runner(
    lock_path: Path, work_dir: Path, *, python: str = sys.executable, out: TextIO | None = None
) -> RoleRunner:
    def _run(entry: ManifestEntry, role: str) -> RoleResult:
        return spawn_role(lock_path, entry, role, work_dir, python=python, out=out)

    return _run


# --------------------------------------------------------------------------
# --write: the smoke_test record and the text-level lock edit
# --------------------------------------------------------------------------


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def _role_record(role: RoleResult) -> dict[str, Any]:
    record: dict[str, Any] = {"status": role.status}
    if role.load_seconds is not None:
        record["load_seconds"] = _round(role.load_seconds)
    if role.inference_seconds is not None:
        record["inference_seconds"] = _round(role.inference_seconds)
    if role.peak_memory_gb is not None:
        record["peak_memory_gb"] = _round(role.peak_memory_gb)
    if role.peak_memory_source is not None:
        record["peak_memory_source"] = role.peak_memory_source
    if role.status == "passed":
        record["result"] = role.result or ""
    else:
        record["result"] = f"failed: {_shorten(role.error or 'unknown error', 200)}"
        record["error"] = _shorten(role.error or "unknown error", 800)
    return record


def smoke_record(
    entry: ManifestEntry,
    result: EntryResult,
    *,
    date: str,
    host_class: str,
    macos: str | None,
) -> dict[str, Any]:
    """The `smoke_test` mapping for the lock: `status`, `date`,
    `host_class`, `load_seconds`, `inference_seconds`, `peak_memory_gb`
    (GiB), `peak_memory_source`, one-line `result`; `error` when failed;
    `macos` for the system entry (its model ships with the OS, so the OS
    version is the pin); `roles` — one such record per role — when the
    entry serves more than one role, in which case the top-level numbers
    are the per-role maxima and `status` is `passed` only if every role
    passed."""
    record: dict[str, Any] = {
        "status": "passed" if result.status == "passed" else "failed",
        "date": date,
        "host_class": host_class,
    }
    if entry.status == STATUS_SYSTEM and macos:
        record["macos"] = macos
    if not result.roles:
        record["result"] = f"failed: {_shorten(result.error or 'no role ran', 200)}"
        record["error"] = _shorten(result.error or "no role ran", 800)
        return record
    if len(result.roles) == 1:
        record.update(_role_record(result.roles[0]))
        record["status"] = "passed" if result.status == "passed" else "failed"
        return record
    loads = [r.load_seconds for r in result.roles if r.load_seconds is not None]
    inferences = [r.inference_seconds for r in result.roles if r.inference_seconds is not None]
    peaks = [r.peak_memory_gb for r in result.roles if r.peak_memory_gb is not None]
    if loads:
        record["load_seconds"] = _round(max(loads))
    if inferences:
        record["inference_seconds"] = _round(max(inferences))
    if peaks:
        record["peak_memory_gb"] = _round(max(peaks))
    sources = sorted({r.peak_memory_source for r in result.roles if r.peak_memory_source})
    if sources:
        record["peak_memory_source"] = ", ".join(sources)
    summary = "; ".join(
        f"{r.role}: {r.status}"
        + (
            f" — {_shorten(r.result or '', 120)}"
            if r.status == "passed"
            else f" — {_shorten(r.error or '', 120)}"
        )
        for r in result.roles
    )
    record["result"] = _shorten(summary, 400)
    record["roles"] = {r.role: _role_record(r) for r in result.roles}
    return record


@dataclass(frozen=True, slots=True)
class LockUpdate:
    entry: str
    smoke_test: Mapping[str, Any]
    artifact_sha256: str | None = None
    """A new digest, or `None` to leave the existing line untouched (a
    system entry, or a download that failed)."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ENTRY_LINE_RE = re.compile(r"^  ([^\s#][^:]*):\s*$")
_KEY_INDENT = "    "
_CONTINUATION_RE = re.compile(r"^ {5,}\S")


def _entry_spans(lines: Sequence[str]) -> dict[str, tuple[int, int]]:
    try:
        models_at = next(i for i, line in enumerate(lines) if line.rstrip() == "models:")
    except StopIteration as exc:
        raise SmokeError("lock has no top-level `models:` line") from exc
    starts: list[tuple[str, int]] = []
    for index in range(models_at + 1, len(lines)):
        line = lines[index]
        if line and not line.startswith(" ") and not line.startswith("#"):
            break  # a later top-level key ends the models mapping
        match = _ENTRY_LINE_RE.match(line)
        if match:
            starts.append((match.group(1).strip(), index))
    spans: dict[str, tuple[int, int]] = {}
    for position, (name, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(lines)
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1  # trailing blank lines belong to nobody
        spans[name] = (start, end)
    return spans


def _key_span(lines: Sequence[str], start: int, end: int, key: str) -> tuple[int, int]:
    prefix = f"{_KEY_INDENT}{key}:"
    hits = [i for i in range(start, end) if lines[i].startswith(prefix)]
    if len(hits) != 1:
        raise SmokeError(f"expected exactly one `{key}:` line in the entry, found {len(hits)}")
    first = hits[0]
    last = first + 1
    while last < end and (_CONTINUATION_RE.match(lines[last]) or not lines[last].strip()):
        last += 1
    while last > first + 1 and not lines[last - 1].strip():
        last -= 1
    return first, last


def render_smoke_test_lines(record: Mapping[str, Any]) -> list[str]:
    """`smoke_test:` as block-style YAML lines at the entry's indent."""
    dumped = yaml.safe_dump(
        dict(record), sort_keys=False, allow_unicode=True, width=100, default_flow_style=False
    )
    return [
        f"{_KEY_INDENT}smoke_test:",
        *(f"{_KEY_INDENT}  {line}" for line in dumped.splitlines()),
    ]


def apply_lock_updates(text: str, updates: Sequence[LockUpdate]) -> str:
    """The lock text with each update's `smoke_test` block (and
    `artifact_sha256` line, when given) replaced — nothing else changes,
    down to the byte. The result is re-parsed and compared with the
    expected document before it is returned."""
    lines = text.split("\n")
    expected = yaml.safe_load(text)
    if not isinstance(expected, dict) or not isinstance(expected.get("models"), dict):
        raise SmokeError("lock text is not a models manifest")
    for update in updates:
        spans = _entry_spans(lines)
        if update.entry not in spans:
            raise SmokeError(f"no entry '{update.entry}' in the lock")
        start, end = spans[update.entry]
        if update.artifact_sha256 is not None:
            if not _SHA256_RE.fullmatch(update.artifact_sha256):
                raise SmokeError(f"not a hex sha256: {update.artifact_sha256!r}")
            first, last = _key_span(lines, start, end, "artifact_sha256")
            if last != first + 1:
                raise SmokeError("artifact_sha256 must be a single line")
            # Quoted: a digest made only of digits (or digits around one
            # `e`) would otherwise load back as a YAML number.
            lines[first] = f"{_KEY_INDENT}artifact_sha256: '{update.artifact_sha256}'"
            expected["models"][update.entry]["artifact_sha256"] = update.artifact_sha256
        start, end = _entry_spans(lines)[update.entry]
        first, last = _key_span(lines, start, end, "smoke_test")
        lines[first:last] = render_smoke_test_lines(update.smoke_test)
        expected["models"][update.entry]["smoke_test"] = json.loads(
            json.dumps(dict(update.smoke_test))
        )
    new_text = "\n".join(lines)
    reparsed = yaml.safe_load(new_text)
    if reparsed != expected:
        raise SmokeError("the rewritten lock does not parse back to the expected document")
    return new_text


def write_lock_updates(path: Path, updates: Sequence[LockUpdate]) -> None:
    text = path.read_text(encoding="utf-8")
    new_text = apply_lock_updates(text, updates)
    tmp = path.with_name(path.name + ".smoke-tmp")
    tmp.write_text(new_text, encoding="utf-8")
    try:
        load_manifest(tmp)  # the project's own parser must still accept it
    except ImsgError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def ordered_entries(lock: ManifestLock, only: Sequence[str]) -> list[ManifestEntry]:
    by_name = {entry.name: entry for entry in lock.entries}
    unknown = [name for name in only if name not in by_name]
    if unknown:
        raise SmokeError(f"unknown entr(y/ies) {unknown}; the lock has {sorted(by_name)}")
    selected = set(only) if only else set(by_name)
    ordered = [name for name in SMOKE_RUN_ORDER if name in by_name]
    ordered += [entry.name for entry in lock.entries if entry.name not in ordered]
    return [by_name[name] for name in ordered if name in selected]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smoke_test_models",
        description=(
            "Download every pinned model in models/manifest.lock.yaml, checksum it, run it "
            "once through imsg.providers.factory on a synthetic input, and record load time, "
            "inference time and peak memory. Never modifies the lock unless --write is given."
        ),
    )
    parser.add_argument("--lock", type=Path, default=None, help="Path to the lock file.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="ENTRY",
        help="Run only this lock entry (repeatable).",
    )
    parser.add_argument(
        "--role",
        action="append",
        default=[],
        metavar="ROLE",
        help="Run only this role of the selected entries (repeatable).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use the already-cached snapshot (huggingface_hub local_files_only) instead of downloading.",
    )
    parser.add_argument(
        "--write", action="store_true", help="Record smoke_test and artifact_sha256 in the lock."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Where synthetic inputs, child results and the report go (default: a temp dir).",
    )
    parser.add_argument(
        "--report-json", type=Path, default=None, help="Where to write the JSON report."
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run every role in this process instead of one child process per (entry, role).",
    )
    parser.add_argument(
        "--hub-cache",
        type=Path,
        default=None,
        help="Hugging Face cache root for --skip-download (default: huggingface_hub's own).",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def _default_work_dir() -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix="imsg-model-smoke-"))


def main(
    argv: Sequence[str] | None = None,
    *,
    deps: SmokeDeps | None = None,
    downloader: Downloader = hf_snapshot_download,
    out: TextIO | None = None,
) -> int:
    """Exit 0 when every selected entry passed, 1 when any failed, 2 when
    the lock or the arguments are unusable."""
    stream = out or sys.stdout
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    lock_path = args.lock or default_manifest_path()
    active_deps = deps or default_deps()
    work_dir = (args.work_dir or _default_work_dir()).resolve()

    if args.child:
        if len(args.only) != 1 or len(args.role) != 1 or args.result_json is None:
            print(
                "smoke child: --child needs exactly one --only, one --role and --result-json",
                file=sys.stderr,
            )
            return 2
        return run_child(
            lock_path, args.only[0], args.role[0], work_dir, args.result_json, active_deps
        )

    try:
        lock = load_manifest(lock_path)
        entries = ordered_entries(lock, args.only)
    except ImsgError as exc:
        print(f"models smoke: {exc}", file=stream)
        return 2
    role_filter = set(args.role)
    print(
        f"manifest: {lock_path} ({len(entries)} of {len(lock.entries)} entries selected); "
        f"host: {active_deps.host_class}; work dir: {scrub_private(str(work_dir))}",
        file=stream,
    )
    runner = (
        in_process_runner(lock, work_dir, active_deps)
        if args.in_process
        else subprocess_runner(lock_path, work_dir, out=stream)
    )

    results: list[EntryResult] = []
    for entry in entries:
        roles = [role for role in entry.roles if not role_filter or role in role_filter]
        free = active_deps.free_memory_gb()
        free_text = (
            f"{free:.1f} GiB free (vm_stat: free+inactive+speculative)"
            if free is not None
            else "free memory unknown"
        )
        print(
            f"\n== {entry.name} ({', '.join(roles) or 'no roles selected'}) — {free_text}",
            file=stream,
        )
        results.append(
            run_entry(
                entry,
                roles,
                downloader=downloader,
                skip_download=args.skip_download,
                runner=runner,
                clock=active_deps.clock,
                out=stream,
                cache_dir=args.hub_cache,
            )
        )

    total_bytes = sum(r.download_bytes or 0 for r in results)
    report: dict[str, Any] = {
        "date": active_deps.today,
        "host_class": active_deps.host_class,
        "macos": macos_version(),
        "lock": scrub_private(str(lock_path)),
        "entries": [result.to_json() for result in results],
        "total_download_bytes": total_bytes,
    }
    report_path = args.report_json or (work_dir / "smoke_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nsummary:", file=stream)
    for result in results:
        size = (
            f"{result.download_bytes / GIB:.2f} GiB" if result.download_bytes is not None else "-"
        )
        sha = result.artifact_sha256 or "-"
        print(f"  {result.status:8} {result.name:22} {size:>10}  {sha}", file=stream)
    print(
        f"  total downloaded: {total_bytes / GIB:.2f} GiB; report: "
        f"{scrub_private(str(report_path))}",
        file=stream,
    )

    if args.write:
        updates = [
            LockUpdate(
                entry=result.name,
                smoke_test=smoke_record(
                    next(e for e in entries if e.name == result.name),
                    result,
                    date=active_deps.today,
                    host_class=active_deps.host_class,
                    macos=macos_version(),
                ),
                artifact_sha256=result.artifact_sha256,
            )
            for result in results
            if result.status != "skipped"
        ]
        try:
            write_lock_updates(lock_path, updates)
        except ImsgError as exc:
            print(f"models smoke: could not write the lock: {exc}", file=stream)
            return 2
        print(f"models smoke: wrote {len(updates)} entr(y/ies) to {lock_path}", file=stream)
    else:
        print("models smoke: lock NOT modified (pass --write to record the results)", file=stream)

    failed = [r.name for r in results if r.status == "failed"]
    if failed:
        print(f"models smoke: {len(failed)} entr(y/ies) failed: {failed}", file=stream)
        return 1
    print("models smoke: all selected entries passed", file=stream)
    return 0


__all__ = [
    "ARTIFACT_FILE_PATTERNS",
    "BOUNDARY_DESIGNED_SPLIT",
    "CAPTION_KEYWORDS",
    "DOWNLOAD_IGNORE_PATTERNS",
    "EMBED_DOCUMENTS",
    "EMBED_QUERY",
    "MLX_ROLES",
    "OCR_LINES",
    "RERANK_DOCUMENTS",
    "RERANK_QUERY",
    "ROLE_SPECS",
    "SMOKE_RUN_ORDER",
    "TRANSCRIPTION_SENTENCE",
    "ArtifactDigest",
    "ArtifactFile",
    "EntryResult",
    "LockUpdate",
    "MlxPeakMemoryProbe",
    "RoleResult",
    "RoleSpec",
    "RssPeakMemoryProbe",
    "SmokeCheckFailed",
    "SmokeDeps",
    "SmokeError",
    "apply_lock_updates",
    "artifact_digest",
    "cached_snapshot_dir",
    "check_boundaries",
    "check_caption",
    "check_multimodal",
    "check_ocr",
    "check_reranker",
    "check_text_embedding",
    "check_transcription",
    "config_from_manifest",
    "default_deps",
    "default_hub_cache_dir",
    "detect_host_class",
    "fetch_snapshot",
    "in_process_runner",
    "iter_artifact_files",
    "main",
    "ordered_entries",
    "parse_vm_stat",
    "render_shape_image",
    "render_smoke_test_lines",
    "render_text_image",
    "run_entry",
    "run_role",
    "scrub_private",
    "smoke_record",
    "spawn_role",
    "synthesize_speech_wav",
    "synthetic_boundary_window",
    "vm_stat_free_gb",
    "write_lock_updates",
]


if __name__ == "__main__":
    raise SystemExit(main())
