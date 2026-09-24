"""The `config.yaml` schema (SPEC §6) and its validation rules.

This module is the load-bearing enforcement mechanism for three of the
non-negotiables in `CLAUDE.md` / SPEC §1:

- **#1** (never write to the live `chat.db`): exactly one path in the
  entire config is permitted to resolve under `~/Library/Messages` —
  the declared `paths.live_chat_db` read-only source. Every other
  configured path that resolves there is rejected.
- **#2** (all derived state on the encrypted volume): every
  derived/output path in the config must resolve under
  `paths.data_root`, symlinks and `..` included.
- **#6** (zero coupling to any other system; dedicated Postgres
  instance): the DSN must target port 5433, and the cluster-fingerprint
  file must live under `data_root`.

Plus: `mcp.public.scope` has no default (D6) — omitting it is a
validation error, never a silent `full`. Secret-marked fields
(`database.password`, `mcp.public.oauth.client_secret`,
`mcp.public.oauth.owner_subject`) only accept `keychain:`/`env:`
references — see `imsg.config.secrets.SecretRef`.

Unknown keys are errors everywhere (`extra="forbid"`), matching SPEC
§6's "unknown keys are errors".
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import phonenumbers
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from imsg import constants, mlx_runtime, shared_vlm_runtime
from imsg.config.secrets import SecretRef
from imsg.embed.batching import DEFAULT_MAX_BATCH_TOKENS
from imsg.paths import is_contained_in, join_under_root, resolve_path

MESSAGES_DIR = Path("~/Library/Messages").expanduser()


class StrictModel(BaseModel):
    """Base for every config section: unknown keys are hard errors."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


class PathsConfig(StrictModel):
    data_root: Path = Field(default=Path("/Volumes/Data-Encrypted/imsgindex"))
    live_chat_db: Path = Field(default=Path("~/Library/Messages/chat.db"))

    @field_validator("data_root", "live_chat_db", mode="after")
    @classmethod
    def _expand_user(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("data_root", mode="after")
    @classmethod
    def _data_root_must_be_absolute(cls, v: Path) -> Path:
        if not v.is_absolute():
            raise ValueError(f"paths.data_root must be an absolute path, got '{v}'")
        return v


# --------------------------------------------------------------------------
# database
# --------------------------------------------------------------------------

_DSN_RE = re.compile(
    r"^postgres(?:ql)?://"
    r"(?:(?P<user>[^:@/]+)(?::[^@/]*)?@)?"
    r"(?P<host>[^:/@]+)"
    r"(?::(?P<port>\d+))?"
    r"/(?P<dbname>[^?]+)"
    r"(?:\?.*)?$"
)

REQUIRED_DB_PORT = 5433


class DatabaseConfig(StrictModel):
    dsn: str = Field(default="postgresql://imsg@127.0.0.1:5433/imsgindex")
    password: SecretRef
    cluster_fingerprint_file: Path = Field(default=Path("pg17/.imsgindex-cluster"))

    @field_validator("dsn", mode="after")
    @classmethod
    def _dsn_must_target_the_dedicated_instance(cls, v: str) -> str:
        m = _DSN_RE.match(v)
        if not m:
            raise ValueError(
                f"database.dsn is not a valid postgresql:// URI: '{v}'"
            )
        port = m.group("port")
        if port is None or int(port) != REQUIRED_DB_PORT:
            raise ValueError(
                f"database.dsn must target port {REQUIRED_DB_PORT} — the dedicated "
                f"imessage-index Postgres instance (CLAUDE.md non-negotiable #6; "
                f"SPEC §1.6, §5.2) — got port {port!r} in '{v}'"
            )
        return v

    @field_validator("cluster_fingerprint_file", mode="after")
    @classmethod
    def _fingerprint_file_must_be_relative_or_checked_later(cls, v: Path) -> Path:
        # Absolute-vs-data_root containment is checked at the root level
        # (Config), where paths.data_root is in scope.
        return v


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------


class SyncSourceConfig(StrictModel):
    name: str = Field(min_length=1)
    chat_db: Path

    @field_validator("chat_db", mode="after")
    @classmethod
    def _expand_user(cls, v: Path) -> Path:
        return v.expanduser()


MIN_SYNC_INTERVAL_SECONDS = 300


class SyncConfig(StrictModel):
    interval_seconds: int = Field(default=900, ge=MIN_SYNC_INTERVAL_SECONDS)
    sources: list[SyncSourceConfig] = Field(default_factory=list)

    @field_validator("sources", mode="after")
    @classmethod
    def _at_least_one_source(
        cls, v: list[SyncSourceConfig]
    ) -> list[SyncSourceConfig]:
        if not v:
            raise ValueError("sync.sources must list at least one source")
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            raise ValueError(f"sync.sources names must be unique, got {names}")
        return v


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


class IdentityConfig(StrictModel):
    default_region: str = "US"
    contacts_import: bool = True

    @field_validator("default_region", mode="after")
    @classmethod
    def _must_be_a_known_region(cls, v: str) -> str:
        upper = v.upper()
        if upper not in phonenumbers.SUPPORTED_REGIONS:
            raise ValueError(
                f"identity.default_region '{v}' is not a region phonenumbers "
                f"recognizes for E.164 normalization"
            )
        return upper


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------


class PolicyConfig(StrictModel):
    index_unsent: bool = False
    index_edit_history: bool = False


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------


class SegmentationConfig(StrictModel):
    session_gap_hours: float = Field(default=3.0, gt=0)
    topical_min_messages: int = Field(default=10, ge=1)
    max_messages: int = Field(default=50, ge=1)
    max_tokens: int = Field(default=2000, ge=1)
    # Hugging Face repo id of the boundary-detection LLM (SPEC §4.1) and
    # the immutable commit it is pinned to. Both default to the pins in
    # models/manifest.lock.yaml (mirrored in imsg.constants). NOTE:
    # `boundary_model` is folded into `seg_config_hash` (D4), so changing
    # it forces re-segmentation of every chat.
    boundary_model: str = Field(default=constants.BOUNDARY_MODEL_REPO, min_length=1)
    boundary_revision: str = Field(default=constants.BOUNDARY_MODEL_REVISION, min_length=1)
    boundary_prompt: Path = Field(default=Path("prompts/segment_boundaries.txt"))

    @model_validator(mode="after")
    def _min_below_max(self) -> SegmentationConfig:
        if self.topical_min_messages > self.max_messages:
            raise ValueError(
                "segmentation.topical_min_messages must be <= segmentation.max_messages"
            )
        return self


# --------------------------------------------------------------------------
# enrichment
# --------------------------------------------------------------------------


class EnrichmentConcurrency(StrictModel):
    ocr: int = Field(default=4, ge=1)
    caption: int = Field(default=1, ge=1)
    transcribe: int = Field(default=1, ge=1)
    pdf: int = Field(default=4, ge=1)


class EnrichmentLimits(StrictModel):
    max_file_bytes: int = Field(default=1_073_741_824, gt=0)
    max_pdf_pages: int = Field(default=1000, gt=0)
    max_media_seconds: int = Field(default=14400, gt=0)
    task_timeout_seconds: int = Field(default=1800, gt=0)
    temp_bytes_per_task: int = Field(default=10_737_418_240, gt=0)


_WINDOW_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")


class EnrichmentConfig(StrictModel):
    egress: Literal["local_only"] = "local_only"
    window: str = "01:00-07:00"
    concurrency: EnrichmentConcurrency = Field(default_factory=EnrichmentConcurrency)
    max_attempts: int = Field(default=5, ge=1)
    video_max_frames: int = Field(default=20, ge=1)
    pdf_scanned_threshold_chars_per_page: int = Field(default=50, ge=0)
    limits: EnrichmentLimits = Field(default_factory=EnrichmentLimits)

    # --- model providers (SPEC §4.1; built by imsg.providers.factory) ---
    # Repo ids + immutable revisions default to models/manifest.lock.yaml's
    # pins (mirrored in imsg.constants).
    transcription_model: str = Field(default=constants.TRANSCRIPTION_MODEL_REPO, min_length=1)
    transcription_revision: str = Field(
        default=constants.TRANSCRIPTION_MODEL_REVISION, min_length=1
    )
    transcription_language: str | None = None
    """BCP-47-ish language hint for Whisper (e.g. 'en'); `null` lets the
    model auto-detect per file."""
    caption_model: str = Field(default=constants.CAPTION_MODEL_REPO, min_length=1)
    caption_revision: str = Field(default=constants.CAPTION_MODEL_REVISION, min_length=1)
    caption_max_image_side: int | None = Field(default=1920, ge=256)
    """Longest side, in pixels, of the image the captioning model is shown;
    a larger image is scaled down (aspect kept, EXIF orientation applied)
    into a temporary copy — the attachment itself is never rewritten.
    `null` shows the model every image at full resolution. A caption's
    cost follows the pixels: measured 2026-09-23 on an M2 Ultra, a
    4032x3024 photo took 37.6 s, the same photo at 1920x1440 5.7 s, and
    the 1440x1920 image behind the production host's 14.2 s-per-caption
    figure 5.2 s. OCR always reads the full-resolution original."""
    caption_prompt: Path = Field(default=Path("prompts/caption.txt"))
    """Fixed captioning prompt (SPEC §4.1) — relative to `paths.data_root`,
    same convention as `segmentation.boundary_prompt`; must resolve
    under data_root (checked at the root level)."""
    ocr_languages: list[str] | None = None
    """Apple Vision recognition languages (e.g. ['en-US']); `null` lets
    Vision detect the language automatically (macOS 13+)."""
    ocr_minimum_text_height: float | None = Field(default=None, gt=0, le=1)
    """Vision's `minimumTextHeight` as a fraction of image height;
    `null` keeps the framework default."""

    # --- yielding to in-flight queries (D10.3's ratified remedy) ---
    yield_to_queries: bool = True
    """Pause between units of work while an MCP server is answering a
    query (`imsg.db.enrichment_yield_locks`). The nightly window overlaps
    the always-on query server on one GPU, and with a single copy of the
    35B resident — no swap, pressure never critical — query p95 still
    trebled while enrichment ran; that residue is GPU contention. The
    asymmetry is deliberate: enrichment is restartable batch work with a
    queue, the MCP server is the externally visible surface with a 2.0 s
    budget. Set false to keep enrichment throughput at the query side's
    expense."""

    yield_poll_interval_seconds: float = Field(
        default=constants.ENRICHMENT_YIELD_POLL_INTERVAL_SECONDS, gt=0
    )
    """How often a paused worker re-checks. Only reached while a query is
    actually in flight — when none is, the check is one round trip and no
    sleep — so this sets how promptly the worker resumes, not what
    yielding costs."""

    yield_max_pause_seconds: float = Field(default=constants.ENRICHMENT_YIELD_MAX_PAUSE_SECONDS, ge=0)
    """How long the worker waits for one unit of work before proceeding
    regardless. This is NOT the crash backstop — a killed MCP server's
    advisory lock dies with its database session, so nothing can wedge
    enrichment paused — it is the "someone is searching continuously and
    the queue still has to drain overnight" backstop."""

    @field_validator("window", mode="after")
    @classmethod
    def _window_must_be_hh_mm_range(cls, v: str) -> str:
        if not _WINDOW_RE.match(v):
            raise ValueError(
                f"enrichment.window must look like 'HH:MM-HH:MM', got '{v}'"
            )
        return v

    @field_validator("ocr_languages", mode="after")
    @classmethod
    def _ocr_languages_non_empty_tags(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not v:
            raise ValueError(
                "enrichment.ocr_languages must be null (automatic detection) or a "
                "non-empty list of language tags such as ['en-US']"
            )
        cleaned = [tag.strip() for tag in v]
        if any(not tag for tag in cleaned):
            raise ValueError("enrichment.ocr_languages entries must be non-empty language tags")
        return cleaned

    @field_validator("transcription_language", mode="after")
    @classmethod
    def _transcription_language_non_empty(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError(
                "enrichment.transcription_language must be null (auto-detect) or a "
                "non-empty language code such as 'en'"
            )
        return v.strip() if v is not None else None


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------


class MultimodalEmbeddingConfig(StrictModel):
    enabled: bool = True
    provider: Literal["local"] = "local"
    scope: Literal["full"] = "full"
    model: str = Field(default=constants.MULTIMODAL_EMBEDDING_MODEL_REPO, min_length=1)
    revision: str = Field(min_length=1)
    dim: int = constants.MULTIMODAL_EMBEDDING_DIM
    batch_size: int = Field(default=16, ge=1)

    @field_validator("dim", mode="after")
    @classmethod
    def _dim_must_match_migration_0002(cls, v: int) -> int:
        if v != constants.MULTIMODAL_EMBEDDING_DIM:
            raise ValueError(
                f"embedding.multimodal.dim must equal "
                f"{constants.MULTIMODAL_EMBEDDING_DIM} to match migration "
                f"0002's attachment_mm_embedding CHECK constraint (SPEC §7.4) — "
                f"a different dim requires a new migration, got {v}"
            )
        return v


class EmbeddingConfig(StrictModel):
    model: str = Field(default=constants.TEXT_EMBEDDING_MODEL_REPO, min_length=1)
    revision: str = Field(min_length=1)
    quantization: str = "mxfp8"
    """Informational: the quantization of the artifact `model`@`revision`
    names (the MLX conversion fixes it; nothing re-quantizes at load
    time). Kept in config so the run log and models/manifest.lock.yaml
    describe the same weights."""
    dim: int = constants.PRIMARY_EMBEDDING_DIM
    batch_size: int = Field(default=32, ge=1)
    """Most rows in one text batch (one forward pass, one transaction)."""
    max_batch_tokens: int = Field(default=DEFAULT_MAX_BATCH_TOKENS, ge=1)
    """Most *padded* tokens in one text batch — `rows x longest row`,
    the cost of a right-padded forward pass — in `imsg.tokens.
    estimate_tokens` units. Batches are length-sorted so this bounds
    activation memory without wasting compute on padding
    (`imsg.embed.batching`)."""
    query_instruction: str = Field(min_length=1)
    multimodal: MultimodalEmbeddingConfig

    @field_validator("dim", mode="after")
    @classmethod
    def _dim_within_index_cap_and_matches_migration(cls, v: int) -> int:
        if v > constants.HALFVEC_INDEX_MAX_DIM:
            raise ValueError(
                f"embedding.dim ({v}) exceeds pgvector's HNSW/IVFFlat index cap "
                f"for halfvec ({constants.HALFVEC_INDEX_MAX_DIM}) — the column "
                f"would be legal DDL but its index could never be created "
                f"(this is the exact bug this spec revision fixed, SPEC §4.1/§7.2)"
            )
        if v != constants.PRIMARY_EMBEDDING_DIM:
            raise ValueError(
                f"embedding.dim must equal {constants.PRIMARY_EMBEDDING_DIM} to "
                f"match migration 0001's segment_embedding/attachment_chunk_embedding "
                f"CHECK constraints — a different dim requires a new migration "
                f"(SPEC §6), got {v}"
            )
        return v


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


HNSW_EF_SEARCH_MAX = 1000
"""pgvector's own upper bound for `hnsw.ef_search` (1..1000): setting
1001 is rejected by the server (checked against pgvector 0.8.6 on the
live instance, 2026-09-17)."""

RERANKER_MODEL_FORMS = (
    "a Hugging Face repo id ('owner/name') or a directory relative to paths.data_root "
    "holding a local MLX conversion (e.g. 'models/<conversion>', the lock's output_dir)"
)


class RetrievalConfig(StrictModel):
    k_fts: int = Field(default=100, ge=1)
    k_vector: int = Field(default=100, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top: int = Field(default=20, ge=1)
    """How many of the fused candidates the reranker scores (SPEC §9.4
    step 7) — never fewer than a request's `limit`; candidates past the
    pool are dropped. The reranker's cost grows with the tokens it reads,
    so this and `rerank_doc_max_tokens` set search latency
    (`scripts/bench_retrieval_latency.py`; the defaults and their measured
    trade-off are in the README)."""
    rerank_doc_max_tokens: int | None = Field(default=256, ge=1)
    """Most tokens (the reranker's own tokenizer) of each candidate's text
    the reranker reads; `null` means no cap beyond the model's 8,192-token
    row. Only the document is cut — the instruction, the query and the
    chat suffix the yes/no score is read after are always kept. The count
    includes the rendered segment's header (`Chat:` / `Time:` lines, ~40
    tokens for two participants): at 32 the reranker sees only part of the
    header, and on the benchmark's proxy it then ordered results worse than
    the fused order it replaces."""
    reranker_model: str = Field(default=constants.RERANKER_MODEL, min_length=1)
    """Either form named by `RERANKER_MODEL_FORMS`. `imsg.providers.factory`
    reads the value as a local directory when `<paths.data_root>/<value>`
    exists (then `revision=None` is passed to the provider) and as a repo
    id otherwise. Containment under `paths.data_root` is enforced: no
    absolute path, no `~`, no `..`, and — at the root level, symlinks
    resolved — nothing that escapes the data root."""
    reranker_revision: str = Field(min_length=1)
    """The pinned commit sha: the repo's own for a Hub id; for a local
    conversion, the UPSTREAM repo's commit the conversion was made from
    (the lock's `upstream_revision`). The provider records
    `<reranker_model>@<reranker_revision>` as its `model_id` either way."""
    default_limit: int = Field(default=10, ge=1)
    hnsw_ef_search: int = Field(default=1000, ge=1, le=HNSW_EF_SEARCH_MAX)
    """`hnsw.ef_search` for every vector channel — the size of the HNSW
    search's dynamic candidate list ("a higher value provides better recall
    at the cost of speed", pgvector 0.8.6 README §Query Options), applied
    with `SET LOCAL` inside each channel's own transaction
    (`imsg.retrieval.vector_search`). pgvector's own range is 1..1000.

    The default is the maximum, chosen by measurement on the live index
    (20 fictional queries, warm pages, 2026-09-17): recall@100 against
    exact search rises 0.944 -> 0.969 -> 0.989 -> 0.997 -> 1.000 (text) and
    0.792 -> 0.887 -> 0.945 -> 0.985 -> 0.998 (multimodal) at 40 / 100 /
    200 / 400 / 1000, and the worst single query rises from 0.79 and 0.38
    to 1.00 and 0.98. It costs about 52 ms of the two channels' p95
    together (20.7 ms at 40, 72.8 ms at 1000) — 2.6 % of a 2 s query
    budget the reranker otherwise dominates. Filtered searches, where the
    iterative scan does the work, were not slower at 1000 than at 40."""

    @field_validator("reranker_model", mode="after")
    @classmethod
    def _reranker_model_is_a_repo_id_or_a_data_root_relative_dir(cls, v: str) -> str:
        value = v.strip()
        if not value:
            raise ValueError(f"retrieval.reranker_model must be {RERANKER_MODEL_FORMS}, got ''")
        path = Path(value)
        if path.is_absolute() or value.startswith("~"):
            raise ValueError(
                f"retrieval.reranker_model must be {RERANKER_MODEL_FORMS} — not an absolute "
                f"or home-relative path, got '{v}'"
            )
        if ".." in path.parts:
            raise ValueError(
                f"retrieval.reranker_model must be {RERANKER_MODEL_FORMS} — a directory may "
                f"not contain '..' segments (it must stay under paths.data_root), got '{v}'"
            )
        return value


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


class ModelsConfig(StrictModel):
    """Which provider backend every model-backed stage constructs
    (`imsg.providers.factory`).

    `real` — the default — loads the pinned MLX / Apple Vision / PE-Core
    models named by the `embedding.*`, `retrieval.reranker_*`,
    `segmentation.boundary_*` and `enrichment.*` fields (runtime
    packages: the `models` extra in pyproject.toml). `fake` substitutes
    the deterministic test stand-ins and is explicit opt-in only: every
    command that builds providers prints `models: backend=fake` so such
    a run can never be mistaken for a real one.

    The three memory fields exist because the nightly enrichment window
    overlaps the always-on query server, and the overlap was measured not
    to fit (docs D10.3): 80.4 GiB of demand on a 64 GB host, critical
    memory pressure, search p95 at 8.73 s against a 2.0 s budget. Each
    field is the configurable form of one fix, defaulted to the value the
    measurements support.
    """

    backend: Literal["real", "fake"] = "real"

    share_boundary_and_caption_weights: bool = True
    """Load one copy of the weights when `segmentation.boundary_model`
    and `enrichment.caption_model` name the same repo and revision
    (`imsg.shared_vlm_runtime`). The pins do name the same checkpoint by
    default, and loading it twice — once through `mlx_lm`, once through
    `mlx_vlm` — was measured at 18.99 -> 37.15 GiB of MLX active memory
    in one process. Boundary detection then runs text-only through the
    vision-language model, which is the same computation on the same
    weights (bit-identical logits, checked 2026-09-17). Set false to go
    back to two independent loaders."""

    query_cache_limit_bytes: int = Field(default=mlx_runtime.DEFAULT_CACHE_LIMIT_BYTES, ge=0)
    """MLX buffer-cache bound for the query-side providers — the text
    embedder and the reranker, so the `mcp public`/`mcp local` process
    (SPEC §10.4). MLX pools freed GPU buffers rather than returning them
    to the OS and its default limit is its memory limit (60.8 GiB on the
    production host), so this is what keeps a months-long server's
    footprint flat. The 8 GiB default held exactly across 755 searches
    and three runs; the process is 18.7 GiB right after load and ~27.5
    GiB in steady state, and the difference is this cache filling to its
    bound."""

    enrichment_cache_limit_bytes: int = Field(
        default=shared_vlm_runtime.DEFAULT_ENRICHMENT_CACHE_LIMIT_BYTES, ge=0
    )
    """MLX buffer-cache bound for the enrichment-side providers — the
    captioner, boundary detection and (via the process-wide effect of
    the same call) `mlx_whisper` transcription. These called nothing
    before, so their limit sat at the runtime default and one enrichment
    process was observed holding 37.25 GiB of pooled freed buffers."""


# --------------------------------------------------------------------------
# memory: admission before a load, stopping between units, emergency release
# --------------------------------------------------------------------------

_GIB = 2**30


def _gib(value: float) -> int:
    return int(value * _GIB)


class ModelFootprints(StrictModel):
    """What each role's model set is expected to occupy once loaded, in
    bytes: the figure `imsg.memory_admission` compares with the host's
    available memory before letting a load start. Each default is a
    measured process footprint on the 64 GiB production host (M4 Pro)
    where one exists, and otherwise is built from measured parts; the
    source is named per field. Raise a figure if a role is seen to use
    more; `imsg status` shows every model process's live footprint."""

    public_server_bytes: int = Field(default=_gib(17.2), gt=0)
    """17.2 GiB: the query process (text embedder, PE-Core text tower,
    0.6B reranker, MLX cache) at its highest sampled footprint over a
    19-minute run on the production host, 2026-09-17 (CHANGELOG
    2026-09-17, "query process, footprint 16.1 GiB (17.2 peak-sampled)")."""

    local_server_bytes: int = Field(default=_gib(17.2), gt=0)
    """17.2 GiB: the same model set as the public server (same source)."""

    embed_bytes: int = Field(default=_gib(24.9), gt=0)
    """24.9 GiB: `imsg embed` (text embedder plus the PE-Core image tower)
    in the 2026-09-24 kernel panic report from the production host
    (CHANGELOG 2026-09-24, "Model-heavy commands run one at a time")."""

    enrich_bytes: int = Field(default=_gib(24.8), gt=0)
    """24.8 GiB: built from measured parts. The enrichment process
    measured 28.2 GiB on the production host (CHANGELOG 2026-09-17) with
    PE-Core's image tower loaded (7.06 GiB, same run), which the real
    worker never loads (`imsg.enrich.pipeline.EnrichmentProviders` holds
    OCR, captioning and transcription only), and without Whisper, which
    it does load for audio (3.66 GiB peak, models/manifest.lock.yaml):
    28.2 - 7.06 + 3.66 = 24.8."""

    segment_bytes: int = Field(default=_gib(21.0), gt=0)
    """21.0 GiB: built from measured parts, no whole-process measurement
    exists. The boundary model through mlx-lm peaked at 19.07 GiB of MLX
    memory (models/manifest.lock.yaml smoke test), and a model process
    carries about 1.94 GiB outside its model runtimes (the enrichment
    process above: 28.2 - 19.00 MLX active - 0.20 MLX cache - 7.06
    torch)."""


class MlxMemoryLimits(StrictModel):
    """`mx.set_memory_limit` per role, in bytes; 0 leaves MLX's own
    default (1.5x the GPU's recommended working set, capped at 95% of
    RAM). The limit is a guideline, not a ceiling: in MLX 0.32.2 it sets
    the point past which a new allocation first releases cached buffers
    (`gc_limit`, `mlx/backend/metal/allocator.cpp`) and past which graph
    evaluation waits for work in flight before queueing more
    (`mlx/transforms.cpp`); no allocation is ever refused because of it
    and no computation changes, so model output is identical. Every
    default is the role's measured MLX peak times 1.25, rounded up to a
    whole GiB, so evaluation never waits in normal running and the only
    effect is on the buffer cache. On the query side that means about
    4 GiB of cache instead of 8, measured not to move query latency (8
    versus 2 GiB of cache: stage p50s 0.644 versus 0.616 s, within
    noise; CHANGELOG 2026-09-17)."""

    public_server_bytes: int = Field(default=12 * _GIB, ge=0)
    """12 GiB: MLX peak 9.56 GiB for the query models (development host,
    CHANGELOG 2026-09-17; 8.70 GiB on the production host)."""

    local_server_bytes: int = Field(default=12 * _GIB, ge=0)
    """12 GiB: the same models as the public server."""

    embed_bytes: int = Field(default=14 * _GIB, ge=0)
    """14 GiB: MLX peak 9-11 GiB for 4k-16k padded tokens including the
    weights (`imsg.mlx_runtime.DEFAULT_CACHE_LIMIT_BYTES`; 8.6 GiB at the
    default 2k budget, `imsg.embed.batching`)."""

    enrich_bytes: int = Field(default=31 * _GIB, ge=0)
    """31 GiB: MLX peak 20.88 GiB for the captioner (production host,
    CHANGELOG 2026-09-17) plus Whisper's 3.66 GiB, both resident when a
    run mixes images and audio."""

    segment_bytes: int = Field(default=24 * _GIB, ge=0)
    """24 GiB: the boundary model's MLX peak of 19.07 GiB
    (models/manifest.lock.yaml)."""


class MemoryConfig(StrictModel):
    """How every model-loading process stays inside the host's memory
    (`imsg.memory_admission`, `imsg.background_gate`,
    `imsg.retrieval.idle_unload`). There is deliberately no switch that
    turns admission off: a load the host cannot measure is refused."""

    reserve_bytes: int = Field(default=8 * _GIB, ge=0)
    """8 GiB left for the rest of the host after a load: a load is
    admitted only if available memory covers its footprint plus this.
    Reasoned from two measured configurations on the 64 GiB production
    host, 2026-09-17: 44.3 GiB of model processes kept pressure normal
    (CHANGELOG 2026-09-17) and 50.8 GiB reached warn (the D10
    measurements). With macOS, the GUI session and Postgres planned at
    about 14 GB (the shared-host budget's estimates), the normal case
    left about 6 GiB and the warn case none. 8 GiB keeps a load clear of
    both."""

    admission_max_pressure: Literal["normal", "warn"] = "normal"
    """A load is refused while the kernel reports worse than this. There
    is no setting that admits a load at critical."""

    admission_wait_seconds: float = Field(default=600.0, ge=0)
    """How long a background command waits, re-checking, for memory
    before it exits `deferred: memory`. 600 s matches
    `mcp.local.idle_unload_seconds`'s default: the most common large and
    temporary holder is an idle local MCP server, which gives its memory
    back within that time."""

    admission_poll_seconds: float = Field(default=30.0, gt=0)
    """How often that wait re-checks, and logs what it saw."""

    admission_retry_seconds: float = Field(default=15.0, gt=0)
    """An MCP server whose load was refused tries again no sooner than
    this: the local server on its next retrieval call, the public server
    by itself. Each try costs one `vm_stat` run."""

    background_stop_at: Literal["warn", "critical"] = "warn"
    """A background worker finishes its current unit and stops when the
    kernel reports this pressure or worse."""

    warn_confirm_seconds: float = Field(default=10.0, ge=0)
    """A warn reading stops a worker only if it is still warn (or worse)
    this long after: in the 2026-09-17 overlap run the host read warn in
    2 of 133 samples, each on its own. Critical stops a worker at once."""

    pressure_check_seconds: float = Field(default=5.0, gt=0)
    """How often an MCP server's watchdog reads the pressure level (one
    sysctl). The 2026-09-17 failure paged in 218 GiB over 11 minutes, so
    a few seconds is early enough to matter."""

    local_server_release_at: Literal["warn", "critical"] = "critical"
    """A local MCP server with its models loaded unloads them at once, not
    waiting for its idle timer, when the kernel reports this pressure.
    It waits for a call in flight to finish first."""

    public_server_release_at: Literal["critical", "never"] = "critical"
    """The same for the public server; `never` keeps its models loaded
    whatever the pressure."""

    public_rewarm_cooldown_seconds: float = Field(default=300.0, ge=0)
    """After releasing its models for pressure, the public server waits
    this long before trying to load them again (and then only if the
    load is admitted). A reload took 41.8-65.4 s on the production
    host's cold starts (`imsg.retrieval.service.ESTIMATED_WARM_UP_SECONDS`),
    so cycling faster than this would spend most of the time loading."""

    footprints: ModelFootprints = Field(default_factory=ModelFootprints)
    mlx_memory_limits: MlxMemoryLimits = Field(default_factory=MlxMemoryLimits)


# --------------------------------------------------------------------------
# background: the pause switch
# --------------------------------------------------------------------------

DEFAULT_HOST_PAUSE_FILE = Path("~/.config/imessage-index/pause-background")


class BackgroundConfig(StrictModel):
    """The pause switch for heavy background work
    (`imsg.background_pause`)."""

    host_pause_file: Path | None = Field(default=DEFAULT_HOST_PAUSE_FILE, validate_default=True)
    """A file outside the encrypted volume that another project on this
    host can create to pause heavy background work and remove to resume,
    without this project's config or volume. Only ever read, never
    written. `null` turns it off; `imsg background pause` works either
    way."""

    @field_validator("host_pause_file", mode="after")
    @classmethod
    def _expand_and_require_absolute(cls, v: Path | None) -> Path | None:
        if v is None:
            return None
        expanded = v.expanduser()
        if not expanded.is_absolute():
            raise ValueError(f"background.host_pause_file must be an absolute path, got '{v}'")
        return expanded


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------


class RenderConfig(StrictModel):
    timezone: str = "America/Los_Angeles"
    attachment_snippet_chars: int = Field(default=200, ge=0)

    @field_validator("timezone", mode="after")
    @classmethod
    def _must_be_a_real_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"render.timezone '{v}' is not a known IANA timezone") from exc
        return v


# --------------------------------------------------------------------------
# mcp
# --------------------------------------------------------------------------


IDLE_UNLOAD_MIN_SECONDS = 60
"""The shortest idle period `mcp.*.idle_unload_seconds` accepts, other
than 0 (off). Mirrors `imsg.retrieval.idle_unload.MIN_IDLE_UNLOAD_SECONDS`
(not imported: the config layer imports nothing from the runtime)."""


def _check_idle_unload_seconds(value: int, key: str) -> int:
    if value != 0 and value < IDLE_UNLOAD_MIN_SECONDS:
        raise ValueError(
            f"{key} must be 0 (never unload) or at least {IDLE_UNLOAD_MIN_SECONDS} "
            f"seconds, got {value}: reloading the models takes tens of seconds, so "
            f"a shorter period would unload them between the queries of one conversation"
        )
    return value


class McpLocalConfig(StrictModel):
    enabled: bool = True
    warm_at_start: bool = False
    """Load the retrieval models as soon as the server starts. Off by
    default: every client session starts its own `imsg mcp local`, most
    never search, and a server that loads at start holds a full model set
    (`memory.footprints.local_server_bytes`, 17.2 GiB) whether or not
    anyone asks it anything — that is how idle sessions filled the 64 GiB
    index host (CHANGELOG 2026-09-24). With it off, the first retrieval
    call starts the load, through the memory admission check like every
    load, and waits for it up to 90 s (else `WARMING_UP`); the idle
    unloader drops the models again afterwards. `true` restores loading
    at start."""
    idle_unload_seconds: int = Field(default=600, ge=0)
    """Drop the retrieval models after this many seconds with no
    retrieval tool call; the next call reloads them (waiting up to 90 s,
    else `WARMING_UP`). 0 keeps them loaded for the life of the process.
    Every client session runs its own `imsg mcp local`, each holding its
    own copy of the models, so this is what keeps idle sessions from
    exhausting memory (`imsg.retrieval.idle_unload`)."""

    @field_validator("idle_unload_seconds", mode="after")
    @classmethod
    def _idle_unload(cls, value: int) -> int:
        return _check_idle_unload_seconds(value, "mcp.local.idle_unload_seconds")


class McpPublicOauthConfig(StrictModel):
    issuer: str = "google"
    client_id: str | None = None
    client_secret: SecretRef | None = None
    owner_subject: SecretRef | None = None
    tokeninfo_cache_ttl_seconds: int = Field(default=60, ge=1)


class McpPublicConfig(StrictModel):
    enabled: bool = False
    bind: str = "127.0.0.1:8700"
    external_url: str | None = None
    # REQUIRED, no default (D6): an omitted key is a validation error, never
    # a silent 'full'. Do not add `= None` or any other default here.
    scope: Literal["full", "allowlist"]
    scope_approval_id: str | None = None
    allowed_origins: list[str] = Field(default_factory=list)
    allowed_hosts: list[str] = Field(default_factory=list)
    protocol_versions: list[str] = Field(
        default_factory=lambda: ["2025-11-25", "2026-07-28"]
    )
    rate_limit_per_minute: int = Field(default=60, ge=1)
    oauth: McpPublicOauthConfig = Field(default_factory=McpPublicOauthConfig)
    idle_unload_seconds: int = Field(default=0, ge=0)
    """As `mcp.local.idle_unload_seconds`, but off by default: the public
    surface is one process, not one per session, and a public retrieval
    call waits only 20 s for the models before answering `WARMING_UP`
    (the Cloudflare edge gives up at 125 s), which is shorter than a
    reload — so with this on, the first public call after an unload is
    answered `WARMING_UP` and has to be retried."""

    @field_validator("idle_unload_seconds", mode="after")
    @classmethod
    def _idle_unload(cls, value: int) -> int:
        return _check_idle_unload_seconds(value, "mcp.public.idle_unload_seconds")

    @model_validator(mode="after")
    def _enabled_requires_full_configuration(self) -> McpPublicConfig:
        if not self.enabled:
            return self
        missing: list[str] = []
        if not self.external_url:
            missing.append("mcp.public.external_url")
        if not self.allowed_origins:
            missing.append("mcp.public.allowed_origins (non-empty)")
        if not self.allowed_hosts:
            missing.append("mcp.public.allowed_hosts (non-empty)")
        if self.oauth.owner_subject is None:
            missing.append("mcp.public.oauth.owner_subject")
        if self.scope == "full" and not self.scope_approval_id:
            missing.append(
                "mcp.public.scope_approval_id (required when scope: full)"
            )
        if missing:
            raise ValueError(
                "mcp.public.enabled=true requires: " + ", ".join(missing) +
                " (SPEC §6). Note: this build does not check cloudflared "
                "installation or AT-1 completion — those are runtime "
                "preconditions for the Phase 6 MCP-surface build, not "
                "config-parse-time checks."
            )
        return self


class McpConfig(StrictModel):
    local: McpLocalConfig = Field(default_factory=McpLocalConfig)
    public: McpPublicConfig


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


class ExportConfig(StrictModel):
    gcp_project: str = Field(min_length=1)
    gcs_bucket: str = Field(min_length=1)
    data_store_id: str = Field(min_length=1)
    format: Literal["txt"] = "txt"
    batch_max_files: int = Field(default=100_000, ge=1)
    # (secret) The GCP service-account key `imsg export push` authenticates
    # with, as a `keychain:<item>` / `env:<VAR>` reference like every other
    # secret field (SPEC §6, `imsg.config.secrets.SecretRef`); the resolved
    # value must be the key's raw JSON text.
    #
    # `None` is the default ON PURPOSE and is not an oversight: with no
    # credential named, `imsg export push` refuses before it so much as
    # imports a Google client library, so a repo checkout with a stock
    # config cannot reach GCS or Discovery Engine at all. Adding a default
    # here — any default — would remove that property.
    gcp_credentials: SecretRef | None = None


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


class EvalConfig(StrictModel):
    seed_queries: Path = Field(default=Path("private/eval/queries.yaml"))
    runs_dir: Path = Field(default=Path("eval/runs"))


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class LoggingConfig(StrictModel):
    level: str = "INFO"
    allow_content_debug: bool = False

    @field_validator("level", mode="after")
    @classmethod
    def _known_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in _LOG_LEVELS:
            raise ValueError(f"logging.level must be one of {sorted(_LOG_LEVELS)}, got '{v}'")
        return upper


# --------------------------------------------------------------------------
# attachments — where copies of attachments can be fetched from (D13)
# --------------------------------------------------------------------------

_LOCATION_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SSH_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")
_REMOTE_ROOT_RE = re.compile(r"^/[A-Za-z0-9._/+@-]*$")


def _check_location_code(code: str, field_name: str) -> str:
    if not _LOCATION_CODE_RE.fullmatch(code) or code in {".", ".."}:
        raise ValueError(
            f"{field_name} must be a location code of letters, digits, '.', '_' or '-' "
            f"(it names a staging directory), got {code!r}"
        )
    return code


class AttachmentPullConfig(StrictModel):
    """A location this host copies from itself, read-only, over SSH (a NAS
    share). `root` is the absolute path on `ssh_host` that the location's
    catalog paths are relative to."""

    location: str
    ssh_host: str
    root: str

    @field_validator("location", mode="after")
    @classmethod
    def _code(cls, v: str) -> str:
        return _check_location_code(v, "attachments.pull[].location")

    @field_validator("ssh_host", mode="after")
    @classmethod
    def _host(cls, v: str) -> str:
        if not _SSH_HOST_RE.fullmatch(v):
            raise ValueError(f"attachments.pull[].ssh_host must be a plain host or alias, got {v!r}")
        return v

    @field_validator("root", mode="after")
    @classmethod
    def _root(cls, v: str) -> str:
        if not _REMOTE_ROOT_RE.fullmatch(v) or "/../" in f"{v}/" or "/./" in f"{v}/":
            raise ValueError(
                f"attachments.pull[].root must be a plain absolute path of [A-Za-z0-9._/+@-], "
                f"got {v!r}"
            )
        return v


class AttachmentsConfig(StrictModel):
    """The attachment fetcher (D13): every section is optional, and with
    none of it set the backfill still tries this host's own Messages folder
    and anything another host has pushed into `staging_dir`."""

    staging_dir: Path = Field(default=Path("attachment-staging"))
    """Where pushed and pulled copies land before they are verified;
    relative to `paths.data_root` (checked to resolve under it)."""
    local_location: str | None = None
    """The code this host's own Messages folder is filed under. Default:
    the name of the `sync.sources` entry that is this host's live chat.db."""
    push_locations: list[str] = Field(default_factory=list)
    """Locations another host copies in with `imsg push-attachments`
    (another Mac's Messages folder, its attached drives)."""
    pull: list[AttachmentPullConfig] = Field(default_factory=list)
    ssh_command: str = "ssh -o BatchMode=yes"
    rsync_command: str = "rsync"

    @field_validator("local_location", mode="after")
    @classmethod
    def _local_code(cls, v: str | None) -> str | None:
        return None if v is None else _check_location_code(v, "attachments.local_location")

    @field_validator("push_locations", mode="after")
    @classmethod
    def _push_codes(cls, v: list[str]) -> list[str]:
        for code in v:
            _check_location_code(code, "attachments.push_locations[]")
        return v

    @field_validator("pull", mode="after")
    @classmethod
    def _unique_pulls(cls, v: list[AttachmentPullConfig]) -> list[AttachmentPullConfig]:
        codes = [p.location for p in v]
        if len(codes) != len(set(codes)):
            raise ValueError(f"attachments.pull locations must be unique, got {codes}")
        return v


# --------------------------------------------------------------------------
# root
# --------------------------------------------------------------------------


class Config(StrictModel):
    """The full, validated `config.yaml`. See module docstring for the
    security-relevant enforcement this class performs."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    database: DatabaseConfig
    sync: SyncConfig
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    mcp: McpConfig
    export: ExportConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    attachments: AttachmentsConfig = Field(default_factory=AttachmentsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)

    # ---- cross-field path-containment validation (hard requirements #1, #2) ----

    @model_validator(mode="after")
    def _exactly_one_path_under_messages(self) -> Config:
        """Hard requirement #1 / SPEC §1, §6.

        `paths.live_chat_db` is the only path anywhere in the config
        permitted to resolve under `~/Library/Messages`. Every other
        configured filesystem path that resolves there — output, cache,
        or a second declared source — is rejected.
        """
        declared_source = resolve_path(self.paths.live_chat_db)
        if not is_contained_in(declared_source, MESSAGES_DIR) and declared_source != resolve_path(
            MESSAGES_DIR
        ):
            raise ValueError(
                f"paths.live_chat_db ('{self.paths.live_chat_db}') must resolve "
                f"under {MESSAGES_DIR} — it is the declared read-only source"
            )

        candidates: list[tuple[str, Path]] = [
            ("paths.data_root", self.paths.data_root),
        ]
        if self.background.host_pause_file is not None:
            candidates.append(("background.host_pause_file", self.background.host_pause_file))
        for i, source in enumerate(self.sync.sources):
            candidates.append((f"sync.sources[{i}] ({source.name}).chat_db", source.chat_db))

        for field_name, raw_path in candidates:
            resolved = resolve_path(raw_path)
            if is_contained_in(resolved, MESSAGES_DIR) and resolved != declared_source:
                raise ValueError(
                    f"{field_name} ('{raw_path}') resolves under {MESSAGES_DIR}, "
                    f"but the only path permitted there is the declared "
                    f"paths.live_chat_db source ('{self.paths.live_chat_db}') — "
                    f"CLAUDE.md non-negotiable #1: never write to the live chat.db, "
                    f"and no second path may even read from that directory "
                    f"un-declared"
                )
        return self

    @model_validator(mode="after")
    def _derived_paths_resolve_under_data_root(self) -> Config:
        """Hard requirement #2 / SPEC §1, §2, §6.

        Every derived/output path in the config must resolve under
        `paths.data_root` — relative paths are joined under it; absolute
        paths must independently resolve there. Symlinks and `..` are
        resolved before comparison (SPEC §5.4: never infer containment
        from a string prefix).
        """
        root = self.paths.data_root
        derived: list[tuple[str, Path]] = [
            ("database.cluster_fingerprint_file", self.database.cluster_fingerprint_file),
            ("segmentation.boundary_prompt", self.segmentation.boundary_prompt),
            ("enrichment.caption_prompt", self.enrichment.caption_prompt),
            ("eval.seed_queries", self.eval.seed_queries),
            ("eval.runs_dir", self.eval.runs_dir),
            ("attachments.staging_dir", self.attachments.staging_dir),
        ]
        for field_name, raw_path in derived:
            resolved = resolve_path(join_under_root(root, raw_path))
            if not is_contained_in(resolved, root):
                raise ValueError(
                    f"{field_name} ('{raw_path}') must resolve under "
                    f"paths.data_root ('{root}') — CLAUDE.md non-negotiable #2: "
                    f"all derived state lives on the encrypted volume, resolved "
                    f"path was '{resolved}'"
                )
            if is_contained_in(resolved, MESSAGES_DIR):
                raise ValueError(
                    f"{field_name} ('{raw_path}') resolves under {MESSAGES_DIR} — "
                    f"derived/output paths may never live there (CLAUDE.md "
                    f"non-negotiable #1)"
                )
        return self

    @model_validator(mode="after")
    def _reranker_model_dir_stays_under_data_root(self) -> Config:
        """`retrieval.reranker_model` may name a directory under
        `paths.data_root` (a local conversion). The field validator has
        already rejected absolute, `~` and `..` forms; this resolves the
        candidate against the real filesystem so a symlink planted under
        data_root cannot point the provider at a directory outside it. A
        Hub repo id (`owner/name`) joins under data_root trivially and
        passes — the factory decides which form it is by existence."""
        root = self.paths.data_root
        value = self.retrieval.reranker_model
        resolved = resolve_path(join_under_root(root, value))
        if not is_contained_in(resolved, root) or is_contained_in(resolved, MESSAGES_DIR):
            raise ValueError(
                f"retrieval.reranker_model ('{value}') must be {RERANKER_MODEL_FORMS}; as a "
                f"directory it must resolve under paths.data_root ('{root}') and never under "
                f"{MESSAGES_DIR} — resolved path was '{resolved}'"
            )
        return self


# Re-exported so `from imsg.config.schema import ...` covers everything
# the rest of the codebase needs without reaching into submodules.
PathLike = Annotated[Path, "resolved via imsg.paths helpers before use"]

__all__ = [
    "DEFAULT_HOST_PAUSE_FILE",
    "HNSW_EF_SEARCH_MAX",
    "RERANKER_MODEL_FORMS",
    "AttachmentPullConfig",
    "AttachmentsConfig",
    "BackgroundConfig",
    "Config",
    "DatabaseConfig",
    "EmbeddingConfig",
    "EnrichmentConfig",
    "EvalConfig",
    "ExportConfig",
    "IdentityConfig",
    "LoggingConfig",
    "McpConfig",
    "McpLocalConfig",
    "McpPublicConfig",
    "McpPublicOauthConfig",
    "MemoryConfig",
    "MlxMemoryLimits",
    "ModelFootprints",
    "ModelsConfig",
    "MultimodalEmbeddingConfig",
    "PathsConfig",
    "PolicyConfig",
    "RenderConfig",
    "RetrievalConfig",
    "SegmentationConfig",
    "SyncConfig",
    "SyncSourceConfig",
]
