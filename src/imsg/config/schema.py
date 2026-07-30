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

from imsg import constants
from imsg.config.secrets import SecretRef
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
    boundary_model: str = "qwen3.5-35b-a3b-4bit"
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

    @field_validator("window", mode="after")
    @classmethod
    def _window_must_be_hh_mm_range(cls, v: str) -> str:
        if not _WINDOW_RE.match(v):
            raise ValueError(
                f"enrichment.window must look like 'HH:MM-HH:MM', got '{v}'"
            )
        return v


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------


class MultimodalEmbeddingConfig(StrictModel):
    enabled: bool = True
    provider: Literal["local"] = "local"
    scope: Literal["full"] = "full"
    model: str = "facebook/PE-Core-G14-448"
    revision: str = Field(min_length=1)
    dim: int = constants.MULTIMODAL_EMBEDDING_DIM

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
    model: str = "Qwen/Qwen3-Embedding-8B"
    revision: str = Field(min_length=1)
    quantization: str = "8bit"
    dim: int = constants.PRIMARY_EMBEDDING_DIM
    batch_size: int = Field(default=32, ge=1)
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


class RetrievalConfig(StrictModel):
    k_fts: int = Field(default=100, ge=1)
    k_vector: int = Field(default=100, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_top: int = Field(default=50, ge=1)
    reranker_model: str = "Qwen/Qwen3-Reranker-8B"
    reranker_revision: str = Field(min_length=1)
    default_limit: int = Field(default=10, ge=1)


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


class McpLocalConfig(StrictModel):
    enabled: bool = True


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
    render: RenderConfig = Field(default_factory=RenderConfig)
    mcp: McpConfig
    export: ExportConfig
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

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
            ("eval.seed_queries", self.eval.seed_queries),
            ("eval.runs_dir", self.eval.runs_dir),
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


# Re-exported so `from imsg.config.schema import ...` covers everything
# the rest of the codebase needs without reaching into submodules.
PathLike = Annotated[Path, "resolved via imsg.paths helpers before use"]

__all__ = [
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
    "MultimodalEmbeddingConfig",
    "PathsConfig",
    "PolicyConfig",
    "RenderConfig",
    "RetrievalConfig",
    "SegmentationConfig",
    "SyncConfig",
    "SyncSourceConfig",
]
