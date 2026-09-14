"""Provider factory: `config.yaml` -> model providers (SPEC §4.1, §6).

Every CLI command that needs a model — `segment`, `embed`, `sync`,
`enrich`, `mcp local`, `mcp public`, `eval run`/`eval pool` — obtains
its providers from the `build_*` functions below and nowhere else, so
backend selection is a single config field:

    models:
      backend: real   # the default
      # backend: fake # deterministic test stand-ins — explicit opt-in only

`real` constructs the MLX / Apple Vision / PE-Core implementations named
in `REAL_PROVIDERS`, from the repo ids and immutable revisions in
`embedding.*`, `retrieval.reranker_*`, `segmentation.boundary_*` and
`enrichment.*` (defaults mirror `models/manifest.lock.yaml`). `fake`
constructs the `Fake*` classes every test uses; because a fake run
reports success while producing meaningless search results, every
command prints `backend_status_line(cfg)` — one line,
`models: backend=<real|fake>` — so a fake run can never be mistaken
for a real one.

The real classes are imported lazily, by dotted path, inside the
builders (never at module-import time): their runtime packages (the
`models` extra: mlx, mlx-lm, mlx-whisper, mlx-vlm, pyobjc Vision/Quartz,
torch, ...) are heavy, macOS-only, and optional for the fake backend.
Any failure to load or construct a real provider surfaces as one
`ProviderUnavailableError` whose message tells the operator what to do
(install the `models` extra / merge the provider module / author the
prompt file), never as a traceback.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from imsg.embed.provider import (
    FakeMultimodalEmbeddingProvider,
    FakeTextEmbeddingProvider,
    MultimodalEmbeddingProvider,
    TextEmbeddingProvider,
)
from imsg.enrich.pipeline import EnrichmentProviders
from imsg.enrich.provider import (
    CaptionProvider,
    FakeCaptionProvider,
    FakeOcrProvider,
    FakeTranscriptionProvider,
    OcrProvider,
    TranscriptionProvider,
)
from imsg.errors import ImsgError, ProviderUnavailableError
from imsg.retrieval.reranker import FakeRerankerProvider, RerankerProvider
from imsg.segment.boundaries import BoundaryProvider, FakeBoundaryProvider

if TYPE_CHECKING:
    from imsg.config.schema import Config

MODELS_EXTRA = "models"
"""Name of the `[project.optional-dependencies]` extra that carries every
real-provider runtime package. Quoted verbatim in error messages."""

INSTALL_HINT = (
    f"install the `{MODELS_EXTRA}` extra (`uv sync --extra {MODELS_EXTRA}`, or "
    f"`pip install 'imsg[{MODELS_EXTRA}]'`)"
)


@dataclass(frozen=True, slots=True)
class RealProviderSpec:
    """Where a real provider class lives. Resolved lazily by
    `importlib.import_module` so the module need not exist (or its
    runtime packages be installed) until a `real`-backend build asks
    for it."""

    role: str
    module: str
    class_name: str

    @property
    def dotted_path(self) -> str:
        return f"{self.module}.{self.class_name}"


REAL_PROVIDERS: dict[str, RealProviderSpec] = {
    "text_embedding": RealProviderSpec(
        "text_embedding", "imsg.embed.mlx_text", "MlxTextEmbeddingProvider"
    ),
    "multimodal_embedding": RealProviderSpec(
        "multimodal_embedding", "imsg.embed.pe_core_multimodal", "PeCoreMultimodalEmbeddingProvider"
    ),
    "boundary": RealProviderSpec("boundary", "imsg.segment.mlx_boundaries", "MlxBoundaryProvider"),
    "reranker": RealProviderSpec("reranker", "imsg.retrieval.mlx_reranker", "MlxRerankerProvider"),
    "ocr": RealProviderSpec("ocr", "imsg.enrich.vision_ocr", "AppleVisionOcrProvider"),
    "transcription": RealProviderSpec(
        "transcription", "imsg.enrich.mlx_whisper_transcription", "MlxWhisperTranscriptionProvider"
    ),
    "caption": RealProviderSpec("caption", "imsg.enrich.mlx_vlm_caption", "MlxVlmCaptionProvider"),
}
"""One entry per provider role. The constructor signatures the builders
below rely on (positional required args, keyword-only options):

- text_embedding:       (model_repo, revision, dim, *, batch_size=32, max_length=8192)
- multimodal_embedding: (model_repo, revision, dim, *, device="mps", batch_size=16,
                         allow_cpu_fallback=False)
- boundary:             (model_repo, revision, prompt_template: str, *, max_tokens=512,
                         timeout_seconds=120.0)
- reranker:             (model_repo, revision, *, instruction=None, batch_size=8,
                         max_length=8192)
- ocr:                  (*, recognition_languages=None, minimum_text_height=None)
- transcription:        (model_repo, revision, *, language=None)
- caption:              (model_repo, revision, prompt: str, *, max_tokens=256)
"""


def backend_status_line(cfg: Config) -> str:
    """The one line every provider-building command prints:
    `models: backend=real` / `models: backend=fake`."""
    return f"models: backend={cfg.models.backend}"


# --------------------------------------------------------------------------
# lazy loading + construction, with operator-facing errors
# --------------------------------------------------------------------------


def _missing_module_error(
    spec: RealProviderSpec, exc: ModuleNotFoundError
) -> ProviderUnavailableError:
    missing = exc.name or spec.module
    if missing == spec.module or missing.startswith("imsg."):
        return ProviderUnavailableError(
            f"this build has no real '{spec.role}' provider: module '{missing}' is not "
            f"present (expected {spec.dotted_path}). Merge/install the provider module, "
            f"or set `models.backend: fake` in config.yaml to run with the deterministic "
            f"stand-ins — every command then prints 'models: backend=fake', and search "
            f"quality from such a run is not representative."
        )
    return ProviderUnavailableError(
        f"the real '{spec.role}' provider ({spec.dotted_path}) needs the runtime package "
        f"'{missing}', which is not installed in this environment — {INSTALL_HINT}. "
        f"See pyproject.toml [project.optional-dependencies].{MODELS_EXTRA} and "
        f"models/manifest.lock.yaml `min_runtime`."
    )


def _load_real_class(spec: RealProviderSpec) -> Any:
    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        raise _missing_module_error(spec, exc) from exc
    except ImportError as exc:  # a present module whose native/runtime import broke
        raise ProviderUnavailableError(
            f"the real '{spec.role}' provider ({spec.dotted_path}) could not be imported: "
            f"{exc}. Its runtime packages may be missing or broken — {INSTALL_HINT}."
        ) from exc
    try:
        return getattr(module, spec.class_name)
    except AttributeError as exc:
        raise ProviderUnavailableError(
            f"module '{spec.module}' has no '{spec.class_name}' — this build's "
            f"'{spec.role}' provider module does not match the factory's expectation "
            f"({spec.dotted_path})"
        ) from exc


def _construct(spec: RealProviderSpec, *args: object, **kwargs: object) -> object:
    """Load and instantiate the real class. Import-time *and*
    construct-time failures both land as `ProviderUnavailableError`:
    a provider may import its runtime lazily inside `__init__`, and
    model download / revision resolution happens there too."""
    cls = _load_real_class(spec)
    try:
        return cls(*args, **kwargs)
    except ImsgError:
        raise
    except ModuleNotFoundError as exc:
        raise _missing_module_error(spec, exc) from exc
    except ImportError as exc:
        raise ProviderUnavailableError(
            f"the real '{spec.role}' provider ({spec.dotted_path}) failed to import a "
            f"runtime dependency while initializing: {exc} — {INSTALL_HINT}."
        ) from exc
    except Exception as exc:  # surfaced as one clean operator message, cause chained
        raise ProviderUnavailableError(
            f"could not construct the real '{spec.role}' provider ({spec.dotted_path}) "
            f"from config: {type(exc).__name__}: {exc}"
        ) from exc


def _check_dim(spec: RealProviderSpec, provider: object, expected: int) -> None:
    """An embedding provider that silently ignores the configured `dim`
    would violate the pgvector CHECK constraints at insert time — catch
    it here instead, at construction."""
    actual = getattr(provider, "dim", expected)
    if actual != expected:
        raise ProviderUnavailableError(
            f"the real '{spec.role}' provider ({spec.dotted_path}) reports dim={actual}, "
            f"but config requires {expected} (the migration DDL's CHECK constraint)"
        )


def _read_prompt_or_raise(path_description: str, path: Path, *, field_name: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProviderUnavailableError(
            f"{path_description} not found at '{path}' — author it before running "
            f"with `models.backend: real` (config {field_name}; the path is relative to "
            f"paths.data_root)"
        ) from exc


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def build_text_provider(cfg: Config) -> TextEmbeddingProvider:
    """S6's primary text embedder / the retrieval service's query
    embedder (SPEC §4.1: Qwen3-Embedding-8B, 2048-dim MRL)."""
    if cfg.models.backend == "fake":
        return FakeTextEmbeddingProvider(dim=cfg.embedding.dim)
    spec = REAL_PROVIDERS["text_embedding"]
    provider = _construct(
        spec,
        cfg.embedding.model,
        cfg.embedding.revision,
        cfg.embedding.dim,
        batch_size=cfg.embedding.batch_size,
    )
    _check_dim(spec, provider, cfg.embedding.dim)
    return cast("TextEmbeddingProvider", provider)


def build_multimodal_provider(cfg: Config) -> MultimodalEmbeddingProvider | None:
    """The secondary visual embedder (D3a: PE-Core-G14-448, 1280-dim), or
    `None` when `embedding.multimodal.enabled` is false — callers already
    treat `None` as "channel C disabled"."""
    if not cfg.embedding.multimodal.enabled:
        return None
    mm = cfg.embedding.multimodal
    if cfg.models.backend == "fake":
        return FakeMultimodalEmbeddingProvider(dim=mm.dim)
    spec = REAL_PROVIDERS["multimodal_embedding"]
    provider = _construct(spec, mm.model, mm.revision, mm.dim, batch_size=mm.batch_size)
    _check_dim(spec, provider, mm.dim)
    return cast("MultimodalEmbeddingProvider", provider)


def build_boundary_provider(cfg: Config, prompt_template: str) -> BoundaryProvider:
    """S4's topical-boundary LLM (SPEC §4.1/D4). `prompt_template` is
    the decoded `segmentation.boundary_prompt` file — the caller reads
    it (and hashes the raw bytes into `seg_config_hash`), so the same
    bytes drive both the hash and the model."""
    if cfg.models.backend == "fake":
        return FakeBoundaryProvider()
    spec = REAL_PROVIDERS["boundary"]
    provider = _construct(
        spec, cfg.segmentation.boundary_model, cfg.segmentation.boundary_revision, prompt_template
    )
    return cast("BoundaryProvider", provider)


def build_reranker(cfg: Config) -> RerankerProvider:
    """SPEC §9.4 step 7's reranker (Qwen3-Reranker-8B)."""
    if cfg.models.backend == "fake":
        return FakeRerankerProvider()
    spec = REAL_PROVIDERS["reranker"]
    provider = _construct(spec, cfg.retrieval.reranker_model, cfg.retrieval.reranker_revision)
    return cast("RerankerProvider", provider)


def read_caption_prompt(cfg: Config) -> str:
    """The fixed captioning prompt (SPEC §4.1), `enrichment.caption_prompt`
    resolved under `paths.data_root` — same convention as the boundary
    prompt. Only the real backend needs it."""
    path = cfg.paths.data_root / cfg.enrichment.caption_prompt
    return _read_prompt_or_raise(
        "enrichment caption prompt", path, field_name="enrichment.caption_prompt"
    )


def build_enrichment_providers(
    cfg: Config, *, caption_prompt: str | None = None
) -> EnrichmentProviders:
    """S5b's three model-backed steps (SPEC §4.1): Apple Vision OCR,
    the local captioning VLM, and Whisper transcription. `caption_prompt`
    may be supplied by a caller that already read it; otherwise the real
    backend reads `enrichment.caption_prompt` itself."""
    if cfg.models.backend == "fake":
        return EnrichmentProviders(
            ocr=FakeOcrProvider(),
            caption=FakeCaptionProvider(),
            transcription=FakeTranscriptionProvider(),
        )
    enrichment = cfg.enrichment
    ocr = _construct(
        REAL_PROVIDERS["ocr"],
        recognition_languages=enrichment.ocr_languages,
        minimum_text_height=enrichment.ocr_minimum_text_height,
    )
    transcription = _construct(
        REAL_PROVIDERS["transcription"],
        enrichment.transcription_model,
        enrichment.transcription_revision,
        language=enrichment.transcription_language,
    )
    prompt = caption_prompt if caption_prompt is not None else read_caption_prompt(cfg)
    caption = _construct(
        REAL_PROVIDERS["caption"], enrichment.caption_model, enrichment.caption_revision, prompt
    )
    return EnrichmentProviders(
        ocr=cast("OcrProvider", ocr),
        caption=cast("CaptionProvider", caption),
        transcription=cast("TranscriptionProvider", transcription),
    )


__all__ = [
    "INSTALL_HINT",
    "MODELS_EXTRA",
    "REAL_PROVIDERS",
    "RealProviderSpec",
    "backend_status_line",
    "build_boundary_provider",
    "build_enrichment_providers",
    "build_multimodal_provider",
    "build_reranker",
    "build_text_provider",
    "read_caption_prompt",
]
