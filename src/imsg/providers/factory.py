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
`enrichment.*` (defaults mirror `models/manifest.lock.yaml`). A model
field may also name a directory relative to `paths.data_root` holding a
local conversion (the lock's `source: local_conversion` entries, their
`output_dir`): `resolve_local_model_dir` reads the value as that
directory when it exists under the data root, and the provider is built
from the directory with `revision=None` and a `model_id` of
`<data-root-relative dir>@<upstream sha>`. `fake`
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
from imsg.paths import is_contained_in, join_under_root, resolve_path
from imsg.retrieval.reranker import FakeRerankerProvider, RerankerProvider
from imsg.segment.boundaries import BoundaryProvider, FakeBoundaryProvider
from imsg.shared_vlm_runtime import SharedVlmRuntime

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

- text_embedding:       (model_repo, revision, dim, *, batch_size=32, max_length=8192,
                         max_batch_tokens=2048, cache_limit_bytes=8 GiB, model_id=None)
- multimodal_embedding: (model_repo, revision, dim, *, device="mps", batch_size=16,
                         allow_cpu_fallback=False)
- boundary:             (model_repo, revision, prompt_template: str, *, max_tokens=512,
                         timeout_seconds=120.0, shared_runtime=None,
                         cache_limit_bytes=8 GiB)
- reranker:             (model_repo, revision, *, instruction=None, batch_size=32,
                         max_length=8192, max_batch_tokens=1024, doc_max_tokens=None,
                         cache_limit_bytes=8 GiB, model_id=None)
- ocr:                  (*, recognition_languages=None, minimum_text_height=None)
- transcription:        (model_repo, revision, *, language=None, temperature=...,
                         cache_limit_bytes=None)
- caption:              (model_repo, revision, prompt: str, *, max_tokens=256,
                         shared_runtime=None, cache_limit_bytes=4 GiB)
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


# --------------------------------------------------------------------------
# local conversions: a model field naming a directory under data_root
# --------------------------------------------------------------------------

LOCAL_MODEL_DIR_PREFIX = "models"
"""First path segment of every local conversion's `output_dir` in
`models/manifest.lock.yaml`. A config value under it that does not exist
on disk is reported as a missing conversion rather than handed to the
Hub as a repo id (which would fail later with an unrelated 401)."""


def resolve_local_model_dir(data_root: Path, value: str) -> Path | None:
    """`data_root/<value>` when it is an existing directory that still
    resolves under `data_root` with symlinks followed (SPEC §5.4) — the
    rule by which a `*_model` config value is read as a local conversion
    rather than a Hugging Face repo id. `None` for any value that is not
    such a directory (a repo id, or a directory not present)."""
    candidate = resolve_path(join_under_root(data_root, value))
    if not candidate.is_dir():
        return None
    if not is_contained_in(candidate, data_root):
        raise ProviderUnavailableError(
            f"model directory '{value}' resolves to '{candidate}', outside paths.data_root "
            f"('{data_root}') — CLAUDE.md non-negotiable #2: every model directory a provider "
            f"opens must live on the encrypted volume"
        )
    return candidate


def local_model_id(value: str, revision: str) -> str:
    """`<data-root-relative dir>@<upstream sha>` — the `model_id` recorded
    for a provider built from a local conversion: the config's relative
    directory (normalised), never the absolute path, plus the upstream
    commit the conversion derives from."""
    return f"{Path(value).as_posix()}@{revision}"


def _looks_like_local_model_dir(value: str) -> bool:
    parts = Path(value).parts
    return len(parts) > 1 and parts[0] == LOCAL_MODEL_DIR_PREFIX


# --------------------------------------------------------------------------
# prompt files: the operator's copy under data_root, else the shipped one
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    """Where a prompt file was found: the operator's copy under
    `paths.data_root` (`source="data_root"`) or the one the repository
    ships at the same relative path (`source="repo"`)."""

    path: Path
    source: str

    @property
    def description(self) -> str:
        return "data_root" if self.source == "data_root" else "repo-shipped default"


def default_prompt_root() -> Path:
    """The repository root, under which the shipped `prompts/*.txt` live —
    the fallback when `paths.data_root` has no operator-authored copy at
    the same relative path. Tests patch this to simulate an absent
    fallback."""
    # src/imsg/providers/factory.py -> providers -> imsg -> src -> repo root
    return Path(__file__).resolve().parents[3]


def resolve_prompt_path(data_root: Path, relative: Path) -> ResolvedPrompt | None:
    """`data_root/<relative>` when it exists, else the repo-shipped file at
    the same relative path, else `None`. The data_root copy always wins,
    so an operator overrides a shipped prompt without touching the
    checkout — and `seg_config_hash` / `prompt_sha256` hash whichever
    bytes were actually used, so the two never disagree with the model."""
    candidate = data_root / relative
    if candidate.is_file():
        return ResolvedPrompt(candidate, "data_root")
    shipped = default_prompt_root() / relative
    if shipped.is_file():
        return ResolvedPrompt(shipped, "repo")
    return None


def read_prompt_text(path: Path, *, field_name: str) -> str:
    """The file's exact bytes decoded as strict UTF-8 — what the providers
    hash into `prompt_sha256` — as one operator-facing error when it
    cannot be read or is not UTF-8 (a `UnicodeDecodeError` would
    otherwise escape the CLI's `ImsgError` boundary as a traceback)."""
    try:
        return path.read_bytes().decode("utf-8")
    except OSError as exc:
        raise ProviderUnavailableError(
            f"prompt file '{path}' (config {field_name}) could not be read: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ProviderUnavailableError(
            f"prompt file '{path}' (config {field_name}) is not valid UTF-8: {exc}"
        ) from exc


def resolve_caption_prompt(cfg: Config) -> ResolvedPrompt:
    """Locate the fixed captioning prompt (SPEC §4.1): `enrichment.
    caption_prompt` under `paths.data_root`, else the repo-shipped copy."""
    relative = cfg.enrichment.caption_prompt
    resolved = resolve_prompt_path(cfg.paths.data_root, relative)
    if resolved is None:
        raise ProviderUnavailableError(
            f"enrichment caption prompt not found at '{cfg.paths.data_root / relative}' "
            f"(and the repository ships no '{relative}' to fall back to) — author it "
            f"before running with `models.backend: real` (config enrichment.caption_prompt; "
            f"the path is relative to paths.data_root)"
        )
    return resolved


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def boundary_and_caption_share_weights(cfg: Config) -> bool:
    """Whether `segmentation.boundary_*` and `enrichment.caption_*` name
    the same checkpoint, so one loaded copy can serve both roles. True by
    default — `imsg.constants` sets `CAPTION_MODEL_REPO =
    BOUNDARY_MODEL_REPO` and the model manifest records one entry with
    both roles — but an operator may point them at different models, and
    then there is nothing to share."""
    return (
        cfg.models.share_boundary_and_caption_weights
        and cfg.segmentation.boundary_model == cfg.enrichment.caption_model
        and cfg.segmentation.boundary_revision == cfg.enrichment.caption_revision
    )


def build_shared_vlm_runtime(cfg: Config) -> SharedVlmRuntime | None:
    """One runtime for a process that will build **both** the captioner
    and the boundary provider, so the shared checkpoint loads once
    (`imsg.shared_vlm_runtime`; D10.3 defect 1 — two independent copies
    cost 18 GiB on a 64 GB host).

    `None` when there is nothing to share: the fake backend, an operator
    who turned sharing off, or two different pins. Callers pass whatever
    they get straight to both builders, which treat `None` as "load the
    way you always did".

    A process that builds only one of the two should pass `None` — the
    boundary provider's own `mlx_lm` loader is 0.83 GiB lighter than the
    vision-language object, and the captioner makes its own private
    runtime regardless.
    """
    if cfg.models.backend == "fake" or not boundary_and_caption_share_weights(cfg):
        return None
    return SharedVlmRuntime(cache_limit_bytes=cfg.models.enrichment_cache_limit_bytes)


def build_text_provider(cfg: Config) -> TextEmbeddingProvider:
    """S6's primary text embedder / the retrieval service's query
    embedder (SPEC §4.1: Qwen3-Embedding-8B, 2048-dim MRL). `embedding.
    model` is a local conversion when `<paths.data_root>/<value>` is an
    existing directory — built with `revision=None` and `model_id`
    `<value>@<embedding.revision>`, the upstream sha (the reranker's
    rule, `build_reranker`) — and a Hub repo id pinned at
    `embedding.revision` otherwise."""
    if cfg.models.backend == "fake":
        return FakeTextEmbeddingProvider(dim=cfg.embedding.dim)
    spec = REAL_PROVIDERS["text_embedding"]
    model, revision = cfg.embedding.model, cfg.embedding.revision
    options: dict[str, object] = {
        "batch_size": cfg.embedding.batch_size,
        "max_batch_tokens": cfg.embedding.max_batch_tokens,
        "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
    }
    local_dir = resolve_local_model_dir(cfg.paths.data_root, model)
    if local_dir is not None:
        provider = _construct(
            spec,
            str(local_dir),
            None,
            cfg.embedding.dim,
            model_id=local_model_id(model, revision),
            **options,
        )
    elif _looks_like_local_model_dir(model):
        raise ProviderUnavailableError(
            f"embedding.model '{model}' names a directory under paths.data_root "
            f"('{cfg.paths.data_root}') that does not exist. A value under "
            f"'{LOCAL_MODEL_DIR_PREFIX}/' is a local conversion (models/manifest.lock.yaml "
            f"`output_dir`): produce it with the `command` recorded there, or set a Hugging "
            f"Face repo id ('owner/name') instead"
        )
    else:
        provider = _construct(spec, model, revision, cfg.embedding.dim, **options)
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


def build_boundary_provider(
    cfg: Config, prompt_template: str, *, shared_runtime: SharedVlmRuntime | None = None
) -> BoundaryProvider:
    """S4's topical-boundary LLM (SPEC §4.1/D4). `prompt_template` is
    the decoded `segmentation.boundary_prompt` file — the caller reads
    it (and hashes the raw bytes into `seg_config_hash`), so the same
    bytes drive both the hash and the model.

    `shared_runtime` (from `build_shared_vlm_runtime`) makes this
    provider generate through an already-loaded vision-language model
    instead of loading its own copy. Same weights, same computation —
    see `imsg.segment.mlx_boundaries` for the equivalence evidence."""
    if cfg.models.backend == "fake":
        return FakeBoundaryProvider()
    spec = REAL_PROVIDERS["boundary"]
    provider = _construct(
        spec,
        cfg.segmentation.boundary_model,
        cfg.segmentation.boundary_revision,
        prompt_template,
        shared_runtime=shared_runtime,
        cache_limit_bytes=cfg.models.query_cache_limit_bytes
        if shared_runtime is None
        else cfg.models.enrichment_cache_limit_bytes,
    )
    return cast("BoundaryProvider", provider)


def build_reranker(cfg: Config) -> RerankerProvider:
    """SPEC §9.4 step 7's reranker (the lock's active `reranker` entry,
    Qwen3-Reranker-0.6B). `retrieval.
    reranker_model` is a local conversion when `<paths.data_root>/<value>`
    is an existing directory — built with `revision=None` and `model_id`
    `<value>@<retrieval.reranker_revision>`, the upstream sha — and a Hub
    repo id pinned at `reranker_revision` otherwise. Either way the
    provider reads at most `retrieval.rerank_doc_max_tokens` tokens of
    each document."""
    if cfg.models.backend == "fake":
        return FakeRerankerProvider()
    spec = REAL_PROVIDERS["reranker"]
    model, revision = cfg.retrieval.reranker_model, cfg.retrieval.reranker_revision
    options: dict[str, object] = {
        "doc_max_tokens": cfg.retrieval.rerank_doc_max_tokens,
        "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
    }
    local_dir = resolve_local_model_dir(cfg.paths.data_root, model)
    if local_dir is not None:
        provider = _construct(
            spec, str(local_dir), None, model_id=local_model_id(model, revision), **options
        )
        return cast("RerankerProvider", provider)
    if _looks_like_local_model_dir(model):
        raise ProviderUnavailableError(
            f"retrieval.reranker_model '{model}' names a directory under paths.data_root "
            f"('{cfg.paths.data_root}') that does not exist. A value under "
            f"'{LOCAL_MODEL_DIR_PREFIX}/' is a local conversion (models/manifest.lock.yaml "
            f"`output_dir`): produce it with the `command` recorded there, or set a Hugging "
            f"Face repo id ('owner/name') instead"
        )
    provider = _construct(spec, model, revision, **options)
    return cast("RerankerProvider", provider)


def read_caption_prompt(cfg: Config) -> str:
    """The fixed captioning prompt (SPEC §4.1) as text — `enrichment.
    caption_prompt` under `paths.data_root`, else the repo-shipped copy
    (`resolve_caption_prompt`). Same convention as the boundary prompt;
    only the real backend needs it."""
    return read_prompt_text(
        resolve_caption_prompt(cfg).path, field_name="enrichment.caption_prompt"
    )


def build_enrichment_providers(
    cfg: Config,
    *,
    caption_prompt: str | None = None,
    shared_runtime: SharedVlmRuntime | None = None,
) -> EnrichmentProviders:
    """S5b's three model-backed steps (SPEC §4.1): Apple Vision OCR,
    the local captioning VLM, and Whisper transcription. `caption_prompt`
    may be supplied by a caller that already read it; otherwise the real
    backend reads `enrichment.caption_prompt` itself.

    `shared_runtime` (from `build_shared_vlm_runtime`) is where the
    captioner's weights come from, so a process that also builds the
    boundary provider holds one copy. Left `None`, the captioner makes
    its own runtime — bounded either way, which is also what bounds
    MLX's process-wide buffer cache for Whisper (D10.2)."""
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
        # Same bound as the captioner. MLX's cache limit is process-wide,
        # so whichever of the two runs first sets it — and a batch that is
        # all audio never loads the captioner at all (D10.2).
        cache_limit_bytes=cfg.models.enrichment_cache_limit_bytes,
    )
    prompt = caption_prompt if caption_prompt is not None else read_caption_prompt(cfg)
    caption = _construct(
        REAL_PROVIDERS["caption"],
        enrichment.caption_model,
        enrichment.caption_revision,
        prompt,
        shared_runtime=shared_runtime,
        cache_limit_bytes=cfg.models.enrichment_cache_limit_bytes,
        max_image_side=enrichment.caption_max_image_side,
    )
    return EnrichmentProviders(
        ocr=cast("OcrProvider", ocr),
        caption=cast("CaptionProvider", caption),
        transcription=cast("TranscriptionProvider", transcription),
    )


__all__ = [
    "INSTALL_HINT",
    "LOCAL_MODEL_DIR_PREFIX",
    "MODELS_EXTRA",
    "REAL_PROVIDERS",
    "RealProviderSpec",
    "ResolvedPrompt",
    "backend_status_line",
    "boundary_and_caption_share_weights",
    "build_boundary_provider",
    "build_enrichment_providers",
    "build_multimodal_provider",
    "build_reranker",
    "build_shared_vlm_runtime",
    "build_text_provider",
    "default_prompt_root",
    "local_model_id",
    "read_caption_prompt",
    "read_prompt_text",
    "resolve_caption_prompt",
    "resolve_local_model_dir",
    "resolve_prompt_path",
]
