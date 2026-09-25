"""`imsg.providers.factory`: backend selection (`models.backend`, default
`real`), lazy real-class resolution by dotted path, operator-facing
errors when a provider module or runtime package is missing, the
`models: backend=<...>` status line, and the config-defaults-equal-the-
manifest invariant.

No model weights, no network: the real provider classes are routed to
in-memory stub modules through a monkeypatched `importlib.import_module`
that delegates every other import to the real thing."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from imsg import constants
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.embed.pe_core_multimodal import resolve_weights_repo
from imsg.embed.provider import FakeMultimodalEmbeddingProvider, FakeTextEmbeddingProvider
from imsg.enrich.pipeline import EnrichmentProviders
from imsg.enrich.provider import FakeCaptionProvider, FakeOcrProvider, FakeTranscriptionProvider
from imsg.errors import ConfigError, ImsgError, ProviderUnavailableError
from imsg.providers import factory
from imsg.providers.factory import (
    REAL_PROVIDERS,
    RealProviderSpec,
    backend_status_line,
    boundary_and_caption_share_weights,
    build_boundary_provider,
    build_enrichment_providers,
    build_multimodal_provider,
    build_reranker,
    build_shared_vlm_runtime,
    build_text_provider,
    local_model_id,
    resolve_local_model_dir,
)
from imsg.providers.manifest import default_manifest_path, entries_by_role, load_manifest
from imsg.retrieval.reranker import FakeRerankerProvider
from imsg.segment.boundaries import FakeBoundaryProvider

pytestmark = pytest.mark.usefixtures("messages_dir")


def _config(config_dict_factory: Any, backend: str | None) -> Config:
    raw = config_dict_factory()
    if backend is None:
        del raw["models"]  # observe the schema default
    else:
        raw["models"]["backend"] = backend
    return load_config_dict(raw)


ConstructorCalls = dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]]


def _install_stub_modules(monkeypatch: pytest.MonkeyPatch, *, init: Any = None) -> ConstructorCalls:
    """Route every `REAL_PROVIDERS` module path to a stub module whose
    class records how it was constructed (and exposes `dim` when it
    was given one, like the real embedding providers). Everything else
    still goes through the real `importlib.import_module`."""
    calls: ConstructorCalls = {}
    modules: dict[str, types.ModuleType] = {}
    for spec in REAL_PROVIDERS.values():
        role_calls = calls.setdefault(spec.role, [])

        def _init(self: Any, *args: Any, _calls: Any = role_calls, **kwargs: Any) -> None:
            _calls.append((args, kwargs))
            if init is not None:
                init(self, *args, **kwargs)
            elif len(args) >= 3 and isinstance(args[2], int):
                self.dim = args[2]

        cls = type(spec.class_name, (), {"__init__": _init, "model_id": f"stub/{spec.role}"})
        module = types.ModuleType(spec.module)
        setattr(module, spec.class_name, cls)
        modules[spec.module] = module

    real_import = importlib.import_module

    def _import(name: str, package: str | None = None) -> Any:
        if name in modules:
            return modules[name]
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _import)
    return calls


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------


def test_backend_defaults_to_real(config_dict_factory: Any) -> None:
    cfg = _config(config_dict_factory, None)
    assert cfg.models.backend == "real"
    assert backend_status_line(cfg) == "models: backend=real"


def test_fake_backend_is_explicit_and_builds_the_deterministic_stand_ins(
    config_dict_factory: Any,
) -> None:
    cfg = _config(config_dict_factory, "fake")
    assert backend_status_line(cfg) == "models: backend=fake"

    text = build_text_provider(cfg)
    assert isinstance(text, FakeTextEmbeddingProvider)
    assert text.dim == cfg.embedding.dim == constants.PRIMARY_EMBEDDING_DIM

    multimodal = build_multimodal_provider(cfg)
    assert isinstance(multimodal, FakeMultimodalEmbeddingProvider)
    assert multimodal.dim == constants.MULTIMODAL_EMBEDDING_DIM

    assert isinstance(build_boundary_provider(cfg, "prompt"), FakeBoundaryProvider)
    assert isinstance(build_reranker(cfg), FakeRerankerProvider)

    enrichment = build_enrichment_providers(cfg)
    assert isinstance(enrichment, EnrichmentProviders)
    assert isinstance(enrichment.ocr, FakeOcrProvider)
    assert isinstance(enrichment.caption, FakeCaptionProvider)
    assert isinstance(enrichment.transcription, FakeTranscriptionProvider)


@pytest.mark.parametrize("backend", ["real", "fake"])
def test_multimodal_disabled_returns_none_without_touching_any_backend(
    config_dict_factory: Any, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(name: str, package: str | None = None) -> Any:
        raise AssertionError(f"no import expected, got {name}")

    monkeypatch.setattr(importlib, "import_module", _boom)
    raw = config_dict_factory()
    raw["models"]["backend"] = backend
    raw["embedding"]["multimodal"]["enabled"] = False
    cfg = load_config_dict(raw)
    assert build_multimodal_provider(cfg) is None


# --------------------------------------------------------------------------
# real backend: each builder resolves the right class by dotted path and
# passes the config through in the agreed constructor shape
# --------------------------------------------------------------------------


def test_real_backend_resolves_each_class_by_dotted_path(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _config(config_dict_factory, None)  # default backend == real
    e, mm, r, s, en = (
        cfg.embedding,
        cfg.embedding.multimodal,
        cfg.retrieval,
        cfg.segmentation,
        cfg.enrichment,
    )

    text = build_text_provider(cfg)
    assert type(text).__name__ == REAL_PROVIDERS["text_embedding"].class_name
    assert calls["text_embedding"] == [
        (
            (e.model, e.revision, e.dim),
            {
                "batch_size": e.batch_size,
                "max_batch_tokens": e.max_batch_tokens,
                "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
            },
        )
    ]

    multimodal = build_multimodal_provider(cfg)
    assert type(multimodal).__name__ == REAL_PROVIDERS["multimodal_embedding"].class_name
    assert calls["multimodal_embedding"] == [
        ((mm.model, mm.revision, mm.dim), {"batch_size": mm.batch_size})
    ]

    boundary = build_boundary_provider(cfg, "PROMPT TEMPLATE")
    assert type(boundary).__name__ == REAL_PROVIDERS["boundary"].class_name
    assert calls["boundary"] == [
        (
            (s.boundary_model, s.boundary_revision, "PROMPT TEMPLATE"),
            {
                "shared_runtime": None,
                "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
            },
        )
    ]

    reranker = build_reranker(cfg)
    assert type(reranker).__name__ == REAL_PROVIDERS["reranker"].class_name
    assert calls["reranker"] == [
        (
            (r.reranker_model, r.reranker_revision),
            {
                "doc_max_tokens": r.rerank_doc_max_tokens,
                "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
            },
        )
    ]

    providers = build_enrichment_providers(cfg, caption_prompt="CAPTION PROMPT")
    assert type(providers.ocr).__name__ == REAL_PROVIDERS["ocr"].class_name
    assert type(providers.caption).__name__ == REAL_PROVIDERS["caption"].class_name
    assert type(providers.transcription).__name__ == REAL_PROVIDERS["transcription"].class_name
    assert calls["ocr"] == [
        (
            (),
            {
                "recognition_languages": en.ocr_languages,
                "minimum_text_height": en.ocr_minimum_text_height,
                "max_image_pixels": en.limits.max_image_pixels,
            },
        )
    ]
    assert calls["transcription"] == [
        (
            (en.transcription_model, en.transcription_revision),
            {
                "language": en.transcription_language,
                "cache_limit_bytes": cfg.models.enrichment_cache_limit_bytes,
            },
        )
    ]
    assert calls["caption"] == [
        (
            (en.caption_model, en.caption_revision, "CAPTION PROMPT"),
            {
                "shared_runtime": None,
                "cache_limit_bytes": cfg.models.enrichment_cache_limit_bytes,
                "max_image_side": en.caption_max_image_side,
            },
        )
    ]


def test_real_backend_passes_operator_settings_through(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    raw = config_dict_factory()
    del raw["models"]
    raw["enrichment"]["ocr_languages"] = ["en-US", "fr-FR"]
    raw["enrichment"]["ocr_minimum_text_height"] = 0.05
    raw["enrichment"]["transcription_language"] = "en"
    raw["embedding"]["batch_size"] = 8
    raw["embedding"]["max_batch_tokens"] = 4096
    raw["embedding"]["multimodal"]["batch_size"] = 4
    cfg = load_config_dict(raw)

    build_text_provider(cfg)
    build_multimodal_provider(cfg)
    build_enrichment_providers(cfg, caption_prompt="p")
    assert calls["text_embedding"][0][1] == {
        "batch_size": 8,
        "max_batch_tokens": 4096,
        "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
    }
    assert calls["multimodal_embedding"][0][1] == {"batch_size": 4}
    assert calls["ocr"][0][1] == {
        "recognition_languages": ["en-US", "fr-FR"],
        "minimum_text_height": 0.05,
        "max_image_pixels": cfg.enrichment.limits.max_image_pixels,
    }
    assert calls["transcription"][0][1] == {
        "language": "en",
        "cache_limit_bytes": cfg.models.enrichment_cache_limit_bytes,
    }


def test_real_backend_passes_the_caption_image_bound_through(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    raw = config_dict_factory()
    del raw["models"]
    raw["enrichment"]["caption_max_image_side"] = None
    build_enrichment_providers(load_config_dict(raw), caption_prompt="p")
    assert calls["caption"][0][1]["max_image_side"] is None


def test_real_backend_reads_the_caption_prompt_from_data_root(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _config(config_dict_factory, None)
    prompt_path = cfg.paths.data_root / cfg.enrichment.caption_prompt
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("Describe the image.\n", encoding="utf-8")

    build_enrichment_providers(cfg)
    assert calls["caption"][0][0][2] == "Describe the image.\n"


def test_real_backend_missing_caption_prompt_is_one_clear_error(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_stub_modules(monkeypatch)
    monkeypatch.setattr(factory, "default_prompt_root", lambda: tmp_path / "no-checkout")
    cfg = _config(config_dict_factory, None)
    with pytest.raises(
        ProviderUnavailableError, match=r"caption prompt not found.*enrichment\.caption_prompt"
    ):
        build_enrichment_providers(cfg)


def test_caption_prompt_falls_back_to_the_repo_shipped_file(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh data_root has no prompts yet; the repository ships the
    canonical text at the same relative path, and that is what the real
    caption provider is built with — verbatim, so `prompt_sha256` is the
    shipped file's hash."""
    calls = _install_stub_modules(monkeypatch)
    cfg = _config(config_dict_factory, None)
    assert not (cfg.paths.data_root / cfg.enrichment.caption_prompt).exists()
    shipped = factory.default_prompt_root() / cfg.enrichment.caption_prompt
    assert shipped.is_file()

    resolved = factory.resolve_caption_prompt(cfg)
    assert (resolved.path, resolved.source) == (shipped, "repo")
    build_enrichment_providers(cfg)
    assert calls["caption"][0][0][2] == shipped.read_bytes().decode("utf-8")


def test_data_root_caption_prompt_wins_over_the_shipped_one(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_stub_modules(monkeypatch)
    cfg = _config(config_dict_factory, None)
    override = cfg.paths.data_root / cfg.enrichment.caption_prompt
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text("Operator override.\n", encoding="utf-8")
    resolved = factory.resolve_caption_prompt(cfg)
    assert (resolved.path, resolved.source) == (override, "data_root")
    assert factory.read_caption_prompt(cfg) == "Operator override.\n"


def test_non_utf8_prompt_is_one_clear_error_not_a_traceback(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_stub_modules(monkeypatch)
    cfg = _config(config_dict_factory, None)
    bad = cfg.paths.data_root / cfg.enrichment.caption_prompt
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(ProviderUnavailableError, match="not valid UTF-8"):
        build_enrichment_providers(cfg)


def test_shipped_boundary_prompt_matches_the_parser_contract() -> None:
    """The repo-shipped `prompts/segment_boundaries.txt` must ask for the
    shape `MlxBoundaryProvider` parses and the range it validates: a
    `{"boundaries": [...]}` object of indices where a NEW segment starts,
    never 0, all inside the window."""
    from imsg.config.schema import SegmentationConfig
    from imsg.segment.mlx_boundaries import parse_boundary_response, validate_boundaries

    relative = SegmentationConfig().boundary_prompt
    text = (factory.default_prompt_root() / relative).read_text(encoding="utf-8")
    assert '{"boundaries": [' in text
    assert "Never return 0" in text
    assert "FIRST message of a new topic" in text
    # The literal answer shape the prompt requests round-trips through the parser.
    assert validate_boundaries(parse_boundary_response('{"boundaries": [2, 5]}'), 8) == [2, 5]
    assert parse_boundary_response('{"boundaries": []}') == []


def test_fake_backend_never_needs_the_caption_prompt(config_dict_factory: Any) -> None:
    cfg = _config(config_dict_factory, "fake")
    assert not (cfg.paths.data_root / cfg.enrichment.caption_prompt).exists()
    assert isinstance(build_enrichment_providers(cfg).caption, FakeCaptionProvider)


# --------------------------------------------------------------------------
# failure modes: one operator-actionable ImsgError, never a traceback
# --------------------------------------------------------------------------


def test_missing_runtime_package_names_the_models_extra(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = REAL_PROVIDERS["text_embedding"]

    def _import(name: str, package: str | None = None) -> Any:
        if name == spec.module:
            # What a present provider module raises when `import mlx` fails inside it.
            raise ModuleNotFoundError("No module named 'mlx'", name="mlx")
        return importlib.__import__(name)

    monkeypatch.setattr(importlib, "import_module", _import)
    cfg = _config(config_dict_factory, None)
    with pytest.raises(ProviderUnavailableError) as excinfo:
        build_text_provider(cfg)
    message = str(excinfo.value)
    assert "'mlx'" in message
    assert "uv sync --extra models" in message
    assert "text_embedding" in message
    assert isinstance(excinfo.value, ImsgError)


def test_missing_provider_module_in_this_build_is_one_clear_error(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unstubbed: a spec pointing at a module that does not exist in any
    build — the shape of running `real` before a provider branch is
    merged. Robust to the real provider modules existing later."""
    absent = RealProviderSpec(
        "reranker", "imsg.retrieval.absent_for_this_test", "MlxRerankerProvider"
    )
    monkeypatch.setitem(REAL_PROVIDERS, "reranker", absent)
    cfg = _config(config_dict_factory, None)
    with pytest.raises(ProviderUnavailableError) as excinfo:
        build_reranker(cfg)
    message = str(excinfo.value)
    assert "this build has no real 'reranker' provider" in message
    assert absent.module in message
    assert "models.backend: fake" in message


def test_constructor_failure_is_wrapped_with_the_cause(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _explode(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("revision deadbeef not found in repo")

    _install_stub_modules(monkeypatch, init=_explode)
    cfg = _config(config_dict_factory, None)
    with pytest.raises(ProviderUnavailableError) as excinfo:
        build_reranker(cfg)
    assert "could not construct the real 'reranker' provider" in str(excinfo.value)
    assert "RuntimeError: revision deadbeef not found in repo" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_provider_raising_imsg_error_passes_through_unwrapped(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Custom(ImsgError):
        pass

    def _raise(self: Any, *args: Any, **kwargs: Any) -> None:
        raise _Custom("provider-specific, already operator-facing")

    _install_stub_modules(monkeypatch, init=_raise)
    cfg = _config(config_dict_factory, None)
    with pytest.raises(_Custom):
        build_reranker(cfg)


def test_embedding_provider_with_the_wrong_dim_is_rejected(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _wrong_dim(self: Any, *args: Any, **kwargs: Any) -> None:
        self.dim = 7

    _install_stub_modules(monkeypatch, init=_wrong_dim)
    cfg = _config(config_dict_factory, None)
    with pytest.raises(ProviderUnavailableError, match=r"dim=7.*requires 2048"):
        build_text_provider(cfg)


# --------------------------------------------------------------------------
# invariants: lazy imports, and config defaults == the manifest lock
# --------------------------------------------------------------------------


def test_real_provider_modules_are_not_imported_at_factory_import_time() -> None:
    """A fresh interpreter: importing the factory (and the CLI, which
    imports it) must not import any real provider module — the `models`
    extra is optional for every `fake`-backend command."""
    src_dir = Path(factory.__file__).resolve().parents[2]
    modules = [spec.module for spec in REAL_PROVIDERS.values()]
    code = (
        "import sys, imsg.providers.factory, imsg.cli\n"
        f"loaded = [m for m in {modules!r} if m in sys.modules]\n"
        "print('LOADED=' + ','.join(loaded))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_dir) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "LOADED=\n" in result.stdout or result.stdout.strip() == "LOADED="


def test_config_defaults_mirror_the_manifest_lock(config_dict_factory: Any) -> None:
    """`models/manifest.lock.yaml` is the source of truth for every pin;
    the schema defaults (via `imsg.constants`) must equal it exactly, so
    an operator who omits a repo/revision gets precisely the locked
    model — never a silently different one."""
    lock = load_manifest(default_manifest_path())
    by_role = entries_by_role(lock)

    raw = config_dict_factory()
    for section, key in [
        ("embedding", "model"),
        ("segmentation", "boundary_model"),
        ("retrieval", "reranker_model"),
    ]:
        del raw[section][key]
    del raw["embedding"]["multimodal"]["model"]
    cfg = load_config_dict(raw)

    text = by_role["text_embedding"]
    assert (cfg.embedding.model, constants.TEXT_EMBEDDING_MODEL_REVISION) == (
        text.repo,
        text.revision,
    )
    assert text.expected_dim == cfg.embedding.dim == constants.PRIMARY_EMBEDDING_DIM

    reranker = by_role["reranker"]
    assert (cfg.retrieval.reranker_model, constants.RERANKER_MODEL_REVISION) == (
        reranker.config_model,
        reranker.config_revision,
    )
    # The reranker is pinned as a local conversion (owner decision
    # 2026-09-15): the config names its data-root-relative output_dir and
    # the UPSTREAM commit the conversion derives from, never a Hub repo.
    assert reranker.local_conversion, "the reranker pin is expected to be a local conversion"
    assert reranker.output_dir == constants.RERANKER_MODEL
    assert reranker.upstream_revision == constants.RERANKER_MODEL_REVISION
    assert constants.RERANKER_MODEL.startswith("models/")

    boundary = by_role["segment_boundaries"]
    assert (cfg.segmentation.boundary_model, cfg.segmentation.boundary_revision) == (
        boundary.repo,
        boundary.revision,
    )

    caption = by_role["image_caption"]
    assert (cfg.enrichment.caption_model, cfg.enrichment.caption_revision) == (
        caption.repo,
        caption.revision,
    )
    assert caption is boundary  # one set of weights for both roles (SPEC §4.1)

    transcription = by_role["transcription"]
    assert (cfg.enrichment.transcription_model, cfg.enrichment.transcription_revision) == (
        transcription.repo,
        transcription.revision,
    )

    multimodal = by_role["multimodal_embedding"]
    assert (cfg.embedding.multimodal.model, constants.MULTIMODAL_EMBEDDING_MODEL_REVISION) == (
        multimodal.repo,
        multimodal.revision,
    )
    assert (
        multimodal.expected_dim
        == cfg.embedding.multimodal.dim
        == constants.MULTIMODAL_EMBEDDING_DIM
    )
    # A revision is a commit of exactly one repo. The PE-Core provider
    # downloads from `resolve_weights_repo(model)` — the open_clip-layout
    # mirror when given Meta's canonical id — so the lock must pin THAT
    # repo and one of its commits. Pinned as the canonical id, the mirror
    # was asked for a sha it does not have (2026-09-14).
    assert resolve_weights_repo(multimodal.repo) == multimodal.repo, (
        f"the multimodal pin names {multimodal.repo!r}, but weights are fetched from "
        f"{resolve_weights_repo(multimodal.repo)!r} — pin the repo the bytes come from"
    )

    # The query side's text tower is a local conversion cut from exactly
    # the multimodal pin (imsg.embed.pe_core_text_tower): the config names
    # its output_dir, and it has no revision of its own to drift.
    text_tower = by_role["multimodal_text_embedding"]
    assert text_tower.local_conversion
    assert cfg.embedding.multimodal.text_tower_model == text_tower.output_dir
    assert text_tower.output_dir == constants.MULTIMODAL_TEXT_TOWER_MODEL
    assert (text_tower.upstream_repo, text_tower.upstream_revision) == (
        multimodal.repo,
        multimodal.revision,
    )
    assert text_tower.expected_dim == constants.MULTIMODAL_EMBEDDING_DIM

    assert by_role["ocr"].status == "system"
    assert set(REAL_PROVIDERS) == {
        "text_embedding",
        "multimodal_embedding",
        "boundary",
        "reranker",
        "ocr",
        "transcription",
        "caption",
    }


# --------------------------------------------------------------------------
# the reranker as a local conversion: a directory under paths.data_root
# --------------------------------------------------------------------------

UPSTREAM_SHA = "3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c"
LOCAL_DIR = "models/example-reranker-mxfp8-3c3c3c3c"


def _local_reranker_config(config_dict_factory: Any, value: str) -> Config:
    raw = config_dict_factory()
    del raw["models"]  # real backend
    raw["retrieval"]["reranker_model"] = value
    raw["retrieval"]["reranker_revision"] = UPSTREAM_SHA
    return load_config_dict(raw)


def test_reranker_directory_under_data_root_builds_with_revision_none_and_a_relative_model_id(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _local_reranker_config(config_dict_factory, LOCAL_DIR + "/")
    directory = cfg.paths.data_root / LOCAL_DIR
    directory.mkdir(parents=True)

    provider = build_reranker(cfg)
    assert type(provider).__name__ == REAL_PROVIDERS["reranker"].class_name
    # The provider gets the resolved directory and no revision; the id it
    # records is the config's relative path plus the upstream sha — never
    # the absolute path.
    assert calls["reranker"] == [
        (
            (str(directory.resolve()), None),
            {
                "model_id": f"{LOCAL_DIR}@{UPSTREAM_SHA}",
                "doc_max_tokens": cfg.retrieval.rerank_doc_max_tokens,
                "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
            },
        )
    ]


def test_reranker_document_cap_flows_from_config_to_the_provider(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    for value in (256, None):
        raw = config_dict_factory()
        del raw["models"]  # real backend
        raw["retrieval"]["reranker_model"] = "example-org/Example-Reranker-8bit"
        raw["retrieval"]["reranker_revision"] = UPSTREAM_SHA
        raw["retrieval"]["rerank_doc_max_tokens"] = value
        build_reranker(load_config_dict(raw))
    assert [kwargs["doc_max_tokens"] for _, kwargs in calls["reranker"]] == [256, None]


def test_reranker_repo_id_stays_a_hub_pin_when_no_such_directory_exists(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _local_reranker_config(config_dict_factory, "example-org/Example-Reranker-8bit")
    assert not (cfg.paths.data_root / "example-org").exists()
    build_reranker(cfg)
    assert calls["reranker"] == [
        (
            ("example-org/Example-Reranker-8bit", UPSTREAM_SHA),
            {
                "doc_max_tokens": cfg.retrieval.rerank_doc_max_tokens,
                "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
            },
        )
    ]


def test_reranker_missing_models_directory_is_one_clear_error_not_a_hub_lookup(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _local_reranker_config(config_dict_factory, "models/not-converted-yet")
    with pytest.raises(ProviderUnavailableError) as excinfo:
        build_reranker(cfg)
    message = str(excinfo.value)
    assert "'models/not-converted-yet' names a directory under paths.data_root" in message
    assert "does not exist" in message
    assert "models/manifest.lock.yaml" in message and "command" in message
    assert "Hugging Face repo id" in message
    assert calls["reranker"] == []  # never handed to the Hub


def test_resolve_local_model_dir_follows_the_containment_rule(tmp_path: Path) -> None:
    data_root = tmp_path / "root"
    (data_root / "models" / "present").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (data_root / "models" / "escape").symlink_to(outside)

    assert resolve_local_model_dir(data_root, "models/present") == (
        data_root / "models" / "present"
    ).resolve()
    assert resolve_local_model_dir(data_root, "models/absent") is None
    assert resolve_local_model_dir(data_root, "example-org/repo") is None
    with pytest.raises(ProviderUnavailableError, match=r"outside paths\.data_root"):
        resolve_local_model_dir(data_root, "models/escape")


def test_local_model_id_is_the_normalised_relative_dir_at_the_upstream_sha() -> None:
    assert local_model_id("models/x/", UPSTREAM_SHA) == f"models/x@{UPSTREAM_SHA}"
    assert local_model_id("models//y", UPSTREAM_SHA) == f"models/y@{UPSTREAM_SHA}"


# --------------------------------------------------------------------------
# the text embedder as a local conversion: the reranker's rule, applied
# to embedding.model (a directory under paths.data_root)
# --------------------------------------------------------------------------

EMBED_LOCAL_DIR = "models/example-embedding-8bit-3c3c3c3c"


def _local_embedding_config(config_dict_factory: Any, value: str) -> Config:
    raw = config_dict_factory()
    del raw["models"]  # real backend
    raw["embedding"]["model"] = value
    raw["embedding"]["revision"] = UPSTREAM_SHA
    return load_config_dict(raw)


def test_embedding_directory_under_data_root_builds_with_revision_none_and_a_relative_model_id(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _local_embedding_config(config_dict_factory, EMBED_LOCAL_DIR)
    directory = cfg.paths.data_root / EMBED_LOCAL_DIR
    directory.mkdir(parents=True)

    provider = build_text_provider(cfg)
    assert type(provider).__name__ == REAL_PROVIDERS["text_embedding"].class_name
    assert calls["text_embedding"] == [
        (
            (str(directory.resolve()), None, cfg.embedding.dim),
            {
                "model_id": f"{EMBED_LOCAL_DIR}@{UPSTREAM_SHA}",
                "batch_size": cfg.embedding.batch_size,
                "max_batch_tokens": cfg.embedding.max_batch_tokens,
                "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
            },
        )
    ]


def test_embedding_repo_id_stays_a_hub_pin_when_no_such_directory_exists(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _local_embedding_config(config_dict_factory, "example-org/Example-Embedding-8bit")
    build_text_provider(cfg)
    assert calls["text_embedding"] == [
        (
            ("example-org/Example-Embedding-8bit", UPSTREAM_SHA, cfg.embedding.dim),
            {
                "batch_size": cfg.embedding.batch_size,
                "max_batch_tokens": cfg.embedding.max_batch_tokens,
                "cache_limit_bytes": cfg.models.query_cache_limit_bytes,
            },
        )
    ]


def test_embedding_missing_models_directory_is_one_clear_error_not_a_hub_lookup(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _local_embedding_config(config_dict_factory, "models/not-converted-yet")
    with pytest.raises(ProviderUnavailableError) as excinfo:
        build_text_provider(cfg)
    message = str(excinfo.value)
    assert "embedding.model 'models/not-converted-yet' names a directory under paths.data_root" in message
    assert "does not exist" in message
    assert "models/manifest.lock.yaml" in message and "command" in message
    assert calls["text_embedding"] == []


# --------------------------------------------------------------------------
# one copy of the shared 35B (D10.3 defect 1)
# --------------------------------------------------------------------------


def _config_with_shipped_pins(config_dict_factory: Any) -> Config:
    """The real backend with both model fields left at their schema
    defaults — the conftest fixture overrides `boundary_model` with a
    placeholder, which would hide the fact that the shipped pins name one
    checkpoint for both roles."""
    raw = config_dict_factory()
    del raw["models"]
    del raw["segmentation"]["boundary_model"]
    return load_config_dict(raw)


def test_the_pins_ship_naming_the_same_checkpoint_for_both_roles(
    config_dict_factory: Any,
) -> None:
    """Sharing is only worth anything because the defaults really do
    point both roles at one model — `imsg.constants` sets
    `CAPTION_MODEL_REPO = BOUNDARY_MODEL_REPO`, and the manifest records
    one entry with both roles."""
    cfg = _config_with_shipped_pins(config_dict_factory)
    assert cfg.segmentation.boundary_model == cfg.enrichment.caption_model
    assert cfg.segmentation.boundary_revision == cfg.enrichment.caption_revision
    assert boundary_and_caption_share_weights(cfg) is True


def test_a_shared_runtime_is_built_for_the_real_backend(config_dict_factory: Any) -> None:
    cfg = _config_with_shipped_pins(config_dict_factory)
    runtime = build_shared_vlm_runtime(cfg)
    assert runtime is not None
    assert runtime.loaded_model_ids == ()  # nothing loads until a provider asks
    assert runtime.cache_limit_bytes == cfg.models.enrichment_cache_limit_bytes


def test_no_shared_runtime_for_the_fake_backend(config_dict_factory: Any) -> None:
    assert build_shared_vlm_runtime(_config(config_dict_factory, "fake")) is None


def test_no_shared_runtime_when_the_operator_turns_sharing_off(
    config_dict_factory: Any,
) -> None:
    raw = config_dict_factory()
    del raw["models"]["backend"]
    del raw["segmentation"]["boundary_model"]
    raw["models"]["share_boundary_and_caption_weights"] = False
    cfg = load_config_dict(raw)
    assert boundary_and_caption_share_weights(cfg) is False
    assert build_shared_vlm_runtime(cfg) is None


def test_no_shared_runtime_when_the_two_roles_name_different_pins(
    config_dict_factory: Any,
) -> None:
    """Nothing to share, and pretending otherwise would silently run
    boundary detection on the captioning model's weights."""
    raw = config_dict_factory()
    del raw["models"]
    raw["segmentation"]["boundary_model"] = "example-org/Boundary-Model-4bit"
    raw["segmentation"]["boundary_revision"] = UPSTREAM_SHA
    cfg = load_config_dict(raw)
    assert boundary_and_caption_share_weights(cfg) is False
    assert build_shared_vlm_runtime(cfg) is None


def test_a_shared_runtime_reaches_both_builders(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _config_with_shipped_pins(config_dict_factory)
    runtime = build_shared_vlm_runtime(cfg)

    build_boundary_provider(cfg, "PROMPT TEMPLATE", shared_runtime=runtime)
    build_enrichment_providers(cfg, caption_prompt="p", shared_runtime=runtime)

    assert calls["boundary"][0][1]["shared_runtime"] is runtime
    assert calls["caption"][0][1]["shared_runtime"] is runtime
    # A shared boundary provider is an enrichment-side consumer, so it
    # gets the enrichment bound rather than the query-side one.
    assert calls["boundary"][0][1]["cache_limit_bytes"] == cfg.models.enrichment_cache_limit_bytes
    assert calls["caption"][0][1]["cache_limit_bytes"] == cfg.models.enrichment_cache_limit_bytes


def test_an_unshared_boundary_provider_gets_the_query_side_bound(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`imsg segment` and `imsg sync` build boundary detection alone, in
    a process that also holds query-shaped models."""
    calls = _install_stub_modules(monkeypatch)
    cfg = _config(config_dict_factory, None)
    build_boundary_provider(cfg, "PROMPT TEMPLATE")
    assert calls["boundary"][0][1]["shared_runtime"] is None
    assert calls["boundary"][0][1]["cache_limit_bytes"] == cfg.models.query_cache_limit_bytes


# --------------------------------------------------------------------------
# the PE-Core text-tower checkpoint: handed to the provider when present
# --------------------------------------------------------------------------


def _real_config(config_dict_factory: Any, **multimodal: Any) -> Config:
    raw = config_dict_factory()
    del raw["models"]  # real backend
    raw["embedding"]["multimodal"].update(multimodal)
    return load_config_dict(raw)


def test_multimodal_provider_is_given_the_text_tower_checkpoint_when_it_exists(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_stub_modules(monkeypatch)
    cfg = _real_config(config_dict_factory)
    directory = cfg.paths.data_root / constants.MULTIMODAL_TEXT_TOWER_MODEL
    directory.mkdir(parents=True)
    build_multimodal_provider(cfg)
    [(_, kwargs)] = calls["multimodal_embedding"]
    assert kwargs == {"batch_size": cfg.embedding.multimodal.batch_size, "text_tower_dir": directory.resolve()}


def test_missing_text_tower_checkpoint_warns_and_builds_the_whole_model_provider(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host that has not run the conversion keeps serving: same vectors,
    slower load. The warning names the fix."""
    from structlog.testing import capture_logs

    calls = _install_stub_modules(monkeypatch)
    cfg = _real_config(config_dict_factory)
    with capture_logs() as logs:
        build_multimodal_provider(cfg)
    [(_, kwargs)] = calls["multimodal_embedding"]
    assert "text_tower_dir" not in kwargs
    [warning] = [e for e in logs if e["event"] == "pe_core.text_tower_checkpoint_missing"]
    assert warning["log_level"] == "warning"
    assert "convert_pe_core_text_tower.py" in warning["fix"]


def test_null_text_tower_model_always_builds_the_whole_model_quietly(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from structlog.testing import capture_logs

    calls = _install_stub_modules(monkeypatch)
    cfg = _real_config(config_dict_factory, text_tower_model=None)
    with capture_logs() as logs:
        build_multimodal_provider(cfg)
    [(_, kwargs)] = calls["multimodal_embedding"]
    assert "text_tower_dir" not in kwargs
    assert not [e for e in logs if e["event"] == "pe_core.text_tower_checkpoint_missing"]


def test_text_tower_directory_symlinked_outside_data_root_is_refused(
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_stub_modules(monkeypatch)
    cfg = _real_config(config_dict_factory, text_tower_model="models/escape")
    outside = tmp_path / "outside-data-root"
    outside.mkdir()
    (cfg.paths.data_root / "models").mkdir(parents=True, exist_ok=True)
    (cfg.paths.data_root / "models" / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProviderUnavailableError, match=r"outside paths\.data_root"):
        build_multimodal_provider(cfg)


@pytest.mark.parametrize("value", ["/abs/text-tower", "~/text-tower", "models/../../escape", "  "])
def test_text_tower_model_must_be_a_data_root_relative_directory(
    config_dict_factory: Any, value: str
) -> None:
    raw = config_dict_factory()
    raw["embedding"]["multimodal"]["text_tower_model"] = value
    with pytest.raises(ConfigError, match="text_tower_model"):
        load_config_dict(raw)
