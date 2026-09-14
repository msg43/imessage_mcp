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
from imsg.errors import ImsgError, ProviderUnavailableError
from imsg.providers import factory
from imsg.providers.factory import (
    REAL_PROVIDERS,
    RealProviderSpec,
    backend_status_line,
    build_boundary_provider,
    build_enrichment_providers,
    build_multimodal_provider,
    build_reranker,
    build_text_provider,
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
    assert calls["text_embedding"] == [((e.model, e.revision, e.dim), {"batch_size": e.batch_size})]

    multimodal = build_multimodal_provider(cfg)
    assert type(multimodal).__name__ == REAL_PROVIDERS["multimodal_embedding"].class_name
    assert calls["multimodal_embedding"] == [
        ((mm.model, mm.revision, mm.dim), {"batch_size": mm.batch_size})
    ]

    boundary = build_boundary_provider(cfg, "PROMPT TEMPLATE")
    assert type(boundary).__name__ == REAL_PROVIDERS["boundary"].class_name
    assert calls["boundary"] == [((s.boundary_model, s.boundary_revision, "PROMPT TEMPLATE"), {})]

    reranker = build_reranker(cfg)
    assert type(reranker).__name__ == REAL_PROVIDERS["reranker"].class_name
    assert calls["reranker"] == [((r.reranker_model, r.reranker_revision), {})]

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
            },
        )
    ]
    assert calls["transcription"] == [
        (
            (en.transcription_model, en.transcription_revision),
            {"language": en.transcription_language},
        )
    ]
    assert calls["caption"] == [((en.caption_model, en.caption_revision, "CAPTION PROMPT"), {})]


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
    raw["embedding"]["multimodal"]["batch_size"] = 4
    cfg = load_config_dict(raw)

    build_text_provider(cfg)
    build_multimodal_provider(cfg)
    build_enrichment_providers(cfg, caption_prompt="p")
    assert calls["text_embedding"][0][1] == {"batch_size": 8}
    assert calls["multimodal_embedding"][0][1] == {"batch_size": 4}
    assert calls["ocr"][0][1] == {
        "recognition_languages": ["en-US", "fr-FR"],
        "minimum_text_height": 0.05,
    }
    assert calls["transcription"][0][1] == {"language": "en"}


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
    config_dict_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_stub_modules(monkeypatch)
    cfg = _config(config_dict_factory, None)
    with pytest.raises(
        ProviderUnavailableError, match=r"caption prompt not found.*enrichment\.caption_prompt"
    ):
        build_enrichment_providers(cfg)


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
        reranker.repo,
        reranker.revision,
    )

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
