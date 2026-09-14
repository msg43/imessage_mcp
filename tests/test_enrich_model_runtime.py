"""Shared plumbing for the real enrichment providers
(`imsg.enrich.model_runtime`): lazy runtime imports that fail with one
typed error, and Hugging Face snapshot pinning."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from _model_runtime_stubs import block_module, install_hub_stub
from imsg.enrich.model_runtime import (
    ModelRuntimeUnavailableError,
    import_runtime_module,
    resolve_model_snapshot,
)
from imsg.errors import EnrichmentError, ImsgError


def test_import_runtime_module_returns_the_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("fake_runtime_for_test")
    monkeypatch.setitem(sys.modules, "fake_runtime_for_test", fake)
    assert import_runtime_module("fake_runtime_for_test", install_hint="n/a") is fake


def test_missing_runtime_is_a_clear_imsg_error_naming_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_module(monkeypatch, "fake_runtime_for_test")
    with pytest.raises(ModelRuntimeUnavailableError) as excinfo:
        import_runtime_module("fake_runtime_for_test", install_hint="install `fake-runtime`")
    message = str(excinfo.value)
    assert "fake_runtime_for_test" in message
    assert "install `fake-runtime`" in message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_runtime_unavailable_is_not_a_per_task_enrichment_error() -> None:
    # process_one_task records EnrichmentError as a retryable per-task
    # failure; a missing runtime must escape that handler and abort the run.
    assert issubclass(ModelRuntimeUnavailableError, ImsgError)
    assert not issubclass(ModelRuntimeUnavailableError, EnrichmentError)


def test_unpinned_repo_id_is_passed_through_without_touching_the_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_module(monkeypatch, "huggingface_hub")
    assert resolve_model_snapshot("example-org/whisper-mlx", None) == "example-org/whisper-mlx"


def test_local_directory_is_passed_through_even_with_a_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    block_module(monkeypatch, "huggingface_hub")
    local = tmp_path / "weights"
    local.mkdir()
    assert resolve_model_snapshot(str(local), "abc123") == str(local)


def test_pinned_revision_resolves_the_local_snapshot_via_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = install_hub_stub(monkeypatch, tmp_path / "snapshot")
    resolved = resolve_model_snapshot("example-org/whisper-mlx", "abc123")
    assert resolved == str(tmp_path / "snapshot")
    assert hub.calls == [{"repo_id": "example-org/whisper-mlx", "revision": "abc123"}]


def test_snapshot_failure_is_runtime_unavailable_not_a_task_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hub = install_hub_stub(monkeypatch, tmp_path)
    hub.exception = OSError("no network and nothing cached")
    with pytest.raises(ModelRuntimeUnavailableError) as excinfo:
        resolve_model_snapshot("example-org/whisper-mlx", "abc123")
    assert "example-org/whisper-mlx@abc123" in str(excinfo.value)
    assert "no network and nothing cached" in str(excinfo.value)


def test_pinned_revision_without_huggingface_hub_is_runtime_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_module(monkeypatch, "huggingface_hub")
    with pytest.raises(ModelRuntimeUnavailableError) as excinfo:
        resolve_model_snapshot("example-org/whisper-mlx", "abc123")
    assert "huggingface_hub" in str(excinfo.value)
