"""Shared `sys.modules` stand-ins for the model runtimes the real
enrichment providers import lazily (`imsg.enrich.model_runtime`).

None of `Vision`, `Foundation`, `mlx_whisper`, `mlx_vlm`, or
`huggingface_hub` may be required by the test suite — these helpers let
each provider test either install a fake module or block the real one,
with `monkeypatch` undoing both.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


def block_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make `import <name>` raise ImportError for the test's duration —
    `None` in `sys.modules` is the interpreter's own "import halted"
    marker — even if the real package happens to be installed."""
    monkeypatch.setitem(sys.modules, name, None)


@dataclass
class HubStub:
    """Records `huggingface_hub.snapshot_download` calls and answers
    them with `snapshot_dir` (or raises `exception`)."""

    snapshot_dir: Path
    exception: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)


def install_hub_stub(monkeypatch: pytest.MonkeyPatch, snapshot_dir: Path) -> HubStub:
    stub = HubStub(snapshot_dir=snapshot_dir)

    def snapshot_download(**kwargs: Any) -> str:
        stub.calls.append(kwargs)
        if stub.exception is not None:
            raise stub.exception
        return str(stub.snapshot_dir)

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return stub
