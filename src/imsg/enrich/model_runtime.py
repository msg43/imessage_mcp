"""Shared plumbing for the real, model-backed enrichment providers
(`vision_ocr`, `mlx_whisper_transcription`, `mlx_vlm_caption`): lazy
runtime imports that fail with one clear, typed error, and Hugging Face
snapshot pinning for the MLX-hosted models.

None of the runtimes — pyobjc's `Vision` bridge, `mlx_whisper`,
`mlx_vlm`, their `huggingface_hub` dependency — are importable in the
build/CI environment, and constructing a provider must stay cheap and
side-effect free: `imsg enrich --dry-run` and every test construct
providers without loading a model. So each provider imports its runtime
on first use, inside the method that needs it, through
`import_runtime_module`, and the tests stand a fake module in
`sys.modules` instead of installing anything.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from imsg.errors import ImsgError


class ModelRuntimeUnavailableError(ImsgError):
    """A model runtime is not importable on this host, or pinned model
    weights could not be resolved or loaded.

    Deliberately a direct `ImsgError` subclass, *not* an
    `EnrichmentError`: `imsg.enrich.pipeline.process_one_task` records an
    `EnrichmentError` as a per-task transient failure and retries it with
    backoff, which for a missing runtime would burn every queued task's
    retry budget one task at a time and finally mark them all `failed`.
    A missing runtime is an environment problem, not a property of the
    attachment, so it must abort the run loudly instead: this propagates
    past the per-task handler to the CLI boundary, which prints the
    install hint and exits nonzero. The one claimed task's lease simply
    expires back into the pool (`imsg.enrich.queue.claim_tasks`).
    """


def import_runtime_module(module_name: str, *, install_hint: str) -> ModuleType:
    """Import `module_name` now (never at package import time), turning
    an `ImportError` into a `ModelRuntimeUnavailableError` that names the
    runtime and says how to install it."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ModelRuntimeUnavailableError(
            f"the '{module_name}' runtime could not be imported on this host "
            f"({exc}) — {install_hint}"
        ) from exc


def resolve_model_snapshot(model_repo: str, revision: str | None) -> str:
    """What to hand an MLX runtime as its model path.

    `model_repo` is either a local directory or a Hugging Face repo id:

    - a local directory is passed through untouched (`revision` is then
      informational — it still appears in the provider's `model_id`, but
      nothing verifies the directory's contents against it);
    - a repo id with no `revision` is passed through, and the runtime
      resolves the Hub's `main` itself, exactly as it would from the
      command line;
    - a repo id with a `revision` is pinned here, through
      `huggingface_hub.snapshot_download(repo_id=..., revision=...)`,
      because `mlx_whisper` offers no way to pass a revision through (its
      own loader always fetches `main`), and the local snapshot directory
      that returns is what the runtime receives. `snapshot_download`
      falls back to an already-cached snapshot when the Hub is
      unreachable, so a pre-downloaded model works offline.

    A pinned revision that cannot be resolved is a
    `ModelRuntimeUnavailableError` (an environment problem — see that
    class), never a per-task failure.
    """
    if revision is None or Path(model_repo).is_dir():
        return model_repo
    hub = import_runtime_module(
        "huggingface_hub", install_hint="it is installed alongside mlx-whisper and mlx-vlm"
    )
    try:
        snapshot = hub.snapshot_download(repo_id=model_repo, revision=revision)
    except Exception as exc:
        raise ModelRuntimeUnavailableError(
            f"could not resolve the model snapshot for {model_repo}@{revision}: {exc}"
        ) from exc
    return str(snapshot)


__all__ = ["ModelRuntimeUnavailableError", "import_runtime_module", "resolve_model_snapshot"]
