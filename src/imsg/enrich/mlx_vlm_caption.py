"""Image captioning through `mlx_vlm` (SPEC §4.1: a local VLM, the fixed
prompt in `prompts/caption.txt`, temperature 0) behind the
`CaptionProvider` Protocol.

The prompt is fixed, not per-image: it is read once at wiring time from
`<data_root>/prompts/caption.txt` (`DEFAULT_CAPTION_PROMPT_PATH` — the
same data-root-relative convention as `segmentation.boundary_prompt`;
the repository ships the canonical text at that same relative path) and
passed in verbatim. Its SHA-256 is exposed as `prompt_sha256`, next to
`model_id`, so enrichment provenance can record exactly which prompt
produced a caption — the role `boundary_prompt_sha256` plays in
`imsg.segment.hashing.compute_seg_config_hash`.

Decoding is greedy (`CAPTION_TEMPERATURE`, 0.0) and capped at
`max_tokens`, so a caption is a deterministic function of (weights,
prompt, image). The output is flattened to one stripped paragraph; a
`<think>…</think>` block, which thinking-capable models can emit ahead
of the answer, is removed rather than indexed. An empty caption is a
failure, not a result — a model that produced nothing for an image has
not captioned it.

Model weights load on the first call and stay loaded for the provider's
lifetime (one queue worker captions thousands of images against one
multi-gigabyte model). The runtime is imported on first use (see
`imsg.enrich.model_runtime`); constructing the provider needs nothing
installed.

Loading goes through an `imsg.shared_vlm_runtime.SharedVlmRuntime`, so a
process that also runs boundary detection against the same pin holds one
copy of the weights rather than two (D10.3 defect 1). Passed no runtime,
the provider makes its own private one and behaves exactly as before.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from imsg.enrich.model_runtime import import_runtime_module
from imsg.errors import ConfigError, EnrichmentError
from imsg.hashing import sha256_text
from imsg.shared_vlm_runtime import (
    DEFAULT_ENRICHMENT_CACHE_LIMIT_BYTES,
    MLX_VLM_INSTALL_HINT,
    LoadedVlm,
    SharedVlmRuntime,
)
from imsg.textnorm import strip_nul

_INSTALL_HINT = MLX_VLM_INSTALL_HINT

DEFAULT_CAPTION_PROMPT_PATH = Path("prompts/caption.txt")
"""Where the fixed captioning prompt lives, relative to
`paths.data_root` — the convention `segmentation.boundary_prompt`
(`prompts/segment_boundaries.txt`) already uses. The repository ships
the canonical prompt at this same relative path."""

CAPTION_TEMPERATURE = 0.0

_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def normalize_caption(text: str) -> str:
    """One stripped paragraph: thinking blocks removed, every whitespace
    run (including the newlines of a multi-paragraph answer) collapsed
    to a single space."""
    without_thoughts = _THINK_BLOCK_RE.sub("", text)
    return strip_nul(" ".join(without_thoughts.split()))


def _output_text(output: Any) -> str:
    """`mlx_vlm.generate` returns a `GenerationResult` whose `.text` is
    the decoded string; older releases returned the string itself."""
    text = getattr(output, "text", output)
    if not isinstance(text, str):
        raise EnrichmentError(
            f"mlx_vlm.generate returned {type(output).__name__} with no text, not a caption"
        )
    return text


class MlxVlmCaptionProvider:
    """`CaptionProvider` over `mlx_vlm.load` / `apply_chat_template` /
    `generate`.

    `model_repo` is a Hugging Face repo id or a local model directory;
    `revision` pins the Hub revision (see
    `imsg.enrich.model_runtime.resolve_model_snapshot`) and is part of
    `model_id` (`<repo>@<revision or 'main'>`). `prompt` is the fixed
    captioning prompt, hashed verbatim into `prompt_sha256`.

    `shared_runtime` is where the weights come from. Hand the *same*
    `SharedVlmRuntime` to this provider and to
    `imsg.segment.mlx_boundaries.MlxBoundaryProvider` and a process
    running both roles against one pin loads one copy of the weights.
    Left `None`, the provider builds a private runtime bounded at
    `cache_limit_bytes`, which is the single-role behaviour.
    """

    def __init__(
        self,
        model_repo: str,
        revision: str | None,
        prompt: str,
        *,
        max_tokens: int = 256,
        shared_runtime: SharedVlmRuntime | None = None,
        cache_limit_bytes: int | None = DEFAULT_ENRICHMENT_CACHE_LIMIT_BYTES,
    ) -> None:
        if not prompt.strip():
            raise ConfigError(
                "the caption prompt is empty — author prompts/caption.txt under data_root "
                "before enabling captioning"
            )
        if max_tokens < 1:
            raise ConfigError(f"caption max_tokens must be at least 1, got {max_tokens}")
        self.model_repo = model_repo
        self.revision = revision
        self.prompt = prompt
        self.prompt_sha256 = sha256_text(prompt)
        self.max_tokens = max_tokens
        self.model_id = f"{model_repo}@{revision or 'main'}"
        self._runtime = shared_runtime or SharedVlmRuntime(cache_limit_bytes=cache_limit_bytes)

    @property
    def shared_runtime(self) -> SharedVlmRuntime:
        """The runtime this provider loads from — the one passed in, or
        the private one it made. Exposed so a caller can hand the same
        object to the boundary provider after the fact, and so tests can
        assert on what is resident."""
        return self._runtime

    def _load(self) -> LoadedVlm:
        """The shared model, loaded once per pin per runtime. A load
        failure is an environment problem (missing or corrupt weights,
        not enough memory for the model at all), so it is a
        `ModelRuntimeUnavailableError`, not a per-task failure."""
        return self._runtime.acquire(self.model_repo, self.revision)

    def caption(self, image_path: Path) -> str:
        if not image_path.is_file():
            raise EnrichmentError(f"caption input is not a file: '{image_path}'")
        mlx_vlm = import_runtime_module("mlx_vlm", install_hint=_INSTALL_HINT)
        prompt_utils = import_runtime_module("mlx_vlm.prompt_utils", install_hint=_INSTALL_HINT)
        loaded = self._load()
        try:
            # `enable_thinking=False`: rendered against the pinned Qwen3.5
            # repo's chat_template.jinja (2026-09-14) it closes an empty
            # <think></think> block so the answer starts at once; left to
            # the template's default the model thinks first and can spend
            # `max_tokens` before any caption appears. mlx_vlm 0.7.1 happens
            # to default this off for qwen3_5_moe; the provider's contract
            # (temperature 0, the budget spent on the caption) must not rest
            # on a third-party default.
            formatted_prompt = prompt_utils.apply_chat_template(
                loaded.processor, loaded.config, self.prompt, num_images=1, enable_thinking=False
            )
            output = mlx_vlm.generate(
                loaded.model,
                loaded.processor,
                formatted_prompt,
                image=[str(image_path)],
                max_tokens=self.max_tokens,
                temperature=CAPTION_TEMPERATURE,
                verbose=False,
            )
        except Exception as exc:
            raise EnrichmentError(
                f"mlx_vlm ({self.model_id}) failed captioning '{image_path}': {exc}"
            ) from exc
        caption = normalize_caption(_output_text(output))
        if not caption:
            raise EnrichmentError(
                f"mlx_vlm ({self.model_id}) returned an empty caption for '{image_path}'"
            )
        return caption


__all__ = [
    "CAPTION_TEMPERATURE",
    "DEFAULT_CAPTION_PROMPT_PATH",
    "MlxVlmCaptionProvider",
    "normalize_caption",
]
