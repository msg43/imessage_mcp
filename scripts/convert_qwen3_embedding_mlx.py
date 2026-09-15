#!/usr/bin/env python3
"""Convert upstream `Qwen/Qwen3-Embedding-8B` into a quantized MLX
checkpoint that `mlx_lm` (and `imsg.embed.mlx_text`) can load.

`mlx_lm.convert` cannot do this on its own: the upstream repo stores
its tensors in the *bare* transformers layout (`embed_tokens.weight`,
`layers.N.*`, `norm.weight` — no `model.` prefix) and ships no LM head
at all, while `config.json` says `tie_word_embeddings: false`.
`mlx_lm`'s Qwen3 `Model` expects `model.*` names and, for an untied
config, allocates an `lm_head` whose weight the checkpoint cannot
supply, so its strict load fails before quantization starts. This
script is `mlx_lm.convert` with the two corrections applied at the
point the weights are read:

1. every tensor is renamed under `model.` (`model.embed_tokens.weight`,
   `model.layers.N.*`, `model.norm.weight`);
2. the saved `config.json` declares `tie_word_embeddings: true` — the
   embedder reads the base transformer's hidden states and never calls
   a head, so no head is allocated, quantized or stored (the loader in
   `imsg.embed.mlx_text` passes the same override for Hub conversions).

Everything else is `mlx_lm`'s own machinery — `quantize_model`,
`save_model`, `save_config` — so the artifact's quantization
parameters, shard layout and metadata match a `mlx_lm.convert` output.

Usage (the project venv with the `models` extra; `$DATA_ROOT` is
`paths.data_root`):

    python scripts/convert_qwen3_embedding_mlx.py \\
        --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \\
        --out $DATA_ROOT/models/qwen3-embedding-8b-8bit-1d8ad4ca \\
        --q-bits 8 --q-group-size 64 --q-mode affine

The pinned upstream snapshot is fetched (or found in the Hugging Face
cache) first and its DIRECTORY is what gets converted, so every input
file is the pinned one. `--out` must not exist yet.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

UPSTREAM_REPO = "Qwen/Qwen3-Embedding-8B"
SNAPSHOT_PATTERNS = ("*.json", "*.safetensors", "*.txt", "*.jinja", "*.py", "*.model", "*.tiktoken")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
)


def prefixed(name: str) -> str:
    """The `mlx_lm` parameter path for an upstream tensor name."""
    if name.startswith(("model.", "lm_head.")):
        return name
    return f"model.{name}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--upstream-repo", default=UPSTREAM_REPO)
    parser.add_argument("--revision", required=True, help="upstream commit sha (40 hex)")
    parser.add_argument("--out", required=True, type=Path, help="output directory (must not exist)")
    parser.add_argument("--q-bits", type=int, default=8)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--q-mode", default="affine", choices=("affine", "mxfp8", "mxfp4", "nvfp4"))
    args = parser.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists; refusing to overwrite a conversion")

    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import _get_classes, quantize_model, save_config, save_model

    started = time.perf_counter()
    src = Path(
        snapshot_download(
            args.upstream_repo, revision=args.revision, allow_patterns=list(SNAPSHOT_PATTERNS)
        )
    )
    print(f"[INFO] upstream snapshot: {args.upstream_repo}@{args.revision}")

    config: dict[str, Any] = json.loads((src / "config.json").read_text())
    config["tie_word_embeddings"] = True  # no head shipped, none used — see the module docstring

    weight_files = sorted(glob.glob(str(src / "model*.safetensors")))
    if not weight_files:
        raise SystemExit(f"no model*.safetensors in {src}")
    weights: dict[str, Any] = {}
    for path in weight_files:
        for name, value in mx.load(path).items():
            weights[prefixed(name)] = value
    print(f"[INFO] {len(weights)} tensors read from {len(weight_files)} shard(s)")

    model_class, args_class = _get_classes(config)
    model = model_class(args_class.from_dict(config))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=True)

    dtype = config.get("torch_dtype")
    if dtype in ("float16", "bfloat16", "float32"):
        target = getattr(mx, dtype)
        model.set_dtype(target)
        print(f"[INFO] parameters cast to {dtype}")

    print(f"[INFO] quantizing: mode={args.q_mode} bits={args.q_bits} group_size={args.q_group_size}")
    model, config = quantize_model(
        model, config, args.q_group_size, args.q_bits, mode=args.q_mode
    )

    save_model(args.out, model, donate_model=True)
    save_config(config, config_path=args.out / "config.json")
    for name in TOKENIZER_FILES:
        candidate = src / name
        if candidate.is_file():
            shutil.copy(candidate, args.out / name)
    for path in glob.glob(str(src / "*.py")):
        shutil.copy(path, args.out)

    written = sorted(p.name for p in args.out.iterdir())
    print(f"[INFO] wrote {args.out}: {written}")
    print(f"[INFO] done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
