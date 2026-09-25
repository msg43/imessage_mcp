#!/usr/bin/env python3
"""Cut the PE-Core text tower out of the pinned open_clip checkpoint into
its own directory, so the MCP servers load 2.0 GiB instead of building and
reading the whole 9.01 GiB model (`imsg.embed.pe_core_text_tower`).

The output is the `pe-core-g14-448-text-tower` local conversion in
`models/manifest.lock.yaml`; this is its recorded `command`. Usage (the
project venv with the `models` extra; `$DATA_ROOT` is `paths.data_root`):

    python scripts/convert_pe_core_text_tower.py \\
        --upstream-repo timm/PE-Core-bigG-14-448 \\
        --revision 17aa0c25addfa14198fa2ff73d845a22d433432e \\
        --out $DATA_ROOT/models/pe-core-bigG-14-448-text-17aa0c25

The pinned snapshot is fetched (or found in the Hugging Face cache, which
`HF_HOME` selects) first: `open_clip_config.json` and
`open_clip_model.safetensors` only. Only the `text.*` tensors are read
from the checkpoint; they are written unchanged. The conversion needs no
GPU and loads no model onto one: peak memory is about twice the 2.0 GiB
text tower. `--out` must not exist. The same input gives the same bytes
on any host, so `imsg models verify --data-root $DATA_ROOT` can check the
result against the lock's `artifact_sha256`.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imsg.embed.pe_core_text_tower import (  # noqa: E402
    TEXT_TOWER_CONFIG_FILENAME,
    UPSTREAM_WEIGHTS_FILENAME,
    write_text_tower_checkpoint,
)
from imsg.providers.manifest import artifact_digest  # noqa: E402

DEFAULT_UPSTREAM_REPO = "timm/PE-Core-bigG-14-448"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--upstream-repo", default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--revision", required=True, help="upstream commit sha (40 hex)")
    parser.add_argument("--out", required=True, type=Path, help="output directory (must not exist)")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="use this already-downloaded snapshot directory instead of fetching one",
    )
    args = parser.parse_args(argv)
    if len(args.revision) != 40:
        parser.error("--revision must be a full 40-hex commit sha")

    torch = importlib.import_module("torch")
    open_clip_model = importlib.import_module("open_clip.model")
    safetensors_torch = importlib.import_module("safetensors.torch")
    safe_open = importlib.import_module("safetensors").safe_open

    if args.snapshot_dir is not None:
        snapshot = args.snapshot_dir
    else:
        hub = importlib.import_module("huggingface_hub")
        snapshot = Path(
            hub.snapshot_download(
                repo_id=args.upstream_repo,
                revision=args.revision,
                allow_patterns=[TEXT_TOWER_CONFIG_FILENAME, UPSTREAM_WEIGHTS_FILENAME],
            )
        )
    print(f"snapshot: {snapshot}", flush=True)

    started = time.perf_counter()
    provenance = write_text_tower_checkpoint(
        snapshot,
        args.out,
        upstream_repo=args.upstream_repo,
        upstream_revision=args.revision,
        torch=torch,
        open_clip_model=open_clip_model,
        safetensors_torch=safetensors_torch,
        safe_open=safe_open,
    )
    elapsed = time.perf_counter() - started
    digest = artifact_digest(args.out)
    print(
        f"wrote {args.out}: {provenance.tensor_count} tensors, "
        f"{provenance.parameter_count:,} parameters, non-persistent buffers "
        f"{list(provenance.non_persistent_buffers)}, in {elapsed:.1f} s"
    )
    print(f"upstream {UPSTREAM_WEIGHTS_FILENAME} sha256: {provenance.upstream_weights_sha256}")
    for item in digest.files:
        print(f"  {item.relative_path}  {item.sha256}  ({item.size_bytes:,} bytes)")
    print(f"artifact_sha256: {digest.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
