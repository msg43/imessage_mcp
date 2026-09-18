#!/usr/bin/env python3
"""Prove that loading only the PE-Core tower a process uses is
numerically free, and measure what it saves.

`imsg.embed.pe_core_multimodal.PeCoreMultimodalEmbeddingProvider` decides
on first use which tower it needs and drops the other (D10.3 defect 2:
both towers were resident everywhere, 9.07 GiB on the query side where
SPEC §4.1 budgeted ~1 GiB, and again on any process that embeds images).
Dropping a tower cannot change what the other computes — they share no
parameters — but "cannot" is an argument, and this is the check.

Three child processes embed the SAME fixed synthetic inputs (one short
phrase and a deterministic generated image — no message content, no real
file is read):

    both    embed_text() then embed_images(), so both towers are resident
            when each vector is produced: the residency this change
            replaces, and the reference vectors
    text    embed_text() only, so the vision tower was dropped
    image   embed_images() only, so the text tower was dropped

and the parent asserts the vectors are **identical** — compared as exact
float bit patterns, not within a tolerance. It also reports each child's
torch allocation and process footprint, which is the saving.

Usage (the project venv with the `models` extra):

    python scripts/verify_pe_core_tower_selection.py
    python scripts/verify_pe_core_tower_selection.py --report-json out.json

Exits 0 when every vector matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imsg import constants  # noqa: E402

PROBE_TEXT = "a photo of a kite over a harbour"
"""Fixed, synthetic, content-free — the whole point is that both runs see
the same bytes."""

PROBE_IMAGE_SIZE = 448
GIB = float(2**30)
MODES = ("both", "text", "image")


def _write_probe_image(path: Path) -> None:
    """A deterministic RGB image, generated rather than read from disk, so
    the check depends on no file outside this repository."""
    from PIL import Image

    pixels = bytearray()
    for y in range(PROBE_IMAGE_SIZE):
        for x in range(PROBE_IMAGE_SIZE):
            pixels += bytes(((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256))
    Image.frombytes("RGB", (PROBE_IMAGE_SIZE, PROBE_IMAGE_SIZE), bytes(pixels)).save(path)


def _bit_pattern(vector: list[float]) -> str:
    """The vector's exact IEEE-754 bits as hex — an equality test that
    cannot be passed by two merely-close vectors, and that distinguishes
    -0.0 from 0.0."""
    return struct.pack(f"<{len(vector)}d", *vector).hex()


def _process_footprint_gib() -> float | None:
    """The process's memory footprint (macOS `footprint`-style), read from
    `ps`. `None` where `ps` cannot answer."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return int(out.stdout.strip()) * 1024 / GIB
    except Exception:
        return None


def _torch_allocated_gib() -> float | None:
    try:
        import torch

        mps = getattr(torch, "mps", None)
        current = getattr(mps, "current_allocated_memory", None)
        if callable(current):
            return float(current()) / GIB
    except Exception:
        return None
    return None


def run_child(mode: str, image_path: Path, out_path: Path) -> None:
    """One provider, one tower decision, the fixed inputs through it."""
    from imsg.embed.pe_core_multimodal import PeCoreMultimodalEmbeddingProvider

    provider = PeCoreMultimodalEmbeddingProvider(
        constants.MULTIMODAL_EMBEDDING_MODEL_REPO,
        constants.MULTIMODAL_EMBEDDING_MODEL_REVISION,
        constants.MULTIMODAL_EMBEDDING_DIM,
    )
    result: dict[str, Any] = {"mode": mode}
    if mode in ("both", "text"):
        result["text_vector"] = provider.embed_text(PROBE_TEXT)
    if mode in ("both", "image"):
        result["image_vector"] = provider.embed_images([image_path])[0]
    runtime = provider._load("text" if mode != "image" else "image")
    result["towers_resident"] = sorted(runtime.towers)
    result["torch_allocated_gib"] = _torch_allocated_gib()
    result["process_footprint_gib"] = _process_footprint_gib()
    out_path.write_text(json.dumps(result), encoding="utf-8")


def _run_modes(work_dir: Path) -> dict[str, dict[str, Any]]:
    image_path = work_dir / "probe.png"
    _write_probe_image(image_path)
    results: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        out_path = work_dir / f"{mode}.json"
        print(f"--- running mode={mode} ---", flush=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", mode,
             "--image", str(image_path), "--out", str(out_path)],
            check=True,
        )
        results[mode] = json.loads(out_path.read_text(encoding="utf-8"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--image", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    if args.child:
        assert args.image is not None and args.out is not None
        run_child(args.child, args.image, args.out)
        return 0

    with tempfile.TemporaryDirectory(prefix="pe-core-tower-check-") as tmp:
        results = _run_modes(Path(tmp))

    reference = results["both"]
    checks = [
        ("text tower", reference["text_vector"], results["text"]["text_vector"]),
        ("vision tower", reference["image_vector"], results["image"]["image_vector"]),
    ]
    ok = True
    print()
    for label, expected, actual in checks:
        identical = _bit_pattern(expected) == _bit_pattern(actual)
        ok = ok and identical
        print(
            f"{label}: {'IDENTICAL' if identical else 'DIFFERENT'} "
            f"({len(expected)} dims, sha of bit pattern "
            f"{_bit_pattern(expected)[:16]}… vs {_bit_pattern(actual)[:16]}…)"
        )

    print()
    print(f"{'mode':<8}{'towers':<18}{'torch alloc GiB':>17}{'footprint GiB':>16}")
    for mode in MODES:
        r = results[mode]
        alloc = r["torch_allocated_gib"]
        foot = r["process_footprint_gib"]
        print(
            f"{mode:<8}{','.join(r['towers_resident']):<18}"
            f"{(f'{alloc:.3f}' if alloc is not None else '—'):>17}"
            f"{(f'{foot:.3f}' if foot is not None else '—'):>16}"
        )

    if args.report_json:
        args.report_json.write_text(
            json.dumps({"identical": ok, "results": results}, indent=2), encoding="utf-8"
        )
    print()
    print("RESULT:", "tower selection is numerically free" if ok else "VECTORS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
