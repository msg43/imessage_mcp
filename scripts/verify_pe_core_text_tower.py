#!/usr/bin/env python3
"""Prove the PE-Core text-tower checkpoint gives bit-identical query
vectors to the full-model load, and time both loads.

`imsg.embed.pe_core_text_tower` lets the query side load 2.0 GiB of text
tower instead of building and reading the whole 9.01 GiB model. That can
only be adopted if the vectors do not move, because the stored image
vectors were made by the full model and every query must land in the
same space. Three child processes, each started fresh so each load pays
its own imports:

    full     the provider as it loads today (open_clip builds both
             towers, the vision tower is dropped): the reference vectors
    tower    the provider with `text_tower_dir` set: the vectors under test
    tensors  both loads in one process, every parameter and buffer of the
             two text towers compared

The parent asserts the query vectors are **identical** — compared as
exact IEEE-754 bit patterns, not within a tolerance — and that every
tensor is equal in dtype, shape and value. It prints each load's time,
peak footprint and device allocation.

The texts are fixed and synthetic (no message content): short and long
(past the 72-token context, so truncation is exercised), punctuation,
accents, CJK and an emoji, and the empty string.

Usage (the project venv with the `models` extra):

    python scripts/verify_pe_core_text_tower.py \\
        --text-tower-dir $DATA_ROOT/models/pe-core-bigG-14-448-text-17aa0c25
    python scripts/verify_pe_core_text_tower.py --text-tower-dir ... --repeat 3 --report-json out.json

Exits 0 when every vector and tensor matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import resource
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from imsg import constants  # noqa: E402

PROBE_TEXTS: tuple[str, ...] = (
    "a photo of a kite over a harbour",
    "receipt",
    "",
    "screenshot of a boarding pass",
    "whiteboard diagram with arrows and three boxes",
    "dog on a beach at sunset",
    "invoice #4471 due 2026-10-01 for $1,250.00",
    "Café crème brûlée — naïve façade",
    "東京タワーの写真",
    "Dinner at 7? 🍝",
    "the quick brown fox jumps over the lazy dog " * 20,
)
GIB = float(2**30)
MODES = ("full", "tower", "tensors")


def _bit_pattern(vector: list[float]) -> str:
    """Exact IEEE-754 bits as hex: two merely-close vectors differ, and
    -0.0 differs from 0.0."""
    return struct.pack(f"<{len(vector)}d", *vector).hex()


def _lifetime_peak_footprint_gib() -> float | None:
    """The kernel's lifetime-maximum physical footprint for this process
    (`proc_pid_rusage`, the number `/usr/bin/time -l` prints as "peak
    memory footprint"). `None` off macOS."""
    try:
        from imsg.host_memory import _libproc, _RusageInfoV4

        info = _RusageInfoV4()
        if _libproc().proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return None
        return int(info.ri_lifetime_max_phys_footprint) / GIB
    except Exception:
        return None


def _max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GIB  # bytes on macOS


def _mps_allocated_gib() -> float | None:
    try:
        import torch

        return float(torch.mps.current_allocated_memory()) / GIB
    except Exception:
        return None


def _provider(text_tower_dir: Path | None, device: str) -> Any:
    from imsg.embed.pe_core_multimodal import PeCoreMultimodalEmbeddingProvider

    return PeCoreMultimodalEmbeddingProvider(
        constants.MULTIMODAL_EMBEDDING_MODEL_REPO,
        constants.MULTIMODAL_EMBEDDING_MODEL_REVISION,
        constants.MULTIMODAL_EMBEDDING_DIM,
        device=device,
        text_tower_dir=text_tower_dir,
    )


def run_vectors_child(mode: str, text_tower_dir: Path, device: str, out_path: Path) -> None:
    """One fresh process, one load, the fixed texts through it. The load
    is timed from before torch is imported, like the server's warm-up
    step ("multimodal text tower ready in N s")."""
    provider = _provider(text_tower_dir if mode == "tower" else None, device)
    started = time.perf_counter()
    runtime = provider._load("text")
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors = [provider.embed_text(text) for text in PROBE_TEXTS]
    embed_seconds = time.perf_counter() - started
    result = {
        "mode": mode,
        "device": device,
        "load_seconds": load_seconds,
        "embed_seconds_all_texts": embed_seconds,
        "towers_resident": sorted(runtime.towers),
        "model_class": type(runtime.model).__name__,
        "vectors": vectors,
        "mps_allocated_gib": _mps_allocated_gib(),
        "max_rss_gib": _max_rss_gib(),
        "peak_footprint_gib": _lifetime_peak_footprint_gib(),
    }
    out_path.write_text(json.dumps(result), encoding="utf-8")


def run_tensors_child(text_tower_dir: Path, device: str, out_path: Path) -> None:
    """Both loads in one process; every tensor of the two text towers
    compared on the CPU with `torch.equal` (exact), plus dtype and shape,
    over parameters and ALL buffers (non-persistent ones included)."""
    import torch

    reference = _provider(None, device)._load("text").model.text
    candidate = _provider(text_tower_dir, device)._load("text").model.text

    def tensors(module: Any) -> dict[str, Any]:
        out = {f"param:{n}": t for n, t in module.named_parameters()}
        out.update({f"buffer:{n}": t for n, t in module.named_buffers()})
        return out

    ref, cand = tensors(reference), tensors(candidate)
    mismatches: list[str] = []
    for name in sorted(set(ref) | set(cand)):
        if name not in ref or name not in cand:
            mismatches.append(f"{name}: present on one side only")
            continue
        a, b = ref[name].detach().to("cpu"), cand[name].detach().to("cpu")
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            mismatches.append(f"{name}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}")
    result = {
        "mode": "tensors",
        "compared": len(set(ref) | set(cand)),
        "parameters": sum(int(t.numel()) for n, t in ref.items() if n.startswith("param:")),
        "mismatches": mismatches,
        "reference_class": type(reference).__name__,
        "candidate_class": type(candidate).__name__,
    }
    out_path.write_text(json.dumps(result), encoding="utf-8")


def _run_child(mode: str, text_tower_dir: Path, device: str, out_path: Path) -> dict[str, Any]:
    print(f"--- {mode} ---", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            mode,
            "--text-tower-dir",
            str(text_tower_dir),
            "--device",
            device,
            "--out",
            str(out_path),
        ],
        check=True,
    )
    loaded: dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8"))
    return loaded


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "—" if value is None else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text-tower-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--repeat", type=int, default=1, help="timed runs of each vector mode")
    parser.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    if args.child == "tensors":
        run_tensors_child(args.text_tower_dir, args.device, args.out)
        return 0
    if args.child:
        run_vectors_child(args.child, args.text_tower_dir, args.device, args.out)
        return 0

    runs: dict[str, list[dict[str, Any]]] = {"full": [], "tower": []}
    with tempfile.TemporaryDirectory(prefix="pe-core-text-tower-check-") as tmp:
        work = Path(tmp)
        for index in range(max(1, args.repeat)):
            for mode in ("full", "tower"):
                runs[mode].append(
                    _run_child(mode, args.text_tower_dir, args.device, work / f"{mode}-{index}.json")
                )
        tensor_report = _run_child("tensors", args.text_tower_dir, args.device, work / "tensors.json")

    ok = True
    reference = runs["full"][0]["vectors"]
    print()
    for mode in ("full", "tower"):
        for index, run in enumerate(runs[mode]):
            if mode == "full" and index == 0:
                continue
            same = [
                _bit_pattern(a) == _bit_pattern(b)
                for a, b in zip(reference, run["vectors"], strict=True)
            ]
            ok = ok and all(same)
            print(
                f"{mode} run {index + 1}: {sum(same)}/{len(same)} query vectors IDENTICAL "
                f"to the full-model reference"
            )
    mismatches = tensor_report["mismatches"]
    ok = ok and not mismatches
    print(
        f"tensors: {tensor_report['compared']} parameters and buffers compared "
        f"({tensor_report['parameters']:,} parameters), {len(mismatches)} differ"
        + (f" — first: {mismatches[0]}" if mismatches else "")
    )

    print()
    print(
        f"{'path':<7}{'run':>4}{'load s':>9}{'peak footprint GiB':>20}"
        f"{'max RSS GiB':>13}{'device alloc GiB':>18}  towers / class"
    )
    for mode in ("full", "tower"):
        for index, run in enumerate(runs[mode]):
            print(
                f"{mode:<7}{index + 1:>4}{run['load_seconds']:>9.2f}"
                f"{_fmt(run['peak_footprint_gib']):>20}{run['max_rss_gib']:>13.2f}"
                f"{_fmt(run['mps_allocated_gib']):>18}  "
                f"{','.join(run['towers_resident'])} / {run['model_class']}"
            )

    if args.report_json:
        payload = {
            "identical": ok,
            "device": args.device,
            "texts": list(PROBE_TEXTS),
            "runs": {
                mode: [{k: v for k, v in run.items() if k != "vectors"} for run in items]
                for mode, items in runs.items()
            },
            "vector_bit_patterns_sha": {
                mode: [
                    hashlib.sha256(
                        "".join(_bit_pattern(v) for v in run["vectors"]).encode()
                    ).hexdigest()
                    for run in items
                ]
                for mode, items in runs.items()
            },
            "tensors": tensor_report,
        }
        args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print("RESULT:", "text-tower checkpoint is bit-identical" if ok else "VECTORS OR TENSORS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
