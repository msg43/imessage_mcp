"""Tests for scripts/copy_backup_offsite.py, the weekly encrypted off-box
copy of the newest nightly backup set.

The destination is a local directory (`--remote ''` runs the destination
commands through /bin/sh), so these need no network. Tests that encrypt
need the `age` and `age-keygen` binaries and skip cleanly without them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "copy_backup_offsite.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("copy_backup_offsite", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cbo() -> ModuleType:
    return _load_module()


needs_age = pytest.mark.skipif(
    shutil.which("age") is None or shutil.which("age-keygen") is None,
    reason="age not installed",
)


def _make_set(backups: Path, name: str, *, complete: bool = True, corrupt: bool = False) -> Path:
    d = backups / name
    d.mkdir(parents=True)
    files = []
    for fname, body in (("postgres.dump", b"PGDMP fake dump " + name.encode()), ("fts.db", b"fts")):
        (d / fname).write_bytes(body)
        files.append(
            {"name": fname, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        )
    if corrupt:
        (d / "fts.db").write_bytes(b"ftx")
    (d / "MANIFEST.json").write_text(json.dumps({"complete": complete, "files": files}))
    return d


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "private").mkdir(parents=True)
    (root / "private" / "config.yaml").write_text("owner: alice\n")
    return root


@pytest.fixture
def key(tmp_path: Path) -> tuple[Path, Path]:
    identity = tmp_path / "keys" / "identity.txt"
    identity.parent.mkdir()
    subprocess.run(["age-keygen", "-o", str(identity)], check=True, capture_output=True)
    recipient = tmp_path / "keys" / "recipient.txt"
    out = subprocess.run(
        ["age-keygen", "-y", str(identity)], check=True, capture_output=True, text=True
    )
    recipient.write_text(out.stdout)
    return identity, recipient


def _args(cbo: ModuleType, root: Path, recipient: Path, dest: Path, keep: int = 8):
    return cbo.parse_args(
        [
            "--data-root",
            str(root),
            "--recipient-file",
            str(recipient),
            "--remote",
            "",
            "--remote-dir",
            str(dest),
            "--keep",
            str(keep),
        ]
    )


def test_newest_complete_set_skips_incomplete_and_foreign(cbo: ModuleType, tmp_path: Path) -> None:
    backups = tmp_path / "backups"
    _make_set(backups, "backup-20261001T080000Z-aaaaaa")
    _make_set(backups, "backup-20261002T080000Z-bbbbbb")
    _make_set(backups, "backup-20261003T080000Z-cccccc", complete=False)
    (backups / "not-a-set").mkdir()
    assert cbo.newest_complete_set(backups).name == "backup-20261002T080000Z-bbbbbb"


def test_verify_source_set_rejects_hash_mismatch(cbo: ModuleType, tmp_path: Path) -> None:
    d = _make_set(tmp_path / "backups", "backup-20261001T080000Z-aaaaaa", corrupt=True)
    with pytest.raises(cbo.OffsiteError, match="sha256 does not match"):
        cbo.verify_source_set(d)


def test_sets_to_delete_keeps_newest(cbo: ModuleType) -> None:
    names = [f"backup-202610{d:02d}T080000Z-aaaaaa" for d in range(1, 11)]
    assert cbo.sets_to_delete(names, 8) == names[:2]
    assert cbo.sets_to_delete(names[:3], 8) == []
    with pytest.raises(cbo.OffsiteError):
        cbo.sets_to_delete(names, 0)


@needs_age
def test_end_to_end_copy_decrypts_and_prunes(cbo: ModuleType, tmp_path: Path, key) -> None:
    identity, recipient = key
    root = _data_root(tmp_path)
    dest = tmp_path / "dest"
    for day in range(1, 4):
        _make_set(root / "backups", f"backup-202610{day:02d}T080000Z-aaaaa{day}")
        # Each week only the newest set is copied; simulate three weeks.
        assert cbo.run(_args(cbo, root, recipient, dest, keep=2)) == 0

    copies = sorted(p.name for p in dest.iterdir())
    assert copies == ["backup-20261002T080000Z-aaaaa2", "backup-20261003T080000Z-aaaaa3"]

    newest = dest / "backup-20261003T080000Z-aaaaa3"
    names = {p.name for p in newest.iterdir()}
    assert names == {
        "MANIFEST.json.age",
        "postgres.dump.age",
        "fts.db.age",
        "private.tar.age",
        "OFFSITE-MANIFEST.json",
    }
    # Nothing at the destination is plaintext except the offsite manifest.
    for p in newest.iterdir():
        if p.name != "OFFSITE-MANIFEST.json":
            assert p.read_bytes().startswith(b"age-encryption.org/v1\n")

    out = tmp_path / "restored"
    out.mkdir()
    subprocess.run(
        [
            "age",
            "-d",
            "-i",
            str(identity),
            "-o",
            str(out / "postgres.dump"),
            str(newest / "postgres.dump.age"),
        ],
        check=True,
    )
    assert (out / "postgres.dump").read_bytes() == b"PGDMP fake dump backup-20261003T080000Z-aaaaa3"
    subprocess.run(
        [
            "age",
            "-d",
            "-i",
            str(identity),
            "-o",
            str(out / "private.tar"),
            str(newest / "private.tar.age"),
        ],
        check=True,
    )
    with tarfile.open(out / "private.tar") as tar:
        tar.extractall(out, filter="data")
    assert (out / "private" / "config.yaml").read_text() == "owner: alice\n"

    state = json.loads((root / "run" / "offsite-backup.last.json").read_text())
    assert state["status"] == "ok"
    assert not (root / "run" / "offsite-backup.FAILED").exists()
    assert not (root / "run" / "offsite-staging" / "backup-20261003T080000Z-aaaaa3").exists()


@needs_age
def test_rerun_same_set_skips_copy(cbo: ModuleType, tmp_path: Path, key) -> None:
    _, recipient = key
    root = _data_root(tmp_path)
    dest = tmp_path / "dest"
    _make_set(root / "backups", "backup-20261001T080000Z-aaaaaa")
    assert cbo.run(_args(cbo, root, recipient, dest)) == 0
    first = (dest / "backup-20261001T080000Z-aaaaaa" / "OFFSITE-MANIFEST.json").read_text()
    assert cbo.run(_args(cbo, root, recipient, dest)) == 0
    assert (dest / "backup-20261001T080000Z-aaaaaa" / "OFFSITE-MANIFEST.json").read_text() == first


@needs_age
def test_read_back_mismatch_fails_and_leaves_no_verified_set(
    cbo: ModuleType, tmp_path: Path, key, monkeypatch
) -> None:
    _, recipient = key
    root = _data_root(tmp_path)
    dest = tmp_path / "dest"
    _make_set(root / "backups", "backup-20261001T080000Z-aaaaaa")
    monkeypatch.setattr(cbo.Remote, "read_back_sha256", lambda self, path: ("0" * 64, 0))
    assert cbo.run(_args(cbo, root, recipient, dest)) == 1
    assert "read-back mismatch" in (root / "run" / "offsite-backup.FAILED").read_text()
    assert not (dest / "backup-20261001T080000Z-aaaaaa").exists()


def test_corrupt_source_fails_before_anything_is_copied(cbo: ModuleType, tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    recipient = tmp_path / "recipient.txt"
    recipient.write_text("age1placeholder\n")
    dest = tmp_path / "dest"
    _make_set(root / "backups", "backup-20261001T080000Z-aaaaaa", corrupt=True)
    args = _args(cbo, root, recipient, dest)
    args.age = sys.executable  # any existing file; source check fails first
    assert cbo.run(args) == 1
    assert "sha256 does not match" in (root / "run" / "offsite-backup.FAILED").read_text()
    assert not any(dest.iterdir())
