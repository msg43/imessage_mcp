"""Materialize + content-address one attachment (SPEC §8 S5a) — exercised
against a real filesystem (tmp_path), no mocking needed here."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from imsg.backfill.materialize import cache_path_for, materialize_attachment


def test_materialize_copies_content_addressed(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"hello world")
    data_root = tmp_path / "data_root"

    result = materialize_attachment(source, data_root)

    expected_sha = hashlib.sha256(b"hello world").hexdigest()
    assert result.sha256 == expected_sha
    assert result.byte_size == len(b"hello world")
    assert result.cache_path == cache_path_for(data_root, expected_sha)
    assert result.cache_path.read_bytes() == b"hello world"
    assert result.cache_path.parent.name == expected_sha[:2]


def test_materialize_deduplicates_identical_content(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_bytes(b"same bytes")
    source_b.write_bytes(b"same bytes")

    result_a = materialize_attachment(source_a, data_root)
    result_b = materialize_attachment(source_b, data_root)

    assert result_a.cache_path == result_b.cache_path
    assert result_a.sha256 == result_b.sha256


def test_materialize_leaves_no_tmp_files_behind(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * 5000)

    materialize_attachment(source, data_root)

    tmp_dir = data_root / "attachments" / ".tmp"
    assert not any(tmp_dir.iterdir()) if tmp_dir.exists() else True


def test_materialize_raises_oserror_for_missing_source(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    with pytest.raises(OSError):
        materialize_attachment(tmp_path / "does-not-exist", data_root)


def test_materialize_handles_large_multi_chunk_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    source = tmp_path / "big.bin"
    payload = b"0123456789" * 500_000  # ~4.9 MB, several read chunks
    source.write_bytes(payload)

    result = materialize_attachment(source, data_root)
    assert result.byte_size == len(payload)
    assert result.cache_path.read_bytes() == payload
