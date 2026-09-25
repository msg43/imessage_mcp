from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from imsg.config.secrets import SecretRef
from imsg.errors import SecretResolutionError


def test_parse_keychain_ref() -> None:
    ref = SecretRef.parse("keychain:imsgindex-pg")
    assert ref.kind == "keychain"
    assert ref.name == "imsgindex-pg"
    assert ref.raw == "keychain:imsgindex-pg"


def test_parse_env_ref() -> None:
    ref = SecretRef.parse("env:IMSG_OAUTH_CLIENT_ID")
    assert ref.kind == "env"
    assert ref.name == "IMSG_OAUTH_CLIENT_ID"


def test_parse_is_idempotent_on_an_existing_secretref() -> None:
    ref = SecretRef.parse("env:FOO")
    assert SecretRef.parse(ref) is ref


@pytest.mark.parametrize(
    "literal",
    [
        "hunter2",
        "sk-abc123def456",
        "postgresql://user:hunter2@host/db",
        "",
        "keychain:",  # empty item name
        "env:",  # empty var name
        "env:lowercase-not-allowed",
        "keychain items with spaces",
    ],
)
def test_literal_values_are_rejected(literal: str) -> None:
    with pytest.raises(ValueError, match=r"keychain:|env:"):
        SecretRef.parse(literal)


def test_non_string_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="string"):
        SecretRef.parse(12345)


def test_repr_never_includes_a_resolved_value() -> None:
    ref = SecretRef.parse("keychain:imsgindex-pg")
    assert "imsgindex-pg" in repr(ref)
    # repr must not attempt resolution as a side effect
    assert repr(ref) == "SecretRef('keychain:imsgindex-pg')"


def test_resolve_env_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMSG_TEST_SECRET", "the-actual-value")
    ref = SecretRef.parse("env:IMSG_TEST_SECRET")
    assert ref.resolve() == "the-actual-value"


def test_resolve_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMSG_TEST_SECRET_ABSENT", raising=False)
    ref = SecretRef.parse("env:IMSG_TEST_SECRET_ABSENT")
    with pytest.raises(SecretResolutionError):
        ref.resolve()


def test_resolve_keychain_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="s3cr3t\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ref = SecretRef.parse("keychain:imsgindex-pg")
    assert ref.resolve() == "s3cr3t"


def test_resolve_keychain_not_found_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=44, stdout="", stderr="not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ref = SecretRef.parse("keychain:missing-item")
    with pytest.raises(SecretResolutionError):
        ref.resolve()


def test_resolve_keychain_missing_security_cli_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("security")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ref = SecretRef.parse("keychain:imsgindex-pg")
    with pytest.raises(SecretResolutionError):
        ref.resolve()


# --------------------------------------------------------------------------
# file:<absolute path>
# --------------------------------------------------------------------------


def _secret_file(tmp_path: Path, content: bytes, mode: int = 0o600, name: str = "secret") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    path.chmod(mode)
    return path


def test_parse_file_ref() -> None:
    ref = SecretRef.parse("file:/Volumes/IMSG-Data/imsgindex/private/env/IMSG_PG_PASSWORD")
    assert ref.kind == "file"
    assert ref.name == "/Volumes/IMSG-Data/imsgindex/private/env/IMSG_PG_PASSWORD"
    assert ref.raw == "file:/Volumes/IMSG-Data/imsgindex/private/env/IMSG_PG_PASSWORD"


@pytest.mark.parametrize(
    "value",
    ["file:", "file:relative/path", "file:./secret", "file:~/secret", "file:/", "file:/a/dir/"],
)
def test_file_refs_must_name_an_absolute_file_path(value: str) -> None:
    with pytest.raises(ValueError, match="file:<absolute path>"):
        SecretRef.parse(value)


def test_file_ref_validates_and_serializes_as_its_reference_in_a_model() -> None:
    class Holder(BaseModel):
        secret: SecretRef

    held = Holder(secret="file:/Volumes/IMSG-Data/secret")  # type: ignore[arg-type]
    assert held.secret.kind == "file"
    assert held.model_dump() == {"secret": "file:/Volumes/IMSG-Data/secret"}


def test_resolve_file_trims_trailing_newlines(tmp_path: Path) -> None:
    path = _secret_file(tmp_path, b"the-actual-value\n\n")
    assert SecretRef.parse(f"file:{path}").resolve() == "the-actual-value"


@pytest.mark.parametrize("mode", [0o400, 0o600, 0o700])
def test_resolve_file_accepts_owner_only_modes(tmp_path: Path, mode: int) -> None:
    path = _secret_file(tmp_path, b"the-actual-value", mode, name=f"secret-{mode:o}")
    assert SecretRef.parse(f"file:{path}").resolve() == "the-actual-value"


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o660, 0o740, 0o602])
def test_resolve_file_refuses_group_or_other_access_without_echoing_it(
    tmp_path: Path, mode: int
) -> None:
    path = _secret_file(tmp_path, b"the-actual-value\n", mode)
    with pytest.raises(SecretResolutionError, match="chmod 600") as excinfo:
        SecretRef.parse(f"file:{path}").resolve()
    assert "the-actual-value" not in str(excinfo.value)


def test_resolve_file_refuses_a_file_owned_by_another_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _secret_file(tmp_path, b"the-actual-value\n")
    other = os.stat(path).st_uid + 1
    monkeypatch.setattr(os, "geteuid", lambda: other)
    with pytest.raises(SecretResolutionError, match="not by the current user") as excinfo:
        SecretRef.parse(f"file:{path}").resolve()
    assert "the-actual-value" not in str(excinfo.value)


def test_resolve_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(SecretResolutionError, match="does not exist"):
        SecretRef.parse(f"file:{tmp_path / 'absent'}").resolve()


def test_resolve_file_refuses_a_directory(tmp_path: Path) -> None:
    directory = tmp_path / "dir"
    directory.mkdir(mode=0o700)
    with pytest.raises(SecretResolutionError, match="not a regular file"):
        SecretRef.parse(f"file:{directory}").resolve()


def test_resolve_file_refuses_a_fifo_instead_of_waiting_for_a_writer(tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(SecretResolutionError, match="not a regular file"):
        SecretRef.parse(f"file:{fifo}").resolve()


def test_resolve_file_follows_a_symlink_and_checks_the_file_it_reaches(tmp_path: Path) -> None:
    target = _secret_file(tmp_path, b"the-actual-value\n")
    link = tmp_path / "link"
    link.symlink_to(target)
    assert SecretRef.parse(f"file:{link}").resolve() == "the-actual-value"
    target.chmod(0o644)
    with pytest.raises(SecretResolutionError, match="chmod 600"):
        SecretRef.parse(f"file:{link}").resolve()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\n", "is empty"),
        (b"", "is empty"),
        (b"\xff\xfe\x00", "not UTF-8"),
        (b"a" * (64 * 1024 + 1), "larger than"),
    ],
)
def test_resolve_file_refuses_content_that_cannot_be_a_secret(
    tmp_path: Path, content: bytes, message: str
) -> None:
    path = _secret_file(tmp_path, content)
    with pytest.raises(SecretResolutionError, match=message):
        SecretRef.parse(f"file:{path}").resolve()
