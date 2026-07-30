from __future__ import annotations

import subprocess
from typing import Any

import pytest

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
