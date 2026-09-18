"""`imsg mcp public --probe` — the operator surface for AT-1 (SPEC §12).

The probe itself (`imsg.mcp.probe.run_auth_probe`) has its own tests.
This file covers everything around it, and the thing it covers hardest
is the negative space: **no path through this command can reach Google
until every precondition has been satisfied.** That is asserted three
ways — the refusals are all raised by a pure function that runs before
anything network-capable is constructed; the CLI is driven through each
refusal with `build_public_gate` replaced by a landmine; and one test
replaces `socket.socket` itself so that any attempt to open a socket
fails the test rather than reaching the internet.

The second theme is that a token must never appear in output. Tokens
reach this command as `keychain:`/`env:` references and the refusals
name the *reference*; a test asserts no message contains the value.

Fictional personas only (D5): Jamie Owner, Robin Nonowner.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import imsg.cli as cli_module
from imsg.cli import app
from imsg.config.schema import Config
from imsg.mcp.probe import ProbeReport, ProbeVerdict
from imsg.mcp.probe_cli import (
    EXIT_CONFIG,
    EXIT_FAIL,
    EXIT_INVALID,
    EXIT_PASS,
    MIN_TOKEN_LENGTH,
    ProbeConfigurationError,
    ProbeTokens,
    check_probe_preconditions,
    format_probe_report,
    validate_token_shape,
    verdict_exit_code,
)
from imsg.mount.guard import MountInfo

runner = CliRunner()

# Shaped like a Google access token (opaque, long, printable ASCII), but
# entirely synthetic — these never leave the test process.
OWNER_TOKEN = "ya29.synthetic-owner-token-for-at1-probe-tests-0000000000"
FOREIGN_TOKEN = "ya29.synthetic-nonowner-token-for-at1-probe-tests-11111"
OWNER_SUB = "100000000000000000001"


def _public_oauth(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "enabled": False,
        "bind": "127.0.0.1:8700",
        "scope": "allowlist",
        "oauth": {
            "client_id": "env:IMSG_TEST_OAUTH_CLIENT_ID",
            "owner_subject": "env:IMSG_TEST_OWNER_SUB",
        },
    }
    base.update(overrides)
    return base


@pytest.fixture
def probe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMSG_TEST_OAUTH_CLIENT_ID", "1234567890-abc.apps.googleusercontent.com")
    monkeypatch.setenv("IMSG_TEST_OWNER_SUB", OWNER_SUB)
    monkeypatch.setenv("IMSG_TEST_OWNER_TOKEN", OWNER_TOKEN)
    monkeypatch.setenv("IMSG_TEST_FOREIGN_TOKEN", FOREIGN_TOKEN)
    monkeypatch.setenv("IMSG_TEST_PG_PASSWORD", "unused")


@pytest.fixture
def cfg(config_dict_factory: Any, probe_env: None) -> Config:
    return Config.model_validate(
        config_dict_factory(**{"mcp.public": _public_oauth()})
    )


# ---------------------------------------------------------------------------
# Missing tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("owner", "foreign", "expected_flag"),
    [
        (None, "env:IMSG_TEST_FOREIGN_TOKEN", "--owner-token-ref"),
        ("env:IMSG_TEST_OWNER_TOKEN", None, "--foreign-token-ref"),
        (None, None, "--owner-token-ref"),
        ("", "env:IMSG_TEST_FOREIGN_TOKEN", "--owner-token-ref"),
        ("   ", "env:IMSG_TEST_FOREIGN_TOKEN", "--owner-token-ref"),
    ],
)
def test_a_missing_token_reference_is_refused_with_the_setup_hint(
    cfg: Config, owner: str | None, foreign: str | None, expected_flag: str
) -> None:
    with pytest.raises(ProbeConfigurationError) as exc:
        check_probe_preconditions(cfg, owner_token_ref=owner, foreign_token_ref=foreign)
    message = str(exc.value)
    assert expected_flag in message
    assert "security add-generic-password" in message, "the refusal must be actionable"


def test_an_unset_environment_reference_is_refused(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IMSG_TEST_OWNER_TOKEN", raising=False)
    with pytest.raises(ProbeConfigurationError, match="could not be resolved"):
        check_probe_preconditions(
            cfg,
            owner_token_ref="env:IMSG_TEST_OWNER_TOKEN",
            foreign_token_ref="env:IMSG_TEST_FOREIGN_TOKEN",
        )


def test_a_missing_keychain_item_is_refused_without_prompting(cfg: Config) -> None:
    """A Keychain item that does not exist resolves to a clean refusal, not
    a hang and not a traceback."""
    with pytest.raises(ProbeConfigurationError, match="could not be resolved"):
        check_probe_preconditions(
            cfg,
            owner_token_ref="keychain:imsgindex-at1-item-that-does-not-exist-1a2b3c",
            foreign_token_ref="env:IMSG_TEST_FOREIGN_TOKEN",
        )


# ---------------------------------------------------------------------------
# A token passed as a literal is the mistake the design exists to prevent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal",
    [
        "ya29.a-real-looking-access-token-pasted-on-the-command-line",
        "Bearer ya29.something",
        "hunter2",
        "/Users/someone/token.txt",
    ],
)
def test_a_literal_token_is_refused_and_says_why(cfg: Config, literal: str) -> None:
    """`argv` is world-readable via `ps -ww` and the shell records it in
    history — on a host that by design has a public tunnel attached."""
    with pytest.raises(ProbeConfigurationError) as exc:
        check_probe_preconditions(
            cfg, owner_token_ref=literal, foreign_token_ref="env:IMSG_TEST_FOREIGN_TOKEN"
        )
    message = str(exc.value)
    assert "secret *reference*" in message
    assert "shell history" in message and "ps -ww" in message
    assert "keychain:" in message


def test_the_same_reference_twice_is_refused_before_resolving_anything(cfg: Config) -> None:
    with pytest.raises(ProbeConfigurationError, match="two-sided by design"):
        check_probe_preconditions(
            cfg,
            owner_token_ref="env:IMSG_TEST_OWNER_TOKEN",
            foreign_token_ref="env:IMSG_TEST_OWNER_TOKEN",
        )


def test_two_references_resolving_to_the_same_token_are_refused(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct names, one credential — AT-1 step 0 requires distinct
    `sub` values, and a shared credential fails the auth design itself."""
    monkeypatch.setenv("IMSG_TEST_FOREIGN_TOKEN", OWNER_TOKEN)
    with pytest.raises(ProbeConfigurationError, match="resolve to the same token"):
        check_probe_preconditions(
            cfg,
            owner_token_ref="env:IMSG_TEST_OWNER_TOKEN",
            foreign_token_ref="env:IMSG_TEST_FOREIGN_TOKEN",
        )


# ---------------------------------------------------------------------------
# Malformed tokens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", "empty value"),
        ("   ", r"empty value|whitespace"),
        (f"{OWNER_TOKEN}\n", "leading or trailing whitespace"),
        (f"  {OWNER_TOKEN}", "leading or trailing whitespace"),
        (f"ya29.part one{OWNER_TOKEN}", "contains whitespace"),
        (f"Bearer {OWNER_TOKEN}", "starts with 'Bearer '"),
        ("ya29.tökén-with-non-ascii-characters-in-it-000000", "non-ASCII"),
        ("ya29.with\x00a-null-byte-in-the-middle-0000000000", "non-printable"),
        ("short", "shorter than any real"),
        ("x" * (MIN_TOKEN_LENGTH - 1), "shorter than any real"),
    ],
)
def test_a_malformed_token_is_refused(value: str, expected: str) -> None:
    with pytest.raises(ProbeConfigurationError, match=expected):
        validate_token_shape(value, ref="keychain:imsgindex-at1-owner", role="owner")


def test_a_token_of_exactly_the_floor_length_is_accepted() -> None:
    validate_token_shape("x" * MIN_TOKEN_LENGTH, ref="env:X", role="owner")


def test_no_refusal_message_ever_contains_the_token(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one place a carefully-protected secret would otherwise land is a
    terminal scrollback. Every malformed-token refusal names the reference
    and describes the defect; none quotes the value."""
    secret = "ya29.SUPERSECRETVALUE-do-not-print-me-abcdefghijklmnop"
    for bad in (f"{secret}\n", f"Bearer {secret}", f"{secret} {secret}"):
        with pytest.raises(ProbeConfigurationError) as exc:
            validate_token_shape(bad, ref="keychain:imsgindex-at1-owner", role="owner")
        message = str(exc.value)
        assert "SUPERSECRETVALUE" not in message
        assert "keychain:imsgindex-at1-owner" in message

    monkeypatch.setenv("IMSG_TEST_OWNER_TOKEN", "tiny")
    with pytest.raises(ProbeConfigurationError) as exc:
        check_probe_preconditions(
            cfg,
            owner_token_ref="env:IMSG_TEST_OWNER_TOKEN",
            foreign_token_ref="env:IMSG_TEST_FOREIGN_TOKEN",
        )
    assert "tiny" not in str(exc.value).replace("env:IMSG_TEST_OWNER_TOKEN", "")


def test_probe_tokens_repr_does_not_leak_either_token() -> None:
    tokens = ProbeTokens(
        owner="ya29.OWNERSECRET-000000000000",
        foreign="ya29.FOREIGNSECRET-0000000000",
        owner_ref="keychain:imsgindex-at1-owner",
        foreign_ref="keychain:imsgindex-at1-nonowner",
    )
    text = repr(tokens)
    assert "OWNERSECRET" not in text
    assert "FOREIGNSECRET" not in text
    assert "keychain:imsgindex-at1-owner" in text


# ---------------------------------------------------------------------------
# Missing configuration
# ---------------------------------------------------------------------------


def test_a_missing_owner_subject_is_refused(
    config_dict_factory: Any, probe_env: None
) -> None:
    cfg = Config.model_validate(
        config_dict_factory(
            **{
                "mcp.public": _public_oauth(
                    oauth={"client_id": "env:IMSG_TEST_OAUTH_CLIENT_ID"}
                )
            }
        )
    )
    with pytest.raises(ProbeConfigurationError, match="owner_subject is not configured"):
        check_probe_preconditions(
            cfg,
            owner_token_ref="env:IMSG_TEST_OWNER_TOKEN",
            foreign_token_ref="env:IMSG_TEST_FOREIGN_TOKEN",
        )


def test_a_missing_client_id_is_refused(config_dict_factory: Any, probe_env: None) -> None:
    cfg = Config.model_validate(
        config_dict_factory(
            **{"mcp.public": _public_oauth(oauth={"owner_subject": "env:IMSG_TEST_OWNER_SUB"})}
        )
    )
    with pytest.raises(ProbeConfigurationError, match="client_id is not configured"):
        check_probe_preconditions(
            cfg,
            owner_token_ref="env:IMSG_TEST_OWNER_TOKEN",
            foreign_token_ref="env:IMSG_TEST_FOREIGN_TOKEN",
        )


def test_the_happy_precondition_path_returns_both_tokens(cfg: Config) -> None:
    tokens = check_probe_preconditions(
        cfg,
        owner_token_ref="env:IMSG_TEST_OWNER_TOKEN",
        foreign_token_ref="env:IMSG_TEST_FOREIGN_TOKEN",
    )
    assert tokens.owner == OWNER_TOKEN
    assert tokens.foreign == FOREIGN_TOKEN
    assert tokens.owner_ref == "env:IMSG_TEST_OWNER_TOKEN"


# ---------------------------------------------------------------------------
# The verdict must be unambiguous
# ---------------------------------------------------------------------------


def test_each_verdict_gets_a_distinct_headline_and_exit_code() -> None:
    headlines = set()
    for verdict in ProbeVerdict:
        lines = format_probe_report(ProbeReport(verdict, ("a reason",)), scope="allowlist")
        headline = next(line for line in lines if line.startswith("AT-1 PROBE:"))
        headlines.add(headline)
    assert len(headlines) == 3
    assert {verdict_exit_code(v) for v in ProbeVerdict} == {EXIT_PASS, EXIT_FAIL, EXIT_INVALID}


def test_invalid_is_rendered_as_not_a_pass() -> None:
    """The failure this wording exists to prevent: reading an empty
    non-owner result as proof of isolation."""
    text = "\n".join(format_probe_report(ProbeReport(ProbeVerdict.INVALID, ()), scope="allowlist"))
    assert "NOT a pass" in text
    assert "scope: allowlist" in text
    assert "equally consistent" in text


def test_pass_names_the_d6_next_action_and_fail_demands_a_fresh_decision() -> None:
    passing = "\n".join(format_probe_report(ProbeReport(ProbeVerdict.PASS, ()), scope="full"))
    assert "'scope: full' is permitted" in passing
    assert "ops/auth-tests/" in passing

    failing = "\n".join(
        format_probe_report(ProbeReport(ProbeVerdict.FAIL, ("breach",)), scope="full")
    )
    assert "FRESH owner decision" in failing
    assert "Do not expose the corpus" in failing


def test_every_reason_is_enumerated_in_the_output() -> None:
    reasons = ("first problem", "second problem", "third problem")
    text = "\n".join(format_probe_report(ProbeReport(ProbeVerdict.INVALID, reasons), scope="full"))
    assert "Evidence (3):" in text
    for reason in reasons:
        assert reason in text


# ---------------------------------------------------------------------------
# End to end through the CLI: refusals never construct anything network-capable
# ---------------------------------------------------------------------------


@pytest.fixture
def probe_cli(
    cfg: Config,
    config_dict_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_env: None,
) -> Any:
    """Writes a config file and turns every network-capable collaborator
    into a landmine. Any refusal path that touches one fails loudly."""

    def _landmine(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "a refusal path constructed something network-capable — the whole "
            "point is that nothing reachable is built before the preconditions pass"
        )

    monkeypatch.setattr(cli_module, "build_public_gate", _landmine)
    monkeypatch.setattr(cli_module, "PostgresAuditSink", _landmine)
    monkeypatch.setattr(cli_module, "run_auth_probe", _landmine)
    monkeypatch.setattr(cli_module, "connect", _landmine)
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="scratch"),
    )

    def _write(**overrides: Any) -> Path:
        path = tmp_path / "config.yaml"
        payload = config_dict_factory(**{"mcp.public": _public_oauth(), **overrides})
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    return _write


def test_cli_probe_without_tokens_exits_ex_config_and_builds_no_gate(probe_cli: Any) -> None:
    result = runner.invoke(app, ["mcp", "public", "--probe", "--config", str(probe_cli())])
    assert result.exit_code == EXIT_CONFIG
    assert "AT-1 probe did not run" in result.output
    assert "--owner-token-ref" in result.output


def test_cli_probe_with_a_literal_token_exits_ex_config(probe_cli: Any) -> None:
    result = runner.invoke(
        app,
        [
            "mcp", "public", "--probe",
            "--config", str(probe_cli()),
            "--owner-token-ref", "ya29.a-literal-token-not-a-reference",
            "--foreign-token-ref", "env:IMSG_TEST_FOREIGN_TOKEN",
        ],
    )
    assert result.exit_code == EXIT_CONFIG
    assert "secret *reference*" in result.output


def test_cli_probe_with_a_malformed_token_exits_ex_config(
    probe_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMSG_TEST_OWNER_TOKEN", "Bearer ya29.header-not-token-0000000")
    result = runner.invoke(
        app,
        [
            "mcp", "public", "--probe",
            "--config", str(probe_cli()),
            "--owner-token-ref", "env:IMSG_TEST_OWNER_TOKEN",
            "--foreign-token-ref", "env:IMSG_TEST_FOREIGN_TOKEN",
        ],
    )
    assert result.exit_code == EXIT_CONFIG
    assert "Bearer" in result.output


def test_cli_probe_with_missing_oauth_config_exits_ex_config(probe_cli: Any) -> None:
    config_path = probe_cli(**{"mcp.public": _public_oauth(oauth={})})
    result = runner.invoke(
        app,
        [
            "mcp", "public", "--probe",
            "--config", str(config_path),
            "--owner-token-ref", "env:IMSG_TEST_OWNER_TOKEN",
            "--foreign-token-ref", "env:IMSG_TEST_FOREIGN_TOKEN",
        ],
    )
    assert result.exit_code == EXIT_CONFIG
    assert "owner_subject is not configured" in result.output


def test_token_references_without_probe_are_refused(probe_cli: Any) -> None:
    """Refusing beats silently ignoring: an operator who meant to probe and
    forgot the flag must not instead start a public server."""
    result = runner.invoke(
        app,
        [
            "mcp", "public",
            "--config", str(probe_cli()),
            "--owner-token-ref", "env:IMSG_TEST_OWNER_TOKEN",
        ],
    )
    assert result.exit_code == 2
    assert "only meaningful with --probe" in result.output


def test_no_socket_can_be_opened_on_any_refusal_path(
    probe_cli: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structural half of 'never contacts Google in a test': socket
    creation itself is replaced, so any attempt — by this command, by a
    transitively-imported client, by anything — fails the test."""
    opened: list[str] = []

    class _NoSockets(socket.socket):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            opened.append("socket")
            raise AssertionError("the probe opened a socket on a refusal path")

    monkeypatch.setattr(socket, "socket", _NoSockets)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("the probe opened a connection on a refusal path")
    ))

    for args in (
        ["--owner-token-ref", "env:IMSG_TEST_OWNER_TOKEN"],
        ["--owner-token-ref", "ya29.literal", "--foreign-token-ref", "env:IMSG_TEST_FOREIGN_TOKEN"],
        [],
    ):
        result = runner.invoke(
            app, ["mcp", "public", "--probe", "--config", str(probe_cli()), *args]
        )
        assert result.exit_code == EXIT_CONFIG, result.output
    assert opened == []


def test_probe_does_not_require_the_public_surface_to_be_enabled(probe_cli: Any) -> None:
    """AT-1 step 0 runs "while the real server stays disabled" — so
    `--probe` must not inherit `mcp public`'s enabled check. Proven by the
    refusal being about tokens, not about `enabled`."""
    config_path = probe_cli()
    assert yaml.safe_load(config_path.read_text())["mcp"]["public"]["enabled"] is False
    result = runner.invoke(app, ["mcp", "public", "--probe", "--config", str(config_path)])
    assert "enabled is false" not in result.output
    assert "--owner-token-ref" in result.output
