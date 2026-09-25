"""`imsg export` end to end through the CLI (SPEC §11): plan → approve →
push, plus purge-person and the weekly unclassified report.

Every command is driven through `typer.testing.CliRunner` against a live
scratch Postgres, so this exercises the real wiring — config load, mount
gate, connection, `ImsgError` caught at the boundary — and not a mock of
it. Nothing here can reach Google: the only transport any test supplies
is `FakeTransport`, and several tests exist specifically to prove that a
credential-less invocation refuses before a Google client library is even
imported.

Fictional personas only (D5): Alice Example, Bob Builder, Jamie Owner.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml
from typer.testing import CliRunner

import imsg.cli as cli_module
from _export_fixtures import (
    add_participant,
    add_raw_participant,
    admin_reachable,
    allow,
    create_scratch_db,
    drop_scratch_db,
    dsn,
    insert_chat,
    insert_message,
    insert_person,
    insert_segment,
)
from imsg.cli import app
from imsg.config.secrets import SecretRef
from imsg.export.documents import segment_document_id
from imsg.export.transport import FakeTransport
from imsg.mount.guard import MountInfo

TEST_DB_NAME = "imsg_index_export_cli_test"

pytestmark = pytest.mark.skipif(
    not admin_reachable(),
    reason="no reachable scratch Postgres instance — set IMSG_TEST_PG_HOST/PORT/USER",
)

runner = CliRunner()

_T = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(TEST_DB_NAME)
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(TEST_DB_NAME)


@pytest.fixture
def cli_env(
    db: psycopg.Connection,
    config_dict_factory: Callable[..., dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., Path]:
    """Returns a factory that writes a config file and points the CLI's
    collaborators at the scratch database.

    `connect` hands back a NEW autocommit connection each time, exactly
    like the real one — every command closes the connection it was given,
    so a shared one would break the second invocation in a test.
    """
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(
            mount_point=data_root, encrypted=True, volume_name="scratch"
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "connect",
        lambda database, **kw: psycopg.connect(dsn(TEST_DB_NAME), autocommit=True),
    )
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: data_root
    )

    def _write(**overrides: Any) -> Path:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config_dict_factory(**overrides)), encoding="utf-8"
        )
        return config_path

    return _write


@dataclass
class Corpus:
    owner_id: int
    alice_id: int
    bob_id: int
    dm_chat_id: int
    dm_stable_key: str

    @property
    def dm_doc_id(self) -> str:
        return segment_document_id(self.dm_stable_key)


@pytest.fixture
def corpus(db: psycopg.Connection) -> Corpus:
    """One allowlisted DM (owner + alice) that exports, and one chat with
    an unresolved handle that never can — the latter is what the weekly
    unclassified report is for."""
    owner_id = insert_person(
        db, display_name="Jamie Owner", short_name="owner", is_owner=True
    )
    alice_id = insert_person(db, display_name="Alice Example", short_name="alice")
    bob_id = insert_person(db, display_name="Bob Builder", short_name="bob")
    allow(db, owner_id)
    allow(db, alice_id)

    dm_chat_id = insert_chat(db, source_guid="chat-dm")
    add_participant(db, dm_chat_id, owner_id)
    add_participant(db, dm_chat_id, alice_id)
    m1 = insert_message(
        db,
        chat_id=dm_chat_id,
        sender_person_id=owner_id,
        is_from_me=True,
        sent_at=_T,
        text="did the revised bid come through?",
    )
    m2 = insert_message(
        db,
        chat_id=dm_chat_id,
        sender_person_id=alice_id,
        is_from_me=False,
        sent_at=_T + timedelta(minutes=4),
        text="yes, sending it over now",
    )
    dm_stable_key = "stable-dm"
    insert_segment(
        db,
        chat_id=dm_chat_id,
        started_at=_T,
        ended_at=_T + timedelta(minutes=4),
        message_ids=[m1, m2],
        stable_key=dm_stable_key,
    )

    # An unclassified thread: bob has no allowlist row and one handle in
    # it never resolved to a person at all.
    other_chat_id = insert_chat(db, source_guid="chat-unclassified", kind="group")
    add_participant(db, other_chat_id, bob_id)
    add_raw_participant(db, other_chat_id, "handle-unresolved")
    insert_message(
        db,
        chat_id=other_chat_id,
        sender_person_id=bob_id,
        is_from_me=False,
        sent_at=datetime.now(UTC) - timedelta(days=2),
        text="are we still on for thursday?",
    )
    db.commit()
    return Corpus(
        owner_id=owner_id,
        alice_id=alice_id,
        bob_id=bob_id,
        dm_chat_id=dm_chat_id,
        dm_stable_key=dm_stable_key,
    )


def _run(*args: str) -> Any:
    return runner.invoke(app, list(args))


def _run_id_from(output: str) -> int:
    """`export plan: run 7 (mode=reconcile) — ...` -> 7."""
    for line in output.splitlines():
        if line.startswith("export plan: run ") or line.startswith(
            "export purge-person: run "
        ):
            return int(line.split("run ", 1)[1].split()[0])
    raise AssertionError(f"no run id in output:\n{output}")


def _fake_transport(monkeypatch: pytest.MonkeyPatch) -> FakeTransport:
    """Substitute the transport at the ONE place the CLI builds it. There
    is deliberately no product flag that does this — a `--fake-transport`
    option on a command whose whole job is to be hard to fire would be a
    bypass, so the seam is a test-only monkeypatch."""
    transport = FakeTransport()
    monkeypatch.setattr(cli_module, "_export_transport_or_die", lambda cfg: transport)
    return transport


# ---------------------------------------------------------------------------
# Command surface
# ---------------------------------------------------------------------------


def test_export_is_a_command_group_not_a_stub() -> None:
    result = _run("export", "--help")
    assert result.exit_code == 0, result.output
    for command in (
        "plan",
        "approve",
        "push",
        "purge-person",
        "unclassified-report",
    ):
        assert command in result.output
    assert "not implemented" not in result.output


def test_bare_export_shows_help_rather_than_doing_anything() -> None:
    result = _run("export")
    assert result.exit_code != 0
    assert "Usage:" in result.output


# ---------------------------------------------------------------------------
# plan → approve → push
# ---------------------------------------------------------------------------


def test_plan_stages_documents_and_names_the_run(
    cli_env: Callable[..., Path], db: psycopg.Connection, corpus: Corpus
) -> None:
    config_path = cli_env()
    result = _run("export", "plan", "--config", str(config_path))
    assert result.exit_code == 0, result.output
    assert "1 upsert(s)" in result.output
    assert "OWNER APPROVAL REQUIRED" in result.output
    assert "first-push" in result.output

    run_id = _run_id_from(result.output)
    with db.cursor() as cur:
        cur.execute(
            "SELECT status, mode FROM export_run WHERE export_run_id = %s", (run_id,)
        )
        assert cur.fetchone() == ("planned", "reconcile")

    report_line = next(
        line for line in result.output.splitlines() if "review report" in line
    )
    report_path = Path(report_line.split("review report ", 1)[1].strip())
    assert report_path.is_file()
    assert "OWNER APPROVAL REQUIRED" in report_path.read_text(encoding="utf-8")


def test_approve_then_push_promotes_exactly_the_plan(
    cli_env: Callable[..., Path],
    db: psycopg.Connection,
    corpus: Corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = cli_env()
    plan = _run("export", "plan", "--config", str(config_path))
    run_id = _run_id_from(plan.output)

    approved = _run("export", "approve", str(run_id), "--config", str(config_path))
    assert approved.exit_code == 0, approved.output
    assert "pinned manifest sha256" in approved.output

    transport = _fake_transport(monkeypatch)
    pushed = _run("export", "push", str(run_id), "--config", str(config_path))
    assert pushed.exit_code == 0, pushed.output
    assert "pushed=1" in pushed.output
    assert corpus.dm_doc_id in transport.imported
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM export_run WHERE export_run_id = %s", (run_id,)
        )
        assert cur.fetchone() == ("ok",)


def test_approve_accepts_an_explicit_approval_id(
    cli_env: Callable[..., Path], db: psycopg.Connection, corpus: Corpus
) -> None:
    config_path = cli_env()
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)
    result = _run(
        "export",
        "approve",
        str(run_id),
        "--config",
        str(config_path),
        "--approval-id",
        "ops-ticket-42",
    )
    assert result.exit_code == 0, result.output
    assert "ops-ticket-42" in result.output
    with db.cursor() as cur:
        cur.execute(
            "SELECT approval_id FROM export_run WHERE export_run_id = %s", (run_id,)
        )
        assert cur.fetchone() == ("ops-ticket-42",)


# ---------------------------------------------------------------------------
# Refusals — the point of this surface
# ---------------------------------------------------------------------------


def test_plan_refuses_an_empty_allowlist(
    cli_env: Callable[..., Path], db: psycopg.Connection, corpus: Corpus
) -> None:
    """Zero classified people is not "an empty plan": a reconcile would
    also schedule the deletion of everything already pushed. Refuse, and
    point at the command that retracts one person on purpose."""
    with db.cursor() as cur:
        cur.execute("DELETE FROM allowlist_person")
    db.commit()

    config_path = cli_env()
    result = _run("export", "plan", "--config", str(config_path))
    assert result.exit_code == 1
    assert "allowlist_person is empty" in result.output
    assert "purge-person" in result.output
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM export_run")
        assert cur.fetchone() == (0,)


def test_approve_refuses_an_unknown_run_id(
    cli_env: Callable[..., Path], corpus: Corpus
) -> None:
    result = _run("export", "approve", "424242", "--config", str(cli_env()))
    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert "Traceback" not in result.output


def test_approve_refuses_after_staging_was_modified(
    cli_env: Callable[..., Path], corpus: Corpus
) -> None:
    config_path = cli_env()
    plan = _run("export", "plan", "--config", str(config_path))
    run_id = _run_id_from(plan.output)
    staging_line = next(
        line for line in plan.output.splitlines() if "staged under" in line
    )
    staging_dir = Path(staging_line.split("staged under ", 1)[1].strip())
    staged = next((staging_dir / "docs").iterdir())
    staged.write_text("TAMPERED", encoding="utf-8")

    result = _run("export", "approve", str(run_id), "--config", str(config_path))
    assert result.exit_code == 1
    assert "re-run `imsg export plan`" in result.output


def test_push_refuses_an_unapproved_run_and_never_builds_a_transport(
    cli_env: Callable[..., Path], corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = cli_env(**{"export.gcp_credentials": "env:IMSG_TEST_GCP_KEY"})
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)

    transport = _fake_transport(monkeypatch)
    result = _run("export", "push", str(run_id), "--config", str(config_path))
    assert result.exit_code == 1
    assert "requires owner approval" in result.output
    assert transport.calls == []


def test_push_refuses_on_live_drift_and_uploads_nothing(
    cli_env: Callable[..., Path],
    db: psycopg.Connection,
    corpus: Corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D9's TOCTOU point: the hash pins prove the bytes did not change,
    not that the world did not."""
    config_path = cli_env()
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)
    _run("export", "approve", str(run_id), "--config", str(config_path))

    # bob joins the allowlisted DM after approval, and is not allowlisted.
    add_participant(db, corpus.dm_chat_id, corpus.bob_id)
    db.commit()

    transport = _fake_transport(monkeypatch)
    result = _run("export", "push", str(run_id), "--config", str(config_path))
    assert result.exit_code == 1
    assert "no longer export-eligible" in result.output
    assert transport.calls == []
    assert transport.objects == {}


def test_push_refuses_without_a_configured_credential(
    cli_env: Callable[..., Path], corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No credential named in config → the command refuses before it opens
    the database, resolves a secret, or imports a Google client."""
    config_path = cli_env()  # no export.gcp_credentials
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)
    _run("export", "approve", str(run_id), "--config", str(config_path))

    def _no_connect(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("push opened the database without a credential")

    def _no_resolve(self: SecretRef) -> str:
        raise AssertionError("push resolved a secret when none was configured")

    monkeypatch.setattr(cli_module, "connect", _no_connect)
    monkeypatch.setattr(SecretRef, "resolve", _no_resolve)

    result = _run("export", "push", str(run_id), "--config", str(config_path))
    assert result.exit_code == 1
    assert "export.gcp_credentials is not set" in result.output
    assert "Nothing was uploaded" in result.output


def test_push_refuses_a_credential_that_does_not_resolve(
    cli_env: Callable[..., Path], corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = cli_env(
        **{"export.gcp_credentials": "env:IMSG_TEST_GCP_KEY_THAT_IS_UNSET"}
    )
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)
    _run("export", "approve", str(run_id), "--config", str(config_path))
    monkeypatch.delenv("IMSG_TEST_GCP_KEY_THAT_IS_UNSET", raising=False)

    result = _run("export", "push", str(run_id), "--config", str(config_path))
    assert result.exit_code == 1
    assert "IMSG_TEST_GCP_KEY_THAT_IS_UNSET" in result.output
    assert "Traceback" not in result.output


def test_push_refuses_an_unknown_run_id(
    cli_env: Callable[..., Path], corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = cli_env()
    _fake_transport(monkeypatch)
    result = _run("export", "push", "424242", "--config", str(config_path))
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_purge_person_refuses_a_name_that_matches_nobody(
    cli_env: Callable[..., Path], db: psycopg.Connection, corpus: Corpus
) -> None:
    result = _run(
        "export", "purge-person", "nobody-by-this-name", "--config", str(cli_env())
    )
    assert result.exit_code == 1
    assert "no person matches" in result.output
    assert "Nothing was revoked" in result.output
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM allowlist_person WHERE text_allowed")
        assert cur.fetchone() == (2,)  # owner + alice, untouched


def test_config_option_before_the_subcommand_is_a_usage_error(
    cli_env: Callable[..., Path],
) -> None:
    """A `--config` the group swallows would silently target the DEFAULT
    database — the same trap `imsg identity` had to close."""
    result = _run("export", "--config", str(cli_env()), "plan")
    assert result.exit_code != 0
    assert "No such option" in result.output or "Usage:" in result.output


# ---------------------------------------------------------------------------
# purge-person
# ---------------------------------------------------------------------------


def test_purge_person_revokes_and_plans_deletes_without_approval(
    cli_env: Callable[..., Path],
    db: psycopg.Connection,
    corpus: Corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = cli_env()
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)
    _run("export", "approve", str(run_id), "--config", str(config_path))
    transport = _fake_transport(monkeypatch)
    _run("export", "push", str(run_id), "--config", str(config_path))
    assert corpus.dm_doc_id in transport.imported

    purge = _run("export", "purge-person", "alice", "--config", str(config_path))
    assert purge.exit_code == 0, purge.output
    assert "revoked alice" in purge.output
    assert "exempt from the approval gate" in purge.output
    purge_run_id = _run_id_from(purge.output)
    with db.cursor() as cur:
        cur.execute(
            "SELECT text_allowed, attachments_allowed FROM allowlist_person "
            "WHERE person_id = %s",
            (corpus.alice_id,),
        )
        assert cur.fetchone() == (False, False)  # row retained, flags off

    # D9.3 end to end: the purge pushes with no approval recorded.
    pushed = _run("export", "push", str(purge_run_id), "--config", str(config_path))
    assert pushed.exit_code == 0, pushed.output
    assert "deleted=1" in pushed.output
    assert corpus.dm_doc_id not in transport.imported
    with db.cursor() as cur:
        cur.execute(
            "SELECT state FROM export_document WHERE document_id = %s",
            (corpus.dm_doc_id,),
        )
        assert cur.fetchone() == ("purged",)
        cur.execute(
            "SELECT approval_id FROM export_run WHERE export_run_id = %s",
            (purge_run_id,),
        )
        assert cur.fetchone() == (None,)


def test_purge_person_accepts_a_numeric_person_id(
    cli_env: Callable[..., Path], db: psycopg.Connection, corpus: Corpus
) -> None:
    result = _run(
        "export", "purge-person", str(corpus.alice_id), "--config", str(cli_env())
    )
    assert result.exit_code == 0, result.output
    assert f"revoked alice (person {corpus.alice_id})" in result.output
    with db.cursor() as cur:
        cur.execute(
            "SELECT text_allowed FROM allowlist_person WHERE person_id = %s",
            (corpus.alice_id,),
        )
        assert cur.fetchone() == (False,)


# ---------------------------------------------------------------------------
# unclassified-report (SPEC §11.5 — what the weekly LaunchAgent invokes)
# ---------------------------------------------------------------------------


def test_unclassified_report_writes_the_file_the_agent_expects(
    cli_env: Callable[..., Path], db: psycopg.Connection, corpus: Corpus, tmp_path: Path
) -> None:
    config_path = cli_env()
    result = _run("export", "unclassified-report", "--config", str(config_path))
    assert result.exit_code == 0, result.output
    assert "1 unclassified active thread(s)" in result.output

    report_path = Path(result.output.split("wrote ", 1)[1].strip())
    assert report_path.parent.name == "export"
    assert report_path.parent.parent.name == "data_root"
    text = report_path.read_text(encoding="utf-8")
    assert "bob" in text
    assert "unresolved handle" in text
    # Identities and counts only — never content (SPEC §11.5).
    assert "thursday" not in text
    # Written outside staging/, so no push can ever select it.
    assert "staging" not in str(report_path)


def test_unclassified_report_command_matches_the_launchagent(
    cli_env: Callable[..., Path], corpus: Corpus
) -> None:
    """The weekly plist renders `imsg export unclassified-report --config
    <path>`. If that ever stops being a real command, the agent fails
    silently every Monday at 08:00."""
    import plistlib

    from imsg.agents.plists import render_agent_plists

    cfg = cli_module.load_config(cli_env())
    plists = render_agent_plists(
        cfg,
        imsg_binary=Path("/usr/local/bin/imsg"),
        postgres_binary=Path("/usr/local/bin/postgres"),
        cloudflared_binary=Path("/usr/local/bin/cloudflared"),
        config_path=Path("/etc/imsg/config.yaml"),
    )
    report_plist = next(
        plistlib.loads(content)
        for label, content in plists.items()
        if label.endswith("report")
    )
    # Every agent runs the supervisor; the service command follows `--`.
    args = report_plist["ProgramArguments"]
    command = args[args.index("--") + 1 :]
    assert command[1:3] == ["export", "unclassified-report"]

    # ...and that spelling really resolves, rather than merely looking right.
    assert _run("export", "unclassified-report", "--help").exit_code == 0


# ---------------------------------------------------------------------------
# --dry-run: the rehearsals, which must change nothing
# ---------------------------------------------------------------------------


def test_plan_dry_run_stages_nothing(
    cli_env: Callable[..., Path], db: psycopg.Connection, corpus: Corpus, data_root: Path
) -> None:
    result = _run("export", "plan", "--config", str(cli_env()), "--dry-run")
    assert result.exit_code == 0, result.output
    assert "would stage 1 document(s)" in result.output
    assert cli_module.DRY_RUN_MARKER in result.output
    assert not (data_root / "export" / "staging").exists()
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM export_run")
        assert cur.fetchone() == (0,)


def test_push_dry_run_verifies_with_no_credential_and_no_transport(
    cli_env: Callable[..., Path], db: psycopg.Connection, corpus: Corpus
) -> None:
    """The rehearsal is provably network-free: it builds no transport, so
    it needs no credential and has nothing to upload through."""
    config_path = cli_env()  # no export.gcp_credentials
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)
    _run("export", "approve", str(run_id), "--config", str(config_path))

    result = _run(
        "export", "push", str(run_id), "--config", str(config_path), "--dry-run"
    )
    assert result.exit_code == 0, result.output
    assert "verifies" in result.output
    assert "would push 1 upsert(s)" in result.output
    assert cli_module.DRY_RUN_MARKER in result.output
    with db.cursor() as cur:
        cur.execute(
            "SELECT status FROM export_run WHERE export_run_id = %s", (run_id,)
        )
        assert cur.fetchone() == ("planned",)  # still unpushed
        cur.execute("SELECT count(*) FROM export_document")
        assert cur.fetchone() == (0,)


def test_push_dry_run_still_refuses_an_unapproved_run(
    cli_env: Callable[..., Path], corpus: Corpus
) -> None:
    config_path = cli_env()
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)
    result = _run(
        "export", "push", str(run_id), "--config", str(config_path), "--dry-run"
    )
    assert result.exit_code == 1
    assert "requires owner approval" in result.output


def test_purge_person_dry_run_leaves_the_allowlist_untouched(
    cli_env: Callable[..., Path],
    db: psycopg.Connection,
    corpus: Corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = cli_env()
    run_id = _run_id_from(_run("export", "plan", "--config", str(config_path)).output)
    _run("export", "approve", str(run_id), "--config", str(config_path))
    _fake_transport(monkeypatch)
    _run("export", "push", str(run_id), "--config", str(config_path))

    result = _run(
        "export", "purge-person", "alice", "--config", str(config_path), "--dry-run"
    )
    assert result.exit_code == 0, result.output
    assert "would revoke alice" in result.output
    assert f"would delete {corpus.dm_doc_id}" in result.output
    assert cli_module.DRY_RUN_MARKER in result.output
    with db.cursor() as cur:
        cur.execute(
            "SELECT text_allowed FROM allowlist_person WHERE person_id = %s",
            (corpus.alice_id,),
        )
        assert cur.fetchone() == (True,)
        cur.execute("SELECT count(*) FROM export_run WHERE mode = 'purge'")
        assert cur.fetchone() == (0,)


def test_unclassified_report_dry_run_writes_no_file(
    cli_env: Callable[..., Path], corpus: Corpus, data_root: Path
) -> None:
    result = _run(
        "export", "unclassified-report", "--config", str(cli_env()), "--dry-run"
    )
    assert result.exit_code == 0, result.output
    assert "1 unclassified active thread(s)" in result.output
    assert cli_module.DRY_RUN_MARKER in result.output
    assert not list((data_root / "export").glob("unclassified-*.txt"))


# ---------------------------------------------------------------------------
# The structural half of "no command can reach Google by accident"
# ---------------------------------------------------------------------------


def test_importing_the_cli_loads_no_google_client() -> None:
    """`imsg.export.gcp_transport` is imported inside
    `_export_transport_or_die`, AFTER the credential check — so merely
    having the export commands available loads no Google client library.
    A future refactor that hoists that import to module scope would
    silently put a network-capable client in every `imsg` invocation;
    this fails when that happens."""
    probe = textwrap.dedent(
        """
        import json, sys
        import imsg.cli  # noqa: F401
        print(json.dumps(sorted(
            m for m in sys.modules
            if m.split(".")[0] == "google" or "gcp_transport" in m
        )))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == []
