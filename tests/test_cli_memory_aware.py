"""The CLI's memory-aware behaviour, end to end through `imsg` commands
with the database and the mount mocked, as in `tests/test_cli.py`.

- `imsg background pause|resume|status`.
- Every heavy background command honours the pause switch — including
  the host pause file another project creates — without loading a model
  or taking the heavy lock, and exits 76.
- `imsg sync` while paused, or with no memory for its models, does its
  light work (snapshot, extract, identity), skips segmentation and
  embedding, says so, and exits 76 or 75.
- A command the host has no memory for waits, then exits 75 with the
  heavy lock released; one stopped between units reports what it did.
- Each command sets its own role's MLX memory limit.
- `imsg status` reports host memory, the pause state and model processes.
- Both MCP servers refuse a load the host has no room for, and watch the
  host's memory pressure.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from typer.testing import CliRunner

import imsg.cli as cli_module
import imsg.stages.sync as sync_module
from imsg.cli import app
from imsg.embed.pipeline import EmbedRunReport
from imsg.heavy_lock import HeavyModelLock, inspect_heavy_lock
from imsg.mount.guard import MountInfo
from imsg.retrieval.background_warm_up import WarmUpPhase
from test_cli import _FakePgConn, _write_config

# Modules this change adds are imported where a test needs them, not
# here: the tests that drive the CLI alone then run (and fail for what
# they check) against a build without them.

GIB = 2**30
runner = CliRunner()


def host_memory() -> ModuleType:
    return importlib.import_module("imsg.host_memory")


class ShortProbe:
    """A host with 10 GiB available: too little for any model set."""

    def __init__(self, available_gib: float = 10.0) -> None:
        self.available_gib = available_gib

    def read(self) -> Any:
        hm = host_memory()
        return hm.HostMemory(64 * GIB, int(self.available_gib * GIB), 0, 0, hm.PressureLevel.NORMAL)

    def pressure(self) -> Any:
        return host_memory().PressureLevel.NORMAL


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A config (fake backend) with the mount and the database mocked."""
    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, data_root, messages_dir)
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _FakePgConn())
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: Path(str(data_root))
    )
    host_pause_file = fake_home / ".config" / "imessage-index" / "pause-background"
    return {"config": config_path, "data_root": data_root, "host_pause_file": host_pause_file}


def limit_memory_waits(env: dict[str, Path], *, wait: float = 0.0, poll: float = 0.01) -> None:
    """Admission waits of `wait` seconds (0: a refusal defers at once)."""
    config = env["config"]
    config.write_text(
        config.read_text()
        + f"memory:\n  admission_wait_seconds: {wait}\n  admission_poll_seconds: {poll}\n"
    )


def pause_from_another_project(env: dict[str, Path], reason: str = "photo import") -> None:
    flag = env["host_pause_file"]
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(f"reason={reason}\n")


def must_not_run(name: str) -> Any:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} ran while it should have been deferred")

    return refuse


@pytest.fixture
def lock_takes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every heavy-lock acquisition, then takes the lock for real."""
    taken: list[str] = []
    real_acquire = HeavyModelLock.acquire

    def acquire(self: HeavyModelLock) -> None:
        taken.append(self._command)
        real_acquire(self)

    monkeypatch.setattr(HeavyModelLock, "acquire", acquire)
    return taken


# --------------------------------------------------------------------------
# imsg background
# --------------------------------------------------------------------------


def test_pause_status_resume(env: dict[str, Path]) -> None:
    config = str(env["config"])
    status = runner.invoke(app, ["background", "status", "--config", config, "--json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["paused"] is False

    paused = runner.invoke(
        app, ["background", "pause", "--config", config, "--reason", "photo import", "--until", "6h"]
    )
    assert paused.exit_code == 0, paused.output
    assert paused.output.startswith("background: paused — photo import; set by imsg background pause")
    assert "until " in paused.output

    report = json.loads(
        runner.invoke(app, ["background", "status", "--config", config, "--json"]).output
    )
    assert report["paused"] is True
    assert report["active"][0]["reason"] == "photo import"
    assert report["active"][0]["until"] is not None
    assert (env["data_root"] / "run" / "background-pause.json").is_file()

    resumed = runner.invoke(app, ["background", "resume", "--config", config])
    assert resumed.exit_code == 0
    assert "resumed" in resumed.output
    assert json.loads(
        runner.invoke(app, ["background", "status", "--config", config, "--json"]).output
    )["paused"] is False


@pytest.mark.parametrize("until", ["1999-01-01T00:00:00+00:00", "whenever"])
def test_pause_refuses_an_until_it_cannot_honour(env: dict[str, Path], until: str) -> None:
    result = runner.invoke(
        app, ["background", "pause", "--config", str(env["config"]), "--until", until]
    )
    assert result.exit_code == 2
    assert "--until" in result.output  # refused for the value, not for an unknown command
    assert not (env["data_root"] / "run" / "background-pause.json").exists()


def test_resume_says_when_another_projects_pause_still_holds(env: dict[str, Path]) -> None:
    pause_from_another_project(env)
    result = runner.invoke(app, ["background", "resume", "--config", str(env["config"])])
    assert result.exit_code == 0
    assert "no `imsg background pause` switch was set" in result.output
    assert "still paused: photo import; set by host pause file" in result.output
    status = runner.invoke(app, ["background", "status", "--config", str(env["config"])])
    assert "background: paused: photo import" in status.output


# --------------------------------------------------------------------------
# every heavy command honours the pause
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "stage"),
    [
        (["segment"], "run_segment"),
        (["embed"], "run_embed"),
        (["enrich"], "claim_tasks"),
        (["backfill-attachments"], "run_backfill"),
    ],
)
def test_a_heavy_command_does_nothing_while_another_project_has_paused_it(
    env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    lock_takes: list[str],
    command: list[str],
    stage: str,
) -> None:
    monkeypatch.setattr(cli_module, stage, must_not_run(stage))
    pause_from_another_project(env)

    result = runner.invoke(app, [*command, "--config", str(env["config"])])

    assert result.exit_code == 76, result.output
    assert f"{command[0]}: deferred: paused — heavy background work is paused: photo import" in (
        result.output
    )
    assert lock_takes == []  # never even queued for the heavy lock


def test_dry_runs_that_load_no_model_are_not_paused(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pause_from_another_project(env)
    monkeypatch.setattr(cli_module, "run_embed", lambda conn, *a, **kw: EmbedRunReport(dry_run=True))
    result = runner.invoke(app, ["embed", "--dry-run", "--config", str(env["config"])])
    assert result.exit_code == 0, result.output


def test_sync_while_paused_does_its_light_work_and_skips_segmentation_and_embedding(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, lock_takes: list[str]
) -> None:
    from test_sync import (
        _fake_extract_result,
        _fake_identity_result,
        _fake_snapshot_result,
        _ok_invariant,
    )

    light: list[str] = []
    monkeypatch.setattr(sync_module, "guard_mount", lambda data_root: None)
    real_all_sources = sync_module.run_sync_all_sources

    def all_sources(**kwargs: Any) -> Any:
        def snapshot(**kw: Any) -> Any:
            light.append("snapshot")
            return _fake_snapshot_result(env["data_root"] / "snapshot.db")

        def extract(**kw: Any) -> Any:
            light.append("extract")
            return _fake_extract_result()

        def identity(**kw: Any) -> Any:
            light.append("identity")
            return _fake_identity_result(_ok_invariant())

        return real_all_sources(
            **kwargs, run_snapshot_fn=snapshot, run_extract_fn=extract, run_identity_fn=identity
        )

    monkeypatch.setattr(cli_module, "run_sync_all_sources", all_sources)
    monkeypatch.setattr(cli_module, "run_segment", must_not_run("run_segment"))
    monkeypatch.setattr(cli_module, "run_embed", must_not_run("run_embed"))
    pause_from_another_project(env)

    result = runner.invoke(app, ["sync", "--config", str(env["config"])])

    assert result.exit_code == 76, result.output
    assert light == ["snapshot", "extract", "identity"]
    assert "segment_ran=False embed_ran=False" in result.output
    assert (
        "sync: source=mini light work done (snapshot, extract, identity); segmentation and "
        "embedding skipped — deferred: paused — heavy background work is paused: photo import"
    ) in result.output
    assert lock_takes == []


def test_sync_with_no_memory_for_its_models_does_its_light_work_and_exits_75(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, lock_takes: list[str]
) -> None:
    from test_sync import (
        _fake_extract_result,
        _fake_identity_result,
        _fake_snapshot_result,
        _ok_invariant,
    )

    monkeypatch.setattr(sync_module, "guard_mount", lambda data_root: None)
    real_all_sources = sync_module.run_sync_all_sources
    monkeypatch.setattr(
        cli_module,
        "run_sync_all_sources",
        lambda **kw: real_all_sources(
            **kw,
            run_snapshot_fn=lambda **k: _fake_snapshot_result(env["data_root"] / "snapshot.db"),
            run_extract_fn=lambda **k: _fake_extract_result(),
            run_identity_fn=lambda **k: _fake_identity_result(_ok_invariant()),
        ),
    )
    monkeypatch.setattr(cli_module, "run_segment", must_not_run("run_segment"))
    monkeypatch.setattr(cli_module, "run_embed", must_not_run("run_embed"))
    monkeypatch.setattr(host_memory(), "default_probe", ShortProbe)
    limit_memory_waits(env)

    result = runner.invoke(app, ["sync", "--config", str(env["config"])])

    assert result.exit_code == 75, result.output
    assert "segmentation and embedding skipped — deferred: memory — host memory busy" in (
        result.output
    )
    assert lock_takes == ["imsg sync"]
    assert inspect_heavy_lock(env["data_root"]).held is False  # given back at once


# --------------------------------------------------------------------------
# deferring for memory, and stopping between units
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "stage", "label"),
    [
        (["embed"], "run_embed", "imsg embed"),
        (["segment"], "run_segment", "imsg segment"),
        (["enrich"], "claim_tasks", "imsg enrich"),
    ],
)
def test_a_command_the_host_has_no_memory_for_exits_75_and_frees_the_lock(
    env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    lock_takes: list[str],
    command: list[str],
    stage: str,
    label: str,
) -> None:
    monkeypatch.setattr(cli_module, stage, must_not_run(stage))
    monkeypatch.setattr(host_memory(), "default_probe", ShortProbe)
    limit_memory_waits(env)

    result = runner.invoke(app, [*command, "--config", str(env["config"])])

    assert result.exit_code == 75, result.output
    assert f"{command[0]}: deferred: memory — host memory busy: 10.0 GiB available" in result.output
    assert lock_takes == [label]
    assert inspect_heavy_lock(env["data_root"]).held is False
    assert not list((env["data_root"] / "run" / "memory-reservations").glob("*.json"))


def test_embed_waits_for_memory_and_then_runs(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = env["config"]
    limit_memory_waits(env, wait=30.0)

    class FreesUp(ShortProbe):
        reads = 0

        def read(self) -> Any:
            FreesUp.reads += 1
            self.available_gib = 10.0 if FreesUp.reads < 3 else 100.0
            return super().read()

    monkeypatch.setattr(host_memory(), "default_probe", FreesUp)
    monkeypatch.setattr(cli_module, "run_embed", lambda conn, *a, **kw: EmbedRunReport())
    monkeypatch.setattr(cli_module, "sync_fts", lambda conn, fts: _FtsReport())

    result = runner.invoke(app, ["embed", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert result.output.count("embed: waiting for memory") == 2
    assert "embed: memory admitted: 100.0 GiB available" in result.output


class _FtsReport:
    events_applied = 0
    upserts = 0
    deletes = 0


def test_embed_stopped_between_batches_reports_what_it_did_and_exits_75(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = importlib.import_module("imsg.background_gate")
    stop = gate.StopReason(gate.DeferralKind.MEMORY, "the kernel reports critical memory pressure")

    def stopped_run_embed(conn: Any, *args: Any, **kwargs: Any) -> EmbedRunReport:
        assert kwargs["stop_check"] is not None  # the gate is wired in
        raise gate.BackgroundWorkDeferred(stop, partial=EmbedRunReport(segments_embedded=3))

    fts_synced: list[bool] = []
    monkeypatch.setattr(cli_module, "run_embed", stopped_run_embed)
    monkeypatch.setattr(cli_module, "sync_fts", lambda conn, fts: fts_synced.append(True) or _FtsReport())

    result = runner.invoke(app, ["embed", "--config", str(env["config"])])

    assert result.exit_code == 75, result.output
    assert "embed: segments_embedded=3" in result.output
    assert "embed: deferred: memory — the kernel reports critical memory pressure" in result.output
    assert fts_synced == [True]  # the FTS sidecar needs no model
    assert inspect_heavy_lock(env["data_root"]).held is False


def test_enrich_stops_after_the_task_in_hand_when_paused_mid_run(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.enrich.queue import EnrichmentTask

    served = iter(range(1, 1000))
    monkeypatch.setattr(
        cli_module,
        "claim_tasks",
        lambda conn, **kw: [EnrichmentTask(attachment_id=next(served), kind="ocr", attempts=0)],
    )

    def process(conn: Any, config: Any, providers: Any, task: Any) -> str:
        pause_from_another_project(env, reason="import starting")  # set while working
        return "done"

    monkeypatch.setattr(cli_module, "process_one_task", process)

    result = runner.invoke(app, ["enrich", "--config", str(env["config"])])

    assert result.exit_code == 76, result.output
    assert "enrich: claimed=1 done=1" in result.output
    assert "enrich: deferred: paused — heavy background work is paused: import starting" in (
        result.output
    )


# --------------------------------------------------------------------------
# per-role MLX memory limits
# --------------------------------------------------------------------------


def test_each_command_sets_its_own_roles_mlx_memory_limit(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    from test_sync import (
        _fake_extract_result,
        _fake_identity_result,
        _fake_snapshot_result,
        _ok_invariant,
    )

    role = importlib.import_module("imsg.memory_admission").ModelRole
    roles: list[Any] = []
    monkeypatch.setattr(
        cli_module, "configure_mlx_memory_limit", lambda cfg, role: roles.append(role)
    )
    monkeypatch.setattr(cli_module, "run_segment", lambda *a, **kw: [])
    monkeypatch.setattr(cli_module, "run_embed", lambda conn, *a, **kw: EmbedRunReport())
    monkeypatch.setattr(cli_module, "sync_fts", lambda conn, fts: _FtsReport())
    monkeypatch.setattr(cli_module, "claim_tasks", lambda conn, **kw: [])
    monkeypatch.setattr(anyio, "run", lambda func, *args: None)
    monkeypatch.setattr(sync_module, "guard_mount", lambda data_root: None)
    real_all_sources = sync_module.run_sync_all_sources
    monkeypatch.setattr(
        cli_module,
        "run_sync_all_sources",
        lambda **kw: real_all_sources(
            **kw,
            run_snapshot_fn=lambda **k: _fake_snapshot_result(env["data_root"] / "snapshot.db"),
            run_extract_fn=lambda **k: _fake_extract_result(),
            run_identity_fn=lambda **k: _fake_identity_result(_ok_invariant()),
        ),
    )
    config = str(env["config"])
    expected = {
        ("segment",): [role.SEGMENT],
        ("embed",): [role.EMBED],
        ("enrich",): [role.ENRICH],
        ("mcp", "local"): [role.LOCAL_SERVER],
        ("sync",): [role.SEGMENT, role.EMBED],
    }
    for command, wanted in expected.items():
        roles.clear()
        result = runner.invoke(app, [*command, "--config", config])
        assert result.exit_code == 0, (command, result.output)
        assert roles == wanted, command


# --------------------------------------------------------------------------
# imsg status
# --------------------------------------------------------------------------


def test_status_reports_host_memory_the_pause_and_model_processes(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(host_memory(), "default_probe", lambda: ShortProbe(36.5))
    process = host_memory().ModelProcess
    monkeypatch.setattr(
        host_memory(),
        "list_model_processes",
        lambda **kw: [
            process(pid=4242, command="imsg mcp public", footprint_bytes=int(16.1 * GIB)),
            process(pid=4343, command="imsg mcp local", footprint_bytes=int(0.35 * GIB)),
        ],
    )
    pause_from_another_project(env)

    result = runner.invoke(app, ["status", "--config", str(env["config"]), "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["host_memory_available_bytes"] == int(36.5 * GIB)
    assert report["host_memory_pressure"] == "normal"
    assert report["background_paused"] is True
    assert report["background_pause_reason"].startswith("paused: photo import")
    assert [p["pid"] for p in report["model_processes"]] == [4242, 4343]
    assert report["model_processes_footprint_bytes"] == int(16.1 * GIB) + int(0.35 * GIB)
    assert "heavy_models_lock_held" in report

    text = runner.invoke(app, ["status", "--config", str(env["config"])]).output
    assert "host_memory_summary: 36.5 GiB available (free + inactive + speculative)" in text
    assert "  pid 4242 imsg mcp public: 16.1 GiB" in text


# --------------------------------------------------------------------------
# the MCP servers
# --------------------------------------------------------------------------


def test_mcp_local_does_not_load_models_the_host_has_no_room_for(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    seen: list[Any] = []

    def serve(func: Any, local: Any) -> None:
        # What run_local_server does first, while the command is serving.
        local.warm_up.start()
        seen.append((local, local.warm_up.wait(timeout=10)))

    monkeypatch.setattr(anyio, "run", serve)
    monkeypatch.setattr(host_memory(), "default_probe", ShortProbe)

    result = runner.invoke(app, ["mcp", "local", "--config", str(env["config"])])
    assert result.exit_code == 0, result.output
    [(local, status)] = seen
    assert local.idle_unloader.watching  # memory pressure is watched
    assert status.phase is WarmUpPhase.MEMORY_BUSY
    assert status.detail.startswith("host memory busy: 10.0 GiB available")


def test_mcp_public_does_not_load_models_the_host_has_no_room_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    from test_cli import _write_public_enabled_config

    fake_home = tmp_path / "home"
    messages_dir = fake_home / "Library" / "Messages"
    messages_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IMSG_TEST_OWNER_SUBJECT", "300000000000000000009")
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages_dir)
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    (data_root / ".imsgindex-volume").write_text("")
    config_path = tmp_path / "config.yaml"
    _write_public_enabled_config(config_path, data_root, messages_dir)
    monkeypatch.setattr(
        cli_module,
        "run_guard_mount_or_exit",
        lambda data_root: MountInfo(mount_point=data_root, encrypted=True, volume_name="fake"),
    )
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: _FakePgConn())
    monkeypatch.setattr(
        cli_module, "verify_data_directory", lambda conn, data_root: Path(str(data_root))
    )
    monkeypatch.setattr(host_memory(), "default_probe", ShortProbe)
    publics: list[Any] = []
    real_build = cli_module.build_public_asgi_app

    def build(public: Any, **kwargs: Any) -> Any:
        publics.append(public)
        return real_build(public, **kwargs)

    seen: list[Any] = []

    def serve(app_arg: Any, **kw: Any) -> None:
        # The command starts the warm-up before it serves.
        seen.append(publics[0].warm_up.wait(timeout=10))

    monkeypatch.setattr(cli_module, "build_public_asgi_app", build)
    monkeypatch.setattr(uvicorn, "run", serve)

    result = runner.invoke(app, ["mcp", "public", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    [public] = publics
    [status] = seen
    assert status.phase is WarmUpPhase.MEMORY_BUSY
    assert public.idle_unloader.watching


def test_mcp_local_loads_nothing_at_start_unless_configured_to(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import anyio

    seen: list[Any] = []

    def serve(func: Any, local: Any) -> None:
        local.begin_serving()  # what run_local_server does once stdio is up
        try:
            seen.append((local.warm_at_start, local.warm_up.wait(timeout=0.5).phase))
        finally:
            local.idle_unloader.stop()

    monkeypatch.setattr(anyio, "run", serve)
    result = runner.invoke(app, ["mcp", "local", "--config", str(env["config"])])
    assert result.exit_code == 0, result.output
    assert seen == [(False, WarmUpPhase.NOT_STARTED)]
    assert "the models load on the first retrieval call" in result.stderr

    config = env["config"]
    config.write_text(config.read_text().replace("mcp:\n", "mcp:\n  local:\n    warm_at_start: true\n", 1))
    seen.clear()

    def serve_and_wait(func: Any, local: Any) -> None:
        local.begin_serving()
        try:
            seen.append((local.warm_at_start, local.warm_up.wait(timeout=10).phase))
        finally:
            local.idle_unloader.stop()

    monkeypatch.setattr(anyio, "run", serve_and_wait)
    result = runner.invoke(app, ["mcp", "local", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert seen == [(True, WarmUpPhase.READY)]
    assert "the models load on the first retrieval call" not in result.stderr
