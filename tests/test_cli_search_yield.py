"""`imsg sync`, `imsg segment` and `imsg embed` wait while a search is in
flight (`imsg.search_yield`), end to end through the CLI with the
database and the mount mocked, as in `tests/test_cli.py`.

- Embedding waits before each batch, segmentation before each
  boundary-model call (`--rebuild` too), with the enrichment worker's
  settings, and each command says how often and how long.
- Nothing changes, and nothing is said, when no search is in flight.
- A pause set while a step waits for a search stops it at once.
- `imsg status` says which step is yielding right now.

The module this change adds is imported inside the tests that need it,
so the rest fail for what they check against a build without it.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import imsg.cli as cli_module
import imsg.stages.sync as sync_module
from imsg.background_gate import BackgroundWorkDeferred
from imsg.cli import app
from imsg.db.enrichment_yield_locks import YieldReport, YieldState
from imsg.embed.pipeline import EmbedRunReport
from imsg.mount.guard import MountInfo
from imsg.segment.models import SegmentationRunReport
from test_cli import _FakePgConn, _patch_status_probes, _write_config

runner = CliRunner()


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


def add_config(env: dict[str, Path], text: str) -> None:
    env["config"].write_text(env["config"].read_text() + text)


class FakeGates:
    """Stands in for `imsg.search_yield.QueryYieldGate`: records how each
    step's gate was built and answers every wait from `waits[step]`."""

    def __init__(self, waits: dict[str, float]) -> None:
        self.waits = waits
        self.built: list[dict[str, Any]] = []
        self.calls: dict[str, int] = {}

    def __call__(self, conn: Any, **kwargs: Any) -> Any:
        self.built.append(kwargs)
        step = kwargs["step"]
        gates = self

        class _Gate:
            enabled = kwargs["enabled"]

            def wait_until_clear(self, *, interrupt: Any = None) -> YieldReport:
                gates.calls[step] = gates.calls.get(step, 0) + 1
                seconds = gates.waits.get(step, 0.0)
                return YieldReport(paused=seconds > 0, waited_seconds=seconds)

        return _Gate()


@pytest.fixture
def gates(monkeypatch: pytest.MonkeyPatch) -> FakeGates:
    fake = FakeGates({"segment": 1.0, "embed": 1.5})
    search_yield = importlib.import_module("imsg.search_yield")
    monkeypatch.setattr(search_yield, "QueryYieldGate", fake)
    return fake


class _FtsReport:
    events_applied = 0
    upserts = 0
    deletes = 0


def _write_boundary_prompt(env: dict[str, Path]) -> None:
    prompt = env["data_root"] / "prompts" / "segment_boundaries.txt"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text("segment this")


def _messages(count: int) -> list[Any]:
    return [SimpleNamespace(text=f"message {i}") for i in range(count)]


def _fake_light_steps(env: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
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


# --------------------------------------------------------------------------
# each command waits where its GPU work starts, and says so
# --------------------------------------------------------------------------


def test_embed_waits_for_searches_before_each_batch_and_says_how_long(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, gates: FakeGates
) -> None:
    def run_embed(conn: Any, *args: Any, **kwargs: Any) -> EmbedRunReport:
        stop_check = kwargs["stop_check"]
        assert stop_check() is None  # before batch 1
        assert stop_check() is None  # before batch 2
        return EmbedRunReport(segments_embedded=2)

    monkeypatch.setattr(cli_module, "run_embed", run_embed)
    monkeypatch.setattr(cli_module, "sync_fts", lambda conn, fts: _FtsReport())

    result = runner.invoke(app, ["embed", "--config", str(env["config"])])

    assert result.exit_code == 0, result.output
    assert gates.calls == {"embed": 2}
    assert "embed: yielded to in-flight searches 2 time(s), 3.0s total" in result.output
    # The enrichment worker's settings: the schema defaults here.
    assert gates.built == [
        {"step": "embed", "enabled": True, "poll_interval_seconds": 0.25, "max_pause_seconds": 300.0}
    ]


def test_segment_waits_for_searches_before_each_boundary_model_call(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, gates: FakeGates
) -> None:
    _write_boundary_prompt(env)

    def run_segment(conn: Any, config: Any, provider: Any, prompt: bytes, **kw: Any) -> Any:
        provider.detect_boundaries(_messages(12))
        provider.detect_boundaries(_messages(12))
        return [SegmentationRunReport(chat_id=1, segments_written=3)]

    monkeypatch.setattr(cli_module, "run_segment", run_segment)

    result = runner.invoke(app, ["segment", "--config", str(env["config"])])

    assert result.exit_code == 0, result.output
    assert gates.calls == {"segment": 2}
    assert "segment: yielded to in-flight searches 2 time(s), 2.0s total" in result.output
    assert gates.built[0]["step"] == "segment"


def test_segment_rebuild_waits_too(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, gates: FakeGates
) -> None:
    _write_boundary_prompt(env)

    def run_segment_for_chat(
        conn: Any, chat_id: int, config: Any, provider: Any, prompt: bytes, **kw: Any
    ) -> SegmentationRunReport:
        provider.detect_boundaries(_messages(12))
        return SegmentationRunReport(chat_id=chat_id, segments_written=2)

    monkeypatch.setattr(cli_module, "run_segment_for_chat", run_segment_for_chat)

    result = runner.invoke(
        app, ["segment", "--rebuild", "--chat", "42", "--config", str(env["config"])]
    )

    assert result.exit_code == 0, result.output
    assert gates.calls == {"segment": 1}
    assert "segment: yielded to in-flight searches 1 time(s), 1.0s total" in result.output


def test_sync_waits_in_both_heavy_steps_and_reports_each(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, gates: FakeGates
) -> None:
    _fake_light_steps(env, monkeypatch)
    _write_boundary_prompt(env)

    def run_segment(conn: Any, config: Any, provider: Any, prompt: bytes, **kw: Any) -> Any:
        provider.detect_boundaries(_messages(12))
        return []

    def run_embed(conn: Any, *args: Any, **kwargs: Any) -> EmbedRunReport:
        assert kwargs["stop_check"]() is None
        assert kwargs["stop_check"]() is None
        return EmbedRunReport()

    monkeypatch.setattr(cli_module, "run_segment", run_segment)
    monkeypatch.setattr(cli_module, "run_embed", run_embed)
    monkeypatch.setattr(cli_module, "sync_fts", lambda conn, fts: _FtsReport())

    result = runner.invoke(app, ["sync", "--config", str(env["config"])])

    assert result.exit_code == 0, result.output
    assert gates.calls == {"segment": 1, "embed": 2}
    assert "sync: segmentation yielded to in-flight searches 1 time(s), 1.0s total" in result.output
    assert "sync: embedding yielded to in-flight searches 2 time(s), 3.0s total" in result.output


# --------------------------------------------------------------------------
# no search in flight: the run is as before
# --------------------------------------------------------------------------


def test_nothing_is_said_when_no_search_was_in_flight(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fake database answers the probe with "nobody holds the query
    lock": one probe per unit, no wait, and the output reads as before."""
    conns: list[_FakePgConn] = []

    def connect(database: Any, **kw: Any) -> _FakePgConn:
        conns.append(_FakePgConn())
        return conns[-1]

    monkeypatch.setattr(cli_module, "connect", connect)
    _fake_light_steps(env, monkeypatch)
    _write_boundary_prompt(env)
    boundary_calls: list[int] = []

    def run_segment(conn: Any, config: Any, provider: Any, prompt: bytes, **kw: Any) -> Any:
        boundary_calls.append(len(provider.detect_boundaries(_messages(12))))
        return []

    def run_embed(conn: Any, *args: Any, **kwargs: Any) -> EmbedRunReport:
        assert kwargs["stop_check"]() is None
        return EmbedRunReport(segments_embedded=1)

    monkeypatch.setattr(cli_module, "run_segment", run_segment)
    monkeypatch.setattr(cli_module, "run_embed", run_embed)
    monkeypatch.setattr(cli_module, "sync_fts", lambda conn, fts: _FtsReport())

    result = runner.invoke(app, ["sync", "--config", str(env["config"])])

    assert result.exit_code == 0, result.output
    assert "yielded" not in result.output
    assert len(boundary_calls) == 1
    probes = [sql for conn in conns for sql in conn.statements if "pg_try_advisory_lock(" in sql]
    assert len(probes) == 2  # one before the boundary call, one before the batch


def test_switching_yield_off_skips_the_probe(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    add_config(env, "enrichment:\n  yield_to_queries: false\n")
    conns: list[_FakePgConn] = []

    def connect(database: Any, **kw: Any) -> _FakePgConn:
        conns.append(_FakePgConn())
        return conns[-1]

    monkeypatch.setattr(cli_module, "connect", connect)

    def run_embed(conn: Any, *args: Any, **kwargs: Any) -> EmbedRunReport:
        assert kwargs["stop_check"]() is None
        return EmbedRunReport()

    monkeypatch.setattr(cli_module, "run_embed", run_embed)
    monkeypatch.setattr(cli_module, "sync_fts", lambda conn, fts: _FtsReport())

    result = runner.invoke(app, ["embed", "--config", str(env["config"])])

    assert result.exit_code == 0, result.output
    assert not [sql for conn in conns for sql in conn.statements if "advisory" in sql]


# --------------------------------------------------------------------------
# the pause switch still wins while a step waits
# --------------------------------------------------------------------------


class _SearchAlwaysInFlight(_FakePgConn):
    """A database where a search holds the query lock the whole time; the
    host pause file appears on the `pause_on_probe`-th probe."""

    def __init__(self, pause_file: Path, pause_on_probe: int) -> None:
        super().__init__()
        self.pause_file = pause_file
        self.pause_on_probe = pause_on_probe
        self.probes = 0

    def cursor(self) -> Any:
        conn = self

        class _Cursor:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def execute(self, sql: str, params: Any = None) -> None:
                conn.statements.append(sql)
                if "pg_try_advisory_lock(" in sql:
                    conn.probes += 1
                    if conn.probes == conn.pause_on_probe:
                        conn.pause_file.parent.mkdir(parents=True, exist_ok=True)
                        conn.pause_file.write_text("reason=photo import\n")

            def fetchone(self) -> tuple[Any, ...]:
                last = conn.statements[-1]
                # The exclusive probe fails: a search holds the shared lock.
                return (False,) if "pg_try_advisory_lock(" in last else (True,)

        return _Cursor()


def test_a_pause_set_while_embedding_waits_for_a_search_stops_it_at_once(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Searches never stop here, so without the pause check inside the wait
    the command would sit out the whole 300 s bound before noticing."""
    conn = _SearchAlwaysInFlight(env["host_pause_file"], pause_on_probe=3)
    monkeypatch.setattr(cli_module, "connect", lambda database, **kw: conn)
    add_config(env, "enrichment:\n  yield_poll_interval_seconds: 0.01\n")
    background_gate = importlib.import_module("imsg.background_gate")

    def run_embed(db: Any, *args: Any, **kwargs: Any) -> EmbedRunReport:
        reason = kwargs["stop_check"]()
        assert reason is not None
        raise BackgroundWorkDeferred(reason, partial=EmbedRunReport(segments_embedded=4))

    monkeypatch.setattr(cli_module, "run_embed", run_embed)
    monkeypatch.setattr(cli_module, "sync_fts", lambda db, fts: _FtsReport())

    started = time.monotonic()
    result = runner.invoke(app, ["embed", "--config", str(env["config"])])

    assert result.exit_code == background_gate.EXIT_DEFERRED_PAUSED, result.output
    assert time.monotonic() - started < 30
    assert "embed: deferred: paused — heavy background work is paused: photo import" in result.output
    assert "embed: yielded to in-flight searches 1 time(s)" in result.output
    assert conn.probes == 3  # it stopped waiting on the poll that saw the pause


# --------------------------------------------------------------------------
# imsg status
# --------------------------------------------------------------------------


def test_status_says_which_step_is_yielding_right_now(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_status_probes(monkeypatch)
    state = YieldState(
        query_in_flight=True, enrichment_paused=False, segment_yielding=False, embed_yielding=True
    )
    monkeypatch.setattr(cli_module, "check_enrichment_yield", lambda config: state)

    result = runner.invoke(app, ["status", "--config", str(env["config"]), "--json"])

    payload = json.loads(result.output)
    assert payload["query_in_flight"] is True
    assert payload["embed_yielding_now"] is True
    assert payload["segment_yielding_now"] is False
    assert payload["enrichment_yielding_now"] is False


def test_status_leaves_the_steps_unknown_when_the_database_is_down(
    env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from imsg.diagnostics import PostgresCheck

    _patch_status_probes(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "check_postgres",
        lambda config: PostgresCheck(reachable=False, cluster_fingerprint_ok=None, reason="down"),
    )
    result = runner.invoke(app, ["status", "--config", str(env["config"]), "--json"])
    payload = json.loads(result.output)
    assert payload["segment_yielding_now"] is None
    assert payload["embed_yielding_now"] is None
