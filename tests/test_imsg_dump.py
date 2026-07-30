from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from imsg.errors import ImsgDumpError
from imsg.stages.imsg_dump import (
    EditVersion,
    ImsgDumpMessage,
    TapbackInfo,
    _parse_line,
    default_binary_path,
    run_imsg_dump,
)


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    """A real, existing file standing in for the compiled `imsg-dump`
    binary — `run_imsg_dump` checks `binary_path.is_file()` before ever
    consulting `run_process`, so tests need a real path there even
    though `run_process` is what actually gets invoked."""
    path = tmp_path / "imsg-dump"
    path.write_text("")
    return path


def _stand_in_process(
    lines: list[str], *, stderr_lines: list[str] | None = None, exit_code: int = 0
) -> list[str]:
    """Build a `python3 -c "..."` argv that prints `lines` to stdout (one
    NDJSON object per line) and `stderr_lines` to stderr, then exits
    with `exit_code` — a real subprocess standing in for the compiled
    Rust binary so the plumbing (streaming, exit codes, stderr capture)
    is exercised for real, without needing `cargo build` to have run.
    """
    stdout_src = "\n".join(lines)
    stderr_src = "\n".join(stderr_lines or [])
    script = (
        "import sys\n"
        f"sys.stdout.write({stdout_src!r} + ('\\n' if {stdout_src!r} else ''))\n"
        f"sys.stderr.write({stderr_src!r} + ('\\n' if {stderr_src!r} else ''))\n"
        "sys.stdout.flush(); sys.stderr.flush()\n"
        f"sys.exit({exit_code})\n"
    )
    return [sys.executable, "-c", script]


class _ArgvCapturingRunner:
    """A `run_process` stand-in that ignores the real argv it's called
    with and instead runs a fixed stand-in script — while recording the
    argv `run_imsg_dump` built, so tests can assert on it."""

    def __init__(self, lines: list[str], *, stderr_lines: list[str] | None = None, exit_code: int = 0) -> None:
        self.lines = lines
        self.stderr_lines = stderr_lines or []
        self.exit_code = exit_code
        self.seen_argv: list[str] | None = None

    def __call__(self, argv: list[str]) -> subprocess.Popen[str]:
        self.seen_argv = argv
        stand_in = _stand_in_process(self.lines, stderr_lines=self.stderr_lines, exit_code=self.exit_code)
        return subprocess.Popen(stand_in, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


ONE_MESSAGE_LINE = (
    '{"rowid": 1, "guid": "msg-1", "chat_guid": "chat-1", "handle": "+15551234567", '
    '"is_from_me": false, "date": "2024-01-01T00:00:00+00:00", "date_edited": null, '
    '"date_retracted": null, "service": "iMessage", "body_text": "hello", '
    '"edit_history": [], "is_unsent": false, "tapback": null, '
    '"attachment_rowids": [], "reply_to_guid": null}'
)

TAPBACK_LINE = (
    '{"rowid": 2, "guid": "msg-2", "chat_guid": "chat-1", "handle": "+15551234567", '
    '"is_from_me": false, "date": "2024-01-01T00:01:00+00:00", "date_edited": null, '
    '"date_retracted": null, "service": "iMessage", "body_text": null, '
    '"edit_history": [], "is_unsent": false, '
    '"tapback": {"kind": "loved", "target_guid": "msg-1"}, '
    '"attachment_rowids": [], "reply_to_guid": null}'
)

EDITED_LINE = (
    '{"rowid": 3, "guid": "msg-3", "chat_guid": "chat-1", "handle": null, '
    '"is_from_me": true, "date": "2024-01-01T00:02:00+00:00", '
    '"date_edited": "2024-01-01T00:03:00+00:00", "date_retracted": null, '
    '"service": "iMessage", "body_text": "final text", '
    '"edit_history": [{"text": "first draft", "edited_at": "2024-01-01T00:02:30+00:00"}], '
    '"is_unsent": false, "tapback": null, "attachment_rowids": [7, 8], "reply_to_guid": "msg-1"}'
)


def test_run_imsg_dump_parses_ordinary_message(fake_binary: Path) -> None:
    runner = _ArgvCapturingRunner([ONE_MESSAGE_LINE])
    result = run_imsg_dump(
        binary_path=fake_binary,
        snapshot_path=Path("/fake/snapshot.db"),
        since_rowid=0,
        run_process=runner,
    )
    assert len(result.messages) == 1
    msg = result.messages[0]
    assert msg == ImsgDumpMessage(
        rowid=1,
        guid="msg-1",
        chat_guid="chat-1",
        handle="+15551234567",
        is_from_me=False,
        date="2024-01-01T00:00:00+00:00",
        date_edited=None,
        date_retracted=None,
        service="iMessage",
        body_text="hello",
        edit_history=(),
        is_unsent=False,
        tapback=None,
        attachment_rowids=(),
        reply_to_guid=None,
    )
    assert result.stderr_lines == ()


def test_run_imsg_dump_parses_tapback_and_edit_history(fake_binary: Path) -> None:
    runner = _ArgvCapturingRunner([TAPBACK_LINE, EDITED_LINE])
    result = run_imsg_dump(
        binary_path=fake_binary,
        snapshot_path=Path("/fake/snapshot.db"),
        since_rowid=0,
        run_process=runner,
    )
    assert len(result.messages) == 2
    tapback_msg, edited_msg = result.messages

    assert tapback_msg.tapback == TapbackInfo(kind="loved", target_guid="msg-1")
    assert tapback_msg.body_text is None

    assert edited_msg.is_from_me is True
    assert edited_msg.reply_to_guid == "msg-1"
    assert edited_msg.attachment_rowids == (7, 8)
    assert edited_msg.edit_history == (
        EditVersion(text="first draft", edited_at="2024-01-01T00:02:30+00:00"),
    )


def test_run_imsg_dump_builds_expected_argv(fake_binary: Path) -> None:
    runner = _ArgvCapturingRunner([])
    run_imsg_dump(
        binary_path=fake_binary,
        snapshot_path=Path("/fake/snapshot.db"),
        since_rowid=42,
        run_process=runner,
    )
    assert runner.seen_argv == [
        str(fake_binary),
        "--db",
        "/fake/snapshot.db",
        "--since-rowid",
        "42",
    ]


def test_run_imsg_dump_captures_stderr_without_raising_on_success(fake_binary: Path) -> None:
    runner = _ArgvCapturingRunner(
        [ONE_MESSAGE_LINE], stderr_lines=["warn: could not decode typedstream for guid=weird-1"]
    )
    result = run_imsg_dump(
        binary_path=fake_binary,
        snapshot_path=Path("/fake/snapshot.db"),
        since_rowid=0,
        run_process=runner,
    )
    assert len(result.messages) == 1
    assert result.stderr_lines == ("warn: could not decode typedstream for guid=weird-1",)


def test_run_imsg_dump_raises_on_nonzero_exit(fake_binary: Path) -> None:
    runner = _ArgvCapturingRunner([], stderr_lines=["fatal: could not open database"], exit_code=1)
    with pytest.raises(ImsgDumpError, match="exited 1"):
        run_imsg_dump(
            binary_path=fake_binary,
            snapshot_path=Path("/fake/snapshot.db"),
            since_rowid=0,
            run_process=runner,
        )


def test_run_imsg_dump_raises_on_malformed_ndjson_line(fake_binary: Path) -> None:
    runner = _ArgvCapturingRunner(["not json at all"])
    with pytest.raises(ImsgDumpError, match="non-JSON"):
        run_imsg_dump(
            binary_path=fake_binary,
            snapshot_path=Path("/fake/snapshot.db"),
            since_rowid=0,
            run_process=runner,
        )


def test_run_imsg_dump_raises_when_binary_missing(tmp_path: Path) -> None:
    with pytest.raises(ImsgDumpError, match="not found"):
        run_imsg_dump(
            binary_path=tmp_path / "does-not-exist",
            snapshot_path=Path("/fake/snapshot.db"),
            since_rowid=0,
        )


def test_parse_line_rejects_missing_required_field() -> None:
    with pytest.raises(ImsgDumpError, match="missing required field"):
        _parse_line('{"guid": "msg-1"}')


def test_parse_line_rejects_malformed_tapback_object() -> None:
    with pytest.raises(ImsgDumpError, match="tapback"):
        _parse_line(
            '{"rowid": 1, "guid": "msg-1", "tapback": {"kind": "loved"}}'
        )


def test_default_binary_path_prefers_release_over_debug(tmp_path: Path) -> None:
    release = tmp_path / "tools" / "imsg-dump" / "target" / "release" / "imsg-dump"
    debug = tmp_path / "tools" / "imsg-dump" / "target" / "debug" / "imsg-dump"
    debug.parent.mkdir(parents=True)
    debug.write_text("")

    # Only debug exists -> debug wins.
    assert default_binary_path(tmp_path) == debug

    release.parent.mkdir(parents=True)
    release.write_text("")
    # Both exist -> release wins.
    assert default_binary_path(tmp_path) == release


def test_default_binary_path_returns_release_path_when_neither_exists(tmp_path: Path) -> None:
    expected = tmp_path / "tools" / "imsg-dump" / "target" / "release" / "imsg-dump"
    assert default_binary_path(tmp_path) == expected
