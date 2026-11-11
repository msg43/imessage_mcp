"""`scripts/run_night_captions.sh` must not mistake "nothing due right now" for
"nothing left to caption".

On 2026-11-11 the window opened at 01:00, ran one chunk at 02:18, claimed
nothing, logged "done: no caption left to claim" and released the models at
02:21 — with 69,719 captions still pending. They were pending but *deferred*:
every one carried `next_attempt_at` 02:42, 24 minutes later, because the
caption experiment had parked the queue. An empty claim is what a deferred
queue looks like, so the job threw away 6.5 hours of GPU window.

The script takes its `imsg` and `psql` from the environment, so these tests
drive the real script with fakes for both and read its progress log. No
database and no models.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_night_captions.sh"


@pytest.fixture
def rig(tmp_path: Path) -> dict[str, Path]:
    """A fake `imsg` and a fake `psql`, both scriptable from files.

    `imsg enrich` prints the claimed= line from `claimed`, one line per call,
    falling back to the last line once the lines run out. `psql` prints the
    contents of `queue` for the pending-caption query.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.log"

    imsg = bin_dir / "imsg"
    imsg.write_text(
        "#!/bin/sh\n"
        f'echo "imsg $*" >> {calls}\n'
        'case "$1 $2" in\n'
        "  'mcp hold-models') exit 0 ;;\n"
        "  'mcp release-models') exit 0 ;;\n"
        "esac\n"
        'if [ "$1" = "enrich" ]; then\n'
        f'  n=$(grep -c "^imsg enrich" {calls})\n'
        f'  line=$(sed -n "${{n}}p" {tmp_path / "claimed"} 2>/dev/null)\n'
        f'  [ -z "$line" ] && line=$(tail -1 {tmp_path / "claimed"})\n'
        '  echo "enrich: claimed=$line"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = "embed" ]; then echo "embed: fts events_applied=0"; exit 0; fi\n'
        "exit 0\n"
    )
    imsg.chmod(0o755)

    psql = bin_dir / "psql"
    psql.write_text(
        "#!/bin/sh\n"
        f'echo "psql $*" >> {calls}\n'
        # Two queries reach psql: the link-preview skip UPDATE, and the question
        # "how many captions are pending, and when is the next due". Only the
        # second selects a count, so match on that — and answer it from the next
        # line of `queue`, so a test can let the queue drain between chunks.
        'for a in "$@"; do\n'
        '  case "$a" in\n'
        # the link-preview skip is an UPDATE that also ends in select count(*),
        # so rule it out first or it eats the queue's answers
        '    *"update enrichment"*) echo 0; exit 0 ;;\n'
        "  esac\n"
        "done\n"
        'for a in "$@"; do\n'
        '  case "$a" in\n'
        "    *\"select count(*)\"*)\n"
        f'      echo q >> {tmp_path / "queue.calls"}\n'
        f'      n=$(wc -l < {tmp_path / "queue.calls"})\n'
        f'      line=$(sed -n "$(echo "$n" | tr -d " ")p" {tmp_path / "queue"})\n'
        f'      [ -z "$line" ] && line=$(tail -1 {tmp_path / "queue"})\n'
        '      echo "$line"\n'
        "      exit 0 ;;\n"
        "  esac\n"
        "done\n"
        "echo 0\n"
    )
    psql.chmod(0o755)

    out = tmp_path / "ops"
    (tmp_path / "queue").write_text("0 0\n")
    (tmp_path / "claimed").write_text("0\n")
    return {
        "tmp": tmp_path,
        "imsg": imsg,
        "psql": psql,
        "out": out,
        "calls": calls,
        "claimed": tmp_path / "claimed",
        "queue": tmp_path / "queue",
    }


def _run(
    rig: dict[str, Path], *, last_start: str = "23:59", min_wait: str = "2", timeout: int = 60
):
    env = dict(os.environ)
    env.update(
        IMSG=str(rig["imsg"]),
        PSQL=str(rig["psql"]),
        DATA_ROOT=str(rig["tmp"] / "data"),
        IMSG_CONFIG=str(rig["tmp"] / "config.yaml"),
        NIGHT_CAPTIONS_DIR=str(rig["out"]),
        NIGHT_CAPTIONS_START="00:00",
        NIGHT_CAPTIONS_LAST_START=last_start,
        NIGHT_CAPTIONS_END="23:59",
        NIGHT_CAPTIONS_UNLOAD_WAIT="0",
        NIGHT_CAPTIONS_CHUNK="250",
        NIGHT_CAPTIONS_MIN_WAIT=min_wait,
    )
    proc = subprocess.run(
        ["/bin/sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=timeout
    )
    log = (rig["out"] / "progress.log").read_text() if (rig["out"] / "progress.log").exists() else ""
    return proc, log


def _enrich_calls(rig: dict[str, Path]) -> int:
    if not rig["calls"].exists():
        return 0
    return sum(1 for line in rig["calls"].read_text().splitlines() if line.startswith("imsg enrich"))


# --- the regression ---------------------------------------------------------


def test_an_empty_claim_with_pending_captions_waits_instead_of_ending_the_window(
    rig: dict[str, Path],
) -> None:
    """The 2026-11-11 case: claimed=0 while 69,719 sit pending, due shortly.

    The job must wait and try again, not declare the queue empty. The claimed
    script gives 0 twice, so the second attempt also claims nothing and the
    window closes at LAST_START — what matters is that it waited at all.
    """
    rig["claimed"].write_text("0\n")
    rig["queue"].write_text("69719 2\n")

    import datetime

    soon = (datetime.datetime.now() + datetime.timedelta(minutes=1)).strftime("%H:%M")
    proc, log = _run(rig, last_start=soon, timeout=180)

    assert proc.returncode == 0, proc.stderr
    assert "done: no caption left to claim" not in log, (
        "a deferred queue is not an empty one:\n" + log
    )
    assert "still pending; waiting" in log, log
    assert "69719 caption(s) still pending" in log, log


def test_an_empty_claim_with_an_empty_queue_still_ends_the_window(
    rig: dict[str, Path],
) -> None:
    """The case the old code got right must keep working."""
    rig["claimed"].write_text("0\n")
    rig["queue"].write_text("0 0\n")

    proc, log = _run(rig)

    assert proc.returncode == 0, proc.stderr
    assert "done: no caption left to claim" in log, log
    assert "waiting" not in log, log
    assert _enrich_calls(rig) == 1, "one chunk, then stop"


def test_it_resumes_captioning_once_the_deferred_work_falls_due(
    rig: dict[str, Path],
) -> None:
    """Waiting is only worth anything if the next chunk actually claims.

    First chunk claims 0 with work pending, then the queue reports work due
    now and the second chunk claims 250, then the queue empties.
    """
    rig["claimed"].write_text("0\n250\n0\n")
    # pending on the first ask, drained by the time the second chunk has run
    rig["queue"].write_text("69719 0\n0 0\n")

    proc, log = _run(rig, timeout=180)

    assert proc.returncode == 0, proc.stderr
    assert _enrich_calls(rig) >= 2, "it must try again after waiting:\n" + log
    assert "claimed=250" in log, log


def test_past_last_start_it_stops_even_with_work_pending(rig: dict[str, Path]) -> None:
    """It must not wait past the window's last start."""
    rig["claimed"].write_text("0\n")
    rig["queue"].write_text("69719 3600\n")

    import datetime

    just_past = (datetime.datetime.now() - datetime.timedelta(minutes=1)).strftime("%H:%M")
    # the loop head refuses to start a chunk at all past LAST_START
    proc, log = _run(rig, last_start=just_past)

    assert proc.returncode == 3, f"outside the window: {proc.returncode}\n{log}"
    assert "outside" in log, log


def test_an_unreadable_queue_stops_rather_than_spinning(rig: dict[str, Path]) -> None:
    """If psql cannot answer, the job must not loop forever on a bad read."""
    rig["claimed"].write_text("0\n")
    rig["queue"].write_text("ERROR: could not connect\n")

    proc, log = _run(rig)

    assert proc.returncode == 0, proc.stderr
    assert "could not read the caption queue" in log, log
    assert _enrich_calls(rig) == 1, "it stopped after one chunk"
