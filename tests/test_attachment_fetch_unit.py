"""The attachment fetcher's pure parts (D13), no database needed: path
rules, catalog codes, the read-only copy commands, the pushing host's
root rule, the wire formats, and the config section.

The guarantee these pin down is that no fetch step can write to a source:
every copy command's source side is rsync's sender, no flag that removes or
deletes anything can reach rsync, a copy list cannot climb out of its
root, a remote path cannot be misread by a shell, and a pushing host can
report on a copy but never declare it fetched.

Fictional content only (D5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from imsg.backfill.locate import catalog_code_for
from imsg.backfill.locations import (
    LocationOutcome,
    MatchQuality,
    fetch_relative_path,
    messages_relative_path,
    name_key,
    safe_relative_path,
)
from imsg.backfill.push import (
    PLAN_FORMAT,
    PushError,
    PushResult,
    parse_plan,
    parse_results,
    parse_root,
    results_to_lines,
)
from imsg.backfill.transfer import (
    DELETE_FLAGS,
    SOURCE_WRITE_FLAGS,
    TransferError,
    assert_read_only,
    pull_command,
    push_command,
)
from imsg.config.loader import load_config_dict
from imsg.errors import ConfigError

GUID = "0AB1C2D3-0000-4000-8000-000000000001"


# --- paths --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "rel"),
    [
        (f"~/Library/Messages/Attachments/0a/10/{GUID}/IMG_0001.heic", f"0a/10/{GUID}/IMG_0001.heic"),
        (f"/Users/alice/Library/Messages/Attachments/0a/10/{GUID}/a.jpg", f"0a/10/{GUID}/a.jpg"),
        (f"Backups/x/Users/alice/Library/Messages/Attachments/1/{GUID}/a.jpg", f"1/{GUID}/a.jpg"),
        ("/private/var/folders/zz/T/IMG_0001.jpeg", None),
        ("~/Library/Messages/StickerCache/abc/sticker.heic", None),
        ("~/Library/Messages/Attachments/../chat.db", None),
    ],
)
def test_messages_relative_path(path: str, rel: str | None) -> None:
    assert messages_relative_path(path) == rel


def test_fetch_relative_path_tells_host_paths_from_drive_paths() -> None:
    assert fetch_relative_path(f"~/Library/Messages/Attachments/0a/{GUID}/a.jpg") == f"0a/{GUID}/a.jpg"
    assert fetch_relative_path(f"Photos/{GUID}/a.jpg") == f"Photos/{GUID}/a.jpg"
    assert fetch_relative_path("/private/tmp/a.jpg") is None
    assert fetch_relative_path("Photos/../../etc/passwd") is None


@pytest.mark.parametrize("rel", ["", "/a", "a/../b", "a//b", "./a", "a/\x00b"])
def test_unsafe_relative_paths(rel: str) -> None:
    assert not safe_relative_path(rel)


def test_names_compare_case_blind_and_unicode_composed_alike() -> None:
    composed = "Café.JPG"
    decomposed = "Café.jpg"
    assert name_key(composed) == name_key(decomposed)


def test_match_quality_order_and_the_flag() -> None:
    assert [m.rank for m in MatchQuality] == [0, 1, 2]
    assert [m.flagged for m in MatchQuality] == [False, False, True]
    assert "name_only" not in {m.value for m in MatchQuality}


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("D-TEST1.files.tsv.gz", "D-TEST1"),
        ("V-TEST2.files.2.tsv.gz", "V-TEST2"),
        ("N-TEST3.files.tsv", "N-TEST3"),
        ("notes.tsv.gz", None),
    ],
)
def test_catalog_code_from_its_name(name: str, code: str | None) -> None:
    assert catalog_code_for(Path("/catalogs") / name) == code


def test_catalog_code_from_a_sweep_run_directory() -> None:
    run = Path("/runs/20260101T000000000000Z_Host-0a1b2c_V-TEST9_0123456789abcdef0123456789abcdef")
    assert catalog_code_for(run / "catalog.files.2.tsv.gz") == "V-TEST9"


# --- no fetch step writes to a source ----------------------------------------


def test_every_copy_command_reads_its_source_and_writes_only_to_staging(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "data_root" / "attachment-staging"
    push = push_command(rsync="rsync", ssh="ssh -o BatchMode=yes", local_root=tmp_path / "src",
                        ssh_host="indexhost", remote_staging_dir=f"{staging}/studio",
                        relpaths=["a/b.jpeg"])
    pull = pull_command(rsync="rsync", ssh="ssh -o BatchMode=yes", ssh_host="nashost",
                        remote_root="/volume1/homes", local_staging_dir=staging / "N-TEST3",
                        relpaths=["alice/c.jpeg"])
    for command, source, destination in (
        (push, f"{tmp_path / 'src'}/", f"indexhost:{staging}/studio/"),
        (pull, "nashost:/volume1/homes/", f"{staging / 'N-TEST3'}/"),
    ):
        flags = {arg.split("=", 1)[0] for arg in command.argv}
        assert not flags & (SOURCE_WRITE_FLAGS | DELETE_FLAGS)
        # rsync's last two arguments are source then destination: the
        # source is the side that is read (rsync's sender), the destination
        # is always a staging directory.
        assert command.argv[-2:] == (source, destination)
        assert command.destination == destination
        assert "--files-from=-" in command.argv and "--from0" in command.argv
        assert not any(a in command.argv for a in ("--links", "-l", "--copy-links", "-L", "-a"))


@pytest.mark.parametrize(
    "flag", sorted(SOURCE_WRITE_FLAGS | DELETE_FLAGS | {"--remove-source-files=yes"})
)
def test_a_copy_that_could_write_to_a_source_is_refused(flag: str) -> None:
    with pytest.raises(TransferError):
        assert_read_only(["rsync", "--from0", flag, "src/", "dst/"])


@pytest.mark.parametrize("rel", ["../escape", "/abs/path", "a/../../b", "a//b", ""])
def test_a_copy_list_cannot_climb_out_of_its_root(tmp_path: Path, rel: str) -> None:
    with pytest.raises(TransferError):
        push_command(rsync="rsync", ssh="ssh", local_root=tmp_path, ssh_host="h",
                     remote_staging_dir="/staging/studio", relpaths=[rel])


@pytest.mark.parametrize("remote", ["relative/path", "/a b/c", "/a/../b", "/a;rm -rf x", "/a/$HOME"])
def test_a_remote_path_that_a_shell_could_misread_is_refused(tmp_path: Path, remote: str) -> None:
    with pytest.raises(TransferError):
        pull_command(rsync="rsync", ssh="ssh", ssh_host="nashost", remote_root=remote,
                     local_staging_dir=tmp_path, relpaths=["a"])


def test_a_pushing_host_cannot_declare_a_copy_fetched() -> None:
    lines = results_to_lines([PushResult(1, LocationOutcome.FETCHED)], [])
    with pytest.raises(PushError, match="may not report"):
        parse_results(lines)


def test_a_root_may_not_expose_the_messages_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import imsg.backfill.push as push_module

    messages = tmp_path / "home" / "Library" / "Messages"
    (messages / "Attachments").mkdir(parents=True)
    monkeypatch.setattr(push_module, "_MESSAGES_DIR", messages)
    assert parse_root(f"studio={messages / 'Attachments'}").root == messages / "Attachments"
    assert parse_root(f"V-TEST2={tmp_path / 'drive'}").location == "V-TEST2"
    for bad in (f"studio={messages}", f"studio={tmp_path / 'home'}", f"studio={tmp_path}",
                "studio=relative/dir", f"bad code={tmp_path / 'drive'}", "no-equals-sign"):
        with pytest.raises(PushError):
            parse_root(bad)




def test_a_plan_with_an_unsafe_path_is_refused() -> None:
    header = f'{{"format": "{PLAN_FORMAT}", "staging_dir": "/data/attachment-staging", "items": 1}}'
    item = (
        '{"attachment_id": 1, "candidates": [{"location_id": 1, "location": "studio", '
        '"rel": "../../etc/passwd", "match": "recorded_path", "byte_size": null, "sha256": null}]}'
    )
    with pytest.raises(PushError):
        parse_plan([header, item])
    with pytest.raises(PushError):
        parse_plan(['{"format": "something else"}'])


def test_push_results_round_trip() -> None:
    results = [PushResult(7, LocationOutcome.STAGED), PushResult(8, LocationOutcome.ABSENT, "gone")]
    parsed, log = parse_results(results_to_lines(results, ["push studio: exit 23\n..."]))
    assert parsed == results and log == ["push studio: exit 23\n..."]


# --- config -------------------------------------------------------------------


def test_the_attachments_section_is_optional(config_dict_factory: Any) -> None:
    cfg = load_config_dict(config_dict_factory())
    assert cfg.attachments.staging_dir == Path("attachment-staging")
    assert cfg.attachments.pull == []


def test_the_attachments_section_validates(config_dict_factory: Any, data_root: Path) -> None:
    cfg = load_config_dict(config_dict_factory(**{"attachments": {
        "push_locations": ["studio", "V-TEST2"],
        "pull": [{"location": "N-TEST3", "ssh_host": "nashost", "root": "/volume1/homes"}],
    }}))
    assert cfg.attachments.pull[0].root == "/volume1/homes"
    for bad in (
        {"staging_dir": "/elsewhere/staging"},
        {"staging_dir": "../outside"},
        {"push_locations": ["has/slash"]},
        {"pull": [{"location": "N-TEST3", "ssh_host": "nas; rm", "root": "/volume1"}]},
        {"pull": [{"location": "N-TEST3", "ssh_host": "nas", "root": "relative"}]},
        {"pull": [{"location": "N-TEST3", "ssh_host": "nas", "root": "/a b"}]},
        {"unknown_key": True},
    ):
        with pytest.raises(ConfigError):
            load_config_dict(config_dict_factory(**{"attachments": bad}))
