"""Which merge rule an extraction run gets (owner decision D12, 2026-09-23).

Only this machine's own `paths.live_chat_db`, copied by S1, is
`MergeMode.LIVE` -- the one source allowed to replace a non-empty value
(a renamed chat, an attachment the Messages app moved). Everything else
is `MergeMode.SEED`, which only inserts and fills: a `--snapshot` seed,
a configured source that points at another Mac's database, and any case
the caller cannot vouch for. No database needed; `imsg extract`'s own
wiring is covered in `test_cli.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import imsg.stages.extract as extract_mod
from conftest import ConfigDictFactory
from imsg.config.loader import load_config_dict
from imsg.config.schema import Config
from imsg.stages.extract import ExtractResult, MergeMode
from imsg.stages.identity import ContactsImportOutcome, IdentityResult, InvariantReport
from imsg.stages.snapshot import SnapshotResult
from imsg.stages.sync import run_sync


def test_only_the_live_database_itself_is_live(tmp_path: Path) -> None:
    """Compared as the seed guard compares: resolved path, then inode, so
    a symlink or a hard link to the live database is still the live
    database, and a copy of it is not."""
    live = tmp_path / "Messages" / "chat.db"
    live.parent.mkdir()
    live.write_text("")
    symlink = tmp_path / "alias.db"
    symlink.symlink_to(live)
    hard_link = tmp_path / "hard-link.db"
    os.link(live, hard_link)
    copy_of_another_mac = tmp_path / "studio-chat.db"
    copy_of_another_mac.write_text("")

    mode_for = extract_mod.merge_mode_for_source
    assert mode_for(live, live) is MergeMode.LIVE
    assert mode_for(symlink, live) is MergeMode.LIVE
    assert mode_for(hard_link, live) is MergeMode.LIVE
    assert mode_for(copy_of_another_mac, live) is MergeMode.SEED
    assert mode_for(tmp_path / "missing.db", live) is MergeMode.SEED
    assert mode_for(None, live) is MergeMode.SEED


# --- run_sync ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_guard_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    import imsg.stages.sync as sync_mod

    monkeypatch.setattr(sync_mod, "guard_mount", lambda data_root: None)


def _extract_result() -> ExtractResult:
    return ExtractResult(
        run_id=1, watermark_before=0, watermark_after=1, chats_upserted=0,
        handles_upserted=0, messages_upserted=0, tapbacks_upserted=0,
        system_messages_skipped=0, attachments_upserted=0, link_previews_upserted=0,
        bodies_missing=0, dump_stderr_line_count=0,
    )


def _identity_result() -> IdentityResult:
    return IdentityResult(
        source_handles_processed=0, persons_created=0, handles_created=0,
        messages_resolved=0, tapbacks_resolved=0, chat_participants_resolved=0,
        contacts=ContactsImportOutcome(
            attempted=False, contacts_loaded=0, degraded=False, degraded_reason=None
        ),
        invariant=InvariantReport(
            unresolved_message_senders=0, unresolved_tapback_senders=0,
            unresolved_chat_participants=0, owner_person_count=1,
        ),
    )


def _mode_run_sync_passes(config: Config, tmp_path: Path, **kwargs: Any) -> object:
    captured: dict[str, Any] = {}

    def fake_extract(**kw: Any) -> ExtractResult:
        captured.update(kw)
        return _extract_result()

    run_sync(
        conn=object(),  # type: ignore[arg-type]
        config=config,
        imsg_dump_binary=tmp_path / "imsg-dump",
        run_snapshot_fn=lambda **kw: SnapshotResult(
            path=tmp_path / "snapshot.db", sha256="a" * 64, byte_size=1, reused_existing=False
        ),
        run_extract_fn=fake_extract,
        run_identity_fn=lambda **kw: _identity_result(),
        **kwargs,
    )
    return captured["merge_mode"]


def test_syncing_the_live_source_is_live(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    config = load_config_dict(config_dict_factory(), source="<test>")

    assert _mode_run_sync_passes(config, tmp_path, source_name="mini") is MergeMode.LIVE


def test_syncing_a_transferred_snapshot_is_a_seed(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    config = load_config_dict(config_dict_factory(), source="<test>")
    transferred = tmp_path / "studio-seed.db"
    transferred.write_text("")

    mode = _mode_run_sync_passes(
        config, tmp_path, source_name="studio-seed", snapshot_override=transferred
    )

    assert mode is MergeMode.SEED


def test_syncing_a_configured_copy_of_another_mac_is_a_seed(
    config_dict_factory: ConfigDictFactory, tmp_path: Path
) -> None:
    """Being listed in `sync.sources` is not what makes a source live:
    `config.example.yaml` shows a transferred Studio snapshot configured
    there, and that witness must only add."""
    studio = tmp_path / "studio" / "chat.db"
    studio.parent.mkdir()
    studio.write_text("")
    config_dict = config_dict_factory()
    config_dict["sync"]["sources"].append({"name": "studio", "chat_db": str(studio)})
    config = load_config_dict(config_dict, source="<test>")

    assert _mode_run_sync_passes(config, tmp_path, source_name="studio") is MergeMode.SEED
