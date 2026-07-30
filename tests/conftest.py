from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from imsg.mount.guard import MountInfo


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A tmp dir that looks like a mounted, encrypted, sentinel-bearing data_root."""
    root = tmp_path / "data_root"
    root.mkdir()
    (root / ".imsgindex-volume").write_text("")
    return root


@pytest.fixture
def messages_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the config schema's notion of `~/Library/Messages` into tmp_path.

    Real HOME on a dev/CI box has no `~/Library/Messages`; tests need a
    real (creatable) directory to exercise path-containment logic
    against, and must never touch the developer's actual home
    directory or its Messages folder.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    messages = fake_home / "Library" / "Messages"
    messages.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    # imsg.config.schema.MESSAGES_DIR is computed at import time from
    # Path("~/Library/Messages").expanduser(); patch it directly too so
    # tests are independent of import order relative to the HOME env var.
    import imsg.config.schema as schema_module

    monkeypatch.setattr(schema_module, "MESSAGES_DIR", messages)
    return messages


ConfigDictFactory = Callable[..., dict[str, Any]]


def _base_config_dict(data_root: Path, live_chat_db: Path) -> dict[str, Any]:
    return {
        "paths": {
            "data_root": str(data_root),
            "live_chat_db": str(live_chat_db),
        },
        "database": {
            "dsn": "postgresql://imsg@127.0.0.1:5433/imsgindex",
            "password": "env:IMSG_TEST_PG_PASSWORD",
            "cluster_fingerprint_file": "pg17/.imsgindex-cluster",
        },
        "sync": {
            "interval_seconds": 900,
            "sources": [{"name": "mini", "chat_db": str(live_chat_db)}],
        },
        "identity": {"default_region": "US", "contacts_import": True},
        "policy": {"index_unsent": False, "index_edit_history": False},
        "segmentation": {
            "session_gap_hours": 3.0,
            "topical_min_messages": 10,
            "max_messages": 50,
            "max_tokens": 2000,
            "boundary_model": "qwen3.5-35b-a3b-4bit",
            "boundary_prompt": "prompts/segment_boundaries.txt",
        },
        "enrichment": {
            "egress": "local_only",
            "window": "01:00-07:00",
            "concurrency": {"ocr": 4, "caption": 1, "transcribe": 1, "pdf": 4},
            "max_attempts": 5,
            "video_max_frames": 20,
            "pdf_scanned_threshold_chars_per_page": 50,
            "limits": {
                "max_file_bytes": 1073741824,
                "max_pdf_pages": 1000,
                "max_media_seconds": 14400,
                "task_timeout_seconds": 1800,
                "temp_bytes_per_task": 10737418240,
            },
        },
        "embedding": {
            "model": "Qwen/Qwen3-Embedding-8B",
            "revision": "deadbeef",
            "quantization": "8bit",
            "dim": 2048,
            "batch_size": 32,
            "query_instruction": "Given a personal message search query, retrieve relevant conversation segments",
            "multimodal": {
                "enabled": True,
                "provider": "local",
                "scope": "full",
                "model": "facebook/PE-Core-G14-448",
                "revision": "cafef00d",
                "dim": 1280,
            },
        },
        "retrieval": {
            "k_fts": 100,
            "k_vector": 100,
            "rrf_k": 60,
            "rerank_top": 50,
            "reranker_model": "Qwen/Qwen3-Reranker-8B",
            "reranker_revision": "f00dcafe",
            "default_limit": 10,
        },
        "render": {"timezone": "America/Los_Angeles", "attachment_snippet_chars": 200},
        "mcp": {
            "local": {"enabled": True},
            "public": {
                "enabled": False,
                "bind": "127.0.0.1:8700",
                "scope": "allowlist",
            },
        },
        "export": {
            "gcp_project": "example-project",
            "gcs_bucket": "example-bucket",
            "data_store_id": "example-datastore",
            "format": "txt",
            "batch_max_files": 100000,
        },
        "eval": {"seed_queries": "private/eval/queries.yaml", "runs_dir": "eval/runs"},
        "logging": {"level": "INFO", "allow_content_debug": False},
    }


@pytest.fixture
def config_dict_factory(
    data_root: Path, messages_dir: Path
) -> ConfigDictFactory:
    """Returns a callable that builds a fresh, valid config dict.

    Callers mutate the returned dict for negative-path tests; each call
    returns a deep copy so tests never leak state into each other.
    """
    live_chat_db = messages_dir / "chat.db"
    live_chat_db.write_text("")

    def _factory(**overrides: Any) -> dict[str, Any]:
        base = _base_config_dict(data_root, live_chat_db)
        for dotted_key, value in overrides.items():
            _set_dotted(base, dotted_key, value)
        return copy.deepcopy(base)

    return _factory


def _set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


@pytest.fixture
def encrypted_mount_info() -> Callable[[Path], MountInfo]:
    """A fake `diskutil_info` that reports `path` itself as a mounted, encrypted volume."""

    def _make(path: Path) -> MountInfo:
        return MountInfo(mount_point=path, encrypted=True, volume_name="Data-Encrypted-Test")

    return _make
