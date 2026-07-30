"""`seg_config_hash` / `stable_key` (SPEC §7.2, D4) — the freeze mechanism."""

from __future__ import annotations

from imsg.segment.hashing import compute_seg_config_hash, compute_stable_key

_KWARGS = {
    "session_gap_hours": 3.0,
    "topical_min_messages": 10,
    "max_messages": 50,
    "max_tokens": 2000,
    "boundary_model": "qwen3.5-35b-a3b-4bit",
    "boundary_prompt_bytes": b"detect topic boundaries",
    "index_unsent": False,
    "index_edit_history": False,
}


def test_hash_is_deterministic() -> None:
    assert compute_seg_config_hash(**_KWARGS) == compute_seg_config_hash(**_KWARGS)  # type: ignore[arg-type]


def test_hash_changes_with_session_gap_hours() -> None:
    a = compute_seg_config_hash(**_KWARGS)  # type: ignore[arg-type]
    kwargs = dict(_KWARGS, session_gap_hours=4.0)
    b = compute_seg_config_hash(**kwargs)  # type: ignore[arg-type]
    assert a != b


def test_hash_changes_with_policy_flags() -> None:
    a = compute_seg_config_hash(**_KWARGS)  # type: ignore[arg-type]
    kwargs = dict(_KWARGS, index_unsent=True)
    b = compute_seg_config_hash(**kwargs)  # type: ignore[arg-type]
    assert a != b


def test_hash_changes_with_prompt_bytes() -> None:
    a = compute_seg_config_hash(**_KWARGS)  # type: ignore[arg-type]
    kwargs = dict(_KWARGS, boundary_prompt_bytes=b"a completely different prompt")
    b = compute_seg_config_hash(**kwargs)  # type: ignore[arg-type]
    assert a != b


def test_hash_changes_with_boundary_model() -> None:
    a = compute_seg_config_hash(**_KWARGS)  # type: ignore[arg-type]
    kwargs = dict(_KWARGS, boundary_model="a-different-model")
    b = compute_seg_config_hash(**kwargs)  # type: ignore[arg-type]
    assert a != b


def test_stable_key_is_deterministic_and_content_addressed() -> None:
    h = compute_seg_config_hash(**_KWARGS)  # type: ignore[arg-type]
    a = compute_stable_key(
        chat_source_guid="chat-1", first_message_guid="m1", last_message_guid="m5", seg_config_hash=h
    )
    b = compute_stable_key(
        chat_source_guid="chat-1", first_message_guid="m1", last_message_guid="m5", seg_config_hash=h
    )
    assert a == b
    assert len(a) == 64  # hex sha256


def test_stable_key_changes_with_seg_config_hash() -> None:
    h1 = compute_seg_config_hash(**_KWARGS)  # type: ignore[arg-type]
    h2 = compute_seg_config_hash(**dict(_KWARGS, session_gap_hours=1.0))  # type: ignore[arg-type]
    a = compute_stable_key(
        chat_source_guid="chat-1", first_message_guid="m1", last_message_guid="m5", seg_config_hash=h1
    )
    b = compute_stable_key(
        chat_source_guid="chat-1", first_message_guid="m1", last_message_guid="m5", seg_config_hash=h2
    )
    assert a != b


def test_stable_key_changes_with_message_range() -> None:
    h = compute_seg_config_hash(**_KWARGS)  # type: ignore[arg-type]
    a = compute_stable_key(
        chat_source_guid="chat-1", first_message_guid="m1", last_message_guid="m5", seg_config_hash=h
    )
    b = compute_stable_key(
        chat_source_guid="chat-1", first_message_guid="m1", last_message_guid="m6", seg_config_hash=h
    )
    assert a != b
