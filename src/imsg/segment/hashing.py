"""`seg_config_hash` and `segment.stable_key` (SPEC §7.2, §8 S4, D4).

`seg_config_hash` is the D4 freeze mechanism: it is stamped on every
segment, and changing any input covered here forces re-segmentation
rather than silently mixing segments built under different rules —
"covers thresholds, policy flags, prompt bytes, boundary-model
revision and renderer version — not only numeric limits" (D4/D6).

`stable_key` is the MCP-facing `segment_key` (SPEC §10.2: "`segment_key`
is `segment.stable_key`") — no separate opaque-key derivation needed
here, unlike `message_key`/`thread_key`/`attachment_key` in
`imsg.keys`, because the formula below already is one.
"""

from __future__ import annotations

from imsg.hashing import sha256_text
from imsg.segment.boundaries import (
    BOUNDARY_WINDOW_OVERLAP_MESSAGES,
    BOUNDARY_WINDOW_TOKENS,
    MIN_MESSAGES_PER_SEGMENT,
)
from imsg.segment.render import RENDERER_VERSION


def compute_seg_config_hash(
    *,
    session_gap_hours: float,
    topical_min_messages: int,
    max_messages: int,
    max_tokens: int,
    boundary_model: str,
    boundary_prompt_bytes: bytes,
    index_unsent: bool,
    index_edit_history: bool,
) -> str:
    """One hash covering every input that changes segment *content* or
    *membership*: the numeric D4 thresholds, the D1 policy flags, the
    exact boundary prompt bytes, the boundary model identifier, the
    renderer's own version, and the fixed (non-configurable) windowing
    constants — any of these changing must force re-segmentation.
    """
    payload = " ".join(
        [
            "seg_config_hash/v1",
            f"session_gap_hours={session_gap_hours!r}",
            f"topical_min_messages={topical_min_messages}",
            f"max_messages={max_messages}",
            f"max_tokens={max_tokens}",
            f"boundary_model={boundary_model}",
            f"boundary_prompt_sha256={sha256_text(boundary_prompt_bytes.decode('utf-8', 'replace'))}",
            f"index_unsent={index_unsent}",
            f"index_edit_history={index_edit_history}",
            f"renderer_version={RENDERER_VERSION}",
            f"boundary_window_tokens={BOUNDARY_WINDOW_TOKENS}",
            f"boundary_window_overlap_messages={BOUNDARY_WINDOW_OVERLAP_MESSAGES}",
            f"min_messages_per_segment={MIN_MESSAGES_PER_SEGMENT}",
        ]
    )
    return sha256_text(payload)


def compute_stable_key(
    *,
    chat_source_guid: str,
    first_message_guid: str,
    last_message_guid: str,
    seg_config_hash: str,
) -> str:
    """`sha256(chat.source_guid || first_msg_guid || last_msg_guid || seg_config_hash)`
    (SPEC §7.2 `segment.stable_key` column comment, verbatim formula)."""
    return sha256_text(
        chat_source_guid + first_message_guid + last_message_guid + seg_config_hash
    )


__all__ = ["compute_seg_config_hash", "compute_stable_key"]
