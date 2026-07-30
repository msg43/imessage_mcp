"""Document identity, GCS object naming, metadata structData, and the
canonical manifest encoding (SPEC §11.3, D6).

**Content hashes are version fields, not part of the document id.**
`segment_document_id` depends only on the segment's stable key, and
`attachment_chunk_document_id` only on structural coordinates — a
content change re-uses the same external document id so a re-push
UPDATES the existing Discovery Engine document instead of minting a
duplicate and orphaning the old one (the exact v1.0 bug D6 fixed).

`gcs_object_for` is a pure function of the document id, so deletes can
always be addressed even if a stored `gcs_uri` were lost — no orphaned
objects from unmapped ids.

**External ids are `d` + 62 hex characters = 63 total.** This is not a
cosmetic choice; both properties are load-bearing against a documented
API constraint, resolved 2026-07-30 before any first push (so no
migration was needed — changing this after export begins would break
the stability guarantee above):

- **Length ≤ 63.** Discovery Engine constrains `Document.id` to
  RFC-1034 with a 1-63 character limit, so the spec's bare 64-char
  sha256 would have been rejected on the first push.
- **Leading letter.** RFC-1034's preferred label syntax requires a
  label to *start with a letter*. A bare hex digest beginning with a
  digit (`7a3f…`) violates that at any length, so truncating alone
  would have fixed the length and left a second, subtler rejection in
  place. The `d` prefix removes that class entirely.

248 bits of entropy remain, which puts the birthday bound around 2^124
documents — irrelevant beside a corpus of ~10^6.
"""

from __future__ import annotations

import json
from typing import Any

from imsg.hashing import sha256_text

MANIFEST_FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.jsonl"
REPORT_FILENAME = "report.txt"
DOCS_SUBDIR = "docs"


#: RFC-1034 allows 1-63 characters; one is spent on the leading letter.
_ID_HEX_CHARS = 62
_ID_PREFIX = "d"


def _external_document_id(payload: str) -> str:
    """Derive an RFC-1034-safe external document id from `payload`.

    See this module's docstring: `d` + 62 hex = 63 chars, starts with a
    letter. Both properties are required by Discovery Engine's stated
    `Document.id` constraint; neither is decorative.
    """
    return f"{_ID_PREFIX}{sha256_text(payload)[:_ID_HEX_CHARS]}"


def segment_document_id(stable_key: str) -> str:
    """SPEC §11.3 (as amended, D9): content-independent id for a segment."""
    return _external_document_id(f"segment:{stable_key}")


def attachment_chunk_document_id(
    stable_key: str, attachment_source_guid: str, kind: str, seq: int
) -> str:
    """SPEC §11.3 (as amended, D9): keyed on structural coordinates only."""
    return _external_document_id(
        f"attachment_chunk:{stable_key}:{attachment_source_guid}:{kind}:{seq}"
    )


def gcs_object_for(document_id: str) -> str:
    """Deterministic `segments/<id>.txt` object name (SPEC §11.1)."""
    return f"segments/{document_id}.txt"


def staged_relpath_for(document_id: str) -> str:
    return f"{DOCS_SUBDIR}/{document_id}.txt"


def canonical_json(value: Any) -> str:
    """The single JSON encoding used for everything that gets hashed or
    compared byte-for-byte (manifest, allowlist snapshot, config
    projection). Sorted keys, no whitespace, ASCII-safe — two
    semantically equal values always encode identically."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def structdata_for(
    *,
    people: tuple[str, ...],
    started_at: str,
    ended_at: str,
    segment_key: str,
    document_kind: str,
) -> dict[str, Any]:
    """Metadata carried as Discovery Engine structData for omnibar
    filtering (SPEC §11.3). Everything here is metadata the segment
    document body already exposes — no additional content classes leak
    through this path, and nothing about non-allowlisted people can
    appear (the document would not exist)."""
    return {
        "people": list(people),
        "started_at": started_at,
        "ended_at": ended_at,
        "segment_key": segment_key,
        "document_kind": document_kind,
    }


def metadata_jsonl_line(
    *, document_id: str, gcs_bucket: str, gcs_object: str, struct_data: dict[str, Any]
) -> str:
    """One line of the GCS-import metadata JSONL (SPEC §11.1: GCS batch
    import with TXT content plus a metadata JSONL)."""
    return canonical_json(
        {
            "id": document_id,
            "structData": struct_data,
            "content": {
                "mimeType": "text/plain",
                "uri": f"gs://{gcs_bucket}/{gcs_object}",
            },
        }
    )


__all__ = [
    "DOCS_SUBDIR",
    "MANIFEST_FILENAME",
    "MANIFEST_FORMAT_VERSION",
    "METADATA_FILENAME",
    "REPORT_FILENAME",
    "attachment_chunk_document_id",
    "canonical_json",
    "gcs_object_for",
    "metadata_jsonl_line",
    "segment_document_id",
    "staged_relpath_for",
    "structdata_for",
]
