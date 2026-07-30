"""The thin, injectable boundary to GCS / Discovery Engine.

This build ships NO production transport: no credentials, no network
client, nothing that could actually push. `push_export` takes an
:class:`ExportTransport` and the only implementation here is
:class:`FakeTransport` (tests, dry runs). The real
GCS-upload + `documents.import` client is a Phase 7 deliverable that
implements this same Protocol — the gate's verification logic must not
change when it lands.

The Protocol is deliberately minimal and expressed in terms the
reconciler needs, not in terms of the Google API surface:

- ``upload_document`` — stage one TXT object into the bucket.
- ``import_documents`` — one incremental `documents.import` batch from
  metadata JSONL entries; returns per-document failures so partial
  import failures can stay `failed` in `export_run_item` and be
  retried (SPEC §11.1).
- ``delete_document`` — remove one document from the data store AND
  its object from the bucket.
- ``document_absent`` — verification primitive for revocation: True
  only when the document is gone from BOTH the data store and the
  bucket (SPEC §11.4/D6: absence is verified by document id, never
  assumed from a successful delete call).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ImportEntry:
    document_id: str
    gcs_object: str
    metadata_jsonl_line: str


@dataclass(frozen=True, slots=True)
class ImportResult:
    failed_document_ids: frozenset[str] = frozenset()


class TransportError(Exception):
    """A transport-level failure for one operation. `push_export`
    records it on the affected item(s) as `failed` — it never aborts
    verification-passed pushes wholesale, so retries stay safe."""


class ExportTransport(Protocol):
    def upload_document(self, *, gcs_object: str, data: bytes) -> str:
        """Upload one staged TXT document; returns the gs:// URI."""
        ...

    def import_documents(self, entries: Sequence[ImportEntry]) -> ImportResult:
        """Run one incremental documents.import batch."""
        ...

    def delete_document(self, *, document_id: str, gcs_object: str) -> None:
        """Delete from the data store and the bucket. Deleting an
        already-absent document must succeed (idempotent)."""
        ...

    def document_absent(self, *, document_id: str, gcs_object: str) -> bool:
        """True only if the document exists in neither the data store
        nor the bucket. Implementations must fail closed: if absence
        cannot be POSITIVELY confirmed, return False."""
        ...


@dataclass
class FakeTransport:
    """In-memory ExportTransport for tests and dry runs.

    Failure knobs let tests exercise every partial-failure path:
    `fail_upload_ids` raise on upload, `fail_import_ids` come back in
    the ImportResult, `fail_delete_ids` raise on delete, and
    `linger_after_delete_ids` simulate a data store that still reports
    the document present after deletion (the revocation-verification
    failure case).
    """

    bucket: str = "fake-bucket"
    objects: dict[str, bytes] = field(default_factory=dict)
    imported: dict[str, str] = field(default_factory=dict)  # document_id -> jsonl line
    fail_upload_ids: set[str] = field(default_factory=set)
    fail_import_ids: set[str] = field(default_factory=set)
    fail_delete_ids: set[str] = field(default_factory=set)
    linger_after_delete_ids: set[str] = field(default_factory=set)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def _doc_id_for_object(self, gcs_object: str) -> str:
        # objects are named segments/<document_id>.txt (documents.gcs_object_for)
        return gcs_object.rsplit("/", 1)[-1].removesuffix(".txt")

    def upload_document(self, *, gcs_object: str, data: bytes) -> str:
        self.calls.append(("upload", gcs_object))
        if self._doc_id_for_object(gcs_object) in self.fail_upload_ids:
            raise TransportError(f"simulated upload failure for {gcs_object}")
        self.objects[gcs_object] = data
        return f"gs://{self.bucket}/{gcs_object}"

    def import_documents(self, entries: Sequence[ImportEntry]) -> ImportResult:
        failed: set[str] = set()
        for entry in entries:
            self.calls.append(("import", entry.document_id))
            if entry.document_id in self.fail_import_ids:
                failed.add(entry.document_id)
                continue
            self.imported[entry.document_id] = entry.metadata_jsonl_line
        return ImportResult(failed_document_ids=frozenset(failed))

    def delete_document(self, *, document_id: str, gcs_object: str) -> None:
        self.calls.append(("delete", document_id))
        if document_id in self.fail_delete_ids:
            raise TransportError(f"simulated delete failure for {document_id}")
        if document_id not in self.linger_after_delete_ids:
            self.imported.pop(document_id, None)
            self.objects.pop(gcs_object, None)

    def document_absent(self, *, document_id: str, gcs_object: str) -> bool:
        self.calls.append(("verify-absent", document_id))
        return document_id not in self.imported and gcs_object not in self.objects


__all__ = [
    "ExportTransport",
    "FakeTransport",
    "ImportEntry",
    "ImportResult",
    "TransportError",
]
