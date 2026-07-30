"""Unit tests for `imsg.export.gcp_transport` — the real GCS +
Discovery Engine `ExportTransport` (SPEC §11.1, Phase 7). No network
access anywhere in this file: every Google client is a small
hand-written stand-in implementing only the methods the transport
actually calls. These tests are about *what request this code builds
and how it interprets a response*, not about live API behavior — see
the module's own docstring for that caveat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from google.api_core.exceptions import NotFound, PermissionDenied

from imsg.config.secrets import SecretRef
from imsg.errors import SecretResolutionError
from imsg.export.gcp_transport import GcsDiscoveryEngineTransport, resolve_gcp_credentials
from imsg.export.transport import ImportEntry, TransportError

# ==========================================================================
# resolve_gcp_credentials — pure logic, no cryptography needed
# ==========================================================================


def test_resolve_gcp_credentials_env_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = {"type": "service_account", "project_id": "example-project"}
    monkeypatch.setenv("IMSG_TEST_GCP_CREDS", json.dumps(fake_info))

    captured: dict[str, Any] = {}

    class _FakeCredentials:
        pass

    def _fake_from_service_account_info(info: dict[str, Any], scopes: list[str]) -> _FakeCredentials:
        captured["info"] = info
        captured["scopes"] = scopes
        return _FakeCredentials()

    monkeypatch.setattr(
        "imsg.export.gcp_transport.service_account.Credentials.from_service_account_info",
        _fake_from_service_account_info,
    )

    ref = SecretRef.parse("env:IMSG_TEST_GCP_CREDS")
    creds = resolve_gcp_credentials(ref)
    assert isinstance(creds, _FakeCredentials)
    assert captured["info"] == fake_info
    assert captured["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]


def test_resolve_gcp_credentials_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMSG_TEST_GCP_CREDS", "not json at all")
    ref = SecretRef.parse("env:IMSG_TEST_GCP_CREDS")
    with pytest.raises(TransportError, match="did not resolve to valid JSON"):
        resolve_gcp_credentials(ref)


def test_resolve_gcp_credentials_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IMSG_TEST_GCP_CREDS_MISSING", raising=False)
    ref = SecretRef.parse("env:IMSG_TEST_GCP_CREDS_MISSING")
    with pytest.raises(SecretResolutionError):
        resolve_gcp_credentials(ref)


def test_resolve_gcp_credentials_wraps_invalid_key_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMSG_TEST_GCP_CREDS", json.dumps({"not": "a real key"}))

    def _raise(*args: object, **kwargs: object) -> None:
        raise ValueError("missing required field 'private_key'")

    monkeypatch.setattr(
        "imsg.export.gcp_transport.service_account.Credentials.from_service_account_info",
        _raise,
    )
    ref = SecretRef.parse("env:IMSG_TEST_GCP_CREDS")
    with pytest.raises(TransportError, match="not a valid service-account key"):
        resolve_gcp_credentials(ref)


# ==========================================================================
# Stand-in Google clients — no network, duck-typed to the methods used
# ==========================================================================


def _not_found(message: str) -> NotFound:
    # google.api_core.exceptions.NotFound's __init__ carries no type
    # annotations even though the api_core package ships py.typed.
    return NotFound(message)  # type: ignore[no-untyped-call]


def _permission_denied(message: str) -> PermissionDenied:
    return PermissionDenied(message)  # type: ignore[no-untyped-call]


class _FakeBlob:
    def __init__(self, bucket: _FakeBucket, name: str) -> None:
        self._bucket = bucket
        self.name = name

    def upload_from_string(self, data: bytes | str, content_type: str) -> None:
        self._bucket.objects[self.name] = data

    def delete(self) -> None:
        if self.name not in self._bucket.objects:
            raise _not_found(f"no such object: {self.name}")
        del self._bucket.objects[self.name]

    def exists(self) -> bool:
        return self.name in self._bucket.objects


@dataclass
class _FakeBucket:
    objects: dict[str, bytes | str] = field(default_factory=dict)

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self, name)


@dataclass
class _FakeStorageClient:
    bucket_obj: _FakeBucket = field(default_factory=_FakeBucket)

    def bucket(self, name: str) -> _FakeBucket:
        return self.bucket_obj


@dataclass
class _FakeDocumentClient:
    """Stands in for `discoveryengine.DocumentServiceClient`."""

    documents: dict[str, object] = field(default_factory=dict)
    import_requests: list[object] = field(default_factory=list)
    import_failure_count: int = 0
    raise_on_get: Exception | None = None
    raise_on_delete: Exception | None = None
    raise_on_import: Exception | None = None

    def branch_path(self, *, project: str, location: str, data_store: str, branch: str) -> str:
        return f"projects/{project}/locations/{location}/dataStores/{data_store}/branches/{branch}"

    def document_path(
        self, *, project: str, location: str, data_store: str, branch: str, document: str
    ) -> str:
        return (
            f"projects/{project}/locations/{location}/dataStores/{data_store}/"
            f"branches/{branch}/documents/{document}"
        )

    def import_documents(self, *, request: object) -> _FakeOperation:
        if self.raise_on_import is not None:
            raise self.raise_on_import
        self.import_requests.append(request)
        return _FakeOperation(failure_count=self.import_failure_count)

    def get_document(self, *, name: str) -> object:
        if self.raise_on_get is not None:
            raise self.raise_on_get
        if name not in self.documents:
            raise _not_found(f"no such document: {name}")
        return self.documents[name]

    def delete_document(self, *, name: str) -> None:
        if self.raise_on_delete is not None:
            raise self.raise_on_delete
        if name not in self.documents:
            raise _not_found(f"no such document: {name}")
        del self.documents[name]


@dataclass
class _FakeResponse:
    failure_count: int = 0


@dataclass
class _FakeOperation:
    failure_count: int = 0

    def result(self) -> _FakeResponse:
        return _FakeResponse(failure_count=self.failure_count)


def _make_transport(
    *, storage_client: _FakeStorageClient | None = None, document_client: _FakeDocumentClient | None = None
) -> GcsDiscoveryEngineTransport:
    return GcsDiscoveryEngineTransport(
        gcp_project="example-project",
        gcs_bucket="example-bucket",
        data_store_id="example-datastore",
        storage_client=storage_client or _FakeStorageClient(),
        document_client=document_client or _FakeDocumentClient(),  # type: ignore[arg-type]
    )


# ==========================================================================
# upload_document
# ==========================================================================


def test_upload_document_writes_to_bucket_and_returns_gs_uri() -> None:
    storage_client = _FakeStorageClient()
    transport = _make_transport(storage_client=storage_client)
    uri = transport.upload_document(gcs_object="segments/dabc123.txt", data=b"hello world")
    assert uri == "gs://example-bucket/segments/dabc123.txt"
    assert storage_client.bucket_obj.objects["segments/dabc123.txt"] == b"hello world"


def test_upload_document_wraps_api_error() -> None:
    class _FailingBlob(_FakeBlob):
        def upload_from_string(self, data: bytes | str, content_type: str) -> None:
            raise _permission_denied("nope")

    class _FailingBucket(_FakeBucket):
        def blob(self, name: str) -> _FakeBlob:
            return _FailingBlob(self, name)

    storage_client = _FakeStorageClient(bucket_obj=_FailingBucket())
    transport = _make_transport(storage_client=storage_client)
    with pytest.raises(TransportError, match="GCS upload failed"):
        transport.upload_document(gcs_object="segments/x.txt", data=b"x")


# ==========================================================================
# import_documents
# ==========================================================================


def test_import_documents_empty_entries_is_a_noop() -> None:
    transport = _make_transport()
    result = transport.import_documents([])
    assert result.failed_document_ids == frozenset()


def test_import_documents_writes_batch_and_calls_import_with_gcs_source() -> None:
    storage_client = _FakeStorageClient()
    document_client = _FakeDocumentClient()
    transport = _make_transport(storage_client=storage_client, document_client=document_client)

    entries = [
        ImportEntry(document_id="d" + "a" * 62, gcs_object="segments/da.txt", metadata_jsonl_line='{"id":"da"}'),
        ImportEntry(document_id="d" + "b" * 62, gcs_object="segments/db.txt", metadata_jsonl_line='{"id":"db"}'),
    ]
    result = transport.import_documents(entries)
    assert result.failed_document_ids == frozenset()

    # exactly one batch object was written, containing both jsonl lines
    batch_objects = [v for k, v in storage_client.bucket_obj.objects.items() if k.startswith("imports/")]
    assert len(batch_objects) == 1
    assert batch_objects[0] == '{"id":"da"}\n{"id":"db"}\n'

    assert len(document_client.import_requests) == 1


def test_import_documents_failure_count_marks_every_entry_failed() -> None:
    """SPEC/module docstring: Discovery Engine gives only an aggregate
    failure_count, so any failure fails the whole batch (safe — a
    retry re-upserts, which is idempotent)."""
    document_client = _FakeDocumentClient(import_failure_count=1)
    transport = _make_transport(document_client=document_client)
    entries = [
        ImportEntry(document_id="d" + "a" * 62, gcs_object="segments/da.txt", metadata_jsonl_line="{}"),
        ImportEntry(document_id="d" + "b" * 62, gcs_object="segments/db.txt", metadata_jsonl_line="{}"),
    ]
    result = transport.import_documents(entries)
    assert result.failed_document_ids == {"d" + "a" * 62, "d" + "b" * 62}


def test_import_documents_wraps_api_error() -> None:
    document_client = _FakeDocumentClient(raise_on_import=_permission_denied("nope"))
    transport = _make_transport(document_client=document_client)
    entries = [ImportEntry(document_id="d" + "a" * 62, gcs_object="segments/da.txt", metadata_jsonl_line="{}")]
    with pytest.raises(TransportError, match="import_documents failed"):
        transport.import_documents(entries)


# ==========================================================================
# delete_document — idempotent per the Protocol's contract
# ==========================================================================


def test_delete_document_removes_from_both_store_and_bucket() -> None:
    document_id = "d" + "c" * 62
    storage_client = _FakeStorageClient()
    storage_client.bucket_obj.objects["segments/dc.txt"] = "content"
    document_client = _FakeDocumentClient()
    document_client.documents[f"projects/example-project/locations/global/dataStores/example-datastore/branches/default_branch/documents/{document_id}"] = object()

    transport = _make_transport(storage_client=storage_client, document_client=document_client)
    transport.delete_document(document_id=document_id, gcs_object="segments/dc.txt")

    assert "segments/dc.txt" not in storage_client.bucket_obj.objects
    assert not document_client.documents


def test_delete_document_already_absent_is_not_an_error() -> None:
    """Idempotent: deleting an already-gone document must succeed."""
    transport = _make_transport()
    transport.delete_document(document_id="d" + "z" * 62, gcs_object="segments/dz.txt")  # no raise


def test_delete_document_wraps_non_notfound_api_error() -> None:
    document_client = _FakeDocumentClient(raise_on_delete=_permission_denied("nope"))
    transport = _make_transport(document_client=document_client)
    with pytest.raises(TransportError, match="delete_document failed"):
        transport.delete_document(document_id="d" + "a" * 62, gcs_object="segments/da.txt")


# ==========================================================================
# document_absent — fail-closed on ambiguity (SPEC §11.4/D6)
# ==========================================================================


def test_document_absent_true_when_gone_from_both() -> None:
    transport = _make_transport()
    assert transport.document_absent(document_id="d" + "a" * 62, gcs_object="segments/da.txt") is True


def test_document_absent_false_when_still_in_data_store() -> None:
    document_id = "d" + "a" * 62
    document_client = _FakeDocumentClient()
    document_client.documents[
        f"projects/example-project/locations/global/dataStores/example-datastore/"
        f"branches/default_branch/documents/{document_id}"
    ] = object()
    transport = _make_transport(document_client=document_client)
    assert transport.document_absent(document_id=document_id, gcs_object="segments/da.txt") is False


def test_document_absent_false_when_still_in_bucket() -> None:
    storage_client = _FakeStorageClient()
    storage_client.bucket_obj.objects["segments/da.txt"] = "still here"
    transport = _make_transport(storage_client=storage_client)
    assert transport.document_absent(document_id="d" + "a" * 62, gcs_object="segments/da.txt") is False


def test_document_absent_fails_closed_on_ambiguous_get_error() -> None:
    """A non-NotFound error from the data store means we could not
    POSITIVELY confirm absence — must return False, not True."""
    document_client = _FakeDocumentClient(raise_on_get=_permission_denied("nope"))
    transport = _make_transport(document_client=document_client)
    assert transport.document_absent(document_id="d" + "a" * 62, gcs_object="segments/da.txt") is False
