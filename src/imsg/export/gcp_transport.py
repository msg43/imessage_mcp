"""Real `ExportTransport` implementation: GCS batch upload + Discovery
Engine `documents.import` (SPEC §11.1).

**UNVERIFIED AGAINST THE LIVE API.** This module has never been
exercised against a real GCS bucket or Discovery Engine data store.
Its first real exercise is a Phase 7 deliverable (SPEC §12 AT-5),
gated behind the pre-push review protocol. Do not treat anything here
— request shapes, field names, error-handling choices — as confirmed
until that happens; treat it as "compiles against the documented API
surface and the installed client library," nothing stronger.

Nothing in this build's test suite calls a Google API over the
network: `tests/test_export_gcp_transport.py` exercises this class
against small hand-written stand-in objects (duck-typed to the
`storage.Client`/`DocumentServiceClient` methods actually called), so
every assertion is about *what request this code builds and how it
interprets a response*, never about live behavior. Constructing a
`GcsDiscoveryEngineTransport` does not itself make a network call
(the underlying Google clients are lazy), and this module is never
imported by `imsg.cli` or wired into any command in this build — see
the module-level scope note in the build's own task description:
"implement the real GCS + Discovery Engine transport behind the
existing Protocol" is explicitly *not* "wire it up to run."

**No credentials in the repo, ever.** `resolve_gcp_credentials` reads
a service-account key's raw JSON from an `imsg.config.secrets.
SecretRef` (env or Keychain only, per that module's contract — never a
literal, never `config.yaml`). There is no default reference baked in
anywhere in this module: a transport only ever gets built by a caller
that explicitly names where the credential lives, so nothing here can
resolve a credential by accident, and `config.schema.ExportConfig`
(out of this build's scope to modify) carries no credential field for
the same reason.

Design note on `import_documents` — SPEC §11.1's GCS-batch-import path:
each `push_export` call already uploaded every document's TXT body via
`upload_document` (one GCS object per document, matching
`imsg.export.documents.gcs_object_for`); `import_documents` writes the
*metadata* JSONL lines it's given to one more GCS object and points
Discovery Engine's `documents.import` at that object
(`data_schema="document"`, since each line is already a full Document
JSON — id, structData, content.uri — per `imsg.export.documents.
metadata_jsonl_line`). Discovery Engine's `ImportDocumentsResponse`
reports only aggregate `success_count`/`failure_count`; `error_samples`
are a handful of sampled `google.rpc.Status` entries with no
structured document-id linkage. Rather than parse error message text
and guess, this fails closed at the *batch* level: any `failure_count
> 0` reports every document in that batch as failed to `push_export`,
which retries the whole batch — safe because re-upserting an
already-succeeded document is idempotent (SPEC §11.3: document ids are
content-independent, a re-push updates in place). A Phase 7 follow-up
can configure `ImportErrorConfig.gcs_prefix` and parse Discovery
Engine's per-line error output, once its real shape is confirmed live,
to narrow the retry set to only the documents that actually failed.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import google.cloud.storage as storage
from google.api_core.exceptions import GoogleAPIError, NotFound
from google.cloud import discoveryengine_v1 as discoveryengine
from google.oauth2 import service_account

from imsg.config.secrets import SecretRef
from imsg.export.transport import ImportEntry, ImportResult, TransportError

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

_DEFAULT_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def resolve_gcp_credentials(
    ref: SecretRef, *, scopes: Sequence[str] = _DEFAULT_SCOPES
) -> Credentials:
    """Resolve GCP service-account credentials from `ref` — env or
    Keychain only (`imsg.config.secrets.SecretRef`'s contract; e.g.
    `keychain:imsgindex-gcp-credentials` or
    `env:IMSG_GCP_CREDENTIALS_JSON`). The resolved secret value must be
    the raw JSON text of a service-account key. Never reads
    `config.yaml`, never embeds a default item name — the caller always
    names the reference explicitly.
    """
    raw = ref.resolve()
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TransportError(
            f"GCP credential secret '{ref.raw}' did not resolve to valid JSON "
            f"(expected a service-account key) — {exc}"
        ) from exc
    try:
        # google-auth's `from_service_account_info` carries no type
        # annotations even though the `oauth2` package ships py.typed —
        # same "typed package, untyped leaf function" situation as
        # `Operation.result()` below.
        creds: Credentials = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
            info, scopes=list(scopes)
        )
    except (ValueError, KeyError) as exc:
        raise TransportError(
            f"GCP credential secret '{ref.raw}' is not a valid service-account key: {exc}"
        ) from exc
    return creds


@dataclass
class GcsDiscoveryEngineTransport:
    """SPEC §11.1's real transport, behind the same `ExportTransport`
    Protocol as `imsg.export.transport.FakeTransport` — `push_export`'s
    verification logic (SPEC §11.1/§11.4, D9) is identical either way.
    See the module docstring: not exercised against the live API in
    this build.
    """

    gcp_project: str
    gcs_bucket: str
    data_store_id: str
    storage_client: storage.Client
    document_client: discoveryengine.DocumentServiceClient
    location: str = "global"
    branch: str = "default_branch"

    def _bucket(self) -> storage.Bucket:
        bucket: storage.Bucket = self.storage_client.bucket(self.gcs_bucket)
        return bucket

    def _branch_path(self) -> str:
        path: str = self.document_client.branch_path(
            project=self.gcp_project,
            location=self.location,
            data_store=self.data_store_id,
            branch=self.branch,
        )
        return path

    def _document_path(self, document_id: str) -> str:
        path: str = self.document_client.document_path(
            project=self.gcp_project,
            location=self.location,
            data_store=self.data_store_id,
            branch=self.branch,
            document=document_id,
        )
        return path

    # -- ExportTransport -----------------------------------------------

    def upload_document(self, *, gcs_object: str, data: bytes) -> str:
        blob = self._bucket().blob(gcs_object)
        try:
            blob.upload_from_string(data, content_type="text/plain; charset=utf-8")
        except GoogleAPIError as exc:
            raise TransportError(f"GCS upload failed for '{gcs_object}': {exc}") from exc
        return f"gs://{self.gcs_bucket}/{gcs_object}"

    def import_documents(self, entries: Sequence[ImportEntry]) -> ImportResult:
        if not entries:
            return ImportResult()

        batch_object = f"imports/{uuid.uuid4().hex}.jsonl"
        payload = "\n".join(e.metadata_jsonl_line for e in entries) + "\n"
        try:
            self._bucket().blob(batch_object).upload_from_string(
                payload, content_type="application/json"
            )
        except GoogleAPIError as exc:
            raise TransportError(f"GCS metadata batch upload failed: {exc}") from exc

        request = discoveryengine.ImportDocumentsRequest(
            parent=self._branch_path(),
            gcs_source=discoveryengine.GcsSource(
                input_uris=[f"gs://{self.gcs_bucket}/{batch_object}"],
                data_schema="document",
            ),
            reconciliation_mode=(
                discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL
            ),
        )
        try:
            operation = self.document_client.import_documents(request=request)
            # `google.api_core.operation.Operation.result` is untyped
            # (same untyped-leaf-function situation as
            # `from_service_account_info` above).
            response = operation.result()  # type: ignore[no-untyped-call]
        except GoogleAPIError as exc:
            raise TransportError(f"Discovery Engine import_documents failed: {exc}") from exc

        # See module docstring: fail-closed, batch-level attribution —
        # Discovery Engine doesn't hand back a structured per-document
        # failure list.
        if getattr(response, "failure_count", 0):
            return ImportResult(failed_document_ids=frozenset(e.document_id for e in entries))
        return ImportResult()

    def delete_document(self, *, document_id: str, gcs_object: str) -> None:
        try:
            self.document_client.delete_document(name=self._document_path(document_id))
        except NotFound:
            pass  # already gone — the Protocol requires this to be idempotent
        except GoogleAPIError as exc:
            raise TransportError(
                f"Discovery Engine delete_document failed for '{document_id}': {exc}"
            ) from exc
        try:
            self._bucket().blob(gcs_object).delete()
        except NotFound:
            pass
        except GoogleAPIError as exc:
            raise TransportError(f"GCS delete failed for '{gcs_object}': {exc}") from exc

    def document_absent(self, *, document_id: str, gcs_object: str) -> bool:
        try:
            self.document_client.get_document(name=self._document_path(document_id))
            return False  # still present in the data store
        except NotFound:
            pass
        except GoogleAPIError:
            return False  # ambiguous — fail closed per the Protocol's contract

        try:
            still_in_bucket: bool = self._bucket().blob(gcs_object).exists()
        except GoogleAPIError:
            return False  # ambiguous — fail closed
        return not still_in_bucket


def build_gcs_discovery_engine_transport(
    *,
    gcp_project: str,
    gcs_bucket: str,
    data_store_id: str,
    credentials: Credentials,
    location: str = "global",
    branch: str = "default_branch",
) -> GcsDiscoveryEngineTransport:
    """Construct the real transport from an already-resolved
    `Credentials` object (see `resolve_gcp_credentials`) and the
    `export.*` values already validated by `Config` (SPEC §6) — this
    function takes them as plain strings rather than a `Config` object
    so it has no dependency on `imsg.config.schema` beyond what the
    caller already read out of it. Constructing the underlying Google
    clients does not itself make a network call.
    """
    storage_client = storage.Client(project=gcp_project, credentials=credentials)
    document_client = discoveryengine.DocumentServiceClient(credentials=credentials)
    return GcsDiscoveryEngineTransport(
        gcp_project=gcp_project,
        gcs_bucket=gcs_bucket,
        data_store_id=data_store_id,
        storage_client=storage_client,
        document_client=document_client,
        location=location,
        branch=branch,
    )


__all__ = [
    "GcsDiscoveryEngineTransport",
    "build_gcs_discovery_engine_transport",
    "resolve_gcp_credentials",
]
