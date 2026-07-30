"""S3 — Identity resolution (SPEC §8 S3) — hard requirement 3 (CLAUDE.md).

Resolves every `source_handle` (S2's raw-provenance staging rows) to a
canonical `person_id`, backfills `message.sender_person_id`,
`tapback.sender_person_id`, and `chat_participant`, and computes the
invariant report S4 must check before it is allowed to run at all:
"nothing keys off a raw handle" only holds if every in-scope sender and
participant has actually been resolved.

**Contacts import (D6, supersedes `BACKFILL.md` §4)**: bootstrapping
names via `CNContactStore`/`CNContactFetchRequest` — the supported
Contacts framework API, never a direct read of the private, unsupported
`AddressBook-v22.abcddb` schema. A new source handle matches an
existing person only on a **unique** phone/email match against
Contacts; zero or multiple matches create a review-stub person instead
of guessing.

**Loud degrade (hard requirement in the build brief)**: Contacts access
requires its own TCC grant, separate from Full Disk Access (SPEC
§5.1a), and — as expected in this build environment —
`CNContactStore.authorizationStatusForEntityType_` currently reports
`CNAuthorizationStatusNotDetermined` here (verified live against the
real `Contacts` framework while building this module, not assumed). If
Contacts access is unavailable, S3 must not silently fall back to
raw-handle-named stub persons as if nothing happened: it raises
`ContactsAccessDeniedError` internally and `run_identity` catches it,
records the degradation on `IdentityResult.contacts` (`degraded=True`,
with the reason), and logs loudly — then proceeds with every handle
becoming a review-stub person, which is the correct behavior (S3 must
still make progress; the loudness is about visibility, not blocking).
`identity.contacts_import: false` in config is a different, intentional
case (not a degrade) and is reported separately (`attempted=False`).

Two ObjC-boundary functions (`_default_contacts_importer`,
`_contact_to_record`) are the only code in this module that touches the
`Contacts` framework directly, imported lazily inside the function
rather than at module level — so every other function here (person
resolution, invariant computation, chat-participant backfill, manual
curation) is plain Postgres logic, fully unit-testable via the injected
`contacts_importer` seam without ever touching pyobjc. Only the denial
path (`test_identity.py`'s `test_default_contacts_importer_...`) is
exercised against the real framework, since that is the one behavior
genuinely reproducible in an environment with no Contacts grant; the
successful-fetch path is implemented against Apple's documented
`CNContact`/`CNLabeledValue`/`CNPhoneNumber` API shape but was not
empirically exercised here (flagged in the build report).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import phonenumbers
import psycopg
import structlog

from imsg.config.schema import Config
from imsg.errors import IdentityError

logger = structlog.get_logger(__name__)


class ContactsAccessDeniedError(IdentityError):
    """Contacts access is unavailable — unauthorized TCC grant, or the
    `Contacts` framework itself could not be used. Caught by
    `run_identity`, which degrades loudly rather than silently."""


# --------------------------------------------------------------------------
# handle normalization
# --------------------------------------------------------------------------


def normalize_handle(raw_value: str, default_region: str) -> tuple[str, str]:
    """`(normalized_value, kind)` per SPEC §8 S3: E.164 for phone numbers
    (`phonenumbers`, region `identity.default_region`), lowercased for
    emails, `kind='unknown'` with `normalized == raw` for anything that
    parses as neither — "unparseable phone (kept as `kind='unknown'`,
    normalized = raw)" (SPEC §8 S3 failure modes)."""
    stripped = raw_value.strip()
    try:
        parsed = phonenumbers.parse(stripped, default_region)
        if phonenumbers.is_valid_number(parsed):
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164), "phone"
    except phonenumbers.NumberParseException:
        pass
    if "@" in stripped:
        return stripped.lower(), "email"
    return stripped, "unknown"


# --------------------------------------------------------------------------
# Contacts import
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContactRecord:
    identifier: str
    """CNContact's own stable identifier — used only to de-duplicate a
    contact appearing more than once within one import pass, never
    stored in Postgres (the schema has no column for it)."""
    display_name: str
    organization: str | None
    normalized_identifiers: tuple[tuple[str, str], ...]
    """`(normalized_value, kind)` for every phone/email this contact has,
    normalized the same way `normalize_handle` normalizes chat.db handles
    — this is what makes cross-referencing them possible at all."""


ContactsImporterFn = Callable[[str], list[ContactRecord]]


def _default_contacts_importer(default_region: str) -> list[ContactRecord]:
    """Real `CNContactStore` import. Raises `ContactsAccessDeniedError`
    immediately on anything short of `CNAuthorizationStatusAuthorized` —
    deliberately does not call `requestAccessForEntityType_` (that
    triggers a GUI prompt; this pipeline runs unattended and headless
    per SPEC §5.1, so a prompt nobody can answer is not a fallback
    path, it is a hang)."""
    try:
        from Contacts import (
            CNAuthorizationStatusAuthorized,
            CNContactEmailAddressesKey,
            CNContactFamilyNameKey,
            CNContactFetchRequest,
            CNContactGivenNameKey,
            CNContactOrganizationNameKey,
            CNContactPhoneNumbersKey,
            CNContactStore,
            CNEntityTypeContacts,
        )
    except ImportError as exc:
        raise ContactsAccessDeniedError(
            f"the Contacts framework could not be imported (pyobjc-framework-Contacts "
            f"not usable on this host): {exc}"
        ) from exc

    status = CNContactStore.authorizationStatusForEntityType_(CNEntityTypeContacts)
    if status != CNAuthorizationStatusAuthorized:
        raise ContactsAccessDeniedError(
            f"Contacts access is not authorized (CNAuthorizationStatus={status}) — grant "
            f"Contacts access to the process running this pipeline in System Settings > "
            f"Privacy & Security > Contacts (SPEC §5.1a: a separate TCC grant from Full "
            f"Disk Access). This build does not fall back to raw handles silently — every "
            f"handle will resolve to a review-stub person until this is granted."
        )

    store = CNContactStore.alloc().init()
    keys = [
        CNContactGivenNameKey,
        CNContactFamilyNameKey,
        CNContactOrganizationNameKey,
        CNContactPhoneNumbersKey,
        CNContactEmailAddressesKey,
    ]
    request = CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)

    records: list[ContactRecord] = []

    def _handle_contact(contact: Any, _stop: Any) -> None:
        records.append(_contact_to_record(contact, default_region))

    ok, error = store.enumerateContactsWithFetchRequest_error_usingBlock_(
        request, None, _handle_contact
    )
    if not ok:
        raise ContactsAccessDeniedError(
            f"CNContactStore enumeration failed even though authorization reported "
            f"'authorized': {error}"
        )
    return records


def _contact_to_record(contact: Any, default_region: str) -> ContactRecord:
    given = str(contact.givenName() or "")
    family = str(contact.familyName() or "")
    org = str(contact.organizationName() or "") or None
    display_name = " ".join(p for p in (given, family) if p) or (org or "Unnamed Contact")

    identifiers: list[tuple[str, str]] = []
    for labeled_phone in contact.phoneNumbers():
        raw = str(labeled_phone.value().stringValue())
        identifiers.append(normalize_handle(raw, default_region))
    for labeled_email in contact.emailAddresses():
        raw = str(labeled_email.value())
        identifiers.append(normalize_handle(raw, default_region))

    return ContactRecord(
        identifier=str(contact.identifier()),
        display_name=display_name,
        organization=org,
        normalized_identifiers=tuple(identifiers),
    )


class ContactsIndex:
    """Looks up Contacts matches by normalized `(value, kind)`."""

    def __init__(self, contacts: list[ContactRecord]) -> None:
        self._by_identifier: dict[tuple[str, str], list[ContactRecord]] = {}
        for contact in contacts:
            for ident in contact.normalized_identifiers:
                self._by_identifier.setdefault(ident, []).append(contact)

    def find_unique(self, normalized_value: str, kind: str) -> ContactRecord | None:
        """The matching contact, or `None` if there are zero or more than
        one — SPEC §8 S3: "zero or multiple matches create a review
        stub/conflict rather than guessing."""
        matches = self._by_identifier.get((normalized_value, kind), [])
        distinct = {m.identifier: m for m in matches}
        if len(distinct) == 1:
            return next(iter(distinct.values()))
        return None


# --------------------------------------------------------------------------
# short_name slugging
# --------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("-", ascii_value.lower()).strip("-")
    return slug or "person"


def _generate_unique_short_name(cur: psycopg.Cursor[Any], base: str) -> str:
    slug = _slugify(base)
    candidate = slug
    suffix = 2
    while True:
        cur.execute("SELECT 1 FROM person WHERE short_name = %s", (candidate,))
        if cur.fetchone() is None:
            return candidate
        candidate = f"{slug}-{suffix}"
        suffix += 1


# --------------------------------------------------------------------------
# person / handle resolution
# --------------------------------------------------------------------------


def _lookup_handle(cur: psycopg.Cursor[Any], normalized_value: str, kind: str) -> tuple[int, int] | None:
    """`(handle_id, person_id)` for an already-canonicalized handle, or `None`."""
    cur.execute(
        "SELECT handle_id, person_id FROM handle WHERE normalized_value = %s AND kind = %s",
        (normalized_value, kind),
    )
    row = cur.fetchone()
    return (int(row[0]), int(row[1])) if row else None


def _create_person(
    cur: psycopg.Cursor[Any], *, display_name: str, organization: str | None, note: str | None
) -> int:
    short_name = _generate_unique_short_name(cur, display_name)
    cur.execute(
        """
        INSERT INTO person (display_name, short_name, organization, notes)
        VALUES (%s, %s, %s, %s)
        RETURNING person_id
        """,
        (display_name, short_name, organization, note),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


def _find_or_create_person(
    cur: psycopg.Cursor[Any], normalized_value: str, kind: str, contacts_index: ContactsIndex | None
) -> int:
    if contacts_index is not None:
        contact = contacts_index.find_unique(normalized_value, kind)
        if contact is not None:
            # Cross-reference this contact's OTHER identifiers: if any of
            # them already resolved to a person (from an earlier handle in
            # this same run, or a prior run), reuse that person rather than
            # creating a duplicate for the same real-world contact — the
            # schema has no stable "CNContact identifier" column, so this
            # cross-reference is the only mechanism available for it.
            for other_value, other_kind in contact.normalized_identifiers:
                if (other_value, other_kind) == (normalized_value, kind):
                    continue
                existing = _lookup_handle(cur, other_value, other_kind)
                if existing is not None:
                    return existing[1]
            return _create_person(
                cur, display_name=contact.display_name, organization=contact.organization, note=None
            )

    return _create_person(
        cur,
        display_name=normalized_value,
        organization=None,
        note="auto-created review stub — zero or multiple Contacts matches (or Contacts unavailable)",
    )


def _resolve_owner_person(cur: psycopg.Cursor[Any]) -> int:
    cur.execute("SELECT person_id FROM person WHERE is_owner")
    row = cur.fetchone()
    if row is not None:
        return int(row[0])
    short_name = _generate_unique_short_name(cur, "owner")
    cur.execute(
        """
        INSERT INTO person (display_name, short_name, is_owner, needs_review)
        VALUES (%s, %s, true, false)
        RETURNING person_id
        """,
        ("Me", short_name),
    )
    row = cur.fetchone()
    assert row is not None
    return int(row[0])


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContactsImportOutcome:
    attempted: bool
    contacts_loaded: int
    degraded: bool
    degraded_reason: str | None


@dataclass(frozen=True, slots=True)
class InvariantReport:
    """The pre-S4 gate (SPEC §8 S3, hard requirement 3's mechanical form)."""

    unresolved_message_senders: int
    unresolved_tapback_senders: int
    unresolved_chat_participants: int
    owner_person_count: int

    @property
    def ok(self) -> bool:
        return (
            self.unresolved_message_senders == 0
            and self.unresolved_tapback_senders == 0
            and self.unresolved_chat_participants == 0
            and self.owner_person_count == 1
        )


@dataclass(frozen=True, slots=True)
class IdentityResult:
    source_handles_processed: int
    persons_created: int
    handles_created: int
    messages_resolved: int
    tapbacks_resolved: int
    chat_participants_resolved: int
    contacts: ContactsImportOutcome
    invariant: InvariantReport
    dry_run: bool = False
    """True when this result came from `run_identity(dry_run=True)`
    (SPEC §8: "takes --dry-run where writes leave the machine") — every
    count and the invariant report reflect what the real resolution
    logic would have produced (computed inside a transaction that was
    then rolled back), but nothing was actually written to Postgres."""


def compute_invariant_report(conn: psycopg.Connection) -> InvariantReport:
    """Pure read — safe to call any time, not just at the end of
    `run_identity` (e.g. `imsg identity review-report`, a later CLI
    build, would call this too)."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM message WHERE sender_person_id IS NULL")
        row = cur.fetchone()
        unresolved_messages = int(row[0]) if row else 0

        cur.execute(
            "SELECT count(*) FROM tapback WHERE NOT is_from_me AND sender_person_id IS NULL"
        )
        row = cur.fetchone()
        unresolved_tapbacks = int(row[0]) if row else 0

        cur.execute(
            """
            SELECT count(*) FROM chat_participant_source cps
            WHERE NOT EXISTS (
                SELECT 1 FROM source_handle_resolution shr
                WHERE shr.source_handle_id = cps.source_handle_id
            )
            """
        )
        row = cur.fetchone()
        unresolved_participants = int(row[0]) if row else 0

        cur.execute("SELECT count(*) FROM person WHERE is_owner")
        row = cur.fetchone()
        owner_count = int(row[0]) if row else 0

    return InvariantReport(
        unresolved_message_senders=unresolved_messages,
        unresolved_tapback_senders=unresolved_tapbacks,
        unresolved_chat_participants=unresolved_participants,
        owner_person_count=owner_count,
    )


def assert_invariant_or_raise(report: InvariantReport) -> None:
    """S4's gate (SPEC §8 S3): raises `IdentityError` naming exactly what
    is still unresolved. S4 itself is not this build's scope, but this
    is the mechanism it must call before sessionizing anything."""
    if report.ok:
        return
    problems = []
    if report.unresolved_message_senders:
        problems.append(f"{report.unresolved_message_senders} message(s) with unresolved sender_person_id")
    if report.unresolved_tapback_senders:
        problems.append(f"{report.unresolved_tapback_senders} non-owner tapback(s) with unresolved sender")
    if report.unresolved_chat_participants:
        problems.append(f"{report.unresolved_chat_participants} chat participant(s) with unresolved handle")
    if report.owner_person_count != 1:
        problems.append(f"expected exactly one owner person, found {report.owner_person_count}")
    raise IdentityError(
        "identity resolution invariant failed — S4 must not run until this is clean: "
        + "; ".join(problems)
    )


class _DryRunRollback(Exception):
    """Internal sentinel: forces the outer `with conn.transaction():`
    block `run_identity`'s dry-run path opens to ROLLBACK, the same
    savepoint-and-forced-rollback pattern `imsg.stages.extract` uses.
    Caught immediately below; never allowed to escape this module."""

    def __init__(self, result: IdentityResult) -> None:
        self.result = result


def _run_identity_body(
    conn: psycopg.Connection,
    config: Config,
    contacts_index: ContactsIndex | None,
    contacts_outcome: ContactsImportOutcome,
) -> IdentityResult:
    """The real resolution logic, factored out of `run_identity` so the
    dry-run path (below) can run it unchanged inside an outer
    transaction it then rolls back."""
    with conn.transaction(), conn.cursor() as cur:
        owner_person_id = _resolve_owner_person(cur)

        cur.execute(
            """
            SELECT sh.source_handle_id, sh.raw_value
            FROM source_handle sh
            WHERE NOT EXISTS (
                SELECT 1 FROM source_handle_resolution shr
                WHERE shr.source_handle_id = sh.source_handle_id
            )
            """
        )
        unresolved = cur.fetchall()

        persons_created_before = _count_persons(cur)
        handles_created = 0
        for source_handle_id, raw_value in unresolved:
            normalized_value, kind = normalize_handle(raw_value, config.identity.default_region)
            existing = _lookup_handle(cur, normalized_value, kind)
            if existing is not None:
                handle_id, _person_id = existing
            else:
                person_id = _find_or_create_person(cur, normalized_value, kind, contacts_index)
                cur.execute(
                    """
                    INSERT INTO handle (person_id, kind, normalized_value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (normalized_value, kind) DO UPDATE SET normalized_value = EXCLUDED.normalized_value
                    RETURNING handle_id
                    """,
                    (person_id, kind, normalized_value),
                )
                row = cur.fetchone()
                assert row is not None
                handle_id = int(row[0])
                handles_created += 1

            cur.execute(
                "INSERT INTO source_handle_resolution (source_handle_id, handle_id) "
                "VALUES (%s, %s) ON CONFLICT (source_handle_id) DO UPDATE SET handle_id = EXCLUDED.handle_id",
                (source_handle_id, handle_id),
            )

        persons_created = _count_persons(cur) - persons_created_before

        cur.execute(
            "UPDATE message SET sender_person_id = %s "
            "WHERE is_from_me AND sender_person_id IS NULL",
            (owner_person_id,),
        )
        cur.execute(
            """
            UPDATE message m
            SET sender_person_id = h.person_id
            FROM source_handle_resolution shr
            JOIN handle h ON h.handle_id = shr.handle_id
            WHERE m.sender_source_handle_id = shr.source_handle_id
              AND m.sender_person_id IS NULL
            """
        )
        cur.execute(
            "UPDATE tapback SET sender_person_id = %s "
            "WHERE is_from_me AND sender_person_id IS NULL",
            (owner_person_id,),
        )
        cur.execute(
            """
            UPDATE tapback t
            SET sender_person_id = h.person_id
            FROM source_handle_resolution shr
            JOIN handle h ON h.handle_id = shr.handle_id
            WHERE t.sender_source_handle_id = shr.source_handle_id
              AND t.sender_person_id IS NULL
            """
        )

        cur.execute(
            """
            INSERT INTO chat_participant (chat_id, person_id)
            SELECT DISTINCT cps.chat_id, h.person_id
            FROM chat_participant_source cps
            JOIN source_handle_resolution shr ON shr.source_handle_id = cps.source_handle_id
            JOIN handle h ON h.handle_id = shr.handle_id
            ON CONFLICT DO NOTHING
            """
        )
        chat_participants_resolved = cur.rowcount or 0

        # The owner participates in every chat they have any message in
        # (SPEC §8 S3: "the owner is inserted as a participant in every
        # chat so participant sets are complete").
        cur.execute(
            """
            INSERT INTO chat_participant (chat_id, person_id)
            SELECT DISTINCT m.chat_id, %s
            FROM message m
            WHERE m.is_from_me
            ON CONFLICT DO NOTHING
            """,
            (owner_person_id,),
        )

        cur.execute("SELECT count(*) FROM message WHERE sender_person_id IS NOT NULL")
        row = cur.fetchone()
        messages_resolved = int(row[0]) if row else 0
        cur.execute("SELECT count(*) FROM tapback WHERE sender_person_id IS NOT NULL")
        row = cur.fetchone()
        tapbacks_resolved = int(row[0]) if row else 0

    invariant = compute_invariant_report(conn)

    return IdentityResult(
        source_handles_processed=len(unresolved),
        persons_created=persons_created,
        handles_created=handles_created,
        messages_resolved=messages_resolved,
        tapbacks_resolved=tapbacks_resolved,
        chat_participants_resolved=chat_participants_resolved,
        contacts=contacts_outcome,
        invariant=invariant,
    )


def run_identity(
    *,
    conn: psycopg.Connection,
    config: Config,
    contacts_importer: ContactsImporterFn = _default_contacts_importer,
    dry_run: bool = False,
) -> IdentityResult:
    """Resolve every unresolved `source_handle`, backfill sender/participant
    `person_id`s, and compute the pre-S4 invariant report (SPEC §8 S3).

    `dry_run=True` (SPEC §8: "takes --dry-run where writes leave the
    machine") runs `_run_identity_body` unchanged inside one outer
    transaction — including `compute_invariant_report`, which reads
    back the (still-uncommitted, but visible within the same
    transaction) resolution it just performed — then forces a
    ROLLBACK before returning, so every count and the invariant report
    are accurate but nothing is actually written to Postgres. Contacts
    import (a read against the `Contacts` framework, not a Postgres
    write) still runs for real either way — dry-run has nothing to
    preview there, it already does no writing.
    """
    contacts_index: ContactsIndex | None = None
    contacts_outcome: ContactsImportOutcome
    if config.identity.contacts_import:
        try:
            records = contacts_importer(config.identity.default_region)
            contacts_index = ContactsIndex(records)
            contacts_outcome = ContactsImportOutcome(
                attempted=True, contacts_loaded=len(records), degraded=False, degraded_reason=None
            )
        except ContactsAccessDeniedError as exc:
            logger.error("identity.contacts_degraded", reason=str(exc))
            contacts_outcome = ContactsImportOutcome(
                attempted=True, contacts_loaded=0, degraded=True, degraded_reason=str(exc)
            )
    else:
        contacts_outcome = ContactsImportOutcome(
            attempted=False, contacts_loaded=0, degraded=False, degraded_reason=None
        )

    if not dry_run:
        return _run_identity_body(conn, config, contacts_index, contacts_outcome)

    try:
        with conn.transaction():
            result = _run_identity_body(conn, config, contacts_index, contacts_outcome)
            raise _DryRunRollback(result)
    except _DryRunRollback as sentinel:
        return replace(sentinel.result, dry_run=True)


def _count_persons(cur: psycopg.Cursor[Any]) -> int:
    cur.execute("SELECT count(*) FROM person")
    row = cur.fetchone()
    return int(row[0]) if row else 0


# --------------------------------------------------------------------------
# manual curation (SPEC §8 S3: `imsg identity merge/rename/assign` —
# CLI wiring is a later build's job; these are the functions it calls)
# --------------------------------------------------------------------------


def merge_persons(conn: psycopg.Connection, *, keep_person_id: int, absorb_person_id: int) -> None:
    """Repoint every handle/message/tapback/participant/allowlist row
    from `absorb_person_id` onto `keep_person_id` in one transaction,
    then delete the absorbed person (SPEC §8 S3: "merges repoint
    handles/messages/participants in one transaction")."""
    if keep_person_id == absorb_person_id:
        raise IdentityError("cannot merge a person into themselves")
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT is_owner FROM person WHERE person_id IN (%s, %s)", (keep_person_id, absorb_person_id))
        rows = cur.fetchall()
        if len(rows) != 2:
            raise IdentityError(f"one or both person_ids not found: {keep_person_id}, {absorb_person_id}")
        if any(r[0] for r in rows):
            raise IdentityError(
                "refusing to merge: one side is the singleton owner person — the owner "
                "identity is not part of ordinary merge flow"
            )

        cur.execute("UPDATE handle SET person_id = %s WHERE person_id = %s", (keep_person_id, absorb_person_id))
        cur.execute(
            "UPDATE message SET sender_person_id = %s WHERE sender_person_id = %s",
            (keep_person_id, absorb_person_id),
        )
        cur.execute(
            "UPDATE tapback SET sender_person_id = %s WHERE sender_person_id = %s",
            (keep_person_id, absorb_person_id),
        )
        cur.execute(
            """
            INSERT INTO chat_participant (chat_id, person_id)
            SELECT chat_id, %s FROM chat_participant WHERE person_id = %s
            ON CONFLICT DO NOTHING
            """,
            (keep_person_id, absorb_person_id),
        )
        cur.execute("DELETE FROM chat_participant WHERE person_id = %s", (absorb_person_id,))
        cur.execute(
            """
            INSERT INTO allowlist_person (person_id, text_allowed, attachments_allowed, note)
            SELECT %s, text_allowed, attachments_allowed, note FROM allowlist_person WHERE person_id = %s
            ON CONFLICT (person_id) DO NOTHING
            """,
            (keep_person_id, absorb_person_id),
        )
        cur.execute("DELETE FROM allowlist_person WHERE person_id = %s", (absorb_person_id,))
        cur.execute("DELETE FROM person WHERE person_id = %s", (absorb_person_id,))


def rename_person(
    conn: psycopg.Connection, *, person_id: int, display_name: str, short_name: str | None = None
) -> None:
    with conn.transaction(), conn.cursor() as cur:
        if short_name is not None:
            cur.execute(
                "UPDATE person SET display_name = %s, short_name = %s, updated_at = now() WHERE person_id = %s",
                (display_name, short_name, person_id),
            )
        else:
            cur.execute(
                "UPDATE person SET display_name = %s, updated_at = now() WHERE person_id = %s",
                (display_name, person_id),
            )
        if cur.rowcount == 0:
            raise IdentityError(f"person_id {person_id} not found")


def assign_handle(conn: psycopg.Connection, *, normalized_value: str, kind: str, person_id: int) -> None:
    """Manual override: repoint an already-canonical handle onto a
    different person (SPEC §8 S3 `identity assign`)."""
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            "UPDATE handle SET person_id = %s WHERE normalized_value = %s AND kind = %s",
            (person_id, normalized_value, kind),
        )
        if cur.rowcount == 0:
            raise IdentityError(f"no handle found for ({normalized_value!r}, {kind!r})")


__all__ = [
    "ContactRecord",
    "ContactsAccessDeniedError",
    "ContactsImportOutcome",
    "ContactsIndex",
    "IdentityResult",
    "InvariantReport",
    "assert_invariant_or_raise",
    "assign_handle",
    "compute_invariant_report",
    "merge_persons",
    "normalize_handle",
    "rename_person",
    "run_identity",
]
