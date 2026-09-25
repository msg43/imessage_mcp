"""Evidence cases: messages and files collected for review, with notes,
saved searches, and a download with exact citations (migration 0012).

The owner adds a message, or one attachment of it, to the active case
with "Add to case"; if no case exists yet, the first add creates one
named for the day. A case page lists its items in the order they were
sent, with a note on each, the case's own notes, and the searches the
owner saved to it with how many of each search's conversations are
marked reviewed. It downloads as Markdown, CSV or JSON, optionally in a
zip with the original files and a SHA-256 list.

Items are keyed by `message_key` / `attachment_key` (opaque, derived from
the Messages database's GUIDs), so a case survives re-segmentation.
Nothing is stored unless the owner clicks. A download goes only to the
owner's browser: it is a copy outside the encrypted volume, and the page
says so. In this project "export" means the allowlisted Gemini pipeline;
the page says "download".
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imsg.search_page.details import FILED_BY, SourceRecord, VersionRecord, service_name
from imsg.search_page.threads import chat_views, is_opaque_key

if TYPE_CHECKING:
    import psycopg

MAX_NAME_CHARS = 200
MAX_NOTES_CHARS = 20000
MAX_ITEM_NOTE_CHARS = 5000
DOWNLOAD_FORMATS = ("md", "csv", "json")


class CaseError(ValueError):
    """A case request the owner can fix (shown on the page)."""


@dataclass(frozen=True, slots=True)
class CaseSummary:
    case_id: int
    name: str
    notes: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    item_count: int
    search_count: int


CaseMarks = frozenset[tuple[str, str | None]]
"""`(message_key, attachment_key or None)` in the active case: which
"Add to case" buttons show "In case"."""


def _summaries(pg: psycopg.Connection, where: str, params: Mapping[str, object]) -> list[CaseSummary]:
    with pg.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.case_id, c.name, c.notes, c.is_active, c.created_at, c.updated_at,
                   (SELECT count(*) FROM search_case_item i WHERE i.case_id = c.case_id),
                   (SELECT count(*) FROM search_case_search s WHERE s.case_id = c.case_id)
            FROM search_case c WHERE {where}
            ORDER BY c.is_active DESC, c.updated_at DESC, c.case_id DESC
            """,
            dict(params),
        )
        return [
            CaseSummary(int(r[0]), str(r[1]), str(r[2]), bool(r[3]), r[4], r[5], int(r[6]), int(r[7]))
            for r in cur.fetchall()
        ]


def list_cases(pg: psycopg.Connection) -> list[CaseSummary]:
    return _summaries(pg, "TRUE", {})


def get_case(pg: psycopg.Connection, case_id: int) -> CaseSummary | None:
    found = _summaries(pg, "c.case_id = %(id)s", {"id": case_id})
    return found[0] if found else None


def active_case(pg: psycopg.Connection) -> CaseSummary | None:
    found = _summaries(pg, "c.is_active", {})
    return found[0] if found else None


def _clean_name(name: str) -> str:
    cleaned = " ".join(name.split())
    if not cleaned:
        raise CaseError("A case needs a name.")
    if len(cleaned) > MAX_NAME_CHARS:
        raise CaseError(f"A case name has at most {MAX_NAME_CHARS} characters.")
    return cleaned


def create_case(pg: psycopg.Connection, name: str) -> int:
    """A new case, which becomes the active one."""
    cleaned = _clean_name(name)
    with pg.transaction(), pg.cursor() as cur:
        cur.execute("UPDATE search_case SET is_active = false WHERE is_active")
        cur.execute(
            "INSERT INTO search_case (name, is_active) VALUES (%s, true) RETURNING case_id",
            (cleaned,),
        )
        row = cur.fetchone()
    assert row is not None
    return int(row[0])


def activate_case(pg: psycopg.Connection, case_id: int) -> None:
    with pg.transaction(), pg.cursor() as cur:
        cur.execute("SELECT 1 FROM search_case WHERE case_id = %s", (case_id,))
        if cur.fetchone() is None:
            raise CaseError("No such case.")
        cur.execute("UPDATE search_case SET is_active = false WHERE is_active AND case_id <> %s", (case_id,))
        cur.execute("UPDATE search_case SET is_active = true WHERE case_id = %s", (case_id,))


def rename_case(pg: psycopg.Connection, case_id: int, name: str) -> None:
    with pg.cursor() as cur:
        cur.execute("UPDATE search_case SET name = %s WHERE case_id = %s", (_clean_name(name), case_id))
        if cur.rowcount != 1:
            raise CaseError("No such case.")


def set_case_notes(pg: psycopg.Connection, case_id: int, notes: str) -> None:
    if len(notes) > MAX_NOTES_CHARS:
        raise CaseError(f"Case notes have at most {MAX_NOTES_CHARS:,} characters.")
    with pg.cursor() as cur:
        cur.execute("UPDATE search_case SET notes = %s WHERE case_id = %s", (notes, case_id))
        if cur.rowcount != 1:
            raise CaseError("No such case.")


def delete_case(pg: psycopg.Connection, case_id: int) -> None:
    with pg.cursor() as cur:
        cur.execute("DELETE FROM search_case WHERE case_id = %s", (case_id,))
        if cur.rowcount != 1:
            raise CaseError("No such case.")


# --------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToggleOutcome:
    case_id: int
    case_name: str
    in_case: bool
    item_count: int


def toggle_item(
    pg: psycopg.Connection,
    *,
    message_key: str,
    attachment_key: str | None,
    add: bool,
    default_name: str,
) -> ToggleOutcome:
    """Add the message (or one of its attachments) to the active case, or
    take it out. The first add with no case creates `default_name`."""
    if not is_opaque_key(message_key) or (attachment_key is not None and not is_opaque_key(attachment_key)):
        raise CaseError("Unknown message.")
    with pg.transaction(), pg.cursor() as cur:
        cur.execute("SELECT message_id FROM message WHERE message_key = %s", (message_key,))
        message = cur.fetchone()
        if message is None:
            raise CaseError("Unknown message.")
        if attachment_key is not None:
            cur.execute(
                "SELECT 1 FROM message_attachment ma JOIN attachment a ON a.attachment_id = ma.attachment_id "
                "WHERE ma.message_id = %s AND a.attachment_key = %s",
                (int(message[0]), attachment_key),
            )
            if cur.fetchone() is None:
                raise CaseError("That file is not part of that message.")
        cur.execute("SELECT case_id, name FROM search_case WHERE is_active")
        row = cur.fetchone()
        if row is None:
            if not add:
                raise CaseError("No case is open.")
            cur.execute(
                "INSERT INTO search_case (name, is_active) VALUES (%s, true) RETURNING case_id, name",
                (_clean_name(default_name),),
            )
            row = cur.fetchone()
        assert row is not None
        case_id, name = int(row[0]), str(row[1])
        if add:
            cur.execute(
                """
                INSERT INTO search_case_item (case_id, message_key, attachment_key)
                SELECT %(c)s, %(m)s, %(a)s
                WHERE NOT EXISTS (
                    SELECT 1 FROM search_case_item WHERE case_id = %(c)s AND message_key = %(m)s
                      AND attachment_key IS NOT DISTINCT FROM %(a)s)
                """,
                {"c": case_id, "m": message_key, "a": attachment_key},
            )
        else:
            cur.execute(
                "DELETE FROM search_case_item WHERE case_id = %s AND message_key = %s "
                "AND attachment_key IS NOT DISTINCT FROM %s",
                (case_id, message_key, attachment_key),
            )
        cur.execute("UPDATE search_case SET updated_at = now() WHERE case_id = %s", (case_id,))
        cur.execute("SELECT count(*) FROM search_case_item WHERE case_id = %s", (case_id,))
        count_row = cur.fetchone()
    return ToggleOutcome(case_id=case_id, case_name=name, in_case=add, item_count=int(count_row[0]) if count_row else 0)


def marks(pg: psycopg.Connection, message_keys: Iterable[str]) -> CaseMarks:
    keys = sorted({k for k in message_keys if k})
    if not keys:
        return frozenset()
    with pg.cursor() as cur:
        cur.execute(
            "SELECT i.message_key, i.attachment_key FROM search_case_item i "
            "JOIN search_case c ON c.case_id = i.case_id "
            "WHERE c.is_active AND i.message_key = ANY(%s::text[])",
            (keys,),
        )
        return frozenset((str(m), str(a) if a is not None else None) for m, a in cur.fetchall())


def set_item_note(pg: psycopg.Connection, item_id: int, note: str) -> int:
    """Returns the item's case id."""
    if len(note) > MAX_ITEM_NOTE_CHARS:
        raise CaseError(f"A note has at most {MAX_ITEM_NOTE_CHARS:,} characters.")
    with pg.cursor() as cur:
        cur.execute(
            "UPDATE search_case_item SET note = %s WHERE item_id = %s RETURNING case_id", (note, item_id)
        )
        row = cur.fetchone()
    if row is None:
        raise CaseError("No such item.")
    return int(row[0])


def remove_item(pg: psycopg.Connection, item_id: int) -> int:
    with pg.cursor() as cur:
        cur.execute("DELETE FROM search_case_item WHERE item_id = %s RETURNING case_id", (item_id,))
        row = cur.fetchone()
    if row is None:
        raise CaseError("No such item.")
    return int(row[0])


@dataclass(frozen=True, slots=True)
class CaseFile:
    attachment_key: str
    filename: str | None
    mime_type: str | None
    byte_size: int | None
    sha256: str | None
    state: str

    @property
    def available(self) -> bool:
        return self.state == "materialized" and bool(self.sha256)


@dataclass(frozen=True, slots=True)
class CaseItem:
    """One item with everything a citation needs."""

    item_id: int
    message_key: str
    attachment_key: str | None
    note: str
    added_at: datetime
    present: bool
    """False when the message is no longer in the index, or hidden by
    `policy.index_unsent`."""
    source_guid: str = ""
    sent_at: datetime | None = None
    is_from_me: bool = False
    sender_name: str = ""
    raw_handle: str | None = None
    service: str = ""
    thread_key: str = ""
    conversation: str = ""
    conversation_kind: str = ""
    chat_evidence: str = ""
    text: str | None = None
    is_edited: bool = False
    date_edited: datetime | None = None
    versions: tuple[VersionRecord, ...] | None = None
    deleted_at: datetime | None = None
    is_unsent: bool = False
    files: tuple[CaseFile, ...] = ()
    """The item's file, or for a message item every file of the message."""
    sources: tuple[SourceRecord, ...] = field(default_factory=tuple)


def case_items(
    pg: psycopg.Connection,
    case_id: int,
    *,
    index_unsent: bool,
    show_edit_history: bool,
    show_raw_handles: bool,
) -> list[CaseItem]:
    """The case's items in the order their messages were sent (items whose
    message is gone come last)."""
    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT i.item_id, i.message_key, i.attachment_key, i.note, i.added_at,
                   m.message_id, m.source_guid, m.sent_at, m.is_from_me, p.display_name,
                   m.service::text, m.chat_id, m.chat_evidence, m.text_original, m.is_edited,
                   m.date_edited, m.deleted_at, m.is_unsent, m.sender_source_handle_id
            FROM search_case_item i
            LEFT JOIN message m ON m.message_key = i.message_key
            LEFT JOIN person p ON p.person_id = m.sender_person_id
            WHERE i.case_id = %s
            ORDER BY m.sent_at NULLS LAST, m.message_id, i.attachment_key NULLS FIRST, i.item_id
            """,
            (case_id,),
        )
        rows = cur.fetchall()
        message_ids = [int(r[5]) for r in rows if r[5] is not None]
        chats = chat_views(pg, {int(r[11]) for r in rows if r[11] is not None})
        files: dict[int, list[CaseFile]] = {}
        if message_ids:
            cur.execute(
                """
                SELECT ma.message_id, a.attachment_key, a.filename, a.mime_type, a.byte_size,
                       a.sha256, a.state::text
                FROM message_attachment ma JOIN attachment a ON a.attachment_id = ma.attachment_id
                WHERE ma.message_id = ANY(%s::bigint[])
                ORDER BY ma.message_id, ma.ordinal NULLS LAST, a.attachment_id
                """,
                (message_ids,),
            )
            for mid, key, name, mime, size, sha, state in cur.fetchall():
                files.setdefault(int(mid), []).append(
                    CaseFile(str(key), name, mime, int(size) if size is not None else None, sha, str(state))
                )
        sources: dict[int, list[SourceRecord]] = {}
        versions: dict[int, list[VersionRecord]] = {}
        handles: dict[int, str] = {}
        if message_ids:
            cur.execute(
                """
                SELECT ms.message_id, ms.source_name, ms.source_rowid, er.merge_mode,
                       coalesce(er.finished_at, er.started_at)
                FROM message_source ms JOIN extraction_run er ON er.run_id = ms.extraction_run_id
                WHERE ms.message_id = ANY(%s::bigint[])
                ORDER BY ms.message_id, (er.merge_mode = 'live') DESC NULLS LAST, ms.source_name
                """,
                (message_ids,),
            )
            for mid, name, rowid, mode, read_at in cur.fetchall():
                sources.setdefault(int(mid), []).append(SourceRecord(str(name), int(rowid), mode, read_at))
            if show_edit_history:
                cur.execute(
                    "SELECT message_id, version_idx, text, edited_at FROM message_version "
                    "WHERE message_id = ANY(%s::bigint[]) ORDER BY message_id, version_idx",
                    (message_ids,),
                )
                for mid, idx, text, edited_at in cur.fetchall():
                    versions.setdefault(int(mid), []).append(VersionRecord(int(idx), str(text), edited_at))
            if show_raw_handles:
                handle_ids = [int(r[18]) for r in rows if r[18] is not None and not r[8]]
                if handle_ids:
                    cur.execute(
                        "SELECT source_handle_id, raw_value FROM source_handle WHERE source_handle_id = ANY(%s::bigint[])",
                        (handle_ids,),
                    )
                    by_handle = {int(h): str(v) for h, v in cur.fetchall()}
                    for r in rows:
                        if r[18] is not None and int(r[18]) in by_handle and r[5] is not None:
                            handles[int(r[5])] = by_handle[int(r[18])]
    out: list[CaseItem] = []
    for r in rows:
        (item_id, mkey, akey, note, added_at, mid, guid, sent_at, from_me, display, service, chat_id,
         evidence, text, edited, date_edited, deleted_at, unsent, _handle_id) = r
        present = mid is not None and (index_unsent or not unsent)
        if not present:
            out.append(CaseItem(int(item_id), str(mkey), akey, str(note), added_at, False))
            continue
        chat = chats.get(int(chat_id))
        own_files = files.get(int(mid), [])
        if akey is not None:
            own_files = [f for f in own_files if f.attachment_key == akey]
        out.append(
            CaseItem(
                item_id=int(item_id),
                message_key=str(mkey),
                attachment_key=str(akey) if akey is not None else None,
                note=str(note),
                added_at=added_at,
                present=True,
                source_guid=str(guid),
                sent_at=sent_at,
                is_from_me=bool(from_me),
                sender_name="Me" if from_me else (str(display) if display else "Unknown"),
                raw_handle=handles.get(int(mid)),
                service=str(service),
                thread_key=chat.thread_key if chat else "",
                conversation=chat.title if chat else "Conversation",
                conversation_kind=(
                    "unfiled"
                    if chat and chat.is_holding
                    else ("group" if chat and chat.kind == "group" else ("one-to-one" if chat else ""))
                ),
                chat_evidence=str(evidence),
                text=text if akey is None else None,
                is_edited=bool(edited),
                date_edited=date_edited,
                versions=tuple(versions.get(int(mid), ())) if show_edit_history else None,
                deleted_at=deleted_at,
                is_unsent=bool(unsent),
                files=tuple(own_files),
                sources=tuple(sources.get(int(mid), ())),
            )
        )
    return out


# --------------------------------------------------------------------------
# saved searches and reviewed conversations
# --------------------------------------------------------------------------

SEARCH_PARAMS = ("people", "from", "to", "att")


def search_params(params: Mapping[str, object]) -> dict[str, str]:
    """The filters a saved search keeps, with empty ones left out."""
    out: dict[str, str] = {}
    for key in SEARCH_PARAMS:
        value = params.get(key)
        if isinstance(value, str) and value.strip() and not (key == "att" and value == "any"):
            out[key] = value.strip()[:500]
    return out


@dataclass(frozen=True, slots=True)
class SavedSearch:
    search_id: int
    case_id: int
    query_text: str
    params: dict[str, str]
    reviewed: frozenset[str]
    created_at: datetime


def save_search(
    pg: psycopg.Connection, *, query_text: str, params: Mapping[str, object], default_name: str
) -> tuple[int, str]:
    """Save a search to the active case (creating one if none is open);
    returns `(search_id, case name)`. Saving it again returns the same
    search."""
    text = " ".join(query_text.split())
    if not text or len(text) > 1000:
        raise CaseError("Save a search that has words.")
    kept = json.dumps(search_params(params), sort_keys=True)
    with pg.transaction(), pg.cursor() as cur:
        cur.execute("SELECT case_id, name FROM search_case WHERE is_active")
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO search_case (name, is_active) VALUES (%s, true) RETURNING case_id, name",
                (_clean_name(default_name),),
            )
            row = cur.fetchone()
        assert row is not None
        case_id, name = int(row[0]), str(row[1])
        cur.execute(
            """
            INSERT INTO search_case_search (case_id, query_text, params) VALUES (%s, %s, %s::jsonb)
            ON CONFLICT (case_id, query_text, params) DO UPDATE SET query_text = EXCLUDED.query_text
            RETURNING search_id
            """,
            (case_id, text, kept),
        )
        found = cur.fetchone()
        cur.execute("UPDATE search_case SET updated_at = now() WHERE case_id = %s", (case_id,))
    assert found is not None
    return int(found[0]), name


def _as_params(value: object) -> dict[str, str]:
    loaded = json.loads(value) if isinstance(value, str) else value
    return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}


def saved_searches(pg: psycopg.Connection, case_id: int) -> list[SavedSearch]:
    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT s.search_id, s.case_id, s.query_text, s.params, s.created_at,
                   coalesce(array_agg(r.thread_key) FILTER (WHERE r.thread_key IS NOT NULL), '{}')
            FROM search_case_search s LEFT JOIN search_case_review r ON r.search_id = s.search_id
            WHERE s.case_id = %s
            GROUP BY s.search_id ORDER BY s.created_at, s.search_id
            """,
            (case_id,),
        )
        return [
            SavedSearch(int(sid), int(cid), str(text), _as_params(params), frozenset(str(k) for k in keys), created)
            for sid, cid, text, params, created, keys in cur.fetchall()
        ]


def active_saved_search(
    pg: psycopg.Connection, query_text: str, params: Mapping[str, object]
) -> SavedSearch | None:
    """The active case's saved search with exactly this text and these
    filters, if the owner saved one."""
    text = " ".join(query_text.split())
    kept = json.dumps(search_params(params), sort_keys=True)
    with pg.cursor() as cur:
        cur.execute(
            "SELECT s.search_id, s.case_id FROM search_case_search s JOIN search_case c ON c.case_id = s.case_id "
            "WHERE c.is_active AND s.query_text = %s AND s.params = %s::jsonb",
            (text, kept),
        )
        row = cur.fetchone()
    if row is None:
        return None
    for search in saved_searches(pg, int(row[1])):
        if search.search_id == int(row[0]):
            return search
    return None


def remove_search(pg: psycopg.Connection, search_id: int) -> int:
    with pg.cursor() as cur:
        cur.execute("DELETE FROM search_case_search WHERE search_id = %s RETURNING case_id", (search_id,))
        row = cur.fetchone()
    if row is None:
        raise CaseError("No such saved search.")
    return int(row[0])


def set_reviewed(pg: psycopg.Connection, *, search_id: int, thread_key: str, reviewed: bool) -> int:
    """Mark or unmark one conversation reviewed for a saved search; returns
    how many are marked."""
    if not is_opaque_key(thread_key):
        raise CaseError("Unknown conversation.")
    with pg.transaction(), pg.cursor() as cur:
        cur.execute("SELECT 1 FROM search_case_search WHERE search_id = %s", (search_id,))
        if cur.fetchone() is None:
            raise CaseError("No such saved search.")
        if reviewed:
            cur.execute(
                "INSERT INTO search_case_review (search_id, thread_key) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (search_id, thread_key),
            )
        else:
            cur.execute(
                "DELETE FROM search_case_review WHERE search_id = %s AND thread_key = %s",
                (search_id, thread_key),
            )
        cur.execute("SELECT count(*) FROM search_case_review WHERE search_id = %s", (search_id,))
        row = cur.fetchone()
    return int(row[0]) if row is not None else 0


# --------------------------------------------------------------------------
# downloads
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchCoverage:
    search: SavedSearch
    conversations: int | None
    """Conversations the search finds now (`None` when it could not run)."""


def _utc(dt: datetime | None) -> str | None:
    """Machine-readable times in a download are UTC, ISO 8601."""
    return dt.astimezone(UTC).isoformat() if dt is not None else None


def _items_as_records(items: Sequence[CaseItem], formatter: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for n, item in enumerate(items, start=1):
        if not item.present:
            records.append(
                {"item": n, "message_id": item.message_key, "attachment_id": item.attachment_key,
                 "missing": True, "note": item.note}
            )
            continue
        records.append(
            {
                "item": n,
                "sent_at": _utc(item.sent_at),
                "sent_local": formatter.exact(item.sent_at) if item.sent_at else None,
                "sender": item.sender_name,
                "sender_handle": item.raw_handle,
                "service": service_name(item.service),
                "conversation": item.conversation,
                "conversation_kind": item.conversation_kind,
                "filed_because": FILED_BY.get(item.chat_evidence, item.chat_evidence),
                "text": item.text,
                "files": [
                    {"name": f.filename, "sha256": f.sha256, "bytes": f.byte_size, "type": f.mime_type,
                     "available": f.available}
                    for f in item.files
                ],
                "edited_at": _utc(item.date_edited) if item.is_edited else None,
                "earlier_text": (
                    [{"text": v.text, "dated": _utc(v.edited_at)} for v in item.versions]
                    if item.versions is not None
                    else None
                ),
                "deleted_at": _utc(item.deleted_at),
                "unsent": item.is_unsent,
                "found_in": [
                    {"source": s.source_name, "kind": s.merge_mode, "row": s.source_rowid,
                     "read_at": _utc(s.read_at)}
                    for s in item.sources
                ],
                "message_id": item.message_key,
                "attachment_id": item.attachment_key,
                "messages_guid": item.source_guid,
                "note": item.note,
            }
        )
    return records


def render_download(
    fmt: str,
    *,
    case: CaseSummary,
    items: Sequence[CaseItem],
    coverage: Sequence[SearchCoverage],
    formatter: Any,
    downloaded_at: datetime,
    timezone: str,
    with_files: bool,
) -> str:
    """The case as Markdown, CSV or JSON. `formatter.exact(dt)` writes a
    time to the second with its zone offset. With `with_files`, each file
    is named by the path it has in the zip."""
    records = _items_as_records(items, formatter)
    if with_files:
        for record, item in zip(records, items, strict=True):
            for f, entry in zip(item.files, record.get("files", []), strict=False):
                entry["path"] = file_path_in_zip(f) if f.available else None
    searches = [
        {
            "search": c.search.query_text,
            "filters": c.search.params,
            "conversations": c.conversations,
            "reviewed": len(c.search.reviewed),
        }
        for c in coverage
    ]
    if fmt == "json":
        return json.dumps(
            {
                "case": case.name,
                "downloaded_at": _utc(downloaded_at),
                "time_zone": timezone,
                "notes": case.notes,
                "items": records,
                "saved_searches": searches,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    if fmt == "csv":
        buffer = io.StringIO()
        columns = [
            "item", "sent_at", "sent_local", "sender", "sender_handle", "service", "conversation",
            "text", "files", "file_sha256", "edited_at", "earlier_text", "deleted_at", "unsent",
            "found_in", "message_id", "attachment_id", "messages_guid", "note",
        ]
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = dict(record)
            files = record.get("files") or []
            row["files"] = "; ".join(str(f.get("path") or f.get("name") or "") for f in files)
            row["file_sha256"] = "; ".join(str(f.get("sha256") or "") for f in files)
            row["found_in"] = "; ".join(
                f"{s['source']} row {s['row']}" for s in record.get("found_in") or []
            )
            earlier = record.get("earlier_text")
            row["earlier_text"] = " | ".join(str(v["text"]) for v in earlier) if earlier else ""
            if record.get("missing"):
                row["text"] = "(no longer in the index)"
            writer.writerow(row)
        return buffer.getvalue()
    lines = [
        f"# {case.name}",
        "",
        f"Downloaded {formatter.exact(downloaded_at)} from the private message search. "
        f"Times are {timezone}. Message IDs are the search page's; Messages GUIDs are the "
        "Messages database's own.",
        "",
    ]
    if case.notes.strip():
        lines += ["## Notes", "", case.notes.strip(), ""]
    lines += ["## Items", ""]
    for record, item in zip(records, items, strict=True):
        n = record["item"]
        if record.get("missing"):
            lines += [f"{n}. (no longer in the index) · Message ID {item.message_key}", ""]
            continue
        who = str(record["sender"]) + (f" ({record['sender_handle']})" if record.get("sender_handle") else "")
        lines.append(
            f"{n}. {record['sent_local']} · {who} · {record['service']} · in "
            f"“{record['conversation']}”"
        )
        if record.get("text"):
            text = " ".join(str(record["text"]).split())
            lines.append(f"   “{text}”")
        for f in record.get("files") or []:
            parts = [str(f.get("path") or f.get("name") or "unnamed file")]
            if f.get("sha256"):
                parts.append(f"SHA-256 {f['sha256']}")
            if f.get("bytes") is not None:
                parts.append(f"{f['bytes']:,} bytes")
            lines.append("   File: " + ", ".join(parts))
        if item.is_edited:
            lines.append(
                "   Edited" + (f" {formatter.exact(item.date_edited)}" if item.date_edited else "")
            )
            for v in item.versions or ():
                lines.append(f"   Earlier text: “{' '.join(v.text.split())}”")
        if item.deleted_at:
            lines.append(f"   Deleted in Messages {formatter.exact(item.deleted_at)}; kept from Recently Deleted")
        if item.is_unsent:
            lines.append("   Unsent")
        if item.sources:
            lines.append(
                "   Found in: "
                + "; ".join(f"{s.source_name} row {s.source_rowid}" for s in item.sources)
            )
        lines.append(f"   Message ID {item.message_key} · Messages GUID {item.source_guid}")
        if item.note.strip():
            lines.append(f"   Note: {' '.join(item.note.split())}")
        lines.append("")
    if coverage:
        lines += ["## Saved searches", ""]
        for c in coverage:
            filters = ", ".join(f"{k} {v}" for k, v in c.search.params.items())
            reviewed = len(c.search.reviewed)
            coverage_text = (
                f"reviewed {reviewed} of {c.conversations} conversation"
                + ("" if c.conversations == 1 else "s")
                if c.conversations is not None
                else f"{reviewed} conversation{'' if reviewed == 1 else 's'} reviewed"
            )
            lines.append(
                f"- \u201c{c.search.query_text}\u201d" + (f" ({filters})" if filters else "") + f": {coverage_text}"
            )
        lines.append("")
    return "\n".join(lines)


def file_path_in_zip(f: CaseFile) -> str:
    """`files/<first 12 of the SHA-256>-<safe name>`: unique per content,
    readable, and free of path separators."""
    name = (f.filename or "file").replace("/", "_").replace("\\", "_").strip() or "file"
    name = "".join(ch for ch in name if ch.isprintable())[:120] or "file"
    return f"files/{(f.sha256 or 'unknown')[:12]}-{name}"


def case_files(items: Sequence[CaseItem]) -> list[CaseFile]:
    """Every available file of the case once, in item order."""
    seen: set[str] = set()
    out: list[CaseFile] = []
    for item in items:
        for f in item.files:
            if f.available and f.sha256 and f.sha256 not in seen:
                seen.add(f.sha256)
                out.append(f)
    return out


__all__ = [
    "DOWNLOAD_FORMATS",
    "MAX_ITEM_NOTE_CHARS",
    "MAX_NAME_CHARS",
    "MAX_NOTES_CHARS",
    "CaseError",
    "CaseFile",
    "CaseItem",
    "CaseMarks",
    "CaseSummary",
    "SavedSearch",
    "SearchCoverage",
    "ToggleOutcome",
    "activate_case",
    "active_case",
    "active_saved_search",
    "case_files",
    "case_items",
    "create_case",
    "delete_case",
    "file_path_in_zip",
    "get_case",
    "list_cases",
    "marks",
    "remove_item",
    "remove_search",
    "rename_case",
    "render_download",
    "save_search",
    "saved_searches",
    "search_params",
    "set_case_notes",
    "set_item_note",
    "set_reviewed",
    "toggle_item",
]
