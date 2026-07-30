/*!
 Connection setup, schema-tiered message queries, and small lookup maps
 (handle rowid -> raw handle string, chat rowid -> chat guid) that the
 `imessage-database` crate's own table structs don't expose directly.
*/

use std::collections::HashMap;
use std::path::Path;

use imessage_database::error::table::TableError;
use imessage_database::tables::handle::Handle;
use imessage_database::tables::table::{Table, get_connection};
use rusqlite::{CachedStatement, Connection};

/// Open `path` read-only via the crate's own connection helper
/// (`SQLITE_OPEN_READ_ONLY`). This never write-opens the database, matching
/// the parent repo's "never write to chat.db" rule even though this shim is
/// only ever expected to run against an already-detached snapshot.
pub fn open(path: &Path) -> Result<Connection, TableError> {
    get_connection(path)
}

/// Map `handle.ROWID -> handle.id` (the raw phone/email/apple-id string).
///
/// Built directly from `Handle::rows`, *not* `Handle::cache`: `Handle::cache`
/// merges handles that share `person_centric_id` into one combined display
/// string (e.g. `"+15551234567 person@example.com"`), which is the right
/// behavior for a human-facing exporter but wrong here -- this shim needs the
/// verbatim per-row handle string so the Python side can do its own identity
/// resolution (see the parent repo's "Identity resolution before
/// segmentation" rule).
pub fn build_handle_map(conn: &Connection) -> Result<HashMap<i32, String>, TableError> {
    let mut map = HashMap::new();
    let mut stmt = Handle::get(conn)?;
    for handle in Handle::rows(&mut stmt, [])? {
        let handle = handle?;
        map.insert(handle.rowid, handle.id);
    }
    Ok(map)
}

/// Map `chat.ROWID -> chat.guid`.
///
/// Deviation from the assumed API: `imessage_database::tables::chat::Chat`
/// does not expose the `guid` column at all -- only `chat_identifier`,
/// `service_name`, and `display_name` -- so this cannot be built from the
/// crate's `Chat` table struct or `Table`/`Cacheable` trait impls. Queried
/// directly with a small raw statement instead.
pub fn build_chat_guid_map(conn: &Connection) -> rusqlite::Result<HashMap<i32, String>> {
    let mut map = HashMap::new();
    let mut stmt = conn.prepare("SELECT ROWID, guid FROM chat")?;
    let rows = stmt.query_map([], |row| {
        let rowid: i32 = row.get(0)?;
        let guid: String = row.get(1)?;
        Ok((rowid, guid))
    })?;
    for row in rows {
        let (rowid, guid) = row?;
        map.insert(rowid, guid);
    }
    Ok(map)
}

// MARK: Message query
//
// The crate's own `Message::stream_rows`/`QueryContext` filters only cover
// date ranges and chat/handle id sets -- `QueryContext`
// (`util/query_context.rs`) has no ROWID field at all -- and the query
// builders that *do* know how to assemble the right column list per schema
// generation (`ios_16_newer_query` / `ios_14_15_query` / `ios_13_older_query`
// in `tables/messages/query_parts.rs`, plus the `COLS` constant in
// `tables/messages/message.rs`) are all `pub(crate)`, unreachable from
// outside the crate. So `--since-rowid` pagination has to be a query this
// shim writes itself. It follows the exact column list documented in the
// crate's own "Making Custom Message Queries" doc comment (module docs on
// `imessage_database::tables::messages::message`) so that
// `Message::from_row`'s positional fast path (`from_row_idx`, which reads
// columns 0..=29 by index before ever trying `from_row_named`) lines up
// column-for-column with what `Message` expects.
//
// Two deliberate deviations from the crate's own query shape, both noted in
// the build report:
//
// 1. `ORDER BY m.ROWID`, not `ORDER BY m.date` (which every one of the
//    crate's own query builders uses). `--since-rowid` is a ROWID cursor, and
//    `date` is not guaranteed monotonic with ROWID (backfilled/imported
//    history, clock skew, etc.); ordering by ROWID is what makes the cursor
//    resumable without gaps or duplicates.
// 2. `chat_id` and `deleted_from` are read via scalar subqueries
//    (`(SELECT ... LIMIT 1)`) instead of the crate's
//    `LEFT JOIN chat_message_join` / `LEFT JOIN chat_recoverable_message_join`.
//    A LEFT JOIN duplicates the message row once per matching chat if a
//    message ever joins more than one chat. The parent task's own notes
//    acknowledge this is possible ("in practice it's 1:1"); the scalar
//    subquery form guarantees exactly one output row per message ROWID and
//    deterministically prefers the lowest chat_id when more than one exists,
//    i.e. "pick the first/primary", as instructed.
//
// Three tiers mirror the crate's own ios_16_newer / ios_14_15 / ios_13_older
// fallback chain, substituting NULL literals (in the same column position)
// for columns that don't exist on older schemas.

const TIER1_SQL: &str = "
SELECT
    m.ROWID, m.guid, m.text, m.service, m.handle_id, m.destination_caller_id, m.subject,
    m.date, m.date_read, m.date_delivered, m.is_from_me, m.is_read, m.item_type, m.other_handle,
    m.share_status, m.share_direction, m.group_title, m.group_action_type, m.associated_message_guid,
    m.associated_message_type, m.balloon_bundle_id, m.expressive_send_style_id, m.thread_originator_guid,
    m.thread_originator_part, m.date_edited, m.associated_message_emoji,
    (SELECT c.chat_id FROM chat_message_join c WHERE c.message_id = m.ROWID ORDER BY c.chat_id LIMIT 1) AS chat_id,
    (SELECT COUNT(*) FROM message_attachment_join a WHERE a.message_id = m.ROWID) AS num_attachments,
    (SELECT d.chat_id FROM chat_recoverable_message_join d WHERE d.message_id = m.ROWID LIMIT 1) AS deleted_from,
    (SELECT COUNT(*) FROM message m2 WHERE m2.thread_originator_guid = m.guid) AS num_replies
FROM message AS m
WHERE m.ROWID > ?1
ORDER BY m.ROWID
";

const TIER2_SQL: &str = "
SELECT
    m.ROWID, m.guid, m.text, m.service, m.handle_id, m.destination_caller_id, m.subject,
    m.date, m.date_read, m.date_delivered, m.is_from_me, m.is_read, m.item_type, m.other_handle,
    m.share_status, m.share_direction, m.group_title, m.group_action_type, m.associated_message_guid,
    m.associated_message_type, m.balloon_bundle_id, m.expressive_send_style_id, m.thread_originator_guid,
    m.thread_originator_part, m.date_edited, m.associated_message_emoji,
    (SELECT c.chat_id FROM chat_message_join c WHERE c.message_id = m.ROWID ORDER BY c.chat_id LIMIT 1) AS chat_id,
    (SELECT COUNT(*) FROM message_attachment_join a WHERE a.message_id = m.ROWID) AS num_attachments,
    NULL AS deleted_from,
    (SELECT COUNT(*) FROM message m2 WHERE m2.thread_originator_guid = m.guid) AS num_replies
FROM message AS m
WHERE m.ROWID > ?1
ORDER BY m.ROWID
";

const TIER3_SQL: &str = "
SELECT
    m.ROWID, m.guid, m.text, m.service, m.handle_id, m.destination_caller_id, m.subject,
    m.date, m.date_read, m.date_delivered, m.is_from_me, m.is_read, m.item_type, m.other_handle,
    m.share_status, m.share_direction, m.group_title, m.group_action_type, m.associated_message_guid,
    m.associated_message_type, m.balloon_bundle_id, m.expressive_send_style_id,
    NULL AS thread_originator_guid,
    NULL AS thread_originator_part,
    NULL AS date_edited,
    NULL AS associated_message_emoji,
    (SELECT c.chat_id FROM chat_message_join c WHERE c.message_id = m.ROWID ORDER BY c.chat_id LIMIT 1) AS chat_id,
    (SELECT COUNT(*) FROM message_attachment_join a WHERE a.message_id = m.ROWID) AS num_attachments,
    NULL AS deleted_from,
    0 AS num_replies
FROM message AS m
WHERE m.ROWID > ?1
ORDER BY m.ROWID
";

/// Prepare the newest compatible `--since-rowid` message query, falling back
/// through older schema shapes exactly like the crate's own
/// `Message::get`/`Message::stream_rows` do (`.or_else` chain over
/// `prepare_cached`, newest schema first).
pub fn prepare_message_query(conn: &Connection) -> rusqlite::Result<CachedStatement<'_>> {
    conn.prepare_cached(TIER1_SQL)
        .or_else(|_| conn.prepare_cached(TIER2_SQL))
        .or_else(|_| conn.prepare_cached(TIER3_SQL))
}
