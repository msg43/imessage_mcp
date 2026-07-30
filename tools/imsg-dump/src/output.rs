/*!
 Converts a decoded `imessage_database::tables::messages::Message` into the
 NDJSON output contract this shim promises the Python `extract.py` stage.
*/

use std::collections::HashMap;

use chrono::Utc;
use imessage_database::error::message::MessageError;
use imessage_database::message_types::edited::EditStatus;
use imessage_database::message_types::variants::{Tapback, TapbackAction, Variant};
use imessage_database::tables::attachment::Attachment;
use imessage_database::tables::messages::Message;
use imessage_database::tables::messages::models::Service;
use imessage_database::util::dates::{get_local_time, get_offset};
use rusqlite::Connection;
use serde::Serialize;

#[derive(Serialize)]
pub struct OutputMessage {
    pub rowid: i64,
    pub guid: String,
    pub chat_guid: Option<String>,
    pub handle: Option<String>,
    pub is_from_me: bool,
    pub date: Option<String>,
    pub date_edited: Option<String>,
    pub date_retracted: Option<String>,
    pub service: String,
    pub body_text: Option<String>,
    pub edit_history: Vec<EditHistoryEntry>,
    pub is_unsent: bool,
    pub tapback: Option<TapbackOut>,
    pub attachment_rowids: Vec<i64>,
    pub reply_to_guid: Option<String>,
}

#[derive(Serialize)]
pub struct EditHistoryEntry {
    pub text: String,
    pub edited_at: Option<String>,
}

#[derive(Serialize)]
pub struct TapbackOut {
    pub kind: String,
    pub emoji: Option<String>,
    pub action: String,
    pub target_guid: String,
}

/// Result of converting one message row: the NDJSON payload to emit, plus an
/// optional non-fatal warning (body-decode failure or attachment-join
/// failure) the caller should log to stderr. A warning never suppresses the
/// output line -- the row is always emitted, degraded as needed.
pub struct ConversionOutcome {
    pub output: OutputMessage,
    pub warning: Option<String>,
}

/// Convert an Apple-epoch (nanoseconds-or-seconds since 2001-01-01, per
/// `imessage_database::util::dates::get_local_time`'s own heuristic) raw
/// timestamp into an ISO-8601 UTC string, or `None` when `raw == 0` (the
/// crate's own convention for "unset", e.g. `Message::is_edited()` is defined
/// as `date_edited != 0`).
///
/// Uses the crate's own `get_local_time` + `get_offset` rather than
/// hand-rolling the epoch math, per the parent task's instruction. One
/// consequence inherited from the crate: `get_local_time` truncates to
/// whole-second precision (it calls `DateTime::from_timestamp(secs, 0)`,
/// discarding the nanosecond remainder), so sub-second precision on message
/// timestamps is lost. Flagged in the build report.
fn convert_timestamp(raw: i64) -> Option<String> {
    if raw == 0 {
        return None;
    }
    let offset = get_offset();
    get_local_time(raw, offset)
        .ok()
        .map(|local| local.with_timezone(&Utc).to_rfc3339())
}

/// Map one `Tapback` variant to the lowercase `kind` string this shim emits,
/// plus the emoji character for `Tapback::Emoji`.
fn tapback_kind(t: &Tapback) -> (&'static str, Option<String>) {
    match t {
        Tapback::Loved => ("loved", None),
        Tapback::Liked => ("liked", None),
        Tapback::Disliked => ("disliked", None),
        Tapback::Laughed => ("laughed", None),
        Tapback::Emphasized => ("emphasized", None),
        Tapback::Questioned => ("questioned", None),
        Tapback::Emoji(e) => ("emoji", e.map(str::to_string)),
        Tapback::Sticker => ("sticker", None),
    }
}

/// Flatten `Message::edited_parts` (one edit history per body *part*) into
/// the single, contract-required `edit_history[]` array of prior versions,
/// oldest first.
///
/// Two judgment calls, both flagged in the build report:
///
/// 1. The contract asks for *prior* versions only ("empty array if never
///    edited"). Per `message_types::edited`'s own module docs, "item 0 is the
///    original text and the last item is the current text" -- i.e. the final
///    entry in each part's history duplicates what `body_text` already
///    holds. This drops that final entry per part.
/// 2. A message can have multiple edited body parts, each with its own
///    history. This flattens all parts into one array and re-sorts by
///    timestamp, so entries from different parts can interleave; the
///    per-part boundary is lost. Multi-part edited messages are rare in
///    practice (most messages are a single text part), but a Python
///    integrator that needs the per-part structure back should be aware this
///    flattening is lossy in that one respect.
fn build_edit_history(message: &Message) -> Vec<EditHistoryEntry> {
    let Some(edited) = &message.edited_parts else {
        return Vec::new();
    };

    let mut all: Vec<(i64, String)> = Vec::new();
    for part in &edited.parts {
        let history = &part.edit_history;
        if history.len() > 1 {
            for event in &history[..history.len() - 1] {
                all.push((event.date, event.text.clone()));
            }
        }
    }
    all.sort_by_key(|(date, _)| *date);

    all.into_iter()
        .map(|(date, text)| EditHistoryEntry {
            text,
            edited_at: convert_timestamp(date),
        })
        .collect()
}

/// `true` when *any* body part was unsent/retracted.
///
/// Deviation: the crate only exposes `Message::is_fully_unsent()` (true only
/// when *every* part is unsent). For a multi-part message where just one part
/// was retracted, `is_fully_unsent()` would report `false` even though the
/// sender did retract something -- the wrong signal for a boolean named
/// `is_unsent`/"message was retracted/unsent by the sender". This instead
/// checks `edited_parts` directly for any part with `EditStatus::Unsent`.
fn compute_is_unsent(message: &Message) -> bool {
    message
        .edited_parts
        .as_ref()
        .is_some_and(|edited| edited.parts.iter().any(|part| matches!(part.status, EditStatus::Unsent)))
}

/// Build the NDJSON payload for one message row.
///
/// `message` is consumed (not borrowed) because `Message::apply_body`
/// requires `&mut self`, and the message isn't needed by the caller
/// afterward.
pub fn build_output(
    conn: &Connection,
    mut message: Message,
    handle_map: &HashMap<i32, String>,
    chat_guid_map: &HashMap<i32, String>,
) -> ConversionOutcome {
    let mut warning: Option<String> = None;

    // Decode the body. `MessageError::NoText` is the crate's own signal for
    // "this message legitimately has no text" (pure attachment messages,
    // tapback rows, etc.) -- not a failure, and not logged. Any other error
    // variant (a malformed typedstream/streamtyped blob, a bad plist, an
    // out-of-range timestamp) is a genuine per-row decode failure: recorded
    // as a warning, but the row is still emitted with `body_text: null`
    // rather than aborting the run.
    match message.parse_body(conn) {
        Ok(parsed) => message.apply_body(parsed),
        Err(MessageError::NoText) => {}
        Err(other) => {
            warning = Some(format!("body decode failed: {other}"));
        }
    }

    let body_text = message.text.clone();

    let attachment_rowids: Vec<i64> = match Attachment::from_message(conn, &message) {
        Ok(list) => list.into_iter().map(|a| i64::from(a.rowid)).collect(),
        Err(e) => {
            if warning.is_none() {
                warning = Some(format!("attachment lookup failed: {e}"));
            }
            Vec::new()
        }
    };

    // `variant()` borrows `message` (a tapback's emoji, when present, is
    // borrowed straight out of `associated_message_emoji`), so everything
    // that needs an owned copy out of `tb`/`action` has to happen before
    // `message` gets consumed into the output below.
    let variant = message.variant();
    let tapback = if let Variant::Tapback(_, action, tb) = &variant {
        message.clean_associated_guid().map(|(_, target_guid)| {
            let (kind, emoji) = tapback_kind(tb);
            TapbackOut {
                kind: kind.to_string(),
                emoji,
                action: match action {
                    TapbackAction::Added => "added".to_string(),
                    TapbackAction::Removed => "removed".to_string(),
                },
                target_guid: target_guid.to_string(),
            }
        })
    } else {
        None
    };

    // `Message::is_from_me()` (the method), not the raw `is_from_me` field:
    // the method additionally accounts for legacy shared-location messages
    // where the raw flag alone doesn't reflect who actually sent it (see the
    // crate's own doc comment on `is_from_me()`).
    let is_from_me = message.is_from_me();
    let handle = if is_from_me {
        None
    } else {
        message.handle_id.and_then(|id| handle_map.get(&id).cloned())
    };

    let chat_guid = message.chat_id.and_then(|id| chat_guid_map.get(&id).cloned());

    // `Service::from_name(...).to_string()` is used verbatim rather than
    // collapsed to the contract's four-value sketch (iMessage | SMS | RCS |
    // unknown): the crate models two more cases the contract didn't
    // anticipate -- `Service::Satellite` (chat.db's literal "iMessageLite",
    // Apple's satellite-messaging service) and `Service::Other(&str)` for any
    // unrecognized raw service string. Emitted strings are exactly:
    // "iMessage", "SMS", "RCS", "Satellite", "Unknown" (missing/absent
    // service field), or the raw service string verbatim for anything else.
    // Flagged in the build report; the Python integrator should treat
    // anything outside {iMessage, SMS, RCS} as needing a fallback bucket
    // rather than assuming only four literal values ever appear.
    let service = Service::from_name(message.service.as_deref()).to_string();

    let output = OutputMessage {
        rowid: i64::from(message.rowid),
        guid: message.guid.clone(),
        chat_guid,
        handle,
        is_from_me,
        date: convert_timestamp(message.date),
        date_edited: convert_timestamp(message.date_edited),
        // Deviation: imessage-database 4.2.0 has no `date_retracted` concept
        // anywhere -- not in `Message`'s field list / `COLS`, not in any
        // query builder, not in `message_types::edited`. `grep -rn retract`
        // across the whole crate source matches only an unrelated
        // `BubbleComponent::Retracted` body-component variant (a rendering
        // marker for an unsent *part*, not a timestamp). Real chat.db schemas
        // may or may not have their own `date_retracted`-ish column, but
        // fabricating a raw-SQL column name against a live database I can't
        // see and the crate doesn't reference felt worse than being explicit
        // about the gap, per the task's own "do not hardcode a
        // plausible-looking but untested value" instruction. Always `null`;
        // `is_unsent` (derived from `edited_parts` part statuses, see
        // `compute_is_unsent`) is the closest available signal for "this
        // message was retracted."
        date_retracted: None,
        service,
        body_text,
        edit_history: build_edit_history(&message),
        is_unsent: compute_is_unsent(&message),
        tapback,
        attachment_rowids,
        reply_to_guid: message.thread_originator_guid.clone(),
    };

    ConversionOutcome { output, warning }
}
