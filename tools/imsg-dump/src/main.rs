/*!
 imsg-dump: reads a macOS `chat.db` snapshot via the `imessage-database`
 crate and emits one NDJSON object per message on stdout.

 GPL-3.0-or-later (see ../LICENSE), same as its `imessage-database`
 dependency. Isolated as its own standalone binary in its own directory
 deliberately: the rest of imessage-index is MIT/Apache-2.0 and must stay
 that way, so this shim is invoked as a subprocess and its stdout parsed as
 plain text/NDJSON -- it is never linked into the Python-facing core. See
 this repo's top-level CLAUDE.md, "Extraction shells out to
 `imessage-exporter` (GPL) -- process boundary only, never linked", which
 this binary follows the same pattern as.

 Usage:
     imsg-dump --db <path-to-chat.db-snapshot> [--since-rowid <N>]

 `--since-rowid` defaults to 0 (from the start). Every message with
 `ROWID > since_rowid` is emitted, ordered by ROWID, oldest first.
*/

mod db;
mod output;

use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use imessage_database::tables::messages::Message;
use imessage_database::tables::table::Table;

struct Args {
    db_path: PathBuf,
    since_rowid: i64,
}

fn parse_args() -> Result<Args, String> {
    let mut db_path: Option<PathBuf> = None;
    let mut since_rowid: i64 = 0;

    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--db" => {
                let value = argv.next().ok_or_else(|| "--db requires a path argument".to_string())?;
                db_path = Some(PathBuf::from(value));
            }
            "--since-rowid" => {
                let value = argv
                    .next()
                    .ok_or_else(|| "--since-rowid requires an integer argument".to_string())?;
                since_rowid = value
                    .parse::<i64>()
                    .map_err(|e| format!("--since-rowid: invalid integer '{value}': {e}"))?;
            }
            "-h" | "--help" => {
                return Err(
                    "usage: imsg-dump --db <path-to-chat.db-snapshot> [--since-rowid <N>]".to_string(),
                );
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }

    let db_path = db_path.ok_or_else(|| "missing required argument: --db <path>".to_string())?;
    Ok(Args { db_path, since_rowid })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;

    // Read-only open, per this repo's "never write to chat.db" rule -- see
    // db::open's doc comment.
    let conn = db::open(&args.db_path).map_err(|e| format!("cannot open database: {e}"))?;

    let handle_map = db::build_handle_map(&conn).map_err(|e| format!("cannot read handle table: {e}"))?;
    let chat_guid_map =
        db::build_chat_guid_map(&conn).map_err(|e| format!("cannot read chat table: {e}"))?;

    let mut stmt = db::prepare_message_query(&conn)
        .map_err(|e| format!("cannot prepare message query (no compatible chat.db schema found): {e}"))?;

    let message_rows = Message::rows(&mut stmt, [args.since_rowid])
        .map_err(|e| format!("cannot run message query: {e}"))?;

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let stderr = io::stderr();

    for message_result in message_rows {
        let message = match message_result {
            Ok(m) => m,
            Err(e) => {
                // No valid rowid/guid to attach to this row, so there is
                // nothing meaningful to emit on stdout for it -- log and
                // move on rather than aborting the whole scan.
                let mut err = stderr.lock();
                let _ = writeln!(err, "imsg-dump: row error: unable to deserialize a message row: {e}");
                continue;
            }
        };

        let outcome = output::build_output(&conn, message, &handle_map, &chat_guid_map);

        if let Some(warning) = &outcome.warning {
            let mut err = stderr.lock();
            let _ = writeln!(
                err,
                "imsg-dump: warning rowid={} guid={}: {}",
                outcome.output.rowid, outcome.output.guid, warning
            );
        }

        if let Err(e) = serde_json::to_writer(&mut out, &outcome.output) {
            let mut err = stderr.lock();
            let _ = writeln!(
                err,
                "imsg-dump: row error: failed to serialize rowid={} guid={}: {}",
                outcome.output.rowid, outcome.output.guid, e
            );
            continue;
        }
        out.write_all(b"\n").map_err(|e| format!("failed writing to stdout: {e}"))?;
        // Flush after every line so a subprocess consumer reading stdout
        // line-by-line sees each message as soon as it's decoded, rather
        // than waiting on BufWriter's internal buffer to fill.
        out.flush().map_err(|e| format!("failed flushing stdout: {e}"))?;
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("imsg-dump: fatal: {message}");
            ExitCode::FAILURE
        }
    }
}
