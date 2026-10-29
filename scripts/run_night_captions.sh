#!/bin/sh
# Caption photos in the nightly window, 01:00-09:00 (owner decision 2026-10-29).
#
# The public MCP server keeps its models loaded all day so Gemini's searches
# never meet a cold server. Captioning needs about 25 GiB and does not fit
# beside them, so this job:
#
#   1. holds the public server's models unloaded until the window's end
#      (`imsg mcp hold-models`); the server unloads within about 15 s;
#   2. captions in chunks of NIGHT_CAPTIONS_CHUNK, running `imsg embed`
#      after each so new captions become searchable, and starts no chunk
#      after NIGHT_CAPTIONS_LAST_START;
#   3. releases the hold (`imsg mcp release-models`) when it finishes, fails
#      or is stopped. The hold also ends by itself at NIGHT_CAPTIONS_END, and
#      a caption chunk still running then stops after the caption in hand,
#      because the reloading server asks background work to give way.
#
# During the window Gemini's searches answer WARMING_UP.
#
# Started by launchd at 01:00. It refuses to start outside 01:00 to
# NIGHT_CAPTIONS_LAST_START, so a Mac that wakes at noon does not run it then.
#
# Exit codes: 0 window done (or nothing to caption), 1 could not start,
# 2 another copy is running, 3 outside the window.

set -u

IMSG=${IMSG:-$HOME/Projects/imessage-index/.venv/bin/imsg}
PSQL=${PSQL:-/opt/homebrew/opt/postgresql@17/bin/psql}
DATA_ROOT=${DATA_ROOT:-/Volumes/Data-Encrypted/imsgindex}
: "${IMSG_CONFIG:=$DATA_ROOT/private/config.yaml}"
export IMSG_CONFIG
OUT=${NIGHT_CAPTIONS_DIR:-$DATA_ROOT/ops/night-captions}
END=${NIGHT_CAPTIONS_END:-09:00}
LAST_START=${NIGHT_CAPTIONS_LAST_START:-08:00}
WINDOW_START=${NIGHT_CAPTIONS_START:-01:00}
CHUNK=${NIGHT_CAPTIONS_CHUNK:-250}
UNLOAD_WAIT=${NIGHT_CAPTIONS_UNLOAD_WAIT:-30}
PG_SOCKET=${PG_SOCKET:-$DATA_ROOT/run}
PG_PORT=${PG_PORT:-5433}

mkdir -p "$OUT" || exit 1
LOG=$OUT/progress.log
night=$(date '+%F')
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
hhmm() { echo "$1" | tr -d ':'; }

now=$(date '+%H%M')
if [ "$now" -lt "$(hhmm "$WINDOW_START")" ] || [ "$now" -ge "$(hhmm "$LAST_START")" ]; then
  log "not started: $(date '+%H:%M') is outside $WINDOW_START-$LAST_START"
  exit 3
fi

# One copy at a time: mkdir is atomic; a stale lock (its pid gone) is taken over.
LOCK=$OUT/running.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  other=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
    log "not started: pid $other is already running the window"
    exit 2
  fi
  rm -rf "$LOCK"
  mkdir "$LOCK" || exit 2
fi
echo $$ > "$LOCK/pid"

finish() {
  "$IMSG" mcp release-models >> "$LOG" 2>&1
  log "window over: public models released"
  rm -rf "$LOCK"
}
trap 'finish' EXIT
trap 'log "stopped by signal"; exit 1' INT TERM HUP

if ! "$IMSG" mcp hold-models --until "$END" --reason "nightly caption window" >> "$LOG" 2>&1; then
  log "could not hold the public models; no captions tonight"
  exit 1
fi
log "window started; captions until $END (no new chunk after $LAST_START)"
sleep "$UNLOAD_WAIT"

i=0
while :; do
  if [ "$(date '+%H%M')" -ge "$(hhmm "$LAST_START")" ]; then
    log "past $LAST_START: no new chunk"
    break
  fi
  i=$((i + 1))
  # Owner decision 2026-09-23: link previews get OCR but no caption.
  skipped=$("$PSQL" -X -h "$PG_SOCKET" -p "$PG_PORT" -U postgres -d imsgindex -At -c "with u as (update enrichment e set state = 'skipped', last_error = 'owner decision 2026-09-23: link previews get OCR but no caption' from attachment a where a.attachment_id = e.attachment_id and e.kind = 'caption' and e.state = 'pending' and a.filename ilike '%.pluginPayloadAttachment' returning 1) select count(*) from u;" 2>> "$LOG")
  chunk_log=$OUT/$night-chunk-$i.log
  "$IMSG" enrich --kinds caption --limit "$CHUNK" > "$chunk_log" 2>&1
  rc=$?
  line=$(grep -E '^enrich: claimed=' "$chunk_log" | tail -1)
  log "chunk $i link-previews-skipped=${skipped:-?} enrich exit $rc $line"
  case $rc in
    75)
      # Deferred for memory. Before the end of the window, wait and try again.
      sleep 60
      continue
      ;;
    0) ;;
    *)
      # 76 paused, 1 failed, anything else: embed what was done and stop.
      "$IMSG" embed > "$OUT/$night-embed-$i.log" 2>&1
      log "chunk $i embed exit $? $(grep -E '^embed:' "$OUT/$night-embed-$i.log" | tail -1)"
      log "stopping: enrich exit $rc"
      break
      ;;
  esac
  "$IMSG" embed > "$OUT/$night-embed-$i.log" 2>&1
  log "chunk $i embed exit $? $(grep -E '^embed:' "$OUT/$night-embed-$i.log" | tail -1)"
  case "$line" in
    *"claimed=0 "*|*"claimed=0")
      log "done: no caption left to claim"
      break
      ;;
  esac
done
exit 0
