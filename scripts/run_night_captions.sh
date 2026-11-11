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
#      after NIGHT_CAPTIONS_LAST_START. A chunk that claims nothing does not
#      end the window unless the queue is genuinely empty: a caption deferred
#      to a later `next_attempt_at` claims nothing while still being pending,
#      so the job waits for the next one to fall due instead;
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
# Floor on the wait after a chunk claims nothing while work is still pending,
# so an already-due-but-unclaimable queue backs off instead of spinning.
MIN_WAIT=${NIGHT_CAPTIONS_MIN_WAIT:-60}
PG_SOCKET=${PG_SOCKET:-$DATA_ROOT/run}
PG_PORT=${PG_PORT:-5433}

mkdir -p "$OUT" || exit 1
LOG=$OUT/progress.log
night=$(date '+%F')
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
hhmm() { echo "$1" | tr -d ':'; }
# Minutes since midnight. A leading zero is stripped rather than forced to
# base 10 with `10#`, which is a bash extension POSIX sh leaves undefined:
# without this, "08:00" would be read as octal and fail.
mins() {
  _h=${1%%:*}
  _m=${1##*:}
  _h=${_h#0}
  _m=${_m#0}
  echo $(( ${_h:-0} * 60 + ${_m:-0} ))
}

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
      # An empty claim means "nothing due right now", which is not the same as
      # "nothing left". A caption whose `next_attempt_at` is in the future —
      # set by a retry backoff, or by an operator parking the queue — leaves
      # the queue full while claiming nothing. Reading that as "done" threw
      # away 6.5 hours of the 2026-11-11 window with 69,719 still pending.
      # So ask the queue, and wait for the next one to fall due.
      pending=$("$PSQL" -X -h "$PG_SOCKET" -p "$PG_PORT" -U postgres -d imsgindex -At -F' ' -c "select count(*), coalesce(ceil(greatest(extract(epoch from (min(next_attempt_at) - now())), 0)), 0)::bigint from enrichment where kind = 'caption' and state = 'pending';" 2>> "$LOG")
      left=${pending%% *}
      due_in=${pending##* }
      case "$left" in '' | *[!0-9]*)
        log "could not read the caption queue; stopping rather than spinning"
        break
        ;;
      esac
      case "$due_in" in '' | *[!0-9]*) due_in=0 ;; esac
      if [ "$left" -eq 0 ]; then
        log "done: no caption left to claim"
        break
      fi
      # Already due yet unclaimable means another worker holds them: back off
      # like the memory case rather than hammering.
      [ "$due_in" -lt "$MIN_WAIT" ] && due_in=$MIN_WAIT
      remaining=$(( ($(mins "$LAST_START") - $(mins "$(date '+%H:%M')")) * 60 ))
      if [ "$remaining" -le 0 ]; then
        log "past $LAST_START with $left caption(s) pending: no new chunk"
        break
      fi
      [ "$due_in" -gt "$remaining" ] && due_in=$remaining
      log "claimed 0 but $left caption(s) still pending; waiting ${due_in}s for the next to fall due"
      sleep "$due_in"
      ;;
  esac
done
exit 0
