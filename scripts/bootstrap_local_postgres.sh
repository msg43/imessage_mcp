#!/usr/bin/env bash
#
# bootstrap_local_postgres.sh -- create the dedicated local Postgres 17
# cluster this project expects, entirely under $DATA_ROOT.
#
# What it does, in plain English:
#   1. Runs `initdb` under `$DATA_ROOT/pg17` (a brand-new cluster
#      directory -- refuses to touch one that already exists).
#   2. Sets the cluster to listen on 127.0.0.1 only, on port 5433 (the
#      dedicated port this project's config requires -- see
#      src/imsg/config/schema.py), using `--auth=trust` for local
#      connections. Trust auth means "anyone who can already reach this
#      port on this machine is allowed in" -- it is scoped to
#      127.0.0.1 so nothing outside this Mac can connect, and this
#      script never asks for or stores a password.
#   3. Starts the cluster with `pg_ctl`.
#   4. Creates the `imsg` role and `imsgindex` database this project's
#      example config expects (config.example.yaml).
#   5. Enables the `vector` (pgvector) and `pg_prewarm` extensions the
#      migrations require (migrations/0001_initial.sql,
#      migrations/0004_pg_prewarm.sql).
#
# It never touches an existing cluster directory, and it never asks
# for or stores a password anywhere -- the role has no password set at
# all; access is controlled by "can you reach 127.0.0.1:5433", which
# only processes on this Mac can do.
#
# Usage:
#   scripts/bootstrap_local_postgres.sh /Volumes/Data-Encrypted/imsgindex
#
# Requires Postgres 17 client/server binaries (e.g. from
# `brew install postgresql@17`) on PATH, or discoverable at the
# Homebrew default install location
# (/opt/homebrew/opt/postgresql@17/bin or /usr/local/opt/postgresql@17/bin).

set -euo pipefail

# Postgres needs a plain C locale on macOS or the postmaster can fail
# at startup with an error that looks like a corrupt installation (see
# CLAUDE.md "Testing").
export LC_ALL=C

PG_PORT=5433
PG_SUBDIR=pg17
DB_NAME=imsgindex
DB_ROLE=imsg

usage() {
  echo "Usage: $0 DATA_ROOT" >&2
  echo "  DATA_ROOT: the directory this project's config.yaml paths.data_root" >&2
  echo "             points at (must be on your encrypted data volume)." >&2
  exit 2
}

if [ "$#" -ne 1 ]; then
  usage
fi

DATA_ROOT="$1"

if [ -z "$DATA_ROOT" ]; then
  usage
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "error: DATA_ROOT '$DATA_ROOT' does not exist or is not a directory." >&2
  echo "Create it first (it should live on your encrypted data volume)." >&2
  exit 1
fi

PG_DATA_DIR="$DATA_ROOT/$PG_SUBDIR"

# --------------------------------------------------------------------------
# Locate the Postgres 17 binaries.
# --------------------------------------------------------------------------

find_pg_bindir() {
  # Prefer whatever's already on PATH, but only if it is actually
  # version 17 -- a stray Postgres 14/15/16 on PATH would otherwise be
  # picked up silently.
  if command -v initdb >/dev/null 2>&1; then
    version_line="$(initdb --version 2>/dev/null || true)"
    case "$version_line" in
      *") 17."*)
        dirname "$(command -v initdb)"
        return 0
        ;;
    esac
  fi

  for candidate in \
    /opt/homebrew/opt/postgresql@17/bin \
    /usr/local/opt/postgresql@17/bin
  do
    if [ -x "$candidate/initdb" ]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

PG_BINDIR="$(find_pg_bindir || true)"
if [ -z "$PG_BINDIR" ]; then
  echo "error: could not find Postgres 17 binaries (initdb/pg_ctl/psql)." >&2
  echo "Install them with: brew install postgresql@17" >&2
  exit 1
fi

INITDB="$PG_BINDIR/initdb"
PG_CTL="$PG_BINDIR/pg_ctl"
PSQL="$PG_BINDIR/psql"
CREATEDB_BIN="$PG_BINDIR/createdb"

for bin in "$INITDB" "$PG_CTL" "$PSQL"; do
  if [ ! -x "$bin" ]; then
    echo "error: expected Postgres binary not found or not executable: $bin" >&2
    exit 1
  fi
done

echo "Using Postgres binaries in: $PG_BINDIR"

# --------------------------------------------------------------------------
# Refuse to touch an existing cluster directory.
# --------------------------------------------------------------------------

if [ -e "$PG_DATA_DIR" ]; then
  echo "error: '$PG_DATA_DIR' already exists." >&2
  echo "This script only initializes a brand-new cluster and will not touch" >&2
  echo "an existing one. Remove or rename it yourself first if you really" >&2
  echo "want to start over, or point DATA_ROOT somewhere else." >&2
  exit 1
fi

# --------------------------------------------------------------------------
# 1. initdb -- create the cluster directory.
# --------------------------------------------------------------------------

echo "Creating a new Postgres cluster at: $PG_DATA_DIR"
echo "(this may take a few seconds)"
"$INITDB" \
  --pgdata="$PG_DATA_DIR" \
  --auth=trust \
  --username="$DB_ROLE" \
  --encoding=UTF8 \
  --no-instructions

# --------------------------------------------------------------------------
# 2. Configure: listen on 127.0.0.1 only, port 5433.
# --------------------------------------------------------------------------

echo "Configuring the cluster to listen on 127.0.0.1:$PG_PORT only"
{
  echo ""
  echo "# --- added by scripts/bootstrap_local_postgres.sh ---"
  echo "listen_addresses = '127.0.0.1'"
  echo "port = $PG_PORT"
} >> "$PG_DATA_DIR/postgresql.conf"

# --------------------------------------------------------------------------
# 3. Start the cluster.
# --------------------------------------------------------------------------

PG_LOG_FILE="$PG_DATA_DIR/startup.log"
echo "Starting Postgres (log: $PG_LOG_FILE)"
"$PG_CTL" \
  --pgdata="$PG_DATA_DIR" \
  --log="$PG_LOG_FILE" \
  --wait \
  --timeout=60 \
  start

stop_on_error() {
  echo "Stopping the cluster because setup failed partway through." >&2
  "$PG_CTL" --pgdata="$PG_DATA_DIR" --mode=fast stop || true
}
trap stop_on_error ERR

# --------------------------------------------------------------------------
# 4. Create the database and confirm the role exists.
# --------------------------------------------------------------------------

# initdb --username already created the superuser role ($DB_ROLE), so
# only the database needs creating.
echo "Creating database '$DB_NAME' owned by role '$DB_ROLE'"
"$CREATEDB_BIN" \
  --host=127.0.0.1 \
  --port="$PG_PORT" \
  --username="$DB_ROLE" \
  --owner="$DB_ROLE" \
  "$DB_NAME"

# --------------------------------------------------------------------------
# 5. Enable the extensions the migrations require.
# --------------------------------------------------------------------------

echo "Enabling extensions: vector (pgvector), pg_prewarm"
"$PSQL" \
  --host=127.0.0.1 \
  --port="$PG_PORT" \
  --username="$DB_ROLE" \
  --dbname="$DB_NAME" \
  --set=ON_ERROR_STOP=1 \
  --command="CREATE EXTENSION IF NOT EXISTS vector;" \
  --command="CREATE EXTENSION IF NOT EXISTS pg_prewarm;"

trap - ERR

echo ""
echo "Done. A local Postgres 17 cluster is running at:"
echo "  data directory: $PG_DATA_DIR"
echo "  listening on:   127.0.0.1:$PG_PORT"
echo "  database:       $DB_NAME (owned by role '$DB_ROLE')"
echo "  extensions:     vector, pg_prewarm"
echo ""
echo "No password was set or stored -- only processes on this Mac can connect."
echo ""
echo "To stop it later:  $PG_CTL --pgdata='$PG_DATA_DIR' --mode=fast stop"
echo "To start it again: $PG_CTL --pgdata='$PG_DATA_DIR' --log='$PG_LOG_FILE' start"
