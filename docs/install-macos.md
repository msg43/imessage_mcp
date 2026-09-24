# Installing imessage-index on your own Mac

This is a from-scratch, non-expert walkthrough for getting your own
iMessages searchable by Claude. It assumes a Mac with Apple Silicon and
no prior familiarity with Rust, Postgres, or MCP servers. Every command
below is meant to be copy-pasted; a few need your own username or
account details filled in, and those are called out.

Sizes and durations marked **[UNVERIFIED]** have not been measured for
this guide and will vary with your Mac and internet connection — treat
them as rough expectations, not promises.

Throughout, replace `alice` with your own macOS username wherever you
see it.

## What you're installing

Four pieces, in order: a database (Postgres, to hold the index),
a small Rust program (`imsg-dump`, to read your Messages database
safely), a handful of local AI models (to understand and search your
messages), and the `imsg` command itself (Python, which ties it all
together and talks to Claude over MCP — the Model Context Protocol).

Nothing here uploads your messages anywhere. Everything runs on your
Mac. (An optional, separate feature lets you push a filtered subset to
a company search index later — you can ignore it entirely for personal
use; see [Optional: enterprise export](#optional-enterprise-export).)

## Step 1: Install the base tools (Homebrew)

If you don't already have [Homebrew](https://brew.sh) installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install `uv` (the Python package/venv manager this project uses),
the Rust toolchain manager, and PostgreSQL 17 with the pgvector
extension:

```bash
brew install uv rustup-init postgresql@17 pgvector
rustup-init -y    # accepts the defaults; installs the stable Rust toolchain
```

Restart your terminal (or `source ~/.zshrc`) after this so `uv`,
`cargo`, and `psql` are all on your `PATH`. Confirm:

```bash
uv --version
cargo --version
psql --version
```

Note: this project pins an exact Rust compiler version in
`tools/imsg-dump/rust-toolchain.toml` — `rustup` reads that file
automatically and will fetch the pinned version the first time you
build, even if it differs from whatever `rustup-init` installed by
default.

## Step 2: Decide where your data lives

Everything the pipeline derives from your messages — the index, the
Postgres data files, the downloaded models — lives under one folder
called `data_root`. Two ways to protect it:

- **Simple default (recommended for most people): turn on FileVault**
  for your whole startup disk (System Settings -> Privacy & Security ->
  FileVault) and put `data_root` in a normal folder in your home
  directory, e.g. `/Users/alice/imsgindex-data`. Most Macs set up since
  2018 already have FileVault on — check the same settings pane.
- **Stricter option: a separate encrypted APFS volume.** Open Disk
  Utility, select your main container, and add a new APFS volume with
  encryption turned on (or use an encrypted disk image). Then point
  `data_root` at a folder on that volume instead, e.g.
  `/Volumes/Data-Encrypted/imsgindex`.

Either way, create the folder and drop a sentinel file in it — this is
how the pipeline proves to itself, every time it starts, that it's
really looking at the encrypted volume you intend and not some
unrelated (or unmounted) path that happens to resolve the same way:

```bash
mkdir -p /Users/alice/imsgindex-data
touch /Users/alice/imsgindex-data/.imsgindex-volume
```

The pipeline refuses to start if this file is missing, or if
`data_root` doesn't resolve onto a mounted, encrypted volume.

## Step 3: Set up the dedicated Postgres instance

The project needs its **own** Postgres instance — not the one Homebrew
starts by default, and not sharing a cluster with anything else on your
Mac. It listens on port **5433** (not Postgres's usual 5432), and its
data files live under your `data_root`, in a subfolder named `pg17`.

> **Shortcut:** if `scripts/bootstrap_local_postgres.sh` exists in your
> checkout, it automates everything in this step. Run
> `scripts/bootstrap_local_postgres.sh --help` to see what it does
> before running it — the manual steps below are what it's automating,
> and are worth understanding once even if you use the script.

Manual steps:

```bash
export DATA_ROOT=/Users/alice/imsgindex-data
export LC_ALL=C   # required — see the gotcha below
mkdir -p "$DATA_ROOT/pg17"
$(brew --prefix postgresql@17)/bin/initdb -D "$DATA_ROOT/pg17" -U imsg
```

> **Gotcha that will cost you an hour if you skip it.** On macOS,
> Postgres needs `LC_ALL=C` set or the postmaster dies at startup with
> *"postmaster became multithreaded during startup"* — an error that
> reads like a corrupted install and isn't. Export it in every shell
> you start Postgres from, or add it to your shell profile.

Start the instance on port 5433, pointed at your data directory:

```bash
$(brew --prefix postgresql@17)/bin/pg_ctl -D "$DATA_ROOT/pg17" \
  -o "-p 5433" -l "$DATA_ROOT/pg17/server.log" start
```

Create the database and role, and enable pgvector:

```bash
$(brew --prefix postgresql@17)/bin/createuser -p 5433 imsg
$(brew --prefix postgresql@17)/bin/createdb -p 5433 imsgindex --owner imsg
psql -p 5433 -d imsgindex -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

Set a password for the `imsg` role and store it in your macOS Keychain
(never in a config file):

```bash
psql -p 5433 -d imsgindex -c "ALTER ROLE imsg WITH PASSWORD 'pick-a-real-password-here';"
security add-generic-password -a "$USER" -s imsgindex-pg -w    # prompts for the same password; no shell history
```

**Check it worked** — this should print a path ending in
`.../pg17` under your `data_root`:

```bash
psql -p 5433 -c 'SHOW data_directory'
```

## Step 4: Build the Rust extraction shim

The pipeline never reads `chat.db` itself in Python — it shells out to
a small Rust program, `imsg-dump`, that does the actual decoding
(Apple's message format is not simple, and this program already handles
it correctly). Build it once:

```bash
cd /Users/alice/imessage-index   # your checkout of this repo
cargo build --release --manifest-path tools/imsg-dump/Cargo.toml
```

This produces `tools/imsg-dump/target/release/imsg-dump`, which the
Python pipeline finds automatically. **[UNVERIFIED]** duration: a clean
build of this crate and its dependencies takes a few minutes on Apple
Silicon.

## Step 5: Install the Python side

```bash
cd /Users/alice/imessage-index
uv sync --extra models
```

`--extra models` installs the real local model runtimes (MLX, Apple
Vision's OCR bridge, Whisper, and the multimodal embedding stack) —
this is what you need for actual local search. It does **not** install
the Google Cloud export libraries; those are a separate, optional
extra (see [Optional: enterprise export](#optional-enterprise-export))
that a local-search setup never needs.

**[UNVERIFIED]** download size: the `models` extra pulls in `torch` and
several other large packages; expect several GB of downloads on first
`uv sync`.

## Step 6: Your config file

Copy the single-Mac example and edit the paths for your own machine:

```bash
cp examples/config.single-mac.yaml config.yaml
export IMSG_CONFIG="$PWD/config.yaml"
```

Open `config.yaml` and replace every `/Users/alice/...` path with your
own username and the `data_root` you picked in Step 2. The file is
commented throughout; `config.example.yaml` in the repo root documents
every field if you want the full picture.

Add `export IMSG_CONFIG=...` to your shell profile (`~/.zshrc`) so it's
set in every new terminal, not just this one.

## Step 7: Full Disk Access and Contacts permissions

macOS will not let any program read your Messages database — including
this one — without explicit permission, granted through a GUI dialog.
This **cannot be done over SSH**: the permission prompt requires an
actual logged-in GUI session, and it's granted to the specific binary
that launches the process (your terminal app, or `imsg` itself,
depending on how you run it).

1. Open **System Settings -> Privacy & Security -> Full Disk Access**.
2. Click **+**, and add your terminal app (Terminal.app, iTerm2,
   whichever you use to run `imsg`).
3. Separately, under **Privacy & Security -> Contacts**, the first time
   `imsg identity import` runs it will prompt you to allow Contacts
   access — approve it so your contacts' names can be matched to phone
   numbers/handles instead of showing raw numbers everywhere.

Check both took effect:

```bash
uv run imsg check-permissions
```

This prints `full_disk_access: True` once granted (and reports mount,
at-rest posture, and Postgres reachability too — worth reading even
after every check is set up).

## Step 8: Download and convert the models

Model choices and their exact pinned versions are recorded in
`models/manifest.lock.yaml` — the source of truth `config.yaml` mirrors.
Most are plain Hugging Face downloads that happen automatically the
first time a pipeline stage needs them. One — the reranker — is a
*local conversion*: you run a command once to produce a quantized copy
under your `data_root`. That command, copied verbatim from
`models/manifest.lock.yaml`:

```bash
python -c "from huggingface_hub import snapshot_download; from mlx_lm.convert import \
convert; src = snapshot_download('Qwen/Qwen3-Reranker-0.6B', \
revision='e61197ed45024b0ed8a2d74b80b4d909f1255473'); convert(hf_path=src, \
mlx_path='$DATA_ROOT/models/qwen3-reranker-0.6b-mxfp8-e61197ed', quantize=True, \
q_mode='mxfp8', q_bits=8, q_group_size=32)"
```

Run this with the project's own `python` (i.e. `uv run python -c "..."`,
or inside `uv run --extra models python`), and with `$DATA_ROOT` set to
the same path as `paths.data_root` in your `config.yaml`.

**[UNVERIFIED]** download size / duration: the 0.6B reranker download
and conversion is small relative to the embedding and captioning
models; the largest model pinned (the 35B-parameter boundary/caption
model, 4-bit quantized) is tens of GB — expect a lengthy download on
the first run that needs it and confirm you have the disk space under
`data_root` before starting.

Verify every pin resolves and every local conversion's checksum matches
what's recorded:

```bash
uv run imsg models verify
```

## Step 9: Run the pipeline, in order

Each stage below corresponds to one `imsg` command. Run them from the
repo root, with `IMSG_CONFIG` set as in Step 6. `--dry-run` is
supported everywhere below and is worth trying first on a stage you're
unsure about — it reports what it would do without changing anything.

```bash
uv run imsg guard-mount                 # confirms data_root is on your encrypted, mounted volume
uv run imsg migrate                     # creates the Postgres schema
uv run imsg check-permissions           # re-check FDA/Contacts/mount/Postgres before the real run
uv run imsg snapshot                    # takes a safe, read-only copy of your live chat.db
uv run imsg extract                     # decodes the snapshot into Postgres (via imsg-dump)
uv run imsg identity import             # resolves phone numbers/handles to people, via Contacts
uv run imsg segment                     # groups messages into topical conversation segments
uv run imsg embed                       # embeds segments for search (loads the embedding model)
uv run imsg enrich                      # OCR / captioning / transcription of attachments
uv run imsg status                      # confirms everything above actually landed
```

After the first full run, `imsg sync` (which the pipeline also runs on
its own schedule, per `sync.interval_seconds` in your config) keeps the
index caught up as new messages arrive — you don't need to re-run
`snapshot`/`extract`/`segment`/`embed` by hand every time.

## Step 10: Connect it to Claude

You built a local MCP server (`imsg mcp local`) in the steps above.
Now tell Claude how to start it. The two Claude apps use different
registration mechanisms:

### Claude Desktop

Claude Desktop's MCP config lives at
`~/Library/Application Support/Claude/claude_desktop_config.json` on
macOS (Settings -> Developer -> Edit Config also opens it). It does
**not** inherit your shell's environment, so both the binary path and
`IMSG_CONFIG` must be absolute paths written directly into the file —
see `examples/claude_desktop_config.json` for the exact shape:

```json
{
  "mcpServers": {
    "imsg-local": {
      "command": "/Users/alice/imessage-index/.venv/bin/imsg",
      "args": ["mcp", "local"],
      "env": {
        "IMSG_CONFIG": "/Users/alice/imessage-index/config.yaml"
      }
    }
  }
}
```

Copy that file (with your own paths substituted) into the location
above, then fully quit and restart Claude Desktop.

### Claude Code

Claude Code registers MCP servers with `claude mcp add`, using `--` to
separate Claude's own flags from the server's command and `-e` for
environment variables:

```bash
claude mcp add \
  --transport stdio \
  --scope user \
  -e "IMSG_CONFIG=/Users/alice/imessage-index/config.yaml" \
  imsg-local \
  -- /Users/alice/imessage-index/.venv/bin/imsg mcp local
```

`examples/claude-code-mcp-add.sh` does this for you, deriving the
paths from wherever you run it — no editing required:

```bash
./examples/claude-code-mcp-add.sh
```

## Step 11: Check it worked

```bash
claude mcp list                # Claude Code: confirms imsg-local is registered and connects
```

In Claude Desktop, restart the app and look for the connector/tools
indicator; `imsg-local` should be listed with its tools available. Ask
Claude something about your own messages — a real answer citing actual
conversations means the whole chain (Postgres, models, MCP) is working.

If you'd rather check outside of Claude first, run the server by hand
and watch its startup log on stderr:

```bash
uv run imsg mcp local
```

A working startup prints `models: backend=real`, then warm-up progress
for each model, then sits waiting for a client on stdio — that's normal
and expected; press Ctrl-C to stop it once you've confirmed no errors
appeared.

## Troubleshooting

- **`imsg: mount gate failed: ...`** — `data_root` isn't resolving onto
  a mounted, encrypted volume, or the `.imsgindex-volume` sentinel file
  from Step 2 is missing. Re-check both.
- **`postmaster became multithreaded during startup`** — you forgot
  `export LC_ALL=C` before starting Postgres (Step 3).
- **`full_disk_access: False` in `imsg check-permissions`** — Full Disk
  Access wasn't actually granted to the terminal app you're running
  `imsg` from, or you granted it to a different app than the one you're
  using now. Re-check Step 7, and remember this can never be fixed over
  SSH — it needs a real GUI session on the Mac itself.
- **Contacts names not resolving (`identity import` fails or leaves
  handles unresolved)** — check Contacts access under Privacy &
  Security; the first `identity import` run should have prompted for
  it, but if you dismissed the prompt you'll need to add the terminal
  app manually the same way as Full Disk Access.
- **`database.dsn must target port 5433`** or similar config errors —
  `imsg check-permissions`/`imsg migrate` print exactly which field
  failed and why; the config loader never lets `config.yaml` silently
  do the wrong thing. Compare against `examples/config.single-mac.yaml`
  and `config.example.yaml`.
- **Claude Desktop doesn't show the server, but `imsg mcp local` runs
  fine by hand** — almost always a relative path or a missing `env` in
  `claude_desktop_config.json` (Step 10); Claude Desktop's own MCP logs
  are at `~/Library/Logs/Claude/mcp*.log` and will show why the process
  failed to start or exited immediately.
- **A pipeline stage is stuck / slow** — `imsg status` reports whether
  another process is holding the host-wide heavy-model lock (only one
  model-loading stage runs at a time by design); `imsg embed --no-wait`
  and similar flags fail fast naming the holder instead of waiting.
- **General diagnostic shortcut:** if `scripts/doctor.py` exists in
  your checkout, `uv run python scripts/doctor.py` runs a plain-English
  version of most of the checks above in one shot.

## Optional: enterprise export

If you later want to push a filtered, allowlisted subset of your
messages into a company (Google Cloud Discovery Engine) search index,
that's a separate feature with its own extra:

```bash
uv sync --extra export
```

Without this extra installed, `imsg export ...` commands refuse with a
clear error naming `uv sync --extra export`, rather than a confusing
Python import failure — you never need to think about this at all for
personal local search.
