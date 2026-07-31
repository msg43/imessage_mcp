# imessage-index

A local-first iMessage retrieval system. Extracts your full message
history, resolves identities, segments conversations topically, embeds
and indexes them, and serves hybrid full-text + vector search over an
MCP server — so an AI assistant can actually answer questions about
your own correspondence.

Designed to run unattended on a headless Mac, with all derived state on
an encrypted volume and nothing leaving the machine unless you
explicitly allowlist it.

---

## ⚠️ Read this before you clone

**This code has never run against a real corpus.** 836 tests pass (639
without a database, plus 197 integration tests that need a live
PostgreSQL and skip cleanly without one), the CLI works, migrations
apply against real PostgreSQL + pgvector — and no message has ever been
indexed by it.

**It ships deterministic *fake* model providers.** The embedding,
captioning, OCR, transcription, and segmentation providers are
correctly-dimensioned stubs (marked `PLACEHOLDER` in `cli.py`) that let
the pipeline run end to end in tests. **Run the pipeline as-is and every
stage will report success while your search results are meaningless.**
Replacing them is the first real task — see
[Replacing the model providers](#replacing-the-model-providers).

Treat this as a **thoroughly tested skeleton with a complete design
behind it**, not a working product. If you want something that works
today, this isn't it. If you want a rigorous starting point that has
already made and documented the non-obvious decisions, read on.

### What is genuinely done

| Area | State |
|---|---|
| Schema + migrations | Complete; applied against live PostgreSQL 17 + pgvector |
| Pipeline stages (snapshot → extract → identity → segment → enrich → embed → sync → export) | Implemented; unit + integration tested |
| Hybrid retrieval (BM25 + text vector + multimodal vector, RRF-fused, reranked) | Implemented |
| Local MCP surface (stdio) | Implemented |
| Public MCP surface (StreamableHTTP + OAuth) | Implemented; never exposed |
| Export gate (default-deny, plan/approve/push) | Implemented; transport never wired to a live API |
| Eval harness (nDCG@k, recall@k, MRR) | Implemented; metrics verified against hand-computed fixtures |
| Real model providers | **Not implemented** |
| Any run against real data | **Never happened** |

---

## Architecture in one pass

```
chat.db ──.backup──▶ snapshot ──▶ extract ──▶ identity resolution
                                                     │
                                              segmentation
                                                     │
                                  ┌──────────────────┴─────────────┐
                             text bodies                     attachments
                                                                   │
                                          ┌────────────────────────┼───────────┐
                                    PDF text layer              images    audio/video
                                          └────────────────────────┼───────────┘
                                                                   │
                                      embed ──▶ pgvector + FTS5 + multimodal vector
                                                                   │
                                          ┌────────────────────────┴──────────┐
                                    local MCP                         filtered export
                                   (full corpus)                       (allowlisted)
```

Four choices that shaped everything else:

- **Segments, not messages.** Retrieval returns topically coherent
  conversation segments. Message-granular results flood a model's
  context with "sounds good" and prevent reconstructing what was
  actually discussed.
- **Everything becomes text first.** OCR output, captions, transcripts,
  and PDF text layers all embed into one text space, so failures stay
  debuggable — when something doesn't surface you can read the
  extracted text and see why. A second multimodal vector runs alongside
  for visual similarity.
- **Identity resolution precedes segmentation.** Nothing downstream keys
  on a raw handle. One person legitimately has many numbers, emails and
  aliases across a decade.
- **Default deny on export.** Nothing reaches an external index unless
  explicitly allowlisted, and a group thread requires *every*
  participant allowlisted.

---

## Requirements

- **macOS on Apple Silicon.** The pipeline depends on macOS-only APIs:
  Full Disk Access to Messages, the Contacts framework, Apple Vision.
- **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/).
- **PostgreSQL 17 + pgvector**, as a dedicated instance.
- **Rust toolchain**, to build the extraction shim.
- **≥ 32 GB unified memory** for 8B-class local models; smaller models
  work with reduced quality.

## Getting started

```bash
git clone https://github.com/msg43/imessage_mcp.git imessage-index
cd imessage-index && uv sync
```
```bash
cargo build --release --manifest-path tools/imsg-dump/Cargo.toml
```
```bash
uv run pytest        # 639 passed, 197 skipped (integration tests need a live DB)
```

Copy `config.example.yaml`, fill it in, and point the CLI at it:

```bash
export IMSG_CONFIG=/path/to/your/config.yaml
```
```bash
uv run imsg check-permissions && uv run imsg migrate
```

Secrets are never stored in config — they resolve from the macOS
Keychain (`keychain:<item>`) or the environment (`env:<VAR>`), and
config validation rejects anything that looks like a literal secret.

> **Two macOS gotchas that will each cost you an hour.**
> PostgreSQL needs `export LC_ALL=C` or the postmaster dies at startup
> with *"postmaster became multithreaded during startup"* — which reads
> like a corrupt installation and is not. And **Full Disk Access cannot
> be granted over SSH**: TCC prompts require a GUI session, and the
> grant goes to the binary that *launches* the job, not to Messages.

Then `uv run imsg --help`. Every stage supports `--dry-run`.

## Replacing the model providers

This is the work between "tests pass" and "it does something." The
interfaces already exist and are correctly dimensioned — implement them
and inject your implementations where `cli.py` currently constructs the
fakes.

| Interface | What it needs |
|---|---|
| `TextEmbeddingProvider` | `embed_documents()` (bare) and `embed_query()` (instruction-prefixed); **2048-dim**, L2-normalized |
| `MultimodalEmbeddingProvider` | `embed_images()` and `embed_text()` (paired towers); **1280-dim** |
| `BoundaryProvider` | Topical boundary indices for a window of messages |
| `OcrProvider` / `CaptionProvider` / `TranscriptionProvider` | One method each |

The reference design uses local MLX-hosted models throughout, on the
reasoning that sending a decade of personal messages to a hosted API is
a categorically different decision from indexing them on your own
machine — and that as of mid-2026 the leading open text-embedding models
top the benchmarks anyway, so there is little quality left to trade for
it. Nothing in the code requires that choice; the interfaces are
provider-agnostic.

⚠️ **The dimensions are load-bearing** — see the first item below.

---

## Things that will bite you

Learned the expensive way; written down so you don't have to.

- **pgvector's index caps are lower than its type limits.** The
  `vector` and `halfvec` types accept up to 16,000 dimensions, but
  **HNSW/IVFFlat indexes cap at 2,000 (`vector`) and 4,000
  (`halfvec`)**. A column can be perfectly legal DDL whose index can
  never be created — the error surfaces at `CREATE INDEX`, and ignoring
  it means silently falling back to sequential scan.
  `scripts/lint_ddl.py` exists solely to catch this. An earlier
  revision of this project specified an unbuildable `halfvec(4096)` for
  exactly this reason.
- **Audience validation and subject validation are not redundant.** On
  the public surface the subject check answers *"is this the owner?"*
  and the audience check answers *"was this token minted for this
  system?"* A user's OAuth subject is identical across every app they
  sign into, so subject-checking alone does not stop a token minted for
  another application being replayed here. Both, or neither works.
- **Applied migrations are immutable**, enforced by hash. Correct a
  mistake in a *later* migration; never edit a shipped one.
- **`updated_at` is enforced by trigger, not convention.**
  Re-segmentation keys off it, so a writer that forgot to bump it would
  strand chats out of reprocessing — no error, stale results the only
  symptom.
- **Filters must overfetch, not post-filter.** Post-filtering a fixed
  top-K silently starves results when filters are selective: you get
  few or zero hits and it looks like "nothing matched."
- **Ingest-time and query-time text normalization must match exactly.**
  If they drift, exact-phrase search silently stops working.

---

## Layout

```
src/imsg/
  config/      config surface + validation (enforces the safety rules)
  db/          connection, migrations, cluster fingerprint
  stages/      snapshot, extract, identity, sync
  segment/  backfill/  enrich/  embed/    indexing pipeline
  retrieval/   hybrid query flow, RRF fusion, reranking
  mcp/         auth boundary, local + public surfaces, tools
  export/      default-deny eligibility, plan/approve/push
  eval/        metrics, runner, diff
  verify/      seed completeness, attachment reconciliation
migrations/         schema, applied in order by a hash-checked runner
tools/imsg-dump/    GPL-3.0 Rust extraction shim (subprocess only)
```

## Development

```bash
uv run ruff check . && uv run mypy . && uv run pytest
```

Integration tests run against a live PostgreSQL when one is reachable
and skip cleanly when it isn't; the unit suite never needs a database.

## Licensing

The core is **MIT** — see [`LICENSE`](LICENSE). Component licensing and
the reasoning behind the split are in [`NOTICE`](NOTICE).

In short: `tools/imsg-dump/` is **GPL-3.0** and carries its own
`LICENSE`. It links the GPL `imessage-database` crate to parse the
`attributedBody` typedstream format, which is not optional — since Big
Sur much of a message's text is not in the `text` column at all, and
readers that only query that column silently return empty strings for
large portions of modern history.

**It is invoked strictly across a process boundary** — spawned as a
subprocess, never linked into the Python code. That boundary is
deliberate and load-bearing for the licensing split. Vendor it
differently and that is yours to reason about.

## What isn't here

Instance configuration, by design: real config values, contact seed
data, allowlists and eval queries live in a separate private overlay
you supply and point at with `IMSG_CONFIG`. This repo is public-safe by
construction — no real names, hosts or secrets have ever been committed
to it, and `config.example.yaml` ships placeholders only.

The design record — architecture rationale, full build spec, and the
decision log explaining *why* each choice above was made — is kept
private, since it's written against a specific deployment.
