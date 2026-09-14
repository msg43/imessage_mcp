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

**No model has been executed end to end yet.** 1,188 tests (990 run
without a database; 198 integration tests need a live PostgreSQL and
skip cleanly without one — measured 2026-09-14), the CLI works,
migrations apply against real PostgreSQL + pgvector, and the snapshot →
extract → identity stages have run against a real corpus. Segmentation,
embedding, enrichment and retrieval have only ever run on the fake
providers or on tiny stand-in models (see
[what is still unverified](#replacing-the-model-providers)).

**It ships deterministic *fake* model providers alongside the real
wiring.** The embedding, reranking, captioning, OCR, transcription, and
segmentation providers each have a correctly-dimensioned stub that lets
the pipeline run end to end in tests. `imsg.providers.factory` selects
between the real implementations and those stubs with one config field,
`models.backend` (`real` is the default; `fake` is explicit opt-in), and
every command that builds providers prints `models: backend=<real|fake>`
first. **Run the pipeline with `fake` and every stage will report success
while your search results are meaningless.** The real implementations
and their pins are described in
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
| Real model providers (MLX text/reranker/boundary LLM, Apple Vision OCR, mlx-whisper, mlx-vlm captions, PE-Core) | Implemented behind `models.backend: real` (the default); **pinned weights never downloaded, never executed end to end** |
| Runs against real data | Snapshot → extract → identity: yes. Segment / embed / enrich with the pinned models: **never** |

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
cd imessage-index && uv sync --extra models --extra dev
```
The `models` extra installs the real model runtimes (mlx, mlx-lm,
mlx-whisper, mlx-vlm, torch, open_clip, pyobjc's Vision bridge,
pillow-heif); plain `uv sync` is enough for the `fake` backend and the
test suite.
```bash
cargo build --release --manifest-path tools/imsg-dump/Cargo.toml
```
```bash
uv run pytest        # 990 passed, 198 skipped (integration tests need a live DB) — 2026-09-14
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
real implementations exist; what is missing is any run of them against
the pinned weights. `imsg.providers.factory` is the only place providers
are constructed. `models.backend` in `config.yaml` selects `real` (the
default — MLX text embedding, reranker and boundary LLM; Apple Vision
OCR; mlx-whisper transcription; mlx-vlm captioning; PE-Core multimodal
embedding) or `fake` (the deterministic stand-ins, explicit opt-in);
every command that builds providers prints `models: backend=<...>`
first. The factory imports the real classes lazily, by dotted path, and
builds them from the repo ids and immutable revisions in `config.yaml`,
whose defaults mirror `models/manifest.lock.yaml` — repo, commit sha,
license, expected dimension, quantization, runtime floors and a
smoke-test record per model. `uv run imsg models verify` (also
`scripts/verify_model_manifest.py`) re-resolves each repo against the
Hugging Face API and checks the installed packages against the floors;
it reports drift and never rewrites the lock unless given `--write`.
The runtime packages live behind the `models` extra; a missing package
fails as one clear `imsg: ...` line, not a traceback. The two fixed
prompts (`prompts/segment_boundaries.txt`, `prompts/caption.txt`) ship
in the repo and are used unless `paths.data_root` holds a copy at the
same relative path; each run prints which file it used, because those
bytes are hashed into `seg_config_hash` and the caption provenance.

**What is still unverified (2026-09-14).** No pinned model has been
downloaded or executed end to end: every hosted entry in the manifest
carries `smoke_test: not_run`. What *has* been checked: every runtime
API name the providers call exists with the assumed signature in the
installed packages; the pinned Qwen3.5 chat template honours
`enable_thinking=False`; the Apple Vision OCR provider ran on this
machine's framework and read a generated image back verbatim (recorded
in the manifest); the transcription chain (ffmpeg → mlx-whisper) returned
an exact transcript of synthesised speech on a 50 MB `whisper-tiny`
stand-in; and the three mlx-lm providers loaded a 79 MB stand-in model
and produced correctly shaped, padding-invariant output. None of that
says anything about retrieval quality with the pinned 8B / 35B weights —
establishing that is the first real task.

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
