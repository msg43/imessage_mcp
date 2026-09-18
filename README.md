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

**No model has run on the pipeline end to end yet.** 1,315 tests (1,116
run without a database; 199 integration tests need a scratch PostgreSQL
and skip cleanly without one — measured 2026-09-15), the CLI works,
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
| Real model providers (MLX text/reranker/boundary LLM, Apple Vision OCR, mlx-whisper, mlx-vlm captions, PE-Core) | Implemented behind `models.backend: real` (the default); every pinned model smoke-run once on a synthetic input (2026-09-14/15, results in the lock); **never run on the pipeline end to end** |
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

**What has been run, and what is still unverified (2026-09-15).**
`uv run python scripts/smoke_test_models.py` (also
`imsg.providers.model_smoke`) downloads every pinned model at its sha,
checksums it (`artifact_sha256`, defined in the lock header), builds the
real provider through the factory and runs one fictional input per role,
recording load time, inference time and peak memory per model into the
lock with `--write`. On an M2 Ultra with 128 GB every entry passes. Still
unverified: `segment` / `embed` / `enrich` on a real chat, batched
throughput and memory (the recorded peaks are single-input), the
deployment host's memory, and retrieval quality with the pinned weights —
establishing that is the first real task.

**The reranker is a local conversion, not a Hub download.** The only
8-bit MLX conversion of Qwen3-Reranker-8B on the Hub ships no `lm_head`
tensor, so the model card's yes/no logits cannot be computed from it (the
2026-09-14 smoke run found this; the evidence is kept in the lock entry's
notes). The lock therefore pins the reranker as `source: local_conversion`:
the upstream repo and commit sha, the license, the converter
(`mlx-lm==0.31.3`), the exact `command` that produces it, its
`output_dir` relative to `paths.data_root`, and the `artifact_sha256` of
that directory. The pinned reranker is **Qwen3-Reranker-0.6B** (0.58 GiB
on disk) since 2026-09-17, for the latency budget below; the 8B
conversion stays in the lock as `status: retained` — pinned and verified
exactly like an active entry, claiming no role, kept for a
retrieval-quality comparison. `imsg models verify` prints which entry is
active for every role and lists the retained ones separately. To
reproduce a conversion, run the recorded command with `$DATA_ROOT` set to
your data root and the `models` extra installed (a few seconds for the
0.6B, about 16 for the 8B), then
`scripts/smoke_test_models.py --only qwen3-reranker-0.6b --data-root
$DATA_ROOT` to confirm the digest — re-running the 0.6B recipe on
2026-09-17 reproduced its artifact byte for byte. In `config.yaml`,
`retrieval.reranker_model` names that directory (data-root-relative) and
`retrieval.reranker_revision` the *upstream* commit; the factory reads
the value as a local directory when it exists under the data root and as
a Hugging Face repo id otherwise, and the provider records `model_id` as
`<dir>@<upstream sha>`. `uv run imsg models verify --data-root
$DATA_ROOT` re-resolves the upstream repo for drift, recomputes the
directory's digest against the lock, and checks the runtimes; `--write`
never advances an upstream pin (the directory was converted from the
pinned commit — re-convert and re-pin by hand to move it).

**Search latency: the reranker's size, and whether the index is in
memory.** Measured end to end through the real MCP surface on an M2 Ultra
on 2026-09-17 — `imsg mcp local` under a stdio client, 20 fictional
queries, three cold starts, with another job using the same disk array at
216-2,177 MB/s throughout — a whole `search_messages` round trip took
**p50 0.87 s, p95 1.14 s, max 1.28 s**. Where it goes, p95 per stage:
reranking 0.67 s, the full-text channel 0.24 s, the two vector channels
0.05 and 0.03 s, the query embedding 0.06 s, the multimodal text tower
0.02 s, summary fetch 0.002 s, the audit row 0.07 s.

Three settings decide most of that, and all three were chosen by
measurement rather than taste:

- **Which reranker.** Qwen3-Reranker-0.6B, at `retrieval.rerank_top` 20
  and `retrieval.rerank_doc_max_tokens` 256. The 8B conversion pinned
  before it reads roughly 650-780 tokens a second here: scoring 50
  uncapped candidates meant ~20,000 tokens and a p95 of 36 s, and its
  fastest setting that fit a budget (10 candidates, 64 tokens, p95 1.97 s
  of reranking) agreed with its own full ranking no better than doing no
  reranking at all. How many candidates are reranked matters more than
  the reranker's size: with 10, the returned top 10 *is* the fused top 10
  in another order, so reranking cannot lift anything from further down.
- **How much of the HNSW index each search reads.**
  `retrieval.hnsw_ef_search`, applied with `SET LOCAL` in each channel's
  own transaction, defaults to 1000 — pgvector's maximum. On the live
  index, recall@100 against exact search rises 0.944 -> 1.000 (text
  channel) and 0.792 -> 0.998 (image channel) from pgvector's default of
  40 to 1000, and the worst single query rises from 0.79 and 0.38 to 1.00
  and 0.98, for about 50 ms more per query.
- **Whether the pages are cached.** See "Things that will bite you" below:
  `shared_buffers` holds the search working set, `pg_prewarm` fills it at
  startup, and `imsg status` says so.

`scripts/bench_retrieval_latency.py` sweeps the reranker settings against
the real index and reports a quality *proxy* — agreement with scoring 50
uncapped candidates, not an evaluation. Re-run it before moving either
setting, and let the eval harness settle what latency costs.

**Startup.** `imsg mcp local` answers the MCP handshake in about 1 s
(Claude Code gives up on a server that has not answered within
`MCP_TIMEOUT`, 30 s by default) and warms up in the background, logging
each step's time on stderr: text embedder 4.8-12.5 s, PE-Core text tower
33.8-50.2 s, reranker 2.7-3.2 s, database buffer pool 0.1-14.3 s —
41.8-65.4 s in all, against 121 s before the 0.6B was pinned. A retrieval
tool call that arrives during warm-up waits up to 90 s for it, then
returns `WARMING_UP` with an estimate of the seconds remaining; a model
that fails to load makes every such call return `WARM_UP_FAILED` with the
cause. `check_permissions` is the exception: it is diagnostics, so it
answers straight away and carries the warm-up's own state (which step is
loading, how many are done, the estimate, the failure) — the way to find
out what a server that is not answering searches is doing. The first
query after warm-up still costs a little more than the rest (0.94-4.08 s
against a 0.87 s median), and the extra is the query embedder's first
forward pass at a real shape: 0.13-3.07 s against 0.055 s afterwards.

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
- **Vector search is only fast while its index is in memory.** An HNSW
  query touches a few thousand pages of an index far larger than the
  default 128 MB `shared_buffers`. Measured on the external encrypted
  volume (2026-09-16): queries whose pages the OS had not cached took p95
  302 ms (text vectors) and 609 ms (image vectors), against 37 ms and
  70 ms once the index files were in the page cache — and with another
  job's heavy I/O on the same disk, several seconds each. The fix is in
  three parts: `shared_buffers` sized to hold what search reads (measured
  with `pg_statio_*` deltas over the benchmark queries — 2,296 MiB here,
  of which 1,193 MiB is the three HNSW indexes and 857 MiB the embedding
  tables' TOAST), `pg_prewarm` (migration 0004) filling it — at every
  server start via `RetrievalService.warm_up()`, and after a reboot via
  `pg_prewarm.autoprewarm` — and `imsg status`, which prints
  `shared_buffers` against the total HNSW index size and warns when the
  pool is the smaller of the two.
- **Two providers pinned to the same checkpoint still load it twice
  unless something makes them share.** Captioning goes through `mlx-vlm`
  and topical boundary detection through `mlx-lm`; both name the same
  35B repo and revision, and each used to build its own model object.
  Measured: MLX active memory 18.99 → 37.15 GiB as the second one
  loaded, in one process, for one set of weights. The fix is
  `imsg.shared_vlm_runtime.SharedVlmRuntime`, an object a caller passes
  to both providers — boundary detection then runs text-only through the
  already-loaded vision-language model, which is the *same* computation
  (the rendered chat prompt is byte-identical and the last-position logit
  row is bit-identical across all 248,320 float32 values). It is
  explicit rather than a module-level cache precisely so the sharing is
  visible at the call site; `models.share_boundary_and_caption_weights`
  turns it off.
- **A CLIP-style model loads both towers whether or not you use both.**
  PE-Core is 9.01 GiB at fp32, of which the vision tower is 7.01 and the
  text tower 2.00 — and the query server only ever calls `embed_text`
  while the embedding pipeline only ever calls `embed_images`. The
  provider now decides on first use and releases the other tower;
  `scripts/verify_pe_core_tower_selection.py` proves the vectors are
  identical bit for bit before and after, because "it cannot change the
  answer" is an argument, not a measurement. A process that does use both
  rebuilds once and keeps both.
- **MLX's buffer cache defaults to its memory limit, which is not a
  bound.** It pools freed GPU buffers rather than returning them, and on
  a 64 GB host the default limit probes at 60.8 GiB; one enrichment
  process was seen holding 37.25 GiB of freed buffers, which is
  indistinguishable from a leak and hides real regressions. Every MLX
  provider now calls `imsg.mlx_runtime.bound_buffer_cache` at load
  (`models.query_cache_limit_bytes` / `models.enrichment_cache_limit_bytes`).
  The call is process-wide, so one provider bounding it covers the rest
  — including `mlx_whisper`, which has no load hook of its own.
- **Two GPU-heavy jobs on one machine need an arbiter, and it has to
  survive a crash.** The nightly enrichment window overlaps the
  always-on MCP server; with the duplicate weights gone and no swap at
  all, query p95 still trebled purely from GPU contention. Enrichment
  now stands aside for in-flight queries
  (`imsg.db.enrichment_yield_locks`, `enrichment.yield_to_queries`), and
  the signal is a **Postgres session-level advisory lock** specifically
  because the server releases it when the session ends, however it ends
  — a killed MCP server cannot leave enrichment paused, with no timeout
  to tune and no stale marker to reap. It is checked between units of
  work, never during one, so no claimed task is ever abandoned;
  `imsg status` reports whether it is yielding right now. **What it
  cannot do** is help when the unit of work is long relative to the
  query rate: measured with it on and off, search p95 during the window
  was 3.00 s versus 3.06 s — one caption takes 14 s on that host and a
  search arrives every 2 s, so a worker that resumes between tasks
  resumes straight into another 14-second caption. The memory fixes
  above are what closed the gap; this is kept because it costs one round
  trip when nobody is searching and it will matter wherever the batch is
  finer-grained than the traffic — not because it earned its place on
  this workload.
- **The planner can abandon an HNSW index at some `ef_search` values, and
  that is not a monotonic effect.** pgvector's own cost estimate bounds
  layer-0 tuples by `ef_search` while its selectivity term carries
  `log(ef_search)` in a denominator, so the estimated cost climbs and
  then drops back: on this index it was 3,252 at `ef_search` 40, 13,287
  at 280 and 3,719 at 285, against a flat 8,827 for the sequential
  alternative — which the planner undercosts anyway, because (pgvector's
  own FAQ) it "doesn't consider out-of-line storage in cost estimates"
  and the vectors it would sort are 642 MiB of TOAST. Between 170 and 284
  the same query took 245 ms instead of 9 ms, exactly and only because of
  the plan. Every vector channel therefore sets `enable_seqscan = off`
  for its own transaction, pgvector's documented remedy.

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
