-- 0001_initial.sql
-- Transcribed verbatim from docs/SPEC.md §7.2 (v1.1, D6-revised) of the
-- imessage-index build specification. Do not hand-edit the DDL here
-- without updating the spec first — migrations are immutable once
-- merged (SPEC §7.1); a real change is a new migration file.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE schema_migrations (
  version     int PRIMARY KEY,
  sha256      text NOT NULL,
  applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE imsg_meta (
  key         text PRIMARY KEY,
  value       text NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now()
);
-- `imsg db init` inserts imsg_meta.cluster_uuid before ordinary startup (§5.2).

CREATE TYPE handle_kind AS ENUM ('phone','email','apple_id','unknown');
CREATE TYPE service_kind AS ENUM ('imessage','sms','rcs','unknown');
CREATE TYPE chat_kind AS ENUM ('dm','group');
CREATE TYPE materialization_state AS ENUM ('dataless','materializing','materialized','missing','error');
CREATE TYPE enrichment_kind AS ENUM ('pdf_text','ocr','caption','transcript','frame_ocr');
CREATE TYPE enrichment_state AS ENUM ('pending','running','done','failed','skipped');
CREATE TYPE index_entity_kind AS ENUM ('segment','attachment_chunk');
CREATE TYPE index_operation AS ENUM ('upsert','delete');

CREATE TABLE person (
  person_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  display_name  text NOT NULL,
  short_name    text NOT NULL UNIQUE,          -- stable slug used in MCP responses
  is_owner      boolean NOT NULL DEFAULT false,
  organization  text,
  needs_review  boolean NOT NULL DEFAULT true, -- auto-created stubs until hand-verified (Phase 1 gate)
  notes         text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX person_owner_singleton ON person ((true)) WHERE is_owner;

CREATE TABLE source_handle (                    -- S2-only raw provenance; never queried downstream (§1.3)
  source_handle_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  raw_value        text NOT NULL,
  service          service_kind NOT NULL DEFAULT 'unknown',
  first_seen       timestamptz,
  last_seen        timestamptz,
  UNIQUE (raw_value, service)
);

CREATE TABLE handle (                           -- S3 canonical identity mapping
  handle_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  person_id        bigint NOT NULL REFERENCES person(person_id),
  kind             handle_kind NOT NULL,
  normalized_value text NOT NULL,              -- E.164 for phones; lowercased for emails
  first_seen       timestamptz,
  last_seen        timestamptz,
  UNIQUE (normalized_value, kind)
);
CREATE INDEX handle_person_idx ON handle (person_id);

CREATE TABLE source_handle_resolution (
  source_handle_id bigint PRIMARY KEY REFERENCES source_handle(source_handle_id),
  handle_id        bigint NOT NULL REFERENCES handle(handle_id)
);

CREATE TABLE chat (
  chat_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_guid  text NOT NULL UNIQUE,           -- chat.db chat.guid
  thread_key   text NOT NULL UNIQUE,           -- opaque sha256-based key returned by MCP (§10.2)
  kind         chat_kind NOT NULL,
  display_name text,
  service      service_kind NOT NULL DEFAULT 'imessage',
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chat_participant (
  chat_id   bigint NOT NULL REFERENCES chat(chat_id),
  person_id bigint NOT NULL REFERENCES person(person_id),
  PRIMARY KEY (chat_id, person_id)
);
CREATE INDEX chat_participant_person_idx ON chat_participant (person_id);

CREATE TABLE chat_participant_source (          -- populated by S2, resolved by S3
  chat_id          bigint NOT NULL REFERENCES chat(chat_id),
  source_handle_id bigint NOT NULL REFERENCES source_handle(source_handle_id),
  PRIMARY KEY (chat_id, source_handle_id)
);

CREATE TABLE extraction_run (
  run_id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_name      text NOT NULL,              -- 'mini' | 'studio-seed' (config sync.sources)
  snapshot_path    text NOT NULL,
  snapshot_sha256  text NOT NULL,
  rowid_before     bigint,
  rowid_after      bigint,
  messages_upserted bigint,
  started_at       timestamptz NOT NULL DEFAULT now(),
  finished_at      timestamptz,
  status           text NOT NULL DEFAULT 'running'
                   CHECK (status IN ('running','ok','failed'))
);

CREATE TABLE message (
  message_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_guid      text NOT NULL UNIQUE,       -- chat.db message.guid; dedupe key across sources
  message_key      text NOT NULL UNIQUE,       -- opaque sha256-based key returned by MCP (§10.2)
  chat_id          bigint NOT NULL REFERENCES chat(chat_id),
  sender_source_handle_id bigint REFERENCES source_handle(source_handle_id), -- S2 provenance
  sender_person_id bigint REFERENCES person(person_id),  -- NULL only until S3 resolves incoming rows
  is_from_me       boolean NOT NULL,
  sent_at          timestamptz NOT NULL,
  service          service_kind NOT NULL,
  text_original    text,                       -- verbatim decoded body
  text_normalized  text,                       -- indexed copy (§9.2 normalization)
  is_unsent        boolean NOT NULL DEFAULT false,
  is_edited        boolean NOT NULL DEFAULT false,
  date_edited      timestamptz,
  reply_to_guid    text,
  has_attachments  boolean NOT NULL DEFAULT false,
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX message_chat_sent_idx   ON message (chat_id, sent_at);
CREATE INDEX message_sender_idx      ON message (sender_person_id, sent_at);
CREATE INDEX message_sent_idx        ON message (sent_at);
CREATE INDEX message_unsent_idx      ON message (is_unsent) WHERE is_unsent;
CREATE INDEX message_unresolved_sender_idx ON message (message_id)
  WHERE sender_person_id IS NULL;              -- S3 invariant + S4 gate scan (§8 S3)

CREATE TABLE message_source (                -- multi-source provenance; no last-source-wins overwrite
  message_id        bigint NOT NULL REFERENCES message(message_id) ON DELETE CASCADE,
  source_name       text NOT NULL,
  source_rowid      bigint NOT NULL,
  extraction_run_id bigint NOT NULL REFERENCES extraction_run(run_id),
  PRIMARY KEY (source_name, source_rowid),
  UNIQUE (message_id, source_name)
);

CREATE TABLE message_version (           -- prior versions of edited messages (D1)
  message_id  bigint NOT NULL REFERENCES message(message_id),
  version_idx smallint NOT NULL,          -- 0 = original
  text        text NOT NULL,
  edited_at   timestamptz,
  PRIMARY KEY (message_id, version_idx)
);

CREATE TABLE tapback (                    -- folded metadata; never standalone documents
  tapback_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_guid       text NOT NULL UNIQUE,
  target_source_guid text NOT NULL,
  target_message_id bigint REFERENCES message(message_id), -- backfilled when target is present
  sender_source_handle_id bigint REFERENCES source_handle(source_handle_id),
  sender_person_id  bigint REFERENCES person(person_id),
  is_from_me        boolean NOT NULL,
  kind              text NOT NULL,        -- loved|liked|disliked|laughed|emphasized|questioned|emoji:<char>|sticker
  removed           boolean NOT NULL DEFAULT false,
  acted_at          timestamptz
);
CREATE INDEX tapback_target_idx ON tapback (target_message_id);
CREATE INDEX tapback_unresolved_target_idx ON tapback (target_source_guid)
  WHERE target_message_id IS NULL;

CREATE TABLE link_preview (
  message_id bigint NOT NULL REFERENCES message(message_id),
  url        text NOT NULL,
  title      text,
  summary    text,
  site_name  text,
  PRIMARY KEY (message_id, url)
);

CREATE TABLE attachment (
  attachment_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_guid   text NOT NULL UNIQUE,     -- chat.db attachment.guid
  attachment_key text NOT NULL UNIQUE,    -- opaque sha256-based key returned by MCP (§10.2)
  filename      text,
  source_path   text,                     -- original ~/Library/Messages/Attachments path
  cache_path    text,                     -- content-addressed copy under $DATA_ROOT/attachments
  uti           text,
  mime_type     text,
  byte_size     bigint,
  sha256        text,
  state         materialization_state NOT NULL DEFAULT 'dataless',
  materialization_attempts smallint NOT NULL DEFAULT 0,
  materialization_next_attempt_at timestamptz NOT NULL DEFAULT now(),
  materialization_last_error text,
  is_sticker    boolean NOT NULL DEFAULT false,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX attachment_state_idx   ON attachment (state);
CREATE INDEX attachment_sha_idx     ON attachment (sha256);

CREATE TABLE attachment_source (
  attachment_id bigint NOT NULL REFERENCES attachment(attachment_id) ON DELETE CASCADE,
  source_name   text NOT NULL,
  source_rowid  bigint NOT NULL,
  PRIMARY KEY (source_name, source_rowid),
  UNIQUE (attachment_id, source_name)
);

CREATE TABLE message_attachment (            -- chat.db uses an explicit join table; preserve it (D6)
  message_id    bigint NOT NULL REFERENCES message(message_id) ON DELETE CASCADE,
  attachment_id bigint NOT NULL REFERENCES attachment(attachment_id) ON DELETE CASCADE,
  ordinal       smallint,
  PRIMARY KEY (message_id, attachment_id)
);
CREATE INDEX message_attachment_attachment_idx ON message_attachment (attachment_id);

CREATE TABLE enrichment (
  attachment_id bigint NOT NULL REFERENCES attachment(attachment_id),
  kind          enrichment_kind NOT NULL,
  state         enrichment_state NOT NULL DEFAULT 'pending',
  model         text,                     -- 'vision.framework' | pinned caption model | 'whisper-large-v3' | 'pdftotext'
  model_version text,
  text          text,
  detail        jsonb,                    -- page map, frame timestamps, confidences
  attempts      smallint NOT NULL DEFAULT 0,
  next_attempt_at timestamptz NOT NULL DEFAULT now(),   -- queryable backoff (D6)
  locked_at     timestamptz,              -- worker lease (D6)
  locked_by     text,
  last_error    text,
  updated_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (attachment_id, kind)
);
CREATE INDEX enrichment_pending_idx ON enrichment (state, updated_at) WHERE state = 'pending';
CREATE INDEX enrichment_ready_idx ON enrichment (next_attempt_at, updated_at)
  WHERE state = 'pending';

CREATE TABLE session (
  session_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  chat_id    bigint NOT NULL REFERENCES chat(chat_id),
  started_at timestamptz NOT NULL,
  ended_at   timestamptz NOT NULL,
  gap_hours  real NOT NULL,               -- threshold used at build time
  UNIQUE (chat_id, started_at)
);

CREATE TABLE segment (
  segment_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  stable_key      text NOT NULL UNIQUE,   -- sha256(chat.source_guid || first_msg_guid || last_msg_guid || seg_config_hash)
  chat_id         bigint NOT NULL REFERENCES chat(chat_id),
  session_id      bigint NOT NULL REFERENCES session(session_id) ON DELETE CASCADE,
  seq_in_session  int NOT NULL,
  started_at      timestamptz NOT NULL,
  ended_at        timestamptz NOT NULL,
  message_count   int NOT NULL,
  token_count     int,
  rendered_text   text NOT NULL,          -- §9.1 rendering; the embedded/indexed unit
  rendered_sha256 text NOT NULL,
  topic_label     text,
  seg_config_hash text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX segment_chat_time_idx ON segment (chat_id, started_at);

CREATE TABLE segment_message (
  segment_id bigint NOT NULL REFERENCES segment(segment_id) ON DELETE CASCADE,
  message_id bigint NOT NULL REFERENCES message(message_id),
  PRIMARY KEY (segment_id, message_id)
);
CREATE UNIQUE INDEX segment_message_single ON segment_message (message_id);

CREATE TABLE segment_embedding (
  segment_id  bigint PRIMARY KEY REFERENCES segment(segment_id) ON DELETE CASCADE,
  model       text NOT NULL,
  dim         int NOT NULL DEFAULT 2048 CHECK (dim = 2048),
  text_sha256 text NOT NULL,              -- idempotency: skip if unchanged
  vec         halfvec(2048) NOT NULL,
  embedded_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX segment_embedding_hnsw ON segment_embedding
  USING hnsw (vec halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);
-- D6: the halfvec TYPE allows up to 16,000 dims, but HNSW/IVFFlat indexes on
-- halfvec cap at 4,000 — v1.0's halfvec(4096) column was legal DDL whose HNSW
-- index could never be created. Qwen3's MRL output at 2048 is the buildable
-- baseline (§4.1); revisit the dim at Phase 4 with eval numbers.

CREATE TABLE attachment_chunk (
  chunk_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  attachment_id bigint NOT NULL REFERENCES attachment(attachment_id) ON DELETE CASCADE,
  kind          enrichment_kind NOT NULL,
  seq           int NOT NULL,
  text          text NOT NULL,
  token_count   int,
  UNIQUE (attachment_id, kind, seq)
);

CREATE TABLE attachment_chunk_embedding (
  chunk_id    bigint PRIMARY KEY REFERENCES attachment_chunk(chunk_id) ON DELETE CASCADE,
  model       text NOT NULL,
  dim         int NOT NULL DEFAULT 2048 CHECK (dim = 2048),
  text_sha256 text NOT NULL,
  vec         halfvec(2048) NOT NULL,
  embedded_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX attachment_chunk_embedding_hnsw ON attachment_chunk_embedding
  USING hnsw (vec halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);

CREATE TABLE search_index_event (             -- transactional outbox for the FTS sidecar (D6)
  event_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  entity_kind    index_entity_kind NOT NULL,
  entity_id      bigint NOT NULL,
  operation      index_operation NOT NULL,
  content_sha256 text,                        -- required for upsert; NULL for delete
  created_at     timestamptz NOT NULL DEFAULT now(),
  CHECK ((operation = 'delete' AND content_sha256 IS NULL)
      OR (operation = 'upsert' AND content_sha256 IS NOT NULL))
);
CREATE INDEX search_index_event_entity_idx
  ON search_index_event (entity_kind, entity_id, event_id);

CREATE TABLE allowlist_person (           -- default deny: absence = denied
  person_id           bigint PRIMARY KEY REFERENCES person(person_id),
  text_allowed        boolean NOT NULL DEFAULT false,
  attachments_allowed boolean NOT NULL DEFAULT false,   -- gated separately (§11.2)
  note                text,
  added_at            timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE thread_classification (
  chat_id     bigint PRIMARY KEY REFERENCES chat(chat_id),
  state       text NOT NULL DEFAULT 'unreviewed'
              CHECK (state IN ('unreviewed','personal','business')),
  reviewed_at timestamptz
);

CREATE TABLE export_run (
  export_run_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  mode               text NOT NULL CHECK (mode IN ('reconcile','purge')),
  allowlist_snapshot jsonb NOT NULL,       -- frozen allowlist at run time (auditability)
  config_sha256      text NOT NULL,
  manifest_sha256    text NOT NULL,        -- exact staged bytes + intended deletes (§11.4)
  doc_count          int,
  approved_at        timestamptz,
  approved_manifest_sha256 text,           -- must equal manifest_sha256 on push
  approval_id        text,
  started_at         timestamptz NOT NULL DEFAULT now(),
  finished_at        timestamptz,
  status             text NOT NULL DEFAULT 'running'
                     CHECK (status IN ('running','planned','pushing','ok','failed'))
);

CREATE TABLE export_document (             -- current external state; id is content-INDEPENDENT (D6)
  document_id    text PRIMARY KEY,         -- sha256("segment:"||stable_key) or attachment-chunk form (§11.3)
  kind           text NOT NULL CHECK (kind IN ('segment','attachment_chunk')),
  segment_id     bigint REFERENCES segment(segment_id) ON DELETE SET NULL,
  attachment_chunk_id bigint REFERENCES attachment_chunk(chunk_id) ON DELETE SET NULL,
  gcs_uri        text,
  current_content_sha256 text,
  state          text NOT NULL
                 CHECK (state IN ('pushed','purged'))
);
CREATE INDEX export_document_segment_idx ON export_document (segment_id);

CREATE TABLE export_run_item (             -- immutable per-run audit history (D6)
  export_run_id bigint NOT NULL REFERENCES export_run(export_run_id),
  document_id   text NOT NULL,
  action        text NOT NULL CHECK (action IN ('upsert','delete')),
  content_sha256 text,
  staged_relpath text,
  result_state  text NOT NULL DEFAULT 'staged'
                CHECK (result_state IN ('staged','pushed','deleted','failed')),
  error         text,
  CHECK ((action = 'delete' AND content_sha256 IS NULL)
      OR (action = 'upsert' AND content_sha256 IS NOT NULL AND staged_relpath IS NOT NULL)),
  PRIMARY KEY (export_run_id, document_id)
);

CREATE TABLE eval_query (                  -- canonical eval store (§13.1, D6)
  query_id   text PRIMARY KEY,
  query_text text NOT NULL,
  notes      text,
  targets    text[] NOT NULL DEFAULT ARRAY['local']::text[],
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE relevance_label (
  query_id    text NOT NULL REFERENCES eval_query(query_id) ON DELETE CASCADE,
  anchor_guid text NOT NULL,
  grade       smallint NOT NULL CHECK (grade BETWEEN 0 AND 2),
  source      text NOT NULL CHECK (source IN ('mark_relevant','manual','pool_judgment')),
  added_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (query_id, anchor_guid)
);

CREATE TABLE mcp_audit (
  audit_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  ts            timestamptz NOT NULL DEFAULT now(),
  surface       text NOT NULL CHECK (surface IN ('local','public')),
  subject       text,                      -- raw OAuth sub on public; 'local' on local
  subject_ok    boolean NOT NULL,
  tool          text,
  params_sha256 text,                      -- never raw params
  result_count  int,
  latency_ms    int,
  error         text
);
CREATE INDEX mcp_audit_ts_idx ON mcp_audit (ts);
CREATE INDEX mcp_audit_reject_idx ON mcp_audit (ts) WHERE NOT subject_ok;

CREATE TABLE sync_state (                  -- e.g. 'watermark.rowid.mini' = '812345'
  key        text PRIMARY KEY,
  value      text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);
