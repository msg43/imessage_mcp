-- 0002_multimodal_vectors.sql
-- Transcribed verbatim from docs/SPEC.md §7.4 (v1.1). Secondary
-- multimodal vector (D3a ratified) — model pinned PE-Core-G14-448,
-- dim 1280. Migrations are immutable once merged (SPEC §7.1); a model
-- swap is a new migration, not an edit to this one.

CREATE TABLE attachment_mm_embedding (     -- secondary multimodal vector (§9.5, D3a)
  attachment_id bigint PRIMARY KEY REFERENCES attachment(attachment_id) ON DELETE CASCADE,
  model         text NOT NULL,             -- pinned model id actually used, e.g. 'facebook/PE-Core-G14-448@<rev>'
  dim           int NOT NULL DEFAULT 1280 CHECK (dim = 1280),
  media_sha256  text NOT NULL,             -- idempotency: one embed per media hash
  vec           halfvec(1280) NOT NULL,    -- PE-Core-G14-448 embedding dim
  embedded_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX attachment_mm_embedding_hnsw ON attachment_mm_embedding
  USING hnsw (vec halfvec_cosine_ops) WITH (m = 16, ef_construction = 200);
