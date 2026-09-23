-- 0009_enrichment_doc_text_and_claim_order.sql
--
-- Two additions for filling and draining the S5b enrichment queue. Neither
-- changes an existing row.
--
-- (1) `enrichment_kind` gains 'doc_text'.
--
--     Text extracted from a text-bearing file that is not a PDF: contact
--     cards (vCard, including Apple's shared-location cards), plain text,
--     Markdown, CSV, calendar files, HTML and SVG, Office and RTF documents
--     (through macOS `textutil`), and OOXML spreadsheets and slides. Before
--     this, the router had no route for any of them, so their text was never
--     searchable. `pdf_text` keeps meaning exactly what it says; reusing it
--     for a contact card would make `imsg enrich --kinds pdf_text` and every
--     per-kind coverage count lie.
--
--     Postgres 12+ permits ADD VALUE inside a transaction; the new value
--     cannot be USED until this transaction commits, and nothing below
--     references it (same note as 0003).
--
-- (2) An index that serves the claim query one kind at a time.
--
--     Workers now claim cheap kinds before captions, and a named list of
--     kinds in the order given (`imsg enrich --kinds`), newest attachment
--     first within a kind. That is one query per kind of the shape
--     `WHERE kind = $1 AND state IN ('pending', 'running') ... ORDER BY
--     attachment_id DESC LIMIT n`, which this index answers without a sort.
--     The partial predicate keeps finished rows out of it: once a kind's
--     backlog is mostly `done`, a claim still starts at a pending row rather
--     than walking past every finished one.
--
-- NO explicit BEGIN/COMMIT, matching 0001-0008: the runner wraps each file
-- in its own transaction.

ALTER TYPE enrichment_kind ADD VALUE IF NOT EXISTS 'doc_text';

CREATE INDEX IF NOT EXISTS enrichment_claim_idx
  ON enrichment (kind, attachment_id DESC)
  WHERE state IN ('pending', 'running');
