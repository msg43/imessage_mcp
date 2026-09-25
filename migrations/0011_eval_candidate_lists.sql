-- 0011_eval_candidate_lists.sql
--
-- Candidate lists from the search page's grading mode (D14), for the
-- reranker evaluation plan of 2026-09-24: grade every candidate a search
-- produced, in random order and without scores, and keep the whole list so
-- any reranker can be scored offline later.
--
-- `eval_candidate_list` is one graded search: the eval query it belongs to,
-- the text and filters as the owner set them, and how the list was ranked
-- (`ranking`: the reciprocal-rank-fusion constant, whether the meaning
-- search ran, the similarity floors, the channel counts). `shuffle_seed`
-- reproduces the random order the candidates were shown in.
--
-- `eval_candidate` is one segment of that list, at its position in
-- reciprocal-rank-fusion order (1 to 30). It is identified the way the eval
-- harness anchors labels (SPEC §13.1): the GUID of the segment's first
-- message, which survives re-segmentation. `segment_key` and
-- `segment_text` record the segment as it was when the list was made, so a
-- reranker scored later reads exactly the text the owner graded.
-- `shown_order` is the candidate's place in the grading view's random
-- order.
--
-- The grades themselves are ordinary `relevance_label` rows (source
-- 'pool_judgment'), keyed by the list's query and each candidate's anchor,
-- so `imsg eval run` and the AT-4 check read them with no conversion.
--
-- Additive only: two new tables. NO explicit BEGIN/COMMIT, matching
-- 0001-0010: the runner wraps each file in its own transaction.

CREATE TABLE IF NOT EXISTS eval_candidate_list (
  list_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  query_id     text NOT NULL REFERENCES eval_query(query_id) ON DELETE CASCADE,
  query_text   text NOT NULL,
  filters      jsonb NOT NULL DEFAULT '{}'::jsonb,
  ranking      jsonb NOT NULL DEFAULT '{}'::jsonb,
  shuffle_seed bigint NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS eval_candidate_list_query_idx
  ON eval_candidate_list (query_id, created_at);

CREATE TABLE IF NOT EXISTS eval_candidate (
  list_id       bigint NOT NULL REFERENCES eval_candidate_list(list_id) ON DELETE CASCADE,
  fused_rank    smallint NOT NULL CHECK (fused_rank BETWEEN 1 AND 30),
  anchor_guid   text NOT NULL,
  segment_key   text NOT NULL,
  fused_score   double precision NOT NULL,
  channel_ranks jsonb NOT NULL,
  segment_text  text NOT NULL,
  shown_order   smallint NOT NULL CHECK (shown_order BETWEEN 1 AND 30),
  PRIMARY KEY (list_id, fused_rank),
  UNIQUE (list_id, anchor_guid),
  UNIQUE (list_id, shown_order)
);

COMMENT ON TABLE eval_candidate_list IS
  'One search graded in the search page''s grading mode: query, filters, how it was ranked, '
  'and the seed of the random order its candidates were shown in.';

COMMENT ON TABLE eval_candidate IS
  'One candidate of a graded search, at its reciprocal-rank-fusion position (1-30), anchored '
  'on its segment''s first message GUID; grades live in relevance_label.';
