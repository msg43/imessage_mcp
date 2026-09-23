-- 0007_unlinked_messages.sql
--
-- File every message chat.db holds, including the ones with no chat link.
--
-- Owner decision D13 (2026-09-23): recall over purity. chat.db keeps some
-- messages with no `chat_message_join` row: Apple's "Recently Deleted" rows
-- (linked only through `chat_recoverable_message_join`, with a delete date),
-- rows whose group chat no longer exists, and rows with no chat evidence at
-- all. Extraction used to skip every one of them. It now files each one into
-- the chat its evidence names, or into a holding chat, and records how it
-- chose (`imsg.stages.unlinked_filing`).
--
-- `chat.unfiled_key` is set only on a holding chat, which extraction creates
-- for messages no real chat can be named for: one per lost group id
-- ('lost-group:<id>'), one per sender ('sender:<handle>'), one for the owner's
-- own sent messages ('owner'), and one for incoming rows with no sender
-- ('unknown-sender'). It is NULL on every real chat, and unique, so each key
-- names one chat. Export and the public surface's allowlist scope deny these
-- chats outright (`imsg.export.eligibility`).
--
-- `message.chat_evidence` says how the message's chat was chosen, strongest
-- first:
--   chat_message_join   chat.db links the message to the chat. Every message
--                       filed before this migration.
--   recoverable_join    Apple's "Recently Deleted" names the chat.
--   ck_1to1             a 1:1-style `ck_chat_id` whose handle is the sender,
--                       or the owner sent it.
--   ck_group_match      a group-style `ck_chat_id` that exactly one indexed
--                       group carries.
--   holding_lost_group  a group-style `ck_chat_id` no single indexed group
--                       carries.
--   holding_sender      no chat evidence at all.
-- A message moves only toward stronger evidence, and never out of a real chat.
--
-- `message.deleted_at` is the delete date from `chat_recoverable_message_join`.
-- Deleted messages stay searchable and render as "[deleted]"; export and the
-- allowlist scope leave them out, as they leave out unsent messages. It is
-- never cleared: a message deleted on one device stays labelled.
--
-- `chat_group_id` records the group ids (`chat.group_id`,
-- `chat.original_group_id`) of every chat every source has shown, so a
-- lost-group message is matched against all indexed chats, not only the ones
-- in the snapshot being read. Insert-only, like the join tables.
--
-- Additive only: two nullable columns, one column with a constant default
-- (Postgres 11+ adds it without rewriting the table), one new table. The
-- default records every existing message as 'chat_message_join', which is how
-- each one was filed. ADD COLUMN fires no row trigger, so 0003's `updated_at`
-- trigger does not move and this migration re-segments nothing.
--
-- NO explicit BEGIN/COMMIT, matching 0001-0006: the runner wraps each file in
-- its own transaction.

ALTER TABLE chat
  ADD COLUMN IF NOT EXISTS unfiled_key text
    CONSTRAINT chat_unfiled_key_key UNIQUE;

ALTER TABLE message
  ADD COLUMN IF NOT EXISTS chat_evidence text NOT NULL DEFAULT 'chat_message_join'
    CONSTRAINT message_chat_evidence_check CHECK (chat_evidence IN (
      'chat_message_join', 'recoverable_join', 'ck_1to1', 'ck_group_match',
      'holding_lost_group', 'holding_sender'
    )),
  ADD COLUMN IF NOT EXISTS deleted_at timestamptz;

-- Every extraction run reads the messages filed on weaker evidence than a
-- chat link (the only ones a run may refile). They are a few thousand of
-- ~700,000 rows, so the lookup gets a partial index; its predicate is the
-- query's own WHERE clause, word for word.
CREATE INDEX IF NOT EXISTS message_weak_evidence_idx
  ON message (source_guid) WHERE chat_evidence <> 'chat_message_join';

CREATE TABLE IF NOT EXISTS chat_group_id (
  group_id text NOT NULL,
  chat_id  bigint NOT NULL REFERENCES chat(chat_id),
  PRIMARY KEY (group_id, chat_id)
);

COMMENT ON COLUMN chat.unfiled_key IS
  'Set only on a holding chat extraction created for messages no real chat '
  'can be named for (lost-group:<id>, sender:<handle>, owner, unknown-sender). '
  'NULL on real chats. Export and allowlist scope deny holding chats (D13).';

COMMENT ON COLUMN message.chat_evidence IS
  'How the chat was chosen: chat_message_join, recoverable_join, ck_1to1, '
  'ck_group_match, holding_lost_group or holding_sender (D13).';

COMMENT ON COLUMN message.deleted_at IS
  'Delete date from chat_recoverable_message_join (Apple''s Recently Deleted). '
  'Searchable and labelled [deleted]; never exported (D13).';

COMMENT ON TABLE chat_group_id IS
  'Group ids (chat.group_id, chat.original_group_id) each source''s chat rows '
  'carry; matches a lost group''s ck_chat_id to an indexed chat (D13).';
