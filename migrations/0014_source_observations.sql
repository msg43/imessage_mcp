-- A named source can observe a message or attachment GUID at more than one SQLite
-- ROWID, and a ROWID can later identify another GUID. Keep each observed
-- association rather than dropping history or refreshing the wrong message.
-- Existing rows and their last-observed extraction runs remain unchanged.
ALTER TABLE message_source
    DROP CONSTRAINT message_source_pkey,
    DROP CONSTRAINT message_source_message_id_source_name_key,
    ADD PRIMARY KEY (message_id, source_name, source_rowid);

CREATE INDEX message_source_source_rowid_idx ON message_source (source_name, source_rowid);

ALTER TABLE attachment_source
    DROP CONSTRAINT attachment_source_pkey,
    DROP CONSTRAINT attachment_source_attachment_id_source_name_key,
    ADD PRIMARY KEY (attachment_id, source_name, source_rowid);

CREATE INDEX attachment_source_source_rowid_idx ON attachment_source (source_name, source_rowid);
