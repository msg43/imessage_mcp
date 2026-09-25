"""Match a search against what people wrote, the way the full-text index does.

The page's segment hits come from the full-text index (`seg_fts`,
`seg_fts_tri`), which holds each segment's whole rendered text
(`imsg.segment.render`): a `Chat:` line with the participants' names and
the group's name, a `Time:` line with the date range and the time zone, a
`[HH:MM] short-name:` label on every message, reaction suffixes such as
`(♥ bob)`, the `[deleted]` label, and the fixed words around attachment
snippets (`pdf`, `full text via`, `caption`, `ocr`). A search for a
contact's name, a year, an hour, "new york" or "text" therefore matches
segments in which no message contains the words.

The page keeps a segment hit only when the words occur in the segment's
content: a message's text, the filename of a PDF or other document (the
renderer writes those names into the segment; it writes none for photos,
video or audio), extracted attachment text (caption, OCR, transcript, PDF
or document text), or, when `policy.index_edit_history` puts them in the
index, an earlier version of an edited message. This module builds the
Postgres regular expressions for that check and runs it
(`segments_matching_content`). The lasting fix, an index column that
holds message words only, needs `imsg fts rebuild` and waits until heavy
work may run again.

**Matching follows the index's own rules.**

- A plain search (the index's BM25 table) matches whole tokens, with
  case and Latin diacritics ignored (`unicode61 remove_diacritics 2`).
  Each query word is tokenized by SQLite's own tokenizer with the index's
  options, so the words and their folding are exactly the index's. Each
  token character becomes a bracket expression holding every character
  that SQLite folds to it (`Café`, `CAFE` and `cafe` all fold to `cafe`),
  and the combining marks SQLite drops inside a token may follow any
  character. The tables come from the tokenizer itself (`fold_tables`),
  not from Python's Unicode rules, which differ (Python folds `ß` to
  `ss`; SQLite keeps `ß`).
- A quoted phrase (the trigram table) matches a substring with case
  ignored and diacritics significant: bracket expressions of case
  variants, with the decomposed spelling of an accented letter allowed
  too, and any run of white space where the phrase has a space.
- An emoji search needs no check here: the page already scans message
  text for it.

**Every expression is locale-independent**: literal characters in
bracket expressions, matched case-sensitively (`~`). Postgres's own
case-insensitive matching (`~*`, `ILIKE`) folds only ASCII letters in a
database whose `LC_CTYPE` is `C`, which is how the test clusters and
possibly the production one are made.

**The check never drops a segment the index matched on content.** Token
boundaries are tested against ASCII letters and digits only, which are
token characters under every tokenizer setting, so the only difference
from the index is that a word glued to a non-ASCII letter (`écafe` for
`cafe`) can be kept. It never removes a true match.
"""

from __future__ import annotations

import itertools
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import apsw

from imsg.embed.fts.schema import PRIMARY_TOKENIZER

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import psycopg

    from imsg.retrieval.query import AnalyzedQuery

MAX_CONTENT_TERMS = 32
"""The same bound the highlighter uses (`imsg.search_page.highlight`)."""

_BEFORE = "(?<![0-9A-Za-z])"
_AFTER = "(?![0-9A-Za-z])"
"""Token boundaries as lookaround constraints: measured faster in Postgres
than `(?:^|[^0-9A-Za-z])` groups (141 against 231 ms over 88,184 messages
of the synthetic corpus, for a word that no message holds)."""
_BETWEEN_TOKENS = "[^0-9A-Za-z]+"
_WHITESPACE = "".join(
    chr(code)
    for code in (
        *range(0x09, 0x0E), 0x20, *range(0x1C, 0x20), 0x85, 0xA0, 0x1680,
        *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
    )
)
"""Every character Python's `\\s` matches, which is what `normalize_text`
collapsed before the index saw the text; raw attachment text may still
hold any of them."""
_QUOTE_VARIANTS = {"'": "\u2018\u2019\u201a\u201b", '"': "\u201c\u201d\u201e\u201f"}
"""`normalize_text` folds curly quotes to ASCII before indexing, and a
query is normalized the same way; raw attachment text keeps them."""

_PROBE_BATCH = 4096
_FIRST_PROBED = 0x21
_LAST_PROBED = 0x2FFFF


# --------------------------------------------------------------------------
# fold tables, read from SQLite's own tokenizers
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FoldTables:
    token_fold: dict[str, str]
    """Character -> the character the index's word tokenizer folds it to
    (only characters it changes)."""
    token_dropped: frozenset[str]
    """Characters the word tokenizer drops inside a token (combining
    marks under `remove_diacritics`)."""
    phrase_fold: dict[str, str]
    """Character -> the character the trigram tokenizer folds it to
    (case only)."""
    token_classes: dict[str, frozenset[str]]
    phrase_classes: dict[str, frozenset[str]]


def _tokenizer_spec() -> tuple[str, list[str]]:
    name, *args = PRIMARY_TOKENIZER.split()
    return name, args


def _may_fold(ch: str) -> bool:
    """A character that has a case mapping, a decomposition, or is a mark:
    the only characters a SQLite tokenizer can fold or drop. Checked
    against a scan of every code point by the tests."""
    return (
        ch.lower() != ch
        or ch.upper() != ch
        or bool(unicodedata.decomposition(ch))
        or unicodedata.category(ch).startswith("M")
    )


def probe_characters(first: int = _FIRST_PROBED, last: int = _LAST_PROBED, *, every: bool = False) -> list[str]:
    out: list[str] = []
    for code in range(first, last + 1):
        if 0xD800 <= code <= 0xDFFF:
            continue
        ch = chr(code)
        if every or _may_fold(ch):
            out.append(ch)
    return out


def _probe(
    tokenizer: apsw.FTS5Tokenizer, characters: Sequence[str], *, template: str
) -> list[tuple[str, int, str]]:
    """Tokenize `template` (with `{}` replaced by each character) for every
    character, in batches, and return `(character, token_length_bytes,
    token)` for the token that starts where each probe starts."""
    out: list[tuple[str, int, str]] = []
    for start in range(0, len(characters), _PROBE_BATCH):
        batch = characters[start : start + _PROBE_BATCH]
        pieces: list[bytes] = []
        offsets: list[int] = []
        position = 0
        for ch in batch:
            encoded = template.format(ch).encode("utf-8")
            offsets.append(position)
            pieces.append(encoded)
            position += len(encoded) + 1
        first_token: dict[int, tuple[int, str]] = {}
        for row in tokenizer(b" ".join(pieces), apsw.FTS5_TOKENIZE_DOCUMENT, None):
            fields: tuple[object, ...] = tuple(row)
            begin, end, token = int(str(fields[0])), int(str(fields[1])), str(fields[2])
            first_token.setdefault(begin, (end - begin, token))
        for ch, offset in zip(batch, offsets, strict=True):
            found = first_token.get(offset)
            if found is not None:
                out.append((ch, found[0], found[1]))
    return out


def build_fold_tables(characters: Sequence[str] | None = None) -> FoldTables:
    """Ask SQLite how its tokenizers fold each character. `characters`
    defaults to every character that could fold (`_may_fold`)."""
    probe = probe_characters() if characters is None else list(characters)
    name, args = _tokenizer_spec()
    conn = apsw.Connection(":memory:")
    try:
        words = conn.fts5_tokenizer(name, args)
        trigram = conn.fts5_tokenizer("trigram", [])
        token_fold: dict[str, str] = {}
        dropped: set[str] = set()
        for ch, length, token in _probe(words, probe, template="a{}b"):
            if length != len(f"a{ch}b".encode()):
                continue  # a separator: "a" and "b" became two tokens
            if token == "ab":
                dropped.add(ch)
            elif token[1:-1] != ch and len(token) == 3:
                token_fold[ch] = token[1:-1]
        phrase_fold: dict[str, str] = {}
        for ch, _length, token in _probe(trigram, probe, template="{}ab"):
            if token and token[0] != ch and unicodedata.category(ch) != "Cn":
                phrase_fold[ch] = token[0]
    finally:
        conn.close()
    return FoldTables(
        token_fold=token_fold,
        token_dropped=frozenset(dropped),
        phrase_fold=phrase_fold,
        token_classes=_invert(token_fold),
        phrase_classes=_invert(phrase_fold),
    )


def _invert(fold: dict[str, str]) -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = {}
    for source, target in fold.items():
        groups.setdefault(target, {target}).add(source)
    return {k: frozenset(v) for k, v in groups.items()}


_tables: FoldTables | None = None
_tables_lock = threading.Lock()


def fold_tables() -> FoldTables:
    """Built once per process (about 60 ms), on first use."""
    global _tables
    with _tables_lock:
        if _tables is None:
            _tables = build_fold_tables()
        return _tables


# --------------------------------------------------------------------------
# expressions
# --------------------------------------------------------------------------


def _escape(code: int) -> str:
    ch = chr(code)
    if ch.isascii() and ch.isalnum():
        return ch
    return f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"


def bracket(characters: Iterable[str]) -> str:
    """A bracket expression matching exactly `characters`, consecutive
    code points merged into ranges, every non-alphanumeric character
    written as a `\\u` escape (so no character is special inside it)."""
    codes = sorted({ord(c) for c in characters})
    parts: list[str] = []
    for _key, run in itertools.groupby(enumerate(codes), key=lambda item: item[1] - item[0]):
        span = [code for _i, code in run]
        if len(span) >= 3:
            parts.append(f"{_escape(span[0])}-{_escape(span[-1])}")
        else:
            parts.extend(_escape(code) for code in span)
    return "[" + "".join(parts) + "]"


def query_tokens(word: str) -> list[str]:
    """The index tokenizer's tokens for one query word, folded as the
    index folds them (`Bid-Rev3` -> `["bid", "rev3"]`)."""
    name, args = _tokenizer_spec()
    conn = apsw.Connection(":memory:")
    try:
        tokenizer = conn.fts5_tokenizer(name, args)
        rows = tokenizer(word.encode("utf-8"), apsw.FTS5_TOKENIZE_QUERY, None, include_offsets=False)
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


_VARIATION_SELECTORS = frozenset(chr(code) for code in range(0xFE00, 0xFE10))
"""`normalize_text` strips these before indexing; raw attachment text and
earlier versions of a message may still hold them."""


def _token_expression(token: str, tables: FoldTables) -> str:
    marks = bracket(tables.token_dropped | _VARIATION_SELECTORS) + "*"
    out: list[str] = []
    for ch in token:
        members = set(tables.token_classes.get(ch, frozenset({ch})))
        members.add(ch)
        out.append(bracket(members) + marks)
    return "".join(out)


def word_expression(word: str, tables: FoldTables | None = None) -> str | None:
    """The expression for one query word, or `None` when the word has no
    token (punctuation alone), which the index ignores too."""
    tokens = query_tokens(word)
    if not tokens:
        return None
    tables = tables or fold_tables()
    body = _BETWEEN_TOKENS.join(_token_expression(t, tables) for t in tokens)
    return f"{_BEFORE}{body}{_AFTER}"


def _phrase_class(ch: str, tables: FoldTables) -> str:
    folded = tables.phrase_fold.get(ch, ch)
    members = set(tables.phrase_classes.get(folded, frozenset({folded})))
    members.update({ch, folded})
    members.update(_QUOTE_VARIANTS.get(ch, ""))
    return bracket(members)


def phrase_expression(phrase: str, tables: FoldTables | None = None) -> str:
    """The expression for a quoted phrase: a substring, case ignored."""
    tables = tables or fold_tables()
    out: list[str] = []
    previous_space = False
    for ch in phrase:
        if ch.isspace():
            if not previous_space:
                out.append(bracket(_WHITESPACE) + "+")
            previous_space = True
            continue
        previous_space = False
        whole = _phrase_class(ch, tables)
        decomposed = unicodedata.normalize("NFD", ch)
        if decomposed != ch:
            parts = "".join(_phrase_class(c, tables) for c in decomposed)
            out.append(f"(?:{whole}|{parts})")
        else:
            out.append(whole)
    return "".join(out)


@dataclass(frozen=True, slots=True)
class ContentQuery:
    """One Postgres regular expression per query term. A segment matches
    when every term matches at least one of its texts."""

    patterns: tuple[str, ...]

    @property
    def matches_nothing(self) -> bool:
        return not self.patterns


def content_query(analyzed: AnalyzedQuery) -> ContentQuery | None:
    """The expressions for `analyzed`, or `None` for an emoji search
    (already a scan of message text)."""
    if analyzed.mode == "bm25":
        patterns: list[str] = []
        seen: set[str] = set()
        for word in analyzed.phrase.split():
            expression = word_expression(word)
            if expression is None or expression in seen:
                continue
            seen.add(expression)
            patterns.append(expression)
            if len(patterns) >= MAX_CONTENT_TERMS:
                break
        return ContentQuery(tuple(patterns))
    if analyzed.mode == "trigram":
        return ContentQuery((phrase_expression(analyzed.phrase),))
    return None


def text_condition(query: ContentQuery, column: str, *, prefix: str) -> tuple[str, dict[str, object]]:
    """`column` matches every term: for one text on its own (a message
    that is in no segment yet)."""
    if query.matches_nothing:
        return "FALSE", {}
    clauses: list[str] = []
    params: dict[str, object] = {}
    for i, pattern in enumerate(query.patterns):
        clauses.append(f"{column} ~ %({prefix}{i})s")
        params[f"{prefix}{i}"] = pattern
    return " AND ".join(clauses), params


# --------------------------------------------------------------------------
# the segment check
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContentMatch:
    matched_attachments: frozenset[tuple[int, int]]
    """`(message_id, attachment_id)` whose filename or extracted text
    matched a term."""
    last_match: datetime
    """When the latest message whose content matched a term was sent."""


def _visibility(
    *,
    index_unsent: bool,
    sent_after: datetime | None,
    sent_before: datetime | None,
    message_filter: tuple[str, Mapping[str, object]] | None = None,
) -> tuple[str, dict[str, object]]:
    clauses = ["(%(cm_unsent)s OR NOT m.is_unsent)"]
    params: dict[str, object] = {"cm_unsent": index_unsent}
    if message_filter is not None:
        clauses.append(f"({message_filter[0]})")
        params.update(message_filter[1])
    if sent_after is not None:
        clauses.append("m.sent_at >= %(cm_after)s")
        params["cm_after"] = sent_after
    if sent_before is not None:
        clauses.append("m.sent_at < %(cm_before)s")
        params["cm_before"] = sent_before
    return " AND ".join(clauses), params


def _term_tests(query: ContentQuery) -> tuple[list[str], str, dict[str, object]]:
    tests: list[str] = []
    params: dict[str, object] = {}
    for i, pattern in enumerate(query.patterns):
        params[f"cm_p{i}"] = pattern
        tests.append(f"(body ~ %(cm_p{i})s) AS t{i}")
    any_term = " OR ".join(f"t{i}" for i in range(len(query.patterns)))
    return tests, any_term, params


def _scan(
    pg: psycopg.Connection,
    content_sql: str,
    query: ContentQuery,
    params: dict[str, object],
) -> list[tuple[int, list[bool], datetime, frozenset[tuple[int, int]]]]:
    """Run one content scan: per segment, which terms matched, the latest
    matching message's time and the matching attachments. Only segments
    where some term matched come back."""
    tests, any_term, term_params = _term_tests(query)
    n = len(query.patterns)
    sql = f"""
        WITH content (segment_id, message_id, sent_at, attachment_id, body) AS ({content_sql}),
        tested AS (
            SELECT segment_id, message_id, sent_at, attachment_id, {", ".join(tests)}
            FROM content WHERE body IS NOT NULL
        )
        SELECT segment_id, max(sent_at) FILTER (WHERE {any_term}),
               array_agg(ARRAY[message_id, attachment_id])
                   FILTER (WHERE attachment_id IS NOT NULL AND ({any_term})),
               {", ".join(f"bool_or(t{i})" for i in range(n))}
        FROM tested
        GROUP BY segment_id
        HAVING {" OR ".join(f"bool_or(t{i})" for i in range(n))}
    """
    with pg.cursor() as cur:
        # Never a prepared statement: after five runs psycopg prepares it and
        # Postgres may switch to a generic plan, which for these lookups by
        # an array of ids was measured at 460 ms against 270 ms.
        cur.execute(sql, {**params, **term_params}, prepare=False)
        rows = cur.fetchall()
    out: list[tuple[int, list[bool], datetime, frozenset[tuple[int, int]]]] = []
    for row in rows:
        pairs = frozenset((int(p[0]), int(p[1])) for p in (row[2] or []))
        out.append((int(row[0]), [bool(v) for v in row[3:]], row[1], pairs))
    return out


RENDERED_FILENAME = (
    "(a.mime_type IS NULL OR NOT (a.mime_type LIKE 'image/%%' OR a.mime_type LIKE 'audio/%%' "
    "OR a.mime_type LIKE 'video/%%'))"
)
"""The attachments whose filename the renderer writes into a segment
(`imsg.segment.pipeline._classify_attachment_kind`: a PDF or any other
kind that is not an image, audio or video). `%%` because the clause runs
with bound parameters."""


def segments_matching_content(
    pg: psycopg.Connection,
    segment_ids: Sequence[int],
    query: ContentQuery,
    *,
    index_unsent: bool,
    include_edit_history: bool,
    sent_after: datetime | None = None,
    sent_before: datetime | None = None,
    message_filter: tuple[str, Mapping[str, object]] | None = None,
) -> dict[int, ContentMatch]:
    """The segments among `segment_ids` whose content matches every term
    of `query`. With `sent_after` / `sent_before`, only messages sent in
    that range count (the page's date filter). `message_filter` is any
    further condition over the `message m` row (the page's "Sent by"
    filter), as SQL with its bound parameters.

    Two passes, because the first settles most segments cheaply:

    1. message text;
    2. for the segments still missing a term: rendered filenames, earlier
       versions under `include_edit_history`, and, when the query has more
       than one term, extracted attachment text. A one-term match in
       extracted text needs no check here: the page's attachment channel
       finds it through the index's `att_fts` table, which holds all of
       it. Several terms can be spread over a message and an attachment,
       which only this pass can join.

    A segment settled by message text is not scanned for matching
    attachments; measured on the synthetic full-size corpus, the passes
    take about 150 and 65 ms for a name found in 16,362 segment headers
    and in no message."""
    if not segment_ids or query.matches_nothing:
        return {}
    visible, params = _visibility(
        index_unsent=index_unsent,
        sent_after=sent_after,
        sent_before=sent_before,
        message_filter=message_filter,
    )
    n = len(query.patterns)
    terms: dict[int, list[bool]] = {}
    latest: dict[int, datetime] = {}
    attachments: dict[int, frozenset[tuple[int, int]]] = {}

    def merge(rows: list[tuple[int, list[bool], datetime, frozenset[tuple[int, int]]]]) -> None:
        for segment_id, bits, last, pairs in rows:
            have = terms.setdefault(segment_id, [False] * n)
            terms[segment_id] = [a or b for a, b in zip(have, bits, strict=True)]
            if last is not None and (segment_id not in latest or last > latest[segment_id]):
                latest[segment_id] = last
            if pairs:
                attachments[segment_id] = attachments.get(segment_id, frozenset()) | pairs

    merge(
        _scan(
            pg,
            f"""SELECT sm.segment_id, m.message_id, m.sent_at, NULL::bigint,
                       coalesce(m.text_normalized, m.text_original)
                FROM segment_message sm JOIN message m ON m.message_id = sm.message_id
                WHERE sm.segment_id = ANY(%(cm_ids)s::bigint[]) AND {visible}""",
            query,
            {**params, "cm_ids": list(segment_ids)},
        )
    )
    unsettled = [sid for sid in segment_ids if not all(terms.get(sid, [False] * n))]
    if unsettled:
        branches = [
            f"""SELECT sm.segment_id, m.message_id, m.sent_at, ma.attachment_id, a.filename
                FROM segment_message sm JOIN message m ON m.message_id = sm.message_id
                JOIN message_attachment ma ON ma.message_id = m.message_id
                JOIN attachment a ON a.attachment_id = ma.attachment_id
                WHERE sm.segment_id = ANY(%(cm_ids)s::bigint[]) AND m.has_attachments
                  AND {visible} AND a.filename IS NOT NULL AND {RENDERED_FILENAME}"""
        ]
        if n > 1:
            branches.append(
                f"""SELECT sm.segment_id, m.message_id, m.sent_at, ma.attachment_id, e.text
                    FROM segment_message sm JOIN message m ON m.message_id = sm.message_id
                    JOIN message_attachment ma ON ma.message_id = m.message_id
                    JOIN enrichment e ON e.attachment_id = ma.attachment_id AND e.state = 'done'
                    WHERE sm.segment_id = ANY(%(cm_ids)s::bigint[]) AND m.has_attachments
                      AND {visible}"""
            )
        if include_edit_history:
            branches.append(
                f"""SELECT sm.segment_id, m.message_id, m.sent_at, NULL::bigint, mv.text
                    FROM segment_message sm JOIN message m ON m.message_id = sm.message_id
                    JOIN message_version mv ON mv.message_id = m.message_id
                    WHERE sm.segment_id = ANY(%(cm_ids)s::bigint[]) AND {visible}"""
            )
        merge(_scan(pg, " UNION ALL ".join(branches), query, {**params, "cm_ids": unsettled}))
    return {
        segment_id: ContentMatch(
            matched_attachments=attachments.get(segment_id, frozenset()),
            last_match=latest[segment_id],
        )
        for segment_id, bits in terms.items()
        if all(bits)
    }


__all__ = [
    "MAX_CONTENT_TERMS",
    "ContentMatch",
    "ContentQuery",
    "FoldTables",
    "bracket",
    "build_fold_tables",
    "content_query",
    "fold_tables",
    "phrase_expression",
    "probe_characters",
    "query_tokens",
    "segments_matching_content",
    "text_condition",
    "word_expression",
]
