"""The content check's expressions match exactly when the index does.

`imsg.search_page.content_match` turns each query word into a Postgres
regular expression built from SQLite's own tokenizers. These tests
compare every expression with an in-memory FTS5 table that uses the
sidecar's tokenizer settings, on text normalized as the sidecar's is.
The page-level behaviour is in `test_search_page_header_matches.py`.

Synthetic text only. Needs the scratch Postgres for the regular
expressions (skips without it)."""

from __future__ import annotations

from collections.abc import Iterator

import apsw
import psycopg
import pytest

from _search_page_fixtures import REACHABLE, SKIP_REASON, create_scratch_db, drop_scratch_db
from imsg.embed.fts.schema import PRIMARY_TOKENIZER
from imsg.retrieval.query import analyze_query, bm25_match_expression, trigram_match_expression
from imsg.search_page.content_match import (
    build_fold_tables,
    content_query,
    fold_tables,
    phrase_expression,
    probe_characters,
    word_expression,
)
from imsg.textnorm import normalize_text

pytestmark = pytest.mark.skipif(not REACHABLE, reason=SKIP_REASON)

DB_NAME = "imsg_sp_content_match_test"


@pytest.fixture(scope="module")
def pg_module() -> Iterator[psycopg.Connection]:
    conn = create_scratch_db(DB_NAME)
    try:
        yield conn
    finally:
        conn.close()
        drop_scratch_db(DB_NAME)


# --------------------------------------------------------------------------
# the expressions agree with SQLite's tokenizers
# --------------------------------------------------------------------------


def test_fold_tables_equal_a_scan_of_every_code_point() -> None:
    """The fast build probes only characters Python says can fold; a scan
    of every code point finds exactly the same folds and dropped marks."""
    fast = build_fold_tables()
    full = build_fold_tables(probe_characters(0x21, 0x10FFFF, every=True))
    assert fast.token_fold == full.token_fold
    assert fast.token_dropped == full.token_dropped
    assert fast.phrase_fold == full.phrase_fold
    assert fast.token_fold["Á"] == "a" and fast.token_fold["G"] == "g"
    assert "ß" not in fast.token_fold  # SQLite keeps ß; Python would fold it to "ss"


def _index_matches(query: str, text: str, *, trigram: bool) -> bool:
    """What the index itself answers: an in-memory FTS5 table with the
    sidecar's tokenizer, fed the text as the sidecar is (normalized)."""
    conn = apsw.Connection(":memory:")
    try:
        tokenizer = "trigram" if trigram else PRIMARY_TOKENIZER
        conn.execute(f"CREATE VIRTUAL TABLE t USING fts5(x, tokenize = '{tokenizer}')")
        conn.execute("INSERT INTO t VALUES (?)", (normalize_text(text),))
        analyzed = analyze_query(query)
        match = (
            trigram_match_expression(analyzed.phrase)
            if trigram
            else bm25_match_expression(analyzed.phrase)
        )
        return bool(conn.execute("SELECT count(*) FROM t WHERE t MATCH ?", (match,)).fetchone()[0])
    finally:
        conn.close()


def _pg_matches(pg: psycopg.Connection, pattern: str, text: str) -> bool:
    row = pg.execute("SELECT %s::text ~ %s::text", (text, pattern)).fetchone()
    assert row is not None
    return bool(row[0])


WORD_CASES = [
    ("cafe", "Meet at the Caf\u00e9 at noon"),
    ("caf\u00e9", "CAFE opens at nine"),
    ("naive", "a na\u00efve plan"),
    ("strasse", "Hauptstra\u00dfe 5"),
    ("stra\u00dfe", "HAUPTSTRASSE"),
    ("\u00df", "GRO\u1e9e"),
    ("istanbul", "\u0130stanbul trip"),
    ("bar", "foo_bar"),
    ("foo_bar", "foo bar"),
    ("800", "paid $4,800.00"),
    ("4800", "paid $4,800"),
    ("rev3", "see bid-rev3.pdf"),
    ("bid-rev3", "see Bid Rev3 attached"),
    ("pergola", "pergolas are nice"),
    ("\u043c\u043e\u0441\u043a\u0432\u0430", "\u041c\u041e\u0421\u041a\u0412\u0410 \u0437\u0438\u043c\u043e\u0439"),
    ("abc", "\uff21\uff22\uff23"),
    ("cafe", "cafe\u0301 later"),
    ("dana", "Dana's truck"),
    ("u", "\u01d6"),
    ("\u03c9\u03bc\u03b5\u03b3\u03b1", "\u03a9\u039c\u0388\u0393\u0391"),
    ("2024", "the 2024 season"),
    ("co", "co-op"),
    ("op", "co-op"),
    ("co-op", "co op"),
]


@pytest.mark.parametrize(("word", "text"), WORD_CASES)
def test_word_expression_matches_exactly_when_the_index_does(
    pg_module: psycopg.Connection, word: str, text: str
) -> None:
    expression = word_expression(word)
    assert expression is not None
    assert _pg_matches(pg_module, expression, text) == _index_matches(word, text, trigram=False)


def test_a_word_glued_to_a_non_ascii_letter_is_the_one_known_difference(
    pg_module: psycopg.Connection,
) -> None:
    """Boundaries are tested against ASCII letters and digits only, so the
    check may keep more than the index, never less."""
    expression = word_expression("cafe")
    assert expression is not None
    assert not _index_matches("cafe", "\u00e9cafe", trigram=False)
    assert _pg_matches(pg_module, expression, "écafe")


PHRASE_CASES = [
    ('"deck stain"', "The DECK  stain arrived"),
    ('"deck stain"', "deck\nstain"),
    ('"caf\u00e9 au"', "CAF\u00c9 AU LAIT"),
    ('"cafe au"', "caf\u00e9 au lait"),
    ('"caf\u00e9 au"', "cafe\u0301 au lait"),
    ('"it\'s late"', "it\u2019s late now"),
    ('"id-rev"', "see bid-rev3.pdf"),
    ('"stra\u00dfe"', "STRASSE"),
]


@pytest.mark.parametrize(("query", "text"), PHRASE_CASES)
def test_phrase_expression_matches_exactly_when_the_index_does(
    pg_module: psycopg.Connection, query: str, text: str
) -> None:
    analyzed = analyze_query(query)
    assert analyzed.mode == "trigram"
    expression = phrase_expression(analyzed.phrase)
    assert _pg_matches(pg_module, expression, text) == _index_matches(query, text, trigram=True)


def test_words_without_tokens_are_ignored_as_the_index_ignores_them() -> None:
    query = content_query(analyze_query("fence - post"))
    assert query is not None and len(query.patterns) == 2
    assert content_query(analyze_query("\U0001f389")) is None  # emoji: scanned directly
    assert fold_tables() is fold_tables()
