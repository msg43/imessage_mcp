"""Attachment-chunk splitting (SPEC §8 S5b: ~1500 tokens, 150 overlap)."""

from __future__ import annotations

from imsg.enrich.chunking import chunk_text
from imsg.tokens import estimate_tokens


def test_empty_text_produces_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text("just one short paragraph of text")
    assert len(chunks) == 1
    assert chunks[0] == "just one short paragraph of text"


def test_long_paragraph_text_splits_into_multiple_chunks() -> None:
    paragraphs = [f"Paragraph number {i} with some filler words to add bulk." for i in range(200)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=20)
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c) <= 200 + 20 + 5  # target + overlap + slack for one extra unit


def test_every_paragraph_survives_in_some_chunk() -> None:
    paragraphs = [f"Unique paragraph marker {i}." for i in range(50)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, target_tokens=50, overlap_tokens=5)
    joined = " ".join(chunks)
    for i in range(50):
        assert f"Unique paragraph marker {i}." in joined


def test_consecutive_chunks_overlap() -> None:
    paragraphs = [f"Paragraph {i} has distinctive content xyz{i}." for i in range(30)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, target_tokens=60, overlap_tokens=20)
    assert len(chunks) >= 2
    # At least one paragraph from the end of chunk N should also appear at
    # the start of chunk N+1 (the overlap window).
    for i in range(len(chunks) - 1):
        tail_of_current = chunks[i].split("\n\n")[-1]
        assert tail_of_current in chunks[i + 1]


def test_paragraph_less_text_falls_back_to_sentence_splitting() -> None:
    # A transcript-shaped blob: no blank-line paragraph breaks at all.
    sentences = [f"This is sentence number {i}." for i in range(100)]
    text = " ".join(sentences)
    chunks = chunk_text(text, target_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    joined = " ".join(chunks)
    for i in range(100):
        assert f"This is sentence number {i}." in joined


def test_single_oversized_unit_is_hard_sliced() -> None:
    huge_sentence = "word " * 5000  # no sentence/paragraph breaks at all
    chunks = chunk_text(huge_sentence, target_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c) <= 100 + 5  # hard-slice budget, small float slack
