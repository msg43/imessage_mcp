"""Coverage and overlap invariants for the isolated comparison, not production."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from runpy import run_path

from imsg.segment.models import MessageForSegmentation

simple_chunks = run_path(str(Path(__file__).parents[1] / "scripts" / "bench_segmentation.py"))[
    "simple_chunks"
]


def message(i, *, hour=0, text="a short line"):
    return MessageForSegmentation(
        i,
        str(i),
        1,
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hour, seconds=i),
        False,
        "Alice",
        text,
        False,
        False,
        False,
    )


def test_size_boundary_overlaps_without_losing_any_message():
    messages = [message(i) for i in range(9)]
    chunks = simple_chunks(messages, max_messages=4, overlap=2)
    assert [[m.message_id for m in c] for c in chunks] == [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [6, 7, 8],
    ]
    assert {m.message_id for c in chunks for m in c} == set(range(9))


def test_time_gap_never_carries_previous_conversation():
    chunks = simple_chunks([message(0), message(1, hour=5)], overlap=3)
    assert [[m.message_id for m in c] for c in chunks] == [[0], [1]]


def test_long_single_message_and_zero_overlap_make_progress():
    chunks = simple_chunks([message(0, text="a" * 100), message(1)], max_tokens=5, overlap=0)
    assert [[m.message_id for m in c] for c in chunks] == [[0], [1]]
    assert simple_chunks([]) == []
