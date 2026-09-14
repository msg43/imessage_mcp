"""`imsg.segment.mlx_boundaries.MlxBoundaryProvider` — the local-LLM
boundary detector against the `_mlx_fakes` runtime: prompt rendering,
JSON parsing (fences, bare lists, think blocks), validation, greedy
decoding, the between-token timeout, and every failure path mapping to
`BoundaryDetectionError`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

import imsg.segment.mlx_boundaries as mlx_boundaries
from _mlx_fakes import FakeRuntime, FakeTokenizer, uninstall_runtime
from imsg.errors import BoundaryDetectionError, SegmentationError
from imsg.mlx_runtime import MlxRuntimeError, MlxRuntimeUnavailableError
from imsg.segment.boundaries import BoundaryProvider
from imsg.segment.mlx_boundaries import (
    MlxBoundaryProvider,
    parse_boundary_response,
    render_boundary_prompt,
    render_window_messages,
    validate_boundaries,
)
from imsg.segment.models import MessageForSegmentation

_BASE = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
TEMPLATE = 'Split the conversation into topics.\nReturn JSON {"boundaries": [...]}.\n'


def _message(i: int, text: str | None = None, **flags: bool) -> MessageForSegmentation:
    return MessageForSegmentation(
        message_id=i,
        source_guid=f"guid-{i}",
        chat_id=1,
        sent_at=_BASE + timedelta(minutes=i),
        is_from_me=(i % 2 == 0),
        sender_short_name="owner" if i % 2 == 0 else "alice",
        text=f"message {i}" if text is None else text,
        is_unsent=flags.get("is_unsent", False),
        is_edited=False,
        has_attachments=flags.get("has_attachments", False),
    )


def _window(n: int) -> list[MessageForSegmentation]:
    return [_message(i) for i in range(n)]


def _provider(**overrides: object) -> MlxBoundaryProvider:
    kwargs: dict[str, object] = {"prompt_template": TEMPLATE}
    kwargs.update(overrides)
    return MlxBoundaryProvider("org/boundary", "rev3", **kwargs)  # type: ignore[arg-type]


# --- identity / construction ---------------------------------------------


def test_model_id_format_and_protocol() -> None:
    provider = _provider()
    assert provider.model_id == "org/boundary@rev3"
    assert MlxBoundaryProvider("org/boundary", None, TEMPLATE).model_id == "org/boundary@main"
    assert isinstance(provider, BoundaryProvider)


def test_template_accepts_bytes_and_rejects_empty_or_invalid() -> None:
    assert _provider(prompt_template=TEMPLATE.encode()).prompt_template == TEMPLATE
    with pytest.raises(SegmentationError):
        _provider(prompt_template="   \n")
    with pytest.raises(SegmentationError):
        _provider(prompt_template=b"\xff\xfe")


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"timeout_seconds": 0.0}])
def test_constructor_rejects_invalid_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _provider(**kwargs)


def test_constructor_rejects_empty_repo() -> None:
    with pytest.raises(ValueError):
        MlxBoundaryProvider("", None, TEMPLATE)


# --- prompt rendering -----------------------------------------------------


def test_render_window_messages_numbers_from_zero_with_sender_time_and_text() -> None:
    window = [
        _message(0, "hello\nthere   friend"),
        _message(1, "", has_attachments=True),
        _message(2, None, is_unsent=True),
        _message(3, "photo", has_attachments=True),
        _message(4, ""),
    ]
    window[2] = _message(2, "", is_unsent=True)
    assert render_window_messages(window).splitlines() == [
        "[0] owner (2024-01-01T12:00+00:00): hello there friend",
        "[1] alice (2024-01-01T12:01+00:00): [attachment]",
        "[2] owner (2024-01-01T12:02+00:00): [unsent message]",
        "[3] alice (2024-01-01T12:03+00:00): photo [attachment]",
        "[4] owner (2024-01-01T12:04+00:00): [empty message]",
    ]


def test_render_boundary_prompt_appends_numbered_messages_to_template() -> None:
    window = _window(3)
    prompt = render_boundary_prompt(TEMPLATE, window)
    assert prompt.startswith(TEMPLATE.rstrip() + "\n\nMessages (3 total, indexed 0 to 2):\n")
    assert prompt.endswith(render_window_messages(window) + "\n")
    assert "[2] owner (2024-01-01T12:02+00:00): message 2" in prompt


# --- response parsing -----------------------------------------------------


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"boundaries": [3, 7]}', [3, 7]),
        ("[3, 7]", [3, 7]),
        ('```json\n{"boundaries": [3]}\n```', [3]),
        ("```\n[4, 5]\n```", [4, 5]),
        ('<think>\nmaybe {"boundaries": [1]}?\n</think>\n{"boundaries": [2]}', [2]),
        ('reasoning without an opening tag</think>\n{"boundaries": [6]}', [6]),
        ('Sure, here it is: {"boundaries": [4]} — done', [4]),
        ('{"boundaries": []}', []),
        ('  \n{"boundaries": [9, 2]}\n  ', [9, 2]),  # raw order preserved; validation sorts
    ],
)
def test_parse_boundary_response_accepts_the_tolerated_shapes(
    response: str, expected: list[int]
) -> None:
    assert parse_boundary_response(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
        "no json here",
        "42",
        '{"other": [1]}',
        '{"boundaries": "3"}',
        '{"boundaries": [1, "2"]}',
        '{"boundaries": [true]}',
        '{"boundaries": [1.5]}',
        '{"boundaries": [1,',
        "<think>still thinking",
        "```json\n```",
    ],
)
def test_parse_boundary_response_rejects_malformed_output(response: str) -> None:
    with pytest.raises(BoundaryDetectionError):
        parse_boundary_response(response)


def test_validate_boundaries_sorts_dedupes_and_enforces_range() -> None:
    assert validate_boundaries([7, 3, 3], 10) == [3, 7]
    assert validate_boundaries([], 10) == []
    assert validate_boundaries([9], 10) == [9]
    for bad in ([0], [10], [-1], [3, 11]):
        with pytest.raises(BoundaryDetectionError):
            validate_boundaries(bad, 10)


# --- end-to-end against the fake runtime ---------------------------------


def test_detect_boundaries_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(stream_chunks=['{"bound', 'aries": [5, 2]}']).install(monkeypatch)
    provider = _provider(max_tokens=64)
    window = _window(8)

    assert provider.detect_boundaries(window) == [2, 5]

    # Greedy decoding, the configured token cap, and the sampler handed to generation.
    assert runtime.sampler_calls == [{"temp": 0.0}]
    [call] = runtime.stream_calls
    assert call["max_tokens"] == 64
    assert call["sampler"] == ("sampler", 0.0)
    assert call["model"] is runtime.model
    # The prompt went through the chat template with thinking disabled ...
    [chat] = runtime.tokenizer.chat_calls
    assert chat["add_generation_prompt"] is True
    assert chat["tokenize"] is False
    assert chat["enable_thinking"] is False
    assert chat["messages"] == [
        {"role": "user", "content": render_boundary_prompt(TEMPLATE, window)}
    ]
    # ... and what reached the model is the template's rendering of exactly that.
    assert call["prompt"] == f"<|user|>{render_boundary_prompt(TEMPLATE, window)}<|assistant|>"
    assert "[7] alice (2024-01-01T12:07+00:00): message 7" in call["prompt"]


def test_base_model_without_chat_template_gets_the_raw_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(
        tokenizer=FakeTokenizer(chat_template=None), stream_chunks=["[1]"]
    ).install(monkeypatch)
    provider = _provider()
    window = _window(3)
    assert provider.detect_boundaries(window) == [1]
    assert runtime.stream_calls[0]["prompt"] == render_boundary_prompt(TEMPLATE, window)
    assert runtime.tokenizer.chat_calls == []


def test_windows_shorter_than_two_return_empty_without_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(stream_chunks=["[1]"]).install(monkeypatch)
    provider = _provider()
    assert provider.detect_boundaries([]) == []
    assert provider.detect_boundaries(_window(1)) == []
    assert runtime.load_calls == []
    assert provider.is_loaded is False


def test_loads_once_with_pinned_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(stream_chunks=["[1]"]).install(monkeypatch)
    provider = _provider()
    provider.detect_boundaries(_window(4))
    provider.detect_boundaries(_window(4))
    assert runtime.load_calls == [
        {"path": "org/boundary", "tokenizer_config": None, "revision": "rev3"}
    ]


@pytest.mark.parametrize("chunks", [["not json at all"], ['{"boundaries": [1,'], [""]])
def test_malformed_output_raises_boundary_detection_error(
    monkeypatch: pytest.MonkeyPatch, chunks: list[str]
) -> None:
    FakeRuntime(stream_chunks=chunks).install(monkeypatch)
    with pytest.raises(BoundaryDetectionError):
        _provider().detect_boundaries(_window(4))


@pytest.mark.parametrize("chunks", [['{"boundaries": [0]}'], ['{"boundaries": [4]}'], ["[-1]"]])
def test_out_of_range_index_raises_boundary_detection_error(
    monkeypatch: pytest.MonkeyPatch, chunks: list[str]
) -> None:
    FakeRuntime(stream_chunks=chunks).install(monkeypatch)
    with pytest.raises(BoundaryDetectionError):
        _provider().detect_boundaries(_window(4))


def test_timeout_between_tokens_raises_boundary_detection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A clock that jumps 100s per reading: deadline check after the 1st token
    # sees t=100 (< 120, fine), after the 2nd sees t=200 (> 120, timed out).
    ticks = count(0, 100)
    monkeypatch.setattr(mlx_boundaries, "_monotonic", lambda: float(next(ticks)))
    runtime = FakeRuntime(stream_chunks=['{"boundaries":', " [1, 2", "]}"]).install(monkeypatch)
    provider = _provider(timeout_seconds=120.0)
    with pytest.raises(BoundaryDetectionError) as excinfo:
        provider.detect_boundaries(_window(4))
    assert "120s" in str(excinfo.value)
    assert runtime.stream_calls  # generation was started, then abandoned


def test_generation_failure_raises_boundary_detection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime(stream_chunks=["{", RuntimeError("metal kernel failed")]).install(monkeypatch)
    with pytest.raises(BoundaryDetectionError) as excinfo:
        _provider().detect_boundaries(_window(4))
    assert "generation failed" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_chat_template_failure_raises_boundary_detection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.chat_error = ValueError("template rejected the conversation")
    FakeRuntime(tokenizer=tokenizer, stream_chunks=["[1]"]).install(monkeypatch)
    with pytest.raises(BoundaryDetectionError) as excinfo:
        _provider().detect_boundaries(_window(4))
    assert "chat template failed" in str(excinfo.value)


def test_missing_runtime_maps_to_boundary_detection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    uninstall_runtime(monkeypatch)
    provider = _provider()
    with pytest.raises(BoundaryDetectionError) as excinfo:
        provider.detect_boundaries(_window(4))
    assert "unavailable" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, MlxRuntimeUnavailableError)
    # The explicit startup check keeps the underlying error type.
    with pytest.raises(MlxRuntimeUnavailableError):
        provider.load()


def test_load_failure_maps_to_boundary_detection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRuntime(load_error=ValueError("Model type qwen3_5_moe not supported")).install(monkeypatch)
    provider = _provider()
    with pytest.raises(BoundaryDetectionError) as excinfo:
        provider.detect_boundaries(_window(4))
    assert "org/boundary@rev3" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, MlxRuntimeError)
