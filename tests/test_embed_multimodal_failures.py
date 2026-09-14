"""`imsg.embed.pipeline._embed_multimodal` (S6, D3a) must not let one
unembeddable attachment abort the run.

`ImageEmbeddingError` is the one member of the `EmbeddingError` family
that describes a single *input* (an unreadable file, a frame the image
tower rejects) rather than the runtime or the model. Before 2026-09-14
the pipeline did not catch it, so one corrupt photo — or, without
`pillow-heif`, the first HEIC attachment — stopped the multimodal pass
and left every attachment after it unembedded. Every other
`EmbeddingError` still aborts the run (SPEC §8 S6: "model load failure
-> abort run, nothing partial").

No database: the Postgres helpers the pass calls are replaced with
recorders, which is enough to observe the control flow the tests are
about — what gets upserted, what gets counted, what propagates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import imsg.embed.pipeline as pipeline
from imsg import constants
from imsg.embed.pipeline import EmbedRunReport, run_embed
from imsg.embed.provider import FakeTextEmbeddingProvider
from imsg.errors import EmbeddingError, ImageEmbeddingError, UnreadableImageError

DIM = 4


class _Provider:
    """A multimodal provider whose behaviour is chosen per file name."""

    model_id = "test/multimodal@rev"
    dim = DIM

    def __init__(self, *, unreadable: set[str] = frozenset(), runtime_failure: bool = False) -> None:
        self.unreadable = set(unreadable)
        self.runtime_failure = runtime_failure
        self.calls: list[list[str]] = []

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        self.calls.append([p.name for p in image_paths])
        if self.runtime_failure:
            raise EmbeddingError("device 'mps' is not available on this host")
        for path in image_paths:
            if path.name in self.unreadable:
                raise UnreadableImageError(path, "cannot identify image file")
        return [[1.0, 0.0, 0.0, 0.0] for _ in image_paths]

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 1.0, 0.0, 0.0]


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand-ins for the Postgres reads/writes: what is pending, and what
    got upserted."""
    state: dict[str, Any] = {"images": [], "videos": [], "upserts": []}
    monkeypatch.setattr(pipeline, "_pending_multimodal_images", lambda conn: list(state["images"]))
    monkeypatch.setattr(pipeline, "_pending_multimodal_videos", lambda conn: list(state["videos"]))

    def upsert(conn: Any, attachment_id: int, provider: Any, media_sha256: str, vec: list[float]) -> None:
        state["upserts"].append((attachment_id, media_sha256, vec))

    monkeypatch.setattr(pipeline, "_upsert_mm_embedding", upsert)
    return state


def test_one_unreadable_image_is_counted_and_the_rest_are_still_embedded(
    recorded: dict[str, Any], tmp_path: Path
) -> None:
    recorded["images"] = [
        (1, str(tmp_path / "first.jpg"), "sha-1"),
        (2, str(tmp_path / "broken.heic"), "sha-2"),
        (3, str(tmp_path / "third.jpg"), "sha-3"),
    ]
    provider = _Provider(unreadable={"broken.heic"})

    written, skipped, failed = pipeline._embed_multimodal(object(), provider)  # type: ignore[arg-type]

    assert (written, skipped, failed) == (2, 0, 1)
    assert [u[0] for u in recorded["upserts"]] == [1, 3]
    assert provider.calls == [["first.jpg"], ["broken.heic"], ["third.jpg"]]


def test_a_rejected_video_frame_fails_that_video_only(
    recorded: dict[str, Any], tmp_path: Path
) -> None:
    frames_a = [tmp_path / "a-0.png", tmp_path / "a-1.png"]
    frames_b = [tmp_path / "b-0.png"]
    for frame in [*frames_a, *frames_b]:
        frame.write_bytes(b"frame")
    recorded["videos"] = [
        (10, {"frames": [{"path": str(p)} for p in frames_a]}),
        (11, {"frames": [{"path": str(p)} for p in frames_b]}),
    ]
    provider = _Provider(unreadable={"a-1.png"})

    written, skipped, failed = pipeline._embed_multimodal(object(), provider)  # type: ignore[arg-type]

    assert (written, skipped, failed) == (1, 0, 1)
    assert [u[0] for u in recorded["upserts"]] == [11]


def test_runtime_failures_still_abort_the_run(recorded: dict[str, Any], tmp_path: Path) -> None:
    """Only the per-input error is survivable; a device/model failure is
    the abort-nothing-partial case and must not be swallowed as one more
    'failed attachment'."""
    recorded["images"] = [(1, str(tmp_path / "first.jpg"), "sha-1")]
    with pytest.raises(EmbeddingError) as excinfo:
        pipeline._embed_multimodal(object(), _Provider(runtime_failure=True))  # type: ignore[arg-type]
    assert not isinstance(excinfo.value, ImageEmbeddingError)
    assert recorded["upserts"] == []


def test_run_embed_reports_the_failed_count(
    recorded: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "_pending_segments", lambda conn: [])
    monkeypatch.setattr(pipeline, "_pending_chunks", lambda conn: [])
    recorded["images"] = [
        (1, str(tmp_path / "ok.jpg"), "sha-1"),
        (2, str(tmp_path / "broken.heic"), "sha-2"),
    ]

    class _Multimodal(_Provider):
        dim = constants.MULTIMODAL_EMBEDDING_DIM

    text_provider = FakeTextEmbeddingProvider(dim=constants.PRIMARY_EMBEDDING_DIM)
    report = run_embed(
        object(),  # type: ignore[arg-type]
        text_provider,
        multimodal_provider=_Multimodal(unreadable={"broken.heic"}),
    )

    assert isinstance(report, EmbedRunReport)
    assert report.attachments_embedded == 1
    assert report.attachments_failed == 1
    assert EmbedRunReport().attachments_failed == 0  # default keeps older constructors valid
