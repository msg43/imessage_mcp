"""`imsg.providers.model_smoke` / `scripts/smoke_test_models.py`: the
`artifact_sha256` definition, the byte-preserving lock rewrite, every
role check against stub providers, `run_role`'s timing and peak-memory
bookkeeping, `run_entry`'s download -> checksum -> roles orchestration,
the `smoke_test` record shape, and `main` end to end with an in-process
runner and a stub downloader. No network, no model weights, and nothing
from the `models` extra is required (the image helper runs only when
Pillow is importable; speech synthesis is exercised with a stubbed
`say`)."""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import platform
import re
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import yaml

from imsg.errors import ImsgError
from imsg.providers import model_smoke as smoke
from imsg.providers.manifest import ManifestEntry, default_manifest_path, load_manifest
from imsg.providers.model_smoke import (
    BOUNDARY_DESIGNED_SPLIT,
    EMBED_DOCUMENTS,
    EMBED_QUERY,
    OCR_LINES,
    RERANK_DOCUMENTS,
    RERANK_QUERY,
    SMOKE_QUERY_INSTRUCTION,
    TRANSCRIPTION_SENTENCE,
    EntryResult,
    LockUpdate,
    RoleResult,
    RoleSpec,
    SmokeCheckFailed,
    SmokeDeps,
    SmokeError,
    apply_lock_updates,
    artifact_digest,
    cached_snapshot_dir,
    check_boundaries,
    check_caption,
    check_multimodal,
    check_ocr,
    check_reranker,
    check_text_embedding,
    check_transcription,
    config_from_manifest,
    fetch_snapshot,
    main,
    ordered_entries,
    parse_vm_stat,
    render_smoke_test_lines,
    run_entry,
    run_role,
    scrub_private,
    smoke_record,
    spawn_role,
    synthesize_speech_wav,
    synthetic_boundary_window,
    write_lock_updates,
)

PIN_A = "0123456789abcdef0123456789abcdef01234567"
PIN_B = "89abcdef0123456789abcdef0123456789abcdef"
PIN_C = "fedcba9876543210fedcba9876543210fedcba98"
PIN_D = "1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a"
PIN_E = "2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b2b"

LOCK_TEXT = f"""# fixture lock — header line one
# header line two, preserved verbatim by --write
schema_version: 1
resolved_at: '2026-01-01'
models:
  text-embedder:
    role: [text_embedding]
    status: resolved
    repo: example-org/Example-Embedder-8bit
    revision: {PIN_A}
    license: apache-2.0
    expected_dim: 2048
    quantization: {{mode: mxfp8, bits: 8, group_size: 32}}
    artifact_sha256: null
    min_runtime: {{mlx: '0.32.2', huggingface_hub: '1.31.0'}}
    smoke_test: {{status: not_run}}
    notes: >-
      folded notes that must
      survive untouched
  reranker:
    role: [reranker]
    status: resolved
    repo: example-org/Example-Reranker-8bit
    revision: {PIN_B}
    license: apache-2.0
    expected_dim: null
    quantization: none
    artifact_sha256: null
    min_runtime: {{mlx: '0.32.2'}}
    smoke_test: {{status: not_run}}
  dual-role-llm:
    role: [segment_boundaries, image_caption]
    status: resolved
    repo: example-org/Example-LLM-4bit
    revision: {PIN_C}
    license: apache-2.0
    expected_dim: null
    quantization: {{mode: affine, bits: 4, group_size: 64}}
    artifact_sha256: null
    min_runtime: {{mlx: '0.32.2', mlx-vlm: '0.7.1'}}
    smoke_test: {{status: not_run}}
  whisper:
    role: [transcription]
    status: resolved
    repo: example-org/Example-Whisper
    revision: {PIN_D}
    license: mit
    expected_dim: null
    quantization: none
    artifact_sha256: null
    min_runtime: {{mlx-whisper: '0.4.3'}}
    smoke_test: {{status: not_run}}
  system-ocr:
    role: [ocr]
    status: system
    repo: null
    revision: null
    license: system framework
    expected_dim: null
    quantization: none
    artifact_sha256: null
    min_runtime: {{macos: '13.0'}}
    smoke_test:
      status: passed
      date: '2026-01-01'
      macos: '26.0'
      method: >-
        two lines of
        folded text
    notes: after the block
  multimodal:
    role: [multimodal_embedding]
    status: resolved
    repo: example-org/Example-PE-Core
    revision: {PIN_E}
    license: apache-2.0
    expected_dim: 1280
    quantization: none
    artifact_sha256: null
    min_runtime: {{torch: '2.14.0'}}
    smoke_test: {{status: not_run}}
"""

ALL_ROLES = {
    "text_embedding",
    "reranker",
    "segment_boundaries",
    "image_caption",
    "transcription",
    "ocr",
    "multimodal_embedding",
}


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.lock.yaml"
    path.write_text(LOCK_TEXT, encoding="utf-8")
    return path


def _entry(
    name: str = "text-embedder",
    roles: tuple[str, ...] = ("text_embedding",),
    *,
    status: str = "resolved",
    repo: str | None = "example-org/Example-Embedder-8bit",
    revision: str | None = PIN_A,
) -> ManifestEntry:
    return ManifestEntry(
        name=name,
        roles=roles,
        status=status,
        repo=repo,
        revision=revision,
        license="apache-2.0",
        expected_dim=None,
        min_runtime={},
        raw={},
    )


def _write_snapshot(root: Path, files: dict[str, bytes]) -> Path:
    for relative, payload in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return root


SNAPSHOT_FILES: dict[str, bytes] = {
    "config.json": b'{"model_type": "example"}',
    "model-00001-of-00002.safetensors": b"weights-one" * 100,
    "model-00002-of-00002.safetensors": b"weights-two" * 100,
    "tokenizer.json": b"{}",
    "merges.txt": b"a b\n",
    "1_Pooling/config.json": b'{"pooling_mode_lasttoken": true}',
    "README.md": b"# not hashed",
    ".gitattributes": b"* text=auto",
    "open_clip_pytorch_model.bin": b"duplicate weights, not hashed",
    ".cache/huggingface/download/x.lock": b"",
}


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    return _write_snapshot(tmp_path / "snapshot", SNAPSHOT_FILES)


# --------------------------------------------------------------------------
# artifact_sha256
# --------------------------------------------------------------------------


def _expected_digest(root: Path, relative_names: list[str]) -> str:
    lines = sorted(
        f"{name}  {hashlib.sha256((root / name).read_bytes()).hexdigest()}\n"
        for name in relative_names
    )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def test_artifact_digest_matches_the_stated_definition(snapshot: Path) -> None:
    digest = artifact_digest(snapshot)
    hashed = [
        "1_Pooling/config.json",
        "config.json",
        "merges.txt",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
    ]
    assert [f.relative_path for f in digest.files] == hashed
    assert digest.sha256 == _expected_digest(snapshot, hashed)
    assert digest.total_bytes == sum((snapshot / name).stat().st_size for name in hashed)
    # README.md, .gitattributes, hidden dirs and the .bin duplicate are outside the definition.
    names = {f.relative_path for f in digest.files}
    assert not names & {"README.md", ".gitattributes", "open_clip_pytorch_model.bin"}
    assert not any(name.startswith(".cache") for name in names)


def test_artifact_digest_is_content_sensitive_and_layout_insensitive(tmp_path: Path) -> None:
    a = _write_snapshot(tmp_path / "a", SNAPSHOT_FILES)
    reordered = dict(reversed(list(SNAPSHOT_FILES.items())))
    b = _write_snapshot(tmp_path / "b", reordered)
    assert artifact_digest(a).sha256 == artifact_digest(b).sha256

    extra_readme = dict(SNAPSHOT_FILES)
    extra_readme["README.md"] = b"# a different readme"
    del extra_readme["open_clip_pytorch_model.bin"]
    c = _write_snapshot(tmp_path / "c", extra_readme)
    assert artifact_digest(c).sha256 == artifact_digest(a).sha256

    changed = dict(SNAPSHOT_FILES)
    changed["model-00002-of-00002.safetensors"] = b"weights-two" * 99 + b"weights-twX"
    d = _write_snapshot(tmp_path / "d", changed)
    assert artifact_digest(d).sha256 != artifact_digest(a).sha256

    renamed = dict(SNAPSHOT_FILES)
    renamed["model-00002-of-00003.safetensors"] = renamed.pop("model-00002-of-00002.safetensors")
    e = _write_snapshot(tmp_path / "e", renamed)
    assert artifact_digest(e).sha256 != artifact_digest(a).sha256


def test_artifact_digest_rejects_missing_or_empty_snapshots(tmp_path: Path) -> None:
    with pytest.raises(SmokeError, match="does not exist"):
        artifact_digest(tmp_path / "nope")
    empty = _write_snapshot(tmp_path / "empty", {"README.md": b"only prose"})
    with pytest.raises(SmokeError, match="no weight or config files"):
        artifact_digest(empty)


def test_fetch_snapshot_downloads_everything_but_duplicate_weights(snapshot: Path) -> None:
    calls: list[dict[str, Any]] = []

    def downloader(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    assert fetch_snapshot(_entry(), downloader, skip_download=False) == snapshot
    assert calls == [
        {
            "repo_id": "example-org/Example-Embedder-8bit",
            "revision": PIN_A,
            "ignore_patterns": ["*.bin", "*.pt", "*.pth"],
        }
    ]
    with pytest.raises(SmokeError, match="no repo/revision"):
        fetch_snapshot(
            _entry(status="system", repo=None, revision=None), downloader, skip_download=False
        )


def test_fetch_snapshot_skip_download_uses_the_cache_directory_without_the_hub(
    tmp_path: Path,
) -> None:
    def downloader(**kwargs: Any) -> str:
        raise AssertionError("--skip-download must not call snapshot_download")

    cache = tmp_path / "hub"
    with pytest.raises(SmokeError, match="not in the Hugging Face cache"):
        fetch_snapshot(_entry(), downloader, skip_download=True, cache_dir=cache)
    cached = cache / "models--example-org--Example-Embedder-8bit" / "snapshots" / PIN_A
    _write_snapshot(cached, SNAPSHOT_FILES)
    assert fetch_snapshot(_entry(), downloader, skip_download=True, cache_dir=cache) == cached
    assert cached_snapshot_dir("example-org/Example-Embedder-8bit", PIN_A, cache) == cached


# --------------------------------------------------------------------------
# the lock rewrite
# --------------------------------------------------------------------------

RECORD = {
    "status": "passed",
    "date": "2026-09-14",
    "host_class": "Apple Example, 128 GB unified memory",
    "load_seconds": 12.34,
    "inference_seconds": 0.56,
    "peak_memory_gb": 8.9,
    "peak_memory_source": "mlx.core.get_peak_memory",
    "result": "dim=2048, unit-norm: cos(query, relevant)=0.71 > cos(query, unrelated)=0.12",
}


def _changed_lines(before: str, after: str) -> list[str]:
    return [
        line
        for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---"))
    ]


def test_apply_lock_updates_touches_only_the_two_keys_of_the_named_entry() -> None:
    new_text = apply_lock_updates(
        LOCK_TEXT, [LockUpdate("text-embedder", RECORD, artifact_sha256="ab" * 32)]
    )
    changed = _changed_lines(LOCK_TEXT, new_text)
    assert changed[:2] == [
        "-    artifact_sha256: null",
        "+    artifact_sha256: '" + "ab" * 32 + "'",
    ]
    assert changed[2] == "-    smoke_test: {status: not_run}"
    assert changed[3] == "+    smoke_test:"
    assert all(line.startswith("+      ") for line in changed[4:]), changed
    # The header and every other entry are byte-identical.
    assert new_text.startswith("# fixture lock — header line one\n# header line two")
    before = yaml.safe_load(LOCK_TEXT)
    after = yaml.safe_load(new_text)
    assert after["models"]["text-embedder"]["smoke_test"] == RECORD
    assert after["models"]["text-embedder"]["artifact_sha256"] == "ab" * 32
    for name in before["models"]:
        if name != "text-embedder":
            assert after["models"][name] == before["models"][name]
    assert after["models"]["text-embedder"]["notes"] == before["models"]["text-embedder"]["notes"]


def test_apply_lock_updates_replaces_a_whole_block_style_record_and_keeps_what_follows() -> None:
    record = {**RECORD, "macos": "26.6.2"}
    new_text = apply_lock_updates(LOCK_TEXT, [LockUpdate("system-ocr", record)])
    assert "method: >-" not in new_text
    assert "two lines of" not in new_text
    assert "    notes: after the block\n" in new_text
    after = yaml.safe_load(new_text)
    assert after["models"]["system-ocr"]["smoke_test"] == record
    assert after["models"]["system-ocr"]["artifact_sha256"] is None
    assert after["models"]["system-ocr"]["notes"] == "after the block"
    # artifact_sha256 stays a null line when no digest is given.
    assert "-    artifact_sha256: null" not in _changed_lines(LOCK_TEXT, new_text)


def test_apply_lock_updates_handles_several_entries_and_a_last_entry() -> None:
    new_text = apply_lock_updates(
        LOCK_TEXT,
        [
            LockUpdate("multimodal", {**RECORD, "status": "failed", "error": "boom"}, "cd" * 32),
            LockUpdate("reranker", RECORD, "ef" * 32),
        ],
    )
    after = yaml.safe_load(new_text)
    assert after["models"]["multimodal"]["artifact_sha256"] == "cd" * 32
    assert after["models"]["multimodal"]["smoke_test"]["error"] == "boom"
    assert after["models"]["reranker"]["artifact_sha256"] == "ef" * 32
    assert new_text.endswith("\n")


def test_apply_lock_updates_rejects_unknown_entries_and_non_manifests() -> None:
    with pytest.raises(SmokeError, match="no entry 'ghost'"):
        apply_lock_updates(LOCK_TEXT, [LockUpdate("ghost", RECORD)])
    with pytest.raises(SmokeError):
        apply_lock_updates("just: a mapping\n", [LockUpdate("x", RECORD)])


def test_write_lock_updates_round_trips_through_the_project_parser(lock_path: Path) -> None:
    write_lock_updates(lock_path, [LockUpdate("whisper", RECORD, "12" * 32)])
    lock = load_manifest(lock_path)
    assert (
        lock.header
        == "# fixture lock — header line one\n# header line two, preserved verbatim by --write\n"
    )
    whisper = next(e for e in lock.entries if e.name == "whisper")
    assert whisper.raw["artifact_sha256"] == "12" * 32
    assert whisper.raw["smoke_test"] == RECORD
    assert not list(lock_path.parent.glob("*.smoke-tmp"))


def test_render_smoke_test_lines_nests_roles_in_block_style() -> None:
    record = {
        "status": "failed",
        "date": "2026-09-14",
        "roles": {
            "segment_boundaries": {"status": "passed"},
            "image_caption": {"status": "failed"},
        },
    }
    lines = render_smoke_test_lines(record)
    assert lines[0] == "    smoke_test:"
    assert "      roles:" in lines
    assert "        segment_boundaries:" in lines
    assert yaml.safe_load("\n".join(lines)) == {"smoke_test": record}


# --------------------------------------------------------------------------
# the role checks, against stub providers
# --------------------------------------------------------------------------


def _unit(*components: float) -> list[float]:
    norm = sum(c * c for c in components) ** 0.5
    return [c / norm for c in components]


class StubEmbedder:
    model_id = "stub/embedder@rev"

    def __init__(self, *, dim: int = 4, query: list[float] | None = None) -> None:
        self.dim = dim
        self.documents = [_unit(1.0, 0.1, 0.0, 0.0)[:dim], _unit(0.0, 0.0, 1.0, 0.1)[:dim]]
        self.query = query if query is not None else _unit(0.9, 0.2, 0.1, 0.0)[:dim]
        self.calls: list[tuple[str, Any]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("documents", list(texts)))
        return [list(v) for v in self.documents[: len(texts)]]

    def embed_query(self, text: str, *, instruction: str) -> list[float]:
        self.calls.append(("query", (text, instruction)))
        return list(self.query)


def test_check_text_embedding_passes_and_records_the_inputs() -> None:
    provider = StubEmbedder()
    line = check_text_embedding(provider, dim=4, instruction="find it")
    assert line.startswith("dim=4, unit-norm; cos(query, relevant)=")
    assert provider.calls == [
        ("documents", list(EMBED_DOCUMENTS)),
        ("query", (EMBED_QUERY, "find it")),
    ]


def test_check_text_embedding_rejects_wrong_dim_non_unit_and_wrong_neighbour() -> None:
    with pytest.raises(SmokeCheckFailed, match="expected dim 8"):
        check_text_embedding(StubEmbedder(), dim=8, instruction="i")
    not_unit = StubEmbedder()
    not_unit.documents[0] = [0.5, 0.0, 0.0, 0.0]
    with pytest.raises(SmokeCheckFailed, match="not unit-norm"):
        check_text_embedding(not_unit, dim=4, instruction="i")
    with pytest.raises(SmokeCheckFailed, match="not closer to the relevant"):
        check_text_embedding(StubEmbedder(query=_unit(0.0, 0.1, 1.0, 0.0)), dim=4, instruction="i")
    nan = StubEmbedder()
    nan.query = [float("nan")] * 4
    with pytest.raises(SmokeCheckFailed, match="non-finite"):
        check_text_embedding(nan, dim=4, instruction="i")


class StubReranker:
    model_id = "stub/reranker@rev"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        return list(self.scores)


def test_check_reranker_passes_and_rejects_bad_scores() -> None:
    provider = StubReranker([0.93, 0.02])
    assert check_reranker(provider) == "P(yes): relevant=0.9300 > irrelevant=0.0200, both in [0, 1]"
    assert provider.calls == [(RERANK_QUERY, list(RERANK_DOCUMENTS))]
    with pytest.raises(SmokeCheckFailed, match="outside \\[0, 1\\]"):
        check_reranker(StubReranker([1.5, 0.1]))
    with pytest.raises(SmokeCheckFailed, match="did not outscore"):
        check_reranker(StubReranker([0.2, 0.8]))
    with pytest.raises(SmokeCheckFailed, match="1 scores for 2 documents"):
        check_reranker(StubReranker([0.9]))


class StubBoundary:
    model_id = "stub/boundary@rev"

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.windows: list[Any] = []

    def detect_boundaries(self, window: Any) -> Any:
        self.windows.append(list(window))
        return self.answer


def test_check_boundaries_accepts_any_sorted_in_range_list_and_reports_the_designed_split() -> None:
    provider = StubBoundary([BOUNDARY_DESIGNED_SPLIT])
    assert check_boundaries(provider) == (
        "boundaries=[6] for a 12-message two-topic window; matches the designed split"
    )
    assert len(provider.windows[0]) == 12
    assert "not asserted" in check_boundaries(StubBoundary([3, 9]))
    assert "boundaries=[]" in check_boundaries(StubBoundary([]))


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ([0], "outside 1..11"),
        ([12], "outside 1..11"),
        ([5, 3], "not sorted and unique"),
        ([4, 4], "not sorted and unique"),
        (["6"], "non-integer"),
        ([True], "non-integer"),
        ((6,), "expected a list"),
    ],
)
def test_check_boundaries_rejects_malformed_answers(answer: Any, message: str) -> None:
    with pytest.raises(SmokeCheckFailed, match=message):
        check_boundaries(StubBoundary(answer))


class StubText:
    model_id = "stub/text@rev"

    def __init__(self, text: str) -> None:
        self.text = text
        self.paths: list[Path] = []

    def recognize_text(self, image_path: Path) -> str:
        self.paths.append(image_path)
        return self.text

    def caption(self, image_path: Path) -> str:
        self.paths.append(image_path)
        return self.text

    def transcribe(self, audio_wav_path: Path) -> str:
        self.paths.append(audio_wav_path)
        return self.text


def test_check_ocr_wants_both_rendered_lines() -> None:
    both = StubText("HARBOR KITE FESTIVAL\n  Gate   opens at nine  ")
    assert check_ocr(both, Path("x.png")).startswith("recovered both rendered lines verbatim")
    assert both.paths == [Path("x.png")]
    with pytest.raises(SmokeCheckFailed, match="did not recover \\['Gate opens at nine'\\]"):
        check_ocr(StubText("Harbor Kite Festival"), Path("x.png"))
    with pytest.raises(SmokeCheckFailed, match="did not recover"):
        check_ocr(StubText(""), Path("x.png"))


def test_check_caption_wants_something_visible() -> None:
    line = check_caption(
        StubText("A poster reading 'Harbor Kite Festival' with a red circle."), Path("x.png")
    )
    assert line.startswith("caption mentions ['kite', 'festival', 'harbor']")
    with pytest.raises(SmokeCheckFailed, match="empty"):
        check_caption(StubText("   "), Path("x.png"))
    with pytest.raises(SmokeCheckFailed, match="nothing visible"):
        check_caption(StubText("Two dogs run across a beach."), Path("x.png"))


def test_check_transcription_compares_words_not_punctuation() -> None:
    same = StubText(" the blue kite drifted over the quiet harbor at dawn ")
    assert check_transcription(same, Path("x.wav")).startswith("transcript matched all 10 words")
    with pytest.raises(SmokeCheckFailed, match="transcript words differ"):
        check_transcription(
            StubText("The blue kite drifted over the quiet harbor at noon."), Path("x.wav")
        )
    assert len(TRANSCRIPTION_SENTENCE.split()) == 10


class StubMultimodal:
    model_id = "stub/multimodal@rev"

    def __init__(self, *, dim: int = 3, flip: bool = False) -> None:
        self.dim = dim
        self.image = _unit(1.0, 0.0, 0.0)[:dim]
        self.match = _unit(0.9, 0.1, 0.0)[:dim]
        self.unrelated = _unit(0.0, 0.0, 1.0)[:dim]
        if flip:
            self.match, self.unrelated = self.unrelated, self.match
        self.texts: list[str] = []

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        return [list(self.image) for _ in image_paths]

    def embed_text(self, text: str) -> list[float]:
        self.texts.append(text)
        return list(self.match) if "red circle" in text else list(self.unrelated)


def test_check_multimodal_passes_and_rejects_wrong_dim_or_order() -> None:
    provider = StubMultimodal()
    assert check_multimodal(provider, Path("x.png"), dim=3).startswith("dim=3, unit-norm;")
    assert provider.texts == [
        "a red circle on a white background",
        "a photograph of a snowy mountain at night",
    ]
    with pytest.raises(SmokeCheckFailed, match="expected dim 1280"):
        check_multimodal(StubMultimodal(), Path("x.png"), dim=1280)
    with pytest.raises(SmokeCheckFailed, match="did not outscore"):
        check_multimodal(StubMultimodal(flip=True), Path("x.png"), dim=3)


# --------------------------------------------------------------------------
# run_role: timing and peak-memory bookkeeping
# --------------------------------------------------------------------------


class FakeProbe:
    source = "fake.peak"

    def __init__(self, peak: float = 3.5) -> None:
        self.peak = peak
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def read_gb(self) -> float:
        return self.peak

    def active_gb(self) -> float:
        return self.peak / 2


class FakeClock:
    """Each call advances by the next step in `steps` (seconds)."""

    def __init__(self, *steps: float) -> None:
        self.steps = list(steps)
        self.now = 100.0

    def __call__(self) -> float:
        value = self.now
        if self.steps:
            self.now += self.steps.pop(0)
        return value


class LoadingProvider:
    model_id = "stub/loading@rev"

    def __init__(self) -> None:
        self.loaded = 0
        self.checked = 0

    def load(self) -> None:
        self.loaded += 1


class LazyProvider:
    model_id = "stub/lazy@rev"

    def __init__(self) -> None:
        self.checked = 0


def _deps(
    spec: RoleSpec, probe: FakeProbe, clock: FakeClock, free: float | None = 40.0
) -> SmokeDeps:
    return SmokeDeps(
        role_specs={spec.role: spec},
        probe_factory=lambda kind: probe,
        host_class="Apple Example, 128 GB unified memory",
        free_memory_gb=lambda: free,
        clock=clock,
        today="2026-09-14",
    )


def _cfg(lock_path: Path, tmp_path: Path) -> Any:
    return config_from_manifest(load_manifest(lock_path), data_root=tmp_path)


def test_run_role_times_an_explicit_load_separately_from_the_check(
    lock_path: Path, tmp_path: Path
) -> None:
    provider = LoadingProvider()

    def check(p: Any, cfg: Any, prepared: Any) -> str:
        p.checked += 1
        assert prepared == "prepared-input"
        return "all good at " + str(Path.home() / "secret")

    spec = RoleSpec(
        "text_embedding", "mlx", lambda wd: "prepared-input", lambda cfg, wd: provider, check
    )
    probe = FakeProbe(peak=7.25)
    # clock calls: load start, load end (+12.0), check start, check end (+0.5)
    result = run_role(
        _entry(),
        "text_embedding",
        _cfg(lock_path, tmp_path),
        tmp_path / "w",
        _deps(spec, probe, FakeClock(12.0, 0.0, 0.5)),
    )
    assert result.status == "passed"
    assert result.result == "all good at ~/secret"
    assert result.model_id == "stub/loading@rev"
    assert provider.loaded == 1 and provider.checked == 1
    assert result.load_seconds == pytest.approx(12.0)
    assert result.inference_seconds == pytest.approx(0.5)
    assert result.peak_memory_gb == 7.25
    assert result.peak_memory_source == "fake.peak"
    assert probe.resets == 1
    assert result.details["active_memory_after_load_gb"] == 3.625
    assert result.timing_method.startswith("explicit load()")
    assert result.free_memory_before_gb == 40.0
    assert result.max_rss_gb is not None and result.max_rss_gb > 0


def test_run_role_derives_load_from_two_calls_for_lazy_providers(
    lock_path: Path, tmp_path: Path
) -> None:
    provider = LazyProvider()

    def check(p: Any, cfg: Any, prepared: Any) -> str:
        p.checked += 1
        return "ok"

    spec = RoleSpec("ocr", "rss", lambda wd: None, lambda cfg, wd: provider, check)
    # first call takes 9.0s (load + inference), second 1.0s
    result = run_role(
        _entry("system-ocr", ("ocr",), status="system", repo=None, revision=None),
        "ocr",
        _cfg(lock_path, tmp_path),
        tmp_path / "w",
        _deps(spec, FakeProbe(), FakeClock(9.0, 0.0, 1.0)),
    )
    assert result.status == "passed"
    assert provider.checked == 2
    assert result.load_seconds == pytest.approx(8.0)
    assert result.inference_seconds == pytest.approx(1.0)
    assert result.details["first_call_seconds"] == pytest.approx(9.0)
    assert "first call minus second call" in (result.timing_method or "")
    assert "active_memory_after_load_gb" not in result.details


def test_run_role_records_failures_without_raising(lock_path: Path, tmp_path: Path) -> None:
    cfg = _cfg(lock_path, tmp_path)

    def failing_check(p: Any, c: Any, prepared: Any) -> str:
        raise SmokeCheckFailed(f"query is not closer; snapshot at {Path.home()}/.cache/x")

    spec = RoleSpec(
        "reranker", "mlx", lambda wd: None, lambda c, wd: LoadingProvider(), failing_check
    )
    result = run_role(
        _entry(), "reranker", cfg, tmp_path / "w", _deps(spec, FakeProbe(2.0), FakeClock())
    )
    assert result.status == "failed"
    assert result.error == "query is not closer; snapshot at ~/.cache/x"
    assert result.peak_memory_gb == 2.0  # still read on failure
    assert result.result is None

    def exploding_build(c: Any, wd: Any) -> Any:
        raise RuntimeError("Model type mystery not supported.")

    spec = RoleSpec("reranker", "mlx", lambda wd: None, exploding_build, failing_check)
    result = run_role(
        _entry(), "reranker", cfg, tmp_path / "w", _deps(spec, FakeProbe(), FakeClock())
    )
    assert result.status == "failed"
    assert result.error == "RuntimeError: Model type mystery not supported."

    result = run_role(
        _entry(), "unknown_role", cfg, tmp_path / "w", _deps(spec, FakeProbe(), FakeClock())
    )
    assert result.status == "failed"
    assert result.error == "no smoke check is defined for role 'unknown_role'"


# --------------------------------------------------------------------------
# run_entry: download -> checksum -> roles
# --------------------------------------------------------------------------


def _downloader(
    snapshot: Path, *, fail: Exception | None = None
) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        if fail is not None:
            raise fail
        return str(snapshot)

    return download, calls


def _passing_runner(entry: ManifestEntry, role: str) -> RoleResult:
    return RoleResult(
        entry=entry.name,
        role=role,
        status="passed",
        result=f"{role} ok",
        load_seconds=1.0,
        inference_seconds=0.1,
        peak_memory_gb=2.0,
        peak_memory_source="fake",
    )


def test_run_entry_downloads_hashes_then_runs_roles_in_order(snapshot: Path) -> None:
    download, calls = _downloader(snapshot)
    ran: list[str] = []

    def runner(entry: ManifestEntry, role: str) -> RoleResult:
        ran.append(role)
        return _passing_runner(entry, role)

    out = io.StringIO()
    entry = _entry("dual-role-llm", ("segment_boundaries", "image_caption"))
    result = run_entry(
        entry,
        ["segment_boundaries", "image_caption"],
        downloader=download,
        skip_download=False,
        runner=runner,
        clock=FakeClock(3.0),
        out=out,
    )
    assert result.status == "passed"
    assert ran == ["segment_boundaries", "image_caption"]
    assert result.artifact_sha256 == artifact_digest(snapshot).sha256
    assert result.download_bytes == artifact_digest(snapshot).total_bytes
    assert result.download_seconds == pytest.approx(3.0)
    assert calls[0]["revision"] == PIN_A and "local_files_only" not in calls[0]
    assert "artifact_sha256 " + result.artifact_sha256 in out.getvalue()
    assert "dual-role-llm/segment_boundaries: PASSED" in out.getvalue()


def test_run_entry_skip_download_and_role_failure(tmp_path: Path) -> None:
    cache = tmp_path / "hub"
    snapshot = _write_snapshot(
        cache / "models--example-org--Example-Embedder-8bit" / "snapshots" / PIN_A, SNAPSHOT_FILES
    )
    download, calls = _downloader(snapshot)

    def runner(entry: ManifestEntry, role: str) -> RoleResult:
        if role == "image_caption":
            return RoleResult(entry=entry.name, role=role, status="failed", error="no caption")
        return _passing_runner(entry, role)

    result = run_entry(
        _entry("dual-role-llm", ("segment_boundaries", "image_caption")),
        ["segment_boundaries", "image_caption"],
        downloader=download,
        skip_download=True,
        runner=runner,
        out=io.StringIO(),
        cache_dir=cache,
    )
    assert calls == []  # --skip-download never touches the Hub
    assert result.artifact_sha256 == artifact_digest(snapshot).sha256
    assert result.status == "failed"
    assert [r.status for r in result.roles] == ["passed", "failed"]
    assert result.artifact_sha256 is not None


def test_run_entry_download_failure_runs_no_role(tmp_path: Path) -> None:
    download, _ = _downloader(tmp_path, fail=OSError(f"HTTP 401 for {Path.home()}/x"))
    calls: list[str] = []

    def runner(entry: ManifestEntry, role: str) -> RoleResult:
        calls.append(role)
        return _passing_runner(entry, role)

    out = io.StringIO()
    result = run_entry(
        _entry(),
        ["text_embedding"],
        downloader=download,
        skip_download=False,
        runner=runner,
        out=out,
    )
    assert result.status == "failed"
    assert calls == []
    assert result.error == "OSError: HTTP 401 for ~/x"
    assert result.artifact_sha256 is None
    assert "FAILED before any model ran" in out.getvalue()


def test_run_entry_system_entry_skips_download_and_unresolved_is_skipped(tmp_path: Path) -> None:
    download, calls = _downloader(tmp_path)
    system = _entry("system-ocr", ("ocr",), status="system", repo=None, revision=None)
    result = run_entry(
        system,
        ["ocr"],
        downloader=download,
        skip_download=False,
        runner=_passing_runner,
        out=io.StringIO(),
    )
    assert result.status == "passed" and calls == [] and result.artifact_sha256 is None
    assert [r.role for r in result.roles] == ["ocr"]

    unresolved = _entry("mystery", ("ocr",), status="unresolved", repo=None, revision=None)
    result = run_entry(
        unresolved,
        ["ocr"],
        downloader=download,
        skip_download=False,
        runner=_passing_runner,
        out=io.StringIO(),
    )
    assert result.status == "skipped" and not result.roles

    result = run_entry(
        system,
        [],
        downloader=download,
        skip_download=False,
        runner=_passing_runner,
        out=io.StringIO(),
    )
    assert result.status == "skipped" and result.error == "no roles selected"


# --------------------------------------------------------------------------
# the smoke_test record
# --------------------------------------------------------------------------


def test_smoke_record_single_role_shape() -> None:
    entry = _entry()
    result = EntryResult(
        name=entry.name,
        status="passed",
        roles=[_passing_runner(entry, "text_embedding")],
        artifact_sha256="ab" * 32,
    )
    record = smoke_record(
        entry,
        result,
        date="2026-09-14",
        host_class="Apple Example, 128 GB unified memory",
        macos="26.6.2",
    )
    assert record == {
        "status": "passed",
        "date": "2026-09-14",
        "host_class": "Apple Example, 128 GB unified memory",
        "load_seconds": 1.0,
        "inference_seconds": 0.1,
        "peak_memory_gb": 2.0,
        "peak_memory_source": "fake",
        "result": "text_embedding ok",
    }
    assert "macos" not in record  # only the system entry pins the OS


def test_smoke_record_multi_role_takes_maxima_and_nests_roles() -> None:
    entry = _entry("dual-role-llm", ("segment_boundaries", "image_caption"))
    roles = [
        RoleResult(
            entry.name,
            "segment_boundaries",
            "passed",
            result="boundaries=[6]",
            load_seconds=10.0,
            inference_seconds=2.0,
            peak_memory_gb=21.0,
            peak_memory_source="mlx.core.get_peak_memory",
        ),
        RoleResult(
            entry.name,
            "image_caption",
            "failed",
            error="Model type qwen3_5_moe not supported. " * 30,
            load_seconds=30.0,
            inference_seconds=1.0,
            peak_memory_gb=24.5,
            peak_memory_source="mlx.core.get_peak_memory",
        ),
    ]
    record = smoke_record(
        entry,
        EntryResult(name=entry.name, status="failed", roles=roles),
        date="2026-09-14",
        host_class="h",
        macos=None,
    )
    assert record["status"] == "failed"
    assert (
        record["load_seconds"] == 30.0
        and record["inference_seconds"] == 2.0
        and record["peak_memory_gb"] == 24.5
    )
    assert record["peak_memory_source"] == "mlx.core.get_peak_memory"
    assert record["result"].startswith(
        "segment_boundaries: passed — boundaries=[6]; image_caption: failed — Model type"
    )
    assert len(record["result"]) <= 400
    assert set(record["roles"]) == {"segment_boundaries", "image_caption"}
    assert record["roles"]["segment_boundaries"]["result"] == "boundaries=[6]"
    caption = record["roles"]["image_caption"]
    assert caption["status"] == "failed" and caption["result"].startswith("failed: Model type")
    assert len(caption["error"]) <= 800 and "roles" not in caption


def test_smoke_record_for_a_failure_before_any_role_and_for_the_system_entry() -> None:
    entry = _entry()
    record = smoke_record(
        entry,
        EntryResult(name=entry.name, status="failed", error="OSError: offline"),
        date="2026-09-14",
        host_class="h",
        macos="26.6.2",
    )
    assert record == {
        "status": "failed",
        "date": "2026-09-14",
        "host_class": "h",
        "result": "failed: OSError: offline",
        "error": "OSError: offline",
    }

    system = _entry("system-ocr", ("ocr",), status="system", repo=None, revision=None)
    record = smoke_record(
        system,
        EntryResult(name=system.name, status="passed", roles=[_passing_runner(system, "ocr")]),
        date="2026-09-14",
        host_class="h",
        macos="26.6.2",
    )
    assert record["macos"] == "26.6.2" and record["status"] == "passed"


# --------------------------------------------------------------------------
# main, end to end (in-process runner, stub downloader, stub roles)
# --------------------------------------------------------------------------


class Recorder:
    def __init__(self) -> None:
        self.built: list[str] = []
        self.checked: list[str] = []
        self.failing: set[str] = set()


def _stub_specs(recorder: Recorder) -> dict[str, RoleSpec]:
    specs: dict[str, RoleSpec] = {}
    for role in ALL_ROLES:

        def build(cfg: Any, wd: Path, _role: str = role) -> Any:
            recorder.built.append(_role)
            provider = LoadingProvider()
            provider.model_id = f"stub/{_role}@rev"
            return provider

        def check(p: Any, cfg: Any, prepared: Any, _role: str = role) -> str:
            recorder.checked.append(_role)
            if _role in recorder.failing:
                raise SmokeCheckFailed(f"{_role} produced garbage")
            return f"{_role} ok"

        specs[role] = RoleSpec(
            role, "mlx" if role != "ocr" else "rss", lambda wd: None, build, check
        )
    return specs


def _main_deps(recorder: Recorder) -> SmokeDeps:
    return SmokeDeps(
        role_specs=_stub_specs(recorder),
        probe_factory=lambda kind: FakeProbe(4.0),
        host_class="Apple Example, 128 GB unified memory",
        free_memory_gb=lambda: 50.0,
        clock=FakeClock(),
        today="2026-09-14",
    )


def _snapshot_downloader(tmp_path: Path) -> tuple[Any, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        root = tmp_path / "hub" / kwargs["repo_id"].replace("/", "--") / kwargs["revision"]
        if not root.exists():
            _write_snapshot(
                root, {"config.json": b"{}", "weights.safetensors": kwargs["repo_id"].encode()}
            )
        return str(root)

    return download, calls


def test_main_runs_everything_in_order_and_only_writes_with_write(
    lock_path: Path, tmp_path: Path
) -> None:
    recorder = Recorder()
    download, calls = _snapshot_downloader(tmp_path)
    report_path = tmp_path / "report.json"
    out = io.StringIO()
    before = lock_path.read_text(encoding="utf-8")

    code = main(
        [
            "--lock",
            str(lock_path),
            "--in-process",
            "--work-dir",
            str(tmp_path / "work"),
            "--report-json",
            str(report_path),
        ],
        deps=_main_deps(recorder),
        downloader=download,
        out=out,
    )
    assert code == 0, out.getvalue()
    assert lock_path.read_text(encoding="utf-8") == before
    assert "lock NOT modified" in out.getvalue()
    # Every hosted entry was downloaded at its pin; the system entry was not.
    assert [c["revision"] for c in calls] == [PIN_A, PIN_B, PIN_C, PIN_D, PIN_E]
    # The fixture names are not in SMOKE_RUN_ORDER, so entries run in lock order.
    assert recorder.checked == [
        "text_embedding",
        "reranker",
        "segment_boundaries",
        "image_caption",
        "transcription",
        "ocr",
        "multimodal_embedding",
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["host_class"] == "Apple Example, 128 GB unified memory"
    assert [e["name"] for e in report["entries"]] == [
        "text-embedder",
        "reranker",
        "dual-role-llm",
        "whisper",
        "system-ocr",
        "multimodal",
    ]
    assert all(e["status"] == "passed" for e in report["entries"])
    assert report["total_download_bytes"] > 0
    assert report["entries"][4]["artifact_sha256"] is None

    code = main(
        ["--lock", str(lock_path), "--in-process", "--work-dir", str(tmp_path / "work"), "--write"],
        deps=_main_deps(Recorder()),
        downloader=download,
        out=io.StringIO(),
    )
    assert code == 0
    after = lock_path.read_text(encoding="utf-8")
    lock = load_manifest(lock_path)
    for entry in lock.entries:
        record = entry.raw["smoke_test"]
        assert record["status"] == "passed", entry.name
        assert record["date"] == "2026-09-14" and record["host_class"].startswith("Apple Example")
        if entry.hosted:
            assert re.fullmatch(r"[0-9a-f]{64}", entry.raw["artifact_sha256"]), entry.name
        else:
            assert entry.raw["artifact_sha256"] is None and record["macos"] == (
                platform.mac_ver()[0] or None
            )
    dual = next(e for e in lock.entries if e.name == "dual-role-llm")
    assert set(dual.raw["smoke_test"]["roles"]) == {"segment_boundaries", "image_caption"}
    # Untouched lines survive byte for byte.
    changed = _changed_lines(before, after)
    assert all(
        re.match(r"^[+-]    (artifact_sha256:|smoke_test:)|^[+-]      |^-        ", line)
        for line in changed
    ), changed
    assert "    notes: after the block\n" in after and "folded notes that must" in after


def test_main_only_and_role_filters_and_failure_exit_code(lock_path: Path, tmp_path: Path) -> None:
    recorder = Recorder()
    recorder.failing.add("image_caption")
    download, calls = _snapshot_downloader(tmp_path)
    out = io.StringIO()
    code = main(
        [
            "--lock",
            str(lock_path),
            "--in-process",
            "--work-dir",
            str(tmp_path / "work"),
            "--only",
            "dual-role-llm",
            "--only",
            "reranker",
            "--role",
            "image_caption",
            "--role",
            "reranker",
            "--write",
        ],
        deps=_main_deps(recorder),
        downloader=download,
        out=out,
    )
    assert code == 1
    assert recorder.checked == ["reranker", "image_caption"]
    assert [c["repo_id"] for c in calls] == [
        "example-org/Example-Reranker-8bit",
        "example-org/Example-LLM-4bit",
    ]
    lock = load_manifest(lock_path)
    by_name = {e.name: e for e in lock.entries}
    assert by_name["text-embedder"].raw["smoke_test"] == {"status": "not_run"}
    assert by_name["reranker"].raw["smoke_test"]["status"] == "passed"
    dual = by_name["dual-role-llm"].raw["smoke_test"]
    assert dual["status"] == "failed" and set(dual) >= {"result", "error", "load_seconds"}
    assert "image_caption produced garbage" in dual["error"]
    assert "roles" not in dual  # only one role ran, so the record is flat
    assert re.fullmatch(r"[0-9a-f]{64}", by_name["dual-role-llm"].raw["artifact_sha256"])
    assert "1 entr(y/ies) failed: ['dual-role-llm']" in out.getvalue()


def test_main_usage_errors(lock_path: Path, tmp_path: Path) -> None:
    out = io.StringIO()
    assert (
        main(
            ["--lock", str(lock_path), "--only", "ghost", "--in-process"],
            deps=_main_deps(Recorder()),
            downloader=lambda **kw: "",
            out=out,
        )
        == 2
    )
    assert "unknown entr(y/ies) ['ghost']" in out.getvalue()
    assert (
        main(
            ["--lock", str(tmp_path / "missing.yaml"), "--in-process"],
            deps=_main_deps(Recorder()),
            downloader=lambda **kw: "",
            out=io.StringIO(),
        )
        == 2
    )
    assert (
        main(
            ["--child", "--lock", str(lock_path)],
            deps=_main_deps(Recorder()),
            downloader=lambda **kw: "",
            out=io.StringIO(),
        )
        == 2
    )


def test_spawn_role_reports_a_child_that_writes_no_result(tmp_path: Path) -> None:
    """A real child interpreter, pointed at a lock that does not exist:
    it exits 2 before writing a result and the parent records that."""
    out = io.StringIO()
    result = spawn_role(
        tmp_path / "missing.yaml",
        _entry(),
        "text_embedding",
        tmp_path / "work",
        python=sys.executable,
        out=out,
    )
    assert result.status == "failed"
    assert result.error is not None and "exited with code 2 before writing a result" in result.error
    assert "not found or unreadable" in result.error
    assert not list((tmp_path / "work").glob("*.result.json"))


# --------------------------------------------------------------------------
# config from the lock, ordering, host probes, synthetic inputs
# --------------------------------------------------------------------------


def test_config_from_manifest_carries_every_pin_and_the_real_backend(
    lock_path: Path, tmp_path: Path
) -> None:
    cfg = config_from_manifest(load_manifest(lock_path), data_root=tmp_path)
    assert cfg.models.backend == "real"
    assert (cfg.embedding.model, cfg.embedding.revision) == (
        "example-org/Example-Embedder-8bit",
        PIN_A,
    )
    assert cfg.embedding.query_instruction == SMOKE_QUERY_INSTRUCTION
    assert (cfg.retrieval.reranker_model, cfg.retrieval.reranker_revision) == (
        "example-org/Example-Reranker-8bit",
        PIN_B,
    )
    assert (cfg.segmentation.boundary_model, cfg.segmentation.boundary_revision) == (
        "example-org/Example-LLM-4bit",
        PIN_C,
    )
    assert (cfg.enrichment.caption_model, cfg.enrichment.caption_revision) == (
        "example-org/Example-LLM-4bit",
        PIN_C,
    )
    assert (cfg.enrichment.transcription_model, cfg.enrichment.transcription_revision) == (
        "example-org/Example-Whisper",
        PIN_D,
    )
    assert (cfg.embedding.multimodal.model, cfg.embedding.multimodal.revision) == (
        "example-org/Example-PE-Core",
        PIN_E,
    )
    assert cfg.paths.data_root == tmp_path
    assert cfg.embedding.dim == 2048 and cfg.embedding.multimodal.dim == 1280


def test_config_from_manifest_against_the_real_lock_matches_the_config_defaults(
    tmp_path: Path,
) -> None:
    from imsg import constants

    cfg = config_from_manifest(load_manifest(default_manifest_path()), data_root=tmp_path)
    assert (cfg.embedding.model, cfg.embedding.revision) == (
        constants.TEXT_EMBEDDING_MODEL_REPO,
        constants.TEXT_EMBEDDING_MODEL_REVISION,
    )
    assert (cfg.retrieval.reranker_model, cfg.retrieval.reranker_revision) == (
        constants.RERANKER_MODEL_REPO,
        constants.RERANKER_MODEL_REVISION,
    )
    assert (cfg.segmentation.boundary_model, cfg.segmentation.boundary_revision) == (
        constants.BOUNDARY_MODEL_REPO,
        constants.BOUNDARY_MODEL_REVISION,
    )
    assert (cfg.enrichment.caption_model, cfg.enrichment.caption_revision) == (
        constants.CAPTION_MODEL_REPO,
        constants.CAPTION_MODEL_REVISION,
    )
    assert (cfg.enrichment.transcription_model, cfg.enrichment.transcription_revision) == (
        constants.TRANSCRIPTION_MODEL_REPO,
        constants.TRANSCRIPTION_MODEL_REVISION,
    )
    assert (cfg.embedding.multimodal.model, cfg.embedding.multimodal.revision) == (
        constants.MULTIMODAL_EMBEDDING_MODEL_REPO,
        constants.MULTIMODAL_EMBEDDING_MODEL_REVISION,
    )


def test_config_from_manifest_needs_every_role(lock_path: Path, tmp_path: Path) -> None:
    text = LOCK_TEXT.replace("role: [multimodal_embedding]", "role: [something_else]")
    lock_path.write_text(text, encoding="utf-8")
    with pytest.raises(SmokeError, match="no entry for role 'multimodal_embedding'"):
        config_from_manifest(load_manifest(lock_path), data_root=tmp_path)


def test_ordered_entries_puts_the_largest_last_and_validates_only(lock_path: Path) -> None:
    lock = load_manifest(lock_path)
    assert [e.name for e in ordered_entries(lock, [])] == [
        "text-embedder",
        "reranker",
        "dual-role-llm",
        "whisper",
        "system-ocr",
        "multimodal",
    ]
    assert [e.name for e in ordered_entries(lock, ["whisper", "text-embedder"])] == [
        "text-embedder",
        "whisper",
    ]
    with pytest.raises(SmokeError, match="unknown entr"):
        ordered_entries(lock, ["ghost"])

    real = load_manifest(default_manifest_path())
    names = [e.name for e in ordered_entries(real, [])]
    assert names[-1] == "qwen3.5-35b-a3b" and names[0] == "qwen3-embedding-8b"
    assert set(names) == {e.name for e in real.entries}


def test_parse_vm_stat() -> None:
    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                                     1000.\n"
        "Pages active:                                3682161.\n"
        "Pages inactive:                                 2000.\n"
        "Pages speculative:                              3000.\n"
        "Pages wired down:                             347172.\n"
    )
    assert parse_vm_stat(text) == pytest.approx(6000 * 16384 / 2**30)
    assert parse_vm_stat("nothing useful") is None
    assert parse_vm_stat("(page size of 4096 bytes)\n") is None


def test_scrub_private_removes_home_and_hostname_and_newlines() -> None:
    text = f"loaded from {Path.home()}/.cache/hub\non {platform.node() or 'nohost'}  twice"
    scrubbed = scrub_private(text, home=Path.home(), hostname=platform.node() or "nohost")
    assert str(Path.home()) not in scrubbed
    assert "\n" not in scrubbed
    assert scrubbed.startswith("loaded from ~/.cache/hub on <host>")
    assert scrub_private("plain", home=Path("/"), hostname="") == "plain"


def test_synthetic_boundary_window_is_fictional_and_well_formed() -> None:
    window = synthetic_boundary_window()
    assert len(window) == 12
    assert {m.sender_short_name for m in window} == {"Alpha", "Bravo"}
    assert all(m.text for m in window)
    assert all(a.sent_at < b.sent_at for a, b in pairwise(window))
    assert window[BOUNDARY_DESIGNED_SPLIT].text.startswith("Different question")
    assert [m.message_id for m in window] == list(range(1, 13))


def test_render_text_image_when_pillow_is_available(tmp_path: Path) -> None:
    pillow = pytest.importorskip("PIL.Image")
    path = smoke.render_text_image(tmp_path / "nested" / "text.png")
    with pillow.open(path) as image:
        assert image.size == (900, 340) and image.format == "PNG"
    shape = smoke.render_shape_image(tmp_path / "shape.png")
    with pillow.open(shape) as image:
        assert image.size == (512, 512)
        assert image.getpixel((256, 256))[0] > 200  # red centre
        assert image.getpixel((5, 5)) == (255, 255, 255)
    assert OCR_LINES == ("Harbor Kite Festival", "Gate opens at nine")


def test_synthesize_speech_wav_runs_say_then_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        Path(command[2]).write_bytes(b"FORM")
        return subprocess.CompletedProcess(command, 0, "", "")

    converted: list[tuple[Path, Path]] = []

    def fake_convert(source: Path, output: Path, *, timeout_seconds: int) -> Path:
        converted.append((source, output))
        output.write_bytes(b"RIFF")
        return output

    monkeypatch.setattr(smoke, "convert_to_whisper_wav", fake_convert)
    wav = synthesize_speech_wav(tmp_path / "audio", run=fake_run)
    assert commands == [
        ["say", "-o", str(tmp_path / "audio" / "smoke_speech.aiff"), TRANSCRIPTION_SENTENCE]
    ]
    assert converted == [(tmp_path / "audio" / "smoke_speech.aiff", wav)]
    assert wav.name == "smoke_speech_16k_mono.wav"

    def broken_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("say: command not found")

    with pytest.raises(SmokeError, match="could not synthesise speech"):
        synthesize_speech_wav(tmp_path / "audio2", run=broken_run)


def test_script_wrapper_help_runs(tmp_path: Path) -> None:
    script = default_manifest_path().parents[1] / "scripts" / "smoke_test_models.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    for flag in ["--only", "--role", "--skip-download", "--write", "--work-dir", "--in-process"]:
        assert flag in result.stdout
    assert "--child" not in result.stdout  # internal


def test_smoke_errors_are_imsg_errors() -> None:
    assert issubclass(SmokeError, ImsgError) and issubclass(SmokeCheckFailed, ImsgError)
