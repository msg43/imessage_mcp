"""`imsg.providers.manifest` / `scripts/verify_model_manifest.py` /
`imsg models verify`: drift detection against a stubbed Hugging Face
API, runtime-floor checks against a stubbed `importlib.metadata`, the
never-write-without-`--write` rule, and structural checks of the real
`models/manifest.lock.yaml` (every resolved entry carries a full commit
sha; runtime floors equal pyproject's `models` extra). No network."""

from __future__ import annotations

import io
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from imsg.cli import app
from imsg.errors import ModelManifestError
from imsg.providers.manifest import (
    MACOS_PSEUDO_PACKAGE,
    default_manifest_path,
    entries_by_role,
    hf_model_url,
    license_from_model_info,
    load_manifest,
    main,
    meets_minimum,
    verify_manifest,
    version_key,
)

PINNED = "0123456789abcdef0123456789abcdef01234567"
MOVED = "ffffffffffffffffffffffffffffffffffffffff"
REPO = "example-org/Example-Embedder-8bit"

LOCK_TEXT = f"""# fixture lock — header line one
# header line two, preserved verbatim by --write
schema_version: 1
resolved_at: '2026-01-01'
models:
  text-embedder:
    role: [text_embedding]
    status: resolved
    repo: {REPO}
    revision: {PINNED}
    license: apache-2.0
    expected_dim: 2048
    quantization: {{mode: mxfp8, bits: 8, group_size: 32}}
    artifact_sha256: null
    min_runtime: {{mlx: '0.32.2', huggingface_hub: '1.31.0'}}
    smoke_test: {{status: not_run}}
    notes: keep me
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
    smoke_test: {{status: not_run}}
  unresolved-thing:
    role: [caption]
    status: unresolved
    repo: null
    revision: null
    license: null
    expected_dim: null
    quantization: none
    artifact_sha256: null
    min_runtime: {{}}
    smoke_test: {{status: not_run}}
"""


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.lock.yaml"
    path.write_text(LOCK_TEXT, encoding="utf-8")
    return path


def _stub_fetch(responses: dict[str, Any]) -> Any:
    """`url -> JSON`, raising `ModelManifestError` for an unknown URL or
    for an entry that is an exception (a 4xx/5xx/transport failure)."""

    def _fetch(url: str) -> dict[str, Any]:
        if url not in responses:
            raise ModelManifestError(f"HTTP 404 from {url}")
        value = responses[url]
        if isinstance(value, Exception):
            raise value
        return dict(value)

    return _fetch


def _info(sha: str, license_id: str = "apache-2.0") -> dict[str, Any]:
    return {"sha": sha, "cardData": {"license": license_id}, "tags": [f"license:{license_id}"]}


def _installed(versions: dict[str, str | None]) -> Any:
    return lambda pkg: versions.get(pkg)


ALL_OK = _installed({"mlx": "0.32.2", "huggingface_hub": "1.31.0", MACOS_PSEUDO_PACKAGE: "26.6.2"})


def _run(lock_path: Path, **kwargs: Any) -> tuple[int, str]:
    out = io.StringIO()
    kwargs.setdefault("installed", ALL_OK)
    code = verify_manifest(lock_path, out=out, **kwargs)
    return code, out.getvalue()


# --------------------------------------------------------------------------
# (a) remote drift
# --------------------------------------------------------------------------


def test_clean_when_the_api_matches_the_lock(lock_path: Path) -> None:
    fetch = _stub_fetch({hf_model_url(REPO): _info(PINNED)})
    code, out = _run(lock_path, fetch=fetch)
    assert code == 0, out
    assert "ok       text-embedder" in out
    assert "skipped  system-ocr" in out
    assert "skipped  unresolved-thing" in out
    assert "models verify: clean" in out
    assert lock_path.read_text(encoding="utf-8") == LOCK_TEXT


def test_drift_is_reported_and_the_lock_is_untouched_without_write(lock_path: Path) -> None:
    fetch = _stub_fetch(
        {hf_model_url(REPO): _info(MOVED, "mit"), hf_model_url(REPO, PINNED): _info(PINNED)}
    )
    code, out = _run(lock_path, fetch=fetch)
    assert code == 1
    assert "DRIFT    text-embedder" in out
    assert f"revision {PINNED} -> {MOVED}" in out
    assert "license 'apache-2.0' -> 'mit'" in out
    assert "NO LONGER resolvable" not in out
    assert "lock NOT modified" in out
    assert lock_path.read_text(encoding="utf-8") == LOCK_TEXT


def test_drift_whose_pinned_revision_vanished_is_called_out(lock_path: Path) -> None:
    fetch = _stub_fetch({hf_model_url(REPO): _info(MOVED)})  # no /revision/<pinned> answer
    code, out = _run(lock_path, fetch=fetch)
    assert code == 1
    assert f"pinned revision {PINNED} is NO LONGER resolvable" in out


def test_write_accepts_drift_preserving_everything_else_and_reverifies_clean(
    lock_path: Path,
) -> None:
    fetch = _stub_fetch(
        {hf_model_url(REPO): _info(MOVED, "mit"), hf_model_url(REPO, PINNED): _info(PINNED)}
    )
    code, out = _run(lock_path, fetch=fetch, write=True)
    assert code == 0, out
    assert "wrote 1 drifted entr" in out

    text = lock_path.read_text(encoding="utf-8")
    assert text.startswith(
        "# fixture lock — header line one\n# header line two, preserved verbatim by --write\n"
    )
    data = yaml.safe_load(text)
    entry = data["models"]["text-embedder"]
    assert entry["revision"] == MOVED
    assert entry["license"] == "mit"
    assert entry["notes"] == "keep me"
    assert entry["quantization"] == {"mode": "mxfp8", "bits": 8, "group_size": 32}
    assert entry["smoke_test"] == {"status": "not_run"}
    assert data["models"]["system-ocr"]["min_runtime"] == {"macos": "13.0"}
    assert data["resolved_at"] == datetime.now(UTC).date().isoformat()
    assert list(data["models"]) == ["text-embedder", "system-ocr", "unresolved-thing"]

    # The rewritten lock re-verifies clean against the same API answers.
    code, out = _run(lock_path, fetch=_stub_fetch({hf_model_url(REPO): _info(MOVED, "mit")}))
    assert code == 0, out


def test_api_error_is_reported_per_entry_not_raised_and_never_written(lock_path: Path) -> None:
    fetch = _stub_fetch({hf_model_url(REPO): ModelManifestError("HTTP 503 from example")})
    code, out = _run(lock_path, fetch=fetch, write=True)
    assert code == 1
    assert "ERROR    text-embedder" in out
    assert "HTTP 503" in out
    assert lock_path.read_text(encoding="utf-8") == LOCK_TEXT


def test_api_answer_without_a_sha_is_an_error(lock_path: Path) -> None:
    fetch = _stub_fetch({hf_model_url(REPO): {"cardData": {"license": "apache-2.0"}}})
    code, out = _run(lock_path, fetch=fetch)
    assert code == 1
    assert "returned no 'sha'" in out


# --------------------------------------------------------------------------
# (b) runtime floors
# --------------------------------------------------------------------------


def test_runtime_floors_report_missing_and_below(lock_path: Path) -> None:
    installed = _installed(
        {"mlx": "0.32.2", "huggingface_hub": "1.30.0", MACOS_PSEUDO_PACKAGE: None}
    )
    code, out = _run(lock_path, skip_remote=True, installed=installed)
    assert code == 1
    assert re.search(r"ok\s+mlx\s+>= 0\.32\.2\s+installed 0\.32\.2", out)
    assert re.search(r"BELOW\s+huggingface_hub\s+>= 1\.31\.0\s+installed 1\.30\.0", out)
    assert re.search(r"MISSING\s+macos\s+>= 13\.0\s+not installed$", out, re.MULTILINE)
    assert "models verify: 2 problem(s)" in out


def test_runtime_floors_pass_when_everything_meets_them(lock_path: Path) -> None:
    code, out = _run(lock_path, skip_remote=True)
    assert code == 0, out
    assert "models verify: clean" in out


def test_skip_flags(lock_path: Path) -> None:
    code, out = _run(lock_path, skip_remote=True, skip_runtime=True, fetch=_stub_fetch({}))
    assert code == 0, out
    assert "remote: skipped" in out
    assert "runtime: skipped" in out


@pytest.mark.parametrize(
    ("installed", "minimum", "ok"),
    [
        ("0.32.2", "0.32.2", True),
        ("0.32.10", "0.32.2", True),
        ("0.32.1", "0.32.2", False),
        ("12.2", "12.2.0", True),
        ("26.6.2", "13.0", True),
        ("0.4.3.dev0", "0.4.3", True),
        ("1.31.0", "1.31", True),
    ],
)
def test_meets_minimum(installed: str, minimum: str, ok: bool) -> None:
    assert meets_minimum(installed, minimum) is ok


def test_version_key_reads_leading_numeric_components_only() -> None:
    assert version_key("12.2.2") == (12, 2, 2)
    assert version_key("2026.9.10") == (2026, 9, 10)
    assert version_key("0.4.3.dev0") == (0, 4, 3)
    assert version_key("1.31.0rc1") == (1, 31, 0)


# --------------------------------------------------------------------------
# lock validation
# --------------------------------------------------------------------------


def test_branch_name_as_revision_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lock.yaml"
    path.write_text(LOCK_TEXT.replace(PINNED, "main"), encoding="utf-8")
    with pytest.raises(ModelManifestError, match=r"40-hex commit sha"):
        load_manifest(path)
    code, out = _run(path, skip_remote=True, skip_runtime=True)
    assert code == 2
    assert "40-hex" in out


def test_missing_lock_is_exit_2(tmp_path: Path) -> None:
    code, out = _run(tmp_path / "nope.yaml", skip_remote=True, skip_runtime=True)
    assert code == 2
    assert "not found" in out


def test_wrong_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lock.yaml"
    path.write_text(LOCK_TEXT.replace("schema_version: 1", "schema_version: 2"), encoding="utf-8")
    with pytest.raises(ModelManifestError, match=r"schema_version"):
        load_manifest(path)


def test_duplicate_role_across_entries_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "lock.yaml"
    path.write_text(LOCK_TEXT.replace("role: [ocr]", "role: [text_embedding]"), encoding="utf-8")
    with pytest.raises(ModelManifestError, match=r"role 'text_embedding' is claimed by both"):
        entries_by_role(load_manifest(path))


def test_license_from_model_info_prefers_card_data_then_tags() -> None:
    assert (
        license_from_model_info({"cardData": {"license": "mit"}, "tags": ["license:apache-2.0"]})
        == "mit"
    )
    assert license_from_model_info({"tags": ["mlx", "license:apache-2.0"]}) == "apache-2.0"
    assert license_from_model_info({"cardData": {}}) is None


def test_hf_model_url_shapes() -> None:
    assert hf_model_url("a/b") == "https://huggingface.co/api/models/a/b"
    assert hf_model_url("a/b", "abc") == "https://huggingface.co/api/models/a/b/revision/abc"


# --------------------------------------------------------------------------
# entry points: argparse script, and `imsg models verify`
# --------------------------------------------------------------------------


def test_main_wires_the_flags(lock_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--lock", str(lock_path), "--skip-remote", "--skip-runtime"]) == 0
    assert "models verify: clean" in capsys.readouterr().out


def test_script_wrapper_runs(lock_path: Path) -> None:
    script = default_manifest_path().parents[1] / "scripts" / "verify_model_manifest.py"
    result = subprocess.run(
        [sys.executable, str(script), "--lock", str(lock_path), "--skip-remote", "--skip-runtime"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "models verify: clean" in result.stdout


def test_cli_models_verify_is_registered_and_wired(lock_path: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["models", "verify", "--help"])
    assert result.exit_code == 0
    for flag in ["--lock", "--write", "--skip-remote", "--skip-runtime"]:
        assert flag in result.output

    result = runner.invoke(
        app, ["models", "verify", "--lock", str(lock_path), "--skip-remote", "--skip-runtime"]
    )
    assert result.exit_code == 0, result.output
    assert "models verify: clean" in result.output

    result = runner.invoke(
        app,
        [
            "models",
            "verify",
            "--lock",
            str(tmp_path / "missing.yaml"),
            "--skip-remote",
            "--skip-runtime",
        ],
    )
    assert result.exit_code == 2
    assert "not found" in result.output


# --------------------------------------------------------------------------
# the real lock file in this repo
# --------------------------------------------------------------------------

EXPECTED_ROLES = {
    "text_embedding",
    "reranker",
    "segment_boundaries",
    "image_caption",
    "transcription",
    "ocr",
    "multimodal_embedding",
}


def test_repo_lock_is_well_formed() -> None:
    lock = load_manifest(default_manifest_path())
    assert set(entries_by_role(lock)) == EXPECTED_ROLES
    for entry in lock.entries:
        if entry.status == "resolved":
            assert entry.repo and "/" in entry.repo, entry.name
            assert entry.revision and re.fullmatch(r"[0-9a-f]{40}", entry.revision), entry.name
            assert entry.license, entry.name
        assert entry.raw["smoke_test"] == {"status": "not_run"}, entry.name
        assert entry.raw["artifact_sha256"] is None, entry.name
    assert not [e.name for e in lock.entries if e.status == "unresolved"]


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def test_repo_lock_runtime_floors_equal_pyprojects_models_extra() -> None:
    """The `models` extra and the lock's `min_runtime` describe the same
    floors; a package present in one but not the other (the PE-Core
    backend placeholder, say) fails here until both are updated."""
    pyproject = tomllib.loads(
        (default_manifest_path().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    extra_floors: dict[str, str] = {}
    for requirement in pyproject["project"]["optional-dependencies"]["models"]:
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*>=\s*([0-9][0-9.]*)", requirement)
        assert m, f"models extra entry must be `name>=floor`, got {requirement!r}"
        extra_floors[_normalize(m.group(1))] = m.group(2)

    lock = load_manifest(default_manifest_path())
    lock_floors: dict[str, set[str]] = {}
    for entry in lock.entries:
        for pkg, floor in entry.min_runtime.items():
            if pkg == MACOS_PSEUDO_PACKAGE:
                continue
            lock_floors.setdefault(_normalize(pkg), set()).add(floor)
    for pkg, floors in lock_floors.items():
        assert pkg in extra_floors, (
            f"{pkg} is in the lock's min_runtime but not pyproject's models extra"
        )
        assert floors == {extra_floors[pkg]}, (
            f"{pkg}: lock floors {floors} != pyproject {extra_floors[pkg]}"
        )
