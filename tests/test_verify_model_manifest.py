"""`imsg.providers.manifest` / `scripts/verify_model_manifest.py` /
`imsg models verify`: drift detection against a stubbed Hugging Face
API, runtime-floor checks against a stubbed `importlib.metadata`, the
never-write-without-`--write` rule, the `source: local_conversion` entry
kind (its provenance fields, upstream drift that `--write` never
accepts, and the on-disk artifact check against `artifact_sha256`), and
structural checks of the real `models/manifest.lock.yaml` (every
resolved entry carries a full commit sha; runtime floors equal
pyproject's `models` extra). No network."""

from __future__ import annotations

import io
import re
import shutil
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
    SOURCE_HUB,
    SOURCE_LOCAL_CONVERSION,
    artifact_digest,
    default_data_root,
    default_manifest_path,
    entries_by_role,
    hf_model_url,
    license_from_model_info,
    load_manifest,
    main,
    meets_minimum,
    normalize_output_dir,
    verify_local_artifacts,
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
    for flag in [
        "--lock",
        "--data-root",
        "--write",
        "--skip-remote",
        "--skip-artifacts",
        "--skip-runtime",
    ]:
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
    "multimodal_text_embedding",
}


def test_repo_lock_is_well_formed() -> None:
    lock = load_manifest(default_manifest_path())
    assert set(entries_by_role(lock)) == EXPECTED_ROLES
    # The lock is public: no home directory, no hostname-bearing path.
    assert "/Users/" not in default_manifest_path().read_text(encoding="utf-8")
    for entry in lock.entries:
        if entry.local_conversion:
            # Provenance is the upstream pin plus the recorded, reproducible
            # command; the directory itself has no Hub repo/revision.
            assert entry.status in {"resolved", "retained"}, entry.name
            assert (entry.repo, entry.revision, entry.license) == (None, None, None), entry.name
            assert entry.upstream_repo and "/" in entry.upstream_repo, entry.name
            assert entry.upstream_revision and re.fullmatch(
                r"[0-9a-f]{40}", entry.upstream_revision
            ), entry.name
            assert entry.upstream_license, entry.name
            tool = entry.tool_package_and_version
            assert tool is not None, entry.name
            package, version = tool
            assert package in entry.min_runtime and meets_minimum(
                version, entry.min_runtime[package]
            ), f"{entry.name}: the conversion tool must satisfy its own min_runtime floor"
            assert entry.output_dir and entry.output_dir.startswith("models/"), entry.name
            assert entry.output_dir == normalize_output_dir(entry.name, entry.output_dir)
            assert entry.command and f"$DATA_ROOT/{entry.output_dir}" in entry.command, (
                f"{entry.name}: the command must write to $DATA_ROOT/<output_dir>"
            )
            assert "/Volumes/" not in entry.command, entry.name
        elif entry.status in {"resolved", "retained"}:
            assert entry.repo and "/" in entry.repo, entry.name
            assert entry.revision and re.fullmatch(r"[0-9a-f]{40}", entry.revision), entry.name
            assert entry.license, entry.name
        # A smoke_test record is either the untouched `{status: not_run}` or
        # what scripts/smoke_test_models.py --write records
        # (imsg.providers.model_smoke.smoke_record): when, on which host
        # class, how long the load and the inference took, the peak memory,
        # and a one-line result — plus the OS version for the `system`
        # entry, whose model ships with the OS. artifact_sha256 is the
        # smoke run's snapshot digest (64 hex) once a hosted entry has been
        # downloaded, and stays null for the system entry.
        smoke = entry.raw["smoke_test"]
        digest = entry.raw["artifact_sha256"]
        assert smoke["status"] in {"not_run", "passed", "failed"}, entry.name
        if smoke["status"] == "not_run":
            assert smoke == {"status": "not_run"}, entry.name
        else:
            assert {"date", "host_class", "result"} <= set(smoke), entry.name
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(smoke["date"])), entry.name
            if entry.status == "system":
                assert "macos" in smoke, entry.name
            if smoke["status"] == "passed":
                assert {"load_seconds", "peak_memory_gb"} <= set(smoke), entry.name
            for key in ("load_seconds", "inference_seconds", "peak_memory_gb"):
                if key in smoke:
                    assert isinstance(smoke[key], int | float) and smoke[key] >= 0, entry.name
            if "roles" in smoke:
                assert set(smoke["roles"]) == set(entry.roles), entry.name
                for role_record in smoke["roles"].values():
                    assert role_record["status"] in {"passed", "failed"}, entry.name
        if entry.status == "system":
            assert digest is None, entry.name
        else:
            assert digest is None or re.fullmatch(r"[0-9a-f]{64}", digest), entry.name
            if smoke["status"] == "passed":
                assert digest is not None, (
                    f"{entry.name}: a passed smoke run must record its digest"
                )
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


# --------------------------------------------------------------------------
# source: local_conversion — provenance, upstream drift, the artifact check
# --------------------------------------------------------------------------

UPSTREAM = "example-org/Upstream-Reranker"
UPSTREAM_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
UPSTREAM_MOVED = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
OUTPUT_DIR = "models/upstream-reranker-mxfp8-aaaaaaaa"
COMMAND = (
    f"python -c \"from mlx_lm.convert import convert; convert(hf_path='{UPSTREAM}', "
    f"revision='{UPSTREAM_SHA}', mlx_path='$DATA_ROOT/{OUTPUT_DIR}', quantize=True, "
    f"q_mode='mxfp8', q_bits=8, q_group_size=32)\""
)
LOCAL_ENTRY_TEXT = f"""  converted-reranker:
    role: [reranker]
    status: resolved
    source: local_conversion
    upstream_repo: {UPSTREAM}
    upstream_revision: {UPSTREAM_SHA}
    upstream_license: apache-2.0
    tool: mlx-lm==0.31.3
    command: >-
      {COMMAND}
    output_dir: {OUTPUT_DIR}
    expected_dim: null
    quantization: {{mode: mxfp8, bits: 8, group_size: 32}}
    artifact_sha256: null
    min_runtime: {{mlx: '0.32.2', mlx-lm: '0.31.3'}}
    smoke_test: {{status: not_run}}
    notes: why it is local
"""

LOCAL_OK = _installed(
    {
        "mlx": "0.32.2",
        "mlx-lm": "0.31.3",
        "huggingface_hub": "1.31.0",
        MACOS_PSEUDO_PACKAGE: "26.6.2",
    }
)


@pytest.fixture
def local_lock(tmp_path: Path) -> tuple[Path, Path]:
    """The fixture lock plus one local conversion, and a data_root that
    holds its converted directory; `artifact_sha256` is left null."""
    path = tmp_path / "manifest.lock.yaml"
    path.write_text(LOCK_TEXT + LOCAL_ENTRY_TEXT, encoding="utf-8")
    data_root = tmp_path / "data_root"
    directory = data_root / OUTPUT_DIR
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"model_type": "example"}', encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"converted-weights" * 64)
    (directory / "README.md").write_text("# outside the digest", encoding="utf-8")
    return path, data_root


def _record_digest(lock_path: Path, digest: str) -> None:
    text = lock_path.read_text(encoding="utf-8")
    entry = LOCAL_ENTRY_TEXT.replace("artifact_sha256: null", f"artifact_sha256: '{digest}'")
    assert LOCAL_ENTRY_TEXT in text
    lock_path.write_text(text.replace(LOCAL_ENTRY_TEXT, entry), encoding="utf-8")


def _both_unchanged() -> Any:
    return _stub_fetch({hf_model_url(REPO): _info(PINNED), hf_model_url(UPSTREAM): _info(UPSTREAM_SHA)})


def _run_local(lock_path: Path, **kwargs: Any) -> tuple[int, str]:
    kwargs.setdefault("installed", LOCAL_OK)
    kwargs.setdefault("fetch", _both_unchanged())
    return _run(lock_path, **kwargs)


def test_local_conversion_entry_parses_with_its_provenance(local_lock: tuple[Path, Path]) -> None:
    lock_path, _ = local_lock
    by_name = {e.name: e for e in load_manifest(lock_path).entries}
    entry = by_name["converted-reranker"]
    assert entry.source == SOURCE_LOCAL_CONVERSION
    assert entry.local_conversion and not entry.hosted
    assert (entry.repo, entry.revision, entry.license) == (None, None, None)
    assert (entry.upstream_repo, entry.upstream_revision, entry.upstream_license) == (
        UPSTREAM,
        UPSTREAM_SHA,
        "apache-2.0",
    )
    assert entry.hub_pin == (UPSTREAM, UPSTREAM_SHA, "apache-2.0")
    assert (entry.config_model, entry.config_revision) == (OUTPUT_DIR, UPSTREAM_SHA)
    assert entry.tool == "mlx-lm==0.31.3"
    assert entry.tool_package_and_version == ("mlx-lm", "0.31.3")
    assert entry.command == COMMAND  # the folded scalar round-trips as one line
    assert entry.output_dir == OUTPUT_DIR
    assert entry.artifact_sha256 is None
    assert entries_by_role(load_manifest(lock_path))["reranker"] is not None

    hub = by_name["text-embedder"]
    assert hub.source == SOURCE_HUB and hub.hosted and not hub.local_conversion
    assert hub.hub_pin == (REPO, PINNED, "apache-2.0")
    assert (hub.config_model, hub.config_revision) == (REPO, PINNED)
    assert hub.tool_package_and_version is None and hub.output_dir is None
    assert by_name["system-ocr"].hub_pin is None and by_name["unresolved-thing"].hub_pin is None


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "    source: local_conversion\n",
            "    source: local_conversion\n    repo: example-org/Nope\n",
            "no Hub repo/revision/license of its own",
        ),
        (
            "    source: local_conversion\n",
            f"    source: local_conversion\n    revision: {PINNED}\n",
            "no Hub repo/revision/license of its own",
        ),
        ("    tool: mlx-lm==0.31.3\n", "    tool: mlx-lm 0.31.3\n", "'<package>==<exact version>'"),
        ("    tool: mlx-lm==0.31.3\n", "", "non-empty 'tool'"),
        ("    command: >-\n", "    command: ''\n    unused: >-\n", "non-empty 'command'"),
        (f"    upstream_revision: {UPSTREAM_SHA}\n", "    upstream_revision: main\n", "40-hex"),
        (f"    upstream_repo: {UPSTREAM}\n", "    upstream_repo: Upstream-Reranker\n", "'owner/name'"),
        (f"    output_dir: {OUTPUT_DIR}\n", "    output_dir: /abs/models/x\n", "relative to paths.data_root"),
        (f"    output_dir: {OUTPUT_DIR}\n", "    output_dir: ~/models/x\n", "relative to paths.data_root"),
        (f"    output_dir: {OUTPUT_DIR}\n", "    output_dir: models/../../x\n", "'..'"),
        (f"    output_dir: {OUTPUT_DIR}\n", "    output_dir: .\n", "may not be empty"),
        ("    source: local_conversion\n", "    source: mirror\n", "source must be one of"),
        (
            "    status: resolved\n    source: local_conversion\n",
            "    status: system\n    source: local_conversion\n",
            "'resolved' (active) or 'retained' (kept, not active)",
        ),
        ("    artifact_sha256: null\n", "    artifact_sha256: not-a-digest\n", "64-hex sha256"),
    ],
)
def test_local_conversion_entries_are_validated(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    assert LOCAL_ENTRY_TEXT.count(old) == 1, old
    path = tmp_path / "lock.yaml"
    path.write_text(LOCK_TEXT + LOCAL_ENTRY_TEXT.replace(old, new), encoding="utf-8")
    with pytest.raises(ModelManifestError, match=re.escape(message)):
        load_manifest(path)


def test_hub_entries_may_not_carry_local_conversion_keys(tmp_path: Path) -> None:
    path = tmp_path / "lock.yaml"
    path.write_text(
        LOCK_TEXT.replace("    notes: keep me\n", "    notes: keep me\n    output_dir: models/x\n"),
        encoding="utf-8",
    )
    with pytest.raises(ModelManifestError, match=r"belong only to 'source: local_conversion'"):
        load_manifest(path)


def test_normalize_output_dir_strips_redundant_segments() -> None:
    assert normalize_output_dir("e", "models//x/./y/") == "models/x/y"
    with pytest.raises(ModelManifestError):
        normalize_output_dir("e", "/models/x")


def test_local_conversion_re_resolves_its_upstream_repo(local_lock: tuple[Path, Path]) -> None:
    lock_path, _ = local_lock
    code, out = _run_local(lock_path, skip_artifacts=True)
    assert code == 0, out
    assert re.search(
        rf"ok\s+converted-reranker\s+upstream {re.escape(UPSTREAM)} @ {UPSTREAM_SHA[:12]} "
        rf"\(apache-2.0\) -> {re.escape(OUTPUT_DIR)}",
        out,
    )
    assert "local artifacts: skipped (--skip-artifacts)" in out
    assert "models verify: clean" in out


def test_upstream_drift_is_reported_but_never_written(local_lock: tuple[Path, Path]) -> None:
    lock_path, _ = local_lock
    before = lock_path.read_text(encoding="utf-8")
    fetch = _stub_fetch(
        {
            hf_model_url(REPO): _info(PINNED),
            hf_model_url(UPSTREAM): _info(UPSTREAM_MOVED, "mit"),
            hf_model_url(UPSTREAM, UPSTREAM_SHA): _info(UPSTREAM_SHA),
        }
    )
    code, out = _run_local(lock_path, fetch=fetch, write=True, skip_artifacts=True)
    assert code == 1
    assert "DRIFT    converted-reranker" in out
    assert f"upstream revision {UPSTREAM_SHA} -> {UPSTREAM_MOVED} (repo's main moved)" in out
    assert "upstream license 'apache-2.0' -> 'mit'" in out
    assert "NO LONGER resolvable" not in out
    assert f"'{OUTPUT_DIR}' still derives from {UPSTREAM_SHA}" in out
    assert "--write never advances an upstream pin" in out
    assert "1 local conversion(s) whose upstream moved" in out
    assert "wrote" not in out
    assert lock_path.read_text(encoding="utf-8") == before


def test_write_accepts_hub_drift_while_leaving_the_local_conversion_alone(
    local_lock: tuple[Path, Path],
) -> None:
    lock_path, _ = local_lock
    fetch = _stub_fetch(
        {
            hf_model_url(REPO): _info(MOVED, "mit"),
            hf_model_url(REPO, PINNED): _info(PINNED),
            hf_model_url(UPSTREAM): _info(UPSTREAM_MOVED),
            hf_model_url(UPSTREAM, UPSTREAM_SHA): _info(UPSTREAM_SHA),
        }
    )
    code, out = _run_local(lock_path, fetch=fetch, write=True, skip_artifacts=True)
    assert code == 1  # the upstream drift remains
    assert "wrote 1 drifted entr" in out
    data = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert data["models"]["text-embedder"]["revision"] == MOVED
    local = data["models"]["converted-reranker"]
    assert local["upstream_revision"] == UPSTREAM_SHA and local["upstream_license"] == "apache-2.0"
    assert local["command"] == COMMAND and local["notes"] == "why it is local"


def test_vanished_upstream_pin_is_called_out(local_lock: tuple[Path, Path]) -> None:
    lock_path, _ = local_lock
    fetch = _stub_fetch({hf_model_url(REPO): _info(PINNED), hf_model_url(UPSTREAM): _info(UPSTREAM_MOVED)})
    code, out = _run_local(lock_path, fetch=fetch, skip_artifacts=True)
    assert code == 1
    assert f"pinned upstream revision {UPSTREAM_SHA} is NO LONGER resolvable" in out


def test_local_artifact_with_a_matching_digest_is_clean(local_lock: tuple[Path, Path]) -> None:
    lock_path, data_root = local_lock
    directory = data_root / OUTPUT_DIR
    expected = artifact_digest(directory)
    assert [f.relative_path for f in expected.files] == ["config.json", "model.safetensors"]
    _record_digest(lock_path, expected.sha256)

    reports = verify_local_artifacts(load_manifest(lock_path), data_root)
    assert [(r.name, r.level, r.computed_sha256) for r in reports] == [
        ("converted-reranker", "ok", expected.sha256)
    ]
    assert reports[0].directory == directory.resolve()

    code, out = _run_local(lock_path, data_root=data_root)
    assert code == 0, out
    assert f"local artifacts (data_root {data_root}):" in out
    assert re.search(
        rf"ok\s+converted-reranker\s+'{re.escape(OUTPUT_DIR)}' — 2 files, 0\.00 GiB; "
        rf"artifact_sha256 matches",
        out,
    )
    assert "models verify: clean" in out


def test_local_artifact_without_a_recorded_digest_is_a_problem(
    local_lock: tuple[Path, Path],
) -> None:
    lock_path, data_root = local_lock
    code, out = _run_local(lock_path, data_root=data_root)
    assert code == 1
    assert "ERROR    converted-reranker" in out
    assert "the lock records no artifact_sha256" in out
    assert "scripts/smoke_test_models.py --only converted-reranker --write" in out
    assert "models verify: 1 problem(s)" in out


def test_local_artifact_digest_mismatch_is_a_problem(local_lock: tuple[Path, Path]) -> None:
    lock_path, data_root = local_lock
    _record_digest(lock_path, "ab" * 32)
    code, out = _run_local(lock_path, data_root=data_root)
    assert code == 1
    assert f"artifact_sha256 MISMATCH for '{OUTPUT_DIR}': lock {'ab' * 32}, directory" in out
    assert "not the pinned conversion" in out


def test_missing_local_artifact_names_the_recorded_command(local_lock: tuple[Path, Path]) -> None:
    lock_path, data_root = local_lock
    _record_digest(lock_path, "ab" * 32)
    shutil.rmtree(data_root / OUTPUT_DIR)
    code, out = _run_local(lock_path, data_root=data_root)
    assert code == 1
    assert f"'{OUTPUT_DIR}' is not a directory under data_root '{data_root}'" in out
    assert "produce it with the recorded command (mlx-lm==0.31.3):" in out
    assert COMMAND in out


def test_unmounted_data_root_is_a_problem_not_a_traceback(
    local_lock: tuple[Path, Path], tmp_path: Path
) -> None:
    lock_path, _ = local_lock
    _record_digest(lock_path, "ab" * 32)
    code, out = _run_local(lock_path, data_root=tmp_path / "not-mounted")
    assert code == 1
    assert "is not a directory (volume not mounted? pass --data-root)" in out


def test_local_artifact_symlink_escaping_data_root_is_rejected(
    local_lock: tuple[Path, Path], tmp_path: Path
) -> None:
    lock_path, data_root = local_lock
    directory = data_root / OUTPUT_DIR
    outside = tmp_path / "outside"
    shutil.move(str(directory), str(outside))
    directory.symlink_to(outside)
    _record_digest(lock_path, artifact_digest(outside).sha256)
    code, out = _run_local(lock_path, data_root=data_root)
    assert code == 1
    assert "outside data_root" in out


def test_lock_without_local_conversions_prints_no_artifact_section(lock_path: Path) -> None:
    code, out = _run(lock_path, skip_remote=True, data_root=Path("/nonexistent/data-root"))
    assert code == 0, out
    assert "local artifacts" not in out


def test_conversion_tool_version_is_noted_never_enforced(local_lock: tuple[Path, Path]) -> None:
    lock_path, data_root = local_lock
    _record_digest(lock_path, artifact_digest(data_root / OUTPUT_DIR).sha256)

    code, out = _run_local(lock_path, data_root=data_root)
    assert code == 0, out
    assert re.search(
        r"ok\s+mlx-lm\s+== 0\.31\.3\s+installed 0\.31\.3 \(as recorded\) "
        r"\(conversion tool for converted-reranker\)",
        out,
    )

    newer = _installed(
        {"mlx": "0.32.2", "mlx-lm": "0.32.0", "huggingface_hub": "1.31.0", MACOS_PSEUDO_PACKAGE: "26.6.2"}
    )
    code, out = _run_local(lock_path, data_root=data_root, installed=newer)
    assert code == 0, out  # the floor is met and the digest matches; only reproduction is affected
    assert re.search(
        r"NOTE\s+mlx-lm\s+== 0\.31\.3\s+installed 0\.32\.0 — re-running the recorded command "
        r"may not reproduce converted-reranker's artifact_sha256",
        out,
    )
    assert "models verify: clean" in out


def test_default_data_root_is_the_config_schemas() -> None:
    from imsg.config.schema import PathsConfig

    assert default_data_root() == PathsConfig().data_root


def test_main_and_cli_accept_data_root_and_skip_artifacts(local_lock: tuple[Path, Path]) -> None:
    lock_path, data_root = local_lock
    _record_digest(lock_path, artifact_digest(data_root / OUTPUT_DIR).sha256)
    out = io.StringIO()
    code = verify_manifest(
        lock_path,
        data_root=data_root,
        skip_remote=True,
        skip_runtime=True,
        out=out,
    )
    assert code == 0, out.getvalue()
    assert "artifact_sha256 matches" in out.getvalue()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "models",
            "verify",
            "--lock",
            str(lock_path),
            "--data-root",
            str(data_root),
            "--skip-remote",
            "--skip-runtime",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "artifact_sha256 matches" in result.output

    result = runner.invoke(
        app,
        ["models", "verify", "--lock", str(lock_path), "--skip-remote", "--skip-runtime", "--skip-artifacts"],
    )
    assert result.exit_code == 0, result.output
    assert "local artifacts: skipped" in result.output

    assert (
        main(
            [
                "--lock",
                str(lock_path),
                "--data-root",
                str(data_root),
                "--skip-remote",
                "--skip-runtime",
            ]
        )
        == 0
    )


def test_repo_lock_pins_the_0_6b_bf16_as_the_active_reranker_and_retains_the_others() -> None:
    """Exactly one reranker is active, the 0.6B's bf16 build; the builds it
    replaced — the 0.6B's mxfp8 build and the 8B — stay in the lock,
    verified, but claim no role, so switching back is a config change."""
    lock = load_manifest(default_manifest_path())
    active = entries_by_role(lock)["reranker"]
    assert (active.name, active.status) == ("qwen3-reranker-0.6b-bf16", "resolved")
    assert active.local_conversion and active.upstream_repo == "Qwen/Qwen3-Reranker-0.6B"
    assert active.output_dir == "models/qwen3-reranker-0.6b-bf16-e61197ed"
    assert active.command is not None
    assert "quantize=False" in active.command and "dtype='bfloat16'" in active.command
    retained = [e for e in lock.entries if e.retained]
    assert [(e.name, e.roles) for e in retained] == [
        ("qwen3-reranker-0.6b", ("reranker",)),
        ("qwen3-reranker-8b", ("reranker",)),
    ]
    mxfp8 = retained[0]
    assert mxfp8.output_dir == "models/qwen3-reranker-0.6b-mxfp8-e61197ed"
    assert (mxfp8.upstream_repo, mxfp8.upstream_revision) == (
        active.upstream_repo,
        active.upstream_revision,
    )
    assert all(e.local_conversion and e.artifact_sha256 is not None for e in retained)
    assert [e.name for e in lock.entries if "reranker" in e.roles and not e.retained] == [
        active.name
    ]


# --------------------------------------------------------------------------
# status: retained — pinned and verified like a resolved entry, serving no role
# --------------------------------------------------------------------------

RETAINED_UPSTREAM = "example-org/Upstream-Reranker-Large"
RETAINED_SHA = "cccccccccccccccccccccccccccccccccccccccc"
RETAINED_OUTPUT_DIR = "models/upstream-reranker-large-mxfp8-cccccccc"
RETAINED_ENTRY_TEXT = (
    LOCAL_ENTRY_TEXT.replace("  converted-reranker:\n", "  retained-reranker:\n")
    .replace("    status: resolved\n", "    status: retained\n")
    .replace(OUTPUT_DIR, RETAINED_OUTPUT_DIR)
    .replace(UPSTREAM_SHA, RETAINED_SHA)
    .replace(UPSTREAM, RETAINED_UPSTREAM)
)


@pytest.fixture
def retained_lock(local_lock: tuple[Path, Path]) -> tuple[Path, Path]:
    """`local_lock` plus a retained conversion for the same role, both
    directories under data_root and both digests recorded."""
    lock_path, data_root = local_lock
    directory = data_root / RETAINED_OUTPUT_DIR
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"model_type": "example-large"}', encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"larger-weights" * 128)
    _record_digest(lock_path, artifact_digest(data_root / OUTPUT_DIR).sha256)
    retained = RETAINED_ENTRY_TEXT.replace(
        "artifact_sha256: null", f"artifact_sha256: '{artifact_digest(directory).sha256}'"
    )
    lock_path.write_text(lock_path.read_text(encoding="utf-8") + retained, encoding="utf-8")
    return lock_path, data_root


def _all_three_unchanged() -> Any:
    return _stub_fetch(
        {
            hf_model_url(REPO): _info(PINNED),
            hf_model_url(UPSTREAM): _info(UPSTREAM_SHA),
            hf_model_url(RETAINED_UPSTREAM): _info(RETAINED_SHA),
        }
    )


def test_a_retained_entry_is_pinned_but_claims_no_role(retained_lock: tuple[Path, Path]) -> None:
    lock_path, _ = retained_lock
    lock = load_manifest(lock_path)
    by_name = {e.name: e for e in lock.entries}
    retained = by_name["retained-reranker"]
    assert retained.retained and retained.status == "retained"
    assert retained.local_conversion and retained.roles == ("reranker",)
    assert retained.hub_pin == (RETAINED_UPSTREAM, RETAINED_SHA, "apache-2.0")
    assert (retained.config_model, retained.config_revision) == (RETAINED_OUTPUT_DIR, RETAINED_SHA)
    assert not by_name["converted-reranker"].retained
    assert entries_by_role(lock)["reranker"].name == "converted-reranker"


def test_verify_names_the_active_entry_per_role_and_checks_the_retained_one_too(
    retained_lock: tuple[Path, Path],
) -> None:
    lock_path, data_root = retained_lock
    code, out = _run_local(lock_path, data_root=data_root, fetch=_all_three_unchanged())
    assert code == 0, out
    assert (
        "active by role: caption=unresolved-thing, ocr=system-ocr, "
        "reranker=converted-reranker, text_embedding=text-embedder\n"
    ) in out
    assert "retained, not active: retained-reranker (reranker)" in out
    assert re.search(
        rf"ok\s+retained-reranker\s+upstream {re.escape(RETAINED_UPSTREAM)} @ "
        rf"{RETAINED_SHA[:12]} \(apache-2.0\) -> {re.escape(RETAINED_OUTPUT_DIR)} "
        rf"\[retained, not active\]",
        out,
    )
    assert re.search(r"ok\s+converted-reranker\s+'.*artifact_sha256 matches", out)
    assert re.search(r"ok\s+retained-reranker\s+'.*artifact_sha256 matches", out)
    assert "models verify: clean" in out


def test_a_retained_conversion_is_verified_like_an_active_one(
    retained_lock: tuple[Path, Path],
) -> None:
    lock_path, data_root = retained_lock
    shutil.rmtree(data_root / RETAINED_OUTPUT_DIR)
    code, out = _run_local(lock_path, data_root=data_root, fetch=_all_three_unchanged())
    assert code == 1
    assert "ERROR    retained-reranker" in out
    assert f"'{RETAINED_OUTPUT_DIR}' is not a directory under data_root" in out

    fetch = _stub_fetch(
        {
            hf_model_url(REPO): _info(PINNED),
            hf_model_url(UPSTREAM): _info(UPSTREAM_SHA),
            hf_model_url(RETAINED_UPSTREAM): _info(MOVED),
            hf_model_url(RETAINED_UPSTREAM, RETAINED_SHA): _info(RETAINED_SHA),
        }
    )
    code, out = _run_local(lock_path, fetch=fetch, skip_artifacts=True, write=True)
    assert code == 1
    assert "DRIFT    retained-reranker" in out
    assert "1 local conversion(s) whose upstream moved" in out


def test_two_active_entries_for_one_role_are_an_unusable_lock(
    retained_lock: tuple[Path, Path],
) -> None:
    lock_path, data_root = retained_lock
    text = lock_path.read_text(encoding="utf-8")
    both_active = text.replace("    status: retained\n", "    status: resolved\n")
    assert both_active != text
    lock_path.write_text(both_active, encoding="utf-8")
    code, out = _run_local(lock_path, data_root=data_root, fetch=_all_three_unchanged())
    assert code == 2
    assert "role 'reranker' is claimed by both 'converted-reranker' and 'retained-reranker'" in out


def test_a_retained_hub_entry_needs_a_full_pin(tmp_path: Path) -> None:
    path = tmp_path / "lock.yaml"
    retained_hub = LOCK_TEXT.replace(
        "  text-embedder:\n    role: [text_embedding]\n    status: resolved\n",
        "  text-embedder:\n    role: [text_embedding]\n    status: retained\n",
    )
    assert retained_hub != LOCK_TEXT
    path.write_text(retained_hub, encoding="utf-8")
    entry = next(e for e in load_manifest(path).entries if e.name == "text-embedder")
    assert entry.retained and entry.hosted
    assert "text_embedding" not in entries_by_role(load_manifest(path))

    path.write_text(retained_hub.replace(f"    revision: {PINNED}\n", "    revision: main\n", 1), encoding="utf-8")
    with pytest.raises(ModelManifestError, match=r"retained entries need a 40-hex commit sha"):
        load_manifest(path)
