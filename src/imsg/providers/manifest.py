"""`models/manifest.lock.yaml` and its verifier (`imsg models verify`,
`scripts/verify_model_manifest.py`).

The SPEC's model-manifest requirement: the lock "records, for every
model: exact repository, immutable revision, license observed at that
revision, expected dimension, quantization artifact checksum, minimum
runtime versions, and a short smoke-test result. A build MUST NOT
silently advance a model or MLX package because 'latest' changed."

Two kinds of entry share the lock, told apart by `source`:

- `hub` (the default): a Hugging Face repo pinned at a commit sha —
  `repo`, `revision`, `license`. The bytes come from the Hub.
- `local_conversion`: a directory under `paths.data_root`, produced
  from an upstream Hub repo by a recorded, reproducible command —
  `upstream_repo`, `upstream_revision` (sha), `upstream_license`,
  `tool` (`<package>==<exact version>` that ran the conversion),
  `command` (the exact invocation), `output_dir` (data-root-relative).
  A local directory has no Hub revision; its provenance is the
  upstream pin plus `artifact_sha256`, the digest of the OUTPUT
  directory (`artifact_digest`, the same definition the smoke harness
  records for hosted snapshots).

This module enforces the *silently* part. `verify_manifest`:

(a) re-resolves every pinned Hub repo's current `main` revision and
    license from `https://huggingface.co/api/models/<repo>` — for a
    local conversion, the upstream repo — and reports drift versus the
    lock. Drift is information, not an automatic update;
(b) checks every local conversion's `output_dir` exists under
    `data_root` and that its recomputed `artifact_digest` equals the
    lock's `artifact_sha256`;
(c) checks the installed runtime packages (`importlib.metadata`) and,
    for the `macos` pseudo-package, the running OS, against each
    entry's `min_runtime`, and notes whether a conversion's recorded
    `tool` version is what is installed now;
(d) never writes the lock unless `--write` is given, and then only
    rewrites drifted Hub entries' `revision`/`license` (plus the
    top-level `resolved_at`), leaving every other field as it was. A
    local conversion's upstream pin is never advanced by `--write`: the
    directory on disk was converted from the pinned sha, so advancing
    the pin means re-converting and re-pinning by hand.

Config-schema defaults for repos/revisions live in `imsg.constants`;
`tests/test_provider_factory.py` asserts they equal this lock.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import platform
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TextIO

import yaml

from imsg.config.schema import PathsConfig
from imsg.errors import ModelManifestError
from imsg.hashing import sha256_file, sha256_text
from imsg.paths import is_contained_in, resolve_path

MANIFEST_SCHEMA_VERSION = 1
HF_API_MODELS_URL = "https://huggingface.co/api/models/"
HF_USER_AGENT = "imsg-model-manifest-verify/1"
DEFAULT_LOCK_RELATIVE_PATH = Path("models") / "manifest.lock.yaml"
MACOS_PSEUDO_PACKAGE = "macos"
"""`min_runtime` key whose "installed version" is the running macOS
release (`platform.mac_ver()`), not a Python distribution."""

STATUS_RESOLVED = "resolved"
STATUS_SYSTEM = "system"
STATUS_UNRESOLVED = "unresolved"
KNOWN_STATUSES = frozenset({STATUS_RESOLVED, STATUS_SYSTEM, STATUS_UNRESOLVED})

SOURCE_HUB = "hub"
SOURCE_LOCAL_CONVERSION = "local_conversion"
KNOWN_SOURCES = frozenset({SOURCE_HUB, SOURCE_LOCAL_CONVERSION})
LOCAL_CONVERSION_KEYS: tuple[str, ...] = (
    "upstream_repo",
    "upstream_revision",
    "upstream_license",
    "tool",
    "command",
    "output_dir",
)
"""The keys only a `local_conversion` entry carries (all required there,
all forbidden on a `hub` entry)."""

ARTIFACT_FILE_PATTERNS: tuple[str, ...] = (
    "*.safetensors",
    "*.npz",
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.jinja",
    "*.py",
    "*.jsonl",
)
"""The files `artifact_sha256` covers (see `artifact_digest`): a
superset of what `mlx_lm.load` fetches for a repo and of what
`imsg.embed.pe_core_multimodal` fetches."""

GIB = float(2**30)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOL_RE = re.compile(r"^(?P<package>[A-Za-z0-9][A-Za-z0-9_.\-]*)==(?P<version>[0-9][^\s=]*)$")

Fetch = Callable[[str], dict[str, Any]]
"""`url -> parsed JSON body`. Injected so tests never touch the network."""

InstalledVersion = Callable[[str], str | None]
"""`package -> installed version string, or None if not installed`."""


# --------------------------------------------------------------------------
# lock model
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    name: str
    roles: tuple[str, ...]
    status: str
    repo: str | None
    revision: str | None
    license: str | None
    expected_dim: int | None
    min_runtime: Mapping[str, str]
    raw: Mapping[str, Any]
    """The entry exactly as loaded — `--write` round-trips it with only
    `revision`/`license` replaced, so notes/quantization/smoke_test survive."""
    source: str = SOURCE_HUB
    """`hub` or `local_conversion` (module docstring)."""
    artifact_sha256: str | None = None
    upstream_repo: str | None = None
    upstream_revision: str | None = None
    upstream_license: str | None = None
    tool: str | None = None
    """`<package>==<exact version>` of the converter that produced
    `output_dir` (local conversions only)."""
    command: str | None = None
    output_dir: str | None = None
    """POSIX path of the converted directory, relative to `paths.data_root`
    (local conversions only)."""

    @property
    def hosted(self) -> bool:
        """True when the bytes come from a Hugging Face repo pinned in
        this entry — the smoke harness downloads such entries."""
        return self.status == STATUS_RESOLVED and bool(self.repo)

    @property
    def local_conversion(self) -> bool:
        return self.source == SOURCE_LOCAL_CONVERSION

    @property
    def hub_pin(self) -> tuple[str, str, str | None] | None:
        """`(repo, sha, license)` to re-resolve against the Hub: the entry's
        own pin for a hosted entry, the upstream pin for a local
        conversion, `None` when there is nothing hosted to check."""
        if self.local_conversion:
            assert self.upstream_repo is not None and self.upstream_revision is not None
            return self.upstream_repo, self.upstream_revision, self.upstream_license
        if self.hosted:
            assert self.repo is not None and self.revision is not None
            return self.repo, self.revision, self.license
        return None

    @property
    def config_model(self) -> str | None:
        """What a config `*_model` field names for this entry: the Hub repo
        id, or a local conversion's data-root-relative `output_dir`."""
        return self.output_dir if self.local_conversion else self.repo

    @property
    def config_revision(self) -> str | None:
        """What the matching `*_revision` field carries: the pinned commit
        sha — for a local conversion, the UPSTREAM commit it was made from."""
        return self.upstream_revision if self.local_conversion else self.revision

    @property
    def tool_package_and_version(self) -> tuple[str, str] | None:
        if self.tool is None:
            return None
        m = _TOOL_RE.match(self.tool)
        if m is None:  # rejected at parse time; defensive
            return None
        return m.group("package"), m.group("version")


@dataclass(frozen=True, slots=True)
class ManifestLock:
    path: Path
    schema_version: int
    resolved_at: str | None
    entries: tuple[ManifestEntry, ...]
    header: str
    """Leading `#` comment lines of the file, preserved verbatim on write."""
    raw: Mapping[str, Any]


def default_manifest_path() -> Path:
    # src/imsg/providers/manifest.py -> providers -> imsg -> src -> repo root
    return Path(__file__).resolve().parents[3] / DEFAULT_LOCK_RELATIVE_PATH


def default_data_root() -> Path:
    """`paths.data_root` as the config schema defaults it — where a local
    conversion's `output_dir` is looked for when no `--data-root` is
    given. Operators whose config names a different data root pass it
    explicitly."""
    return PathsConfig().data_root


def _require_str(entry_name: str, mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelManifestError(
            f"manifest entry '{entry_name}': '{key}' must be a string or null, got "
            f"{type(value).__name__}"
        )
    return value


def normalize_output_dir(entry_name: str, value: str) -> str:
    """A local conversion's `output_dir`, validated as a relative POSIX
    path that cannot leave `data_root` by construction (no absolute
    path, no `~`, no `..` segment) and normalized (`a//b/` -> `a/b`).
    Containment is re-checked against the real filesystem — symlinks
    included — wherever the directory is actually opened."""
    path = Path(value)
    if not value.strip() or path.is_absolute() or value.startswith("~"):
        raise ModelManifestError(
            f"manifest entry '{entry_name}': 'output_dir' must be a directory relative to "
            f"paths.data_root (e.g. 'models/<conversion>'), got {value!r}"
        )
    parts = [p for p in path.parts if p not in ("", ".")]
    if not parts or ".." in parts:
        raise ModelManifestError(
            f"manifest entry '{entry_name}': 'output_dir' may not be empty or contain '..' "
            f"segments (it must stay under paths.data_root), got {value!r}"
        )
    return Path(*parts).as_posix()


def _parse_local_conversion(name: str, raw: Mapping[str, Any]) -> dict[str, str]:
    """The six `LOCAL_CONVERSION_KEYS`, each validated."""
    values: dict[str, str] = {}
    for key in LOCAL_CONVERSION_KEYS:
        value = _require_str(name, raw, key)
        if not value or not value.strip():
            raise ModelManifestError(
                f"manifest entry '{name}': local_conversion entries need a non-empty '{key}'"
            )
        values[key] = value
    if "/" not in values["upstream_repo"]:
        raise ModelManifestError(
            f"manifest entry '{name}': 'upstream_repo' must be a Hugging Face repo id "
            f"('owner/name'), got {values['upstream_repo']!r}"
        )
    if not _SHA_RE.match(values["upstream_revision"]):
        raise ModelManifestError(
            f"manifest entry '{name}': 'upstream_revision' must be the upstream repo's 40-hex "
            f"commit sha, got {values['upstream_revision']!r} — a branch name or tag is not "
            f"immutable"
        )
    if not _TOOL_RE.match(values["tool"]):
        raise ModelManifestError(
            f"manifest entry '{name}': 'tool' must name the converter as "
            f"'<package>==<exact version>' (e.g. 'mlx-lm==0.31.3'), got {values['tool']!r}"
        )
    values["output_dir"] = normalize_output_dir(name, values["output_dir"])
    return values


def _parse_entry(name: str, raw: Any) -> ManifestEntry:
    if not isinstance(raw, Mapping):
        raise ModelManifestError(f"manifest entry '{name}' must be a mapping")
    roles_raw = raw.get("role")
    roles: tuple[str, ...]
    if isinstance(roles_raw, str):
        roles = (roles_raw,)
    elif isinstance(roles_raw, list) and roles_raw and all(isinstance(r, str) for r in roles_raw):
        roles = tuple(roles_raw)
    else:
        raise ModelManifestError(
            f"manifest entry '{name}': 'role' must be a string or non-empty list of strings"
        )
    status = _require_str(name, raw, "status") or STATUS_RESOLVED
    if status not in KNOWN_STATUSES:
        raise ModelManifestError(
            f"manifest entry '{name}': status must be one of {sorted(KNOWN_STATUSES)}, got '{status}'"
        )
    source = _require_str(name, raw, "source") or SOURCE_HUB
    if source not in KNOWN_SOURCES:
        raise ModelManifestError(
            f"manifest entry '{name}': source must be one of {sorted(KNOWN_SOURCES)}, got '{source}'"
        )
    repo = _require_str(name, raw, "repo")
    revision = _require_str(name, raw, "revision")
    license_id = _require_str(name, raw, "license")
    local: dict[str, str] = {}
    if source == SOURCE_LOCAL_CONVERSION:
        if status != STATUS_RESOLVED:
            raise ModelManifestError(
                f"manifest entry '{name}': a local_conversion entry is always 'resolved' "
                f"(its provenance is the upstream pin), got status '{status}'"
            )
        if repo or revision or license_id:
            raise ModelManifestError(
                f"manifest entry '{name}': a local_conversion has no Hub repo/revision/license "
                f"of its own — leave 'repo', 'revision' and 'license' null; its provenance is "
                f"'upstream_repo' @ 'upstream_revision' ('upstream_license')"
            )
        local = _parse_local_conversion(name, raw)
    else:
        stray = [key for key in LOCAL_CONVERSION_KEYS if raw.get(key) is not None]
        if stray:
            raise ModelManifestError(
                f"manifest entry '{name}': {stray} belong only to 'source: local_conversion' "
                f"entries"
            )
        if status == STATUS_RESOLVED:
            if not repo:
                raise ModelManifestError(f"manifest entry '{name}': resolved entries need a 'repo'")
            if not revision or not _SHA_RE.match(revision):
                raise ModelManifestError(
                    f"manifest entry '{name}': resolved entries need a 40-hex commit sha in "
                    f"'revision', got {revision!r} — a branch name or tag is not immutable"
                )
    expected_dim = raw.get("expected_dim")
    if expected_dim is not None and not isinstance(expected_dim, int):
        raise ModelManifestError(f"manifest entry '{name}': 'expected_dim' must be an int or null")
    min_runtime_raw = raw.get("min_runtime") or {}
    if not isinstance(min_runtime_raw, Mapping):
        raise ModelManifestError(f"manifest entry '{name}': 'min_runtime' must be a mapping")
    min_runtime: dict[str, str] = {}
    for pkg, floor in min_runtime_raw.items():
        if not isinstance(pkg, str) or not isinstance(floor, str | int | float):
            raise ModelManifestError(
                f"manifest entry '{name}': min_runtime entries must be package -> version"
            )
        min_runtime[pkg] = str(floor)
    if "smoke_test" not in raw:
        raise ModelManifestError(f"manifest entry '{name}': missing 'smoke_test'")
    if "artifact_sha256" not in raw:
        raise ModelManifestError(f"manifest entry '{name}': missing 'artifact_sha256'")
    artifact_sha256 = _require_str(name, raw, "artifact_sha256")
    if artifact_sha256 is not None and not _SHA256_RE.match(artifact_sha256):
        raise ModelManifestError(
            f"manifest entry '{name}': 'artifact_sha256' must be null or a 64-hex sha256, got "
            f"{artifact_sha256!r}"
        )
    return ManifestEntry(
        name=name,
        roles=roles,
        status=status,
        repo=repo,
        revision=revision,
        license=license_id,
        expected_dim=expected_dim,
        min_runtime=min_runtime,
        raw=raw,
        source=source,
        artifact_sha256=artifact_sha256,
        upstream_repo=local.get("upstream_repo"),
        upstream_revision=local.get("upstream_revision"),
        upstream_license=local.get("upstream_license"),
        tool=local.get("tool"),
        command=local.get("command"),
        output_dir=local.get("output_dir"),
    )


def _leading_comment_block(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
            continue
        break
    # Drop trailing blank lines so the header is exactly the comment block.
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


def load_manifest(path: Path) -> ManifestLock:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelManifestError(
            f"model manifest lock not found or unreadable: '{path}' ({exc})"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ModelManifestError(f"model manifest lock '{path}' is not valid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ModelManifestError(
            f"model manifest lock '{path}' must be a YAML mapping at the top level"
        )
    schema_version = data.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ModelManifestError(
            f"model manifest lock '{path}': schema_version must be {MANIFEST_SCHEMA_VERSION}, "
            f"got {schema_version!r}"
        )
    models = data.get("models")
    if not isinstance(models, Mapping) or not models:
        raise ModelManifestError(
            f"model manifest lock '{path}': 'models' must be a non-empty mapping"
        )
    entries = tuple(_parse_entry(str(name), raw) for name, raw in models.items())
    resolved_at = data.get("resolved_at")
    return ManifestLock(
        path=path,
        schema_version=MANIFEST_SCHEMA_VERSION,
        resolved_at=str(resolved_at) if resolved_at is not None else None,
        entries=entries,
        header=_leading_comment_block(text),
        raw=data,
    )


def entries_by_role(lock: ManifestLock) -> dict[str, ManifestEntry]:
    out: dict[str, ManifestEntry] = {}
    for entry in lock.entries:
        for role in entry.roles:
            if role in out:
                raise ModelManifestError(
                    f"model manifest lock: role '{role}' is claimed by both "
                    f"'{out[role].name}' and '{entry.name}'"
                )
            out[role] = entry
    return out


# --------------------------------------------------------------------------
# artifact checksum
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    sha256: str
    files: tuple[ArtifactFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)


def iter_artifact_files(
    snapshot: Path, patterns: Sequence[str] = ARTIFACT_FILE_PATTERNS
) -> list[Path]:
    """The regular files under `snapshot` (symlinks into the Hub cache's
    blob store followed) whose name matches `patterns`, hidden entries
    skipped, in relative-path order."""
    matched: list[tuple[str, Path]] = []
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            continue
        matched.append((relative.as_posix(), path))
    matched.sort(key=lambda item: item[0])
    return [path for _, path in matched]


def artifact_digest(
    snapshot: Path, patterns: Sequence[str] = ARTIFACT_FILE_PATTERNS
) -> ArtifactDigest:
    """`artifact_sha256` — exact definition: take every regular file under
    `snapshot` (a Hub snapshot or a local conversion's `output_dir`) whose
    *name* matches one of `patterns` (weights: `*.safetensors`, `*.npz`;
    config/tokenizer: `*.json`, `*.txt`, `*.model`, `*.tiktoken`,
    `*.jinja`, `*.py`, `*.jsonl`), skipping hidden files and directories;
    for each, one line `"<relative POSIX path>  <hex sha256 of the file
    bytes>\\n"` (two spaces, like `sha256sum`); sort the lines by relative
    path (plain string order); the digest is the hex SHA-256 of the UTF-8
    bytes of those lines concatenated. `README.md`, `.gitattributes` and
    the `*.bin`/`*.pt` duplicates of safetensors weights (PE-Core's mirror
    ships both) are deliberately outside the definition, so the digest is
    the same whether or not a full snapshot was fetched. Returns the
    digest plus the per-file lines it was computed from."""
    if not snapshot.is_dir():
        raise ModelManifestError(f"snapshot directory does not exist: '{snapshot}'")
    files: list[ArtifactFile] = []
    for path in iter_artifact_files(snapshot, patterns):
        relative = path.relative_to(snapshot).as_posix()
        files.append(ArtifactFile(relative, sha256_file(path), path.stat().st_size))
    if not files:
        raise ModelManifestError(
            f"snapshot '{snapshot}' holds no weight or config files matching {list(patterns)}"
        )
    files.sort(key=lambda item: item.relative_path)
    listing = "".join(f"{item.relative_path}  {item.sha256}\n" for item in files)
    return ArtifactDigest(sha256=sha256_text(listing), files=tuple(files))


ArtifactDigester = Callable[[Path], ArtifactDigest]
"""`directory -> ArtifactDigest`; `artifact_digest` by default, injected
by tests."""


# --------------------------------------------------------------------------
# remote resolution (Hugging Face API)
# --------------------------------------------------------------------------


def fetch_hf_json(url: str, *, timeout_seconds: float = 20.0) -> dict[str, Any]:
    """Default `Fetch`: an anonymous GET of a Hugging Face API URL.
    Raises `ModelManifestError` for HTTP and transport failures alike,
    with the status code in the message — a 401 from the HF API is how
    a *nonexistent* public repo reports, so say so."""
    request = urllib.request.Request(url, headers={"User-Agent": HF_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        hint = (
            " (the HF API answers 401 for repos that do not exist or are private)"
            if exc.code == 401
            else ""
        )
        raise ModelManifestError(f"HTTP {exc.code} from {url}{hint}") from exc
    except urllib.error.URLError as exc:
        raise ModelManifestError(f"could not reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ModelManifestError(f"timed out fetching {url}") from exc
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise ModelManifestError(f"non-JSON response from {url}") from exc
    if not isinstance(parsed, dict):
        raise ModelManifestError(f"unexpected JSON shape from {url}: expected an object")
    return parsed


def hf_model_url(repo: str, revision: str | None = None) -> str:
    base = f"{HF_API_MODELS_URL}{repo}"
    return f"{base}/revision/{revision}" if revision else base


def license_from_model_info(info: Mapping[str, Any]) -> str | None:
    """`cardData.license` when present, else the `license:<id>` tag —
    the two places the HF API exposes the model-card license."""
    card = info.get("cardData")
    if isinstance(card, Mapping):
        value = card.get("license")
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            return ",".join(value)
    tags = info.get("tags")
    if isinstance(tags, list):
        licenses = [
            t.removeprefix("license:")
            for t in tags
            if isinstance(t, str) and t.startswith("license:")
        ]
        if licenses:
            return ",".join(licenses)
    return None


@dataclass(frozen=True, slots=True)
class EntryReport:
    name: str
    level: str
    """`ok` | `drift` | `error` | `skipped`."""
    messages: tuple[str, ...]
    current_revision: str | None = None
    current_license: str | None = None
    pinned_still_resolvable: bool | None = None
    writable: bool = False
    """True when `--write` may accept this drift (a Hub entry's own pin);
    a local conversion's upstream drift is reported only."""


def verify_remote(lock: ManifestLock, fetch: Fetch) -> list[EntryReport]:
    """(a): re-resolve every hosted entry and every local conversion's
    upstream repo. Never raises for a single entry's failure — that
    entry reports `error` and the rest continue."""
    reports: list[EntryReport] = []
    for entry in lock.entries:
        if entry.status == STATUS_SYSTEM:
            reports.append(
                EntryReport(
                    entry.name, "skipped", ("system framework — nothing hosted to re-resolve",)
                )
            )
            continue
        pin = entry.hub_pin
        if pin is None:
            reports.append(
                EntryReport(
                    entry.name,
                    "skipped",
                    (
                        "status: unresolved — no confirmed repo/revision recorded; resolve it by hand",
                    ),
                )
            )
            continue
        repo, pinned_sha, pinned_license = pin
        what = "upstream " if entry.local_conversion else ""
        try:
            info = fetch(hf_model_url(repo))
        except ModelManifestError as exc:
            reports.append(EntryReport(entry.name, "error", (str(exc),)))
            continue
        current_sha = info.get("sha")
        current_license = license_from_model_info(info)
        messages: list[str] = []
        if not isinstance(current_sha, str) or not current_sha:
            reports.append(
                EntryReport(entry.name, "error", (f"HF API returned no 'sha' for {repo}",))
            )
            continue
        if current_sha != pinned_sha:
            messages.append(f"{what}revision {pinned_sha} -> {current_sha} (repo's main moved)")
        if current_license != pinned_license:
            messages.append(f"{what}license {pinned_license!r} -> {current_license!r}")
        pinned_ok: bool | None = None
        if messages:
            # The pinned sha must remain fetchable even after main moves;
            # if it does not, a rebuild could not reproduce this lock.
            try:
                fetch(hf_model_url(repo, pinned_sha))
                pinned_ok = True
            except ModelManifestError as exc:
                pinned_ok = False
                messages.append(
                    f"pinned {what}revision {pinned_sha} is NO LONGER resolvable: {exc}"
                )
            if entry.local_conversion:
                messages.append(
                    f"the local conversion '{entry.output_dir}' still derives from {pinned_sha}; "
                    f"--write never advances an upstream pin — re-run the recorded command "
                    f"against the new revision and re-pin by hand to move it"
                )
        reports.append(
            EntryReport(
                entry.name,
                "drift" if messages else "ok",
                tuple(messages)
                or (
                    f"{what}{repo} @ {current_sha[:12]} ({current_license})"
                    + (f" -> {entry.output_dir}" if entry.local_conversion else ""),
                ),
                current_revision=current_sha,
                current_license=current_license,
                pinned_still_resolvable=pinned_ok,
                writable=bool(messages) and not entry.local_conversion,
            )
        )
    return reports


# --------------------------------------------------------------------------
# local artifacts
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalArtifactReport:
    name: str
    level: str
    """`ok` | `error`."""
    messages: tuple[str, ...]
    directory: Path | None = None
    computed_sha256: str | None = None


def local_conversion_dir(entry: ManifestEntry, data_root: Path) -> Path:
    """`data_root/<output_dir>`, resolved, and proven to still lie under
    `data_root` after symlink resolution (SPEC §5.4: containment is
    never inferred from a string prefix)."""
    if entry.output_dir is None:
        raise ModelManifestError(f"manifest entry '{entry.name}' is not a local conversion")
    resolved = resolve_path(data_root / entry.output_dir)
    if not is_contained_in(resolved, data_root):
        raise ModelManifestError(
            f"manifest entry '{entry.name}': output_dir '{entry.output_dir}' resolves to "
            f"'{resolved}', outside data_root '{data_root}'"
        )
    return resolved


def verify_local_artifacts(
    lock: ManifestLock, data_root: Path, *, digest: ArtifactDigester = artifact_digest
) -> list[LocalArtifactReport]:
    """(b): every local conversion's directory exists under `data_root`
    and hashes to the lock's `artifact_sha256`. Entries of other kinds
    produce no report."""
    reports: list[LocalArtifactReport] = []
    for entry in lock.entries:
        if not entry.local_conversion:
            continue
        assert entry.output_dir is not None
        if not data_root.is_dir():
            reports.append(
                LocalArtifactReport(
                    entry.name,
                    "error",
                    (
                        f"data_root '{data_root}' is not a directory (volume not mounted? pass "
                        f"--data-root), so '{entry.output_dir}' cannot be checked",
                    ),
                )
            )
            continue
        try:
            directory = local_conversion_dir(entry, data_root)
        except ModelManifestError as exc:
            reports.append(LocalArtifactReport(entry.name, "error", (str(exc),)))
            continue
        if not directory.is_dir():
            reports.append(
                LocalArtifactReport(
                    entry.name,
                    "error",
                    (
                        f"'{entry.output_dir}' is not a directory under data_root '{data_root}' — "
                        f"produce it with the recorded command ({entry.tool}):",
                        entry.command or "<no command recorded>",
                    ),
                    directory=directory,
                )
            )
            continue
        try:
            computed = digest(directory)
        except ModelManifestError as exc:
            reports.append(
                LocalArtifactReport(entry.name, "error", (str(exc),), directory=directory)
            )
            continue
        size = f"{len(computed.files)} files, {computed.total_bytes / GIB:.2f} GiB"
        if entry.artifact_sha256 is None:
            reports.append(
                LocalArtifactReport(
                    entry.name,
                    "error",
                    (
                        f"'{entry.output_dir}' is present ({size}; digest {computed.sha256}) but "
                        f"the lock records no artifact_sha256 — run "
                        f"scripts/smoke_test_models.py --only {entry.name} --write",
                    ),
                    directory=directory,
                    computed_sha256=computed.sha256,
                )
            )
            continue
        if computed.sha256 != entry.artifact_sha256:
            reports.append(
                LocalArtifactReport(
                    entry.name,
                    "error",
                    (
                        f"artifact_sha256 MISMATCH for '{entry.output_dir}': lock "
                        f"{entry.artifact_sha256}, directory {computed.sha256} ({size}) — the "
                        f"directory is not the pinned conversion; re-run the recorded command "
                        f"into a fresh directory, or re-pin it with the smoke test --write",
                    ),
                    directory=directory,
                    computed_sha256=computed.sha256,
                )
            )
            continue
        reports.append(
            LocalArtifactReport(
                entry.name,
                "ok",
                (f"'{entry.output_dir}' — {size}; artifact_sha256 matches",),
                directory=directory,
                computed_sha256=computed.sha256,
            )
        )
    return reports


# --------------------------------------------------------------------------
# runtime floors
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    package: str
    minimum: str
    installed: str | None
    ok: bool
    required_by: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolReport:
    """(c), second half: the converter version a local conversion records
    versus what is installed now — informational, never a problem: the
    directory on disk is checked by digest, and only *re-running* the
    recorded command depends on the same tool version."""

    name: str
    package: str
    recorded: str
    installed: str | None

    @property
    def matches(self) -> bool:
        return self.installed == self.recorded


def version_key(version: str) -> tuple[int, ...]:
    """Leading dotted numeric components only (`12.2.2` -> (12, 2, 2),
    `0.32.2.dev0` -> (0, 32, 2)). Deliberately dependency-free — the
    `packaging` distribution is not a declared runtime dependency — and
    sufficient for floor comparisons of release versions."""
    parts: list[int] = []
    for piece in version.split("."):
        m = re.match(r"(\d+)", piece)
        if not m:
            break
        parts.append(int(m.group(1)))
    return tuple(parts)


def meets_minimum(installed: str, minimum: str) -> bool:
    a, b = version_key(installed), version_key(minimum)
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) >= b + (0,) * (width - len(b))


def installed_version(package: str) -> str | None:
    """Default `InstalledVersion`: `importlib.metadata`, with `macos`
    mapped to the running OS release (`None` off-macOS)."""
    if package == MACOS_PSEUDO_PACKAGE:
        release = platform.mac_ver()[0]
        return release or None
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def verify_runtime(
    lock: ManifestLock, *, installed: InstalledVersion = installed_version
) -> list[RuntimeReport]:
    """(c): the *highest* floor any entry declares per package, checked
    once against what is installed."""
    floors: dict[str, str] = {}
    required_by: dict[str, list[str]] = {}
    for entry in lock.entries:
        for pkg, floor in entry.min_runtime.items():
            required_by.setdefault(pkg, []).append(entry.name)
            if pkg not in floors or not meets_minimum(floors[pkg], floor):
                floors[pkg] = floor
    reports: list[RuntimeReport] = []
    for pkg in sorted(floors):
        have = installed(pkg)
        ok = have is not None and meets_minimum(have, floors[pkg])
        reports.append(RuntimeReport(pkg, floors[pkg], have, ok, tuple(required_by[pkg])))
    return reports


def verify_conversion_tools(
    lock: ManifestLock, *, installed: InstalledVersion = installed_version
) -> list[ToolReport]:
    reports: list[ToolReport] = []
    for entry in lock.entries:
        tool = entry.tool_package_and_version
        if tool is None:
            continue
        package, version = tool
        reports.append(ToolReport(entry.name, package, version, installed(package)))
    return reports


# --------------------------------------------------------------------------
# --write
# --------------------------------------------------------------------------


def apply_drift(lock: ManifestLock, reports: list[EntryReport]) -> ManifestLock:
    """A new lock with each drifted Hub entry's `revision`/`license`
    replaced by what the API reported now, and `resolved_at` set to
    today (UTC). Local conversions are left alone (their drift is not
    `writable`). Pure — the caller decides whether to `write_manifest`
    it."""
    drifted = {r.name: r for r in reports if r.level == "drift" and r.writable}
    if not drifted:
        return lock
    new_entries: list[ManifestEntry] = []
    new_models: dict[str, Any] = {}
    for entry in lock.entries:
        report = drifted.get(entry.name)
        if report is None:
            new_entries.append(entry)
            new_models[entry.name] = dict(entry.raw)
            continue
        raw = dict(entry.raw)
        raw["revision"] = report.current_revision
        raw["license"] = report.current_license
        new_entries.append(
            replace(
                entry, revision=report.current_revision, license=report.current_license, raw=raw
            )
        )
        new_models[entry.name] = raw
    today = datetime.now(UTC).date().isoformat()
    new_raw = dict(lock.raw)
    new_raw["resolved_at"] = today
    new_raw["models"] = new_models
    return replace(lock, resolved_at=today, entries=tuple(new_entries), raw=new_raw)


def write_manifest(lock: ManifestLock, path: Path | None = None) -> Path:
    target = path or lock.path
    body = yaml.safe_dump(dict(lock.raw), sort_keys=False, allow_unicode=True, width=100)
    target.write_text(lock.header + body, encoding="utf-8")
    return target


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------


def verify_manifest(
    path: Path | None = None,
    *,
    fetch: Fetch = fetch_hf_json,
    installed: InstalledVersion = installed_version,
    digest: ArtifactDigester = artifact_digest,
    data_root: Path | None = None,
    write: bool = False,
    skip_remote: bool = False,
    skip_artifacts: bool = False,
    skip_runtime: bool = False,
    out: TextIO | None = None,
) -> int:
    """Run (a), (b) and (c), print a report, return the exit code: 0
    clean, 1 drift / unresolvable / missing or mismatched local artifact
    / runtime below floor, 2 unusable lock. With `write=True`, Hub
    drift is written back and no longer counts against the exit code;
    upstream drift of a local conversion, errors and runtime problems
    still do."""
    stream = out or sys.stdout
    lock_path = path or default_manifest_path()
    root = data_root or default_data_root()
    try:
        lock = load_manifest(lock_path)
    except ModelManifestError as exc:
        print(f"models verify: {exc}", file=stream)
        return 2

    print(
        f"manifest: {lock_path} ({len(lock.entries)} entries, schema v{lock.schema_version}, "
        f"resolved_at {lock.resolved_at})",
        file=stream,
    )
    problems = 0
    drift_count = 0
    unwritable_drift = 0
    if skip_remote:
        print("  remote: skipped (--skip-remote)", file=stream)
        remote_reports: list[EntryReport] = []
    else:
        remote_reports = verify_remote(lock, fetch)
        for report in remote_reports:
            label = {"ok": "ok", "drift": "DRIFT", "error": "ERROR", "skipped": "skipped"}[
                report.level
            ]
            print(f"  {label:8} {report.name:22} {report.messages[0]}", file=stream)
            for extra in report.messages[1:]:
                print(f"  {'':8} {'':22} {extra}", file=stream)
            if report.level == "drift":
                drift_count += 1
                if not report.writable:
                    unwritable_drift += 1
            elif report.level == "error":
                problems += 1

    local_entries = [e for e in lock.entries if e.local_conversion]
    if skip_artifacts:
        print("  local artifacts: skipped (--skip-artifacts)", file=stream)
    elif local_entries:
        print(f"local artifacts (data_root {root}):", file=stream)
        for artifact in verify_local_artifacts(lock, root, digest=digest):
            label = "ok" if artifact.level == "ok" else "ERROR"
            print(f"  {label:8} {artifact.name:22} {artifact.messages[0]}", file=stream)
            for extra in artifact.messages[1:]:
                print(f"  {'':8} {'':22} {extra}", file=stream)
            if artifact.level != "ok":
                problems += 1

    if skip_runtime:
        print("  runtime: skipped (--skip-runtime)", file=stream)
    else:
        print("runtime:", file=stream)
        for rt in verify_runtime(lock, installed=installed):
            if rt.ok:
                state, detail = "ok", f"installed {rt.installed}"
            elif rt.installed is None:
                state, detail = "MISSING", "not installed"
                if rt.package != MACOS_PSEUDO_PACKAGE:
                    detail += " (install the `models` extra)"
            else:
                state, detail = "BELOW", f"installed {rt.installed}"
            print(f"  {state:8} {rt.package:24} >= {rt.minimum:10} {detail}", file=stream)
            if not rt.ok:
                problems += 1
        for tool in verify_conversion_tools(lock, installed=installed):
            if tool.matches:
                state, detail = "ok", f"installed {tool.installed} (as recorded)"
            else:
                state = "NOTE"
                detail = (
                    f"installed {tool.installed or 'nothing'} — re-running the recorded command "
                    f"may not reproduce {tool.name}'s artifact_sha256"
                )
            print(
                f"  {state:8} {tool.package:24} == {tool.recorded:10} {detail} "
                f"(conversion tool for {tool.name})",
                file=stream,
            )

    written = False
    writable_drift = drift_count - unwritable_drift
    if writable_drift and write:
        updated = apply_drift(lock, remote_reports)
        write_manifest(updated)
        written = True
        print(
            f"models verify: wrote {writable_drift} drifted entr(y/ies) to {lock_path}",
            file=stream,
        )
    elif writable_drift:
        print(
            f"models verify: {writable_drift} drifted entr(y/ies) — lock NOT modified "
            f"(pass --write to accept the new revision/license)",
            file=stream,
        )
    if unwritable_drift:
        print(
            f"models verify: {unwritable_drift} local conversion(s) whose upstream moved — "
            f"not writable; re-convert and re-pin by hand",
            file=stream,
        )

    remaining = problems + unwritable_drift + (0 if written else writable_drift)
    if remaining == 0:
        print("models verify: clean", file=stream)
        return 0
    print(f"models verify: {remaining} problem(s)", file=stream)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_model_manifest",
        description="Re-resolve models/manifest.lock.yaml against the Hugging Face API, check "
        "every local conversion's directory under data_root against its recorded digest, and "
        "check installed runtime versions against its floors.",
    )
    parser.add_argument("--lock", type=Path, default=None, help="Path to the lock file.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="paths.data_root, under which local conversions' output_dir live "
        "(default: the config schema's default data root).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Accept Hub drift: rewrite revision/license in the lock (never an upstream pin).",
    )
    parser.add_argument("--skip-remote", action="store_true", help="Do not contact Hugging Face.")
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Do not check local conversions' directories (e.g. when the volume is not mounted).",
    )
    parser.add_argument(
        "--skip-runtime", action="store_true", help="Do not check installed packages."
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return verify_manifest(
        args.lock,
        data_root=args.data_root,
        write=args.write,
        skip_remote=args.skip_remote,
        skip_artifacts=args.skip_artifacts,
        skip_runtime=args.skip_runtime,
    )


__all__ = [
    "ARTIFACT_FILE_PATTERNS",
    "DEFAULT_LOCK_RELATIVE_PATH",
    "GIB",
    "HF_API_MODELS_URL",
    "KNOWN_SOURCES",
    "LOCAL_CONVERSION_KEYS",
    "MANIFEST_SCHEMA_VERSION",
    "SOURCE_HUB",
    "SOURCE_LOCAL_CONVERSION",
    "ArtifactDigest",
    "ArtifactFile",
    "EntryReport",
    "LocalArtifactReport",
    "ManifestEntry",
    "ManifestLock",
    "RuntimeReport",
    "ToolReport",
    "apply_drift",
    "artifact_digest",
    "default_data_root",
    "default_manifest_path",
    "entries_by_role",
    "fetch_hf_json",
    "hf_model_url",
    "installed_version",
    "iter_artifact_files",
    "license_from_model_info",
    "load_manifest",
    "local_conversion_dir",
    "main",
    "meets_minimum",
    "normalize_output_dir",
    "verify_conversion_tools",
    "verify_local_artifacts",
    "verify_manifest",
    "verify_remote",
    "verify_runtime",
    "version_key",
    "write_manifest",
]
