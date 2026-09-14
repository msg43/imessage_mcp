"""`models/manifest.lock.yaml` and its verifier (`imsg models verify`,
`scripts/verify_model_manifest.py`).

The SPEC's model-manifest requirement: the lock "records, for every
model: exact repository, immutable revision, license observed at that
revision, expected dimension, quantization artifact checksum, minimum
runtime versions, and a short smoke-test result. A build MUST NOT
silently advance a model or MLX package because 'latest' changed."

This module enforces the *silently* part. `verify_manifest`:

(a) re-resolves every Hugging Face-hosted entry's current `main`
    revision and license from `https://huggingface.co/api/models/<repo>`
    and reports drift versus the lock — drift is information, not an
    automatic update;
(b) checks the installed runtime packages (`importlib.metadata`) and,
    for the `macos` pseudo-package, the running OS, against each
    entry's `min_runtime`;
(c) never writes the lock unless `--write` is given, and then only
    rewrites the drifted entries' `revision`/`license` (plus the
    top-level `resolved_at`), leaving every other field as it was.

Config-schema defaults for repos/revisions live in `imsg.constants`;
`tests/test_provider_factory.py` asserts they equal this lock.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TextIO

import yaml

from imsg.errors import ModelManifestError

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

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

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

    @property
    def hosted(self) -> bool:
        """True when there is a Hugging Face repo to re-resolve."""
        return self.status == STATUS_RESOLVED and bool(self.repo)


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
    repo = _require_str(name, raw, "repo")
    revision = _require_str(name, raw, "revision")
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
    return ManifestEntry(
        name=name,
        roles=roles,
        status=status,
        repo=repo,
        revision=revision,
        license=_require_str(name, raw, "license"),
        expected_dim=expected_dim,
        min_runtime=min_runtime,
        raw=raw,
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


def verify_remote(lock: ManifestLock, fetch: Fetch) -> list[EntryReport]:
    """(a): re-resolve every hosted entry. Never raises for a single
    entry's failure — that entry reports `error` and the rest continue."""
    reports: list[EntryReport] = []
    for entry in lock.entries:
        if entry.status == STATUS_SYSTEM:
            reports.append(
                EntryReport(
                    entry.name, "skipped", ("system framework — nothing hosted to re-resolve",)
                )
            )
            continue
        if entry.status == STATUS_UNRESOLVED or not entry.repo:
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
        try:
            info = fetch(hf_model_url(entry.repo))
        except ModelManifestError as exc:
            reports.append(EntryReport(entry.name, "error", (str(exc),)))
            continue
        current_sha = info.get("sha")
        current_license = license_from_model_info(info)
        messages: list[str] = []
        if not isinstance(current_sha, str) or not current_sha:
            reports.append(
                EntryReport(entry.name, "error", (f"HF API returned no 'sha' for {entry.repo}",))
            )
            continue
        if current_sha != entry.revision:
            messages.append(f"revision {entry.revision} -> {current_sha} (repo's main moved)")
        if current_license != entry.license:
            messages.append(f"license {entry.license!r} -> {current_license!r}")
        pinned_ok: bool | None = None
        if messages:
            # The pinned sha must remain fetchable even after main moves;
            # if it does not, a rebuild could not reproduce this lock.
            try:
                fetch(hf_model_url(entry.repo, entry.revision))
                pinned_ok = True
            except ModelManifestError as exc:
                pinned_ok = False
                messages.append(f"pinned revision {entry.revision} is NO LONGER resolvable: {exc}")
        reports.append(
            EntryReport(
                entry.name,
                "drift" if messages else "ok",
                tuple(messages) or (f"{entry.repo} @ {current_sha[:12]} ({current_license})",),
                current_revision=current_sha,
                current_license=current_license,
                pinned_still_resolvable=pinned_ok,
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
    """(b): the *highest* floor any entry declares per package, checked
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


# --------------------------------------------------------------------------
# --write
# --------------------------------------------------------------------------


def apply_drift(lock: ManifestLock, reports: list[EntryReport]) -> ManifestLock:
    """A new lock with each drifted entry's `revision`/`license` replaced
    by what the API reported now, and `resolved_at` set to today (UTC).
    Pure — the caller decides whether to `write_manifest` it."""
    drifted = {r.name: r for r in reports if r.level == "drift"}
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
    write: bool = False,
    skip_remote: bool = False,
    skip_runtime: bool = False,
    out: TextIO | None = None,
) -> int:
    """Run (a) and (b), print a report, return the exit code: 0 clean,
    1 drift / unresolvable / runtime below floor, 2 unusable lock.
    With `write=True`, drift is written back and no longer counts
    against the exit code; errors and runtime problems still do."""
    stream = out or sys.stdout
    lock_path = path or default_manifest_path()
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
            elif report.level == "error":
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

    written = False
    if drift_count and write:
        updated = apply_drift(lock, remote_reports)
        write_manifest(updated)
        written = True
        print(f"models verify: wrote {drift_count} drifted entr(y/ies) to {lock_path}", file=stream)
    elif drift_count:
        print(
            f"models verify: {drift_count} drifted entr(y/ies) — lock NOT modified "
            f"(pass --write to accept the new revision/license)",
            file=stream,
        )

    remaining = problems + (0 if written else drift_count)
    if remaining == 0:
        print("models verify: clean", file=stream)
        return 0
    print(f"models verify: {remaining} problem(s)", file=stream)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_model_manifest",
        description="Re-resolve models/manifest.lock.yaml against the Hugging Face API and "
        "check installed runtime versions against its floors.",
    )
    parser.add_argument("--lock", type=Path, default=None, help="Path to the lock file.")
    parser.add_argument(
        "--write", action="store_true", help="Accept drift: rewrite revision/license in the lock."
    )
    parser.add_argument("--skip-remote", action="store_true", help="Do not contact Hugging Face.")
    parser.add_argument(
        "--skip-runtime", action="store_true", help="Do not check installed packages."
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return verify_manifest(
        args.lock, write=args.write, skip_remote=args.skip_remote, skip_runtime=args.skip_runtime
    )


__all__ = [
    "DEFAULT_LOCK_RELATIVE_PATH",
    "HF_API_MODELS_URL",
    "MANIFEST_SCHEMA_VERSION",
    "EntryReport",
    "ManifestEntry",
    "ManifestLock",
    "RuntimeReport",
    "apply_drift",
    "default_manifest_path",
    "entries_by_role",
    "fetch_hf_json",
    "hf_model_url",
    "installed_version",
    "license_from_model_info",
    "load_manifest",
    "main",
    "meets_minimum",
    "verify_manifest",
    "verify_remote",
    "verify_runtime",
    "version_key",
    "write_manifest",
]
