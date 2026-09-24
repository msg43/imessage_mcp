#!/usr/bin/env python3
"""Scan files and commit messages for real contact data before it reaches the
public repo.

On 2026-09-24 real contact names, a real phone number, and real message
snippets were committed to this repo — some only in commit messages, which
`git grep` at HEAD never sees — and the history had to be rewritten to
remove them. This script is the guard that stops it happening again: it
runs in CI on every push and pull request (see `.github/workflows/ci.yml`)
and can be run by hand against a diff, the whole tree, or a commit range.

What it checks, and how hard it enforces each one:

* **Phone numbers** (hard failure). Flags any US/Canada-shaped number
  whose area code and exchange are not fictional, i.e. not the reserved
  `555` range in either the area-code or exchange position (this repo's
  own fixtures use both shapes: `(415) 555-2671` and `+1 555 222 0000`).
  `PHONE_ALLOWLIST` below holds specific real, public numbers that are
  fine to keep (Apple's published support line).
* **Emails** (hard failure). Flags any address whose domain is not a
  reserved-fictional or example domain (`example.*`, `*.example`,
  `*.invalid`, `*.test`, `fictional.example`) UNLESS the local part
  (before any `+tag` or trailing annotation like `(filtered)`) is one of
  this repo's known fictional personas — covers fixtures like
  `Alice.Example@ICLOUD.com`, which pair a fictional name with a real
  mail provider's domain to test normalization.
* **`/Users/<name>` paths** (hard failure). Flags any macOS home-directory
  path whose `<name>` segment is not a known fictional persona.
* **Names** (report-only, i.e. warnings, unless `--strict`): capitalized
  First-Last pairs, three-word `display_name=`/`name=` fixture values, and
  emoji-decorated first names, each checked against `PERSON_ALLOWLIST`
  below. **This check cannot catch a single common first name used alone**
  ("Jeff" with no surname, no fixture keyword, no emoji) — it only
  recognizes name-*shaped* patterns (a capitalized pair, a quoted fixture
  value, an emoji decoration). A bare common first name slips through by
  design; a human sweep is still required before anything sensitive is
  committed. See SECURITY.md.
* **Commit-message shape** (hard failure, `--commits` only): a line naming
  two colon-prefixed brands/institutions and the word "recipient" — the
  shape of the real leaked commit message this script exists to prevent
  ("PayPal: ... Wells Fargo: ... recipient ...").

Usage:
    uv run python scripts/check_public_safety.py <file> [<file> ...]
    uv run python scripts/check_public_safety.py --all-files
    uv run python scripts/check_public_safety.py --commits <git-range>
    uv run python scripts/check_public_safety.py --all-files --strict

Exit status is non-zero if any hard finding is reported (or, with
`--strict`, any finding at all).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlists
#
# Every entry here is either a fictional persona already used in this
# repo's own tests/fixtures (found by scanning `tests/` for
# `display_name=`, `full_name=`, `short_name=`, `/Users/...` and email
# literals), or a specific real-but-public number that is safe to name.
# Extend these, don't loosen the regexes/checks above them.
# ---------------------------------------------------------------------------

# Real, public phone numbers that are fine to keep verbatim (not a leak).
# Apple's published AppleCare/support line, referenced in docs/help text.
PHONE_ALLOWLIST: set[str] = {
    "+18002752273",  # Apple support, (800) 275-2273 — public, not a contact
}

# First names (and a few first+last combos) used throughout tests/ and
# src/ as fictional personas — the "Alice and Bob" cast plus this repo's
# own additions. Matched case-insensitively against a bare first name,
# an email local-part, or a /Users/<name> path segment.
PERSON_FIRST_NAMES: set[str] = {
    "alice",
    "bob",
    "carol",
    "dana",
    "dave",
    "erin",
    "evan",
    "frank",
    "gina",
    "jamie",
    "mallory",
    "someone",
    "example",
    "user",
    "test",
    "nobody",
    "owner",
    "alex",
    "jane",
    "joe",
    "joseph",
    "robert",
}

# Full "First Last" (or "First Last-Last") persona names seen in this
# repo's fixtures — kept as a separate set because a first name alone
# (above) is not enough to clear a capitalized-pair or display-name check
# for an unfamiliar surname.
PERSON_FULL_NAMES: set[str] = {
    "alice bell carter",
    "alice carter",
    "alice example",
    "alice hand-named",
    "bob builder",
    "bob example",
    "bob feldman",
    "bob hand-named",
    "bob mason",
    "bob pool",
    "carol carpenter",
    "carol chen",
    "carol chen-okafor",
    "carol example",
    "dana driver",
    "dana example",
    "dave carter",
    "erin delgado",
    "evan estimator",
    "frank example",
    "frank framer",
    "gina glazier",
    "jamie owner",
    "jamie renamed",
    "jane doe",
    "joe marsh",
    "joseph marsh",
    "mallory mason",
    "not the owner",
    "robert builder",
    "acme pool",
    "acme construction",
    "n pool",
    "r pool",
}

# Email domains treated as inherently fictional/reserved. A domain
# "clears" if it equals one of these, starts with "example.", or ends
# with one of the dotted suffixes.
EMAIL_ALLOWED_DOMAIN_EXACT: set[str] = {"fictional.example"}
EMAIL_ALLOWED_DOMAIN_PREFIXES: tuple[str, ...] = ("example.",)
EMAIL_ALLOWED_DOMAIN_SUFFIXES: tuple[str, ...] = (".example", ".invalid", ".test")

# Local parts that are a structural convention, not a person, regardless
# of domain: "noreply@<anything>" is a bot placeholder (this repo's own
# commit trailers use `noreply@anthropic.com` for AI co-author
# attribution, the same convention GitHub uses for
# `<id>+<user>@users.noreply.github.com`).
EMAIL_ALLOWED_LOCAL_PARTS: set[str] = {"noreply", "no-reply"}

# Known non-name capitalized pairs that would otherwise false-positive the
# capitalized-pair name scan (project/doc vocabulary, not people).
PAIR_FALSE_POSITIVES: set[str] = {
    "book club",
    "contacts conflict",
    "deck project",
    "weekend plans",
    "not the",
}


def _persona_ok(name: str) -> bool:
    return name.strip().lower() in PERSON_FIRST_NAMES


def _full_name_ok(name: str) -> bool:
    return name.strip().lower() in PERSON_FULL_NAMES


@dataclass
class Finding:
    path: str
    line: int
    category: str
    hard: bool
    detail: str

    def __str__(self) -> str:
        level = "FAIL" if self.hard else "WARN"
        return f"[{level}] {self.path}:{self.line}: {self.category}: {self.detail}"


# ---------------------------------------------------------------------------
# Phone numbers
# ---------------------------------------------------------------------------

_PHONE_SEP = re.compile(r"(?<!\d)\(?([2-9]\d{2})\)?[-.\s]([2-9]\d{2})[-.\s](\d{4})(?!\d)")
_PHONE_PLUS1 = re.compile(r"(?<!\d)\+1[-.\s]?([2-9]\d{2})[-.\s]?([2-9]\d{2})[-.\s]?(\d{4})(?!\d)")


def check_phones(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    seen_spans: set[tuple[int, int]] = set()
    for rx in (_PHONE_PLUS1, _PHONE_SEP):
        for m in rx.finditer(text):
            span = m.span()
            if any(span[0] < e and s < span[1] for s, e in seen_spans):
                continue  # already matched by the other pattern
            area, exch, line = m.group(1), m.group(2), m.group(3)
            e164 = f"+1{area}{exch}{line}"
            if area == "555" or exch == "555":
                continue  # fictional NANP range, either position
            if e164 in PHONE_ALLOWLIST:
                continue
            seen_spans.add(span)
            lineno = text.count("\n", 0, m.start()) + 1
            findings.append(
                Finding(
                    path,
                    lineno,
                    "phone",
                    hard=True,
                    detail=f"non-fictional-looking number {area}-{exch}-{line}",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Emails
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")


def _domain_ok(domain: str) -> bool:
    d = domain.lower()
    if d in EMAIL_ALLOWED_DOMAIN_EXACT:
        return True
    if d.startswith(EMAIL_ALLOWED_DOMAIN_PREFIXES):
        return True
    return d.endswith(EMAIL_ALLOWED_DOMAIN_SUFFIXES)


def _local_part_ok(local: str) -> bool:
    # Strip a "+tag" suffix and anything after the first non
    # local-part-ish decoration some fixtures append, e.g.
    # "Bob@Example.com(filtered)" gets email-regexed as local "Bob".
    base = local.split("+", 1)[0]
    if base.lower() in EMAIL_ALLOWED_LOCAL_PARTS:
        return True
    # Fixtures sometimes use "Alice.Example" as a local part.
    first = re.split(r"[._-]", base)[0]
    return _persona_ok(first)


def check_emails(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for m in _EMAIL.finditer(text):
        local, domain = m.group(1), m.group(2)
        if _domain_ok(domain) or _local_part_ok(local):
            continue
        lineno = text.count("\n", 0, m.start()) + 1
        findings.append(
            Finding(path, lineno, "email", hard=True, detail=f"non-fictional-looking address {local}@{domain}")
        )
    return findings


# ---------------------------------------------------------------------------
# /Users/<name> paths
# ---------------------------------------------------------------------------

_USERS_PATH = re.compile(r"/Users/([A-Za-z][A-Za-z0-9_.-]*)")


def check_home_paths(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for m in _USERS_PATH.finditer(text):
        name = m.group(1)
        first = re.split(r"[._-]", name)[0]
        if _persona_ok(first) or _persona_ok(name):
            continue
        lineno = text.count("\n", 0, m.start()) + 1
        findings.append(
            Finding(path, lineno, "home-path", hard=True, detail=f"/Users/{name} is not an allowlisted persona")
        )
    return findings


# ---------------------------------------------------------------------------
# Names (report-only unless --strict)
# ---------------------------------------------------------------------------

_CAP_PAIR = re.compile(r"\b([A-Z][a-z]+)[ \t]+([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b")
_DISPLAY_NAME_FIXTURE = re.compile(
    r'\b(?:display_name|full_name|name)\s*[=:]\s*"([^"]+)"'
)
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "]"
)


def check_names(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []

    for m in _CAP_PAIR.finditer(text):
        first, second = m.group(1), m.group(2)
        pair = f"{first} {second}".lower()
        if pair in PAIR_FALSE_POSITIVES:
            continue
        if _full_name_ok(pair) or (_persona_ok(first) and _persona_ok(second)):
            continue
        lineno = text.count("\n", 0, m.start()) + 1
        findings.append(
            Finding(path, lineno, "name-pair", hard=False, detail=f"capitalized pair '{first} {second}'")
        )

    for m in _DISPLAY_NAME_FIXTURE.finditer(text):
        value = m.group(1)
        words = value.split()
        if len(words) >= 3 and value.lower() not in PERSON_FULL_NAMES:
            lineno = text.count("\n", 0, m.start()) + 1
            findings.append(
                Finding(path, lineno, "display-name-three-word", hard=False, detail=f"'{value}'")
            )

    for m in re.finditer(r"\b([A-Z][a-zA-Z]{1,20})\b", text):
        name = m.group(1)
        start, end = m.span()
        window = text[max(0, start - 4) : end + 4]
        if _EMOJI.search(window) and not _persona_ok(name):
            lineno = text.count("\n", 0, m.start()) + 1
            findings.append(
                Finding(path, lineno, "emoji-name", hard=False, detail=f"'{name}' decorated with an emoji")
            )

    return findings


# ---------------------------------------------------------------------------
# Commit-message-only: brand/bank/recipient shape
# ---------------------------------------------------------------------------

_BRAND_RECIPIENT_LINE = re.compile(
    r"\b[A-Z][A-Za-z]+:\s.*?\b[A-Z][A-Za-z]+:\s.*?\brecipient\b", re.IGNORECASE | re.DOTALL
)


def check_commit_message_shape(sha: str, message: str) -> list[Finding]:
    """Two colon-prefixed brand/institution names plus the word
    "recipient" anywhere after them, within a short span of text — the
    shape of the real leaked commit message this check exists to catch
    ("PayPal: ... Wells Fargo: ... recipient ..."). Checked over the whole
    message, not line by line, since the real leak wrapped across lines."""
    findings: list[Finding] = []
    m = _BRAND_RECIPIENT_LINE.search(message)
    if m and (m.end() - m.start()) < 400:
        lineno = message.count("\n", 0, m.start()) + 1
        findings.append(
            Finding(
                f"commit:{sha}",
                lineno,
                "commit-message-shape",
                hard=True,
                detail="message names two brands and a recipient, like the leaked message this check exists to catch",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def scan_text(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(check_phones(path, text))
    findings.extend(check_emails(path, text))
    findings.extend(check_home_paths(path, text))
    findings.extend(check_names(path, text))
    return findings


def scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text(str(path), text)


def git_ls_files(repo_root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def git_commit_messages(repo_root: Path, commit_range: str) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "log", "--format=%x00%H%x01%B", commit_range],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    commits: list[tuple[str, str]] = []
    for chunk in out.stdout.split("\x00"):
        if not chunk.strip():
            continue
        sha, _, message = chunk.partition("\x01")
        commits.append((sha.strip(), message))
    return commits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Specific files to scan.")
    parser.add_argument("--all-files", action="store_true", help="Scan every file tracked by git (git ls-files).")
    parser.add_argument(
        "--commits",
        metavar="RANGE",
        help="Also (or instead) scan commit messages in this git revision range, e.g. HEAD~20..HEAD.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat report-only findings (name checks) as hard failures too.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root for --all-files and --commits (default: cwd).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    all_findings: list[Finding] = []

    paths_to_scan: list[Path] = [Path(f) for f in args.files]
    if args.all_files:
        paths_to_scan.extend(repo_root / f for f in git_ls_files(repo_root))

    for path in paths_to_scan:
        all_findings.extend(scan_file(path))

    if args.commits:
        for sha, message in git_commit_messages(repo_root, args.commits):
            all_findings.extend(check_commit_message_shape(sha, message))
            all_findings.extend(scan_text(f"commit:{sha}", message))

    if not paths_to_scan and not args.commits:
        parser.error("nothing to scan: pass files, --all-files, and/or --commits <range>")

    hard = [f for f in all_findings if f.hard]
    soft = [f for f in all_findings if not f.hard]

    for f in hard:
        print(f, file=sys.stderr)
    for f in soft:
        print(f, file=sys.stderr)

    if hard:
        print(f"\n{len(hard)} hard finding(s), {len(soft)} warning(s).", file=sys.stderr)
        return 1
    if soft:
        print(f"\n0 hard findings, {len(soft)} warning(s) (name checks are report-only).", file=sys.stderr)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
