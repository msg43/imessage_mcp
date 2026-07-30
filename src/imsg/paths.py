"""Path containment helpers.

Hard requirements 1 and 2 (CLAUDE.md; SPEC §1, §6) are enforced by
config validation, and the enforcement is only as good as the path
comparison underneath it. A naive string-prefix check
(``str(path).startswith(str(root))``) is defeated by ``..`` segments
and by symlinks that point off the intended root — SPEC §5.4 calls
this out explicitly ("do not infer the mount by string prefix").

Every containment check in this codebase MUST go through
:func:`resolve_path` and :func:`is_contained_in` rather than reimplementing
prefix comparisons.
"""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: Path | str) -> Path:
    """Expand ``~`` and resolve to an absolute, symlink-free path.

    Uses ``strict=False`` because config validation runs before the
    directories necessarily exist (e.g. on first run, before
    ``paths.data_root`` has been created). ``Path.resolve`` still
    resolves every symlink in the portion of the path that *does*
    exist, and normalizes ``..``/``.`` segments in the rest — which is
    what defeats both attack/mistake shapes named in the spec.
    """
    return Path(path).expanduser().resolve(strict=False)


def is_contained_in(candidate: Path | str, root: Path | str) -> bool:
    """True if ``candidate`` resolves to ``root`` itself or somewhere beneath it.

    Both sides are resolved (symlinks + ``..``) before comparison, so a
    symlink under ``root`` that points outside it is correctly treated
    as *not* contained, and a path that merely looks contained as a
    string (e.g. via ``..``) is resolved to its real location first.
    """
    resolved_candidate = resolve_path(candidate)
    resolved_root = resolve_path(root)
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def join_under_root(root: Path | str, value: Path | str) -> Path:
    """Join ``value`` under ``root``, the way a "relative-to-root" config field should.

    If ``value`` is relative, this is exactly ``root / value``. If
    ``value`` is already absolute, ``pathlib`` semantics have
    ``root / value`` discard ``root`` and evaluate to ``value`` itself —
    which is exactly right here: an absolute derived-path field is still
    required to resolve under ``root``, and returning it unjoined lets
    :func:`is_contained_in` catch the case where it does not.
    """
    return Path(root) / Path(value)


__all__ = ["is_contained_in", "join_under_root", "resolve_path"]
