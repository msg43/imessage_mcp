"""Errors the search page and its internal model API raise. Kept in this
package, beside the code that raises them, so the shared hierarchy in
`imsg.errors` stays untouched; every class still derives from
`imsg.errors.ImsgError`, which the CLI boundary catches."""

from __future__ import annotations

from imsg.errors import ImsgError


class SearchPageError(ImsgError):
    """Base class for search-page failures."""


class SecretFileError(SearchPageError):
    """A secret file is missing, too permissive, not ours, a symlink, or
    resolves outside `paths.data_root`. The message names the path and
    the rule, never the file's contents."""


class SearchPageStartupError(SearchPageError):
    """The page or the model API refused to start: fail closed."""


class ModelApiUnavailable(SearchPageError):
    """The internal model API could not answer (down, warming up, busy,
    wrong secret, timed out). Full-text search carries on without it."""

    def __init__(self, reason: str, *, retryable: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable


class SearchInputError(SearchPageError):
    """A search request the owner can fix: an empty query, a bad date, an
    unknown or ambiguous person. The message is shown on the page."""


__all__ = [
    "ModelApiUnavailable",
    "SearchInputError",
    "SearchPageError",
    "SearchPageStartupError",
    "SecretFileError",
]
