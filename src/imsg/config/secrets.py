"""Secret references: ``keychain:<item>`` / ``env:<VAR>`` — never literals.

SPEC §3.2 and §6: every config field marked *(secret)* must hold a
reference, resolved at the point of use via the macOS Keychain or an
environment variable. A literal value typed into the config file
(including something that merely *looks* like a real secret) is
rejected at parse time — this is the enforcement mechanism, not a
convention operators are trusted to follow, because the config file is
exactly the artifact most likely to be committed by accident.

Downstream modules should depend on this module's :class:`SecretRef`
type for any new secret-shaped config field rather than inventing a
parallel convention.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from imsg.errors import SecretResolutionError

_KEYCHAIN_RE = re.compile(r"^keychain:(?P<name>[A-Za-z0-9][A-Za-z0-9_.\-]*)$")
_ENV_RE = re.compile(r"^env:(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")

_FORMAT_ERROR = (
    "secret fields must be 'keychain:<item-name>' or 'env:<VAR>' — literal "
    "values (including anything that looks like a real secret) are rejected; "
    "see SPEC §6"
)


@dataclass(frozen=True, slots=True)
class SecretRef:
    """A reference to a secret. Holding one of these never means you hold the secret."""

    kind: Literal["keychain", "env"]
    name: str
    raw: str

    _keychain_re: ClassVar[re.Pattern[str]] = _KEYCHAIN_RE
    _env_re: ClassVar[re.Pattern[str]] = _ENV_RE

    @classmethod
    def parse(cls, value: Any) -> SecretRef:
        if isinstance(value, SecretRef):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"{_FORMAT_ERROR} (got a {type(value).__name__}, not a string)"
            )
        if m := cls._keychain_re.match(value):
            return cls(kind="keychain", name=m.group("name"), raw=value)
        if m := cls._env_re.match(value):
            return cls(kind="env", name=m.group("name"), raw=value)
        raise ValueError(_FORMAT_ERROR)

    def resolve(self) -> str:
        """Resolve to the actual secret value.

        Callers must never log, print, or include the return value in an
        error message or exception.
        """
        if self.kind == "env":
            value = os.environ.get(self.name)
            if value is None:
                raise SecretResolutionError(
                    f"environment variable '{self.name}' is not set "
                    f"(referenced as '{self.raw}')"
                )
            return value
        return self._resolve_keychain()

    def _resolve_keychain(self) -> str:
        try:
            proc = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-a",
                    os.environ.get("USER", ""),
                    "-s",
                    self.name,
                    "-w",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SecretResolutionError(
                "the 'security' CLI is not available — Keychain secrets can only "
                "be resolved on macOS"
            ) from exc
        if proc.returncode != 0:
            raise SecretResolutionError(
                f"Keychain item '{self.name}' not found or inaccessible "
                f"(referenced as '{self.raw}')"
            )
        return proc.stdout.rstrip("\n")

    def __repr__(self) -> str:
        # Deliberately never includes a resolved value.
        return f"SecretRef({self.raw!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_before_validator_function(
            cls.parse,
            core_schema.is_instance_schema(cls),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.raw
            ),
        )


__all__ = ["SecretRef"]
