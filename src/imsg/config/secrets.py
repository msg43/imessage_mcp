"""Secret references: ``keychain:<item>`` / ``env:<VAR>`` / ``file:<path>`` — never literals.

SPEC §3.2 and §6: every config field marked *(secret)* must hold a
reference, resolved at the point of use via the macOS Keychain, an
environment variable, or a file only its owner can read. A literal value typed into the config file
(including something that merely *looks* like a real secret) is
rejected at parse time — this is the enforcement mechanism, not a
convention operators are trusted to follow, because the config file is
exactly the artifact most likely to be committed by accident.

``file:<absolute path>`` exists for headless hosts. The login Keychain
cannot be read over SSH or from a launchd job before someone logs in,
and an ``env:`` reference makes every caller (launchd agents, SSH
wrappers, MCP client entries) set the variable itself, so one secret
ends up copied into many places. A file reference keeps the secret in
one place. It resolves only when the file is a regular file owned by the
current user with no group or other permission bits (``chmod 600``), the
same rule ``ssh`` applies to private keys, so a copy that other local
accounts can read is refused rather than used.

Downstream modules should depend on this module's :class:`SecretRef`
type for any new secret-shaped config field rather than inventing a
parallel convention.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from imsg.errors import SecretResolutionError

_KEYCHAIN_RE = re.compile(r"^keychain:(?P<name>[A-Za-z0-9][A-Za-z0-9_.\-]*)$")
_ENV_RE = re.compile(r"^env:(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
# Absolute paths only: a relative path would resolve against whatever
# directory a launchd job or SSH session happens to start in.
_FILE_RE = re.compile(r"^file:(?P<name>/[^\x00\n\r]*[^/\x00\n\r])$")

# A secret file is a password or token, never a document; anything larger
# is a mistaken path, and reading it whole would be pointless.
_MAX_SECRET_FILE_BYTES = 64 * 1024

_FORMAT_ERROR = (
    "secret fields must be 'keychain:<item-name>', 'env:<VAR>' or "
    "'file:<absolute path>' — literal values (including anything that looks "
    "like a real secret) are rejected; see SPEC §6"
)


@dataclass(frozen=True, slots=True)
class SecretRef:
    """A reference to a secret. Holding one of these never means you hold the secret."""

    kind: Literal["keychain", "env", "file"]
    name: str
    raw: str

    _keychain_re: ClassVar[re.Pattern[str]] = _KEYCHAIN_RE
    _env_re: ClassVar[re.Pattern[str]] = _ENV_RE
    _file_re: ClassVar[re.Pattern[str]] = _FILE_RE

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
        if m := cls._file_re.match(value):
            return cls(kind="file", name=m.group("name"), raw=value)
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
        if self.kind == "file":
            return self._resolve_file()
        return self._resolve_keychain()

    def _resolve_file(self) -> str:
        # Open first, then check the opened file: checking the path and
        # opening it afterwards would let the file be swapped in between.
        # A symlink is followed; the checks apply to the file it reaches.
        # O_NONBLOCK so a FIFO at the path is refused below instead of
        # blocking the caller until something writes to it.
        try:
            fd = os.open(
                self.name,
                os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
            )
        except FileNotFoundError:
            raise SecretResolutionError(
                f"secret file '{self.name}' does not exist (referenced as '{self.raw}')"
            ) from None
        except OSError as exc:
            raise SecretResolutionError(
                f"secret file '{self.name}' cannot be opened: {exc.strerror} "
                f"(referenced as '{self.raw}')"
            ) from None
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise SecretResolutionError(
                    f"secret file '{self.name}' is not a regular file "
                    f"(referenced as '{self.raw}')"
                )
            if info.st_uid != os.geteuid():
                raise SecretResolutionError(
                    f"secret file '{self.name}' is owned by uid {info.st_uid}, not "
                    f"by the current user (uid {os.geteuid()}); refusing to use it "
                    f"(referenced as '{self.raw}')"
                )
            mode = stat.S_IMODE(info.st_mode)
            if mode & 0o077:
                raise SecretResolutionError(
                    f"secret file '{self.name}' is accessible by group or others "
                    f"(mode {mode:04o}); run 'chmod 600' on it. Refusing to use it "
                    f"(referenced as '{self.raw}')"
                )
            chunks: list[bytes] = []
            total = 0
            while total <= _MAX_SECRET_FILE_BYTES:
                chunk = os.read(fd, _MAX_SECRET_FILE_BYTES + 1 - total)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(fd)
        if len(data) > _MAX_SECRET_FILE_BYTES:
            raise SecretResolutionError(
                f"secret file '{self.name}' is larger than "
                f"{_MAX_SECRET_FILE_BYTES} bytes, so it is not a secret file "
                f"(referenced as '{self.raw}')"
            )
        try:
            value = data.decode("utf-8")
        except UnicodeDecodeError:
            raise SecretResolutionError(
                f"secret file '{self.name}' is not UTF-8 text "
                f"(referenced as '{self.raw}')"
            ) from None
        # Same trimming as the Keychain path: `printf 'x\n' > file` and
        # `echo x > file` both mean the value `x`.
        value = value.rstrip("\n")
        if not value:
            raise SecretResolutionError(
                f"secret file '{self.name}' is empty (referenced as '{self.raw}')"
            )
        return value

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
