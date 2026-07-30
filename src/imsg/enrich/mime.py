"""MIME sniffing via the `file` command's content-based magic detection
(SPEC §8 S5b, D6: "the extension is not trusted").

macOS ships `file` (and its bundled magic database) at `/usr/bin/file`;
shelling out to it gets the same content-based sniffing SPEC §4's
`python-magic / libmagic` dependency line calls for, without adding a
compiled libmagic *binding* dependency — not available via Homebrew in
this build/test sandbox. Same outcome (content sniffed, not the
filename extension trusted), subprocess boundary instead of a C
extension; worth revisiting at Phase 3/5 if a bundled libmagic wheel
turns out to be the simpler real-deployment path.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from imsg.errors import UntrustedAttachmentError

MimeSnifferFn = Callable[[Path], str]

# `type/subtype` — deliberately simple; only used to reject `file`'s prose
# error messages ("cannot open `...' (No such file or directory)"), which
# `file` prints to *stdout* with a **0** exit code (verified empirically —
# do not trust returncode/stderr alone to catch this).
_MIME_TYPE_RE = re.compile(r"^[\w.+-]+/[\w.+-]+$")


def real_sniff_mime(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["file", "--brief", "--mime-type", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UntrustedAttachmentError(f"MIME sniffing failed for '{path}': {exc}") from exc
    output = proc.stdout.strip()
    if proc.returncode != 0 or not _MIME_TYPE_RE.match(output):
        raise UntrustedAttachmentError(
            f"MIME sniffing failed for '{path}': 'file' exited {proc.returncode} "
            f"with unparseable output {output!r} (stderr: {proc.stderr.strip()!r})"
        )
    return output


__all__ = ["MimeSnifferFn", "real_sniff_mime"]
