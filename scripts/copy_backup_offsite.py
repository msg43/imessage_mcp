#!/usr/bin/env python3
"""scripts/copy_backup_offsite.py -- weekly encrypted off-box copy of the
newest nightly backup set.

`imsg backup` writes its sets under `<data_root>/backups/`, on the same
device as the data they protect (SPEC §14 "no disaster backup"). This
script is the off-box half: it takes the newest *complete* nightly set,
re-checks every file against the set's MANIFEST.json, adds a tar of the
instance's `private/` overlay (config, identity overrides -- which the
nightly set does not contain), and:

1. encrypts every file locally with `age` to a public recipient key. Only
   ciphertext ever leaves this machine. The matching identity (decryption)
   key is NOT needed here and must not be stored at the destination;
2. streams each ciphertext to the destination over ssh into a
   `<set>.partial/` directory;
3. reads every file back over the same ssh connection and compares its
   SHA-256 with the local ciphertext -- a copy that cannot be read back
   is a failed copy;
4. writes OFFSITE-MANIFEST.json (names, sizes, ciphertext hashes -- no
   message content), reads it back, and renames `<set>.partial` to
   `<set>` so a complete-looking set is always a verified one;
5. keeps the newest `--keep` verified sets at the destination and deletes
   older ones plus stale `.partial` leftovers;
6. records the outcome in `<data_root>/run/`: `offsite-backup.last.json`
   on success, `offsite-backup.FAILED` (and an optional macOS
   notification) on failure, exit status 1.

The real destination host, path and recipient file are instance
configuration: they belong in the private LaunchAgent / overlay, never in
this repo.

Usage:
    python3 scripts/copy_backup_offsite.py --data-root DIR \\
        --recipient-file FILE --remote SSH_HOST --remote-dir PATH \\
        [--keep 8] [--notify] [--dry-run]

`--remote ''` runs the destination commands through a local shell instead
of ssh (used by the tests). Standard library only, plus the `age` binary.

Run it from a LaunchAgent through `src/imsg/agents/supervise.py` like the
other `com.imsgindex.*` agents (mount gate, logs on the volume). Pointing
launchd's own StandardOut/ErrPath at the encrypted volume fails at spawn
with exit 78.

Restoring (on a FileVault-encrypted disk only; never at the destination):
    age -d -i IDENTITY -o postgres.dump postgres.dump.age
    age -d -i IDENTITY -o private.tar private.tar.age && tar -xf private.tar
    # check the decrypted dump against MANIFEST.json's sha256, then restore
    # into a throwaway cluster first, e.g. one table:
    pg_restore -d scratch --no-owner --section=pre-data postgres.dump
    pg_restore -d scratch --no-owner --data-only -t person postgres.dump
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

SET_NAME_RE = re.compile(r"^backup-\d{8}T\d{6}Z-[0-9a-f]{6}$")
OFFSITE_MANIFEST = "OFFSITE-MANIFEST.json"
PARTIAL_SUFFIX = ".partial"
AGE_HEADER = b"age-encryption.org/v1\n"
CHUNK = 4 * 1024 * 1024
FAILED_MARKER = "offsite-backup.FAILED"
LAST_STATE = "offsite-backup.last.json"
LOCK_FILE = "offsite-backup.lock"


class OffsiteError(Exception):
    """A step failed; the message says which and why, in plain words."""


def log(msg: str) -> None:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} offsite: {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- source side


def newest_complete_set(backups_dir: Path) -> Path:
    """Newest set whose MANIFEST.json says complete. Set names start with a
    UTC timestamp, so name order is time order."""
    if not backups_dir.is_dir():
        raise OffsiteError(f"backup directory not found: {backups_dir}")
    for entry in sorted(backups_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not (entry.is_dir() and SET_NAME_RE.match(entry.name)):
            continue
        manifest = entry / "MANIFEST.json"
        try:
            data = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if data.get("complete") is True and data.get("files"):
            return entry
    raise OffsiteError(f"no complete backup set in {backups_dir}")


def verify_source_set(set_dir: Path) -> list[Path]:
    """Check every file the manifest lists against its recorded size and
    SHA-256. Returns the files to copy (manifest first)."""
    data = json.loads((set_dir / "MANIFEST.json").read_text())
    files = [set_dir / "MANIFEST.json"]
    for entry in data["files"]:
        path = set_dir / entry["name"]
        if not path.is_file():
            raise OffsiteError(f"{set_dir.name}: manifest lists {entry['name']} but it is missing")
        size = path.stat().st_size
        if "bytes" in entry and size != entry["bytes"]:
            raise OffsiteError(
                f"{set_dir.name}/{entry['name']}: size {size} != manifest {entry['bytes']}"
            )
        if "sha256" not in entry:
            raise OffsiteError(f"{set_dir.name}/{entry['name']}: manifest has no sha256")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise OffsiteError(f"{set_dir.name}/{entry['name']}: sha256 does not match manifest")
        files.append(path)
    return files


def tar_private(data_root: Path, out: Path) -> None:
    private = data_root / "private"
    if not private.is_dir():
        raise OffsiteError(f"private overlay not found: {private}")
    with tarfile.open(out, "w") as tar:
        tar.add(private, arcname="private")


def age_encrypt(age: str, recipient_file: Path, src: Path, dst: Path) -> None:
    proc = subprocess.run(
        [age, "--encrypt", "-R", str(recipient_file), "-o", str(dst), str(src)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise OffsiteError(f"age failed on {src.name}: {proc.stderr.strip()}")
    with dst.open("rb") as fh:
        if fh.read(len(AGE_HEADER)) != AGE_HEADER:
            raise OffsiteError(f"{dst.name} does not start with an age header")


# ----------------------------------------------------------- destination side


@dataclass
class Remote:
    host: str  # '' = run through a local shell (tests)

    def argv(self, command: str) -> list[str]:
        if not self.host:
            return ["/bin/sh", "-c", command]
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", self.host, command]

    def run(self, command: str, stdin: IO[bytes] | None = None) -> str:
        proc = subprocess.run(self.argv(command), stdin=stdin, capture_output=True)
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace").strip()
            raise OffsiteError(f"destination command failed ({command.split()[0]}): {err}")
        return proc.stdout.decode(errors="replace")

    def upload(self, local: Path, remote_path: str) -> None:
        with local.open("rb") as fh:
            self.run(f"cat > {shlex.quote(remote_path)}", stdin=fh)

    def read_back_sha256(self, remote_path: str) -> tuple[str, int]:
        """Stream the file back and hash it here, so the check exercises
        the same read path a restore would use."""
        proc = subprocess.Popen(
            self.argv(f"cat {shlex.quote(remote_path)}"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        h = hashlib.sha256()
        n = 0
        while chunk := proc.stdout.read(CHUNK):
            h.update(chunk)
            n += len(chunk)
        err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
        if proc.wait() != 0:
            raise OffsiteError(f"read-back of {remote_path} failed: {err}")
        return h.hexdigest(), n

    def list_sets(self, remote_dir: str) -> tuple[list[str], list[str]]:
        """(verified set names, partial dir names) at the destination."""
        q = shlex.quote(remote_dir)
        out = self.run(
            f"cd {q} && for d in *; do "
            f'[ -d "$d" ] || continue; '
            f'if [ -f "$d/{OFFSITE_MANIFEST}" ]; then echo "S $d"; else echo "P $d"; fi; '
            f"done"
        )
        sets, partials = [], []
        for line in out.splitlines():
            kind, _, name = line.partition(" ")
            if kind == "S" and SET_NAME_RE.match(name):
                sets.append(name)
            elif kind == "P" and (
                SET_NAME_RE.match(name)
                or (
                    name.endswith(PARTIAL_SUFFIX)
                    and SET_NAME_RE.match(name[: -len(PARTIAL_SUFFIX)])
                )
            ):
                partials.append(name)
        return sorted(sets), sorted(partials)


def sets_to_delete(verified: list[str], keep: int) -> list[str]:
    if keep < 1:
        raise OffsiteError("--keep must be at least 1")
    ordered = sorted(verified)
    return ordered[:-keep] if len(ordered) > keep else []


# ------------------------------------------------------------------ main flow


def copy_set(
    *,
    data_root: Path,
    set_dir: Path,
    recipient_file: Path,
    remote: Remote,
    remote_dir: str,
    age: str,
) -> dict:
    name = set_dir.name
    staging = data_root / "run" / "offsite-staging" / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        sources = verify_source_set(set_dir)
        log(f"{name}: source files match MANIFEST.json")
        private_tar = staging / "private.tar"
        tar_private(data_root, private_tar)
        sources.append(private_tar)

        encrypted: list[Path] = []
        for src in sources:
            dst = staging / f"{src.name}.age"
            age_encrypt(age, recipient_file, src, dst)
            encrypted.append(dst)
        private_tar.unlink()  # plaintext copy of private/ never lingers
        log(f"{name}: encrypted {len(encrypted)} files")

        partial = f"{remote_dir}/{name}{PARTIAL_SUFFIX}"
        final = f"{remote_dir}/{name}"
        remote.run(f"rm -rf {shlex.quote(partial)} && mkdir -p {shlex.quote(partial)}")
        entries = []
        for path in encrypted:
            local_hash = sha256_file(path)
            size = path.stat().st_size
            target = f"{partial}/{path.name}"
            remote.upload(path, target)
            back_hash, back_size = remote.read_back_sha256(target)
            if (back_hash, back_size) != (local_hash, size):
                raise OffsiteError(f"read-back mismatch for {path.name}")
            entries.append({"name": path.name, "bytes": size, "sha256": local_hash})
            log(f"{name}: {path.name} {size} B copied and read back")

        manifest = {
            "source_set": name,
            "copied_at": dt.datetime.now(dt.UTC).isoformat(),
            "encryption": "age (X25519 recipient); decryption key is not stored here",
            "recipient": recipient_file.read_text().strip().splitlines()[-1],
            "files": entries,
        }
        manifest_path = staging / OFFSITE_MANIFEST
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        remote.upload(manifest_path, f"{partial}/{OFFSITE_MANIFEST}")
        back_hash, _ = remote.read_back_sha256(f"{partial}/{OFFSITE_MANIFEST}")
        if back_hash != sha256_file(manifest_path):
            raise OffsiteError("read-back mismatch for OFFSITE-MANIFEST.json")
        remote.run(f"rm -rf {shlex.quote(final)} && mv {shlex.quote(partial)} {shlex.quote(final)}")
        log(f"{name}: verified copy in place")
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def apply_retention(remote: Remote, remote_dir: str, keep: int, current: str) -> list[str]:
    verified, partials = remote.list_sets(remote_dir)
    if current not in verified:
        raise OffsiteError(f"{current} is not listed as verified after copy; not pruning")
    doomed = sets_to_delete(verified, keep) + [
        p for p in partials if p != f"{current}{PARTIAL_SUFFIX}"
    ]
    for d in doomed:
        remote.run(f"rm -rf {shlex.quote(f'{remote_dir}/{d}')}")
        log(f"retention: removed {d}")
    return doomed


def notify(message: str) -> None:
    script = f'display notification {json.dumps(message)} with title "iMessage index offsite backup FAILED"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    run_dir = data_root / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = (run_dir / LOCK_FILE).open("w")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another offsite copy is running; exiting")
        return 1
    try:
        age = shutil.which(args.age) or args.age
        if not Path(age).is_file():
            raise OffsiteError(f"age binary not found: {args.age}")
        recipient_file = Path(args.recipient_file)
        if not recipient_file.is_file():
            raise OffsiteError(f"recipient file not found: {recipient_file}")
        remote = Remote(args.remote)
        set_dir = newest_complete_set(data_root / "backups")
        log(f"newest complete set: {set_dir.name}")
        if args.dry_run:
            verify_source_set(set_dir)
            log("dry run: source verified; nothing encrypted or copied")
            return 0
        remote.run(f"mkdir -p {shlex.quote(args.remote_dir)}")
        verified, _ = remote.list_sets(args.remote_dir)
        if set_dir.name in verified:
            log(f"{set_dir.name}: already copied and verified earlier; skipping copy")
            manifest = {"source_set": set_dir.name, "skipped": "already present"}
        else:
            manifest = copy_set(
                data_root=data_root,
                set_dir=set_dir,
                recipient_file=recipient_file,
                remote=remote,
                remote_dir=args.remote_dir,
                age=age,
            )
        removed = apply_retention(remote, args.remote_dir, args.keep, set_dir.name)
        state = {
            "status": "ok",
            "finished_at": dt.datetime.now(dt.UTC).isoformat(),
            "source_set": set_dir.name,
            "removed": removed,
            "files": manifest.get("files", []),
        }
        (run_dir / LAST_STATE).write_text(json.dumps(state, indent=2) + "\n")
        (run_dir / FAILED_MARKER).unlink(missing_ok=True)
        log("done")
        return 0
    except (OffsiteError, OSError, subprocess.SubprocessError) as exc:
        msg = f"{type(exc).__name__}: {exc}"
        log(f"FAILED: {msg}")
        stamp = dt.datetime.now(dt.UTC).isoformat()
        (run_dir / FAILED_MARKER).write_text(f"{stamp} {msg}\n")
        if args.notify:
            notify(msg[:200])
        return 1
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-root", required=True)
    p.add_argument("--recipient-file", required=True, help="age public key file (recipient)")
    p.add_argument("--remote", required=True, help="ssh host alias; '' runs locally (tests)")
    p.add_argument("--remote-dir", required=True)
    p.add_argument("--keep", type=int, default=8)
    p.add_argument("--age", default="age")
    p.add_argument("--notify", action="store_true", help="macOS notification on failure")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    os.umask(0o077)
    sys.exit(run(parse_args()))
