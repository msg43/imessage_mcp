"""Thumbnails and browser-friendly previews of attachments, cached on the
encrypted volume.

- Image thumbnails (480 px) and full-size previews (2048 px, for HEIC and
  TIFF, which most browsers cannot draw) come from Apple's ImageIO,
  called through `ctypes` in a short-lived helper process.
- Video posters come from `ffmpeg`, when it is installed.
- Voice notes (CAF, AMR) are converted to AAC with `afconvert`.

Every converter is an untrusted-content decoder, so each runs in its own
process under `sandbox-exec` with network access denied and file writes
allowed only in a fresh work directory under the thumbnail cache, the
same containment `imsg.enrich.doc_text` gives `textutil`. `sips` and
`qlmanage` were rejected: `sips` writes its output through the boot
volume's temporary directory, and `qlmanage` refuses to start inside a
sandbox and keeps its own thumbnail cache off the encrypted volume.

Conversions are bounded: two at a time, a timeout each, a size ceiling on
the input, and a failure marker so a file that cannot be converted is
not retried on every page view (it is retried after a day).
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from imsg.paths import resolve_path

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
AFCONVERT = "/usr/bin/afconvert"
FFMPEG_CANDIDATES = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
THUMBNAIL_PIXELS = 480
PREVIEW_PIXELS = 2048
JPEG_QUALITY = 0.72
FAILURE_RETRY_SECONDS = 86400.0
MAX_SOURCE_BYTES = 1024 * 1024 * 1024
MAX_IMAGE_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_AUDIO_OUTPUT_BYTES = 256 * 1024 * 1024

ArgvBuilder = Callable[[Path, Path], list[str]]
"""Builds a converter's argv from `(source, output)`."""

IMAGEIO_HELPER = r"""
import ctypes, sys

def main(src, dst, max_pixels, quality):
    cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    iio = ctypes.CDLL("/System/Library/Frameworks/ImageIO.framework/ImageIO")
    vp = ctypes.c_void_p
    cf.CFURLCreateFromFileSystemRepresentation.restype = vp
    cf.CFURLCreateFromFileSystemRepresentation.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
    cf.CFNumberCreate.restype = vp
    cf.CFNumberCreate.argtypes = [vp, ctypes.c_long, vp]
    cf.CFDictionaryCreate.restype = vp
    cf.CFDictionaryCreate.argtypes = [vp, ctypes.POINTER(vp), ctypes.POINTER(vp), ctypes.c_long, vp, vp]
    cf.CFStringCreateWithCString.restype = vp
    cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
    iio.CGImageSourceCreateWithURL.restype = vp
    iio.CGImageSourceCreateWithURL.argtypes = [vp, vp]
    iio.CGImageSourceCreateThumbnailAtIndex.restype = vp
    iio.CGImageSourceCreateThumbnailAtIndex.argtypes = [vp, ctypes.c_size_t, vp]
    iio.CGImageDestinationCreateWithURL.restype = vp
    iio.CGImageDestinationCreateWithURL.argtypes = [vp, vp, ctypes.c_size_t, vp]
    iio.CGImageDestinationAddImage.argtypes = [vp, vp, vp]
    iio.CGImageDestinationFinalize.restype = ctypes.c_bool
    iio.CGImageDestinationFinalize.argtypes = [vp]
    key_callbacks = ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeDictionaryKeyCallBacks"))
    value_callbacks = ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeDictionaryValueCallBacks"))
    true = vp.in_dll(cf, "kCFBooleanTrue").value

    def const(name):
        return vp.in_dll(iio, name).value

    def url(path):
        raw = path.encode()
        return cf.CFURLCreateFromFileSystemRepresentation(None, raw, len(raw), False)

    def dictionary(pairs):
        keys = (vp * len(pairs))(*[k for k, _ in pairs])
        values = (vp * len(pairs))(*[v for _, v in pairs])
        return cf.CFDictionaryCreate(None, keys, values, len(pairs), key_callbacks, value_callbacks)

    pixels = ctypes.c_int32(int(max_pixels))
    pixels_number = cf.CFNumberCreate(None, 3, ctypes.byref(pixels))
    q = ctypes.c_double(float(quality))
    quality_number = cf.CFNumberCreate(None, 6, ctypes.byref(q))
    source = iio.CGImageSourceCreateWithURL(url(src), None)
    if not source:
        return 2
    options = dictionary([
        (const("kCGImageSourceCreateThumbnailFromImageAlways"), true),
        (const("kCGImageSourceCreateThumbnailWithTransform"), true),
        (const("kCGImageSourceThumbnailMaxPixelSize"), pixels_number),
    ])
    image = iio.CGImageSourceCreateThumbnailAtIndex(source, 0, options)
    if not image:
        return 3
    jpeg = cf.CFStringCreateWithCString(None, b"public.jpeg", 0x08000100)
    destination = iio.CGImageDestinationCreateWithURL(url(dst), jpeg, 1, None)
    if not destination:
        return 4
    properties = dictionary([(const("kCGImageDestinationLossyCompressionQuality"), quality_number)])
    iio.CGImageDestinationAddImage(destination, image, properties)
    return 0 if iio.CGImageDestinationFinalize(destination) else 5

sys.exit(main(*sys.argv[1:5]))
"""
"""Self-contained so the helper imports nothing from this package: it
runs with the interpreter's own standard library and Apple's frameworks
only."""


def _sbpl_string(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def sandbox_profile(work_dir: Path) -> str:
    """No network; writes only inside `work_dir`."""
    return (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        f"(allow file-write* (subpath {_sbpl_string(work_dir)}))"
    )


def find_ffmpeg() -> str | None:
    for candidate in FFMPEG_CANDIDATES:
        if os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("ffmpeg")


class MediaConverter:
    def __init__(
        self,
        cache_dir: Path,
        *,
        python: str = sys.executable,
        ffmpeg: str | None = None,
        afconvert: str = AFCONVERT,
        sandbox_exec: str = SANDBOX_EXEC,
        timeout_seconds: float = 30.0,
        max_concurrent: int = 2,
    ) -> None:
        self._cache_dir = cache_dir
        self._python = python
        self._ffmpeg = ffmpeg if ffmpeg is not None else find_ffmpeg()
        self._afconvert = afconvert
        self._sandbox_exec = sandbox_exec
        self._timeout = timeout_seconds
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _lock_for(self, name: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(name)
            if lock is None:
                if len(self._locks) > 4096:
                    self._locks.clear()
                lock = threading.Lock()
                self._locks[name] = lock
            return lock

    def _target(self, sha256: str, suffix: str) -> Path:
        return self._cache_dir / sha256[:2] / f"{sha256}.{suffix}"

    def cached(self, sha256: str, suffix: str) -> Path | None:
        target = self._target(sha256, suffix)
        return target if target.is_file() else None

    def _recent_failure(self, marker: Path) -> bool:
        try:
            return time.time() - marker.stat().st_mtime < FAILURE_RETRY_SECONDS
        except OSError:
            return False

    def _convert(
        self, source: Path, sha256: str, suffix: str, argv_for: ArgvBuilder, max_output: int
    ) -> Path | None:
        target = self._target(sha256, suffix)
        if target.is_file():
            return target
        marker = target.with_name(target.name + ".failed")
        if self._recent_failure(marker):
            return None
        try:
            if source.stat().st_size > MAX_SOURCE_BYTES:
                return None
        except OSError:
            return None
        with self._lock_for(target.name):
            if target.is_file():
                return target
            if not self._slots.acquire(timeout=self._timeout):
                return None
            try:
                return self._run(source, target, marker, argv_for, max_output)
            finally:
                self._slots.release()

    def _run(
        self, source: Path, target: Path, marker: Path, argv_for: ArgvBuilder, max_output: int
    ) -> Path | None:
        work = self._cache_dir / ".work" / secrets.token_hex(8)
        work.mkdir(parents=True, exist_ok=True, mode=0o700)
        work_real = resolve_path(work)
        output = work_real / f"out.{target.suffix.lstrip('.')}"
        argv = [self._sandbox_exec, "-p", sandbox_profile(work_real), *argv_for(source, output)]
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=work_real,
                timeout=self._timeout,
                check=False,
                env={"PATH": "/usr/bin:/bin", "TMPDIR": str(work_real) + "/"},
            )
            ok = (
                completed.returncode == 0
                and output.is_file()
                and 0 < output.stat().st_size <= max_output
            )
            if not ok:
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                marker.write_bytes(b"")
                return None
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.replace(output, target)
            marker.unlink(missing_ok=True)
            return target
        except (subprocess.TimeoutExpired, OSError):
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            marker.write_bytes(b"")
            return None
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # -- conversions --------------------------------------------------------

    def _imageio(self, pixels: int) -> ArgvBuilder:
        def argv(source: Path, output: Path) -> list[str]:
            return [
                self._python, "-I", "-S", "-c", IMAGEIO_HELPER,
                str(source), str(output), str(pixels), str(JPEG_QUALITY),
            ]

        return argv

    def image_thumbnail(self, source: Path, sha256: str) -> Path | None:
        return self._convert(source, sha256, "thumb.jpg", self._imageio(THUMBNAIL_PIXELS), MAX_IMAGE_OUTPUT_BYTES)

    def image_preview(self, source: Path, sha256: str) -> Path | None:
        return self._convert(source, sha256, "view.jpg", self._imageio(PREVIEW_PIXELS), MAX_IMAGE_OUTPUT_BYTES)

    def video_poster(self, source: Path, sha256: str) -> Path | None:
        ffmpeg = self._ffmpeg
        if ffmpeg is None:
            return None

        def argv(src: Path, output: Path) -> list[str]:
            return [
                ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-ss", "0.5", "-i", str(src),
                "-frames:v", "1", "-vf", f"scale='min({THUMBNAIL_PIXELS},iw)':-2",
                "-f", "image2", "-c:v", "mjpeg", "-q:v", "5", str(output),
            ]

        return self._convert(source, sha256, "poster.jpg", argv, MAX_IMAGE_OUTPUT_BYTES)

    def audio_preview(self, source: Path, sha256: str) -> Path | None:
        afconvert = self._afconvert

        def argv(src: Path, output: Path) -> list[str]:
            return [afconvert, "-f", "m4af", "-d", "aac", str(src), str(output)]

        return self._convert(source, sha256, "audio.m4a", argv, MAX_AUDIO_OUTPUT_BYTES)


__all__ = [
    "IMAGEIO_HELPER",
    "PREVIEW_PIXELS",
    "THUMBNAIL_PIXELS",
    "MediaConverter",
    "find_ffmpeg",
    "sandbox_profile",
]
