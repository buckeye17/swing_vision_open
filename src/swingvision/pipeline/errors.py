"""Plain-language explanations for stage failures (shown on the Jobs page).

The traceback still goes to the job's log and ``jobs.error``; the message says what
probably happened and what to do about it.
"""

from __future__ import annotations

import errno
import subprocess


def is_gpu_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(exc)


def describe(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    if is_gpu_oom(exc):
        return (
            "the GPU ran out of memory (another program may be using it). Close other GPU "
            "programs (games, video editors, other AI tools) and retry the job."
        )
    if "CUDA error" in text or "CUBLAS" in text or "cuDNN" in text or "NVDEC" in text:
        return (
            f"a GPU error ({text.splitlines()[0]}). Retry the job; if it keeps happening, "
            "restart the computer or update the NVIDIA driver."
        )
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "the output disk is full. Free some space (or pick another output folder) and retry."
    if isinstance(exc, PermissionError):
        return (
            f"permission denied for {exc.filename or 'a file'}. Another program may have it "
            "open (close it, e.g. a video player or spreadsheet), then retry."
        )
    if isinstance(exc, FileNotFoundError):
        return f"a file is missing: {exc.filename or text}."
    if type(exc).__name__ == "FFmpegError":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return f"ffmpeg failed ({lines[-1].strip() if lines else 'no output'})."
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = exc.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        last = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit code {exc.returncode}"
        return f"an external tool failed ({last})."
    return text
