"""Pretrained model weights: where they come from, their licenses, and a local cache.

Base weights live in the app data directory (``settings.data_dir() / "models"``), not in the
output folder, so every output folder shares them. Downloads are verified against a pinned
SHA-256 and written atomically.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from swingvision.settings import data_dir
from swingvision.storage.fsutil import atomic_write

ULTRALYTICS_RELEASES = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0",
    "https://github.com/ultralytics/assets/releases/download/v0.0.0",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    urls: tuple[str, ...]
    sha256: str | None
    license: str
    task: str
    description: str
    size_mb: float

    @property
    def path(self) -> Path:
        return weights_dir() / self.filename

    def available(self) -> bool:
        return self.path.exists()


def _yolo(name: str, sha256: str, size_mb: float, description: str) -> ModelSpec:
    filename = f"{name}.pt"
    return ModelSpec(
        name=name,
        filename=filename,
        urls=tuple(f"{base}/{filename}" for base in ULTRALYTICS_RELEASES),
        sha256=sha256,
        license="AGPL-3.0 (Ultralytics)",
        task="person detection (COCO)",
        description=description,
        size_mb=size_mb,
    )


REGISTRY: dict[str, ModelSpec] = {
    m.name: m
    for m in (
        _yolo(
            "yolo11s",
            "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
            19.3,
            "Fastest; misses more far-court players",
        ),
        _yolo(
            "yolo11m",
            "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
            40.7,
            "Default: reliable on the far player at 1920 px",
        ),
        _yolo(
            "yolo11l",
            "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f",
            51.4,
            "Slower; slightly better on small, distant players",
        ),
    )
}


def weights_dir() -> Path:
    return data_dir() / "models"


def get(name: str) -> ModelSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}. Known: {', '.join(REGISTRY)}") from None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure(name: str, progress: Callable[[float], None] | None = None) -> Path:
    """Path to the model's weights, downloading (and verifying) them on first use."""
    spec = get(name)
    if spec.path.exists():
        return spec.path
    errors = []
    for url in spec.urls:
        try:
            atomic_write(spec.path, lambda tmp, url=url: _download(url, tmp, spec, progress))
            return spec.path
        except (OSError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(f"Could not download {spec.name}:\n" + "\n".join(errors))


def _download(url: str, tmp: Path, spec: ModelSpec, progress) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "swingvision"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        h = hashlib.sha256()
        while block := resp.read(1 << 20):
            out.write(block)
            h.update(block)
            done += len(block)
            if progress and total:
                progress(done / total)
    if spec.sha256 and h.hexdigest() != spec.sha256:
        raise ValueError(f"checksum mismatch (got {h.hexdigest()[:12]}…)")
